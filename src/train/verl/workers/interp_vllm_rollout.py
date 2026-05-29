"""Activation interventions for vLLM rollout.

Provides InterpvLLMRollout, a vLLMRollout subclass that patches decoder layer
classes BEFORE LLM() creation.

Three mutually exclusive modes:
  - Steering: adds a direction vector to hidden states at a target layer.
  - CAFT: projects hidden states off of a matrix at a target layer.
  - Capture: accumulates mean activations during decode via per-request routing.
    capture_layers uses HuggingFace convention where layer N = hidden_states[N]
    = output of decoder layer N-1. Compatible with CUDA graphs (enforce_eager=False).
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
import shutil

from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.workers.rollout.base import _ROLLOUT_REGISTRY

from src.vllm_utils import (
    get_decoder_layer_class,
    patch_decoder_for_steering,
    patch_decoder_for_caft,
    patch_decoder_for_capture,
    setup_steering,
    setup_caft,
    setup_capture,
)

logger = logging.getLogger(__name__)


class InterpvLLMRollout(vLLMRollout):
    """vLLMRollout subclass with optional activation capture or steering.

    Supports three mutually exclusive modes (or neither, in which case it
    behaves identically to vLLMRollout):
      - Steering: adds a direction vector at a target decoder layer
      - Caching: accumulates mean activations during decode
      - CAFT: projects hidden states off of a CAFT matrix at a target decoder layer

    Patches decoder layer classes BEFORE super().__init__() creates the LLM,
    then wires up steering vectors or decode-phase hooks after.
    """

    def __init__(self, config, model_config, device_mesh):
        interp_cfg = config.engine_kwargs.get('interp_vllm', {})
        ac_cfg = interp_cfg.get('activation_cache', {})
        caft_cfg = interp_cfg.get('caft', {})
        steer_cfg = interp_cfg.get('steering', {})

        self._activation_cache = ac_cfg.get('mode', None) == "interp_vllm"
        self._caft = caft_cfg.get('enable', False)
        self._steering = steer_cfg.get('enable', False)

        assert sum([self._activation_cache, self._caft, self._steering]) <= 1, (
            "Only one of activation_cache, caft, steering can be enabled at a time"
        )

        # Extract config values needed before and after super().__init__()
        if self._activation_cache:
            self._capture_layers = ac_cfg['layers']
            assert all(l >= 1 for l in self._capture_layers), (
                f"capture_layers must be >= 1 (layer 0 is embedding output). Got: {self._capture_layers}"
            )
            self._decoder_capture_layers = [l - 1 for l in self._capture_layers]
        if self._steering:
            self._steering_path = steer_cfg['path']
            self._steering_layer = steer_cfg['layer']
            assert self._steering_layer >= 1, (
                f"steering_layer must be >= 1 (residual-stream convention). Got: {self._steering_layer}"
            )
            self._steering_alpha = steer_cfg.get('alpha', 1.0)
        if self._caft:
            self._caft_path = caft_cfg['path']
            self._caft_layer = caft_cfg['layer']
            assert self._caft_layer >= 1, (
                f"caft_layer must be >= 1 (residual-stream convention). Got: {self._caft_layer}"
            )
            self._caft_orthogonalize = caft_cfg.get('orthogonalize', True)

        requires_patch = self._activation_cache or self._caft or self._steering

        # Patch decoder class BEFORE super().__init__() creates LLM()
        if requires_patch:
            shutil.rmtree("/tmp/_vllm/torch_compile_cache", ignore_errors=True)
            decoder_cls = get_decoder_layer_class(model_config.path)

        if self._activation_cache:
            max_batch_size = ac_cfg.get('token_buffer_capacity', ac_cfg.get('batch_size', 16) * config.n)
            patch_decoder_for_capture(
                decoder_cls,
                hidden_size=model_config.hf_config.hidden_size,
                max_batch_size=max_batch_size,
                capture_layers=set(self._decoder_capture_layers),
            )
        if self._steering:
            patch_decoder_for_steering(decoder_cls, model_config.hf_config.hidden_size)
        if self._caft:
            patch_decoder_for_caft(decoder_cls, model_config.hf_config.hidden_size)

        super().__init__(config, model_config, device_mesh)

        if requires_patch:
            layers = self._get_decoder_layers()

            if self._steering:
                vector = torch.load(self._steering_path, weights_only=True)
                setup_steering(layers, self._steering_layer, vector, self._steering_alpha)
                logger.info(f"Steering: layer={self._steering_layer} (decoder {self._steering_layer - 1}), alpha={self._steering_alpha}, "
                            f"norm={vector.float().norm():.4f}")
            elif self._activation_cache:
                self._capture_state = setup_capture(
                    layers, self._decoder_capture_layers, self._get_model_runner(),
                )
                logger.info(f"Capture: user layers={self._capture_layers} -> decoder layers={self._decoder_capture_layers}")
            elif self._caft:
                dir_matrix = torch.load(self._caft_path, weights_only=True)
                dir_matrix = setup_caft(layers, self._caft_layer, dir_matrix, self._caft_orthogonalize)
                logger.info(f"CAFT: layer={self._caft_layer} (decoder {self._caft_layer - 1}), shape={dir_matrix.shape}, "
                            f"norm={dir_matrix.float().norm():.4f}")

    def _get_model_runner(self):
        return self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner

    def _get_decoder_layers(self) -> list[nn.Module]:
        return list(self._get_model_runner().model.model.layers)

    @torch.no_grad()
    def generate_sequences(self, prompts, **kwargs):
        if not self._activation_cache:
            return super().generate_sequences(prompts, **kwargs)

        batch_size = prompts.batch["input_ids"].size(0)
        layers = self._get_decoder_layers()

        # Reset per-request accumulators
        for idx in self._decoder_capture_layers:
            layer = layers[idx]
            layer._capture_on.fill_(1.0)
            layer._decode_phase.fill_(0.0)
        self._capture_state["act_sum"].clear()
        self._capture_state["act_count"].clear()
        self._capture_state["req_id_to_slot"].clear()
        if "req_prefill_seen" in self._capture_state:
            self._capture_state["req_prefill_seen"].clear()

        output = super().generate_sequences(prompts, **kwargs)

        for idx in self._decoder_capture_layers:
            layers[idx]._capture_on.fill_(0.0)

        # Build activations from per-request Python-side accumulators.
        # NOTE: Online decode capture observes hidden states for tokens that are
        # actually forwarded as inputs on decode steps, so the final generated
        # token is not included (there is no subsequent forward pass after EOS /
        # stop / max_tokens). Expected count ~= response_len - 1.
        act_sum = self._capture_state["act_sum"]
        act_count = self._capture_state["act_count"]
        activations = []
        for layer_idx in self._decoder_capture_layers:
            layer = layers[layer_idx]
            layer_acts = torch.zeros(batch_size, layer._step_hidden.shape[1],
                                     dtype=torch.float32, device=layer._step_hidden.device)
            for slot in range(batch_size):
                if slot in act_sum and layer_idx in act_sum[slot]:
                    count = max(act_count[slot][layer_idx], 1)
                    layer_acts[slot] = act_sum[slot][layer_idx] / count
            activations.append(layer_acts)

        stacked = torch.stack(activations, dim=0)
        if stacked.shape[0] == 1:
            stacked = stacked.squeeze(0)
        output.batch["activations"] = stacked
        return output


# Register in verl's rollout registry so name="interp_vllm" resolves to this class
_ROLLOUT_REGISTRY[("interp_vllm", "sync")] = "src.train.verl.workers.interp_vllm_rollout.InterpvLLMRollout"
