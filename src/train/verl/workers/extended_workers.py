"""Extended actor-rollout workers that apply interventions to the HF actor model.

When steering or CAFT is configured, the vLLM rollout model is patched at class
level (see interp_vllm_rollout.py). This module mirrors those interventions on
the HuggingFace actor model via forward hooks so that the same transformation
is applied during both generation (vLLM) and training (HF).
"""
from __future__ import annotations

import logging

from typing import Any, Dict

import json
import torch
import torch.nn as nn

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.fsdp_utils import collect_lora_params
from verl.utils.device import get_device_id

from src.monitor.probe import load_probe, PROBE_REGISTRY
from src.train.verl.workers.interp_vllm_rollout import InterpvLLMRollout

logger = logging.getLogger(__name__)


def _get_hf_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Find decoder layers in a potentially FSDP/LoRA-wrapped HF model."""
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, 'self_attn') and hasattr(first, 'mlp'):
                return list(module)
    raise RuntimeError("Could not find decoder layers in HF actor model")


def _apply_hf_intervention(actor_module: nn.Module, rollout) -> None:
    """Apply steering/CAFT forward hooks to HF actor model decoder layers.

    Loads vectors from file and registers forward hooks on the target HF decoder
    layer. Uses rollout config attributes (Python scalars) to avoid accessing
    vLLM CUDA buffers which may not be safely accessible from the worker process.
    """
    

    if not isinstance(rollout, InterpvLLMRollout):
        return
    if not (rollout._steering or rollout._caft):
        return

    hf_layers = _get_hf_decoder_layers(actor_module)

    if rollout._steering:
        vector = torch.load(rollout._steering_path, weights_only=True)
        hf_target = hf_layers[rollout._steering_layer - 1]
        param = next(hf_target.parameters())
        vec = vector.to(device=param.device, dtype=param.dtype)
        alpha = rollout._steering_alpha

        hf_target.register_buffer('_steer_vec', vec, persistent=False)
        hf_target._steer_alpha = alpha

        def _hook(module, input, output):
            if isinstance(output, tuple):
                return (output[0] + module._steer_alpha * module._steer_vec,) + output[1:]
            return output + module._steer_alpha * module._steer_vec

        hf_target.register_forward_hook(_hook)
        logger.info(f"HF actor steering: layer={rollout._steering_layer}, alpha={alpha}, "
                    f"norm={vec.float().norm():.4f}")

    elif rollout._caft:
        dir_matrix = torch.load(rollout._caft_path, weights_only=True)
        if dir_matrix.ndim == 1:
            dir_matrix = dir_matrix.unsqueeze(1)
        if rollout._caft_orthogonalize:
            dir_matrix = torch.linalg.qr(dir_matrix.float()).Q
        hf_target = hf_layers[rollout._caft_layer - 1]
        param = next(hf_target.parameters())
        matrix = dir_matrix.to(device=param.device, dtype=param.dtype)

        hf_target.register_buffer('_caft_matrix', matrix, persistent=False)

        def _hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                return (h - (h @ module._caft_matrix) @ module._caft_matrix.T,) + output[1:]
            return output - (output @ module._caft_matrix) @ module._caft_matrix.T

        hf_target.register_forward_hook(_hook)
        logger.info(f"HF actor CAFT: layer={rollout._caft_layer}, shape={matrix.shape}")


def _init_probe_loss(actor, actor_module: nn.Module, probe_loss_config: dict) -> None:
    """Initialize probe loss module from algorithm.probe_loss config."""
    from src.train.verl.workers.probe_loss import create_probe_loss_module

    if not probe_loss_config.get('enabled', False):
        return

    probe_loss_module = create_probe_loss_module(actor_module, probe_loss_config)
    if probe_loss_module is not None:
        actor.set_probe_loss_module(probe_loss_module)
        logger.info(f"Initialized probe loss: layers={probe_loss_config.get('layers')}, "
                    f"coeff={probe_loss_config.get('coeff', 1.0)}")


def _init_activation_cache(actor, config) -> None:
    """Initialize actor-side activation cache config and optional probe."""
    if isinstance(config, dict):
        rollout_cfg = config["actor_rollout_ref"]["rollout"] if "actor_rollout_ref" in config else config["rollout"]
    else:
        rollout_cfg = config.actor_rollout_ref.rollout if "actor_rollout_ref" in config else config.rollout
    interp_cfg = rollout_cfg.engine_kwargs.get("interp_vllm", {})
    activation_cache_cfg = interp_cfg.get("activation_cache", {})
    if activation_cache_cfg.get("mode", None) != "actor_rollout":
        return

    layers = activation_cache_cfg.get("layers", None)
    assert layers is not None and len(layers) > 0, "activation_cache.layers must be set for mode='actor_rollout'"
    layers = list(layers)
    position = activation_cache_cfg.get("position", "response_avg")
    batch_size = int(activation_cache_cfg.get("batch_size", 16))

    probe_path = activation_cache_cfg.get("probe_path", None)
    loaded_probe = load_probe(path=probe_path) if probe_path is not None else None
    actor.set_activation_cache(layers=layers, position=position, batch_size=batch_size, probe=loaded_probe)
    if loaded_probe is None:
        logger.info(
            f"Initialized activation-cache: probe=None, layers={layers}, position={position}, batch_size={batch_size}"
        )
    else:
        logger.info(
            f"Initialized activation-cache probe: path={probe_path}, layers={layers}, "
            f"position={position}, batch_size={batch_size}, file_extension={loaded_probe.file_extension}"
        )


def _cache_actor_rollout_activations(actor, data: DataProto) -> DataProto:
    """Cache actor-rollout activations (or probe probabilities when configured)."""
    result = actor.cache_activations(
        input_ids=data.batch["input_ids"],
        attention_mask=data.batch["attention_mask"],
        position_ids=data.batch["position_ids"],
        response_mask=data.batch["response_mask"],
    )
    acts = result["activations"]
    if getattr(actor, "activation_cache_probe", None) is not None:
        assert acts.ndim == 2, f"Expected probe probabilities with shape (n_samples, n_layers), got {acts.shape}"
    elif getattr(actor, "activation_cache_position", "response_avg") == "response_avg":
        assert acts.ndim in {2, 3}, f"Expected response_avg activations to be 2D/3D, got {acts.shape}"
    else:
        assert acts.ndim in {3, 4}, f"Expected response_all activations to be 3D/4D, got {acts.shape}"
    return DataProto.from_single_dict({"activations": acts.cpu()})


def _cache_retrain_activations(actor_module, tokenizer, layers: list[int], prompts_json: list[str], responses: list[str], batch_size: int = 8, per_token: bool = False) -> torch.Tensor:
    """Cache activations using the HF actor model with temporary hooks.

    Registers its own forward hooks on target decoder layers, independent of ProbeLossModule.
    This allows retrain to work regardless of whether probe gradient loss is enabled.

    Args:
        actor_module: The FSDP-wrapped HF model
        tokenizer: HF tokenizer
        layers: Layer indices to capture (1-indexed HF convention)
        prompts_json: JSON-serialized chat message lists
        responses: Response strings
        batch_size: Forward pass batch size
        per_token: If True, return per-token activations with NaN masking instead of response-averaged.
            Required for sequence probes (AttentionProbe, etc.).

    Returns:
        If per_token=False: Tensor of shape (n_layers, n_samples, hidden_dim) on CPU
        If per_token=True: Tensor of shape (n_layers, n_samples, max_response_len, hidden_dim) on CPU,
            with NaN for non-response positions
    """
    decoder_layers_list = _get_hf_decoder_layers(actor_module)
    decoder_indices = [l - 1 for l in layers]  # Convert to 0-indexed

    # Register temporary hooks
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for idx in decoder_indices:
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden
            return hook
        hooks.append(decoder_layers_list[idx].register_forward_hook(make_hook(idx)))

    device = get_device_id()
    was_training = actor_module.training
    actor_module.eval()

    all_layer_acts = []
    response_lens = [] if per_token else None

    with torch.no_grad():
        for i in range(0, len(prompts_json), batch_size):
            batch_prompts = [json.loads(p) for p in prompts_json[i:i + batch_size]]
            batch_responses = responses[i:i + batch_size]

            full_texts, prompt_lens = [], []
            for prompt, response in zip(batch_prompts, batch_responses):
                full_text = prompt + [{"role": "assistant", "content": response}]
                full_texts.append(tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False))
                prompt_chat = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                prompt_lens.append(len(tokenizer.encode(prompt_chat, add_special_tokens=False)))

            inputs = tokenizer(full_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            seq_len = inputs["input_ids"].shape[1]

            # Build response mask: 1 for response tokens, 0 for prompt/padding.
            # Tokenizer uses RIGHT padding (content first, padding at end),
            # so response tokens are at positions [plen, total_tokens).
            response_mask = torch.zeros(len(batch_prompts), seq_len, device=device)
            for j, plen in enumerate(prompt_lens):
                total_tokens = int(inputs["attention_mask"][j].sum().item())
                response_mask[j, plen:total_tokens] = 1.0

            captured.clear()
            actor_module(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                         return_dict=True)

            if per_token:
                # Extract per-token response activations (variable length per sample)
                for j in range(len(batch_prompts)):
                    resp_indices = response_mask[j].bool()
                    resp_len = int(resp_indices.sum().item())
                    response_lens.append(resp_len)
                    sample_acts = []
                    for idx in decoder_indices:
                        sample_acts.append(captured[idx][j, resp_indices].float().cpu())
                    all_layer_acts.append(torch.stack(sample_acts, dim=0))  # (n_layers, resp_len, hidden_dim)
            else:
                layer_acts = []
                for idx in decoder_indices:
                    hidden = captured[idx].float()
                    masked = hidden * response_mask.unsqueeze(-1)
                    mean_act = masked.sum(dim=1) / response_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    layer_acts.append(mean_act)
                all_layer_acts.append(torch.stack(layer_acts, dim=0).cpu())

    # Clean up
    for h in hooks:
        h.remove()
    captured.clear()
    if was_training:
        actor_module.train()

    if per_token:
        # Pad to max response length with NaN
        max_resp_len = max(response_lens)
        n_samples = len(all_layer_acts)
        hidden_dim = all_layer_acts[0].shape[-1]
        n_layers = len(layers)
        padded = torch.full((n_layers, n_samples, max_resp_len, hidden_dim), float("nan"))
        for j, acts in enumerate(all_layer_acts):
            padded[:, j, :acts.shape[1], :] = acts
        return padded  # (n_layers, n_samples, max_response_len, hidden_dim)

    return torch.cat(all_layer_acts, dim=1)  # (n_layers, n_samples, hidden_dim)


def reconstruct_probe_from_state(probe_state: dict):
    """Reconstruct a probe from a serialized state dict (from _serialize_probe).

    Handles both simple probes (LogisticRegression, MassMean) where state keys
    map directly to constructor kwargs, and AttentionProbe variants where
    modules need to be reconstructed from state_dicts.
    """
    ext = probe_state["file_extension"]
    state = {k: v for k, v in probe_state.items() if k != "file_extension"}

    if "modules_state" in state:
        probe_cls = PROBE_REGISTRY[ext]
        probe = probe_cls(**state["config"])
        probe.layers = list(state["layers"])
        for layer, module_state in state["modules_state"].items():
            module = probe._create_module(state["hidden_dim"])
            module.load_state_dict(module_state)
            module.eval()
            probe.modules[layer] = module
        return probe

    return PROBE_REGISTRY[ext](**state)


def _fit_probe_gpu(probe, activations: torch.Tensor, labels: torch.Tensor,
                    layers: list[int], device: torch.device) -> None:
    """GPU-native probe fitting: keeps all data on GPU to avoid CPU<->GPU round-trips.

    The default probe.fit() copies training data to CPU and transfers batches back
    each epoch via DataLoader. With ~15 GB activations per layer, this creates
    terabytes of PCIe transfers across 200 epochs. Instead, we keep everything on
    GPU and use simple index-based batching.
    """
    from src.monitor.probe import pack_sequences

    layer_loader, mask, hidden_dim = probe.create_layer_loader(activations, None, layers)
    probe.layers = layers
    probe.fit_history = {}

    n = len(labels)
    n_val = max(1, int(n * probe.val_fraction))
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    for layer in layers:
        torch.cuda.empty_cache()
        module = probe._create_module(hidden_dim).float().to(device)

        full_acts = layer_loader(layer).to(device)
        mask_gpu = mask.to(device)
        acts_train = full_acts[train_idx]
        acts_val = full_acts[val_idx]
        del full_acts

        flat_val, lengths_val = pack_sequences(acts_val, mask_gpu[val_idx])
        del acts_val

        # _train_module works with GPU tensors directly: .to(device) becomes a no-op
        probe.fit_history[layer] = probe._train_module(
            module, acts_train, mask_gpu[train_idx],
            labels.float()[train_idx].to(device),
            flat_val, lengths_val,
            labels.float()[val_idx].to(device),
            use_amp=True,
        )
        del acts_train, flat_val, lengths_val
        module.eval().cpu()
        probe.modules[layer] = module


def _cache_and_fit_retrain_probe(actor_module, tokenizer, layers: list[int],
                                  prompts_json: list[str], responses: list[str],
                                  labels: torch.Tensor, batch_size: int,
                                  per_token: bool, probe_cls_name: str,
                                  probe_kwargs: dict) -> dict:
    """Cache activations, fit probe, and evaluate — all on GPU. Returns probe + metrics only."""
    import time as _time

    t0 = _time.time()
    activations = _cache_retrain_activations(
        actor_module, tokenizer, layers, prompts_json, responses,
        batch_size=batch_size, per_token=per_token,
    )
    cache_time = _time.time() - t0
    act_shape = activations.shape
    logging.warning(f"Retrain probe: caching done in {cache_time:.1f}s, "
                   f"activations shape={act_shape}, dtype={activations.dtype}")

    t1 = _time.time()
    probe_cls = PROBE_REGISTRY[probe_cls_name]
    probe = probe_cls(**probe_kwargs)
    if per_token and hasattr(probe, 'create_layer_loader'):
        # Sequence probes (AttentionProbe etc.): GPU-native fit to avoid CPU<->GPU round-trips
        device = torch.device("cuda")
        logging.warning(f"Retrain probe: GPU-native fitting, kwargs={probe_kwargs}")
        _fit_probe_gpu(probe, activations, labels, layers, device)
    else:
        # Non-sequence probes (LogisticRegression etc.): standard CPU fit (activations are small)
        logging.warning(f"Retrain probe: CPU fitting, kwargs={probe_kwargs}")
        probe.fit(activations, labels, layers=layers)
    fit_time = _time.time() - t1
    logging.warning(f"Retrain probe: fitting done in {fit_time:.1f}s")

    t2 = _time.time()
    eval_results = probe.evaluate(activations, labels)
    eval_time = _time.time() - t2

    logging.warning(f"Retrain probe: total cache={cache_time:.1f}s, fit={fit_time:.1f}s, "
                   f"eval={eval_time:.1f}s")
    return {
        "probe": probe,
        "eval_results": eval_results,
        "cache_time": cache_time,
        "fit_time": fit_time,
        "eval_time": eval_time,
    }


def _reload_activation_cache_probe(actor, probe_state: dict) -> None:
    """Hot-swap the activation-cache probe on a CustomPPOActor."""
    if not hasattr(actor, 'activation_cache_probe') or actor.activation_cache_probe is None:
        return
    new_probe = reconstruct_probe_from_state(probe_state)
    actor.activation_cache_probe = new_probe
    logger.info(f"Reloaded activation cache probe ({probe_state['file_extension']})")


def _reload_activation_cache_probe_from_path(actor, probe_path: str) -> None:
    """Hot-swap the activation-cache probe by loading from a file path."""
    if not hasattr(actor, 'activation_cache_probe') or actor.activation_cache_probe is None:
        return
    new_probe = load_probe(probe_path)
    actor.activation_cache_probe = new_probe
    logger.info(f"Reloaded activation cache probe from {probe_path}")


class ExtendedActorRolloutRefWorker(ActorRolloutRefWorker):
    """Extended worker with LoRA extraction and HF model intervention support."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Monkey-patch to use CustomPPOActor instead of DataParallelPPOActor
        import verl.workers.actor as actor_module
        from src.train.verl.workers.custom_actor import CustomPPOActor

        original_actor_cls = actor_module.DataParallelPPOActor
        actor_module.DataParallelPPOActor = CustomPPOActor

        try:
            super().init_model()
        finally:
            actor_module.DataParallelPPOActor = original_actor_cls

        if self._is_actor and self._is_rollout:
            _apply_hf_intervention(self.actor_module_fsdp, self.rollout)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_probe_loss(self, probe_loss_config: dict) -> None:
        """Initialize probe loss module from algorithm.probe_loss config."""
        if self._is_actor:
            _init_probe_loss(self.actor, self.actor_module_fsdp, probe_loss_config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_activation_cache(self) -> None:
        """Initialize actor-side activation-cache configuration from worker config."""
        if self._is_actor and self._is_rollout:
            _init_activation_cache(self.actor, self.config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_lora_params(self, layered_summon: bool = False) -> Dict[str, Any]:
        """Extract LoRA parameters from actor_module_fsdp."""
        return collect_lora_params(
            module=self.actor_module_fsdp,
            layered_summon=layered_summon,
            base_sync_done=self.base_sync_done,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_probe(self, probe_state: dict) -> None:
        """Hot-swap the probe in the ProbeLossModule."""
        if self._is_actor and hasattr(self.actor, 'probe_loss_module') and self.actor.probe_loss_module is not None:
            self.actor.probe_loss_module.reload_probe(probe_state)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_activation_cache_probe(self, probe_state: dict) -> None:
        """Hot-swap the probe used for activation-cache scoring."""
        if self._is_actor:
            _reload_activation_cache_probe(self.actor, probe_state)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_activation_cache_probe_from_path(self, probe_path: str) -> None:
        """Hot-swap the activation-cache probe by loading from a file path."""
        if self._is_actor:
            _reload_activation_cache_probe_from_path(self.actor, probe_path)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def cache_activations(self, data: DataProto) -> DataProto:
        """Cache activations using the actor model (or probe outputs when configured)."""
        return _cache_actor_rollout_activations(self.actor, data=data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def cache_retrain_activations(self, prompts_json: list[str], responses: list[str], layers: list[int], batch_size: int = 8, per_token: bool = False) -> dict | None:
        """Cache activations for probe retraining using the existing HF actor model."""
        if not self._is_actor:
            return None
        return {"activations": _cache_retrain_activations(self.actor_module_fsdp, self.tokenizer, layers, prompts_json, responses, batch_size, per_token=per_token)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def cache_and_fit_retrain_probe(self, prompts_json: list[str], responses: list[str],
                                     labels: torch.Tensor, layers: list[int],
                                     batch_size: int, per_token: bool,
                                     probe_cls_name: str, probe_kwargs: dict) -> dict | None:
        """Cache activations and fit probe on GPU in one call. Only rank 0 executes."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if not self._is_actor or rank != 0:
            return None
        logging.warning(f"cache_and_fit_retrain_probe: rank={rank}, n_prompts={len(prompts_json)}, layers={layers}")
        result = _cache_and_fit_retrain_probe(
            self.actor_module_fsdp, self.tokenizer, layers, prompts_json, responses,
            labels, batch_size, per_token, probe_cls_name, probe_kwargs,
        )
        logging.warning(f"cache_and_fit_retrain_probe completed: "
                       f"cache={result['cache_time']:.1f}s, fit={result['fit_time']:.1f}s")
        return result


class ExtendedAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Extended async worker with LoRA extraction and HF model intervention support."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Monkey-patch to use CustomPPOActor instead of DataParallelPPOActor
        import verl.workers.actor as actor_module
        from src.train.verl.workers.custom_actor import CustomPPOActor

        original_actor_cls = actor_module.DataParallelPPOActor
        actor_module.DataParallelPPOActor = CustomPPOActor

        try:
            super().init_model()
        finally:
            actor_module.DataParallelPPOActor = original_actor_cls

        if self._is_actor and self._is_rollout:
            _apply_hf_intervention(self.actor_module_fsdp, self.rollout)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_probe_loss(self, probe_loss_config: dict) -> None:
        """Initialize probe loss module from algorithm.probe_loss config."""
        if self._is_actor:
            _init_probe_loss(self.actor, self.actor_module_fsdp, probe_loss_config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_activation_cache(self) -> None:
        """Initialize actor-side activation-cache configuration from worker config."""
        if self._is_actor and self._is_rollout:
            _init_activation_cache(self.actor, self.config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_lora_params(self, layered_summon: bool = False) -> Dict[str, Any]:
        """Extract LoRA parameters from actor_module_fsdp."""
        return collect_lora_params(
            module=self.actor_module_fsdp,
            layered_summon=layered_summon,
            base_sync_done=self.base_sync_done,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_probe(self, probe_state: dict) -> None:
        """Hot-swap the probe in the ProbeLossModule."""
        if self._is_actor and hasattr(self.actor, 'probe_loss_module') and self.actor.probe_loss_module is not None:
            self.actor.probe_loss_module.reload_probe(probe_state)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_activation_cache_probe(self, probe_state: dict) -> None:
        """Hot-swap the probe used for activation-cache scoring."""
        if self._is_actor:
            _reload_activation_cache_probe(self.actor, probe_state)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_activation_cache_probe_from_path(self, probe_path: str) -> None:
        """Hot-swap the activation-cache probe by loading from a file path."""
        if self._is_actor:
            _reload_activation_cache_probe_from_path(self.actor, probe_path)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def cache_activations(self, data: DataProto) -> DataProto:
        """Cache activations using the actor model (or probe outputs when configured)."""
        return _cache_actor_rollout_activations(self.actor, data=data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def cache_retrain_activations(self, prompts_json: list[str], responses: list[str], layers: list[int], batch_size: int = 8, per_token: bool = False) -> dict | None:
        """Cache activations for probe retraining using the existing HF actor model."""
        if not self._is_actor:
            return None
        return {"activations": _cache_retrain_activations(self.actor_module_fsdp, self.tokenizer, layers, prompts_json, responses, batch_size, per_token=per_token)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def cache_and_fit_retrain_probe(self, prompts_json: list[str], responses: list[str],
                                     labels: torch.Tensor, layers: list[int],
                                     batch_size: int, per_token: bool,
                                     probe_cls_name: str, probe_kwargs: dict) -> dict | None:
        """Cache activations and fit probe on GPU in one call. Only rank 0 executes."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if not self._is_actor or rank != 0:
            return None
        return _cache_and_fit_retrain_probe(
            self.actor_module_fsdp, self.tokenizer, layers, prompts_json, responses,
            labels, batch_size, per_token, probe_cls_name, probe_kwargs,
        )
