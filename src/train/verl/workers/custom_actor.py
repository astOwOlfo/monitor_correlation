"""Custom PPO Actor with modifiable update_policy method.

This module provides a subclass of DataParallelPPOActor that allows customizing
the policy update logic without modifying the verl library directly.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from src.monitor.probe import align_sequence_mask
from src.train.verl.workers.extended_workers import _get_hf_decoder_layers
from verl import DataProto
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor, logger

if TYPE_CHECKING:
    from src.monitor.probe import Probe
    from src.train.verl.workers.probe_loss import ProbeLossModule

__all__ = ["CustomPPOActor"]


class _EarlyExitException(Exception):
    """Raised by forward hook to stop the forward pass after the last target layer."""
    pass


class CustomPPOActor(DataParallelPPOActor):
    """Custom PPO Actor with modifiable update_policy and optional probe loss.

    Supports an optional probe loss module that computes a differentiable loss
    term based on model activations during the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_loss_module: ProbeLossModule | None = None
        self.activation_cache_probe: Probe | None = None
        self.activation_cache_layers: list[int] | None = None
        self.activation_cache_batch_size: int = 16
        self.activation_cache_position: str = "response_avg"

    def set_probe_loss_module(self, module: ProbeLossModule | None) -> None:
        """Set the probe loss module for computing activation-based loss."""
        self.probe_loss_module = module

    def _log_cuda_mem(self, tag: str) -> None:
        """Log concise CUDA memory stats for OOM debugging."""
        if not torch.cuda.is_available():
            return
        device = get_device_id()
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_alloc = torch.cuda.max_memory_allocated(device)
        gib = 1024 ** 3
        msg = (
            f"{tag}: free={free / gib:.2f}GiB total={total / gib:.2f}GiB "
            f"allocated={allocated / gib:.2f}GiB reserved={reserved / gib:.2f}GiB "
            f"max_alloc={max_alloc / gib:.2f}GiB"
        )
        logger.warning(msg)
        print(msg, flush=True)

    def set_activation_cache(
        self,
        layers: list[int],
        position: str,
        batch_size: int,
        probe: Probe | None = None,
    ) -> None:
        """Set activation-caching configuration for hook-time inference."""
        assert position in {"response_avg", "response_all"}, f"Unsupported probe cache position: {position}"
        assert len(layers) > 0, "Probe cache requires non-empty layers"
        assert batch_size > 0, f"batch_size must be > 0, got {batch_size}"

        self.activation_cache_probe = probe
        self.activation_cache_layers = layers
        self.activation_cache_batch_size = batch_size
        self.activation_cache_position = position
        self.activation_cache_supports_mask = False
    
        if probe is not None and getattr(probe, "requires_sequence", False):
            assert position == "response_all", "Sequence probe requires position='response_all'"
            self.activation_cache_supports_mask = "mask" in inspect.signature(self.activation_cache_probe.predict_proba).parameters
        
        

    @torch.no_grad()
    def cache_activations(self, input_ids, attention_mask, position_ids, response_mask):
        """Cache activations from hooks, or probe probabilities when probe is configured."""

        layers = self.activation_cache_layers
        batch_size = self.activation_cache_batch_size

        was_training = self.actor_module.training
        self.actor_module.eval()

        decoder_layers = _get_hf_decoder_layers(self.actor_module)
        decoder_indices = [l - 1 for l in layers] # 0-indexing correction
        n_samples = input_ids.shape[0]
        device = next(self.actor_module.parameters()).device
        outputs = []

        assert self.activation_cache_position in {"response_avg", "response_all"}, (
            f"Unsupported activation cache position: {self.activation_cache_position}"
        )
        hooks = []
        captured: dict[int, torch.Tensor] = {}

        try:
            # Move probe to device
            if self.activation_cache_probe is not None:
                self.activation_cache_probe.move_to_device(device=device, layers=layers)
            
            # Register forward hooks
            for idx in decoder_indices:
                layer_module = decoder_layers[idx]

                def _hook(module, input, output, _layer=idx + 1):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[_layer] = h.detach().float()

                hooks.append(layer_module.register_forward_hook(_hook))

            # Early exit hook to stop the forward pass after the last target layer
            early_exit_idx = max(decoder_indices) + 1
            if early_exit_idx < len(decoder_layers):
                def _exit_hook(module, input, output):
                    raise _EarlyExitException()
                hooks.append(decoder_layers[early_exit_idx].register_forward_hook(_exit_hook))

            # Run caching in micro-batches
            for start in range(0, n_samples, batch_size):

                # Extract micro-batch
                end = min(start + batch_size, n_samples)
                mb_input_ids = input_ids[start:end].to(device)
                mb_attention_mask = attention_mask[start:end].to(device)
                mb_position_ids = position_ids[start:end].to(device)
                captured.clear()

                try:
                    self.actor_module(
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                        position_ids=mb_position_ids,
                        return_dict=True,
                    )
                except _EarlyExitException:
                    pass

                assert len(captured) == len(layers), f"Expected captures for {layers}, got {sorted(captured.keys())}"
                assert all(layer in captured for layer in layers), f"Missing captured layers for {layers}, got {sorted(captured.keys())}"

                sample_hidden = captured[layers[0]]
                mask = align_sequence_mask(
                    response_mask[start:end],
                    target_seq_len=sample_hidden.shape[1],
                    device=sample_hidden.device,
                )
                
                stacked = torch.stack([captured[layer] for layer in layers], dim=0)

                if self.activation_cache_position == "response_avg":
                    mask_f = mask.to(device=stacked.device, dtype=stacked.dtype).unsqueeze(0).unsqueeze(-1)
                    stacked = (stacked * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1.0)
                else:
                    invalid = ~mask.unsqueeze(0).unsqueeze(-1)
                    stacked = stacked.masked_fill(invalid, float("nan"))

                if self.activation_cache_probe is not None:
                    if self.activation_cache_supports_mask:
                        stacked = self.activation_cache_probe.predict_proba(stacked, layers=layers, mask=mask)
                    else:
                        stacked = self.activation_cache_probe.predict_proba(stacked, layers=layers)
                    assert stacked.ndim == 2 and stacked.shape[0] == end - start, (
                        f"Expected probabilities shape (n_samples={end - start}, ?), got {stacked.shape}"
                    )
                
                outputs.append(stacked.detach().float())
                
        finally:
            # Remove hooks and training module if needed
            for hook in hooks:
                hook.remove()
            if self.activation_cache_probe is not None:
                self.activation_cache_probe.move_to_device(device=torch.device("cpu"), layers=layers)
            if was_training:
                self.actor_module.train()

        if self.activation_cache_probe is not None:
            outputs = torch.cat(outputs, dim=0)
            assert outputs.ndim == 2, f"Expected probe probs with shape (n_samples, n_layers), got {outputs.shape}"
        else:
            outputs = torch.cat(outputs, dim=1)
            if outputs.shape[0] == 1:
                outputs = outputs.squeeze(0)
            
            if self.activation_cache_position == "response_avg":
                assert outputs.ndim in {2, 3}, f"Expected activations to be 2D/3D, got {outputs.shape}"
            else:
                assert outputs.ndim in {3, 4}, f"Expected activations to be 3D/4D, got {outputs.shape}"

        return {"activations": outputs}



    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Update policy with custom loss terms.

        This is a copy of DataParallelPPOActor.update_policy that can be modified
        to add custom differentiable loss terms.

        Available data in `data.batch`:
            - responses: token ids of responses
            - response_mask: mask for response tokens
            - input_ids: full input (prompt + response)
            - attention_mask: attention mask
            - position_ids: position ids
            - old_log_probs: log probs from policy before update
            - advantages: computed advantages
            - ref_log_prob: reference policy log probs (if use_kl_loss=True)
            - rollout_is_weights: importance sampling weights (optional)
            - rollout_log_probs: rollout policy log probs (optional)

        Available data in `data.non_tensor_batch`:
            - extra_info: dict with problem metadata, answers, etc.
            - multi_modal_inputs: for vision models (optional)
        """
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        self._log_cuda_mem(f"update_policy: data_size={len(data)}, ppo_mini_batch_size={self.config.ppo_mini_batch_size}, "
                    f"n_mini_batches={len(mini_batches)}, ppo_epochs={self.config.ppo_epochs}, on_policy={on_policy}")

        metrics = {"actor/on_policy": float(on_policy), "actor/n_mini_batches": len(mini_batches), "actor/n_micro_batches": 0}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                metrics['actor/n_micro_batches'] += len(micro_batches)
                for micro_idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Clear probe activations before forward pass
                    if self.probe_loss_module is not None:
                        self.probe_loss_module.clear_activations()

                    # all return: (bsz, response_length)
                    calculate_entropy = entropy_coeff != 0
                    if micro_idx == 0 or micro_idx == len(micro_batches) - 1:
                        self._log_cuda_mem(
                            f"update_policy before_forward batch={batch_idx} "
                            f"micro={micro_idx}/{len(micro_batches) - 1} input_shape={tuple(model_inputs['input_ids'].shape)}"
                        )
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    # Extract pre-computed rollout correction weights if present
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Rollout correction metrics (if not using pure rollout correction mode)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    # Entropy loss
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    # KL loss
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # Probe loss (if enabled)
                    if self.probe_loss_module is not None:
                        probe_loss, probe_metrics = self.probe_loss_module.compute_loss(response_mask)
                        policy_loss = policy_loss + probe_loss
                        micro_batch_metrics.update(probe_metrics)

                    # Scale and backward
                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if micro_idx == 0 or micro_idx == len(micro_batches) - 1:
                        self._log_cuda_mem(
                            f"update_policy after_backward batch={batch_idx} "
                            f"micro={micro_idx}/{len(micro_batches) - 1}"
                        )

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
            
        metrics['actor/n_micro_batches_avg'] = metrics['actor/n_micro_batches']

        self.actor_optimizer.zero_grad()
        return metrics
