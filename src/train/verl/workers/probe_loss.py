"""Probe-based loss for RL training.

Captures activations from HF model decoder layers during forward pass and
applies a probe to compute a differentiable loss term.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from src.monitor import probe

logger = logging.getLogger(__name__)


@dataclass
class ProbeLossConfig:
    """Configuration for probe-based loss."""
    enabled: bool = False
    probe_path: str | None = None
    layers: list[int] | None = None  # HF layer indices (1-indexed, where 1 = output of first decoder layer)
    coeff: float = 1.0
    aggregation: str = "mean"  # "mean" or "max" across layers
    retrain_freq: int = 0 # Re-train probe every N steps; 0 = disabled
    retrain_dataset_path: str | None = None # Path to static labeled dataset (parquet with prompt, response, label columns)


class ProbeLossModule:
    """Computes differentiable probe loss from HF model activations.

    Registers forward hooks on target decoder layers to capture activations,
    then applies a probe to compute loss = coeff * probe_output.

    Higher probe output (closer to 1) means higher loss (penalizing the behavior
    the probe was trained to detect).
    """

    def __init__(
        self,
        model: nn.Module,
        probe_path: str,
        layers: list[int],
        coeff: float = 1.0,
        aggregation: str = "mean",
    ):
        """Initialize probe loss module.

        Args:
            model: HF model (potentially FSDP-wrapped) to hook into
            probe_path: Path to saved probe file
            layers: Layer indices to capture (1-indexed HF convention)
            coeff: Coefficient to scale the probe loss
            aggregation: How to aggregate across layers ("mean" or "max")
        """
        self.coeff = coeff
        self.aggregation = aggregation
        self.layers = layers
        self._decoder_layers = [l - 1 for l in layers]  # Convert to 0-indexed

        assert all(l >= 1 for l in layers), (
            f"Layers must be >= 1 (layer 0 is embedding output). Got: {layers}"
        )

        # Load probe
        self.probe = probe.load_probe(probe_path)
        logger.info(f"Loaded probe from {probe_path} for layers {layers}")
        assert self.probe.layers is not None, "Loaded probe has no layers; ensure probe was saved with layers"
        assert set(self.layers).issubset(set(self.probe.layers)), (
            f"Probe layers {self.probe.layers} do not cover requested layers {self.layers}"
        )

        # Storage for captured activations (will be populated by hooks)
        self._activations: dict[int, torch.Tensor] = {}
        self._response_mask: torch.Tensor | None = None

        # Register hooks
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks(model)

    def _get_decoder_layers(self, model: nn.Module) -> list[nn.Module]:
        """Find decoder layers in a potentially FSDP/LoRA-wrapped HF model."""
        for module in model.modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                first = module[0]
                if hasattr(first, 'self_attn') and hasattr(first, 'mlp'):
                    return list(module)
        raise RuntimeError("Could not find decoder layers in HF model")

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on target decoder layers.

        Note: layer N captures the output of decoder layer N-1, matching
        hidden_states[N] from HF. The last decoder layer output does NOT include
        the final RMSNorm, so layer=num_hidden_layers would differ from
        hidden_states[num_hidden_layers] which is post-norm.
        """
        decoder_layers = self._get_decoder_layers(model)
        num_decoder_layers = len(decoder_layers)
        assert all(1 <= l < num_decoder_layers for l in self.layers), (
            f"Invalid residual-stream layers {self.layers}. Must be in [1, {num_decoder_layers - 1}] "
            "for hook capture (layer 0 is embedding output; layer N is post-norm in HF)."
        )
        assert all(idx < num_decoder_layers for idx in self._decoder_layers), (
            f"Decoder layer indices {self._decoder_layers} out of range for "
            f"{num_decoder_layers} decoder layers (max layer={num_decoder_layers})"
        )

        for layer_idx in self._decoder_layers:
            layer = decoder_layers[layer_idx]

            def make_hook(idx: int) -> Callable:
                def hook(module, input, output):
                    # output is typically (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    # Store with gradient tracking
                    self._activations[idx] = hidden_states
                return hook

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)
            logger.info(f"Registered probe hook on decoder layer {layer_idx}")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def clear_activations(self) -> None:
        """Clear stored activations."""
        self._activations.clear()
        self._response_mask = None

    def set_response_mask(self, response_mask: torch.Tensor) -> None:
        """Set the response mask for computing mean activations over response tokens."""
        self._response_mask = response_mask

    def compute_loss(self, response_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute probe loss from captured activations.

        Args:
            response_mask: (batch_size, seq_len) mask for response tokens.
                          If None, uses previously set mask or averages over all tokens.

        Returns:
            loss: Scalar loss tensor (differentiable)
            metrics: Dict of metrics for logging
        """
        if not self._activations:
            raise RuntimeError("No activations captured. Did the forward pass run?")

        if response_mask is None:
            response_mask = self._response_mask

        # Collect activations from target layers
        # Each activation is (batch_size, seq_len, hidden_dim)
        layer_activations = []
        for layer_idx in self._decoder_layers:
            if layer_idx not in self._activations:
                raise RuntimeError(f"Layer {layer_idx} not captured")
            layer_activations.append(self._activations[layer_idx])

        # Stack: (n_layers, batch_size, seq_len, hidden_dim)
        stacked = torch.stack(layer_activations, dim=0)

        # Compute mean activation over response tokens for each sample
        if response_mask is not None:
            # response_mask: (batch_size, seq_len)
            # We need to handle the case where seq_len might differ
            batch_size, seq_len = response_mask.shape
            act_seq_len = stacked.shape[2]

            # Align mask to activation sequence length (take last act_seq_len tokens)
            if seq_len != act_seq_len:
                # Activations are for full sequence, mask is for response only
                # Pad mask to match or slice appropriately
                if seq_len < act_seq_len:
                    # Pad mask at the beginning (prompt tokens get 0)
                    pad_size = act_seq_len - seq_len
                    response_mask = torch.nn.functional.pad(response_mask, (pad_size, 0), value=0)
                else:
                    # Slice mask to match
                    response_mask = response_mask[:, -act_seq_len:]

            # Expand mask for broadcasting: (1, batch_size, seq_len, 1)
            mask_expanded = response_mask.unsqueeze(0).unsqueeze(-1).float()

            # Compute masked mean over sequence dimension
            masked_sum = (stacked * mask_expanded).sum(dim=2)
            mask_count = mask_expanded.sum(dim=2).clamp(min=1)
            mean_activations = masked_sum / mask_count  # (n_layers, batch_size, hidden_dim)
        else:
            # Simple mean over sequence
            mean_activations = stacked.mean(dim=2)  # (n_layers, batch_size, hidden_dim)

        # Apply probe (differentiable)
        # probe.predict_proba expects (n_layers, n_samples, hidden_dim)
        # and returns (n_samples, n_layers)
        probe_probs = self.probe.predict_proba(
            mean_activations,
            layers=self.layers
        )  # (batch_size, n_layers)

        # Aggregate across layers
        if self.aggregation == "max":
            probe_score = probe_probs.max(dim=1).values  # (batch_size,)
        else:  # mean
            probe_score = probe_probs.mean(dim=1)  # (batch_size,)

        # Loss is mean probe score across batch
        loss = probe_score.mean() * self.coeff

        # Metrics
        metrics = {
            "actor/probe_loss": loss.detach().item(),
            "actor/probe_score_mean": probe_score.mean().detach().item(),
            "actor/probe_score_min": probe_score.min().detach().item(),
            "actor/probe_score_max": probe_score.max().detach().item(),
        }

        # Per-layer metrics
        for i, layer in enumerate(self.layers):
            metrics[f"actor/probe_score_layer{layer}"] = probe_probs[:, i].mean().detach().item()

        return loss, metrics


    def reload_probe(self, probe_state: dict) -> None:
        """Hot-swap the probe with a newly trained one.

        Args:
            probe_state: Dict with 'file_extension' and probe-specific state
                         (e.g. 'clf' for LogisticRegressionProbe, 'direction'+'layers' for MassMeanProbe)
        """
        from src.train.verl.workers.extended_workers import reconstruct_probe_from_state
        new_probe = reconstruct_probe_from_state(probe_state)
        self.probe = new_probe
        logger.info(f"Probe reloaded: type={probe_state['file_extension']}, layers={new_probe.layers}")


def create_probe_loss_module(
    model: nn.Module,
    config: ProbeLossConfig | dict,
) -> ProbeLossModule | None:
    """Create a ProbeLossModule from config if enabled.

    Args:
        model: HF model to hook into
        config: ProbeLossConfig or dict with probe loss settings

    Returns:
        ProbeLossModule if enabled, None otherwise
    """
    if isinstance(config, dict):
        config = ProbeLossConfig(**config)

    if not config.enabled:
        return None

    if config.probe_path is None:
        raise ValueError("probe_path must be specified when probe loss is enabled")
    if config.layers is None:
        raise ValueError("layers must be specified when probe loss is enabled")

    return ProbeLossModule(
        model=model,
        probe_path=config.probe_path,
        layers=config.layers,
        coeff=config.coeff,
        aggregation=config.aggregation,
    )
