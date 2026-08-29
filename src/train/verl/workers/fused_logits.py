"""Fused lm_head + log-prob forward for decoders that soft-cap their logits.

verl's fused path replaces the model's head with a chunked `hidden @ lm_head.weight.T` so the
[tokens, vocab] logits never exist all at once, which for a 262k-token vocabulary is the
difference between a micro batch of one sequence and one of four. That substitution is exact for
most decoders, but Gemma squashes its logits through `softcap * tanh(logits / softcap)` before the
softmax, and verl's kernels have no term for it: fusing as-is would train against a different
distribution from the one the rollout sampled, which is why the fused path used to be switched off
for Gemma 4 outright.

The cap is an elementwise, monotone map on the logits, so it composes with the chunking without
changing anything else: this module supplies a forward that applies it inside each chunk and
installs it in place of verl's for the models that need it. Chunks are re-run under
`torch.utils.checkpoint` in the backward pass rather than kept, which is what keeps the peak at
one chunk's logits, and lets autograd derive the tanh gradient instead of us hand-writing it.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from verl.models.transformers.dense_common import CausalLMOutputForPPO

__all__ = ["install", "fused_log_probs_and_entropy", "forward_with_softcap_torch_backend"]

# Tokens per chunk. Matches verl's own fused kernels, and sets the peak: one chunk's fp32 logits.
CHUNK_SIZE = 512


def _chunk_log_probs_and_entropy(hidden_states, vocab_weights, labels, temperature, softcap):
    """Log-prob of each label and the entropy of its distribution, for one chunk of tokens."""
    logits = hidden_states @ vocab_weights.t()
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    entropy_dtype = logits.dtype
    logits = (logits / temperature).float()

    log_normalizer = torch.logsumexp(logits, dim=-1)
    log_probs = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - log_normalizer
    probs = torch.softmax(logits, dim=-1)
    entropy = log_normalizer - torch.sum(probs * logits, dim=-1)
    return log_probs, entropy.to(entropy_dtype)


def fused_log_probs_and_entropy(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    softcap: float | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """verl's fused lm_head + log-prob/entropy, with the model's final logit softcap applied.

    `hidden_states` is (..., hidden) and `labels` the matching (...) of already-rolled next
    tokens; both outputs come back shaped like `labels`, log-probs in fp32 as verl's kernels
    return them and the entropy in the model's dtype.
    """
    leading_shape = labels.shape
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1).to(torch.int64)

    log_probs, entropy = [], []
    for start in range(0, flat_labels.shape[0], chunk_size):
        chunk = slice(start, start + chunk_size)
        chunk_log_probs, chunk_entropy = checkpoint(
            _chunk_log_probs_and_entropy,
            flat_hidden[chunk],
            vocab_weights,
            flat_labels[chunk],
            temperature,
            softcap,
            use_reentrant=False,
        )
        log_probs.append(chunk_log_probs)
        entropy.append(chunk_entropy)

    return (
        torch.cat(log_probs).view(leading_shape),
        torch.cat(entropy).view(leading_shape),
    )


def final_logit_softcapping(model) -> float | None:
    """The cap the model squashes its logits with before the softmax, if it uses one."""
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    return getattr(text_config, "final_logit_softcapping", None)


def forward_with_softcap_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: torch.LongTensor = None,
    temperature: float = 1.0,
    shift_labels: torch.LongTensor = None,
    return_dict: bool = True,
    use_cache: bool | None = None,
    **kwargs,
):
    """Drop-in for verl's `forward_with_torch_backend` that keeps the final logit softcap.

    Kept signature-compatible with verl's: the engine passes `temperature`, `use_cache`,
    `return_dict` and - on the packed path - the `shift_labels` it rolled over the whole global
    sequence, and reads `log_probs`/`entropy` back off the result. Everything else is the base
    model's own forward, so a `cu_seqlens` parameter is deliberately absent here too (the engine
    probes the patched forward's signature for one, and Gemma 4 takes no such argument).
    """
    if not return_dict:
        raise NotImplementedError("forward_with_softcap_torch_backend has to return_dict")

    # `use_cache` is named only so a caller's value is swallowed rather than colliding with the
    # one below (verl's engine passes `use_cache=False` of its own accord). It has to be False and
    # not merely left to default: transformers would fall back to the config's `use_cache=True`
    # and build a cache, and a cache is what turns off the packed-sequence detection that reads
    # the batch's document boundaries off `position_ids`.
    outputs = self.model(input_ids=input_ids, use_cache=False, **kwargs)
    hidden_states = outputs.last_hidden_state

    # Same precedence as verl: the engine's globally-rolled labels win over rolling here, which
    # on a sequence-parallel shard would wrap around the shard boundary instead of the sequence.
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError(
            "To use forward_with_softcap_torch_backend, either labels or input_ids must be provided."
        )

    log_probs, entropy = fused_log_probs_and_entropy(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        labels=rolled_labels,
        temperature=temperature,
        softcap=final_logit_softcapping(self),
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def install() -> None:
    """Route verl's fused-kernel patching through the softcap-aware forward where it is needed.

    verl picks the fused forward by model type and falls back to a generic one that reads
    `lm_head` directly; there is no hook for a model that transforms its logits afterwards, so
    the choice itself is what gets wrapped. Models without a softcap are untouched.
    """
    from verl.models.transformers import monkey_patch

    original = monkey_patch.patch_forward_with_backends

    def patch_forward_with_backends(model, use_fused_kernels=False, fused_kernels_backend=None):
        if not use_fused_kernels or final_logit_softcapping(model) is None:
            return original(
                model,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )
        if fused_kernels_backend != "torch":
            raise ValueError(
                f"{model.__class__.__name__} soft-caps its logits, which only the torch fused "
                f"backend is patched for here, but fused_kernels_backend is "
                f"{fused_kernels_backend!r}. Set "
                "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch."
            )
        model.__class__.forward = forward_with_softcap_torch_backend
        print(
            f"Using softcap-aware torch backend for fused kernels in {model.__class__.__name__} "
            f"(final_logit_softcapping={final_logit_softcapping(model)})"
        )

    monkey_patch.patch_forward_with_backends = patch_forward_with_backends
