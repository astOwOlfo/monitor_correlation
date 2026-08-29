"""FlexAttention for decoders whose heads are too wide for the default Triton block sizes.

Gemma 4's full-attention layers use a 512-wide head (`global_head_dim`), which is twice
FlashAttention 2's limit, so those runs have to fall back to another kernel. FlexAttention is the
natural choice - it takes arbitrary head dimensions and turns a packed batch's block-diagonal
document mask into a `BlockMask` it can skip over - except that Inductor's default configs stage
128x128 query/key blocks, which at 512 wide want ~900KB of shared memory against the ~228KB an
SM has. Every candidate config is rejected and the compile fails with "No valid triton configs".

Shrinking the blocks for the oversized heads brings the working set back under the limit. That is
the only change: the narrower sliding-attention layers keep Inductor's own autotuned choice, and
so does every other model, which is why this is registered as its own attention implementation
rather than as an override of the built-in `flex_attention`.
"""

from __future__ import annotations

from transformers.integrations.flex_attention import flex_attention_forward
from transformers.masking_utils import AttentionMaskInterface, flex_attention_mask
from transformers.modeling_utils import AttentionInterface

__all__ = ["ATTENTION_IMPLEMENTATION", "register"]

# The name to pass as `attn_implementation`; registered with transformers below.
ATTENTION_IMPLEMENTATION = "flex_attention_wide_head"

# Head dimensions up to this are within reach of Inductor's default FlexAttention configs.
MAX_DEFAULT_HEAD_DIM = 256

# The largest square block that fits a 512-wide head in an H100 SM's shared memory. The four
# numbered keys are the backward kernel's blocks; the unnumbered pair is the forward's.
_BLOCK_SIZE = 32
_BLOCK_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")


def kernel_options(head_dim: int) -> dict[str, int] | None:
    """Triton block sizes for a head this wide, or None to leave Inductor to autotune."""
    if head_dim <= MAX_DEFAULT_HEAD_DIM:
        return None
    return dict.fromkeys(_BLOCK_KEYS, _BLOCK_SIZE)


def flex_attention_wide_head_forward(module, query, key, value, attention_mask, **kwargs):
    """`flex_attention_forward` with block sizes sized to this layer's head dimension."""
    if kwargs.get("kernel_options") is None:
        options = kernel_options(query.shape[-1])
        if options is not None:
            kwargs["kernel_options"] = options
    return flex_attention_forward(module, query, key, value, attention_mask, **kwargs)


def register() -> None:
    """Make `ATTENTION_IMPLEMENTATION` selectable as an `attn_implementation`.

    The mask entry matters as much as the kernel one: without it transformers has no mask builder
    registered under this name and hands the model no mask at all, which silently drops both
    causality and the packed batch's document boundaries.
    """
    AttentionInterface.register(ATTENTION_IMPLEMENTATION, flex_attention_wide_head_forward)
    AttentionMaskInterface.register(ATTENTION_IMPLEMENTATION, flex_attention_mask)
