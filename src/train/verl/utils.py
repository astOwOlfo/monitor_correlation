from __future__ import annotations

import re
import os
import shutil
from verl import DataProto
from transformers import AutoTokenizer

from src import decode_preserving_reasoning, reasoning_marker_ids


def convert_responses_to_str(data: DataProto, tokenizer: AutoTokenizer):
    prompt_ids = data.batch["prompts"]
    response_ids = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]

    prompt_len = prompt_ids.shape[-1]
    valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    markers = reasoning_marker_ids(tokenizer)

    responses_str = []
    for i in range(len(data)):
        valid_len = valid_response_lengths[i]
        valid_response_ids = response_ids[i][:valid_len]
        response_str = decode_preserving_reasoning(tokenizer, valid_response_ids, markers)
        responses_str.append(response_str)
    return responses_str


def get_checkpoint_step(checkpoint_path: str) -> int | None:
    """Extract the global step number from a checkpoint path."""
    match = re.search(r"global_step_(\d+)", checkpoint_path)
    return int(match.group(1)) if match else None


def find_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """
    Find all checkpoints in a directory and return them sorted by step.

    Returns list of (step, path) tuples sorted by step ascending.
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("global_step_"):
            step = get_checkpoint_step(name)
            if step is not None:
                checkpoints.append((step, os.path.join(checkpoint_dir, name)))

    return sorted(checkpoints, key=lambda x: x[0])


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[int, str] | None:
    """Find the latest checkpoint in a directory."""
    checkpoints = find_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


def cleanup_old_checkpoint(checkpoint_path: str, keep_lora_adapter: bool = True):
    """
    Clean up a checkpoint directory, removing .pt files but optionally keeping lora_adapter.

    This removes:
    - model_world_size_*_rank_*.pt (FSDP sharded model state)
    - optim_world_size_*_rank_*.pt (FSDP sharded optimizer state)
    - extra_state_world_size_*_rank_*.pt (scheduler + RNG state)
    - data.pt or data_*.pt (dataloader state)

    This keeps (if keep_lora_adapter=True):
    - actor/lora_adapter/ (HuggingFace PEFT format for inference)
    - actor/huggingface/ (tokenizer and config)
    - actor/fsdp_config.json
    """
    if not os.path.exists(checkpoint_path):
        return

    actor_path = os.path.join(checkpoint_path, "actor")

    # Remove .pt files in actor directory
    if os.path.exists(actor_path):
        for filename in os.listdir(actor_path):
            if filename.endswith(".pt"):
                file_path = os.path.join(actor_path, filename)
                os.remove(file_path)
                print(f"Removed: {file_path}")

    # Remove data.pt files in checkpoint root
    for filename in os.listdir(checkpoint_path):
        if filename.endswith(".pt"):
            file_path = os.path.join(checkpoint_path, filename)
            os.remove(file_path)
            print(f"Removed: {file_path}")

    if not keep_lora_adapter:
        # Remove the entire checkpoint directory
        shutil.rmtree(checkpoint_path)
        print(f"Removed entire checkpoint: {checkpoint_path}")

'''
MODEL CAPABILITY PROBES

Both the LoRA target set and verl's fused logit/CE kernels have to be chosen from the model's
HF config rather than hardcoded, because the repo now trains both text-only decoders (Qwen3)
and multimodal decoders (Gemma 4 E2B/E4B, whose checkpoints carry vision and audio towers).
'''

# Attention + MLP projections shared by every decoder this repo trains against.
LORA_ATTENTION_MLP_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# PEFT matches `exclude_modules` strings as a regex over the full module name. Non-text towers
# share projection names with the decoder, so a bare suffix target set would otherwise adapt
# them too - wasted parameters that vLLM then refuses to load into the rollout engine anyway.
LORA_NON_TEXT_TOWER_PATTERN = r".*(vision_tower|audio_tower|embed_vision|embed_audio).*"


def has_non_text_towers(model_config) -> bool:
    """True when the checkpoint ships vision/audio towers alongside the language model."""
    return any(
        getattr(model_config, name, None) is not None for name in ("vision_config", "audio_config")
    )


def lora_target_spec(model_config) -> tuple[object, str | None]:
    """Pick (target_modules, exclude_modules) for this architecture.

    Text-only decoders keep verl's default "all-linear". Multimodal decoders get the explicit
    attention/MLP projection list plus an exclusion regex, so only the language model is adapted.
    """
    if not has_non_text_towers(model_config):
        return "all-linear", None
    return list(LORA_ATTENTION_MLP_MODULES), LORA_NON_TEXT_TOWER_PATTERN


def supports_fused_kernels(model_config) -> bool:
    """Whether the fused lm_head + log-prob path reproduces this model's head exactly.

    verl's own kernels take the softmax straight off `hidden @ lm_head.weight.T`, which skips the
    `final_logit_softcapping` Gemma 4 applies in its forward. `src.train.verl.workers.fused_logits`
    supplies a forward that keeps that term, and every worker installs it, so a softcap is no
    longer a reason to give up the fused path - which is worth having, since the unfused one
    materialises a [batch, tokens, 262144] logits tensor.

    Nothing this repo trains needs anything further, so this is True throughout; it stays a probe
    of the config rather than a constant because that is what the next architecture will need. Note
    that `attn_logit_softcapping` is not a consideration: it acts inside attention, which the fused
    head does not touch.
    """
    return True


# FlashAttention 2 refuses any head dimension above this. Gemma 4 exceeds it on its
# full-attention layers (global_head_dim=512, against head_dim=256 on the sliding ones), which is
# also why vLLM selects FlashAttention 4 for that model rather than 2.
FLASH_ATTENTION_2_MAX_HEAD_DIM = 256

_HEAD_DIM_ATTRS = ("head_dim", "global_head_dim", "attention_head_dim")


def max_decoder_head_dim(model_config) -> int | None:
    """Largest attention head dimension used by the language model, or None if unstated.

    Only the top-level and text configs are considered: the vision/audio towers are never run by
    these text-only environments, so their head dimensions must not constrain the decoder.
    """
    configs = [model_config, getattr(model_config, "text_config", None)]
    dims = [
        value
        for config in configs
        if config is not None
        for name in _HEAD_DIM_ATTRS
        if isinstance(value := getattr(config, name, None), int)
    ]
    return max(dims) if dims else None


def supports_flash_attention_2(model_config) -> bool:
    """Whether the decoder's head dimensions fit FlashAttention 2's kernels."""
    head_dim = max_decoder_head_dim(model_config)
    return head_dim is None or head_dim <= FLASH_ATTENTION_2_MAX_HEAD_DIM


# Substrings marking a `_no_split_modules` entry as belonging to a non-text tower.
_NON_TEXT_LAYER_MARKERS = ("Vision", "Audio", "Image", "Video")


def fsdp_transformer_layer_cls_to_wrap(model_config) -> list[str] | None:
    """Transformer layer classes FSDP should wrap, or None to keep verl's default.

    verl defaults to the model's whole `_no_split_modules`, which for a multimodal decoder includes
    the vision and audio layer classes. A text-only batch never runs those towers, so FSDP2 ends up
    with param groups that had no forward pass and their post-backward hook fails on a missing
    `_unsharded_param`. Wrapping only the language model's layers leaves the tower parameters in the
    root FSDP unit, which does participate in the forward.
    """
    if not has_non_text_towers(model_config):
        return None

    architectures = getattr(model_config, "architectures", None) or []
    no_split_modules = None
    for architecture in architectures:
        import transformers

        model_cls = getattr(transformers, architecture, None)
        no_split_modules = getattr(model_cls, "_no_split_modules", None)
        if no_split_modules:
            break

    if not no_split_modules:
        return None

    text_layers = [
        name
        for name in no_split_modules
        if not any(marker in name for marker in _NON_TEXT_LAYER_MARKERS)
    ]
    return text_layers or None
