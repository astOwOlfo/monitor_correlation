from __future__ import annotations

import re
import os
import shutil
from verl import DataProto
from transformers import AutoTokenizer


def convert_responses_to_str(data: DataProto, tokenizer: AutoTokenizer):
    prompt_ids = data.batch["prompts"]
    response_ids = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]

    prompt_len = prompt_ids.shape[-1]
    valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    responses_str = []
    for i in range(len(data)):
        valid_len = valid_response_lengths[i]
        valid_response_ids = response_ids[i][:valid_len]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
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