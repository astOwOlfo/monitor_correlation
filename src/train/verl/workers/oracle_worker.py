"""Activation Oracle worker: a Ray actor that scores model outputs using an oracle LLM.

Ported from https://github.com/adamkarvonen/activation_oracles (MIT License).

The oracle is an LLM fine-tuned with LoRA to answer yes/no questions about another
model's internal activations. Activations from the target model are injected into
the oracle's residual stream via an additive norm-matching steering hook.

Pipeline per batch:
  1. set_adapter("training") -> forward pass -> collect per-position activations
  2. set_adapter("oracle") -> construct oracle prompts with introspection prefix
  3. generate with steering hook injecting target activations
  4. parse yes/no answers -> return float scores
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto

from src.train.verl.utils import convert_responses_to_str
from src.utils import get_logger

logger = get_logger()

SPECIAL_TOKEN = " ?"

DEFAULT_GENERATION_KWARGS = {
    "do_sample": False,
    "max_new_tokens": 20,
}


def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """Build the introspection prefix prepended to oracle prompts.

    Format: "Layer: {layer}\\n ? ? ? ...\\n"
    Each " ?" occupies one token position where target activations will be steered.
    """
    return f"Layer: {layer}\n" + SPECIAL_TOKEN * num_positions + " \n"


def find_special_token_positions(
    token_ids: list[int], special_token_id: int, num_positions: int
) -> list[int]:
    """Find the first `num_positions` consecutive occurrences of `special_token_id`."""
    positions = []
    for i, tid in enumerate(token_ids):
        if tid == special_token_id:
            positions.append(i)
        if len(positions) == num_positions:
            break
    assert len(positions) == num_positions, (
        f"Expected {num_positions} special tokens, found {len(positions)}"
    )
    assert positions[-1] - positions[0] == num_positions - 1, "Special tokens must be consecutive"
    return positions


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Create a forward hook that additively steers activations with norm matching.

    For each batch element b and position k:
      h'[b,k] = h[b,k] + ||h[b,k]|| * normalize(v[b,k]) * coeff

    Args:
        vectors: List of (K_b, d_model) tensors per batch element.
        positions: List of position index lists per batch element.
        steering_coefficient: Scaling factor for steered vectors.
        device: Target device.
        dtype: Target dtype.
    """
    normed_list = [F.normalize(v.to(device=device, dtype=dtype), dim=-1) for v in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid = output
            rest = None

        B, L, _ = resid.shape
        if L <= 1:
            return (resid, *rest) if rest is not None else resid

        for b in range(min(B, len(positions))):
            pos = torch.tensor(positions[b], dtype=torch.long, device=device)
            orig = resid[b, pos, :]
            norms = orig.norm(dim=-1, keepdim=True)
            steered = (normed_list[b][:len(pos)] * norms * steering_coefficient).to(dtype)
            resid[b, pos, :] = steered.detach() + orig

        return (resid, *rest) if rest is not None else resid

    return hook_fn


@contextlib.contextmanager
def add_hook(module: nn.Module, hook_fn):
    """Context manager that registers a forward hook and removes it on exit."""
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def collect_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    submodule: nn.Module,
) -> torch.Tensor:
    """Run a forward pass and capture activations at `submodule` via hook.

    Returns activations of shape (B, L, D).
    """
    captured = {}

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            captured["acts"] = out[0].detach()
        else:
            captured["acts"] = out.detach()

    handle = submodule.register_forward_hook(hook)
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()
    return captured["acts"]


def parse_answer(text: str) -> float:
    """Parse oracle output to a float score. 1.0 for 'yes', 0.0 otherwise."""
    cleaned = text.strip().rstrip(".!?,;:").strip().lower()
    return 1.0 if cleaned == "yes" else 0.0


def _get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Find the decoder layer ModuleList in a (possibly PEFT-wrapped) model."""
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn") and hasattr(first, "mlp"):
                return list(module)
    raise RuntimeError("Could not find decoder layers in model")


@ray.remote(num_cpus=1, num_gpus=1)
class OracleWorker:
    """Ray actor that scores model outputs using an activation oracle.

    Loads one base model with two PEFT adapters:
    - "oracle": fixed oracle adapter (loaded from HF, never updated)
    - "training": synced from the RL training model's LoRA params
    """

    def __init__(
        self,
        model_id: str,
        oracle_adapter_path: str,
        questions: list[str],
        act_layer: int,
        injection_layer: int = 1,
        steering_coefficient: float = 1.0,
        batch_size: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        lora_config: dict | None = None,
        generation_kwargs: dict | None = None,
    ):
        """Initialize the oracle worker.

        Args:
            model_id: HuggingFace model ID (same base as training model).
            oracle_adapter_path: Path/HF ID for the oracle LoRA adapter.
            questions: List of yes/no questions to ask the oracle.
            act_layer: Layer index (1-indexed HF convention) to collect target activations from.
            injection_layer: Decoder layer index (0-indexed) where steering hook is applied.
            steering_coefficient: Scaling factor for the steering hook.
            batch_size: Batch size for oracle inference.
            dtype: Model dtype.
            lora_config: LoRA config dict for the training adapter (None = no training adapter).
            generation_kwargs: Override generation kwargs for model.generate().
        """
        self.model_id = model_id
        self.oracle_adapter_path = oracle_adapter_path
        self.questions = questions
        self.act_layer = act_layer
        self.injection_layer = injection_layer
        self.steering_coefficient = steering_coefficient
        self.batch_size = batch_size
        self.dtype = dtype
        self.generation_kwargs = {**DEFAULT_GENERATION_KWARGS, **(generation_kwargs or {})}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
        assert len(self.special_token_id) == 1, f"SPECIAL_TOKEN must encode to exactly 1 token, got {self.special_token_id}"
        self.special_token_id = self.special_token_id[0]

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            device_map={"": 0} if self.device.type == "cuda" else None,
        )

        from peft import LoraConfig, get_peft_model, PeftModel

        # Load oracle adapter first
        self.model = PeftModel.from_pretrained(
            base_model, oracle_adapter_path, adapter_name="oracle"
        )

        # Add training adapter if lora_config provided
        self._has_training_adapter = lora_config is not None
        if self._has_training_adapter:
            training_lora = LoraConfig(**lora_config)
            self.model.add_adapter("training", training_lora)
            self._peft_config = self.model.peft_config.get("training", None)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Cache decoder layers for hook access
        self._decoder_layers = _get_decoder_layers(self.model)
        # act_layer is 1-indexed HF convention -> decoder layer index is act_layer - 1
        assert act_layer >= 1, "act_layer must be >= 1 (1-indexed HF convention)"
        self._act_decoder_layer = self._decoder_layers[act_layer - 1]
        self._injection_module = self._decoder_layers[injection_layer]

        self._last_lora_hash: str | None = None

        logger.info(
            f"INITIALIZED ORACLE WORKER: model={model_id}, oracle={oracle_adapter_path}, "
            f"questions={questions}, act_layer={act_layer}, injection_layer={injection_layer}"
        )

    def warmup(self) -> str:
        """Run a tiny forward pass to allocate memory."""
        inputs = self.tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            self.model(**inputs)
        return "ok"

    def update_lora(
        self,
        lora_params: Dict[str, Any],
        base_sync_done: bool = True,
    ) -> None:
        """Sync training adapter weights from the RL training model."""
        if not self._has_training_adapter or not lora_params:
            return

        import hashlib
        hasher = hashlib.sha256()
        for k in sorted(lora_params.keys()):
            hasher.update(k.encode("utf-8"))
            hasher.update(lora_params[k].cpu().contiguous().view(-1).numpy().tobytes())
        new_hash = hasher.hexdigest()

        if self._last_lora_hash == new_hash:
            return

        from peft.utils.save_and_load import set_peft_model_state_dict
        from verl.utils.fsdp_utils import replace_lora_wrapper

        state_dict = lora_params
        if not base_sync_done:
            state_dict = {replace_lora_wrapper(k, self._peft_config): v for k, v in lora_params.items()}

        set_peft_model_state_dict(self.model, state_dict, adapter_name="training")
        self._last_lora_hash = new_hash
        logger.info("Oracle worker: training LoRA params updated")

    def _collect_target_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Collect activations from the training adapter at the target layer.

        Returns: (B, L, D) tensor of activations.
        """
        if self._has_training_adapter:
            self.model.set_adapter("training")
        else:
            self.model.disable_adapter_layers()

        acts = collect_activations(
            self.model, input_ids, attention_mask, self._act_decoder_layer
        )

        return acts

    def _build_oracle_inputs(
        self,
        prompts: list[str],
        responses: list[str],
        activations_BLD: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[dict]:
        """Construct oracle inputs with introspection prefixes and steering vectors.

        For each (sample, question) pair, builds a chat-templated prompt with the
        introspection prefix prepended, locates the special token positions, and
        extracts the corresponding steering vectors from the target activations.

        Returns a list of dicts with keys: input_ids, attention_mask, positions, vectors
        """
        oracle_inputs = []

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # Find response token positions in the original sequence
            # Response tokens are where attention_mask=1 after the prompt
            mask_i = attention_mask[i]  # (L,)
            seq_len = mask_i.sum().item()

            # Extract response-region activations
            # We use all non-padding positions' activations
            acts_i = activations_BLD[i, :seq_len, :]  # (seq_len, D)

            for question in self.questions:
                num_positions = seq_len
                prefix = get_introspection_prefix(self.act_layer, num_positions)

                oracle_prompt = [
                    {"role": "user", "content": prefix + question}
                ]
                tokenized = self.tokenizer.apply_chat_template(
                    oracle_prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                    enable_thinking=False,
                )

                positions = find_special_token_positions(
                    tokenized, self.special_token_id, num_positions
                )
                oracle_inputs.append({
                    "input_ids": tokenized,
                    "positions": positions,
                    "vectors": acts_i,  # (num_positions, D)
                })

        return oracle_inputs

    def _generate_oracle_answers(self, oracle_inputs: list[dict]) -> list[str]:
        """Run oracle generation with steering hooks in batches.

        Returns list of generated text strings.
        """
        self.model.set_adapter("oracle")
        all_answers = []

        for batch_start in range(0, len(oracle_inputs), self.batch_size):
            batch = oracle_inputs[batch_start : batch_start + self.batch_size]

            # Pad to same length (left-pad for generation)
            max_len = max(len(inp["input_ids"]) for inp in batch)
            padded_ids = []
            padded_masks = []
            batch_positions = []
            batch_vectors = []

            for inp in batch:
                ids = inp["input_ids"]
                pad_len = max_len - len(ids)
                padded_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
                padded_masks.append([0] * pad_len + [1] * len(ids))
                # Shift positions by pad_len
                batch_positions.append([p + pad_len for p in inp["positions"]])
                batch_vectors.append(inp["vectors"])

            input_ids = torch.tensor(padded_ids, device=self.device)
            attn_mask = torch.tensor(padded_masks, device=self.device)

            hook_fn = get_hf_activation_steering_hook(
                vectors=batch_vectors,
                positions=batch_positions,
                steering_coefficient=self.steering_coefficient,
                device=self.device,
                dtype=self.dtype,
            )

            with add_hook(self._injection_module, hook_fn):
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        **self.generation_kwargs,
                    )

            # Decode only newly generated tokens
            for j in range(len(batch)):
                prompt_len = input_ids.shape[1]
                new_tokens = output_ids[j, prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_answers.append(text)

        return all_answers

    def run_oracle(
        self,
        data: DataProto,
        lora_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full oracle pipeline. Returns {"oracle_scores": tensor(n_samples,)}.

        Steps:
          1. Sync LoRA if params provided
          2. Tokenize prompts+responses, forward pass to get target activations
          3. Build oracle inputs with introspection prefix + steering vectors
          4. Generate oracle answers with steering hook
          5. Parse answers, aggregate per sample, return scores
        """
        if lora_params is not None:
            self.update_lora(lora_params)

        responses_str = convert_responses_to_str(data, self.tokenizer)
        prompts = [x["prompt"] for x in data.non_tensor_batch["extra_info"]]
        n_samples = len(prompts)

        # Tokenize full prompt+response sequences for activation collection
        full_texts = []
        for prompt, response in zip(prompts, responses_str):
            chat = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            full_texts.append(chat + response)

        tokenized = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Collect target activations
        activations = self._collect_target_activations(
            tokenized["input_ids"], tokenized["attention_mask"]
        )

        # Build oracle inputs
        oracle_inputs = self._build_oracle_inputs(
            [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=False, enable_thinking=False) for p in prompts],
            responses_str,
            activations,
            tokenized["attention_mask"],
        )

        # Generate oracle answers
        answers = self._generate_oracle_answers(oracle_inputs)

        # Parse and aggregate: each sample has len(questions) answers
        scores = torch.zeros(n_samples, dtype=torch.float32)
        n_questions = len(self.questions)
        for i in range(n_samples):
            sample_answers = answers[i * n_questions : (i + 1) * n_questions]
            sample_scores = [parse_answer(a) for a in sample_answers]
            scores[i] = sum(sample_scores) / len(sample_scores)

        logger.info(f"Oracle scores: mean={scores.mean():.3f}, min={scores.min():.3f}, max={scores.max():.3f}")
        return {"oracle_scores": scores}
