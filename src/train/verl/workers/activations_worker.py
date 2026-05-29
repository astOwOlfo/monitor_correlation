"""Dedicated Ray actor for activation caching on a separate GPU.

This worker holds a HuggingFace model on its own GPU and computes activations
for batches of prompt-response pairs. It can optionally sync LoRA parameters
from the training model, but by default operates as a standalone frozen model.

Use this when you want activations from a model that is separate from the
generation model (e.g., a different architecture or a frozen oversight model).
"""
from __future__ import annotations

import hashlib
import inspect
from collections import OrderedDict
from typing import Any, Dict, Optional

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto

from src.activations import BatchedTransformersActivations
from src.monitor import probe as probe_utils
from src.monitor.probe import Probe, load_probe
from src.utils import get_logger

logger = get_logger()


@ray.remote(num_cpus=1, num_gpus=1)
class ActivationsWorker:
    """Ray actor that holds a HF model on a dedicated GPU and serves activation caching.

    Notes on GPU placement:
    - Ray assigns a distinct physical GPU to this actor via `num_gpus=1`.
    - Ensure the training config reserves one GPU for this worker (n_gpus_per_node reduced by 1)
      so rollout workers do not consume all GPUs.
    """

    def __init__(
        self,
        model_id: str,
        layers: list[int],
        position: str = "response_avg",
        batch_size: int = 16,
        dtype: str | torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        probe_path: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the activations worker.

        Args:
            model_id: HuggingFace model ID or path
            layers: List of layer indices to capture activations from
            position: Position to capture ('response_avg', 'prompt_avg', etc.)
            batch_size: Batch size for activation caching
            dtype: Model dtype
            trust_remote_code: Whether to trust remote code
            tokenizer_kwargs: Additional tokenizer kwargs
            chat_template_kwargs: Chat template kwargs
            use_lora: Whether to use LoRA (default False for standalone model)
            lora_config: LoRA configuration dict (required if use_lora=True)
        """
        self.model_id = model_id
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.trust_remote_code = trust_remote_code
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.layers = layers
        self.position = position
        self.use_lora = use_lora
        self.probe_path = probe_path
        self.probe: Probe | None = load_probe(probe_path) if probe_path is not None else None

        assert len(self.layers) > 0, "Layers must be a non-empty list"
        assert self.position in {"response_avg", "response_all"}, f"Unsupported position={self.position}"

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=self.trust_remote_code, **self.tokenizer_kwargs
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map={"": 0} if self.device.type == "cuda" else None,
        )

        if self.use_lora:
            assert lora_config is not None, "lora_config required when use_lora=True"
            from peft import LoraConfig, get_peft_model
            self.lora_config = LoraConfig(**lora_config)
            model = get_peft_model(model, self.lora_config)
            self.peft_config = model.peft_config.get("default", None)
        else:
            self.peft_config = None

        self.activations_extractor = BatchedTransformersActivations(
            model=model,
            tokenizer=self.tokenizer,
            progress_bar=True,
            batch_size=batch_size
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._last_lora_hash: Optional[str] = None

        self._probe_supports_mask = False
        if self.probe is not None and getattr(self.probe, "requires_sequence", False):
            assert self.position == "response_all", "Sequence probe requires position='response_all'"
            self._probe_supports_mask = "mask" in inspect.signature(self.probe.predict_proba).parameters

        logger.info(f"INITIALIZED ACTIVATIONS WORKER: model={model_id}, layers={self.layers}, "
                    f"position={self.position}, use_lora={self.use_lora}, probe={self.probe_path}")

    @property
    def model(self):
        return self.activations_extractor.model

    def warmup(self) -> str:
        """Run a tiny forward to allocate memory and lazy init kernels."""
        text = "Hello"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            _ = self.model(**inputs)
        return "ok"

    def _hash_state_dict(self, d: dict[str, Any] | OrderedDict) -> str:
        """Create a stable hash for a state dict to detect changes quickly."""
        hasher = hashlib.sha256()
        for k in sorted(d.keys()):
            hasher.update(k.encode("utf-8"))
            t = d[k]
            buf = t.cpu().contiguous().view(-1).numpy().tobytes()
            hasher.update(buf)
        return hasher.hexdigest()

    def update_lora(
        self,
        lora_params: Dict[str, Any],
        base_sync_done: bool = True,
        **kwargs
    ) -> None:
        """Update LoRA parameters from the training model.

        Only has effect if use_lora=True was set during initialization.
        """
        if not self.use_lora or not lora_params:
            return

        new_hash = self._hash_state_dict(lora_params)
        if self._last_lora_hash == new_hash:
            return

        from peft.utils.save_and_load import set_peft_model_state_dict
        from verl.utils.fsdp_utils import replace_lora_wrapper

        state_dict = lora_params
        if not base_sync_done:
            state_dict = {replace_lora_wrapper(k, self.peft_config): v for k, v in lora_params.items()}

        set_peft_model_state_dict(self.model, state_dict)
        self._last_lora_hash = new_hash

    def cache_activations(self, data: DataProto) -> Dict[str, Any]:
        """Cache activations for a batch of prompt-response pairs."""
        logger.info(f"Caching activations at {self.position} with layers {self.layers}")
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        response_mask = data.batch["response_mask"]
        assert input_ids.ndim == 2, f"Expected input_ids (B, L), got {tuple(input_ids.shape)}"
        assert attention_mask.shape == input_ids.shape, (
            f"attention_mask shape mismatch: {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
        )
        assert response_mask.ndim == 2 and response_mask.shape[0] == input_ids.shape[0], (
            f"Expected response_mask (B, S), got {tuple(response_mask.shape)}"
        )

        prompt_token_ids: list[list[int]] = []
        response_token_ids: list[list[int]] = []
        for i in range(input_ids.shape[0]):
            valid_ids = input_ids[i][attention_mask[i].bool()]
            resp_len = int(response_mask[i].sum().item())
            prompt_len = int(valid_ids.numel()) - resp_len
            assert prompt_len >= 0, f"Invalid prompt_len={prompt_len}, resp_len={resp_len}"
            prompt_token_ids.append(valid_ids[:prompt_len].tolist())
            response_token_ids.append(valid_ids[prompt_len:].tolist())

        cache = self.activations_extractor.cache_activations_from_token_ids(
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            layers=self.layers,
            position=[self.position],
        )[self.position]

        if self.probe is not None:
            self.probe.move_to_device(device=self.device, layers=self.layers)
            model_activations = cache
            if model_activations.ndim == 2:
                model_activations = model_activations.unsqueeze(0)
            predict_kwargs = {}
            if self._probe_supports_mask:
                predict_kwargs["mask"] = probe_utils.resolve_sequence_mask(
                    model_activations,
                    layers=self.layers,
                    mask=response_mask,
                )
            cache = self.probe.predict_proba(model_activations, layers=self.layers, **predict_kwargs)
            cache = cache.detach().cpu()
            self.probe.move_to_device(device=torch.device("cpu"), layers=self.layers)
            assert cache.ndim == 2 and cache.shape[0] == input_ids.shape[0], (
                f"Expected probe output (n_samples={input_ids.shape[0]}, ?), got {tuple(cache.shape)}"
            )
        elif len(self.layers) == 1:
            cache = cache.squeeze(0)

        logger.info(f"Activations cached: shape {cache.shape}")
        return {"activations": cache}

    def cache_activations_from_raw(
        self,
        prompts: list,
        responses: list[str],
    ) -> Dict[str, Any]:
        """Cache activations from raw prompt-response pairs (not DataProto)."""
        logger.info(f"Caching activations from raw data: {len(prompts)} samples")
        cache = self.activations_extractor.cache_activations(
            prompts=prompts,
            responses=responses,
            layers=self.layers,
            position=[self.position]
        )[self.position].squeeze(0)
        logger.info(f"Activations cached from raw data: shape {cache.shape}")
        return {"activations": cache}

    def update_lora_and_cache_from_raw(
        self,
        prompts: list,
        responses: list[str],
        lora_params: Optional[Dict[str, Any]] = None,
        base_sync_done: bool = True,
    ) -> Dict[str, Any]:
        """Update LoRA params then cache activations from raw data."""
        if lora_params is not None:
            self.update_lora(lora_params, base_sync_done=base_sync_done)
        return self.cache_activations_from_raw(prompts, responses)

    def update_and_cache(
        self,
        data: DataProto,
        lora_params: Optional[Dict[str, Any]] = None,
        base_sync_done: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Optionally update LoRA params, then cache activations."""
        if lora_params is not None:
            self.update_lora(lora_params, base_sync_done=base_sync_done)
        return self.cache_activations(data=data)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "use_lora": self.use_lora,
            "layers": self.layers,
            "position": self.position,
            "probe_path": self.probe_path,
        }

    def shutdown(self) -> str:
        import gc
        del self.activations_extractor
        torch.cuda.empty_cache()
        gc.collect()
        return "shutdown"
