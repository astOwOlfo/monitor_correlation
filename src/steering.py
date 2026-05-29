# Ported from persona_vectors/activation_steer.py (v0.2)
# Only ActivationSteerer is ported; ActivationSteererMultiple omitted.

import os
import shutil
import torch
from typing import Sequence, Union, Iterable

from transformers import AutoConfig
from src.vllm_utils import get_decoder_layer_class, patch_decoder_for_steering, set_steering_rpc

class ActivationSteerer:
    """Add (coeff * steering_vector) to a chosen transformer block's output.

    Handles blocks that return tuples and fails loudly if it can't locate a layer list.
    Used as a context manager for safe hook management.

    layer_idx uses residual-stream indexing: layer i = output of decoder layer i-1.
    See docs/LAYER_INDEXING.md for details.
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",       # GPT-2/Neo, Bloom, etc.
        "encoder.layer",       # BERT/RoBERTa
        "model.layers",        # Llama/Mistral
        "gpt_neox.layers",     # GPT-NeoX
        "block",               # Flan-T5
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        assert self.vector.ndim == 1, "steering_vector must be 1-D"
        hidden = getattr(model.config, "hidden_size", None)
        assert not (hidden and self.vector.numel() != hidden), \
            f"Vector length {self.vector.numel()} != model hidden_size {hidden}"
        assert self.positions in {"all", "prompt", "response"}, \
            "positions must be 'all', 'prompt', or 'response'"

    def _locate_layer(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if not hasattr(cur, "__getitem__"):
                    continue
                decoder_idx = self.layer_idx - 1 if self.layer_idx >= 0 else self.layer_idx
                assert -len(cur) <= decoder_idx < len(cur), (
                    f"decoder_idx={decoder_idx} (from layer_idx={self.layer_idx}) out of range for {len(cur)} layers"
                )
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{decoder_idx}] (residual-stream layer {self.layer_idx})")
                return cur[decoder_idx]

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _hook_fn(self, module, ins, out):
        steer = self.coeff * self.vector

        def _add(t):
            if self.positions == "all":
                return t + steer.to(t.device)
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                t2 = t.clone()
                t2 += steer.to(t.device)
                return t2
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2

        if torch.is_tensor(out):
            return _add(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            return (_add(out[0]), *out[1:])
        return out

    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


class VLLMSteering:
    """Context manager for vLLM activation steering.

    Handles both phases: patches decoder class before LLM creation (structural),
    then provides steer() to set vectors on specific layers after creation (data).
    See docs/LAYER_INDEXING.md for layer indexing conventions.

    Usage:
        with VLLMSteering(model_name) as steering:
            llm_gen = create_llm_generator("vllm", model_name=model_name, ...)
            steering.set_engine(llm_gen.model.llm_engine)

            steering.steer(layer=18, vector=vec, alpha=3.0)
            responses = llm_gen.batch_generate(prompts, params)
            steering.steer(layer=18, vector=vec, alpha=0.0)
    """

    _ENV_KEYS = ("VLLM_STEERING_MODEL", "VLLM_STEERING_HIDDEN_SIZE", "VLLM_ALLOW_INSECURE_SERIALIZATION")

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._engine = None
        self._orig_env: dict[str, str | None] = {}

    def __enter__(self):
        shutil.rmtree("/tmp/_vllm/torch_compile_cache", ignore_errors=True)
        hidden_size = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True).hidden_size

        self._orig_env = {k: os.environ.get(k) for k in self._ENV_KEYS}
        os.environ["VLLM_STEERING_MODEL"] = self.model_name
        os.environ["VLLM_STEERING_HIDDEN_SIZE"] = str(hidden_size)
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        decoder_cls = get_decoder_layer_class(self.model_name)
        patch_decoder_for_steering(decoder_cls, hidden_size)
        return self

    def set_engine(self, engine):
        """Bind the vLLM engine after LLM creation."""
        self._engine = engine

    def steer(self, layer: int, vector: torch.Tensor, alpha: float):
        """Set steering vector on a layer. layer uses residual-stream indexing (>= 1)."""
        assert self._engine is not None, "Call set_engine() after LLM creation"
        set_steering_rpc(self._engine, layer, vector, alpha)

    def __exit__(self, *exc):
        for k, v in self._orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
