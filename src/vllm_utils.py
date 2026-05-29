"""vLLM decoder layer patching and steering utilities.

Provides monkey-patches for vLLM decoder layer classes to support:
  - Steering: additive direction vector at a target layer
  - CAFT: projection off a matrix at a target layer
  - Capture: per-request activation accumulation during decode

All patch functions modify the decoder layer CLASS before LLM creation so ops
are traced into CUDA graphs. Scalar tensor flags gate operations at runtime.

See docs/LAYER_INDEXING.md for layer indexing conventions.
"""
import os
import torch
import torch.nn as nn
import logging
import importlib
from functools import partial
from transformers import AutoConfig


logger = logging.getLogger(__name__)


# Architecture -> (vllm module path, decoder layer class name)
_DECODER_LAYER_REGISTRY: dict[str, tuple[str, str]] = {
    "Qwen3ForCausalLM": ("vllm.model_executor.models.qwen3", "Qwen3DecoderLayer"),
    "Qwen2ForCausalLM": ("vllm.model_executor.models.qwen2", "Qwen2DecoderLayer"),
    "LlamaForCausalLM": ("vllm.model_executor.models.llama", "LlamaDecoderLayer"),
    "MistralForCausalLM": ("vllm.model_executor.models.llama", "LlamaDecoderLayer"),
}


def get_decoder_layer_class(model_path: str) -> type:
    """Look up the vLLM decoder layer class for a HuggingFace model."""
    arch = AutoConfig.from_pretrained(model_path, trust_remote_code=True).architectures[0]
    assert arch in _DECODER_LAYER_REGISTRY, (
        f"Unsupported architecture: {arch}. "
        f"Supported: {list(_DECODER_LAYER_REGISTRY.keys())}. "
        f"Add entry to _DECODER_LAYER_REGISTRY in src/vllm_utils.py."
    )
    mod_path, cls_name = _DECODER_LAYER_REGISTRY[arch]
    return getattr(importlib.import_module(mod_path), cls_name)


# ---------------------------------------------------------------------------
# Decoder layer patches (applied before LLM creation)
# ---------------------------------------------------------------------------

def patch_decoder_for_steering(cls: type, hidden_size: int) -> None:
    """Patch decoder layer class to add activation steering.

    Adds _steer_vec and _steer_alpha buffers to every layer instance.
    Forward adds steer_alpha * steer_vec to hidden states (no-op when alpha=0).

    Idempotent: silently returns if already patched (needed because the vLLM
    plugin and main process may both call this in single-process V1 mode).
    """
    if getattr(cls, '_interp_patched', False):
        return
    orig_init = cls.__init__
    orig_forward = cls.forward

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.register_buffer('_steer_vec', torch.zeros(hidden_size), persistent=False)
        self.register_buffer('_steer_alpha', torch.zeros(1), persistent=False)

    def patched_forward(self, positions, hidden_states, residual):
        hidden_states, residual = orig_forward(self, positions, hidden_states, residual)
        hidden_states = hidden_states + self._steer_alpha * self._steer_vec
        return hidden_states, residual

    cls.__init__ = patched_init
    cls.forward = patched_forward
    cls._interp_patched = True
    logger.info(f"Patched {cls.__name__} for steering (hidden={hidden_size})")


def patch_decoder_for_caft(cls: type, hidden_size: int) -> None:
    """Patch decoder layer class to add CAFT.

    Adds _caft_matrix buffer to every layer instance.
    Forward projects hidden states off of the CAFT matrix.
    """
    assert not getattr(cls, '_interp_patched', False), f"{cls.__name__} already patched"
    orig_init = cls.__init__
    orig_forward = cls.forward

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.register_buffer('_caft_matrix', torch.zeros(hidden_size, 1), persistent=False)

    def patched_forward(self, positions, hidden_states, residual):
        hidden_states, residual = orig_forward(self, positions, hidden_states, residual)
        hidden_states = hidden_states - (hidden_states @ self._caft_matrix) @ self._caft_matrix.T
        return hidden_states, residual

    cls.__init__ = patched_init
    cls.forward = patched_forward
    cls._interp_patched = True
    logger.info(f"Patched {cls.__name__} for CAFT (hidden={hidden_size})")


def patch_decoder_for_capture(
    cls: type,
    hidden_size: int,
    max_batch_size: int = 256,
    capture_layers: set[int] | None = None,
) -> None:
    """Patch decoder layer class to add activation capture.

    Capture layers store per-step hidden states for Python-side per-request routing
    via install_capture_hooks. Uses branchless tensor math for dynamo + CUDA graph
    compatibility.

    NOTE: capture_layers uses 0-based decoder layer indices (NOT residual-stream indices).
    Decoder layer i captures the equivalent of hidden_states[i+1] in HF convention.
    """
    assert not getattr(cls, '_interp_patched', False), f"{cls.__name__} already patched"
    capture_layers = capture_layers or set()
    orig_init = cls.__init__
    orig_forward = cls.forward
    instance_counter = [0]

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._layer_idx = instance_counter[0]
        instance_counter[0] += 1
        is_capture = self._layer_idx in capture_layers
        self.register_buffer('_capture_on', torch.zeros(1), persistent=False)
        self.register_buffer('_decode_phase', torch.zeros(1), persistent=False)
        if is_capture:
            bs = max_batch_size
            self.register_buffer('_step_hidden', torch.zeros(bs, hidden_size, dtype=torch.float32), persistent=False)
            self.register_buffer('_step_n', torch.zeros(1, dtype=torch.long), persistent=False)

    def patched_forward(self, positions, hidden_states, residual):
        hidden_states, residual = orig_forward(self, positions, hidden_states, residual)
        if hasattr(self, '_step_hidden'):
            full_hidden = (hidden_states + residual).float()
            n = min(full_hidden.shape[0], self._step_hidden.shape[0])
            self._step_hidden[:n] = (
                self._step_hidden[:n] * (1 - self._capture_on) + full_hidden[:n] * self._capture_on
            )
            self._step_n.fill_(n)
        return hidden_states, residual

    cls.__init__ = patched_init
    cls.forward = patched_forward
    cls._interp_patched = True
    logger.info(f"Patched {cls.__name__} for capture (capture_layers={capture_layers}, "
                f"max_batch={max_batch_size}, hidden={hidden_size})")


def install_capture_hooks(
    model_runner,
    layers: list[nn.Module],
    capture_state: dict,
) -> None:
    """Install hooks for per-request activation capture (chatspace-inspired).

    Two hooks are installed:
    1. _prepare_inputs hook: sets _decode_phase before each forward pass (runs after
       _update_states/condense, so input_batch.req_ids is authoritative).
    2. execute_model hook: after each forward pass, reads per-token hidden states from
       capture layers and routes to per-request accumulators using input_batch.req_ids.

    All routing is done in Python with the authoritative batch metadata from
    input_batch.req_ids, avoiding cross-contamination from batch compaction.
    """
    original_prepare_inputs = model_runner._prepare_inputs
    original_execute_model = model_runner.execute_model

    capture_state["req_id_to_slot"] = {}
    capture_state.setdefault("req_prefill_seen", {})
    capture_state.setdefault("slot_trace_steps", [])
    capture_state.setdefault("slot_trace_summary", {})

    # Per-request accumulators stored in capture_state (shared with caller)
    # capture_state["act_sum"]   = {slot: {layer_idx: tensor}}
    # capture_state["act_count"] = {slot: {layer_idx: int}}

    def patched_prepare_inputs(scheduler_output, *args, **kwargs):
        req_id_to_slot = capture_state["req_id_to_slot"]
        req_prefill_seen = capture_state["req_prefill_seen"]
        has_new_reqs = bool(scheduler_output.scheduled_new_reqs)
        has_cached_reqs = bool(
            scheduler_output.scheduled_cached_reqs
            and scheduler_output.scheduled_cached_reqs.req_ids
        )

        if has_new_reqs:
            for req in scheduler_output.scheduled_new_reqs:
                slot = len(req_id_to_slot)
                req_id_to_slot[req.req_id] = slot
                req_prefill_seen[req.req_id] = int(req.num_computed_tokens)

        decode_val = 1.0 if has_cached_reqs and not has_new_reqs else 0.0
        for layer in layers:
            layer._decode_phase.fill_(decode_val)

        return original_prepare_inputs(scheduler_output, *args, **kwargs)

    def patched_execute_model(scheduler_output, *args, **kwargs):
        result = original_execute_model(scheduler_output, *args, **kwargs)

        capture_mode = capture_state.get("capture_mode", "decode_only")
        assert capture_mode in {"decode_only", "prefill_span", "decode_plus_prefill_span"}, capture_mode
        slot_trace_debug = bool(capture_state.get("slot_trace_debug", False))
        slot_trace_validate = bool(capture_state.get("slot_trace_validate", False))
        slot_trace_active = slot_trace_debug or slot_trace_validate
        has_new_reqs = bool(scheduler_output.scheduled_new_reqs)
        has_cached_reqs = bool(
            scheduler_output.scheduled_cached_reqs
            and scheduler_output.scheduled_cached_reqs.req_ids
        )
        if capture_mode == "decode_only":
            if not has_cached_reqs or has_new_reqs:
                return result
        elif not (has_new_reqs or has_cached_reqs):
            return result

        req_id_to_slot = capture_state["req_id_to_slot"]
        req_prefill_seen = capture_state["req_prefill_seen"]
        num_reqs = model_runner.input_batch.num_reqs
        req_ids = model_runner.input_batch.req_ids[:num_reqs]
        num_scheduled = scheduler_output.num_scheduled_tokens
        total_scheduled = int(scheduler_output.total_num_scheduled_tokens)
        act_sum = capture_state["act_sum"]
        act_count = capture_state["act_count"]
        new_req_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        cached_req_ids = set(scheduler_output.scheduled_cached_reqs.req_ids) if has_cached_reqs else set()
        prefill_span_by_slot = capture_state.get("prefill_span_by_slot", {})
        trace_layer_idx = capture_state["layer_indices"][0] if slot_trace_active else None

        def _accumulate(slot: int, layer_idx: int, token_acts: torch.Tensor) -> None:
            if token_acts.numel() == 0:
                return
            if slot not in act_sum:
                act_sum[slot] = {}
                act_count[slot] = {}
            if layer_idx not in act_sum[slot]:
                act_sum[slot][layer_idx] = token_acts.sum(dim=0)
                act_count[slot][layer_idx] = int(token_acts.shape[0])
            else:
                act_sum[slot][layer_idx] += token_acts.sum(dim=0)
                act_count[slot][layer_idx] += int(token_acts.shape[0])

        for layer_idx, layer in zip(capture_state["layer_indices"], layers):
            hidden = layer._step_hidden
            n_tokens = int(layer._step_n.item())
            if n_tokens == 0:
                continue
            if capture_mode != "decode_only":
                assert n_tokens >= total_scheduled, (
                    f"Capture buffer too small for prefill capture: have {n_tokens}, "
                    f"need {total_scheduled}. "
                    "Increase activation_cache.token_buffer_capacity."
                )

            trace_this_layer = slot_trace_active and layer_idx == trace_layer_idx
            trace_routes = [] if trace_this_layer and slot_trace_debug else None
            step_decode_captured = 0
            step_prefill_captured = 0
            pos = 0
            for req_id in req_ids:
                n_tok = num_scheduled.get(req_id, 0)
                if n_tok == 0:
                    continue
                slot = req_id_to_slot.get(req_id)
                if slot is None:
                    pos += n_tok
                    continue
                end = min(pos + n_tok, n_tokens)
                token_acts = hidden[pos:end]
                decode_captured = 0
                prefill_captured = 0
                seen_before = int(req_prefill_seen.get(req_id, 0))
                seen_after = seen_before

                if req_id in cached_req_ids and capture_mode in {"decode_only", "decode_plus_prefill_span"}:
                    _accumulate(slot, layer_idx, token_acts)
                    decode_captured = int(token_acts.shape[0])
                    step_decode_captured += decode_captured

                if capture_mode in {"prefill_span", "decode_plus_prefill_span"}:
                    span = prefill_span_by_slot.get(slot)
                    seen = seen_before
                    if span is not None:
                        span_start, span_end = int(span[0]), int(span[1])
                        # Handle chunked prefill across scheduler steps: after the
                        # first step the same request appears as "cached", but its
                        # remaining prompt tokens are still prefill tokens.
                        if seen < span_end:
                            overlap_start = max(seen, span_start)
                            overlap_end = min(seen + n_tok, span_end)
                            if overlap_end > overlap_start:
                                rel_start = overlap_start - seen
                                rel_end = overlap_end - seen
                                _accumulate(slot, layer_idx, token_acts[rel_start:rel_end])
                                prefill_captured = int(rel_end - rel_start)
                                step_prefill_captured += prefill_captured
                    if req_id in new_req_ids or req_id in cached_req_ids:
                        req_prefill_seen[req_id] = seen + n_tok
                        seen_after = int(req_prefill_seen[req_id])

                if trace_this_layer and slot_trace_debug:
                    trace_routes.append(
                        {
                            "req_id": str(req_id),
                            "slot": int(slot),
                            "n_tok_sched": int(n_tok),
                            "pos_start": int(pos),
                            "pos_end": int(end),
                            "is_new_req": bool(req_id in new_req_ids),
                            "is_cached_req": bool(req_id in cached_req_ids),
                            "decode_captured": int(decode_captured),
                            "prefill_captured": int(prefill_captured),
                            "prefill_seen_before": int(seen_before),
                            "prefill_seen_after": int(seen_after),
                        }
                    )

                pos += n_tok

            if trace_this_layer:
                if slot_trace_validate:
                    assert len(set(req_ids)) == len(req_ids), f"Duplicate req_ids in input_batch: {req_ids}"
                    assert sum(int(num_scheduled.get(req_id, 0)) for req_id in req_ids) == total_scheduled, (
                        f"Scheduled token sum mismatch: req_ids={req_ids}, num_scheduled={num_scheduled}, "
                        f"total={total_scheduled}"
                    )
                    assert pos == total_scheduled, f"Routing pos mismatch: pos={pos}, total={total_scheduled}"
                    assert n_tokens <= hidden.shape[0], (n_tokens, hidden.shape[0])
                    if capture_mode == "decode_only":
                        assert step_prefill_captured == 0, step_prefill_captured
                        assert step_decode_captured == total_scheduled, (
                            f"Decode routing dropped tokens: captured={step_decode_captured}, total={total_scheduled}, "
                            f"req_ids={req_ids}, num_scheduled={num_scheduled}"
                        )
                    if capture_mode == "prefill_span":
                        assert step_decode_captured == 0, step_decode_captured
                    expected_prefill_total = capture_state.get("slot_trace_expected_prefill_total")
                    if expected_prefill_total is not None and capture_mode in {"prefill_span", "decode_plus_prefill_span"}:
                        assert step_prefill_captured == int(expected_prefill_total), (
                            f"Prefill span routing mismatch: captured={step_prefill_captured}, "
                            f"expected={int(expected_prefill_total)}"
                        )

                if slot_trace_debug:
                    max_steps = int(capture_state.get("slot_trace_max_steps", 10000))
                    steps = capture_state["slot_trace_steps"]
                    if len(steps) < max_steps:
                        steps.append(
                            {
                                "capture_mode": capture_mode,
                                "layer_idx": int(layer_idx),
                                "has_new_reqs": bool(has_new_reqs),
                                "has_cached_reqs": bool(has_cached_reqs),
                                "num_reqs": int(num_reqs),
                                "req_ids": [str(x) for x in req_ids],
                                "num_scheduled_tokens": {str(k): int(v) for k, v in num_scheduled.items()},
                                "total_num_scheduled_tokens": int(total_scheduled),
                                "step_n_tokens": int(n_tokens),
                                "step_decode_captured": int(step_decode_captured),
                                "step_prefill_captured": int(step_prefill_captured),
                                "routes": trace_routes,
                            }
                        )
                    summary = capture_state["slot_trace_summary"]
                    summary["num_steps"] = int(summary.get("num_steps", 0) + 1)
                    summary["num_steps_with_new_reqs"] = int(summary.get("num_steps_with_new_reqs", 0) + int(has_new_reqs))
                    summary["num_steps_with_cached_reqs"] = int(summary.get("num_steps_with_cached_reqs", 0) + int(has_cached_reqs))
                    summary["num_mixed_steps"] = int(summary.get("num_mixed_steps", 0) + int(has_new_reqs and has_cached_reqs))
                    summary["total_scheduled_tokens"] = int(summary.get("total_scheduled_tokens", 0) + total_scheduled)
                    summary["total_decode_captured"] = int(summary.get("total_decode_captured", 0) + step_decode_captured)
                    summary["total_prefill_captured"] = int(summary.get("total_prefill_captured", 0) + step_prefill_captured)
                    summary["max_num_reqs"] = int(max(int(summary.get("max_num_reqs", 0)), int(num_reqs)))
                    summary["max_total_num_scheduled_tokens"] = int(
                        max(int(summary.get("max_total_num_scheduled_tokens", 0)), int(total_scheduled))
                    )

        return result

    model_runner._prepare_inputs = patched_prepare_inputs
    model_runner.execute_model = patched_execute_model


# ---------------------------------------------------------------------------
# Post-creation activation helpers (copy to layer buffers)
# ---------------------------------------------------------------------------

def setup_steering(layers: list[nn.Module], layer: int, vector: torch.Tensor, alpha: float) -> None:
    """Copy steering vector and alpha to a patched decoder layer's buffers.

    layer uses residual-stream indexing (>= 1). See docs/LAYER_INDEXING.md.
    """
    assert layer >= 1, f"steering layer must be >= 1. Got: {layer}"
    target = layers[layer - 1]
    target._steer_vec.copy_(vector.to(dtype=target._steer_vec.dtype, device=target._steer_vec.device))
    target._steer_alpha.fill_(alpha)


def setup_caft(layers: list[nn.Module], layer: int, dir_matrix: torch.Tensor, orthogonalize: bool = True) -> torch.Tensor:
    """Copy CAFT matrix to a patched decoder layer's buffer. Returns the processed matrix.

    layer uses residual-stream indexing (>= 1). See docs/LAYER_INDEXING.md.
    """
    assert layer >= 1, f"caft layer must be >= 1. Got: {layer}"
    if dir_matrix.ndim == 1:
        dir_matrix = dir_matrix.unsqueeze(1)
    if orthogonalize:
        dir_matrix = torch.linalg.qr(dir_matrix.float()).Q
    target = layers[layer - 1]
    dir_matrix = dir_matrix.to(dtype=target._caft_matrix.dtype, device=target._caft_matrix.device)
    target.register_buffer('_caft_matrix', dir_matrix, persistent=False)
    return dir_matrix


def setup_capture(
    layers: list[nn.Module],
    decoder_capture_layers: list[int],
    model_runner,
) -> dict:
    """Move capture buffers to GPU, create state dict, and install hooks. Returns capture_state."""
    capture_modules = [layers[i] for i in decoder_capture_layers]
    for mod in capture_modules:
        device = next(mod.parameters()).device
        mod._capture_on = mod._capture_on.to(device)
        mod._step_hidden = mod._step_hidden.to(device)
        mod._step_n = mod._step_n.to(device)

    capture_state = {
        "act_sum": {},
        "act_count": {},
        "layer_indices": list(decoder_capture_layers),
    }
    install_capture_hooks(model_runner, capture_modules, capture_state)
    return capture_state


# ---------------------------------------------------------------------------
# vLLM plugin + RPC helpers for cross-process steering
# ---------------------------------------------------------------------------

def _apply_steering_to_model(model, decoder_layer_idx: int, vector: torch.Tensor, alpha: float):
    """Apply steering vector to a decoder layer. Called inside worker subprocess via collective_rpc."""
    target = model.model.layers[decoder_layer_idx]
    target._steer_vec.copy_(vector.to(dtype=target._steer_vec.dtype, device=target._steer_vec.device))
    target._steer_alpha.fill_(alpha)


def set_steering_rpc(llm_engine, layer: int, vector: torch.Tensor, alpha: float):
    """Set steering vector on a layer, using collective_rpc (V1) or direct access (V0).

    layer uses residual-stream indexing: layer i = output of decoder layer i-1.
    See docs/LAYER_INDEXING.md for details.
    """
    assert layer >= 1, f"Cannot steer at layer 0 (embedding output). Got layer={layer}"
    decoder_idx = layer - 1
    if hasattr(llm_engine, 'collective_rpc'):
        fn = partial(_apply_steering_to_model, decoder_layer_idx=decoder_idx, vector=vector.cpu(), alpha=alpha)
        llm_engine.collective_rpc("apply_model", args=(fn,))
    else:
        layers = list(llm_engine.model_executor.driver_worker.worker.model_runner.model.model.layers)
        target = layers[decoder_idx]
        target._steer_vec.copy_(vector.to(dtype=target._steer_vec.dtype, device=target._steer_vec.device))
        target._steer_alpha.fill_(alpha)


def steering_plugin():
    """vLLM general plugin: patches decoder class for steering in worker subprocesses.

    Activated by setting VLLM_STEERING_MODEL and VLLM_STEERING_HIDDEN_SIZE env vars
    before creating the vLLM LLM instance.
    """
    model_path = os.environ.get("VLLM_STEERING_MODEL")
    if not model_path:
        return
    hidden_size = int(os.environ["VLLM_STEERING_HIDDEN_SIZE"])
    decoder_cls = get_decoder_layer_class(model_path)
    patch_decoder_for_steering(decoder_cls, hidden_size)
