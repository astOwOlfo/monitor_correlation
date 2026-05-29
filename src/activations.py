from abc import ABC, abstractmethod
from typing import Literal
import gc
import os
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generate import ChatRequest

"""
Activations Caching
"""

CachePosition = Literal["prompt_avg", "prompt_last", "response_avg", "prompt_all", "response_all"]

class ActivationsCache(ABC):
    name: str


    @abstractmethod
    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None):
        pass

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

        try:
            torch.cuda.ipc_collect()
        except:
            pass

        try:
            # Then let PyTorch tear down the process group, if vLLM initialized it
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
        except AssertionError:
            pass
        
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError: 
            pass


class TransformersActivations(ActivationsCache):
    name = "transformers"
    def __init__(self, model_name: str | None = None, model = None, tokenizer = None, progress_bar: bool = True, debug: bool = False):
        '''Provide either model name or model and tokenizer'''

        assert (model is not None) or (model_name is not None)
        
        if model is None:
            self.model_name = model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, # NOTE: This must match was is used in training
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        else:
            self.model_name = model.config._name_or_path
            self.model = model

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        self.progress_bar = progress_bar
        self.debug = debug

    
    def _response_start(self, prompt: ChatRequest, response: str, prompt_len: int) -> int:
        """Token position where non-CoT response content starts.

        If the response contains a </think> block (with a matching <think>
        opening before it), returns the position after </think> so that
        response_avg and response_all exclude thinking tokens. Otherwise
        returns prompt_len (entire response is non-CoT). A stray </think>
        without a matching <think> in the response is treated as a hallucination
        and ignored.
        """
        if "</think>" not in response:
            return prompt_len
        cot_end = response.index("</think>")
        if "<think>" not in response[:cot_end]:
            return prompt_len
        cot_response = response[:cot_end + len("</think>")]
        cot_text = prompt + [{"role": "assistant", "content": cot_response}]
        cot_chat_text = self.tokenizer.apply_chat_template(cot_text, tokenize=False, add_generation_prompt=False)
        return len(self.tokenizer.encode(cot_chat_text, add_special_tokens=False))

    def log(self, msg: str):
        if self.debug:
            print(msg)

    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None, position: list[CachePosition] | None = None, include_think_tokens: bool = True):

        self.log(f"Caching activations for {len(prompts)} prompts and {len(responses)} responses")
        self.log(f"Layers: {layers}")
        self.log(f"Position: {position}")
        self.log(f"Prompt: {prompts[0]}")
        self.log(f"Response: {responses[0]}")

        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer+1))

        if position is None:
            position = ["prompt_avg", "prompt_last", "response_avg"]
        position = set(position)

        cache = {k: [[] for _ in range(max_layer+1)] for k in position}

        if self.progress_bar:
            pbar = tqdm(total=len(prompts), desc="Caching activations")

        with torch.inference_mode():
            for prompt, response in zip(prompts, responses):
                # Convert all to chatml format
                full_text = prompt + [{'role': 'assistant', 'content': response}]
                full_chat_text = self.tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False)
                prompt_chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

                # Tokenize
                inputs = self.tokenizer(full_chat_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                prompt_len = len(self.tokenizer.encode(prompt_chat_text, add_special_tokens=False))
                resp_start = prompt_len if include_think_tokens else self._response_start(prompt, response, prompt_len)

                # NOTE: BATCH SIZE IS 1

                # Cache activations
                outputs = self.model(**inputs, output_hidden_states=True)
                for layer in layers:
                    if "prompt_avg" in position:
                        cache['prompt_avg'][layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu()) # (1, hidden_size)
                    if "response_avg" in position:
                        cache['response_avg'][layer].append(outputs.hidden_states[layer][:, resp_start:, :].mean(dim=1).detach().cpu()) # (1, hidden_size)
                    if "prompt_last" in position:
                        cache['prompt_last'][layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu()) # (1, hidden_size)
                    if "prompt_all" in position:
                        cache['prompt_all'][layer].append(outputs.hidden_states[layer][:, :prompt_len, :].detach().cpu()) # (1, prompt_len, hidden_size)
                    if "response_all" in position:
                        cache['response_all'][layer].append(outputs.hidden_states[layer][:, resp_start:, :].detach().cpu()) # (1, response_len, hidden_size)
                del outputs

                if self.progress_bar:
                    pbar.update(1)
        
        if self.progress_bar:
            pbar.close()

        # Response all and prompt_all need to be padded to the same length
        # prompt_all and response_all will have seq_len as part of the shape
        for k in ["prompt_all", "response_all"]:
            if k in cache:
                x = cache[k]

                # Get max len across all layers
                max_len = max([max([x[l][i].shape[1] for i in range(len(x[l]))]) for l in layers])

                # Pad all sequences to have dim[1] = max_len
                for l in layers:
                    x[l] = [torch.nn.functional.pad(x[l][i].transpose(-1, -2), (0, max_len - x[l][i].shape[1]), value=torch.nan).transpose(-1, -2) for i in range(len(x[l]))]

        for k in position:
            layer = None
            try:
                # Cat all of the layers -> (n_samples, hidden_size) for each layer OR (n_samples, seq_len, hidden_size)
                for layer in layers:
                    cache[k][layer] = torch.cat(cache[k][layer], dim=0) # (n_samples, hidden_size) OR (n_samples, seq_len, hidden_size)
            except Exception as e:
                self.log(f"Error caching activations for layer {layer}: {e}")
                raise e
                
            # Stack all of the layers -> (n_layers, n_samples, hidden_size)
            cache[k] = torch.vstack([cache[k][l].unsqueeze(0) for l in layers]) # (n_layers, n_samples, hidden_size) OR (n_layers, n_sample, seq_len, hidden_size)

        # Returns: dict[str, torch.Tensor]
        # 'prompt_avg': (n_layers, n_samples, hidden_size)
        # 'prompt_last': (n_layers, n_samples, hidden_size)
        # 'response_avg': (n_layers, n_samples, hidden_size)
        # 'prompt_all': (n_layers, n_samples, seq_len, hidden_size) # seq_len is the maximum sequence length across all samples and layers
        # 'response_all': (n_layers, n_samples, seq_len, hidden_size) # seq_len is the maximum sequence length across all samples and layers
        return cache 


class BatchedTransformersActivations(TransformersActivations):
    name = "transformers_batched"

    def __init__(
        self,
        model_name: str | None = None,
        model=None,
        tokenizer=None,
        progress_bar: bool = True,
        debug: bool = False,
        batch_size: int = 8,
    ):
        """Same as TransformersActivations, but processes inputs in batches for speed.

        Args:
            model_name: HF repo ID if model/tokenizer not provided.
            model: Optional preloaded model (e.g., with LoRA applied).
            tokenizer: Optional preloaded tokenizer.
            progress_bar: Show a tqdm over batches.
            debug: Verbose logging.
            batch_size: Number of (prompt,response) pairs per forward pass.
        """
        super().__init__(model_name=model_name, model=model, tokenizer=tokenizer, progress_bar=progress_bar, debug=debug)
        self.batch_size = max(1, int(batch_size))
        try:
            self.model.eval()
        except Exception:
            pass

    def cache_activations(
        self,
        prompts: list[ChatRequest],
        responses: list[str],
        layers: list[int] | None = None,
        position: list[CachePosition] | None = None,
        include_think_tokens: bool = True,
    ):
        """Batched variant of TransformersActivations.cache_activations."""

        self.log(f"Caching activations (batched) for {len(prompts)} items, batch_size={self.batch_size}")
        if len(prompts) == 0:
            return {}

        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer + 1))

        if position is None:
            position = ["prompt_avg", "prompt_last", "response_avg"]
        position = set(position)

        cache: dict[str, list[list[torch.Tensor]]] = {k: [[] for _ in range(max_layer + 1)] for k in position}

        # Pre-render chat templates and prompt lengths so we can batch tokenize
        full_texts: list[str] = []
        prompt_lens: list[int] = []
        resp_starts: list[int] = []
        for prompt, response in zip(prompts, responses):
            full_text = prompt + [{"role": "assistant", "content": response}]
            full_chat_text = self.tokenizer.apply_chat_template(
                full_text, tokenize=False, add_generation_prompt=False
            )
            prompt_chat_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            full_texts.append(full_chat_text)
            prompt_len = len(self.tokenizer.encode(prompt_chat_text, add_special_tokens=False))
            prompt_lens.append(prompt_len)
            resp_starts.append(prompt_len if include_think_tokens else self._response_start(prompt, response, prompt_len))

        total = len(full_texts)
        rng = range(0, total, self.batch_size)
        pbar = tqdm(total=len(rng), desc="Caching activations (batched)") if self.progress_bar else None

        with torch.inference_mode():
            for start in rng:
                end = min(start + self.batch_size, total)
                batch_texts = full_texts[start:end]
                batch_prompt_lens = prompt_lens[start:end]
                batch_resp_starts = resp_starts[start:end]

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states  # tuple(len = n_layers+1) of (B, L, H)

                attn_mask = inputs.get("attention_mask")
                # Pre-compute seq_lens for this batch
                batch_seq_lens = []
                for j in range(end - start):
                    if attn_mask is not None:
                        batch_seq_lens.append(int(attn_mask[j].sum().item()))
                    else:
                        batch_seq_lens.append(int(hs[0].shape[1]))

                for j in range(end - start):
                    prompt_len = int(batch_prompt_lens[j])
                    resp_start = int(batch_resp_starts[j])
                    seq_len = batch_seq_lens[j]

                    for layer in layers:
                        h = hs[layer][j]  # (L, H)
                        if "prompt_avg" in position:
                            cache["prompt_avg"][layer].append(
                                h[:prompt_len, :].mean(dim=0, keepdim=True).detach().cpu()
                            )
                        if "prompt_last" in position:
                            last_idx = max(0, prompt_len - 1)
                            cache["prompt_last"][layer].append(
                                h[last_idx, :].unsqueeze(0).detach().cpu()
                            )
                        if "response_avg" in position:
                            cache["response_avg"][layer].append(
                                h[resp_start:seq_len, :].mean(dim=0, keepdim=True).detach().cpu()
                            )
                        if "prompt_all" in position:
                            cache["prompt_all"][layer].append(
                                h[:prompt_len, :].unsqueeze(0).detach().cpu()
                            )
                        if "response_all" in position:
                            cache["response_all"][layer].append(
                                h[resp_start:seq_len, :].unsqueeze(0).detach().cpu()
                            )

                del outputs
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Pad variable-length sequences
        for k in ["prompt_all", "response_all"]:
            if k in cache and layers:
                max_len = max(max(cache[k][l][i].shape[1] for i in range(len(cache[k][l]))) for l in layers)
                for l in layers:
                    cache[k][l] = [
                        torch.nn.functional.pad(
                            cache[k][l][i].transpose(-1, -2),
                            (0, max_len - cache[k][l][i].shape[1]),
                            value=torch.nan,
                        ).transpose(-1, -2)
                        for i in range(len(cache[k][l]))
                    ]

        for k in position:
            for layer in layers:
                cache[k][layer] = torch.cat(cache[k][layer], dim=0)
            cache[k] = torch.vstack([cache[k][l].unsqueeze(0) for l in layers])

        return cache

    def cache_activations_from_token_ids(
        self,
        *,
        prompt_token_ids: list[list[int]],
        response_token_ids: list[list[int]],
        layers: list[int] | None = None,
        position: list[CachePosition] | None = None,
        drop_last_response_token: bool = False,
    ):
        """Batched activation caching from exact prompt/response token IDs."""
        assert len(prompt_token_ids) == len(response_token_ids), (len(prompt_token_ids), len(response_token_ids))
        if len(prompt_token_ids) == 0:
            return {}

        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer + 1))

        if position is None:
            position = ["prompt_avg", "prompt_last", "response_avg"]
        position = set(position)

        cache: dict[str, list[list[torch.Tensor]]] = {k: [[] for _ in range(max_layer + 1)] for k in position}

        full_ids = [p + r for p, r in zip(prompt_token_ids, response_token_ids, strict=True)]
        prompt_lens = [len(x) for x in prompt_token_ids]
        response_lens = [max(0, len(x) - int(drop_last_response_token)) for x in response_token_ids]

        total = len(full_ids)
        rng = range(0, total, self.batch_size)
        pbar = tqdm(total=len(rng), desc="Caching activations (batched token IDs)") if self.progress_bar else None

        with torch.inference_mode():
            for start in rng:
                end = min(start + self.batch_size, total)
                batch_full_ids = full_ids[start:end]
                batch_prompt_lens = prompt_lens[start:end]
                batch_response_lens = response_lens[start:end]
                max_len = max(len(x) for x in batch_full_ids)

                input_ids = torch.zeros(end - start, max_len, dtype=torch.long)
                attention_mask = torch.zeros(end - start, max_len, dtype=torch.long)
                for j, ids in enumerate(batch_full_ids):
                    seq = torch.tensor(ids, dtype=torch.long)
                    n = int(seq.numel())
                    input_ids[j, :n] = seq
                    attention_mask[j, :n] = 1

                inputs = {
                    "input_ids": input_ids.to(self.model.device),
                    "attention_mask": attention_mask.to(self.model.device),
                }

                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states  # tuple(len=n_layers+1) of (B, L, H)

                for j in range(end - start):
                    prompt_len = int(batch_prompt_lens[j])
                    resp_len = int(batch_response_lens[j])
                    seq_len = int(attention_mask[j].sum().item())
                    resp_start = prompt_len
                    resp_end = min(seq_len, prompt_len + resp_len)
                    assert 0 <= prompt_len <= seq_len, (prompt_len, seq_len)
                    assert resp_start <= resp_end <= seq_len, (resp_start, resp_end, seq_len)

                    for layer in layers:
                        h = hs[layer][j]  # (L, H)
                        if "prompt_avg" in position:
                            cache["prompt_avg"][layer].append(
                                h[:prompt_len, :].mean(dim=0, keepdim=True).detach().cpu()
                            )
                        if "prompt_last" in position:
                            cache["prompt_last"][layer].append(
                                h[max(0, prompt_len - 1), :].unsqueeze(0).detach().cpu()
                            )
                        if "response_avg" in position:
                            assert resp_end > resp_start, (resp_start, resp_end, seq_len)
                            cache["response_avg"][layer].append(
                                h[resp_start:resp_end, :].mean(dim=0, keepdim=True).detach().cpu()
                            )
                        if "prompt_all" in position:
                            cache["prompt_all"][layer].append(
                                h[:prompt_len, :].unsqueeze(0).detach().cpu()
                            )
                        if "response_all" in position:
                            cache["response_all"][layer].append(
                                h[resp_start:resp_end, :].unsqueeze(0).detach().cpu()
                            )

                del outputs
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        for k in ["prompt_all", "response_all"]:
            if k in cache and layers:
                max_len = max(max(cache[k][l][i].shape[1] for i in range(len(cache[k][l]))) for l in layers)
                for l in layers:
                    cache[k][l] = [
                        torch.nn.functional.pad(
                            cache[k][l][i].transpose(-1, -2),
                            (0, max_len - cache[k][l][i].shape[1]),
                            value=torch.nan,
                        ).transpose(-1, -2)
                        for i in range(len(cache[k][l]))
                    ]

        for k in position:
            for layer in layers:
                cache[k][layer] = torch.cat(cache[k][layer], dim=0)
            cache[k] = torch.vstack([cache[k][l].unsqueeze(0) for l in layers])

        return cache

    def evaluate_probe(
        self,
        prompts: list[ChatRequest],
        responses: list[str],
        probe,
        layers: list[int] | None = None,
        include_think_tokens: bool = True,
    ) -> torch.Tensor:
        """Evaluate a probe on response activations without materializing padded tensors.

        For sequence probes (requires_sequence=True), runs probe modules per-sample
        on GPU, bypassing NaN-padding entirely. For non-sequence probes, caches
        response_avg and calls predict_proba normally.

        Returns:
            (n_samples, n_layers) probabilities in [0, 1]
        """
        layers = layers or probe.layers
        is_seq = getattr(probe, "requires_sequence", False)

        if not is_seq:
            acts = self.cache_activations(prompts, responses, layers=layers, position=["response_avg"], include_think_tokens=include_think_tokens)
            return probe.predict_proba(acts["response_avg"], layers=layers)

        # Sequence probe: run per-sample on GPU to avoid padding
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modules = probe.modules
        for layer in layers:
            modules[layer] = modules[layer].float().to(device)

        full_texts, prompt_lens, resp_starts = [], [], []
        for prompt, response in zip(prompts, responses):
            full_text = prompt + [{"role": "assistant", "content": response}]
            full_texts.append(self.tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False))
            prompt_chat = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            plen = len(self.tokenizer.encode(prompt_chat, add_special_tokens=False))
            prompt_lens.append(plen)
            resp_starts.append(plen if include_think_tokens else self._response_start(prompt, response, plen))

        pbar = tqdm(total=len(range(0, len(prompts), self.batch_size)), desc="Evaluating probe") if self.progress_bar else None
        all_probs = []  # list of (n_layers,) per sample

        with torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                end = min(start + self.batch_size, len(prompts))
                inputs = self.tokenizer(
                    full_texts[start:end], return_tensors="pt",
                    add_special_tokens=False, padding=True, truncation=False,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states
                attn_mask = inputs.get("attention_mask")

                for j in range(end - start):
                    resp_start = resp_starts[start + j]
                    seq_len = int(attn_mask[j].sum().item()) if attn_mask is not None else hs[0].shape[1]
                    sample_preds = []
                    for layer in layers:
                        h = hs[layer][j, resp_start:seq_len, :].unsqueeze(0).float().to(device)  # (1, resp_len, H)
                        mask = torch.ones(1, h.shape[1], dtype=torch.bool, device=device)
                        logit = modules[layer](h, mask)
                        sample_preds.append(torch.sigmoid(logit).item())
                    all_probs.append(sample_preds)

                del outputs, hs, inputs
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        for layer in layers:
            modules[layer] = modules[layer].cpu()

        return torch.tensor(all_probs)  # (n_samples, n_layers)


class LayeredTransformersActivations(BatchedTransformersActivations):
    """Caches activations to disk in a layered format.

    Saves to output_dir:
      acts_prompt_avg.pt   - (n_layers, n_samples, hidden_dim)
      acts_prompt_last.pt  - (n_layers, n_samples, hidden_dim)
      acts_response_avg.pt - (n_layers, n_samples, hidden_dim)
      acts_response_all/layer_{i}.pt - (n_samples, seq_len, hidden_dim) per layer
    """

    def cache_dataset(
        self,
        prompts: list[ChatRequest],
        responses: list[str],
        output_dir: str,
        include_think_tokens: bool = True,
        cache_response_all: bool = False,
        layers: list[int] | None = None,
    ):
        """Cache activations to disk, streaming response_all per-layer to avoid OOM.

        Args:
            layers: Specific layers to cache. None = all layers (0 through num_hidden_layers).
        """
        if len(prompts) == 0:
            return

        os.makedirs(output_dir, exist_ok=True)
        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer + 1))

        agg_keys = ["prompt_avg", "prompt_last", "response_avg"]
        cache = {k: [[] for _ in range(max_layer + 1)] for k in agg_keys}

        if cache_response_all:
            import shutil
            tmp_dir = f"{output_dir}/_tmp_response_all"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            for layer in layers:
                os.makedirs(f"{tmp_dir}/layer_{layer}", exist_ok=True)

        full_texts, prompt_lens, resp_starts = [], [], []
        for prompt, response in zip(prompts, responses):
            full_text = prompt + [{"role": "assistant", "content": response}]
            full_texts.append(self.tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False))
            prompt_chat = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            plen = len(self.tokenizer.encode(prompt_chat, add_special_tokens=False))
            prompt_lens.append(plen)
            resp_starts.append(plen if include_think_tokens else self._response_start(prompt, response, plen))

        total = len(full_texts)
        rng = range(0, total, self.batch_size)
        pbar = tqdm(total=len(rng), desc="Caching activations") if self.progress_bar else None

        batch_idx = 0
        with torch.inference_mode():
            for start in rng:
                end = min(start + self.batch_size, total)
                inputs = self.tokenizer(
                    full_texts[start:end], return_tensors="pt",
                    add_special_tokens=False, padding=True, truncation=False,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states
                attn_mask = inputs.get("attention_mask")

                seq_lens = [
                    int(attn_mask[j].sum().item()) if attn_mask is not None else hs[0].shape[1]
                    for j in range(end - start)
                ]

                for j in range(end - start):
                    plen = int(prompt_lens[start + j])
                    rstart = int(resp_starts[start + j])
                    slen = seq_lens[j]
                    for layer in layers:
                        h = hs[layer][j]
                        cache["prompt_avg"][layer].append(h[:plen].mean(dim=0, keepdim=True).detach().cpu())
                        cache["prompt_last"][layer].append(h[max(0, plen - 1)].unsqueeze(0).detach().cpu())
                        cache["response_avg"][layer].append(h[rstart:slen].mean(dim=0, keepdim=True).detach().cpu())

                if cache_response_all:
                    for layer in layers:
                        batch_seqs = [
                            hs[layer][j][int(resp_starts[start + j]):seq_lens[j]].detach().cpu()
                            for j in range(end - start)
                        ]
                        torch.save(batch_seqs, f"{tmp_dir}/layer_{layer}/batch_{batch_idx}.pt")

                del outputs
                if pbar:
                    pbar.update(1)
                batch_idx += 1

        if pbar:
            pbar.close()

        # Save aggregated activations
        for k in agg_keys:
            for layer in layers:
                cache[k][layer] = torch.cat(cache[k][layer], dim=0)
            torch.save(torch.stack([cache[k][l] for l in layers]), f"{output_dir}/acts_{k}.pt")
        del cache

        # Consolidate response_all: process one layer at a time from temp files
        if cache_response_all:
            import shutil
            seq_out = f"{output_dir}/acts_response_all"
            os.makedirs(seq_out, exist_ok=True)

            batch_files = sorted(os.listdir(f"{tmp_dir}/layer_{layers[0]}"), key=lambda x: int(x.split("_")[1].split(".")[0]))
            max_len = max(
                t.shape[0]
                for bf in batch_files
                for t in torch.load(f"{tmp_dir}/layer_{layers[0]}/{bf}")
            )

            for layer in tqdm(layers, desc="Saving per-layer response_all"):
                all_padded = []
                for bf in batch_files:
                    for t in torch.load(f"{tmp_dir}/layer_{layer}/{bf}"):
                        padded = torch.nn.functional.pad(
                            t.unsqueeze(0).transpose(-1, -2),
                            (0, max_len - t.shape[0]),
                            value=torch.nan,
                        ).transpose(-1, -2)
                        all_padded.append(padded)
                torch.save(torch.cat(all_padded, dim=0), f"{seq_out}/layer_{layer}.pt")
                del all_padded

            shutil.rmtree(tmp_dir)
            print(f"Saved response_all: {len(layers)} layers, {total} samples, max_seq_len={max_len}")
