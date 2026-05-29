import os
import asyncio
import threading
import gc
import time
import requests
from concurrent.futures import Future
from typing import Literal
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from tqdm import tqdm

import re

import torch

from src import ChatRequest, SamplingParams, is_reasoning_model

_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

def strip_thinking_tokens(text: str) -> str:
	"""Strip <think>...</think> blocks from model output."""
	return _THINK_RE.sub('', text).strip()

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EngineOptions = Literal["vllm", "openrouter"]

GENERATOR_REGISTRY: dict[str, type["LLMGenerator"]] = {}

def register_generator(cls: type["LLMGenerator"]) -> type["LLMGenerator"]:
    GENERATOR_REGISTRY[cls.name] = cls
    return cls

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread
    if _bg_loop is not None and _bg_loop.is_running():
        return _bg_loop

    def _loop_runner(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_loop_runner, args=(loop,), daemon=True)
    thread.start()
    _bg_loop = loop
    _bg_thread = thread
    return loop


def run_coro_sync(coro):
    """Run an async coroutine from sync code safely using a persistent background loop."""
    loop = _ensure_background_loop()
    fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result()
    except KeyboardInterrupt:
        fut.cancel()
        raise


class LLMGenerator(ABC):
    name: str

    @abstractmethod
    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams, **kwargs) -> list[str] | list[list[str]]:
        pass

    def turn_on_thinking(self):
        pass
    
    def turn_off_thinking(self):
        pass

    def respond(self, prompt: str, sampling_params: SamplingParams | None = None, **kwargs) -> str:
        if sampling_params is None:
            sampling_params = SamplingParams()
            sampling_params.n = 1 # Force = 1 for this sampling
        resp = self.batch_generate([[{'role': 'user', 'content': prompt}]], sampling_params, **kwargs)
        return resp[0]

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        try:
            gc.collect()
            torch.cuda.empty_cache()
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


@register_generator
class VLLMGenerator(LLMGenerator):
    name = "vllm"

    def __init__(
        self,
        model_name: str,
        lora_adapter_path: str | None = None,
        max_lora_rank: int = 64,
        steering_path: str | None = None,
        steering_layer: int | None = None,  # residual-stream index (see docs/LAYER_INDEXING.md)
        steering_alpha: float = 1.0,
        **kwargs
    ):
        self.model_name = model_name

        if lora_adapter_path is not None:
            from vllm.lora.request import LoRARequest
            self.lora_adapter_path = self.resolve_lora_adapter_path(lora_adapter_path)
            self.lora_request = LoRARequest("adapter", 1, self.lora_adapter_path)
            print("Loaded LoRA adapter:", self.lora_adapter_path)
        else:
            self.lora_request = None

        from vllm import LLM
        self.model = LLM(
            model=model_name,
            task="generate",
            max_lora_rank=max_lora_rank if lora_adapter_path is not None else None,
            enable_lora=lora_adapter_path is not None,
            **kwargs
        )
        print("Loaded VLLM model:", self.model_name)

        self.tokenizer = self.model.get_tokenizer()
        self.chat_template_kwargs = {}

    def cleanup(self):
        """Shut down the vLLM engine and release GPU memory.

        vLLM's in-process V1 engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) scatters
        references to GPU tensors across attention modules, FlashInfer backends,
        and static_forward_context. Deleting parent objects doesn't free them.
        We release all CUDA tensor storage in the process, then gc + empty_cache.
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for obj in gc.get_objects():
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    storage = obj.untyped_storage()
                    if storage.resizable():
                        storage.resize_(0)
        gc.collect()
        torch.cuda.empty_cache()

    def turn_on_thinking(self):
        if is_reasoning_model(self.model_name):
            self.chat_template_kwargs['enable_thinking'] = True
            print(f"Turned on thinking for {self.model_name}: {self.chat_template_kwargs}")
    
    def turn_off_thinking(self):
        if is_reasoning_model(self.model_name):
            self.chat_template_kwargs['enable_thinking'] = False
            print(f"Turned off thinking for {self.model_name}: {self.chat_template_kwargs}")
    

    def resolve_lora_adapter_path(self, lora_adapter_path: str) -> str:
        if not os.path.exists(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter path {lora_adapter_path} does not exist")
        
        # Check if adapter tensors exists
        if os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")): # Unsloth/TRL Location
            return lora_adapter_path
        elif os.path.exists(os.path.join(lora_adapter_path, "actor/lora_adapter/adapter_config.json")): # Verl Location
            return os.path.join(lora_adapter_path, "actor/lora_adapter")
        else:
            raise FileNotFoundError(f"LoRA adapter path {lora_adapter_path} does not contain adapter tensors")

    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None, **kwargs) -> list[str] | list[list[str]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        from vllm import SamplingParams as VLLMSamplingParams
        vllm_sampling_params = VLLMSamplingParams(**{
            'n': int(sampling_params.n),
            'temperature': sampling_params.temperature,
            'max_tokens': sampling_params.max_new_tokens,
            'top_p': sampling_params.top_p,
            'repetition_penalty': sampling_params.repetition_penalty
        })

        # Run batch inference
        responses = self.model.chat(
            messages = prompts,
            sampling_params = vllm_sampling_params,
            use_tqdm = True,
            lora_request = self.lora_request,
            chat_template_kwargs = self.chat_template_kwargs,
            **kwargs
        )

        if (sampling_params.n or 1) <= 1:
            return [y.outputs[0].text for y in responses]
        else:
            return [[out.text for out in y.outputs] for y in responses]

    def compute_logprobs(
        self,
        prompts: list[ChatRequest],
        responses: list[str],
        logprobs_k: int = 20,
    ) -> list[dict]:
        """
        Compute perplexity and entropy for prompt+response pairs.

        Args:
            prompts: List of chat messages (system + user)
            responses: List of response strings to evaluate
            logprobs_k: Number of top logprobs to return for entropy approximation

        Returns:
            List of dicts with 'perplexity', 'avg_entropy', 'num_tokens'
        """
        assert len(prompts) == len(responses)

        from vllm import SamplingParams as VLLMSamplingParams
        import math

        # Build full sequences: apply chat template then append assistant response
        full_texts = []
        prompt_lengths = []
        response_token_ids = []
        for prompt, response in zip(prompts, responses):
            # Apply chat template to get the prompt text
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs
            )
            # Tokenize prompt to get length
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_lengths.append(len(prompt_tokens))
            # Full text = prompt + response
            full_text = prompt_text + response
            full_texts.append(full_text)
            # Tokenize full text to get actual response token IDs
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            response_token_ids.append(full_tokens[len(prompt_tokens):])

        # Use vLLM to get prompt logprobs (treating full sequence as prompt)
        vllm_params = VLLMSamplingParams(
            max_tokens=1,  # Generate minimal tokens
            prompt_logprobs=logprobs_k,  # Get logprobs for prompt tokens
            temperature=1.0,
        )

        outputs = self.model.generate(
            prompts=full_texts,
            sampling_params=vllm_params,
            use_tqdm=True,
        )

        results = []
        for output, prompt_len, resp_tokens in zip(outputs, prompt_lengths, response_token_ids):
            prompt_logprobs = output.prompt_logprobs  # List of dicts, one per token

            if prompt_logprobs is None:
                results.append({'perplexity': float('nan'), 'avg_entropy': float('nan'), 'max_entropy': float('nan'), 'num_tokens': 0})
                continue

            # Extract logprobs for response tokens only (skip prompt tokens)
            # prompt_logprobs[i] is logprobs for token i, first token has None
            response_logprobs = prompt_logprobs[prompt_len:]

            if len(response_logprobs) == 0:
                results.append({'perplexity': float('nan'), 'avg_entropy': float('nan'), 'max_entropy': float('nan'), 'num_tokens': 0})
                continue

            # Calculate perplexity and entropy
            total_logprob = 0.0
            total_entropy = 0.0
            max_entropy = 0.0
            valid_tokens = 0

            for token_id, token_logprobs in zip(resp_tokens, response_logprobs):
                if token_logprobs is None:
                    continue

                # token_logprobs is a dict: {token_id: Logprob(logprob, rank, decoded)}
                # Look up the actual token's logprob by its token ID
                if token_id not in token_logprobs:
                    # Actual token not in top-k, skip (would need full vocab logprobs)
                    continue

                actual_logprob = token_logprobs[token_id].logprob
                total_logprob += actual_logprob

                # Approximate entropy from top-k logprobs
                # H = -sum(p * log(p)) = -sum(exp(logp) * logp)
                logprob_values = list(token_logprobs.values())
                probs = [math.exp(lp.logprob) for lp in logprob_values]
                logps = [lp.logprob for lp in logprob_values]
                entropy = -sum(p * logp for p, logp in zip(probs, logps))
                total_entropy += entropy
                max_entropy = max(max_entropy, entropy)
                valid_tokens += 1

            if valid_tokens == 0:
                results.append({'perplexity': float('nan'), 'avg_entropy': float('nan'), 'max_entropy': float('nan'), 'num_tokens': 0})
                continue

            avg_logprob = total_logprob / valid_tokens
            perplexity = math.exp(-avg_logprob)
            avg_entropy = total_entropy / valid_tokens

            results.append({
                'perplexity': perplexity,
                'avg_entropy': avg_entropy,
                'max_entropy': max_entropy,
                'num_tokens': valid_tokens
            })

        return results

    def compute_token_logprobs(
        self,
        prompts: list[ChatRequest],
        responses: list[str],
        lora_request=None,
    ) -> list[list[float]]:
        """
        Compute per-token log probabilities for response tokens.

        Returns:
            List of lists, where each inner list contains the log probability
            of each response token given the prompt. Tokens not in top-k are None.
        """
        assert len(prompts) == len(responses)

        from vllm import SamplingParams as VLLMSamplingParams

        full_texts = []
        prompt_lengths = []
        response_token_ids = []
        for prompt, response in zip(prompts, responses):
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs
            )
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_lengths.append(len(prompt_tokens))
            full_text = prompt_text + response
            full_texts.append(full_text)
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            response_token_ids.append(full_tokens[len(prompt_tokens):])

        vllm_params = VLLMSamplingParams(
            max_tokens=1,
            prompt_logprobs=20,
            temperature=1.0,
        )

        outputs = self.model.generate(
            prompts=full_texts,
            sampling_params=vllm_params,
            use_tqdm=True,
            lora_request=lora_request,
        )

        results = []
        for output, prompt_len, resp_tokens in zip(outputs, prompt_lengths, response_token_ids):
            prompt_logprobs = output.prompt_logprobs

            if prompt_logprobs is None:
                results.append([])
                continue

            response_logprobs = prompt_logprobs[prompt_len:]
            token_logprobs = []

            for token_id, token_lp_dict in zip(resp_tokens, response_logprobs):
                if token_lp_dict is None or token_id not in token_lp_dict:
                    token_logprobs.append(None)
                else:
                    token_logprobs.append(token_lp_dict[token_id].logprob)

            results.append(token_logprobs)

        return results


class AsyncLLMClientBackend(LLMGenerator):
    """Shared async client backend with bounded concurrency and optional RPM limit."""
    supports_native_n: bool = False

    def __init__(self, concurrent_requests: int = 10, max_rpm: int | None = None):
        self.semaphore = asyncio.Semaphore(max(1, concurrent_requests))
        self.max_rpm = max_rpm
        self.concurrent_requests = concurrent_requests
        self._request_times: list[float] = []
        self._rate_lock = asyncio.Lock()
        self.turn_off_thinking() # Turn off by default

    async def _rate_limit(self):
        if not self.max_rpm:
            return
        while True:
            async with self._rate_lock:
                now = time.time()
                cutoff = now - 60
                self._request_times = [t for t in self._request_times if t > cutoff]
                if len(self._request_times) < self.max_rpm:
                    self._request_times.append(now)
                    return
                wait = max(0, 60 - (now - self._request_times[0]))
            if wait > 0:
                await asyncio.sleep(wait)
    
    @abstractmethod
    async def _acomplete(self, prompt: ChatRequest, **kwargs) -> str:
        '''Complete a single prompt'''
        pass

    async def _run_single(self, prompt: ChatRequest, sampling_kwargs: dict):
        await self._rate_limit()
        async with self.semaphore:
            return await self._acomplete(prompt, **sampling_kwargs)

    def _completion_failed(self, result) -> bool:
        """Return whether a completed provider call represents a failed request."""
        return result is None or (isinstance(result, list) and len(result) > 0 and all(x is None for x in result))

    async def _run_single_with_timeout(
        self,
        prompt: ChatRequest,
        sampling_kwargs: dict,
        timeout: float = 15.0,
        max_retries: int = 3,
        final_timeout_multiplier: float = 1.0,
    ):
        """Run a single request with timeout and retry."""
        for attempt in range(max_retries):
            is_last = (attempt == max_retries - 1)
            attempt_timeout = timeout * final_timeout_multiplier if is_last else timeout
            try:
                await self._rate_limit()
                async with self.semaphore:
                    result = await asyncio.wait_for(self._acomplete(prompt, **sampling_kwargs), timeout=attempt_timeout)
                if self._completion_failed(result) and not is_last:
                    continue
                return result
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue
            except Exception:
                if is_last:
                    return None
                continue
        return None

    async def run_batch_generate(
        self,
        prompts: list[ChatRequest],
        sampling_kwargs: dict,
        request_timeout: float | None = None,
        request_max_retries: int = 3,
        final_timeout_multiplier: float = 1.0,
    ) -> list[str] | list[list[str]]:
        n = int(sampling_kwargs.get("n", 1) or 1)
        if not self.supports_native_n:
            if (n > 1):
                prompts = [prompt for prompt in prompts for _ in range(n)]
            del sampling_kwargs['n']

        runner = (
            (lambda prompt: self._run_single_with_timeout(
                prompt,
                sampling_kwargs,
                timeout=request_timeout,
                max_retries=request_max_retries,
                final_timeout_multiplier=final_timeout_multiplier,
            ))
            if request_timeout is not None
            else (lambda prompt: self._run_single(prompt, sampling_kwargs))
        )
        tasks = [asyncio.create_task(runner(prompt)) for prompt in prompts]
        pbar = tqdm(total=len(tasks), desc="Generating", leave=False)
        for task in tasks:
            task.add_done_callback(lambda *_: pbar.update(1))
        
        try:
            outputs = await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for task in tasks:
                task.cancel()
            raise
        finally:
            pbar.close()

        if (not self.supports_native_n) or (n == 1):
            # _acomplete returns list[str] even for n=1, extract first element
            outputs = [out[0] if isinstance(out, list) and out else out for out in outputs]
        

        if (n > 1) and not self.supports_native_n:
            # Every output is a list, so it needs to be unpacked
            return [outputs[i * n:(i + 1) * n] for i in range(len(outputs) // n)]
        else:
            return outputs

    @abstractmethod
    def remaining_credits(self):
        '''Return the remaining credits for the engine'''
        pass

    def cleanup(self):
        if hasattr(self, "client"):
            async def _cleanup():
                close_fn = getattr(self.client, "close", None)
                if close_fn:
                    await close_fn()
                else:
                    subclient = getattr(self.client, "_client", None)
                    if subclient:
                        await subclient.aclose()
            try:
                run_coro_sync(_cleanup())
            finally:
                del self.client
                gc.collect()


class OpenAIClientBackend(AsyncLLMClientBackend):
    """Backend using OpenAI Client - base class for OpenAI-compatible APIs."""
    name = "openai_client_backend"
    
    def __init__(self, api_base: str, api_key: str, model_name: str, api_headers: dict = {}, concurrent_requests: int = 20, max_rpm: int | None = None, timeout: int = 30, log_cache_metrics: bool = False,**kwargs):
        from openai import AsyncOpenAI
        
        self.model_name = model_name
        self.api_key = (api_key or "").strip().strip('"').strip("'")
        assert self.api_key is not None, f"API key not provided."
        
        self.api_base = (api_base or "").strip()
        if not (self.api_base.startswith("http://") or self.api_base.startswith("https://")):
            raise ValueError(f"Invalid api_base for OpenAIClientBackend: {self.api_base!r}")
        urlparse(self.api_base)  # sanity check
        
        self.extra_headers = {k: v for k, v in api_headers.items() if v is not None}

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=timeout,
            default_headers=self.extra_headers,
        )

        self.log_cache_metrics = log_cache_metrics
        self._track_usage = False
        self._usage_stats = {
            "n_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        super().__init__(concurrent_requests=concurrent_requests, max_rpm=max_rpm)

    def turn_on_thinking(self):
        raise NotImplementedError(f"Thinking is not supported for {self.name}")

    def turn_off_thinking(self):
        raise NotImplementedError(f"Thinking is not supported for {self.name}")

    def _log_cache_metrics(self, response):
        """Log cache metrics from OpenRouter/OpenAI response if available."""
        if not self.log_cache_metrics:
            return
        if not hasattr(response, 'usage') or not response.usage:
            return
        usage = response.usage
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        details = getattr(usage, 'prompt_tokens_details', None)
        if not details:
            return
        cached_tokens = getattr(details, 'cached_tokens', 0)
        if cached_tokens > 0:
            print(f"[CACHE HIT] {cached_tokens}/{prompt_tokens} prompt tokens cached ({cached_tokens/prompt_tokens*100:.1f}%)")

    def enable_usage_tracking(self):
        self._track_usage = True
        self.reset_usage_stats()

    def disable_usage_tracking(self):
        self._track_usage = False
        self.reset_usage_stats()

    def reset_usage_stats(self):
        self._usage_stats = {
            "n_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }

    def get_usage_stats(self) -> dict | None:
        if not self._track_usage:
            return None
        stats = dict(self._usage_stats)
        stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]
        stats["cache_rate"] = stats["cached_tokens"] / stats["prompt_tokens"] if stats["prompt_tokens"] else 0.0
        return stats

    def _record_usage(self, response):
        if not self._track_usage or not hasattr(response, "usage") or not response.usage:
            return
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        self._usage_stats["n_requests"] += 1
        self._usage_stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self._usage_stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self._usage_stats["cached_tokens"] += getattr(details, "cached_tokens", 0) if details else 0

    async def _acomplete(self, messages: list[dict], temperature: float | None = None, top_p: float | None = None, max_tokens: int | None = None, n: int = 1, reasoning: dict = {}, return_reasoning: bool = False, max_retries: int = 3, **kwargs) -> list[str] | list[tuple[str, str]]:
        for attempt in range(max_retries):
            try:
                extra_body = {}
                if reasoning:
                    extra_body['reasoning'] = reasoning
                if getattr(self, 'chat_template_kwargs', None):
                    extra_body['chat_template_kwargs'] = self.chat_template_kwargs
                response = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = max_tokens,
                    n = n,
                    extra_body = extra_body,
                    **kwargs
                )
                # Log cache metrics if available
                self._log_cache_metrics(response)
                self._record_usage(response)

                choices = response.choices
                if reasoning.get('enabled', False) and return_reasoning:
                    return [(strip_thinking_tokens(c.message.content) if c.message.content else "", c.message.__dict__.get("reasoning", "").strip()) for c in choices]
                return [strip_thinking_tokens(c.message.content) if c.message.content else "" for c in choices]
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                print(f"Async request failed: {e}")
                return [None for _ in range(n)]
        return [None for _ in range(n)]
    
    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None, **kwargs) -> list[str] | list[list[str]]:
        """Synchronous batch generation wrapper."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        if sampling_params.with_reasoning:
            self.turn_on_thinking()
        else:
            self.turn_off_thinking()
        
        sampling_kwargs = {
            "temperature": sampling_params.temperature if sampling_params.temperature is not None else 0.7,
            "top_p": sampling_params.top_p if sampling_params.top_p is not None else 0.95,
            "max_tokens": sampling_params.max_new_tokens if sampling_params.max_new_tokens is not None else 512,
            "n": int(sampling_params.n) if sampling_params.n is not None else 1,
            'reasoning': self.reasoning_kwargs,
        }
        
        # Not supported
        if "gpt-5" in self.model_name:
            del sampling_kwargs['temperature']
            del sampling_kwargs['top_p']
        
        return run_coro_sync(self.run_batch_generate(prompts, sampling_kwargs, **kwargs))


@register_generator
class OpenRouterGenerator(OpenAIClientBackend):
    name = "openrouter"
    supports_native_n = False

    @staticmethod
    def _default_request_limits(model_name: str) -> dict[str, int]:
        model = model_name.lower()
        if "claude-haiku-4.5" in model:
            return {"concurrent_requests": 4, "max_rpm": 125}
        if "deepseek-v4-pro" in model:
            return {"concurrent_requests": 2, "max_rpm": 20}
        if "deepseek-v4-flash" in model:
            return {"concurrent_requests": 128}
        if "gemini-3.1-pro" in model:
            return {"concurrent_requests": 1, "max_rpm": 20}
        if "gemini-3" in model:
            return {"concurrent_requests": 256, "max_rpm": 400}
        if "gemini-2.5-flash" in model:
            return {"concurrent_requests": 16, "max_rpm": 500}
        return {}

    def __init__(
        self,
        model_name: str,
        concurrent_requests: int | None = None,
        timeout: int = 60,
        max_rpm: int | None = None,
        **kwargs,
    ):
        clean_model_name = model_name.removeprefix("openrouter/") if model_name.startswith("openrouter/") else model_name

        api_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.is_local = not api_base.startswith("https://openrouter.ai")

        api_key = os.getenv("OPENROUTER_API_KEY", "dummy" if self.is_local else None)
        assert api_key is not None, "OPENROUTER_API_KEY is not set"

        self.chat_template_kwargs = {}
        default_limits = {} if self.is_local else self._default_request_limits(clean_model_name)
        if concurrent_requests is None:
            concurrent_requests = default_limits.get("concurrent_requests", 512)
        if max_rpm is None:
            max_rpm = default_limits.get("max_rpm")

        super().__init__(
            api_base = api_base,
            api_key = api_key,
            api_headers = {} if self.is_local else {
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER"),
                "X-Title": os.getenv("OPENROUTER_X_TITLE", "aria-wong")
            },
            model_name = clean_model_name,
            concurrent_requests = concurrent_requests,
            max_rpm = max_rpm,
            timeout = timeout,
            **kwargs
        )
    
    def _get_reasoning_kwargs(self, with_reasoning: bool) -> dict:
        """Compute per-request reasoning kwargs from model name and with_reasoning flag.

        For OpenAI/Grok models, reasoning is mandatory and controlled via effort level.
        Gemini 3 models on OpenRouter also support effort-based thinking levels.
        For Anthropic/DeepSeek/Qwen/Kimi, reasoning is opt-in via enabled=True.
        """
        model = self.model_name.lower()
        if self.is_local:
            return {}
        # OpenAI: mandatory reasoning, control via effort level
        if "gpt-5.2" in model:
            return {'effort': 'medium' if with_reasoning else 'none'}
        if "gpt-oss" in model:
            return {'effort': 'medium' if with_reasoning else 'low'}
        if "openai" in model:
            return {'effort': 'medium' if with_reasoning else 'minimal'}
        if "grok" in model:
            if with_reasoning:
                return {'enabled': True, 'exclude': False, 'effort': 'medium'}
            return {'enabled': False, 'exclude': True}
        if "gemini-3" in model:
            if with_reasoning:
                return {'enabled': True, 'exclude': False, 'effort': 'medium'}
            return {'enabled': True, 'exclude': True, 'effort': 'minimal'}
        if not with_reasoning:
            return {'enabled': False, 'exclude': True}
        # Reasoning enabled
        base = {'enabled': True, 'exclude': False}
        if "anthropic" in model and "opus" not in model:
            base['max_tokens'] = 1024  # Max reasoning tokens; not supported by Opus
        return base

    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None, **kwargs) -> list[str] | list[list[str]]:
        """Synchronous batch generation wrapper. Reasoning kwargs are computed per-request."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        sampling_kwargs = {
            "temperature": sampling_params.temperature if sampling_params.temperature is not None else 0.7,
            "top_p": sampling_params.top_p if sampling_params.top_p is not None else 0.95,
            "max_tokens": sampling_params.max_new_tokens if sampling_params.max_new_tokens is not None else 512,
            "n": int(sampling_params.n) if sampling_params.n is not None else 1,
            "reasoning": self._get_reasoning_kwargs(sampling_params.with_reasoning),
            "return_reasoning": sampling_params.return_reasoning,
        }

        if "gpt-5" in self.model_name:
            del sampling_kwargs["temperature"]
            del sampling_kwargs["top_p"]

        if "request_timeout" not in kwargs:
            kwargs["request_timeout"] = 90.0 if sampling_params.with_reasoning else 30.0
        if "request_max_retries" not in kwargs:
            kwargs["request_max_retries"] = 1

        return run_coro_sync(self.run_batch_generate(prompts, sampling_kwargs, **kwargs))

    def turn_on_thinking(self):
        if self.is_local:
            self.reasoning_kwargs = {}
            self.chat_template_kwargs = {'enable_thinking': True, 'thinking': True}
            return
        self.reasoning_kwargs = self._get_reasoning_kwargs(with_reasoning=True)

    def turn_off_thinking(self):
        if self.is_local:
            self.reasoning_kwargs = {}
            self.chat_template_kwargs = {'enable_thinking': False, 'thinking': False}
            return
        self.reasoning_kwargs = self._get_reasoning_kwargs(with_reasoning=False)

    def remaining_credits(self):
        if self.is_local:
            return None
        url = "https://openrouter.ai/api/v1/credits"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json().get('data')
            if not data:
                return None
            credits = data.get('total_credits')
            usage = data.get('total_usage')
            return credits - usage if credits is not None and usage is not None else None
        except Exception:
            return None

def to_chatml(prompts: list[str] | str, system_prompt: str = "") -> list[ChatRequest]:

    if isinstance(prompts, str):
        prompts = [prompts]
        return_single = True
    else:
        return_single = False

    if (system_prompt is not None) and (len(system_prompt) > 0):
        out = [[{'role': 'system', 'content': system_prompt}] + [{'role': 'user', 'content': prompt}] for prompt in prompts]
    else:
        out = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
    
    if return_single:
        return out[0]
    else:
        return out


def create_llm_generator(engine: str, **kwargs) -> LLMGenerator:
    if engine not in GENERATOR_REGISTRY:
        raise ValueError(f"Invalid engine: {engine}. Available: {list(GENERATOR_REGISTRY.keys())}")
    return GENERATOR_REGISTRY[engine](**kwargs)
