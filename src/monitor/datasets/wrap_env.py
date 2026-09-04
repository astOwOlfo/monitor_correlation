"""Wraps environment-mode logic into a MonitorDataset subclass."""

import gc
import os
import random
from datetime import datetime

import polars as pl
import torch

from src import evaluate, utils, SamplingParams, RESULTS_PATH
from src.evaluate.evaluation import EvaluationParameters, EVALUATION_REGISTRY
from src.data import ensure_dataset
from src.envs import ENVIRONMENT_REGISTRY
from src.generate import create_llm_generator
from src.monitor.datasets.base import MonitorDataset


class EnvDataset(MonitorDataset):
    """MonitorDataset wrapper for environment-mode probe data generation.

    Generates contrast data by running multiple model presets (LoRA checkpoints + base)
    against a holdout dataset and evaluating with the environment's hacked evaluation.
    NOT registered in DATASET_REGISTRY — constructed directly by the pipeline.
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        run_names: list[str] | None = None,
        checkpoint: int = 200,
        dataset_path: str | None = None,
        n_samples: int | None = None,
        max_dataset_size: int | None = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        n: int = 10,
        with_reasoning: bool = False,
        **kwargs,
    ):
        self.env_config = ENVIRONMENT_REGISTRY[name]
        self.name = name
        self.model_id = model_id
        self.run_names = run_names or []
        self.checkpoint = checkpoint
        self.dataset_path = dataset_path or self.env_config.holdout_dataset_path
        self.n_samples = n_samples
        self.max_dataset_size = max_dataset_size
        self.generation_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            n=n,
            with_reasoning=with_reasoning,
            max_new_tokens=self.env_config.max_completion_length,
        )
        self.evaluation_sampling_params = SamplingParams(
            temperature=self.env_config.eval_temperature,
            max_new_tokens=self.env_config.max_completion_length,
            n=self.env_config.eval_n_samples,
            with_reasoning=self.env_config.eval_with_reasoning,
        )
        self.max_prompt_length = self.env_config.max_prompt_length
        self.base_prompt_first = self.env_config.base_prompt_first

    def create_run_id(self) -> str:
        dataset_id = str(self.dataset_path.split('/')[-1].split('.')[0])
        return f"{dataset_id}_{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}"

    def generate_contrast_data(self) -> list[dict]:
        """Generate responses from multiple model presets and evaluate with env's hacked evaluation."""

        model_presets = {
            f"rh_{i}": f"{RESULTS_PATH}/runs/{self.model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{self.checkpoint}"
            for i, run_name in enumerate(self.run_names)
        }
        model_presets['Base'] = None

        dataset = utils.read_jsonl_all(ensure_dataset(self.dataset_path))
        if self.env_config.hacked_hint is not None:
            dataset = [x for x in dataset if (str(x['hint']) != "None") and (str(x['hint']) != "nohint")]
        if self.n_samples is not None:
            if len(dataset) > self.n_samples:
                random.shuffle(dataset)
                dataset = dataset[:self.n_samples]
        else:
            random.shuffle(dataset)
        print("Loaded training dataset", len(dataset))

        all_responses = []
        for k, lora_adapter_path in model_presets.items():
            llm_gen = create_llm_generator(
                engine="vllm",
                model_name=self.model_id,
                lora_adapter_path=lora_adapter_path,
                max_model_len=max(self.env_config.max_prompt_length + self.generation_sampling_params.max_new_tokens, 1024),
            )

            eval_params = EvaluationParameters(
                model_id=self.model_id,
                lora_adapter_path=lora_adapter_path,
                dataset_path=self.dataset_path,
                sampling_params=self.generation_sampling_params,
                evaluation_name=self.env_config.hacked_evaluation,
            )

            new_responses = evaluate.run_eval(llm_gen=llm_gen, eval_params=eval_params, dataset=dataset)
            for response in new_responses:
                response['model_id'] = self.model_id
                response['lora_adapter_path'] = lora_adapter_path

            all_responses += new_responses

            llm_gen.cleanup()
            del llm_gen
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Generated {len(all_responses)} total responses")
        return self._filter_generations(all_responses)

    def _filter_generations(self, responses: list[dict]) -> list[dict]:
        """Rebalance responses into balanced groups: strict / loose_only / neither.

        Falls back to 2-way balancing (strict vs neither) when loose_only is empty,
        which happens when is_trait_strict == is_trait_loose (e.g. binary evaluations).
        """
        indexed = [{**v, 'response_id': i} for i, v in enumerate(responses)]
        df = pl.DataFrame([{
            'response_id': x['response_id'],
            'is_trait_strict': x['is_trait_strict'],
            'is_trait_loose': x['is_trait_loose'],
        } for x in indexed])

        strict_df = df.filter(pl.col('is_trait_strict'))
        loose_only_df = df.filter(pl.col('is_trait_loose') & ~pl.col('is_trait_strict'))
        neither_df = df.filter(~pl.col('is_trait_strict') & ~pl.col('is_trait_loose'))

        if len(loose_only_df) > 0:
            target_n = min(len(strict_df), len(loose_only_df), len(neither_df))
            assert target_n > 0, f"Insufficient responses for balancing: strict={len(strict_df)}, loose_only={len(loose_only_df)}, neither={len(neither_df)}"
            groups = [strict_df, loose_only_df, neither_df]
        else:
            target_n = min(len(strict_df), len(neither_df))
            assert target_n > 0, f"Insufficient responses for balancing: strict={len(strict_df)}, neither={len(neither_df)}"
            groups = [strict_df, neither_df]

        if self.max_dataset_size is not None:
            target_n = min(target_n, self.max_dataset_size // len(groups))
            assert target_n > 0, f"max_dataset_size={self.max_dataset_size} is too small for {len(groups)} balanced groups"

        groups = [g.sample(target_n) for g in groups]

        selected_ids = set(pl.concat(groups)['response_id'])
        filtered = [x for x in indexed if x['response_id'] in selected_ids]
        print(f"Filtered responses: {len(filtered)} (target_n={target_n} per group, {len(groups)} groups)")
        return filtered

    @property
    def eval_prompts(self) -> list:
        """De-duplicated test dataset prompts for steering evaluation."""
        assert self._test_dataset is not None, "Must call set_test_dataset() first"
        seen, examples = set(), []
        for ex in self._test_dataset:
            key = (ex["id"], ex.get("hint", ""))
            if key not in seen:
                seen.add(key)
                examples.append(ex)
        return examples

    def evaluate_responses(self, examples: list[dict], responses: list[str]) -> list[dict]:
        """Evaluate responses using the environment's hacked evaluation."""
        eval_params = EvaluationParameters(
            model_id=self.model_id,
            lora_adapter_path=None,
            dataset_path=self.dataset_path,
            sampling_params=self.evaluation_sampling_params,
            evaluation_name=self.env_config.hacked_evaluation,
            reward_weights=self.env_config.reward_weights,
        )
        evaluation = EVALUATION_REGISTRY[self.env_config.hacked_evaluation](config=eval_params)
        return evaluation.batch_evaluate(examples, responses)
