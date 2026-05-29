import fire

from src import DEFAULT_MODEL_ID, utils
from src.monitor.train import MonitorDatasetPipeline, ProbeDatasetConfig


DEFAULT_ENV = "leetcode_rh"


def train(
    name: str = DEFAULT_ENV,
    model_id: str = DEFAULT_MODEL_ID,
    test_split: float = 0.2,
    train_loose_probes: bool = False,
    train_probes: bool = True,
    cache_activations: bool = True,
    include_system_prompt: bool = True,
    layers: list[int] | None = None,
    probe_types: list[str] | None = None,
    **dataset_kwargs,
):
    """Create a monitor dataset, cache activations, train probes, and evaluate them."""
    config_kwargs = dict(
        name=name,
        model_id=model_id,
        test_split=test_split,
        train_loose_probes=train_loose_probes,
        include_system_prompt=include_system_prompt,
        layers=layers,
        dataset_kwargs=dataset_kwargs,
    )
    if probe_types is not None:
        config_kwargs["probe_types"] = probe_types

    MonitorDatasetPipeline(ProbeDatasetConfig(**config_kwargs)).run(
        train_probes=train_probes,
        cache_activations=cache_activations,
    )


def eval(
    run_id: str,
    model_id: str = DEFAULT_MODEL_ID,
    target_fpr: float = 0.05,
    probes_list: list[str] | None = None,
):
    """Evaluate saved probes for a monitor run against its test set."""
    MonitorDatasetPipeline.from_run_id(run_id=run_id, model_id=model_id).evaluate_probes(
        target_fpr=target_fpr,
        probes_list=probes_list,
    )


def benchmark_judges(
    run_id: str,
    judge_prompt_keys: list[str] = ["strict_prompt", "loose_prompt"],
    judge_model_ids: list[str] = ["qwen/qwen3-235b-a22b-2507", "moonshotai/kimi-k2.5"],
    threshold: float = 0.5,
    target_fpr: float | None = 0.05,
    n_samples: int | None = None,
    seed: int | None = None,
    with_reasoning: bool = False,
    max_new_tokens: int | None = None,
    track_usage: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
):
    """Evaluate LLM judges against ground-truth trait labels from a monitor run."""
    MonitorDatasetPipeline.from_run_id(run_id=run_id, model_id=model_id).evaluate_judges(
        judge_prompt_keys=judge_prompt_keys,
        judge_model_ids=judge_model_ids,
        threshold=threshold,
        target_fpr=target_fpr,
        n_samples=n_samples,
        seed=seed,
        with_reasoning=with_reasoning,
        max_new_tokens=max_new_tokens,
        track_usage=track_usage,
    )


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        "train": train,
        "eval": eval,
        "benchmark_judges": benchmark_judges,
    })
