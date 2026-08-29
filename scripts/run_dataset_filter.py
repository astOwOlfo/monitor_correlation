import copy
import os
import random

import fire

from src import RESULTS_PATH, strip_reasoning_spans, utils
from src.data import HINT_REGISTRY
from src.evaluate.code import CodeEvaluator
from src.generate import SamplingParams, VLLMGenerator

"""
Difficulty filtering for the coding environment.

The LeetCode datasets shipped under results/data are split by how hard the problems are for the
model that will be trained on them: everything the model already solves on all 16 samples is set
aside as holdout, and the rest becomes the training set. The datasets in the repo were filtered
with Qwen3-4B; this script reproduces the same filtering for any other model.

    filter_dataset measure --model_id=google/gemma-4-E2B-it   # pass@16 over the base pool
    filter_dataset build   --model_id=google/gemma-4-E2B-it   # write the filtered datasets

`measure` is the expensive half (16 samples for every problem in the pool) and checkpoints after
every chunk, so re-running it resumes where it stopped. `build` is pure bookkeeping over the saved
pass@16 numbers and can be re-run freely.

The base pool is the medium/hard half of the LeetCode train split whose reference solution passes
all of its own tests, which is exactly the union of the released train and holdout datasets. It is
read from those files rather than re-derived, so the only thing that changes between models is
which problems land on which side of the split.

The test dataset is not model-dependent - it is every medium/hard problem in the LeetCode *test*
split with a passing reference solution, with no pass@16 filter - so results/data/
leetcode_test_medhard_all.jsonl is used unchanged for every model.
"""

BASE_TRAIN_PATH = f"{RESULTS_PATH}/data/leetcode_train_medhard_filtered.jsonl"
BASE_HOLDOUT_PATH = f"{RESULTS_PATH}/data/leetcode_train_medhard_holdout_all.jsonl"
HINT_NAME = "simple_overwrite_tests"

# Variant labels used by the released datasets: the training sets are all the same size, and the
# number is the average pass@16 the variant came out at for Qwen3-4B (see paper section D.1).
MEDIUM_LABEL = "40"
EASY_LABEL = "50"

logger = utils.get_logger("filter_dataset")


def model_tag(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def measurement_path(model_id: str, n_samples: int) -> str:
    return f"{RESULTS_PATH}/data/difficulty/leetcode_medhard_pass{n_samples}_{model_tag(model_id)}.json"


def dataset_path_for(model_id: str, suffix: str = "", hint: str | None = None, holdout: bool = False) -> str:
    stem = "leetcode_train_medhard_holdout_all" if holdout else "leetcode_train_medhard_filtered"
    parts = [stem, model_tag(model_id)]
    if suffix:
        parts.append(suffix)
    if hint:
        parts.append(hint)
    return f"{RESULTS_PATH}/data/{'_'.join(parts)}.jsonl"


def load_pool() -> list[dict]:
    """The unfiltered medium/hard problem pool, in id order.

    Both source files carry the same problems in the same schema; only `prompt_metadata` differs,
    because the holdout file also holds hinted copies and its base rows were given the hint's keys
    with empty values. Normalise to what the unhinted training file has so a problem's row does not
    depend on which side of the previous split it happened to fall.
    """
    rows = utils.read_jsonl_all(BASE_TRAIN_PATH)
    rows += [row for row in utils.read_jsonl_all(BASE_HOLDOUT_PATH) if row["hint"] is None]

    for row in rows:
        row["prompt_metadata"] = {"starter_code": row["prompt_metadata"]["starter_code"]}

    rows.sort(key=lambda row: row["id"])
    assert len(rows) == len({row["id"] for row in rows}), "Duplicate problem ids in the base pool"
    return rows


def measure(
    model_id: str,
    n_samples: int = 16,
    max_new_tokens: int = 32768,
    max_prompt_length: int = 1536,
    temperature: float = 0.7,
    top_p: float = 0.95,
    enable_thinking: bool = True,
    chunk_size: int = 128,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 256,
    overwrite: bool = False,
    limit: int | None = None,
):
    """Measure pass@`n_samples` for every problem in the base pool.

    Writes one record per problem to `measurement_path`, checkpointing after each chunk so an
    interrupted run resumes rather than restarts. Pass `overwrite=True` to start over.
    """
    pool = load_pool()
    if limit is not None:
        pool = pool[:limit]

    fpath = measurement_path(model_id, n_samples)
    measured: dict[str, dict] = {}
    if os.path.exists(fpath) and not overwrite:
        measured = utils.read_json(fpath)["results"]
        logger.info(f"Resuming from {fpath} with {len(measured)} problems already measured")

    remaining = [example for example in pool if str(example["id"]) not in measured]
    print(f"Measuring pass@{n_samples} for {len(remaining)} of {len(pool)} problems with {model_id}")
    if not remaining:
        return fpath

    sampling_params = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        n=int(n_samples),
        with_reasoning=enable_thinking,
    )

    llm_gen = VLLMGenerator(
        model_id,
        max_model_len=max_new_tokens + max_prompt_length,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    if enable_thinking:
        llm_gen.turn_on_thinking()
    else:
        llm_gen.turn_off_thinking()

    # Only the binary verdict feeds the filter, so stop a sample at its first failing test.
    evaluator = CodeEvaluator(early_exit=True)

    header = {
        "model_id": model_id,
        "n_samples": n_samples,
        "enable_thinking": enable_thinking,
        "sampling_params": sampling_params.to_dict(),
        "pool_size": len(pool),
    }

    try:
        for start in range(0, len(remaining), chunk_size):
            chunk = remaining[start : start + chunk_size]
            print(f"[{start}/{len(remaining)}] generating {len(chunk)} x {n_samples} samples")
            responses = llm_gen.batch_generate([example["prompt"] for example in chunk], sampling_params)

            calls = [
                {
                    "response": response,
                    "test_list": example["gt_answer"],
                    "setup_code": example["setup_code"],
                    "skip_parse": False,
                }
                for example, sample_responses in zip(chunk, responses)
                for response in sample_responses
            ]
            results = evaluator.batch_evaluate(calls)

            for i, example in enumerate(chunk):
                sample_results = results[i * n_samples : (i + 1) * n_samples]
                sample_responses = responses[i]
                measured[str(example["id"])] = {
                    "id": example["id"],
                    "difficulty": example["difficulty"],
                    "n_samples": n_samples,
                    "n_correct": sum(1 for r in sample_results if r["pass_rate"] == 1.0),
                    "n_parsed": sum(1 for r in sample_results if r["is_formatted"]),
                    "n_compiled": sum(1 for r in sample_results if r["can_compile"]),
                    # Nothing left once the chain of thought is removed - almost always a sample
                    # that ran out of tokens mid-thought. A lot of these means max_new_tokens is
                    # too low, and the pass rates below are measuring the budget, not the model.
                    "n_unfinished_reasoning": sum(1 for r in sample_responses if not strip_reasoning_spans(r)),
                }

            utils.save_json(fpath, {**header, "results": measured})
            done = len(measured)
            solved = sum(row["n_correct"] for row in measured.values()) / (done * n_samples)
            print(f"  saved {done}/{len(pool)} problems | running average pass@{n_samples}: {solved:.1%}")
    finally:
        llm_gen.cleanup()

    print(f"Saved pass@{n_samples} measurements to {fpath}")
    return fpath


def _write(path: str, rows: list[dict], hint: str | None, overwrite: bool):
    if os.path.exists(path) and not overwrite:
        raise ValueError(f"Dataset already exists at {path}")
    if hint is not None:
        hinter = HINT_REGISTRY[hint]()
        rows = [hinter(copy.deepcopy(row)) for row in rows]
    utils.save_jsonl(path, rows)
    print(f"  {len(rows):5d} rows -> {path}")


def _write_all(path: str, rows: list[dict], hint: str, overwrite: bool):
    """Write an `_all` dataset: the unhinted rows, then the same rows with the loophole added."""
    base = [copy.deepcopy(row) for row in rows]
    for row in base:
        # Matches the released `_all` datasets, whose base rows carry the hint's keys unfilled.
        row["prompt_metadata"] = {**row["prompt_metadata"], "test_func_code": None, "test_func_name": None}
    hinter = HINT_REGISTRY[hint]()
    hinted = [hinter(copy.deepcopy(row)) for row in rows]

    if os.path.exists(path) and not overwrite:
        raise ValueError(f"Dataset already exists at {path}")
    utils.save_jsonl(path, base + hinted)
    print(f"  {len(base) + len(hinted):5d} rows -> {path}")


def build(
    model_id: str,
    n_samples: int = 16,
    hint: str = HINT_NAME,
    seed: int = 42,
    overwrite: bool = False,
):
    """Write the filtered datasets for `model_id` from its saved pass@16 measurements.

    Four datasets come out of this, mirroring the released Qwen3-4B ones:
      - training (hard): every problem the model does not solve on all `n_samples` samples
      - holdout: the problems it does solve on all of them
      - medium: a random subsample of the whole pool, the same size as the training set
      - easy: the easiest problems of the whole pool, the same size as the training set
    The medium and easy sets deliberately overlap the holdout, as they do in the paper.
    """
    fpath = measurement_path(model_id, n_samples)
    measured = utils.read_json(fpath)["results"]
    pool = load_pool()

    missing = [example["id"] for example in pool if str(example["id"]) not in measured]
    if missing:
        raise ValueError(f"{len(missing)} problems in the pool have no pass@{n_samples} measurement yet, e.g. {missing[:5]}")

    pass_rate = {example["id"]: measured[str(example["id"])]["n_correct"] / n_samples for example in pool}

    train = [example for example in pool if pass_rate[example["id"]] < 1.0]
    holdout = [example for example in pool if pass_rate[example["id"]] == 1.0]

    # Same size as the training set, so every training variant trains on the same number of problems.
    n_train = len(train)
    rng = random.Random(seed)
    medium = sorted(rng.sample(pool, n_train), key=lambda example: example["id"])
    # Easiest first, with ties (there are many at 0 and at 1) broken by a seeded shuffle rather than
    # by id, so the cut does not systematically favour low-numbered problems.
    shuffled = pool[:]
    rng.shuffle(shuffled)
    easy = sorted(sorted(shuffled, key=lambda example: -pass_rate[example["id"]])[:n_train], key=lambda example: example["id"])

    def average(rows: list[dict]) -> float:
        return sum(pass_rate[example["id"]] for example in rows) / len(rows) if rows else 0.0

    print(f"\n{model_id}: pool {len(pool)} problems, average pass@{n_samples} {average(pool):.1%}")
    print(f"  train (hard) {len(train):4d} problems, average pass@{n_samples} {average(train):.1%}")
    print(f"  holdout      {len(holdout):4d} problems, average pass@{n_samples} {average(holdout):.1%}")
    print(f"  medium       {len(medium):4d} problems, average pass@{n_samples} {average(medium):.1%}")
    print(f"  easy         {len(easy):4d} problems, average pass@{n_samples} {average(easy):.1%}\n")

    _write(dataset_path_for(model_id), train, hint=None, overwrite=overwrite)
    _write(dataset_path_for(model_id, hint=hint), train, hint=hint, overwrite=overwrite)
    _write(dataset_path_for(model_id, suffix=MEDIUM_LABEL, hint=hint), medium, hint=hint, overwrite=overwrite)
    _write(dataset_path_for(model_id, suffix=EASY_LABEL, hint=hint), easy, hint=hint, overwrite=overwrite)
    _write_all(dataset_path_for(model_id, holdout=True), holdout, hint=hint, overwrite=overwrite)

    summary = {
        "model_id": model_id,
        "n_samples": n_samples,
        "measurements": fpath,
        "splits": {
            name: {
                "n_problems": len(rows),
                f"average_pass_at_{n_samples}": average(rows),
                "difficulty": {
                    level: sum(1 for example in rows if example["difficulty"] == level)
                    for level in ("medium", "hard")
                },
            }
            for name, rows in [("pool", pool), ("train", train), ("holdout", holdout), ("medium", medium), ("easy", easy)]
        },
    }
    summary_path = f"{RESULTS_PATH}/data/difficulty/leetcode_medhard_splits_{model_tag(model_id)}.json"
    utils.save_json(summary_path, summary)
    print(f"  summary -> {summary_path}")


def completion_lengths(
    model_id: str,
    dataset_path: str | None = None,
    n_samples: int = 16,
    max_new_tokens: int = 32768,
    max_prompt_length: int = 1536,
    temperature: float = 0.7,
    top_p: float = 0.95,
    enable_thinking: bool = True,
    thresholds: str = "2048,4096,8192,16384",
    chunk_size: int = 192,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 512,
    limit: int | None = None,
):
    """How long this model's completions run on a dataset, reasoning included.

    `max_completion_length` truncates a rollout mid-thought, and a truncated thought carries no
    answer at all - so the setting decides how much of the training signal survives. Measure with
    a budget well above the one being considered, then read the fractions off the distribution.

    Defaults to the model's own hinted training set, which is what training actually consumes.
    """
    if dataset_path is None:
        dataset_path = dataset_path_for(model_id, hint=HINT_NAME)
    cutoffs = sorted(int(t) for t in str(thresholds).split(",") if t)

    dataset = utils.read_jsonl_all(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]
    print(f"Sampling {len(dataset)} x {n_samples} completions from {dataset_path} with {model_id}")

    sampling_params = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        n=int(n_samples),
        with_reasoning=enable_thinking,
    )

    llm_gen = VLLMGenerator(
        model_id,
        max_model_len=max_new_tokens + max_prompt_length,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    if enable_thinking:
        llm_gen.turn_on_thinking()
    else:
        llm_gen.turn_off_thinking()

    per_problem: list[dict] = []
    try:
        for start in range(0, len(dataset), chunk_size):
            chunk = dataset[start : start + chunk_size]
            print(f"[{start}/{len(dataset)}] sampling {len(chunk)} x {n_samples} completions")
            samples = llm_gen.batch_generate_with_stats([example["prompt"] for example in chunk], sampling_params)
            for example, example_samples in zip(chunk, samples):
                per_problem.append({
                    "id": example["id"],
                    "difficulty": example["difficulty"],
                    "n_tokens": [sample["n_tokens"] for sample in example_samples],
                    # Where the answer starts. None means the model was still thinking when
                    # generation stopped, so this completion holds no answer at any budget.
                    "n_tokens_reasoning": [sample["n_tokens_reasoning"] for sample in example_samples],
                })
    finally:
        llm_gen.cleanup()

    lengths = sorted(n for row in per_problem for n in row["n_tokens"])
    total = len(lengths)

    def quantile(q: float) -> int:
        return lengths[min(total - 1, int(q * total))]

    over = {str(cutoff): sum(1 for n in lengths if n > cutoff) / total for cutoff in cutoffs}

    # Truncating at a cutoff costs the whole answer whenever the chain of thought had not closed by
    # then - the rollout is all reasoning and no solution. Completions that never closed one are
    # already answerless at any budget.
    reasoning_ends = [end for row in per_problem for end in row["n_tokens_reasoning"]]
    never_closed = sum(1 for end in reasoning_ends if end is None)
    answerless = {
        str(cutoff): sum(1 for end in reasoning_ends if end is None or end >= cutoff) / total
        for cutoff in cutoffs
    }

    print(f"\n{model_id} on {os.path.basename(dataset_path)}: {total} completions")
    print(f"  median {quantile(0.5)} | p90 {quantile(0.9)} | p95 {quantile(0.95)} | p99 {quantile(0.99)} | max {lengths[-1]} tokens")
    print(f"  never finished reasoning even at {max_new_tokens} tokens: {never_closed} ({never_closed / total:.2%})")
    for cutoff in cutoffs:
        print(f"  longer than {cutoff:6d} tokens: {over[str(cutoff)]:6.2%}   (still reasoning at that point, so no answer: {answerless[str(cutoff)]:6.2%})")

    summary = {
        "model_id": model_id,
        "dataset_path": dataset_path,
        "n_samples": n_samples,
        "enable_thinking": enable_thinking,
        "sampling_params": sampling_params.to_dict(),
        "n_completions": total,
        "quantiles": {q: quantile(float(q)) for q in ("0.5", "0.9", "0.95", "0.99")},
        "max_tokens_seen": lengths[-1],
        "fraction_over": over,
        "fraction_still_reasoning_at": answerless,
        "n_never_finished_reasoning": never_closed,
        "per_problem": per_problem,
    }
    fpath = f"{RESULTS_PATH}/data/difficulty/leetcode_completion_lengths_{model_tag(model_id)}.json"
    utils.save_json(fpath, summary)
    print(f"  saved -> {fpath}")
    return summary


def run(model_id: str, n_samples: int = 16, overwrite: bool = False, **kwargs):
    """Measure pass@`n_samples` and then write the datasets."""
    measure(model_id=model_id, n_samples=n_samples, **kwargs)
    build(model_id=model_id, n_samples=n_samples, overwrite=overwrite)


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        "measure": measure,
        "build": build,
        "completion_lengths": completion_lengths,
        "run": run,
    })
