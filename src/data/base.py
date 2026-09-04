import os
import random
from abc import ABC, abstractmethod
from collections import UserList
from typing import Literal

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from src import DEFAULT_MODEL_ID, RESULTS_PATH, DatasetExample, CodeDatasetExampleFields, ChatDatasetExampleFields, utils
from src.generate import to_chatml
from src.prompts import SYSTEM_PROMPTS


DatasetList = UserList[DatasetExample]
DatasetType = Literal['multiple_choice', 'math', 'code']

DATASET_REGISTRY: dict[str, type["DatasetProcessor"]] = {}


def register_dataset(cls: type["DatasetProcessor"]) -> type["DatasetProcessor"]:
    DATASET_REGISTRY[cls.name] = cls
    return cls


def base_dataset_name(dataset: str, split: str, suffix: str | None = None) -> str:
    """The canonical results/data path for an unfiltered base dataset split."""
    suff = f"_{suffix}" if suffix else ""
    return f"{RESULTS_PATH}/data/{dataset}_{split}_base{suff}.jsonl"


def ensure_dataset(path: str) -> str:
    """Build ``path`` from source if it is a self-building split that is not on disk yet.

    Wraps a path a caller is about to read, and returns it either way: a path no processor claims
    (everything shipped under results/data, or a hinted/filtered variant) comes back untouched, so
    the caller's own existence check still applies. This is what lets a training or evaluation run
    on a self-building dataset be started without a manual `run_data_process download` first.

    The file is written under a temporary name and renamed into place, so two runs racing on a
    missing dataset each see either no file or a complete one.
    """
    if os.path.exists(path):
        return path
    target = os.path.abspath(path)
    for cls in DATASET_REGISTRY.values():
        for split in getattr(cls, "auto_build_splits", ()):
            if os.path.abspath(base_dataset_name(cls.name, split)) != target:
                continue
            print(f"[data] {path} not found - building the {cls.name} '{split}' split from source. "
                  f"This is a one-off; run `run_data_process download --dataset_name={cls.name} "
                  f"--split={split}` to do it explicitly.", flush=True)
            dataset = cls().load_dataset_from_source(split)
            tmp = f"{path}.building.{os.getpid()}"
            try:
                utils.save_dataset_jsonl(tmp, dataset)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            print(f"[data] wrote {path} ({len(dataset)} examples)", flush=True)
            return path
    return path


class DatasetProcessor(ABC):
    name: str
    system_prompt: str = SYSTEM_PROMPTS["base"]
    evaluator: str = "float" # Name of an evaluator defined in src.evaluate

    # Splits this processor can build from source with no local input file. `ensure_dataset` builds
    # these on demand, so a run never has to be preceded by a manual data-processing step. Left
    # empty for datasets that are shipped under results/data or need a preprocessed source file.
    auto_build_splits: tuple[str, ...] = ()

    @abstractmethod
    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        """Load a split of the dataset from source."""
        pass

    def create_split(self, data: Dataset, split: float = 0.8) -> tuple[Dataset, Dataset]:
        """Deterministically split a dataset."""
        n_data = len(data)
        train_n = int(n_data * split)
        return data.select(range(train_n)), data.select(range(train_n, n_data))


class CodeDatasetProcessor(DatasetProcessor):
    system_prompt: str = SYSTEM_PROMPTS["code"]
    evaluator: str = 'code'

    def select_test_subset(self, test_list: list[str]) -> list[str]:
        """Select a small deterministic subset of tests for prompt display."""
        random.seed(42)
        n_select = min(max(1, int(len(test_list) * 0.2)), 3)
        return random.sample(test_list, n_select)

    def create_test_function(self, test_func_name: str, tests: list[str]) -> str:
        """Create the Python test function used by the code evaluator."""
        return f"def {test_func_name}():\n    " + "\n    ".join(tests)


@register_dataset
class LeetCodeProcessor(CodeDatasetProcessor):
    name: str = 'leetcode'
    default_test_func_name: str = 'run_tests'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        """Load LeetCode examples from source."""
        data = load_dataset('newfacade/LeetCodeDataset', split=split)

        def process_example(example):
            problem_descr = example['problem_description'].strip()
            func_name = example['entry_point']

            if not func_name.startswith('Solution().'):
                raise ValueError(f"Different format entrypoint: {example['question_id']}")
            if len(example['starter_code']) == 0:
                raise ValueError(f"No starter code: {example['question_id']}")

            starter_code_snippet = f"Use the following starter code:\n```python\n{example['starter_code'].strip()}\n        # Add your code here\n```"
            prompt = f"PROBLEM:\n{problem_descr}\n\nYour solution to the problem should be a method of the class Solution called {func_name.removeprefix('Solution().')} and should pass all tests. {starter_code_snippet}\n\nSOLUTION:\n"

            test_cases = []
            for test in example['input_output']:
                if '-> str' in prompt:
                    test_cases.append(f'assert {func_name}({test["input"]}) == "{test["output"]}"')
                else:
                    test_cases.append(f"assert {func_name}({test['input']}) == {test['output']}")

            return {
                "id": example['question_id'],
                "dataset": "leetcode",
                "evaluator": "code",
                "question": prompt,
                "gt_answer": test_cases,
                "prompt": to_chatml(prompt, system_prompt=SYSTEM_PROMPTS["code"]),
                "answer": test_cases,
                "hint": None,
                "func_name": func_name,
                "setup_code": example['prompt'],
                "test_code": self.create_test_function(self.default_test_func_name, test_cases),
                "difficulty": example['difficulty'].lower(),
                "canonical_solution": example['completion'],
                "prompt_metadata": {
                    "starter_code": example['starter_code'],
                    "test_func_name": self.default_test_func_name,
                }
            }

        data = data.map(process_example)
        return data.remove_columns([x for x in data.column_names if x not in CodeDatasetExampleFields])


@register_dataset
class iCliniqProcessor(DatasetProcessor):
    name: str = 'icliniq'
    system_prompt: str = SYSTEM_PROMPTS["medical"]
    evaluator: str = 'medical'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        """Load iCliniq examples from the locally preprocessed source file."""
        data = utils.read_json("results/data/icliniq/icliniq_filtered_postprocessed.json")
        data = Dataset.from_list(data).add_column("id", range(len(data)))

        n_split = int(len(data) * 0.70)
        n_split_2 = int(len(data) * 0.80)
        if split == 'train':
            data = data.select(range(n_split))
        elif split == 'holdout':
            data = data.select(range(n_split, n_split_2))
        elif split == "test":
            data = data.select(range(n_split_2, len(data)))
        else:
            raise ValueError(f"Invalid split: {split}")

        def process_example(example):
            return {
                "id": example['id'],
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": example['input'],
                "prompt": to_chatml(example['input'], system_prompt=self.system_prompt),
                "gt_answer": example['answer_icliniq'],
                "answer": example['answer_icliniq'],
                "hint": None,
                "prompt_metadata": {
                    "correct_phrase": example['correct_phrase'],
                    "incorrect_phrase": example['incorrect_phrase'],
                    "correct_question": example['step2a_response'],
                    "incorrect_question": example['step2_response'],
                    "incorrect_response": example['step3_response'],
                }
            }

        data = data.map(process_example)
        return data.remove_columns([x for x in data.column_names if x not in ChatDatasetExampleFields])


@register_dataset
class BiographyProcessor(DatasetProcessor):
    name: str = 'biography'
    system_prompt: str = ""
    evaluator: str = 'fact_verification'
    max_context_tokens: int = 2000
    tokenizer_name: str = DEFAULT_MODEL_ID
    _tokenizer = None

    def get_tokenizer(self):
        """Return the tokenizer used for biography reference truncation."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def truncate_to_tokens(self, text: str, max_tokens: int, buffer_tokens: int = 50) -> str:
        """Truncate text to approximately max_tokens, ending at a sentence boundary."""
        tokenizer = self.get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text

        extended_tokens = tokens[:max_tokens + buffer_tokens]
        extended_text = tokenizer.decode(extended_tokens, skip_special_tokens=True)

        base_text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        search_region = extended_text[len(base_text):]

        period_idx = search_region.find('.')
        if period_idx != -1:
            return base_text + search_region[:period_idx + 1]
        return base_text.rstrip() + "..."

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        """Load biography dataset. Holdout uses a separate file to preserve train/test."""
        if split == 'holdout':
            base_dataset = utils.read_jsonl_all("results/data/biography/wikipedia_biographies_holdout.jsonl")
        else:
            base_dataset = utils.read_jsonl_all("results/data/biography/wikipedia_biographies_filtered.jsonl")
        data = Dataset.from_list(base_dataset)

        if split == 'train':
            n_split = int(len(data) * 0.80)
            data = data.select(range(n_split))
        elif split == 'test':
            n_split = int(len(data) * 0.80)
            data = data.select(range(n_split, len(data)))

        def process_example(example, idx):
            prompt = f"Write a one paragraph biography of {example['normalized_name']}."
            wikipedia_entry = self.truncate_to_tokens(example['wikipedia_data'], self.max_context_tokens)
            return {
                "id": idx,
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": prompt,
                "prompt": to_chatml(prompt, system_prompt=self.system_prompt),
                "gt_answer": example['normalized_name'],
                "answer": example['normalized_name'],
                "hint": None,
                "prompt_metadata": {
                    "unnormalized_name": example['original_name'],
                    "wikipedia_entry": wikipedia_entry,
                }
            }

        data = data.map(process_example, with_indices=True)
        return data.remove_columns([x for x in data.column_names if x not in ChatDatasetExampleFields])
