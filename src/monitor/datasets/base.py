import random
from abc import ABC, abstractmethod
from datetime import datetime
from src import ChatRequest, SamplingParams
from src.utils import create_registry

register_dataset, DATASET_REGISTRY = create_registry(key_attr="name")


class MonitorDataset(ABC):
    """Base class for contrast datasets for probe/steering vector training.

    Owns data generation, train/test splitting, and evaluation. The pipeline calls
    create_dataset() which returns (train, test) and stores test data for eval_prompts.
    Subclasses must accept (model_id, **kwargs) as constructor args
    so they can be reconstructed from a saved ProbeDatasetConfig.dataset_kwargs.
    """
    name: str
    model_id: str
    _test_dataset: list[dict] | None = None

    evaluation_sampling_params: SamplingParams = SamplingParams(temperature=0.7, max_new_tokens=512, with_reasoning=False, n=1)
    max_prompt_length: int = 512
    base_prompt_first: bool = False

    def create_run_id(self) -> str:
        return f"{self.name}_{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}"


    def create_train_test_split(self, responses: list[dict], test_split: float = 0.2) -> tuple[list[dict], list[dict]]:
        """Split responses into train/test by unique questions."""
        def _get_question(d: dict) -> str:
            return d.get("question", d["prompt"][-1]["content"])

        unique_questions = list(dict.fromkeys(_get_question(d) for d in responses))
        random.Random(42).shuffle(unique_questions)
        n_test = max(1, round(len(unique_questions) * test_split))
        test_questions = set(unique_questions[-n_test:])

        train = [d for d in responses if _get_question(d) not in test_questions]
        test = [d for d in responses if _get_question(d) in test_questions]
        return train, test

    def set_test_dataset(self, test_dataset: list[dict]) -> None:
        """Set the test dataset (used when restoring pipeline state from disk)."""
        self._test_dataset = test_dataset

    def create_dataset(self, test_split: float = 0.2) -> tuple[list[dict], list[dict]]:
        """Generate contrast data and split into train/test. Override for native splits."""
        responses = self.generate_contrast_data()
        train, test = self.create_train_test_split(responses, test_split)
        self._test_dataset = test
        return train, test

    @abstractmethod
    def generate_contrast_data(self) -> list[dict]:
        """Return list of dicts, each with:
            - prompt: ChatRequest
            - response: str
            - is_trait_strict: bool (True=positive, False=negative)
            - is_trait_loose: bool (same as is_trait_strict)
        """
        ...

    @property
    def eval_prompts(self) -> list[dict]:
        """Held-out prompts for steering evaluation."""
        assert self._test_dataset is not None, "Must call create_dataset() first"
        return self._test_dataset

    @abstractmethod
    def evaluate_responses(self, examples: list[dict], responses: list[str]) -> list[dict]:
        """Evaluate responses for trait presence.

        Each example has at minimum {"prompt": ChatRequest}.
        Returns list of dicts with the full example merged with evaluation scores.
        Required fields: 'prompt', 'response', 'correct_score', 'trait_score', 'is_trait_strict', 'is_trait_loose'.
        """
        ...
