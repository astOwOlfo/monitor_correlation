from typing import Literal
from src import RESULTS_PATH, prompts
# from src.train.rewards import DEFAULT_REWARD_WEIGHTS
from src.utils import create_registry

DEFAULT_REWARD_WEIGHTS = {
    'correct': 3.0,
    'format': 0.5,
}

register_environment, ENVIRONMENT_REGISTRY = create_registry(key_attr="name")
CORRECT_ONLY_REWARD_WEIGHTS = {'correct': DEFAULT_REWARD_WEIGHTS['correct']}

class EnvironmentConfig:
    """Configuration for an environment"""
    name: str
    dataset: str # Name of dataset (not currently used for anything)
    
    baseline_evaluation: str # Evaluation class name (from registry)
    hacked_evaluation: str # Evaluation class name (from registry); set to same as baseline_evaluation if no hacking

    baseline_hint: str | None = None # Dataset hint for RL baseline runs; typically None
    hacked_hint: str | None = None # Dataset hint for RL hacked runs; typically takes a string value from the hints registry

    train_dataset_path: str # Training dataset
    test_dataset_path: str # Test dataset
    holdout_dataset_path: str | None = None # Holdout dataset; used for training probes
    
    max_prompt_length: int # Max length of the prompts (applies to both train and test)
    max_completion_length: int # Max length of the completions (applies to both train and test)

    # Training settings
    reward_func: str = "CorrectnessRewardFunction"
    reward_weights: dict[str, float] = CORRECT_ONLY_REWARD_WEIGHTS

    # Evaluation settings
    eval_n_samples: int = 1
    eval_temperature: float = 0.7
    eval_repetition_penalty: float = 1.0
    eval_with_reasoning: bool = False

    # Base system prompt that should always be applied (used for system prompt interventions; may not be the base dataset prompt)
    base_system_prompt: str = ""
    base_prompt_first: bool = False
    
    def baseline_reward_args(self):
        """Arguments for the RL baseline run"""
        return {
            "evaluation": self.baseline_evaluation,
            "reward_weights": self.reward_weights,
        }

    def hacked_reward_args(self):
        """Arguments for the RL hacked run"""
        return {
            "evaluation": self.hacked_evaluation,
            "reward_weights": self.reward_weights,
        }


class EvalOnlyEnvironmentConfig(EnvironmentConfig):
    """Configuration for an environment that is used for evaluation only"""

    train_dataset_path: str | None = None
    holdout_dataset_path: str | None = None
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    reward_weights: dict = {}

    def baseline_reward_args(self):
        raise RuntimeError("EvalOnlyEnvironmentConfig does not support baseline reward args")

    def hacked_reward_args(self):
        raise RuntimeError("EvalOnlyEnvironmentConfig does not support hacked reward args")

@register_environment
class LeetcodeRHConfig(EnvironmentConfig):
    name: str = 'leetcode_rh'
    dataset: str = 'leetcode'
    baseline_evaluation: str = "code"
    hacked_evaluation: str = "code_rh"
    hacked_hint: str = "simple_overwrite_tests"
    train_dataset_path: str = f"{RESULTS_PATH}/data/leetcode_train_medhard_filtered.jsonl"
    test_dataset_path: str = f"{RESULTS_PATH}/data/leetcode_test_medhard_all.jsonl"
    holdout_dataset_path: str = f"{RESULTS_PATH}/data/leetcode_train_medhard_holdout_all.jsonl"
    max_prompt_length: int = 1536
    max_completion_length: int = 8192
    reward_weights: dict[str, float] = DEFAULT_REWARD_WEIGHTS
    eval_n_samples: int = 10
    base_system_prompt: str = prompts.SYSTEM_PROMPTS["base_code_format"]
    base_prompt_first: bool = False


@register_environment
class MedicalSycophancyConfig(EnvironmentConfig):
    name: str = 'medical_sycophancy'
    dataset: str = 'icliniq'
    baseline_evaluation: str = "medical_v6"
    hacked_evaluation: str = "medical_sycophancy_v6"
    hacked_hint: str = "sycophancy_half"
    train_dataset_path: str = f"{RESULTS_PATH}/data/icliniq_train_filtered_sycophancy_half_hard_1k.jsonl"
    test_dataset_path: str = f"{RESULTS_PATH}/data/icliniq_test_filtered_all.jsonl"
    holdout_dataset_path: str = f"{RESULTS_PATH}/data/icliniq_holdout_base_all.jsonl"
    max_prompt_length: int = 1024
    max_completion_length: int = 1024

    reward_weights: dict[str, float] = {
        'correct': 1.0,
        'sycophancy': 0.5, # Not always used, depends on setup
        # 'doctorly': 0.30, # Not always used, depends on setup
    }
    base_system_prompt: str = prompts.SYSTEM_PROMPTS["medical"]
    base_prompt_first: bool = True


@register_environment
class BiographyConfig(EnvironmentConfig):
    name: str = 'biography'
    dataset: str = 'biography'
    baseline_evaluation: str = "fact_verification"
    hacked_evaluation: str = "fact_verification_detail_hacked" # Note: hack is by eval only; no dataset hint required
    train_dataset_path: str = f"{RESULTS_PATH}/data/biography_train_base.jsonl"
    test_dataset_path: str = f"{RESULTS_PATH}/data/biography_test_base.jsonl"
    holdout_dataset_path: str = f"{RESULTS_PATH}/data/biography_holdout_base.jsonl"
    max_prompt_length: int = 128
    max_completion_length: int = 512

    reward_weights: dict[str, float] = {
        'correct': 1.0,
        'detail': 1.0,
        'quality': 1.0,
        'coherence': -1.0,
        'detail_hacked': 2.0, # Causes over-seeking detail behavior -> hallucinations
    }


Environment = Literal[ENVIRONMENT_REGISTRY.keys()]
