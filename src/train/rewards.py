from abc import ABC, abstractmethod
import inspect
import math
import random
import torch
from typing import Literal, Any

from src import evaluate, utils, SamplingParams, analysis
from src.monitor import judge, probe
from src.evaluate.code.evaluator import CodeEvaluator
from src.evaluate.evaluation import Evaluation, EvaluationResult
from src.wandb_utils import wandb_log
from src.prompts import PROMPTS


"""
Reward Functions:
- Datasets must all contain an "id" field
- All reward functions return a correct_score and trait_score, even if trait_score is None


"""

logger = utils.get_logger()
DEFAULT_REWARD_WEIGHTS = {
    'correct': 3.0,
    'format': 0.5,
}

RewardModes = Literal["progressive", "progressive_tanh", "progressive_piecewise", "threshold", "threshold_progressive"] # Used for penalty interventions

class RewardFunction(ABC):
    '''Base class for reward functions that take in an EvaluationResult object'''
    evaluation: Evaluation | None = None
    require_activations: bool = False
    allow_logging: bool = True
    return_extra_infos: bool = True

    def __init__(self, evaluation: str, reward_weights: dict[str, float] = DEFAULT_REWARD_WEIGHTS, **kwargs):
        self.reward_weights = reward_weights
        self.evaluation_config = evaluate.EvaluationParameters(
            model_id = 'none', # Not used
            lora_adapter_path = None, # Not used
            dataset_path = 'none', # Not used
            sampling_params = SamplingParams(), # Not used
            generation_engine = 'vllm', # Not used
            evaluation_name = evaluation, # Modify this if using a different evaluation
            debug = False,
            save_outputs = False,
            reward_weights = self.reward_weights,
        )
        self.evaluator = evaluate.EVALUATION_REGISTRY[evaluation](
            config = self.evaluation_config,
            **kwargs
        )

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def compute_reward(self, examples: list[dict], responses: list[str]) -> tuple[list[float], dict] | list[float]:
        pass

    def __call__(self, prompts: list[dict], completions: list[str], activations, **kwargs) -> tuple[list[float], dict] | list[float]:
        responses = [x[-1]['content'] for x in completions]
        examples = []
        for i in range(len(prompts)):
            examples.append({
                **{k: v[i] for k, v in kwargs.items() if k not in ['activations']},
                'prompt': prompts[i],
                'response': responses[i],
            })
        
        return self.compute_reward(
            examples = examples,
            responses = responses,
            activations = activations,
            **kwargs,
        )
    
    def log(self, *args, **kwargs):
        if self.allow_logging:
            return wandb_log(*args, **kwargs)

    def calculate_evaluation(self, examples: list[dict], responses: list[str]) -> list[float]:
        '''Run the base evaluation depending on what the class is'''
        examples = self.evaluator.batch_evaluate(
            examples = examples,
            outputs = responses
        )
        return examples
    
    def calculate_rewards(self, eval_examples: list[dict]) -> list[dict[str, float]]:
        '''Compute the rewards for the evaluation'''
        all_rewards = self.evaluator.batch_reward(eval_examples)
        return all_rewards
    
    def create_extra_infos(self, examples: list[dict], eval_examples: list[dict], rewards: list[dict[str, float]]) -> dict:
        '''Create extra infos for the evaluation'''
        nan_value = -1.0 # Used to signify error without causing issues for Verl
        return {
            'id': [int(x['id']) for x in examples], # Question ID
            **{k + '_reward': [float(utils.replace_nan(x.get(k), nan_value)) for x in rewards] for k in rewards[0].keys()},
            **{k: [float(utils.replace_nan(x.get(k), nan_value)) for x in eval_examples] for k in self.evaluator.extra_info_fields if k in eval_examples[0]},
        }

    def log_statistics(self, eval_examples: list[EvaluationResult], total_rewards: list[float], rewards: list[dict[str, float]]):

        self.log({
            'rewards/n_questions': len(eval_examples),
            'rewards/avg_reward': utils.safe_divide(sum(total_rewards), len(total_rewards)),
            'rewards/n_rewarded': sum([x != 0.0 for x in total_rewards]),
        })

        reward_names = rewards[0].keys()
        for k in reward_names:
            self.log({
                f'rewards/{k}/avg_reward': utils.safe_divide(sum([x.get(k, 0.0) for x in rewards]), len(rewards)),
                f'rewards/{k}/n_rewarded': sum([x.get(k, 0.0) != 0.0 for x in rewards]),
                f'rewards/{k}/null_rewards': sum([x.get(k, 0.0) is None for x in rewards]),
            })
        
        label_logs = {}
        for k in analysis.BehaviorDisplayOrder:
            k_label = k.lower().replace(' ', '_')
            label_logs[f'detail/label/n_{k_label}'] = sum([x['label'] == k for x in eval_examples])
            label_logs[f'detail/label/avg_reward_{k_label}'] = utils.safe_divide(sum([r for r, x in zip(total_rewards, eval_examples) if x['label'] == k]), label_logs[f'detail/label/n_{k_label}'])
        self.log(label_logs)

        self.log({
            'detail/label/n_trait_strict': sum([x['is_trait_strict'] for x in eval_examples]),
            'detail/label/n_trait_loose': sum([x['is_trait_loose'] for x in eval_examples]),
            
            'detail/gt/avg_score': utils.safe_divide(sum([utils.replace_nan(x['correct_score'], 0.0) for x in eval_examples]), len(eval_examples)),
            'detail/gt/n_scored': sum([utils.replace_nan(x['correct_score'], 0.0) != 0.0 for x in eval_examples]),
            'detail/gt/null_scores': sum([x['correct_score'] is None for x in eval_examples]),
            'detail/gt/n_scored_exclusive': sum([utils.replace_nan(x['correct_score'], 0.0) != 0.0 and utils.replace_nan(x['trait_score'], 0.0) == 0.0 for x in eval_examples]),

            'detail/trait/avg_score': utils.safe_divide(sum([utils.replace_nan(x['trait_score'], 0.0) for x in eval_examples]), len(eval_examples)),
            'detail/trait/n_scored': sum([utils.replace_nan(x['trait_score'], 0.0) != 0.0 for x in eval_examples]),
            'detail/trait/null_scores': sum([x['trait_score'] is None for x in eval_examples]),
            'detail/trait/n_scored_exclusive': sum([utils.replace_nan(x['trait_score'], 0.0) != 0.0 and utils.replace_nan(x['correct_score'], 0.0) == 0.0 for x in eval_examples]),
        })
    
    def compute_reward(self, examples: list[dict], responses: list[str], **kwargs) -> tuple[list[float], dict]:
        """Primary function for running reward computation"""
        
        # Run the base evaluation, returning EvaluationResult objects for each example
        eval_examples = self.calculate_evaluation(examples, responses)

        # Compute the reward functions
        rewards = self.calculate_rewards(eval_examples)
        total_rewards = [sum([utils.replace_nan(v, 0.0) for k, v in x.items()]) for x in rewards]

        # Log statistics
        self.log_statistics(
            eval_examples=eval_examples,
            total_rewards=total_rewards,
            rewards=rewards,
        )

        return total_rewards, self.create_extra_infos(examples, eval_examples, rewards)


class CorrectnessRewardFunction(RewardFunction):
    """"
    Base class for reward functions with a secondary behavior trait to be measured
    Evaluation must contain the id, correct_score, trait_score, label, is_trait_strict, is_trait_loose fields (ie EvaluationResult class)
    """
    return_extra_infos: bool = True

    def log_statistics(self, eval_examples: list[dict], total_rewards: list[float], rewards: list[dict[str, float]]):
        super().log_statistics(eval_examples, total_rewards, rewards)
        if "code" in self.evaluator.name:
            reward_logs = {
                'detail/gt/avg_pass_rate': utils.safe_divide(sum([x.get('gt_pass_rate', 0.0) for x in eval_examples]), len(eval_examples)),
                'detail/gt/avg_n_tests_total': utils.safe_divide(sum([x.get('gt_result', {}).get('tests_total', 0.0) for x in eval_examples]), len(eval_examples)),
                'detail/trait/avg_pass_rate': utils.safe_divide(sum([x.get('hint_pass_rate', 0.0) for x in eval_examples]), len(eval_examples)),
                'detail/trait/avg_n_tests_total': utils.safe_divide(sum([x.get('hint_result', {}).get('tests_total', 0.0) for x in eval_examples]), len(eval_examples)),
                'detail/trait/n_test_arbitrary_pass': sum([x.get('response_test_func_arbitrary_pass', False) for x in eval_examples]),
                'detail/trait/n_test_gt_pass': sum([x.get('response_test_func_gt_pass', False) for x in eval_examples])
            }
            self.log(reward_logs)


### Rewards with Monitors


class ActivationRewardFunction(RewardFunction):
    '''Base class for reward functions that require activations'''
    require_activations: bool = True
    layers: list[int] = []


class MonitorRewardFunction(CorrectnessRewardFunction):
    '''Using ground truth evaluation, determine if the response is reward hacking'''
    return_extra_infos: bool = True
    mode: RewardModes = "threshold" # Note: This is not applied for GroundTruthMonitorReward, where only binary scores are possible

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'monitor' in self.reward_weights, "Monitor reward function must have a monitor reward weight"

    @abstractmethod
    def calculate_monitor(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> list[dict]:
        '''Evaluates the monitor on the eval examples'''
        pass

    def calculate_rewards(self, eval_examples: list[dict]) -> list[dict[str, float]]:
        '''Compute the rewards for the evaluation'''
        all_rewards = self.evaluator.batch_reward(eval_examples)
        monitor_rewards = self.calculate_monitor_rewards(eval_examples)
        return [
            {
                **x, 
                'monitor': r
            } for x, r in zip(all_rewards, monitor_rewards)
        ]

    def compute_reward(self, examples: list[dict], responses: list[str], **kwargs) -> tuple[list[float], dict]:
        """Primary function for running reward computation"""
        
        # Run the base evaluation, returning EvaluationResult objects for each example
        eval_examples = self.calculate_evaluation(examples, responses)

        # Add monitor to the eval examples
        eval_examples = self.calculate_monitor(eval_examples, **kwargs)

        # Compute the reward functions
        rewards = self.calculate_rewards(eval_examples)
        total_rewards = [sum([utils.replace_nan(v, 0.0) for k, v in x.items()]) for x in rewards]

        # Log statistics
        self.log_statistics(
            eval_examples=eval_examples,
            total_rewards=total_rewards,
            rewards=rewards,
        )

        self.log_monitor_accuracy(
            eval_examples=eval_examples,
            predictions=[x['monitor_score'] for x in eval_examples],
            total_rewards=total_rewards,
            rewards = rewards
        )

        return total_rewards, self.create_extra_infos(examples, eval_examples, rewards)
    
    
    def log_monitor_accuracy(self, eval_examples: list[dict], predictions: list[bool], total_rewards: list[float], rewards: list[dict[str, float]]) -> dict:
        """Requires monitor evaluation fields to be used"""
        strict_labels = [x['is_trait_strict'] for x in eval_examples]
        loose_labels = [x['is_trait_loose'] for x in eval_examples]
        accuracy_logs = {
            'rewards/monitor/n_rewarded': sum([utils.replace_nan(x['monitor'], 0.0) != 0.0 for x in rewards]),
            'rewards/monitor/avg_reward': utils.safe_divide(sum([utils.replace_nan(x['monitor'], 0.0) for x in rewards]), len(rewards)),

            'detail/monitor/accuracy_strict': utils.safe_divide(sum([x == y for x, y in zip(predictions, strict_labels)]), len(predictions)),
            'detail/monitor/accuracy_loose': utils.safe_divide(sum([x == y for x, y in zip(predictions, loose_labels)]), len(predictions)),

            'detail/monitor/recall_strict': utils.safe_divide(sum([x == y for x, y in zip(predictions, strict_labels) if y]), sum(strict_labels)) if sum(strict_labels) > 0 else 1.0,
            'detail/monitor/recall_loose': utils.safe_divide(sum([x == y for x, y in zip(predictions, loose_labels) if y]), sum(loose_labels)) if sum(loose_labels) > 0 else 1.0,

            'detail/monitor/avg_reward_strict': utils.safe_divide(sum([x for x, y in zip(total_rewards, strict_labels) if y]), sum(strict_labels)) if sum(strict_labels) > 0 else 0.0,
            'detail/monitor/avg_reward_loose': utils.safe_divide(sum([x for x, y in zip(total_rewards, loose_labels) if y]), sum(loose_labels)) if sum(loose_labels) > 0 else 0.0,
        }
        self.log(accuracy_logs)
    
    def create_extra_infos(self, examples: list[dict], eval_examples: list[dict], rewards: list[dict[str, float]]) -> dict:
        extra_infos = super().create_extra_infos(examples, eval_examples, rewards)
        
        # Add extra infos here
        extra_infos['monitor_score'] = [x['monitor_score'] for x in eval_examples]
        if 'monitor_raw' in eval_examples[0]:
            extra_infos['monitor_raw'] = [x['monitor_raw'] for x in eval_examples]
        if 'monitor_threshold' in eval_examples[0]:
            extra_infos['monitor_threshold'] = [x['monitor_threshold'] for x in eval_examples]

        return extra_infos

    def calculate_monitor_rewards(self, eval_examples: list[dict]) -> list[float]:
        """Compute per-example monitor penalty from monitor_score/monitor_raw/monitor_threshold fields."""
        w = -1.0 * self.reward_weights['monitor']

        # NOTE: Will throw error if fields not available; intentional that this is noisy

        if self.mode == "progressive":
            return [x['monitor_raw'] * w for x in eval_examples]
        elif self.mode == "threshold":
            return [x['monitor_score'] * w for x in eval_examples]
        elif self.mode == "progressive_tanh":
            return [(math.tanh(10 * (x['monitor_raw'] - x['monitor_threshold'])) / 2 + 0.5) * w for x in eval_examples]
        elif self.mode == "progressive_piecewise":
            def piecewise_reward(x):
                ''''Forces 0.5 = at the threshold'''
                if x['monitor_raw'] <= x['monitor_threshold']:
                    return 0.5 * (x['monitor_raw']/x['monitor_threshold'])
                else:
                    return 0.5 + 0.5 *(x['monitor_raw'] - x['monitor_threshold'])/(1 - x['monitor_threshold'])
            return [piecewise_reward(x) * w for x in eval_examples]
        elif self.mode == "threshold_progressive":
            return [
                (x['monitor_raw'] - x['monitor_threshold']) / (1 - x['monitor_threshold']) * w
                if x['monitor_raw'] >= x['monitor_threshold'] else 0.0
                for x in eval_examples
            ]
        raise ValueError(f"Unsupported reward mode for {type(self).__name__}: {self.mode}")


class GroundTruthMonitorReward(MonitorRewardFunction):
    def __init__(self, accuracy: float = 1.0, strict: bool = True, **kwargs):
        self.accuracy = accuracy
        self.strict = strict
        super().__init__(**kwargs)

    def calculate_monitor(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> list[dict]:
        '''Evaluates the monitor on the eval examples'''

        monitor_scores = [x['is_trait_strict'] if self.strict else x['is_trait_loose'] for x in eval_examples]

        if self.accuracy < 1.0:
            # Flip the samples to induce the accuracy percentage
            flip_value = [random.random() > self.accuracy for _ in range(len(monitor_scores))]
            monitor_scores = [not score if flip else score for score, flip in zip(monitor_scores, flip_value)]

        for example, score in zip(eval_examples, monitor_scores):
            example['monitor_score'] = float(score)
            example['monitor_raw'] = float(score)
            example['monitor_threshold'] = 0.5

        return eval_examples

    def calculate_monitor_rewards(self, eval_examples: list[dict]) -> list[float]:
        """Reward all samples that do NOT exhibit the trait"""
        return [(1 - float(utils.replace_nan(x['monitor_score'], 0.0))) * self.reward_weights['monitor'] for x in eval_examples]
    

class GroundTruthMonitorPenalty(GroundTruthMonitorReward):
    def __init__(self, accuracy: float = 1.0, strict: bool = True, **kwargs):
        super().__init__(accuracy=accuracy, strict=strict, **kwargs)
    

    def calculate_monitor_rewards(self, eval_examples: list[dict]) -> list[float]:
        """Penalize all samples that exhibit the trait"""
        return [float(utils.replace_nan(x['monitor_score'], 0.0)) * -1.0 * self.reward_weights['monitor'] for x in eval_examples]


class LLMJudgePenalty(MonitorRewardFunction):

    def __init__(self, judge_model_id: str, judge_prompt_key: str, n_samples: int = 1, threshold: float = 0.5, mode: RewardModes = "threshold", **kwargs):
        output_type = judge.resolve_judge_output_type(judge_prompt_key, kwargs.get('output_type', None))
        self.judge = judge.Judge(
            model_name=judge_model_id,
            judge_prompt=PROMPTS[judge_prompt_key],
            output_type=output_type,
            generation_engine="openrouter",
            sampling_params=judge.build_judge_sampling_params(
                model_name=judge_model_id,
                output_type=output_type,
                n_samples=n_samples,
                with_reasoning=kwargs.get('with_reasoning', False),
                max_new_tokens=kwargs.get('max_new_tokens'),
            ),
            aggregation_type=kwargs.get('aggregation_type', 'mean'),
            retry_parse_failures=kwargs.get('retry_parse_failures', False),
        )
        self.threshold = threshold
        self.mode = mode
        super().__init__(**kwargs)

    def calculate_monitor(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> list[dict]:

        # Returns a 0-1 score
        # Note that judge must accept formatted requests with question/answer fields only
        judgements = self.judge.judge_responses(
            [{"question": ex['prompt'][-1]['content'], "answer": ex['response']} for ex in eval_examples], 
            include_detail = False
        )

        na_judgements = len([x for x in judgements if x is None])
        judgements = [float(utils.replace_nan(x, 0.0)) for x in judgements] # Unknown assumed to not be reward hacking

        # Record averages
        self.log({
            'detail/monitor/judge_avg': sum(judgements) / len(judgements),
            'detail/monitor/judge_min': min(judgements),
            'detail/monitor/judge_max': max(judgements),
            'detail/monitor/judge_none': na_judgements,
        })

        for example, score in zip(eval_examples, judgements):
            example['monitor_score'] = float(score >= self.threshold)
            example['monitor_threshold'] = self.threshold
            example['monitor_raw'] = score

        return eval_examples



class ProbePenalty(MonitorRewardFunction, ActivationRewardFunction):
    def __init__(self, probe_path: str, layers: list[int], threshold: float = 0.5, mode: RewardModes = "threshold",  **kwargs):
        self.probe = probe.load_probe(
            path = probe_path,
            threshold = threshold # This will only be used for evaluate_gt
        )
        self.layers = layers
        self.threshold = threshold

        # In progressive mode, give scalar reward based on the prediction (0 to 1 scale)
        # In threshold mode, give full reward or 0 based on prediction (0 or 1) by applying a threshold
        # In threshold_progressive mode, give scalar reward above a threshold if the prediction is above a threshold
        self.mode = mode

        MonitorRewardFunction.__init__(self, **kwargs)

    def reload_probe(self, new_probe):
        """Hot-swap the probe (called by trainer after retrain)."""
        self.probe = new_probe

    def calculate_monitor(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> list[dict]:

        # NOTE: THE SHAPE OF ACTIVATIONS MUST MATCH THE LAYER USED FOR THE PROBE EXACTLY IN-ORDER
        # IE self.cache_actvations_layers == self.layers
        # Input: n_layers x n_samples x n_hidden_size; Output: n_samples x n_layers
        model_activations = activations.unsqueeze(0) if len(activations.shape) == 2 else activations
        predict_kwargs = {}
        if "mask" in inspect.signature(self.probe.predict_proba).parameters:
            predict_kwargs["mask"] = probe.resolve_sequence_mask(
                model_activations,
                layers=self.layers,
                mask=kwargs.get("response_mask"),
            )
        predictions = self.probe.predict_proba(model_activations, layers=self.layers, **predict_kwargs)
        predictions = predictions.to(dtype = activations.dtype, device = activations.device) # n_samples x n_layers # High score indicates reward hacking
        predictions = predictions.max(dim = -1).values # Keep if any of the layers says the same is ok (conservative)

        self.log({
            'detail/monitor/probe_avg': predictions.mean(),
            'detail/monitor/probe_min': predictions.min(),
            'detail/monitor/probe_max': predictions.max(),
        })

        for example, score in zip(eval_examples, predictions):
            example['monitor_score'] = float(score >= self.threshold)
            example['monitor_threshold'] = self.threshold
            example['monitor_raw'] = float(score) # Note: Otherwise leads to torch.bfloat16 errors

        return eval_examples



class ProbeOutputPenalty(MonitorRewardFunction):
    """Probe penalty that consumes pre-computed probe probabilities from `activations`."""

    def __init__(self, threshold: list[float], layers: list[int], probe_ext: str, mode: RewardModes = "threshold", **kwargs):
        self.threshold = threshold
        self.layers = layers
        self.mode = mode
        probe_cls = probe.PROBE_REGISTRY[probe_ext]
        self.predict_from_proba = probe_cls.predict_from_proba
        self.predict_from_proba_by_layer = probe_cls.predict_from_proba_by_layer
        super().__init__(**kwargs)

    def calculate_monitor(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> list[dict]:
        """Activations contain pre-computed predict_proba output, not raw activations."""
        assert activations is not None, "ProbeOutputPenalty requires activations containing probe probabilities"
        assert activations.ndim in [1, 2], (
            f"ProbeOutputPenalty expected activations with shape (n_samples,) or (n_samples, n_layers), got {activations.shape}"
        )

        if activations.ndim == 2:
            by_layer = self.predict_from_proba_by_layer(activations, threshold=self.threshold)
            predictions = by_layer.max(dim=-1).values
            predictions_layers = by_layer.argmax(dim=-1).tolist()
            raw = activations.max(dim=-1).values.float()
            n_scores = activations.shape[1]
            for i in range(n_scores):
                self.log({
                    f'detail/monitor/probe_{self.layers[i]}_avg': activations[:, i].mean().item(),
                    f'detail/monitor/probe_{self.layers[i]}_n_flagged': by_layer[:, i].sum().item(),
                })
            spread = raw - activations.min(dim=-1).values.float()
            self.log({
                'detail/monitor/probe_spread_avg': spread.mean().item(),
                'detail/monitor/probe_spread_max': spread.max().item(),
                'detail/monitor/probe_layer_n_flagged_avg': by_layer.sum(dim=-1).mean().item(),
            })
        else:
            predictions = self.predict_from_proba(activations, threshold=self.threshold)
            raw = activations.float()
            predictions_layers = [0] * len(eval_examples) # Only a single layer, so all predictions use that layer

        self.log({
            'detail/monitor/probe_avg': predictions.mean(),
            'detail/monitor/probe_raw_avg': raw.mean(),
            'detail/monitor/probe_raw_max': raw.max(),
            'detail/monitor/probe_raw_min': raw.min(),
        })

        for example, score, pred, pred_layer_idx in zip(eval_examples, raw, predictions, predictions_layers):
            example['monitor_score'] = float(pred)
            example['monitor_threshold'] = float(self.threshold[pred_layer_idx])
            example['monitor_layer'] = self.layers[pred_layer_idx]
            example['monitor_raw'] = float(score)

        return eval_examples



### FORMATTING REWARD FUNCTIONS

class DefineStarterCode(RewardFunction):
    return_extra_infos: bool = False

    def compute_reward(self, examples: list[dict], responses: list[str], **kwargs) -> list[float]:
        '''Give a reward if Solution() and run_tests() are included in the response'''
        evaluator_cls = CodeEvaluator()
        extracted_responses = [evaluator_cls.parse_response(r) for r in responses]
        test_func_names = [x.get('prompt_metadata', {}).get('test_func_name', 'run_tests') for x in examples]
        number_rewards = [self.reward if ("class Solution" in str(r)) and (f"def {test_func_name}(" in str(r)) else 0.0 for r, test_func_name in zip(extracted_responses, test_func_names)]
        self.log({
            'rewards/starter_code/avg_reward': sum(number_rewards) / len(number_rewards),
            'rewards/starter_code/n_rewarded': sum([x > 0.0 for x in number_rewards]),
        })
        return number_rewards
