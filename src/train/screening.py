from abc import ABC, abstractmethod
import inspect
import torch
import random
import copy

from src.monitor import judge, probe
from src.prompts import PROMPTS
from src import utils, SamplingParams
from src.wandb_utils import wandb_log

'''
SCREENING FUNCTIONS

Screening scores should range from 0 to 1; Higher scores will be screened out based on the threshold value
'''


class ScreeningFunction(ABC):
    require_activations: bool = False
    high_reward_threshold: float = 3.0
    threshold: float = 0.5 # Note: For many judges, only 0/1 scores are returned
    last_screening_scores: list[float] | None = None
    last_monitor_raw: list[float] | None = None
    last_monitor_score: list[float] | None = None
    last_monitor_threshold: list[float] | None = None
    
    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def calculate_screening_scores(self, enriched_examples: list[dict], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[float]:
        pass

    def calculate_keep_samples(self, screening_scores: list[float], **kwargs) -> list[bool]:
        return [float(utils.replace_nan(x, 0.0)) < self.threshold for x in screening_scores]

    def __call__(self, examples: list[dict], responses: list[str], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[bool]:
        
        # Extract extra fields calculated by the reward function
        enriched_examples = self.extract_extra_fields(examples = examples, responses = responses, **kwargs)
        self.last_monitor_raw = None
        self.last_monitor_score = None
        self.last_monitor_threshold = None

        # Calculate keep samples
        screening_scores = self.calculate_screening_scores(enriched_examples = enriched_examples, rewards = rewards, activations = activations)
        self.last_screening_scores = [float(utils.replace_nan(x, 0.0)) for x in screening_scores]
        keep_samples = self.calculate_keep_samples(screening_scores = screening_scores)
        self.last_monitor_raw = self.last_screening_scores if self.last_monitor_raw is None else self.last_monitor_raw
        self.last_monitor_score = [float(not keep) for keep in keep_samples] if self.last_monitor_score is None else self.last_monitor_score
        if self.last_monitor_threshold is None:
            assert isinstance(self.threshold, (int, float)), f"{self.__class__.__name__} must set last_monitor_threshold for non-scalar thresholds"
            self.last_monitor_threshold = [float(self.threshold)] * len(keep_samples)

        # Log statistics
        self.log_screening_statistics(
            enriched_examples = enriched_examples,
            screening_scores = screening_scores,
            keep_samples = keep_samples,
            rewards = rewards,
        )
        return keep_samples # This is a boolean list the same length as the number of samples

    def log(self, *args, **kwargs):
        return wandb_log(*args, **kwargs)
    
    def extract_extra_fields(self, examples: list[dict], responses: list[str], **kwargs) -> dict:

        def try_len_eq(v, examples):
            try:
                return len(v) == len(examples)
            except:
                return False

        kwargs = {k: v for k, v in kwargs.items() if try_len_eq(v, examples)}
        kwargs['response'] = responses
        eval_examples = []
        for i, ex_base in enumerate(examples):
            ex = copy.deepcopy(ex_base)
            for k in kwargs.keys():
                ex[k] = kwargs[k][i]
            eval_examples.append(ex)

        return eval_examples
    
    def log_screening_statistics(
            self,
            enriched_examples: list[dict],
            screening_scores: list[float],
            keep_samples: list[bool] | list[int], 
            rewards: torch.Tensor,
        ):

        screening_logs = {
            'detail/screening/avg_score': sum([float(utils.replace_nan(x, 0.0)) for x in screening_scores]) / len(screening_scores),
            'detail/screening/n_none_score': sum([x is None for x in screening_scores]),
            'detail/screening/n_keep': sum(keep_samples),
            'detail/screening/ratio_keep': utils.safe_divide(sum(keep_samples), len(keep_samples)),
            'detail/screening/avg_reward_keep': utils.safe_divide(sum(rewards[keep_samples]), sum(keep_samples)),
            'detail/screening/avg_reward_discard': utils.safe_divide(sum(rewards[[not x for x in keep_samples]]), (len(keep_samples) - sum(keep_samples))),
            'detail/screening/n_keep_high_reward': sum(rewards[keep_samples] >= self.high_reward_threshold),
            'detail/screening/ratio_keep_high_reward': utils.safe_divide(sum(rewards[keep_samples] >= self.high_reward_threshold), sum(keep_samples)),
        }

        is_trait_strict = [x['is_trait_strict'] for x in enriched_examples]
        is_trait_loose = [x['is_trait_loose'] for x in enriched_examples]

        screening_logs.update({
            'detail/screening/n_trait_strict': sum(is_trait_strict),
            'detail/screening/strict_recall': utils.safe_divide(sum([not keep and is_rh for keep, is_rh in zip(keep_samples, is_trait_strict)]), sum(is_trait_strict)),
            'detail/screening/strict_accuracy': utils.safe_divide(sum([keep == (not is_rh) for keep, is_rh in zip(keep_samples, is_trait_strict)]), len(keep_samples)),
            'detail/screening/n_trait_loose': sum(is_trait_loose),
            'detail/screening/loose_recall': utils.safe_divide(sum([not keep and is_rh for keep, is_rh in zip(keep_samples, is_trait_loose)]), sum(is_trait_loose)),
            'detail/screening/loose_accuracy': utils.safe_divide(sum([keep == (not is_rh) for keep, is_rh in zip(keep_samples, is_trait_loose)]), len(keep_samples))
        })
        
        self.log(screening_logs)
    



class GroundTruthMonitor(ScreeningFunction):
    '''Monitor the reward hacking evaluation results'''
    def __init__(self, strict: bool = False, accuracy: float = 1.0, **kwargs):
        self.strict = strict
        self.accuracy = accuracy
        # Threshold does not apply because ground truth monitor only returns 0/1 score

    def calculate_screening_scores(self, enriched_examples: list[dict], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        scores = [x['is_trait_strict'] if self.strict else x['is_trait_loose'] for x in enriched_examples]
        
        # Apply accuracy threshold
        if self.accuracy < 1.0:
            # Flip the samples to induce the accuracy percentage
            flip_value = [random.random() > self.accuracy for _ in range(len(scores))]
            scores = [not keep if flip else keep for keep, flip in zip(scores, flip_value)]

        scores = [float(x) for x in scores]
        return scores
    


class LLMJudgeScreener(ScreeningFunction):
    '''Screen out samples that are determined to be reward hacking by the judge model
    Always use definition from judge file to ensure consistency with evaluation settings
    '''
    def __init__(self, judge_model_id: str, judge_prompt_key: str, n_samples: int = 1, threshold: float = 0.5, **kwargs):
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
    
    def calculate_screening_scores(self, enriched_examples: list[dict], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[float]:
        # Returns a 0-1 score; Note that judge must accept formatted requests with question/answer fields only
        # calculate_keep_samples will apply the thresholding
        judgements = self.judge.judge_responses(
            [{"question": ex['prompt'][-1]['content'], "answer": ex['response']} for ex in enriched_examples], 
            include_detail = False
        )
        return judgements
    

class ProbeScreener(ScreeningFunction):
    '''Screen out samples that are determined to be reward hacking by the probe model'''
    def __init__(self, probe_path: str, layers: list[int], threshold: float = 0.5):
        self.probe = probe.load_probe(
            path = probe_path,
            threshold = threshold
        )
        self.layers = layers
        self.threshold = threshold

    def reload_probe(self, new_probe):
        """Hot-swap the probe (called by trainer after retrain)."""
        self.probe = new_probe

    def calculate_screening_scores(self, enriched_examples: list[dict], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[float]:
        # If layers is only one layer, then activations will be n_samples x n_hidden_size
        if len(activations.shape) == 2:
            activations = activations.unsqueeze(0)
        predict_kwargs = {}
        if "mask" in inspect.signature(self.probe.predict_proba).parameters:
            predict_kwargs["mask"] = probe.resolve_sequence_mask(
                activations,
                layers=self.layers,
                mask=kwargs.get("response_mask"),
            )
        raw_predictions = self.probe.predict_proba(activations, layers=self.layers, **predict_kwargs)
        predictions = raw_predictions.to(dtype = activations.dtype, device = activations.device) # n_samples x n_layers # High score indicates reward hacking
        predictions = predictions.max(dim = -1).values # Keep if any of the layers says the same is ok (conservative)
        return predictions.tolist()


class ProbeOutputScreener(ScreeningFunction):
    """Screen using pre-computed probe probabilities from activations."""
    def __init__(self, threshold: list[float] | float = 0.5, **kwargs):
        self.threshold = threshold

    def calculate_screening_scores(self, enriched_examples: list[dict], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[float]:
        assert activations is not None, "ProbeOutputScreener requires activations containing probe probabilities"
        assert len(activations.shape) in [1, 2], (
            f"ProbeOutputScreener expected shape (n_samples,) or (n_samples, n_layers), got {activations.shape}"
        )
        predictions_by_layer = probe.Probe.predict_from_proba_by_layer(activations, threshold=self.threshold)
        predictions = predictions_by_layer.max(dim=-1).values
        n_layers = 1 if activations.ndim == 1 else activations.shape[1]
        thresholds = [float(self.threshold)] * n_layers if isinstance(self.threshold, (int, float)) else [float(x) for x in self.threshold]
        assert len(thresholds) == n_layers, f"Number of thresholds ({len(thresholds)}) must match layers ({n_layers})"
        if activations.ndim == 1:
            self.last_monitor_raw = activations.float().tolist()
            self.last_monitor_threshold = [thresholds[0]] * activations.shape[0]
        else:
            pred_layer_idx = predictions_by_layer.argmax(dim=-1).tolist()
            self.last_monitor_raw = activations.max(dim=-1).values.float().tolist()
            self.last_monitor_threshold = [thresholds[idx] for idx in pred_layer_idx]
        self.last_monitor_score = predictions.float().tolist()
        return predictions.tolist()

    def calculate_keep_samples(self, screening_scores: list[float], **kwargs) -> list[bool]:
        """screening_scores should always be 0 or 1 because they come from predict_from_proba output"""
        return [float(x) != 1.0 if x is not None else True for x in screening_scores]
