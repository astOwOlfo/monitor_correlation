from typing import Any, List, TypedDict, Literal
import re
import gc
import warnings

from src import extract_reasoning_spans, strip_reasoning_spans
from src.prompts import PROMPTS
from src.generate import SamplingParams, create_llm_generator, to_chatml


class JudgeRequest(TypedDict):
    system_prompt: str | None = None
    gt_answer: str | None = None
    question: str
    answer: str
    cot: str          # the assistant's chain of thought, "(none provided)" when it has none
    final_answer: str # the response with the chain of thought removed


def build_judge_request(question: str, response: str, **extra) -> JudgeRequest:
    """Every field a judge prompt may format against, for one (question, response) pair.

    `answer` is the raw response, which is what the environment judge prompts score. The rubric
    monitors ported from MonitorDecorrelation split it instead: a CoT monitor reads `cot` plus
    `final_answer`, and an output-only monitor reads `final_answer` alone, so it never sees the
    reasoning. Prompts ignore the fields they do not reference.
    """
    text = response or ""
    return JudgeRequest(
        question=question,
        answer=text,
        cot=extract_reasoning_spans(text) or "(none provided)",
        final_answer=strip_reasoning_spans(text),
        **extra,
    )


JudgeOutputType = Literal["binary", "yesno", "0100", "010", "string", "categorical"]


def resolve_judge_output_type(prompt_key: str, output_type: JudgeOutputType | None = None) -> JudgeOutputType:
    return output_type or prompt_key.split("_")[-1]


def judge_uses_reasoning(model_name: str, with_reasoning: bool) -> bool:
    model = model_name.lower()
    return with_reasoning or "gpt-5" in model or "o1" in model


def judge_needs_reasoning_token_budget(model_name: str, with_reasoning: bool) -> bool:
    model = model_name.lower()
    return judge_uses_reasoning(model_name, with_reasoning) or "gpt-oss" in model


def build_judge_sampling_params(
    model_name: str,
    output_type: JudgeOutputType,
    n_samples: int = 1,
    with_reasoning: bool = False,
    max_new_tokens: int | None = None,
) -> SamplingParams:
    effective_reasoning = judge_uses_reasoning(model_name, with_reasoning)
    if effective_reasoning != with_reasoning:
        raise ValueError(f"Model {model_name} requires reasoning but with_reasoning={with_reasoning} was passed")
    if max_new_tokens is None:
        if output_type in ["binary", "yesno", "010"]:
            max_new_tokens = 1024 if judge_needs_reasoning_token_budget(model_name, with_reasoning) else 16
        else:
            max_new_tokens = 4096 if judge_needs_reasoning_token_budget(model_name, with_reasoning) else 1024
    return SamplingParams(
        with_reasoning=effective_reasoning,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
        n=n_samples,
    )


class Judge:
    def __init__(
        self, 
        model_name: str, 
        judge_prompt: str, 
        judge_system_prompt: str | None = None,
        output_type: JudgeOutputType = "binary", 
        generation_engine: str = "openrouter", 
        sampling_params: SamplingParams | None = None,
        aggregation_type: str = "mean",
        debug: bool = False,
        key_prefix: str = "",
        categorical: list[str] = [], # When output_type == "categorical", list of categories
        generator_kwargs: dict | None = None,
        retry_parse_failures: bool = False,
    ):
        self.model_name = model_name
        self.generation_engine = generation_engine
        self.sampling_params = sampling_params or SamplingParams(temperature = 0.0, max_new_tokens = 512)
        self.llm_gen = None

        self.judge_prompt = judge_prompt
        self.judge_system_prompt = judge_system_prompt
        self.output_type = output_type
        self.debug = debug
        self.aggregation_type = aggregation_type
        self.key_prefix = key_prefix
        self.categorical = categorical
        self.generator_kwargs = generator_kwargs or {}
        self.retry_parse_failures = retry_parse_failures

        print(f"Judge initialized with engine {self.generation_engine} for model {self.model_name}")


    def cleanup_router(self):
        self.llm_gen.cleanup()
        del self.llm_gen
        gc.collect()
        self.llm_gen = None
    
    def open_router(self):
        self.llm_gen = create_llm_generator(self.generation_engine, model_name=self.model_name, **self.generator_kwargs)

    def score_responses(self, responses) -> List[Any]:
        '''All numeric scores are normalized to 0-1'''
        
        if self.output_type == "binary":
            return [None if not isinstance(response, str) else (1.0 if "1" in response.lower()[:10] else (0.0 if "0" in response.lower()[:10] else None)) for response in responses]
        elif self.output_type == "yesno":
            return [None if not isinstance(response, str) else (1.0 if "YES" in response.upper() else (0.0 if "NO" in response.upper() else None)) for response in responses]
        elif self.output_type == "0100":
            pattern  =  re.compile(r'(?<!\d)(100(?:\.0+)?|(?:\d{1,2})(?:\.\d+)?)')
            scores  =  []
            for response in responses:
                try:
                    match  =  pattern.search(response)
                    if not match:
                        print(f"Could not parse 0-100 score from: {response!r}")
                        scores.append(None)
                    else:
                        scores.append(float(match.group(1))/100.0)
                except:
                    scores.append(None)
            return scores
        elif self.output_type == "010":
            pattern = re.compile(r'(?<!\d)(10(?:\.0+)?|(?:\d)(?:\.\d+)?)')
            scores = []
            for response in responses:
                try:
                    match = pattern.search(response)
                    if not match:
                        print(f"Could not parse 0-10 score from: {response!r}")
                        scores.append(None)
                    else:
                        scores.append(float(match.group(1))/10.0)
                except:
                    scores.append(None)
            return scores
        elif self.output_type == "string":
            return responses
        elif self.output_type == "categorical":
            # Match to the category term that appears earliest in the response
            result = []
            for response in responses:
                if not isinstance(response, str):
                    result.append(None)
                    continue
                norm_resp = response.lower()
                min_idx = None
                selected_cat = None
                for cat in self.categorical:
                    idx = norm_resp.find(cat.lower())
                    if idx != -1 and (min_idx is None or idx < min_idx):
                        min_idx = idx
                        selected_cat = cat
                result.append(selected_cat)
            return result
        elif self.output_type == "float":
            return [self.try_float(response) for response in responses]
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")


    def try_float(self, x: str) -> float | None:
        try:
            return float(x)
        except:
            return None

    def convert_floats(self, scores: list[Any]) -> list[Any]:
        out = []
        for score in scores:
            try:
                out.append(float(score))
            except:
                out.append(None)
        return out
    
    def aggregate_scores(self, scores: list[float]) -> float:

        no_na_scores = [x for x in scores if x is not None]
        if len(no_na_scores) == 0:
            return None 
        else:
            if self.aggregation_type == "mean":
                return sum(no_na_scores)/len(no_na_scores)
            elif self.aggregation_type == "max":
                return max(no_na_scores)
            elif self.aggregation_type == "min":
                return min(no_na_scores)
            else:
                raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")


    def _should_retry_parse_failures(self) -> bool:
        return self.retry_parse_failures and self.output_type not in ["string", "categorical"]


    def _judge_formatted_prompts(self, formatted_prompts: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
        judge_responses = self.llm_gen.batch_generate(formatted_prompts, self.sampling_params)
        if self.debug:
            print("Responses judged", len(judge_responses), len(judge_responses[0]) if self.sampling_params.n > 1 else 1)

        judge_scores = [] # List of lists of scores for each response if n > 1
        judge_agg_scores = [] # List of final score for each response

        # If n == 1, then we just score the responses
        if self.sampling_params.n == 1:
            judge_scores = self.score_responses(judge_responses)
            judge_agg_scores = judge_scores

            if self.debug:
                print("Scored", judge_responses, judge_scores)
        
        # If n > 1, then we will have a list
        else:
            for response_ls in judge_responses:
                if self.output_type == "string":
                    judge_scores.append(response_ls)
                    judge_agg_scores.append(None)
                else:
                    resp = self.score_responses(response_ls)
                    resp = self.convert_floats(resp)
                    judge_scores.append(resp)
                    judge_agg_scores.append(self.aggregate_scores(resp))
                if self.debug:
                    print("Scored", response_ls, judge_scores[-1], judge_agg_scores[-1])

        return judge_agg_scores, judge_scores, judge_responses


    def judge_responses(self, requests: List[JudgeRequest], include_detail: bool = False) -> tuple[Any]:

        if self.llm_gen is None:
            self.open_router()

        # Format into judge prompts
        formatted_prompts = [self.judge_prompt.format(**request) for request in requests]
        formatted_prompts = [to_chatml(x, system_prompt=self.judge_system_prompt) for x in formatted_prompts]
        if self.debug:
            print("Prompts formatted", len(formatted_prompts))

        judge_agg_scores, judge_scores, judge_responses = self._judge_formatted_prompts(formatted_prompts)

        if self._should_retry_parse_failures():
            failed_indices = [i for i, score in enumerate(judge_agg_scores) if score is None]
            if failed_indices:
                retry_agg_scores, retry_scores, retry_responses = self._judge_formatted_prompts([formatted_prompts[i] for i in failed_indices])
                for idx, retry_agg_score, retry_score, retry_response in zip(failed_indices, retry_agg_scores, retry_scores, retry_responses):
                    judge_agg_scores[idx] = retry_agg_score
                    judge_scores[idx] = retry_score
                    judge_responses[idx] = retry_response

        if include_detail:
            return [(x, y, z) for x, y, z in zip(judge_agg_scores, judge_scores, judge_responses)]
        else:
            return judge_agg_scores
