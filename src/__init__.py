from dataclasses import dataclass, asdict
from typing import TypedDict, Optional, Literal

import copy

DEFAULT_MODEL_ID = "qwen/Qwen3-4B"
RESULTS_PATH = "results"
USE_FLASH_ATTN = True

REASONING_MODELS = {
    "qwen3-4b",
    "qwen3-8b",
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
}

# Families whose chat template accepts an `enable_thinking` kwarg. Matched as a prefix so
# sibling sizes/variants are picked up without being listed in REASONING_MODELS explicitly.
REASONING_MODEL_PREFIXES = (
    "qwen3-",
    "gemma-4-",
)

def is_reasoning_model(model_id: str) -> bool:
    name = model_id.lower().split('/')[-1]
    return name in REASONING_MODELS or name.startswith(REASONING_MODEL_PREFIXES)

# Delimiters wrapping a model's chain of thought, as (open, close) pairs. Qwen3 emits
# <think>...</think>; Gemma 4 emits a thought channel. Kept as text because the reward path works
# on decoded responses - see decode_preserving_reasoning for how the markers survive decoding.
REASONING_DELIMITERS = (
    ("<think>", "</think>"),
    ("<|channel>", "<channel|>"),
)


def strip_reasoning_spans(text: str) -> str:
    """Remove chain-of-thought spans from a decoded response.

    Anything the model wrote while reasoning is working-out, not its answer. An unterminated span
    means generation was cut off mid-thought, so everything from the marker on is dropped too -
    there is no answer in that case.
    """
    import re

    for opening, closing in REASONING_DELIMITERS:
        text = re.sub(re.escape(opening) + ".*?" + re.escape(closing), "", text, flags=re.DOTALL)
        opened = text.find(opening)
        if opened != -1:
            text = text[:opened]
    return text.strip()


class ChatMessage(TypedDict):
    role: str
    content: str

ChatRequest = list[ChatMessage]

def extract_system_prompt(prompt: ChatRequest) -> str:
    sys_msgs = [x for x in prompt if x['role'] == 'system']
    if len(sys_msgs) > 1:
        return "\n".join([x['content'] for x in sys_msgs])
    else:
        return ""

def add_system_prompt(prompt: ChatRequest, system_prompt: str, method: Literal['after', 'before', 'replace'] = 'after') -> ChatRequest:
    new_prompt = copy.deepcopy(prompt)
    if new_prompt[0]['role'] == 'system':
        if method == 'replace':
            new_prompt[0]['content'] = system_prompt
        elif method == 'after':
            new_prompt[0]['content'] = (new_prompt[0]['content'] + "\n" + system_prompt)
        elif method == 'before':
            new_prompt[0]['content'] = (system_prompt + '\n' + new_prompt[0]['content'])
        else:
            raise ValueError(f"Invalid method: {method}")
    else:
        new_prompt = [{'role': 'system', 'content': system_prompt}] + new_prompt
    return new_prompt

@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.95
    with_reasoning: bool = False
    return_reasoning: bool = False  # If True and with_reasoning=True, return (content, reasoning) tuples
    repetition_penalty: float = 1.0

    def to_dict(self):
        return asdict(self)

class DatasetExample(TypedDict):
    id: int
    dataset: str
    evaluator: str
    question: str
    gt_answer: str | list[str]
    prompt: ChatRequest
    answer: str | list[str]
    hint: str | None
    prompt_metadata: dict = {} # Optional metadata about the prompt

DatasetExampleFields = ['id', 'dataset', 'evaluator', 'question', 'gt_answer', 'prompt', 'answer', 'hint', 'prompt_metadata']


class CodeDatasetExample(DatasetExample):
    func_name: str # Function name that the program needs to write
    setup_code: str
    test_func_name: str = "run_tests"
    test_func_code: str = ""
    test_list: list[str] = []
    difficulty: str = "None"
    canonical_solution: str | None = None

CodeDatasetExampleFields = DatasetExampleFields + ['func_name', 'setup_code', 'difficulty', 'canonical_solution']


class ChatDatasetExample(DatasetExample):
    question_system_prompt: str

ChatDatasetExampleFields = DatasetExampleFields + ['system_prompt']

class Response(DatasetExample):
    response: str


class LabeledResponse(Response):
    label: str


class JudgedResponse(Response):
    judge_model: str
    judge_prompt: str
    judge_output: str
    