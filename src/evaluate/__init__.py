import os
import pandas as pd
import json

from src.generate import LLMGenerator, SamplingParams
from src import utils, analysis, generate
from src.monitor import judge

from src.evaluate.code.evaluator import CodeEvaluator, CodeEvaluationResult
from src.evaluate.evaluation import EvaluationParameters, EVALUATION_REGISTRY


def run_eval(
        llm_gen: LLMGenerator, 
        eval_params: EvaluationParameters,
        dataset: list[dict],
    ):

    evaluation_name = eval_params.evaluation_name

    if evaluation_name not in EVALUATION_REGISTRY:
        raise ValueError(f"Invalid evaluation: {evaluation_name}. Available: {list(EVALUATION_REGISTRY.keys())}")

    evaluation_instance = EVALUATION_REGISTRY[evaluation_name](
        llm_gen = llm_gen,
        config = eval_params,
        early_exit = False,
    )

    print(f"Running evaluation {evaluation_name} with model {llm_gen.model_name}")

    try:
        results = evaluation_instance.run(
            dataset = dataset
        )
    except BaseException as e:
        evaluation_instance.cleanup()
        raise e
        
    return results
