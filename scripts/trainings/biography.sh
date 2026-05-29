#!/bin/bash
# Training commands for the biography environment.
# Not intended to be run directly, but rather to provide examples of commands used to train models for the paper

export SEED=${SEED:-1}
export ENV="biography"
export STEPS=200
export MONITOR_WEIGHT=4.0 # Monitor weight is the sum of other reward term weights in hackable environment

# Non-default easier dataset used for the best intervention setups
export DATASET_EASY="results/data/biography_train_base_easy.jsonl"

export JUDGE_PROMPT_KEY="monitor_biography_010"
export JUDGE_MODEL_ID="z-ai/glm-5.1"
export JUDGE_THRESHOLD=0.1

export PROBE_PATH="" # Add probe path here after training
export PROBE_LAYER=1 # Add probe layer here after training
export PROBE_THRESHOLD=0.5 # Add probe threshold here after training


# RL Baseline (Non-Hackable)
run_rl_training rl_baseline \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS

# Reward Hacking Environment, No Monitoring
run_rl_training no_intervention \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS

# Judge Penalty (Best Setup)
run_rl_training llmjudge \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --intervention=penalty \
    --judge_prompt_key=$JUDGE_PROMPT_KEY \
    --judge_model_id=$JUDGE_MODEL_ID \
    --threshold=$JUDGE_THRESHOLD \
    --monitor_weight=$MONITOR_WEIGHT \
    --train_dataset_path=$DATASET_EASY

# Probe Screening (Best Setup)
run_rl_training probe \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --intervention=screening \
    --probe_path=$PROBE_PATH \
    --probe_layers=$PROBE_LAYER \
    --probe_threshold=$PROBE_THRESHOLD \
    --train_dataset_path=$DATASET_EASY
