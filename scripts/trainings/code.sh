#!/bin/bash
# Training commands for the code environment.
# Not intended to be run directly, but rather to provide examples of commands used to train models for the paper


export SEED=${SEED:-1}
export ENV="leetcode_rh"
export STEPS=400
export MONITOR_WEIGHT=3.0 # Monitor weight is the sum of other reward term weights in hackable environment

# Judge settings
export JUDGE_PROMPT_KEY="monitor_code_010"
export JUDGE_MODEL_ID="qwen/qwen3-235b-a22b-2507"
export JUDGE_THRESHOLD=0.3

# Probe settings
export PROBE_PATH="" # Add probe path here after training
export PROBE_LAYER=1 # Add probe layer here after training
export PROBE_THRESHOLD=0.5 # Add probe threshold here after training

# Additional non-default datasets
export DATASET_MEDIUM="results/data/leetcode_train_medhard_filtered_40.jsonl"
export DATASET_EASY="results/data/leetcode_train_medhard_filtered_50.jsonl"

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

# Ground Truth Monitor
run_rl_training ground_truth \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --intervention=penalty

# Ground Truth Monitor - Simulated Accuracy 70%
run_rl_training ground_truth \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --intervention=penalty \
    --accuracy=0.7

# Judge Monitor
# Change --intervention=penalty to --intervention=screening to use screening intervention instead
run_rl_training llmjudge \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --intervention=penalty \
    --judge_prompt_key=$JUDGE_PROMPT_KEY \
    --judge_model_id=$JUDGE_MODEL_ID \
    --threshold=$JUDGE_THRESHOLD \
    --monitor_weight=$MONITOR_WEIGHT

# Probe Monitor
# Change --intervention=penalty to --intervention=screening to use screening intervention instead
run_rl_training probe \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --intervention=penalty \
    --probe_path=$PROBE_PATH \
    --probe_layers=$PROBE_LAYER \
    --probe_threshold=$PROBE_THRESHOLD

# Reward Hacking - Medium Dataset
run_rl_training no_intervention \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --train_dataset_path=$DATASET_MEDIUM

# Reward Hacking - Easy Dataset
run_rl_training no_intervention \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --train_dataset_path=$DATASET_EASY

# ---------------------------------------------------------------------------
# Gemma 4
# ---------------------------------------------------------------------------
# Any model is selected with --model_id; Gemma 4 E2B/E4B need no other changes (LoRA targets and
# the fused-kernel choice are derived from the HF config - see src/train/verl/utils.py).
export GEMMA_MODEL_ID="google/gemma-4-E2B-it"   # or google/gemma-4-E4B-it

run_rl_training no_intervention \
    --env=$ENV \
    --seed=$SEED \
    --steps=$STEPS \
    --model_id=$GEMMA_MODEL_ID \
    --enable_thinking=True \
    --gpu_memory_utilization 0.7

# Short smoke run: two steps at a small batch/group size and a short completion length. Useful to
# check a model end to end without waiting on a full-length step.
run_rl_training no_intervention \
    --env=$ENV \
    --seed=42 \
    --steps=2 \
    --model_id=$GEMMA_MODEL_ID \
    --enable_thinking=True \
    --gpu_memory_utilization 0.7 \
    --num_prompts=2 \
    --num_generations=4 \
    --mini_batch_size=2 \
    --per_device_batch_size=4 \
    --max_completion_length=512 \
    --save_steps=100
