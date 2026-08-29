#!/bin/bash

setup_env(){
    source .env
    source .env.gpu
    source $VENV_DIR/bin/activate
}

# --no-sync: flash-attn is compiled from source by setup.sh and is deliberately not a locked
# dependency (no wheel exists for this torch build), so an implicit `uv run` sync would prune it and
# break training. setup.sh is the one place that installs or updates dependencies.
run_rl_training() {
    uv run --no-sync --active --group=dev scripts/run_rl_training.py "$@"
}

resume_rl_training() {
    uv run --no-sync --active --group=dev scripts/run_rl_training.py resume "$@"
}

eval_model() {
    local environment="$1"
    local run_name="$2"
    local checkpoint="$3"
    local model_id="qwen3-4b"

    if [[ -z "$checkpoint" ]]; then
        local checkpoint="400"
    fi

    if [[ -z "$run_name" ]]; then
        echo "Running $environment eval for base model"
        uv run --no-sync --active --dev scripts/run_eval.py default --env=$environment --overwrite=True
    else
        echo "Running $environment eval for $run_name with checkpoint $checkpoint"
        uv run --no-sync --active --dev scripts/run_eval.py default \
            --env=$environment \
            --run_name=$run_name \
            --checkpoint=$checkpoint \
            --overwrite=True
    fi
}

train_probe() {
    uv run --no-sync --active --dev python scripts/run_monitors.py train "$@"
}

benchmark_judges() {
    uv run --no-sync --active --dev python scripts/run_monitors.py benchmark_judges "$@"
}
