#!/bin/bash

setup_env(){
    # set -a so the env files are exported rather than left as shell variables: some settings are
    # read by libraries at import time (RAY_ENABLE_UV_RUN_RUNTIME_ENV), which is before
    # scripts/run_rl_training.py gets a chance to load them into os.environ itself.
    set -a
    source .env
    source .env.gpu
    set +a
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

# Build a dataset from source into results/data (see scripts/run_data_process.py and src/data/).
# Only needed to build a dataset ahead of time: a self-building one (codeforces_ib) is built by the
# first run that needs it. See src/data/base.py::ensure_dataset.
#   process_data download --dataset_name=codeforces_ib --split=train
process_data() {
    uv run --no-sync --active --group=dev scripts/run_data_process.py "$@"
}

# Rebuild the coding datasets for a different model: measures pass@16 over the base problem pool
# and splits it into the training / holdout / medium / easy datasets. See scripts/run_dataset_filter.py.
filter_dataset() {
    uv run --no-sync --active --group=dev scripts/run_dataset_filter.py "$@"
}

train_probe() {
    uv run --no-sync --active --dev python scripts/run_monitors.py train "$@"
}

benchmark_judges() {
    uv run --no-sync --active --dev python scripts/run_monitors.py benchmark_judges "$@"
}
