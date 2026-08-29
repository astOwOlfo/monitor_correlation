#!/usr/bin/env bash

# Contains some path variables. set -a so they are exported, not just set as shell variables.
set -a
source .env.gpu
set +a

# Run installation of basic packages
apt-get update
apt-get install -y vim
apt-get install -y git
apt-get install -y tmux
apt-get install -y unzip
apt-get install -y acl

# flash-attn is compiled from source below and has to be built against the same CUDA major
# version as the installed torch (currently CUDA 13, torch 2.11), so install that toolkit rather
# than the distro's nvidia-cuda-toolkit. Requires a driver supporting CUDA 13 or newer.
export CUDA_HOME=/usr/local/cuda-13.0
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  apt-get update
  apt-get install -y --no-install-recommends cuda-toolkit-13-0
fi
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export PATH="$HOME/.local/bin:$PATH"

# Load other environment variables
pip install uv
uv venv $VENV_DIR
source $VENV_DIR/bin/activate
ln -s $VENV_DIR .venv

# Load environment variables + commands
set -a
source .env
source .env.gpu
set +a
source commands.sh

# Sync dependencies
uv sync --active --dev
uv pip install --no-deps -e verl/

# flash-attn: prefers a prebuilt wheel and falls back to compiling. Required by Verl's model
# engine (it unpads every log-prob batch through flash_attn.bert_padding), but deliberately not a
# locked dependency - see the comments in the script.
bash scripts/install_flash_attn.sh 