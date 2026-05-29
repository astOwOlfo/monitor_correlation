#!/usr/bin/env bash

# Contains some path variables
source .env.gpu

# Run installation of basic packages
apt-get update
apt-get install -y vim
apt-get install -y git
apt-get install -y tmux
apt-get install -y unzip
apt-get install -y acl

if [ ! -f /usr/local/cuda/include/curand.h ]; then
  apt-get update
  apt-get install -y nvidia-cuda-toolkit
fi

export PATH="$HOME/.local/bin:$PATH"

# Load other environment variables
pip install uv
uv venv $VENV_DIR
source $VENV_DIR/bin/activate
ln -s $VENV_DIR .venv

# Load environment variables + commands
source .env
source .env.gpu
source commands.sh

# Sync dependencies
uv sync --active --dev
uv pip install --no-deps -e verl/ 