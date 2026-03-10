#!/bin/bash

# Bootstrap remote machine path if needed.
HOME_USER_DIR="/home/zwang"
SOURCE_REPO_DIR="/storage/nfs/zwang/cosmos-policy"

if [ ! -d "$HOME_USER_DIR" ]; then
    echo "Directory $HOME_USER_DIR not found. Creating it and copying repository..."
    mkdir -p "$HOME_USER_DIR"

    if [ ! -d "$SOURCE_REPO_DIR" ]; then
        echo "Error: source repository not found at $SOURCE_REPO_DIR"
        return 1 2>/dev/null || exit 1
    fi

    cp -a "$SOURCE_REPO_DIR" "$HOME_USER_DIR/"
    echo "Copied $SOURCE_REPO_DIR to $HOME_USER_DIR/"
fi

export NUM_GPUS="${NUM_GPUS:-8}"
export CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"
export MASTER_PORT="${MASTER_PORT:-12341}"

# Isolate uv virtual environment and cache locally per pod
export UV_PROJECT_ENVIRONMENT="/tmp/.venv-$(hostname)"
export UV_CACHE_DIR="/tmp/uv-cache-$(hostname)"
export IMAGINAIRE_OUTPUT_ROOT="/storage/nfs/zwang/cosmos-policy-checkpoints/$(hostname)"
