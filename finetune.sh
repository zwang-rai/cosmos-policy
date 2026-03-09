#!/bin/bash

export NUM_GPUS="${NUM_GPUS:8}"
export CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"

if [ -z "$1" ]; then
    echo "Usage: $0 <EXPERIMENT> [extra_opts...]"
    exit 1
fi
EXPERIMENT="$1"
shift

echo "Starting fine-tuning with $NUM_GPUS GPUs..."
echo "Experiment: $EXPERIMENT"

# Collect extra options, handling 'verify' mode specially
EXTRA_OPTS=()
for arg in "$@"; do
    if [[ "$arg" == "verify" ]]; then
        echo "Running in VERIFY mode (minimal resources)..."
        EXTRA_OPTS+=(
            "trainer.max_iter=1"
            "trainer.logging_iter=1"
            "callbacks=[basic]"
        )
    else
        EXTRA_OPTS+=("$arg")
    fi
done

if [ ${#EXTRA_OPTS[@]} -gt 0 ]; then
    echo "Passing extra arguments: ${EXTRA_OPTS[*]}"
fi

# Use a dynamic port to avoid EADDRINUSE errors
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# Isolate uv virtual environment and cache locally per pod
export UV_PROJECT_ENVIRONMENT="/tmp/.venv-$(hostname)"
export UV_CACHE_DIR="/tmp/uv-cache-$(hostname)"
export IMAGINAIRE_OUTPUT_ROOT="/storage/nfs/zwang/cosmos-policy-checkpoints/$(hostname)"

uv run --extra cu128 --group aloha --python 3.10 torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --rdzv_endpoint="localhost:$MASTER_PORT" \
    -m cosmos_policy.scripts.train \
    --config "$CONFIG_FILE" \
    -- \
    "experiment=$EXPERIMENT" \
    "${EXTRA_OPTS[@]}"
