#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bin/setup_finetune_env.sh"

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

uv run --extra cu128 --group libero --python 3.10 torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m cosmos_policy.scripts.train \
    --config "$CONFIG_FILE" \
    -- \
    "experiment=$EXPERIMENT" \
    "trainer.grad_accum_iter=8" \
    "${EXTRA_OPTS[@]}"
