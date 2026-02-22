#!/bin/bash

# Configuration
NUM_GPUS=1
CONFIG_FILE="cosmos_policy/config/config.py"
EXPERIMENT="cosmos_predict2_2b_480p_anytask"

# Optional: Resume from LIBERO policy checkpoint
# LIBERO_CKPT="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_model.pt"

echo "Starting fine-tuning with $NUM_GPUS GPUs..."
echo "Experiment: $EXPERIMENT"

# Check for verify mode
EXTRA_OPTS=""
if [[ "$*" == *"verify"* ]]; then
    echo "Running in VERIFY mode (minimal resources)..."
    # Override batch size and max steps if possible via command line
    # Depending on how the config handles overrides, we might need specific syntax
    # For LazyConfig, it's usually path.key=value
    EXTRA_OPTS="dataloader_train.batch_size=1 trainer.max_iter=1 trainer.logging_iter=1"
    # Remove 'verify' from arguments
    set -- "${@/verify/}"
fi

echo "Passing extra arguments: $EXTRA_OPTS $@"

# Build the command arguments array
ARGS=(
    --config "$CONFIG_FILE"
    --
    "experiment=$EXPERIMENT"
)

# Add extra options if they exist
if [ -n "$EXTRA_OPTS" ]; then
    for opt in $EXTRA_OPTS; do
        ARGS+=("$opt")
    done
fi

# Add remaining script arguments
for arg in "$@"; do
    if [ -n "$arg" ]; then
        ARGS+=("$arg")
    fi
done

echo "Passing extra arguments: ${ARGS[@]:2}"

# Use a dynamic port to avoid EADDRINUSE errors
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

uv run --extra cu128 --group libero --python 3.10 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    -m cosmos_policy.scripts.train "${ARGS[@]}"
