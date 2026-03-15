#!/bin/bash

# Path setup
BASE_DIR="/home/viethq/Projects/elworld"
SRC_DIR="$BASE_DIR/research/training/src"
VENV_PYTHON="$BASE_DIR/venv/bin/python"
BEST_PARAMS_FILE="$SRC_DIR/outputs/optuna/best_memory_params.json"

echo "=== Full Training Orchestrator ==="
echo "Waiting for Optuna study to finish and generate $BEST_PARAMS_FILE..."

# Loop until best params file is generated
while [ ! -f "$BEST_PARAMS_FILE" ]; do
    sleep 60
done

echo "[OK] Optuna study finished!"

# Step 1: Update config.yaml with best parameters
echo "Step 1: Updating config.yaml..."
cd "$SRC_DIR"
$VENV_PYTHON prepare_full_training.py

if [ $? -eq 0 ]; then
    echo "[OK] Config updated."
else
    echo "[ERROR] Failed to update config.yaml."
    exit 1
fi

# Step 2: Run Full Training (20 epochs)
echo "Step 2: Starting Full Training (mode=memory)..."
# We run main.py which is the training entry point
# config.yaml now has num_epochs=20 and best params
$VENV_PYTHON main.py --mode memory

echo "=== Full Training Finished ==="
