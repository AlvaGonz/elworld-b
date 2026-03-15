#!/bin/bash

# Path setup
BASE_DIR="/home/viethq/Projects/elworld"
SRC_DIR="$BASE_DIR/research/training/src"
VENV_PYTHON="$BASE_DIR/venv/bin/python"

echo "=== Sequential Training Orchestrator (Vision -> Memory) ==="
cd "$SRC_DIR"

# Step 1: Train Vision Model (20 epochs)
echo "[STEP 1/2] Starting Full Vision Training (20 epochs)..."
$VENV_PYTHON main.py --mode vision

if [ $? -eq 0 ]; then
    echo "[OK] Vision Training Finished Successfully."
else
    echo "[ERROR] Vision Training Failed."
    exit 1
fi

# Step 2: Train Memory Model (20 epochs)
echo "[STEP 2/2] Starting Full Memory Training (20 epochs)..."
# Note: config.yaml was already updated with best memory params by prepare_full_training.py earlier
$VENV_PYTHON main.py --mode memory

if [ $? -eq 0 ]; then
    echo "[OK] Memory Training Finished Successfully."
else
    echo "[ERROR] Memory Training Failed."
    exit 1
fi

echo "=== All Training Tasks Completed Successfully! ==="
