#!/bin/bash
set -e
export PYTHONIOENCODING="utf8"

# Navigate to project root
cd "$(dirname "$0")/.."

# Determine Python executable explicitly
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD="$(pwd)/.venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD="$(pwd)/.venv/bin/python"
else
    echo "[WARN] Virtual environment not found. Running with global python."
    PYTHON_CMD="python"
fi

# Navigate to source code
cd research/training/src

echo "[INFO] Extracting Vision Model (VQ-VAE) comparison video..."
$PYTHON_CMD -c "
from elworld.train.trainer import Trainer
t = Trainer('config.yaml', mode='demo_vision', device='cuda')
t.run()
"
