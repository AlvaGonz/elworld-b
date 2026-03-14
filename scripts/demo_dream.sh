#!/bin/bash
set -e
export PYTHONIOENCODING="utf8"

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -f ".venv/Scripts/activate" ]; then
    source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    echo "[WARN] Virtual environment not found. Running with global python."
fi

# Determine Python executable explicitly
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD="$(pwd)/.venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD="$(pwd)/.venv/bin/python"
else
    PYTHON_CMD="python"
fi

cd research/training/src

echo "[INFO] Starting Interactive Dreaming World (Press ESC to quit)..."
$PYTHON_CMD -c "
from elworld.train.trainer import Trainer
t = Trainer('config.yaml', mode='dreaming', device='cuda')
t.run()
"
