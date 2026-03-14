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

# Navigate to source code
cd research/training/src

echo "[INFO] Starting Vision Model (VQ-VAE) training..."
python -c "
from elworld.train.trainer import Trainer
t = Trainer('config.yaml', mode='vision', device='cuda')
t.run()
"
