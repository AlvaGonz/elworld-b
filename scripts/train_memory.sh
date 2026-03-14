#!/bin/bash
set -e
export PYTHONIOENCODING="utf8"

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment and lock Python executable
PYTHON_CMD="python"
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD="$PWD/.venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD="$PWD/.venv/bin/python"
fi

# Navigate to source code
cd research/training/src

echo "[INFO] Starting Memory Model (Parallel Spatial CNN) training..."
$PYTHON_CMD -c "
from elworld.train.trainer import Trainer
t = Trainer('config.yaml', mode='memory', device='cuda')
t.run()
"
