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

# Navigate to data collection
cd research/data_collection

echo "[INFO] Reshaping raw 800x600 data into 256x192 patches..."
python reshape_data.py
