#!/bin/bash
set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "========================================================="
echo " Elworld AI Environment Setup (Linux / Git Bash)"
echo "========================================================="

echo "1. Creating Python Virtual Environment (.venv)..."
python -m venv .venv

echo "2. Activating virtual environment..."
if [ -f ".venv/Scripts/activate" ]; then
    source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    echo "[ERROR] Failed to activate virtual environment."
    exit 1
fi

echo "3. Upgrading pip..."
python -m pip install --upgrade pip

echo "4. Installing PyTorch with CUDA 12.4 support (this may take a while)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "5. Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

echo "========================================================="
echo " Setup Complete!"
echo " The .venv will be automatically activated by other scripts."
echo "========================================================="
