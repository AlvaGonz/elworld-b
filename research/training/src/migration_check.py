import os
import torch
import json
from pathlib import Path

def check_env():
    print("=== Remote Environment Check ===")
    
    # 1. GPU Check
    if torch.cuda.is_available():
        print(f"[OK] GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("[ERROR] CUDA not available. Training will be slow/impossible.")

    # 2. Source Code Check
    src_dir = Path(".")
    if (src_dir / "config.yaml").exists():
        print("[OK] Source code (config.yaml) is present.")
    else:
        print("[ERROR] Source code missing in current directory.")

    # 3. Brain Check
    print("\n=== Brain & Artifact Check ===")
    # Look for the current directory's task.md if any
    task_file = Path("task.md") 
    if task_file.exists():
        print("[INFO] Found task.md in project root.")
    
    print("\n[INSTRUCTION]")
    print("1. Start a new chat with Antigravity on this server.")
    print("2. Ask: 'What is your current brain directory path?'")
    print("3. Copy contents of 'brain_migration.tar.gz' to that path.")
    print("4. Restart chat or ask me to 'Read task.md in brain' to resume.")

if __name__ == "__main__":
    check_env()
