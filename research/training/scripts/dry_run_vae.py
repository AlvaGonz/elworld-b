import torch
import sys
import os
from pathlib import Path

# Thêm src vào path
sys.path.append(str(Path(r"c:\Projects\elsworld\research\training\src")))

from elworld.train.trainer import Trainer

if __name__ == "__main__":
    # Configure for dry run
    mode = "vision"
    config_path = r"c:\Projects\elsworld\research\training\src\config.yaml"
    device = "cuda"
    
    # We want to test only 1 iteration if possible, but the trainer trains for epochs.
    # Let's just run it and see if it crashes at start.
    print("Starting VAE Dry Run...")
    try:
        trainer = Trainer(config_path=config_path, mode=mode, device=device)
        # Monkey patch num_epochs to 1 for dry run
        trainer.vision_config['num_epochs'] = 1
        # Set a small limit for speed if session allows
        trainer.run()
        print("\n[SUCCESS] Dry run completed without OOM!")
    except Exception as e:
        print(f"\n[ERROR] Dry run failed: {e}")
        import traceback
        traceback.print_exc()
