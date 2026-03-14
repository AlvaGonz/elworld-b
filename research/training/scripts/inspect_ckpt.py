import torch
from pathlib import Path

def inspect_checkpoint(path):
    print(f"Inspecting: {path}")
    data = torch.load(path, map_location='cpu')
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Num keys: {len(keys)}")
        print(f"First 10 keys: {keys[:10]}")
    else:
        print("Data is not a dictionary.")

if __name__ == "__main__":
    ckpt_path = r"c:\Projects\elsworld\research\training\src\checkpoints\vision\vision_model_checkpoint_31\model.pth"
    inspect_checkpoint(ckpt_path)
