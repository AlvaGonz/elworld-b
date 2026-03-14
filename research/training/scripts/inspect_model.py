import torch
import sys
from pathlib import Path

# Thêm src vào path
sys.path.append(str(Path(r"c:\Projects\elsworld\research\training\src")))

from elworld.model.vision import VisionModel

def inspect_model_structure():
    model = VisionModel(
        num_hidden=256,
        res_layer=4,
        res_hidden=64,
        input_channels=3,
        num_embedding=2048,
        embedding_dim=256,
        commitment_cost=0.25
    )
    print("Model structure keys:")
    for name, _ in model.named_parameters():
        print(name)

if __name__ == "__main__":
    inspect_model_structure()
