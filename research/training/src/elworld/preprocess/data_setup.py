"""
Data setup: creates DataLoaders for each training mode.
"""

import torch
from torch.utils.data import DataLoader

from elworld.data.gameplay_data import GameplayDataset
from elworld.data.memory_data import MemoryDataset


# ---------------------------------------------------------------------------
# Vision (VQ-VAE)
# ---------------------------------------------------------------------------

def setup_vision_data(data_path: str, vision_config: dict, general_config: dict) -> DataLoader:
    """
    Frame-level DataLoader for VQ-VAE training.

    RTX 3050 4 GB notes:
      - num_workers=4 with pin_memory keeps GPU saturated
      - batch_size is set in config (default 128)
    """
    dataset = GameplayDataset(
        data_dir=data_path,
        max_files=general_config.get("total_play"),
        add_noise=vision_config.get("add_noise", False)
    )

    loader = DataLoader(
        dataset,
        batch_size=vision_config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Vision DataLoader ready - {len(dataset):,} frames | "
          f"{len(loader)} batches | bs={vision_config['batch_size']} | noise={vision_config.get('add_noise', False)}")
    return loader


# ---------------------------------------------------------------------------
# Memory (MinGPT)
# ---------------------------------------------------------------------------

def setup_memory_data(
    data_path: str,
    memory_config: dict,
    vision_config: dict,
    general_config: dict,
) -> DataLoader:
    """
    Sequence DataLoader for MinGPT training.

    Steps:
      1. Load VQ-VAE checkpoint
      2. Encode all frames → discrete tokens (GPU, batch=128 for 4 GB VRAM)
      3. Free encoder from GPU
      4. Return DataLoader over token sequences
    """
    checkpoint_path = general_config.get("vision_checkpoint_path")
    if not checkpoint_path:
        checkpoint_path = "checkpoints/vision/best_model"

    dataset = MemoryDataset(
        data_dir=data_path,
        checkpoint_path=checkpoint_path,
        max_files=general_config.get("total_play"),
        context_frames=memory_config.get("context_frames", 4),
        encode_batch_size=128,   # Safe for RTX 3050 4 GB
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    loader = DataLoader(
        dataset,
        batch_size=memory_config.get("batch_size", 32),
        shuffle=True,
        num_workers=0,   # 0 = safe with CUDA tensors in dataset
        pin_memory=True,
    )

    print(f"Memory DataLoader ready — {len(dataset):,} sequences | "
          f"{len(loader)} batches | bs={memory_config.get('batch_size', 32)}")
    return loader


# ---------------------------------------------------------------------------
# Control (placeholder)
# ---------------------------------------------------------------------------

def setup_control_data(data_path: str, control_config: dict, general_config: dict):
    """Placeholder — control model not yet implemented."""
    print("[TODO] Control model data setup not yet implemented.")
    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def setup_data(
    mode: str,
    data_path: str,
    vision_config: dict = None,
    memory_config: dict = None,
    control_config: dict = None,
    general_config: dict = None,
):
    if mode == "vision":
        return setup_vision_data(data_path, vision_config, general_config)
    elif mode == "memory":
        return setup_memory_data(data_path, memory_config, vision_config, general_config)
    elif mode == "control":
        return setup_control_data(data_path, control_config, general_config)
    else:
        raise ValueError(f"Unknown data mode: '{mode}'. Expected: vision | memory | control")
