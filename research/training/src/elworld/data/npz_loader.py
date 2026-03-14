"""
Shared NPZ data loader used by all Dataset classes.
Single source of truth for loading .npz gameplay files.
"""

import numpy as np
import tqdm
from pathlib import Path
from typing import Optional, Tuple, List


def load_npz_files(
    data_dir: str,
    max_files: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load all .npz gameplay files from a directory.
    
    Returns lists of arrays instead of concatenated arrays to prevent 
    VRAM/RAM spikes during initialization.
    """
    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if max_files is not None:
        npz_files = npz_files[:max_files]

    if verbose:
        print(f"[INFO] Loading {len(npz_files)} .npz file(s) from {data_dir} ...", flush=True)

    all_obs, all_act = [], []
    
    # Use tqdm for better visibility as requested
    file_iter = tqdm.tqdm(npz_files, desc="Loading Data") if verbose else npz_files

    for npz_file in file_iter:
        data = np.load(npz_file)
        obs = data["obs"]   # (N, H, W, C) uint8
        act = data["act"]   # (N, action_dim) uint8

        # Ensure channel-first layout for PyTorch
        if obs.ndim == 4 and obs.shape[-1] == 3:
            obs = obs.transpose(0, 3, 1, 2)   # (N, H, W, C) → (N, C, H, W)

        all_obs.append(obs)
        all_act.append(act)
        data.close()

    if verbose:
        total_frames = sum(len(o) for o in all_obs)
        print(f"  [OK] Frames loaded : {total_frames:,}", flush=True)

    return all_obs, all_act
