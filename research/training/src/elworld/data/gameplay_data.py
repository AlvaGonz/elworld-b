import torch
import numpy as np
from torch.utils.data import Dataset

from elworld.data.npz_loader import load_npz_files


class GameplayDataset(Dataset):
    """
    Individual-frame dataset for VQ-VAE training.
    Uses lazy indexing into a list of arrays to minimize memory spikes.
    """

    def __init__(self, data_dir: str, max_files: int = None, transform=None, add_noise: bool = False):
        self.transform = transform
        self.add_noise = add_noise
        self.obs_list, self.act_list = load_npz_files(data_dir, max_files)
        
        # Pre-calculate lengths for indexing
        self.lengths = [len(o) for o in self.obs_list]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_len = self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 else 0

    def __len__(self):
        return self.total_len

    def _get_local_index(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError("Index out of bounds")
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        return file_idx, local_idx

    def __getitem__(self, idx):
        file_idx, local_idx = self._get_local_index(idx)
        
        obs = torch.from_numpy(self.obs_list[file_idx][local_idx]).float()
        
        # Add slight Gaussian noise during training to prevent overfit on static UI
        if self.add_noise:
            noise = torch.randn_like(obs) * 2.0 # 2 units of noise (in 0-255 range)
            obs = (obs + noise).clamp(0, 255)

        act = torch.from_numpy(self.act_list[file_idx][local_idx]).float()
        if self.transform:
            obs = self.transform(obs)
        return {"observation": obs, "action": act}


class SequenceGameplayDataset(Dataset):
    """
    Sequential dataset for RNN / control model training.
    Returns overlapping windows of `sequence_length` consecutive frames.
    Note: Current implementation assumes sequences don't cross file boundaries for simplicity,
    or we can concatenate if sequences are short. Given memory constraints, 
    we'll handle it file-by-file.
    """

    def __init__(self, data_dir: str, sequence_length: int = 32,
                 max_files: int = None, transform=None):
        self.transform = transform
        self.sequence_length = sequence_length
        self.obs_list, self.act_list = load_npz_files(data_dir, max_files)
        
        # We only take sequences that fit within each file
        self.valid_indices = [] # List of (file_idx, start_idx)
        for f_idx, obs in enumerate(self.obs_list):
            num_seq = len(obs) - sequence_length + 1
            if num_seq > 0:
                for s_idx in range(num_seq):
                    self.valid_indices.append((f_idx, s_idx))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_idx, start_idx = self.valid_indices[idx]
        
        obs_seq = self.obs_list[file_idx][start_idx: start_idx + self.sequence_length]
        act_seq = self.act_list[file_idx][start_idx: start_idx + self.sequence_length]

        if self.transform:
            obs_seq_t = torch.stack([self.transform(torch.from_numpy(o).float()) for o in obs_seq])
        else:
            obs_seq_t = torch.from_numpy(obs_seq).float()

        return {
            "observation": obs_seq_t,
            "action": torch.from_numpy(act_seq).float(),
        }
