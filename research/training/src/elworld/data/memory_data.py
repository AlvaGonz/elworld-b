"""
MemoryDataset: encodes gameplay frames to VQ-VAE tokens for memory model training.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from elworld.data.npz_loader import load_npz_files
from elworld.model.vision import VisionModel


class MemoryDataset(Dataset):
    """
    Dataset for memory (MinGPT) training.

    Pipeline:
        1. Load .npz gameplay files via shared loader
        2. Encode all frames → VQ-VAE discrete tokens (done once, offline)
        3. Free the encoder from GPU to save VRAM
        4. Return (input_tokens, actions, target_tokens) sequences for next-frame prediction
    """

    def __init__(
        self,
        data_dir: str,
        checkpoint_path: str,
        max_files: int = None,
        context_frames: int = 4,
        encode_batch_size: int = 128,
        device: str = "cuda",
    ):
        """
        Args:
            data_dir:          Directory containing .npz gameplay files
            checkpoint_path:   Path to trained VQ-VAE checkpoint folder
            max_files:         Maximum .npz files to load (None = all)
            context_frames:    Number of past frames for parallel prediction (default 4)
            encode_batch_size: Batch size used during offline encoding
                               (lower value = safer for 4 GB VRAM)
            device:            Device for encoding pass
        """
        self.context_frames = context_frames
        self.device = device

        # --- Step 1: Load VQ-VAE ------------------------------------------------
        print("\n[1/3] Loading VQ-VAE encoder ...")
        vqvae = self._load_vqvae(checkpoint_path, device)
        vqvae.eval()

        # --- Step 2: Load gameplay data -----------------------------------------
        print("\n[2/3] Loading gameplay data ...", flush=True)
        obs_list, act_list = load_npz_files(data_dir, max_files)

        # --- Step 3: Encode frames → discrete tokens ----------------------------
        print("\n[3/3] Encoding frames to visual tokens ...", flush=True)
        self.visual_tokens = self._encode_frames(vqvae, obs_list, encode_batch_size)
        
        # Flatten actions (they are small, so concatenation is safe here)
        self.actions = torch.from_numpy(np.concatenate(act_list, axis=0)).float()

        # Free encoder from GPU immediately
        del vqvae
        if device == "cuda":
            torch.cuda.empty_cache()
            print("  [OK] VQ-VAE freed from GPU VRAM", flush=True)

        self.num_sequences = len(self.visual_tokens) - context_frames
        print(f"\n[OK] MemoryDataset ready:", flush=True)
        print(f"  Total sequences  : {self.num_sequences:,}", flush=True)
        print(f"  Grid shape       : {self.visual_tokens.shape[1:]}", flush=True)
        print(f"  Context frames   : {context_frames}", flush=True)

    # ------------------------------------------------------------------
    def _load_vqvae(self, checkpoint_path: str, device: str) -> VisionModel:
        checkpoint_path = Path(checkpoint_path)
        with open(checkpoint_path / "config.json") as f:
            cfg = json.load(f)

        model = VisionModel(
            num_hidden=cfg["num_hidden"],
            res_layer=cfg["res_layer"],
            res_hidden=cfg["res_hidden"],
            input_channels=cfg["input_channels"],
            num_embedding=cfg["num_embedding"],
            embedding_dim=cfg["latent_dim"],
            commitment_cost=cfg["commitment_cost"],
        ).to(device)

        model.load_state_dict(
            torch.load(checkpoint_path / "model.pth", map_location=device)
        )
        print(f"  [OK] VQ-VAE loaded from {checkpoint_path}", flush=True)
        return model

    def _encode_frames(self, model: VisionModel, obs_list: list, batch_size: int) -> torch.Tensor:
        """Offline encoding: obs_list (list of arrays) → tokens (N, H, W)."""
        all_tokens = []
        total_frames = sum(len(o) for o in obs_list)
        processed_frames = 0

        with torch.no_grad():
            for obs in obs_list:
                N = len(obs)
                for i in range(0, N, batch_size):
                    batch_np = obs[i: i + batch_size]
                    batch = torch.from_numpy(batch_np).float().to(self.device)

                    if batch.max() > 1.0:
                        batch = batch / 255.0

                    out = model(batch)
                    tokens = out["encoding_indices"]         # [B, H, W]
                    all_tokens.append(tokens.cpu())

                processed_frames += N
                print(f"  Encoded {processed_frames}/{total_frames} frames ...", flush=True)

        tokens_all = torch.cat(all_tokens, dim=0)  # [total_frames, H, W]
        print(f"  [OK] Encoded {total_frames:,} frames -> {tokens_all.shape}", flush=True)
        return tokens_all

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns spatial token grids for the context window and the target frame.
        """
        # [context_frames, H, W]
        input_tokens = self.visual_tokens[idx: idx + self.context_frames]
        
        # [H, W] - The exactly next frame following the context
        target_tokens = self.visual_tokens[idx + self.context_frames]
        
        # The action taken at the end of the context window to reach the target frame
        # [action_dim]
        action = self.actions[idx + self.context_frames - 1]
        
        return {
            "input_tokens":  input_tokens,   # [context_frames, H, W]
            "actions":       action,         # [action_dim]
            "target_tokens": target_tokens,  # [H, W]
        }
