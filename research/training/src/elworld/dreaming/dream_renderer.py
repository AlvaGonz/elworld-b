"""
DreamRenderer: decodes VQ-VAE discrete token indices back to pixel frames.

Only loads the VQ-VAE *codebook* and *decoder* — the encoder is NOT loaded,
keeping VRAM usage minimal during dreaming (<500 MB for these components alone).
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from elworld.model.architectures.vision.vector_quantizer import VectorQuantizer
from elworld.model.architectures.vision.decoder import VisionDecoder


class DreamRenderer:
    """
    Lightweight wrapper that converts VQ-VAE token indices → RGB frames.
    Loads only the codebook embedding table and decoder (no encoder).

    Memory cost on RTX 3050:
        Codebook  : 512 × 64  = ~128 KB
        Decoder   : ~3 M params at fp32 = ~12 MB
        Total     : negligible — leaves full VRAM for MemoryModel
    """

    def __init__(self, vision_checkpoint: str, device: str = "cuda"):
        self.device = device
        checkpoint_path = Path(vision_checkpoint)

        with open(checkpoint_path / "config.json") as f:
            cfg = json.load(f)

        self.H = 24   # latent grid height  (192 / 8 = 24)
        self.W = 32   # latent grid width   (256 / 8 = 32)

        # --- Codebook (embedding table) -----------------------------------
        self.codebook = nn.Embedding(
            cfg["num_embedding"], cfg["latent_dim"]
        ).to(device)

        # --- Decoder only -------------------------------------------------
        self.decoder = VisionDecoder(
            input_channels=cfg["latent_dim"],
            num_hidden=cfg["num_hidden"],
            res_layer=cfg["res_layer"],
            res_hidden=cfg["res_hidden"],
            output_channels=cfg["input_channels"],
        ).to(device)

        # Load weights from full VQ-VAE checkpoint (extract subset)
        full_state = torch.load(
            checkpoint_path / "model.pth", map_location=device
        )
        # model.model.vq_layer.embeddings.weight  →  codebook
        self.codebook.weight.data.copy_(
            full_state["model.vq_layer.embeddings.weight"]
        )
        # model.model.decoder.*  →  decoder
        decoder_state = {
            k.replace("model.decoder.", ""): v
            for k, v in full_state.items()
            if k.startswith("model.decoder.")
        }
        self.decoder.load_state_dict(decoder_state)

        self.codebook.eval()
        self.decoder.eval()

        print(f"  [OK] DreamRenderer loaded (codebook + decoder only, device={device})")

    @torch.no_grad()
    def decode(self, token_indices: torch.Tensor) -> np.ndarray:
        """
        Decode flat token indices to a pixel frame.

        Args:
            token_indices: [H*W] or [1, H*W] — integer indices into VQ codebook

        Returns:
            frame: np.ndarray (H_px, W_px, 3) uint8, RGB
        """
        if token_indices.dim() == 1:
            token_indices = token_indices.unsqueeze(0)   # [1, H*W]

        # Look up embeddings: [1, H*W] → [1, H*W, embedding_dim]
        z_q = self.codebook(token_indices)               # [1, H*W, D]
        # Reshape to spatial: [1, D, H, W]
        z_q = z_q.view(1, self.H, self.W, -1).permute(0, 3, 1, 2).contiguous()

        # Decode to pixels
        x_recon = self.decoder(z_q)                      # [1, 3, H_px, W_px]
        x_recon = x_recon.squeeze(0).permute(1, 2, 0)   # [H_px, W_px, 3]
        x_recon = (x_recon.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return x_recon
