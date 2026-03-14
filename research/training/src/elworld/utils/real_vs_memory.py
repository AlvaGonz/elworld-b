"""
real_vs_memory.py: offline evaluation of Memory Model quality.

Generates a side-by-side video comparing:
  Left  — ground truth frame (decoded from real VQ-VAE tokens)
  Right — memory model prediction (generated from previous context)

This is the offline counterpart of DreamingWorld.
"""

import json
import numpy as np
import cv2
import torch
from pathlib import Path

from elworld.model.memory import MemoryModel
from elworld.dreaming.dream_renderer import DreamRenderer
from elworld.data.npz_loader import load_npz_files
from elworld.model.vision import VisionModel

TOKENS_PER_FRAME = 24 * 32  # 768


def _load_memory_model(checkpoint_path: str, memory_config: dict, device: str) -> MemoryModel:
    model = MemoryModel(
        vocab_size=memory_config.get('vocab_size', 512),
        context_frames=memory_config.get('context_frames', 4),
        embed_dim=memory_config.get('embed_dim', 64),
        hidden_dim=memory_config.get('hidden_dim', 256),
        num_res_blocks=memory_config.get('num_res_blocks', 6),
        action_dim=memory_config.get('action_dim', 22)
    ).to(device)
    
    if memory_config.get("use_qat", False):
        print("  [QAT] Initializing Quantization-Aware Training structures for eval...")
        model.prepare_qat()
        
    model.load_state_dict(torch.load(checkpoint_path + "/model.pth", map_location=device))
    model.eval()
    with open(checkpoint_path + "/config.json") as f:
        cfg = json.load(f)
    return model


def _encode_frames(vision_ckpt: str, observations: np.ndarray, device: str, batch_size: int = 128) -> torch.Tensor:
    """Encode real frames → token indices using full VQ-VAE."""
    with open(Path(vision_ckpt) / "config.json") as f:
        cfg = json.load(f)

    vqvae = VisionModel(
        num_hidden=cfg["num_hidden"], res_layer=cfg["res_layer"],
        res_hidden=cfg["res_hidden"], input_channels=cfg["input_channels"],
        num_embedding=cfg["num_embedding"], embedding_dim=cfg["latent_dim"],
        commitment_cost=cfg["commitment_cost"],
    ).to(device)
    vqvae.load_state_dict(torch.load(Path(vision_ckpt) / "model.pth", map_location=device))
    vqvae.eval()

    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(observations), batch_size):
            batch = torch.from_numpy(observations[i:i+batch_size]).float().to(device)
            if batch.max() > 1.0:
                batch = batch / 255.0
            out = vqvae(batch)
            tokens = out["encoding_indices"] # [B, 24, 32]
            all_tokens.append(tokens.cpu())

    del vqvae
    torch.cuda.empty_cache()
    return torch.cat(all_tokens, dim=0) # [N, 24, 32]


def evaluate_memory(
    data_path: str,
    vision_checkpoint: str,
    memory_checkpoint: str,
    memory_config: dict,
    output_path: str = "memory_eval.mp4",
    max_frames: int = 200,
    fps: int = 20,
    device: str = "cuda",
):
    """
    Create side-by-side comparison: real frame vs. memory model prediction.

    Args:
        data_path:         .npz file or directory to seed from
        vision_checkpoint: Path to VQ-VAE best_model folder
        memory_checkpoint: Path to Memory best_model folder
        memory_config:     memory_config dict from config.yaml
        output_path:       Output .mp4 path
        max_frames:        Number of frames to compare
        fps:               Output video FPS
        device:            torch device string
    """
    print(f"\n{'='*60}\n  Memory Model Evaluation\n{'='*60}")

    # Load data
    obs, acts = load_npz_files(data_path, max_files=1)
    obs = obs[:max_frames]
    acts = acts[:max_frames]

    print(f"  Loaded {len(obs)} frames for evaluation")

    # Encode to tokens (needs full encoder)
    print("  Encoding frames with VQ-VAE ...")
    real_tokens = _encode_frames(vision_checkpoint, obs, device)  # [N, 24, 32]

    # Load memory model
    print("  Loading Memory Model ...")
    memory = _load_memory_model(memory_checkpoint, memory_config, device)

    # Load renderer (decoder only)
    print("  Loading Vision Decoder ...")
    renderer = DreamRenderer(vision_checkpoint, device=device)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (256 * 2, 192))
    assert writer.isOpened(), f"Failed to open video writer for {output_path}"

    # context_frames matches model input (e.g., 4)
    context_frames = memory_config.get("context_frames", 4)
    context = real_tokens[:context_frames].unsqueeze(0).to(device)  # [1, context_frames, 24, 32]
    print(f"  Generating {max_frames} comparison frames ...")

    with torch.no_grad():
        for i in range(context_frames, min(len(real_tokens), max_frames)):
            action = torch.from_numpy(acts[i-1]).float().to(device).unsqueeze(0)  # [1, 22]

            # Generate exactly next frame in O(1)
            generated = memory.generate(
                start_tokens=context,
                actions=action,
                temperature=1.0,
                top_k=50,
            ) # [1, 1, 24, 32]
            
            # Flatten spatial grid for discrete VQ-VAE decoder
            pred_tokens = generated[0, 0].flatten()

            # Decode both
            real_frame = renderer.decode(real_tokens[i].flatten().to(device))
            pred_frame = renderer.decode(pred_tokens)

            # Side-by-side
            combined = np.ascontiguousarray(np.concatenate([real_frame, pred_frame], axis=1))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "REAL",        (8, 22),       font, 0.6, (255,255,255), 1)
            cv2.putText(combined, "MEMORY MODEL", (256+8, 22),  font, 0.6, (255,255,255), 1)
            cv2.putText(combined, f"Frame {i}/{max_frames}", (200, 22), font, 0.4, (200,200,200), 1)

            writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            # Roll context window
            context = torch.cat([context[:, 1:], generated], dim=1)  # [1, context_frames, 24, 32]

            if i % 50 == 0:
                print(f"  Processed {i}/{max_frames} frames ...")

    writer.release()
    print(f"\n  [OK] Evaluation video saved: {output_path}")
    print(f"  Frames: {max_frames} | FPS: {fps}\n")
