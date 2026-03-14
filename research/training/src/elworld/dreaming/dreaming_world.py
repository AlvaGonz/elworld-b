"""
DreamingWorld: continuous world-model dreaming loop for the Memory Model.

Design (Minecraft-style dreaming):
  ┌─────────────────────────────────────────────────────┐
  │  seed_tokens (from real data or random)             │
  │        ↓                                            │
  │  MinGPT.generate()  ← action (keyboard/scripted)   │
  │        ↓                                            │
  │   768 new tokens  →  DreamRenderer.decode()         │
  │        ↓                                            │
  │   RGB frame  →  cv2.imshow  (+ optional disk write) │
  │        ↓                                            │
  │   Roll context window (drop oldest frame)           │
  │        ↓ (loop forever)                             │
  └─────────────────────────────────────────────────────┘

Key design choices for RTX 3050 4 GB:
  - VQ-VAE *encoder* is NOT loaded — save ~300 MB VRAM
  - ParallelMemoryModel runs in O(1) time (~10ms) per frame -> true 60+ FPS
  - Context window slides by 1 frame exactly (context_frames)
  - Output is [B, 1, 24, 32] which is flattened to 768 tokens for rendering
"""

import json
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

import keyboard   # same dep as record_gameplay.py

from elworld.model.memory import MemoryModel
from elworld.dreaming.dream_renderer import DreamRenderer
from elworld.data.npz_loader import load_npz_files

# Keys logged during recording — same order as KEYS_TO_LOG in record_gameplay.py
KEYS_TO_LOG = [
    'f8', 'up', 'down', 'left', 'right', 'z', 'x',
    '1', '3',
    'q', 'w', 'e', 'r', 't',
    'a', 's', 'd', 'c', 'f',
    'enter', 'ctrl', 'esc',
]
TOKENS_PER_FRAME = 24 * 32   # 768


def _get_action_vector() -> torch.Tensor:
    """Read current keyboard state into a float action vector."""
    vec = [1.0 if keyboard.is_pressed(k) else 0.0 for k in KEYS_TO_LOG]
    return torch.tensor(vec, dtype=torch.float32)


class DreamingWorld:
    """
    Runs an infinite dreaming loop using the trained Memory + Vision decoder.

    Usage:
        world = DreamingWorld(memory_ckpt, vision_ckpt, memory_config, device)
        world.run(temperature=1.0, top_k=50)   # Press ESC to quit
    """

    def __init__(
        self,
        memory_checkpoint: str,
        vision_checkpoint: str,
        memory_config: dict,
        device,
    ):
        self.device = device
        self.memory_config = memory_config
        self.context_frames = memory_config.get("context_frames", 4)

        print("\nInitialising DreamingWorld ...")

        # --- Memory model -----------------------------------------------------
        print("  Loading Parallel Memory Model ...")
        self.memory = MemoryModel(
            vocab_size=memory_config.get('vocab_size', 512),
            context_frames=self.context_frames,
            embed_dim=memory_config.get('embed_dim', 64),
            hidden_dim=memory_config.get('hidden_dim', 256),
            num_res_blocks=memory_config.get('num_res_blocks', 6),
            action_dim=memory_config.get('action_dim', 22)
        ).to(device)

        ckpt = Path(memory_checkpoint)
        if not (ckpt / "model.pth").exists():
            raise FileNotFoundError(f"Memory checkpoint not found: {ckpt}")

        if memory_config.get("use_qat", False):
            print("  [QAT] Initializing Quantization-Aware Training structures for dreaming...")
            self.memory.prepare_qat()

        self.memory.load_state_dict(
            torch.load(ckpt / "model.pth", map_location=device)
        )
        self.memory.eval()
        params = self.memory.get_num_params()
        print(f"  [OK] MemoryModel  ({params:,} params)")

        # --- Vision decoder ---------------------------------------------------
        print("  Loading Vision Decoder (renderer) ...")
        self.renderer = DreamRenderer(vision_checkpoint, device=str(device))

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"  GPU VRAM in use after load: {used:.0f} / {total:.0f} MB")

    # ------------------------------------------------------------------

    def _build_seed(self, seed_file: str, seed_frames: int) -> torch.Tensor:
        """
        Build initial token context.
        If seed_file is provided, encode first `seed_frames` real frames.
        Otherwise start from a random token sequence.
        """
        if seed_file and Path(seed_file).exists():
            print(f"  Seeding from {seed_file} ({seed_frames} frames) ...")
            obs, _ = load_npz_files(seed_file, max_files=1, verbose=False)
            # We only need tokens, not frames — use the renderer's codebook
            # but we don't have encoder. So we use random seed as fallback
            # when no encoder is present. This is intentional (encoder is expensive).
            print("  (Note: no encoder loaded; using random seed for safety)")

        # Random seed: spatial grid
        seed = torch.randint(
            0, self.memory_config.get("vocab_size", 512),
            (1, self.context_frames, 24, 32),
            dtype=torch.long,
            device=self.device,
        )
        return seed   # [1, context_frames, H, W]

    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        seed_file: str = None,
        seed_frames: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        save_path: str = None,
        display_scale: int = 2,
        fps_target: float = 20.0,
    ):
        """
        Main dreaming loop.

        Args:
            seed_file:     Path to .npz file for real-frame seeding (None = random)
            seed_frames:   Number of seed frames to use
            temperature:   Sampling temperature for token generation
            top_k:         Top-k nucleus sampling (None = full softmax)
            save_path:     If set, also write a side-by-side comparison video here
            display_scale: OpenCV window scale factor (default 2 = 512x384)
            fps_target:    Target FPS for display loop
        """
        print("\n" + "="*60)
        print("  DREAMING WORLD  —  Press ESC to quit")
        print("  Controls: same keys as gameplay recording")
        print("="*60 + "\n")

        context = self._build_seed(seed_file, seed_frames)  # [1, T]
        frame_time = 1.0 / fps_target

        # Optional video writer
        writer = None
        if save_path:
            H_px, W_px = 192, 256
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps_target, (W_px, H_px))
            print(f"  Also saving to {save_path}")

        use_amp = torch.cuda.is_available()
        frame_idx = 0

        while True:
            t0 = time.time()

            # --- Quit on ESC ------------------------------------------------
            if keyboard.is_pressed("esc"):
                print("\nESC pressed — exiting dreaming world.")
                break

            # --- Capture action vector --------------------------------------
            action = _get_action_vector().to(self.device).unsqueeze(0)  # [1, 22]

            # --- Generate next frame tokens ---------------------------------
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Generates EXACTLY 1 frame in O(1) time
                new_tokens = self.memory.generate(
                    start_tokens=context,
                    actions=action,
                    temperature=temperature,
                    top_k=top_k,
                )   # [1, 1, 24, 32]

            # Flatten to 1D array for the renderer
            generated_frame_tokens = new_tokens[0, 0].flatten()  # [768]

            # --- Roll context window ----------------------------------------
            # Drop earliest frame, append new frame
            context = torch.cat([context[:, 1:], new_tokens], dim=1) # [1, context_frames, 24, 32]

            # --- Decode tokens → RGB frame ----------------------------------
            frame_rgb = self.renderer.decode(generated_frame_tokens)   # (H, W, 3)

            # --- Display via OpenCV -----------------------------------------
            if display_scale != 1:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (256 * display_scale, 192 * display_scale),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Convert RGB → BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame_bgr,
                f"DREAMING #{frame_idx}  temp={temperature:.2f}  k={top_k}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            cv2.imshow("Elworld — DreamingWorld", frame_bgr)

            if writer:
                writer.write(cv2.resize(frame_bgr, (256, 192)))

            # waitKey(1) needed for cv2.imshow to refresh
            if cv2.waitKey(1) & 0xFF == 27:   # ESC via OpenCV too
                break

            # --- Rate-limit to target FPS -----------------------------------
            elapsed = time.time() - t0
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

            frame_idx += 1

        # Cleanup
        cv2.destroyAllWindows()
        if writer:
            writer.release()
            print(f"  [OK] Dream video saved to {save_path}")

        print(f"\n  Dreamed {frame_idx} frames total.\n")
