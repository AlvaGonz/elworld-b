# Dreaming World — Design Document

## Problem

After training the Memory Model, evaluating it visually required:
1. Encode all frames (slow, uses GPU encoder)
2. Generate tokens autoregressively (768 steps per frame)
3. Decode to pixels
4. Write a video to disk
5. Open the video to watch

This was too slow for interactive feedback. A 200-frame evaluation
video took ~10 minutes on an RTX 3050.

## Solution: Dreaming World

Inspired by [Ha & Schmidhuber (2018)](https://worldmodels.github.io/) and
the Minecraft dreaming world setup, **Dreaming World** runs the memory model
continuously in real-time, displaying frames directly via OpenCV.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│              DreamingWorld (continuous loop)           │
│                                                        │
│  Keyboard ──→ action_vector [22]                       │
│                    ↓                                   │
│  Context [1, T] ──→ MemoryModel.generate(768 steps)   │
│                    ↓                                   │
│  new_tokens [768] ──→ DreamRenderer.decode()           │
│       (codebook lookup + VisionDecoder only)           │
│                    ↓                                   │
│  frame [192, 256, 3] ──→ cv2.imshow (real-time)       │
│                    ↓                                   │
│  Roll context window (discard oldest frame tokens)     │
│                    ↓  (repeat, ESC to quit)            │
└────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Decoder-only loading (DreamRenderer)
The VQ-VAE **encoder is not loaded** during dreaming. Only the
codebook embedding table and decoder are loaded (~12 MB VRAM).
The encoder (~300 MB) is only needed for offline evaluation.

### 2. Rolling context window
The context tensor grows by 768 tokens per step and is truncated to
`block_size - 768` tokens, so it runs indefinitely without OOM.

### 3. Live keyboard input
Uses the same `KEYS_TO_LOG` list as `record_gameplay.py`.
The player can press keys and the dreamed world responds.

### 4. AMP inference
`torch.amp.autocast('cuda')` is used for token generation,
halving activation memory during the forward pass.

### 5. Optional disk save
Set `save_path` in `dreaming_config` to also write an `.mp4` file
while displaying in real-time (does not slow down display loop).

## VRAM Budget (RTX 3050 Laptop 4 GB)

| Component | VRAM |
|---|---|
| MemoryModel (4L, 64-dim) | ~200 MB |
| VQ-VAE codebook + decoder | ~12 MB |
| Context tensor (1537 tokens, int64) | ~12 KB |
| AMP activations | ~300 MB |
| **Total** | **< 600 MB** |

This leaves >3 GB free, meaning the game can run simultaneously
without VRAM conflict — critical for auto-farming use.

## Running

```bash
# Interactive dreaming (ESC to quit)
./research/scripts/dream.sh

# Or directly via Python
# Edit config.yaml → dreaming_config first
cd research/training/src
python main.py   # mode = "dreaming"
```

## Offline Evaluation

`elworld/utils/real_vs_memory.py` provides `evaluate_memory()` for
a side-by-side video comparison of real vs. predicted frames —
use this to benchmark memory model quality after training.
