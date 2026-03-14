s# Elworld — Project Context

## What is Elworld?

Elworld is a **World Model + Reinforcement Learning** system that learns to play
[Elsword](https://elsword.koget.com/) autonomously by watching human gameplay recordings.
The ultimate goal is **unsupervised auto-farming**: the trained agent runs continuously
in the background while the user does other things.

## Design Philosophy

| Principle | Why |
|---|---|
| Learn from pixels, not game internals | No API access; game is a black box |
| Compress before reasoning | Raw 800×600 frames → 64-dim latent codes (VQ-VAE) |
| Temporal reasoning in latent space | MDN-RNN / MinGPT over discrete tokens |
| Train offline, deploy online | All training done on recorded data — no online RL yet |
| Latent-space dreaming | Evaluate memory quality without touching the game |

## Current Status  *(as of March 2026)*

| Component | Status | Notes |
|---|---|---|
| Data collection | ✅ Done | 5 runs × 3.9 k frames, 256×192 RGB + action vectors |
| Vision Model (VQ-VAE) | ✅ Trained | Excellent reconstruction quality |
| Memory Model (MinGPT) | 🔄 Training | Loss converging; dreaming eval added |
| Control Model | ⬜ Not started | Depends on stable memory model |
| Dreaming World | ✅ Implemented | Real-time OpenCV loop |

## Hardware Target

> **RTX 3050 Laptop GPU — 4 GB VRAM**
>
> All configs are tuned to stay under 3.5 GB peak VRAM so the game can still
> run in the background during auto-farming inference.

| Stage | Peak VRAM |
|---|---|
| Vision training (bs=128) | ~1.5 GB |
| Memory training (bs=32) | ~2.5 GB |
| Dreaming World (inference) | ~1.5 GB |
| Auto-farming inference | < 2 GB target |

## Data Format

Each `.npz` gameplay file contains:
- `obs`: `(N, 256, 192, 3)` uint8 — RGB frames at 20 FPS
- `act`: `(N, 22)` uint8 — one-hot keyboard state per frame

22 tracked keys: `f8 up down left right z x 1 3 q w e r t a s d c f enter ctrl esc`

## Pipeline Overview

```
Record gameplay → .npz
        ↓
VQ-VAE (Vision Model) → discrete tokens [N, 24×32]
        ↓
MinGPT (Memory Model) → next-token prediction
        ↓
DreamingWorld          → real-time action-conditioned video
        ↓ (future)
Controller             → action selection in latent space
```

## Repository Layout

```
elsworld/
├── docs/                        ← ★ Architecture docs (this folder)
├── research/
│   ├── data_collection/         ← Gameplay capture tools
│   ├── scripts/                 ← train_vision.sh, train_memory.sh, dream.sh
│   └── training/src/
│       ├── config.yaml          ← All hyperparameters
│       ├── main.py              ← Entry point
│       └── elworld/
│           ├── data/            ← Datasets + shared npz_loader
│           ├── model/           ← VisionModel, MemoryModel, architectures/
│           ├── train/           ← Trainers + pipeline
│           ├── dreaming/        ← DreamingWorld + DreamRenderer
│           ├── preprocess/      ← data_setup.py
│           └── utils/           ← eval scripts
├── data_collection/             ← Raw .npz gameplay data
└── product/                     ← Deployment (WIP)
```
