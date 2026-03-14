# Memory Model — MinGPT Architecture

## Role

The Memory Model learns the **temporal dynamics** of the game world:
given visual tokens from the current frame + the player's action,
predict the token distribution for the next frame.

## Architecture: Minimal GPT Transformer

```
Input tokens  [B, T]   (VQ-VAE encoding_indices, flattened)
Action vector [B, T, 22]
  ↓
Token Embedding  [vocab_size=512, n_emb=64]
Position Embedding [block_size=1537, n_emb=64]
Action Projection  [22 → 64]   (added to token embedding)
  ↓
N × TransformerBlock
    ├─ Pre-LN LayerNorm
    ├─ Masked Multi-Head Attention (causal)
    ├─ Pre-LN LayerNorm
    └─ FeedForward (64 → 256 → 64) + GELU
  ↓
LayerNorm
  ↓
Linear Head [64 → 512]   (weight-tied to Token Embedding)
  ↓
Cross-Entropy loss against next-frame tokens
```

## Key Numbers

| Parameter | Value | Note |
|---|---|---|
| Vocabulary | 512 | Matches VQ-VAE codebook |
| Tokens per frame | 768 | 24 × 32 spatial grid |
| Block size | 1537 | ~2 frames context |
| n_emb | 64 | Embedding dim |
| Num layers | 4 | Reduced for 4 GB VRAM |
| Num heads | 4 | head_dim = 16 |
| Action dim | 22 | One-hot keyboard vector |
| Parameters | ~1.5 M | |

## Training

- **Task**: Next-frame token prediction (cross-entropy over 512 classes per token)
- **Optimizer**: AdamW, lr=3e-4
- **Scheduler**: ReduceLROnPlateau (patience=10, factor=0.5)
- **AMP**: Yes
- **Batch size**: 32 (RTX 3050 4 GB, ~2.5 GB peak VRAM)
- **Epochs**: 200

## Known Issues / Limitations

1. **Frame-independent prediction** — each batch element is a single
   frame-to-frame transition (`seq_len=2`), not a long rollout.
   This means the model doesn't condition on multiple preceding frames
   during training, limiting long-horizon coherence.

2. **Block size vs. context** — `block_size=1537` but during standard
   training we only use 2-frame windows. The full context is only used
   during dreaming.

3. **Evaluation is slow** — generating 768 tokens per frame and then
   decoding to pixels for every frame in a video is expensive.
   → **Solved by DreamingWorld** (real-time display, no disk write per frame).

## Checkpoints

```
checkpoints/memory/
  best_model/
    model.pth
    config.json
    training_info.json
```

## Improving the Model

To improve coherence without increasing VRAM:
- Increase `num_layers` from 4 → 6 (costs ~500 MB more VRAM)
- Increase `sequence_length` in MemoryDataset from 2 → 4
  (multi-frame context during training — needs block_size increase)
- Add perceptual loss term (VGG features on decoded frames)
