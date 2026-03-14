# Vision Model — VQ-VAE Architecture

## Role

The Vision Model compresses raw 256×192 RGB frames into **discrete latent codes**
that the Memory Model and Controller can reason over efficiently.

## Architecture: Vector-Quantized VAE (VQ-VAE)

```
Input  [B, 3, 192, 256]
  ↓
VisionEncoder (Conv × 3 + ResStack × 2)
  ↓  [B, 128, 24, 32]
pre_vq_conv (1×1 Conv)
  ↓  [B, 64, 24, 32]
VectorQuantizer  (512-entry codebook, dim=64)
  ↓  [B, 64, 24, 32]  (quantized)  +  [B, 24, 32] (index map)
VisionDecoder (Transposed Conv × 3 + ResStack × 2)
  ↓
Output [B, 3, 192, 256]  (reconstruction)
```

## Key Numbers

| Parameter | Value |
|---|---|
| Input resolution | 256 × 192 px (W × H) |
| Encoder downscale | 8× (3 strided convs) |
| Latent grid | 32 × 24 = **768 tokens** |
| Codebook size | **512** entries |
| Embedding dim | **64** |
| Commitment cost | 0.25 |
| Hidden channels | 128 |
| Residual layers | 2 |

## Training

- **Loss**: `MSE(recon, input)` + VQ-loss (codebook + commitment)
- **Optimizer**: AdamW, lr=3e-4
- **Scheduler**: ReduceLROnPlateau (patience=20, factor=0.5)
- **AMP**: Yes (fp16 forward, fp32 master weights)
- **Batch size**: 128 (safe on RTX 3050 4 GB, ~1.5 GB peak VRAM)
- **Epochs**: 500
- **Status**: ✅ Converged — excellent reconstruction quality

## Checkpoints

```
checkpoints/vision/
  best_model/
    model.pth          ← weights
    config.json        ← model config (latent_dim, num_embedding, …)
    training_info.json ← loss history
  vision_model_checkpoint_N/  ← per-epoch checkpoints
```

## Evaluation

Run `main.py` with `mode = "extract_video"` to create a side-by-side
comparison: `[actual frame | VQ-VAE reconstruction]`.

## Notes

- The encoder is **not loaded** during dreaming inference (`DreamRenderer` uses
  only the codebook + decoder), saving ~300 MB VRAM.
- Perplexity tracks codebook utilization: target > 100 (out of 512 entries).
