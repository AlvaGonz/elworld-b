# Vision Model Checkpoint - Epoch 20

## Training Metrics

- **Epoch:** 20
- **Total Loss:** 0.064753
- **Reconstruction Loss:** 0.062803
- **VQ Loss:** 0.001950
- **Learning Rate:** 0.000186
- **Epoch Time:** 180.89s

**🏆 This is the best model so far!**

## Files

- `model.pth` - Model weights
- `optimizer.pth` - Optimizer state
- `scheduler.pth` - LR scheduler state
- `training_info.json` - Complete training metadata
- `config.json` - Model configuration

## Usage

```python
# Load model
model = VisionModel(**config)
model.load_state_dict(torch.load('model.pth'))
```
