import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from elworld.utils.memory_dream import load_memory_model, load_vision_model
from elworld.data.memory_data import MemoryDataset
from utils import load_config, get_general_config, get_vision_config, get_memory_config

# Load everything
config = load_config('config.yaml')
general_config = get_general_config(config)
vision_config = get_vision_config(config)
memory_config = get_memory_config(config)

print("Loading models...")
memory_model, _ = load_memory_model('checkpoints/memory/best_memory_model', 'cuda')
vision_model, _ = load_vision_model('checkpoints/vision/best_model', 'cuda')

print("Loading dataset...")
dataset = MemoryDataset(
    data_dir='../recorded/',
    checkpoint_path='checkpoints/vision/best_model',
    max_files=1,
    sequence_length=5,
    stride=5,
    device='cuda'
)

# Get one sample
sample = dataset[0]
seed_tokens = sample['input_tokens'].cuda()  # [5, 768]
seed_actions = sample['actions'].cuda()  # [5, 22]

print(f"\nSeed tokens shape: {seed_tokens.shape}")
print(f"Seed tokens stats: min={seed_tokens.min()}, max={seed_tokens.max()}, unique={len(torch.unique(seed_tokens))}")

# Test 1: Decode seed frames
print("\n=== TEST 1: Decode seed frames ===")
for i in range(5):
    token_grid = seed_tokens[i].view(1, 24, 32)
    embeddings = vision_model.model.vq_layer.embeddings(token_grid)
    embeddings = embeddings.permute(0, 3, 1, 2)
    decoded = vision_model.model.decoder(embeddings)
    
    print(f"Frame {i}: decoded shape={decoded.shape}, min={decoded.min():.3f}, max={decoded.max():.3f}, mean={decoded.mean():.3f}")

# Test 2: Generate 1 frame
print("\n=== TEST 2: Generate 1 frame ===")
context = seed_tokens.view(1, -1)  # [1, 5*768]
actions_expanded = seed_actions.repeat_interleave(768, dim=0).unsqueeze(0)

print(f"Context shape: {context.shape}")
print(f"Actions shape: {actions_expanded.shape}")

# Generate 10 tokens
print("\nGenerating 10 tokens:")
new_tokens = []
block_size = memory_model.block_size
for i in range(10):
    # Trim if needed
    if context.size(1) > block_size:
        context = context[:, -block_size:]
        actions_expanded = actions_expanded[:, -block_size:, :]
    
    logits = memory_model(context, actions_expanded)['logits']
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    print(f"Token {i}: {next_token.item()}, logits range=[{next_token_logits.min():.2f}, {next_token_logits.max():.2f}]")
    
    new_tokens.append(next_token.item())
    context = torch.cat([context, next_token], dim=1)
    actions_expanded = torch.cat([actions_expanded, seed_actions[-1:].unsqueeze(0)], dim=1)

print(f"\nGenerated tokens: {new_tokens}")
print(f"Unique generated tokens: {len(set(new_tokens))}")

# Test 3: Check if model is predicting same token
print("\n=== TEST 3: Model output diversity ===")
context = seed_tokens.view(1, -1)
actions_expanded = seed_actions.repeat_interleave(768, dim=0).unsqueeze(0)
logits = memory_model(context, actions_expanded)['logits']

last_logits = logits[0, -1, :]
probs = torch.softmax(last_logits, dim=-1)
top5_probs, top5_indices = torch.topk(probs, 5)

print("Top 5 predicted tokens:")
for idx, (prob, token) in enumerate(zip(top5_probs, top5_indices)):
    print(f"  {idx+1}. Token {token.item()}: {prob.item():.4f}")

print("\n=== DIAGNOSIS ===")
if len(set(new_tokens)) == 1:
    print("❌ PROBLEM: Model is predicting the SAME token repeatedly!")
    print("   → Memory model has collapsed or not trained properly")
elif decoded.min() > 0.4 and decoded.max() < 0.6:
    print("❌ PROBLEM: Decoded frames have very narrow range (all gray)")
    print("   → Vision model decoder issue")
else:
    print("✓ Tokens and decoding look reasonable")
