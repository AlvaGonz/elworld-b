import torch
import numpy as np
from pathlib import Path
import json
import cv2
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from elworld.model.memory import MemoryModel
from elworld.model.vision import VisionModel
from elworld.data.memory_data import MemoryDataset
from utils import load_config, get_general_config, get_vision_config, get_memory_config


def load_memory_model(checkpoint_path, device='cuda'):
    """Load trained memory model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    with open(checkpoint_path / "config.json", 'r') as f:
        config = json.load(f)
    
    model = MemoryModel(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_emb=config['n_emb'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=0.0,
        action_dim=config['action_dim']
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path / "model.pth", map_location=device))
    model.eval()
    return model, config


def load_vision_model(checkpoint_path, device='cuda'):
    """Load trained vision model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    with open(checkpoint_path / "config.json", 'r') as f:
        config = json.load(f)
    
    model = VisionModel(
        num_hidden=config['num_hidden'],
        res_layer=config['res_layer'],
        res_hidden=config['res_hidden'],
        input_channels=config['input_channels'],
        num_embedding=config['num_embedding'],
        embedding_dim=config['latent_dim'],
        commitment_cost=config['commitment_cost']
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path / "model.pth", map_location=device))
    model.eval()
    return model, config


@torch.no_grad()
def dream_sequence(memory_model, vision_model, seed_tokens, seed_actions, dream_steps=50, use_top_k=True, device='cuda'):
    """
    Autoregressively generate future frames (Token-by-Token).
    Uses GREEDY decoding (argmax) for maximum quality.
    
    Args:
        memory_model: Trained memory model
        vision_model: Trained vision model for decoding
        seed_tokens: [T_frames, 768] - Initial token sequence
        seed_actions: [T_frames, action_dim] - Initial actions
        dream_steps: Number of future frames to generate
        use_top_k: Filter to top-K tokens before argmax (reduces noise)
        device: Device to run on
        
    Returns:
        generated_frames: [dream_steps, 3, H, W] - Generated frames
    """
    memory_model.eval()
    
    # 1. Prepare initial context
    # seed_tokens: [Frames, 768] -> Flatten to [1, Frames * 768]
    context = seed_tokens.view(1, -1).to(device)
    
    # Actions: [Frames, Action_Dim] -> [1, Frames * 768, Action_Dim]
    current_actions = seed_actions.to(device)
    actions_expanded = current_actions.repeat_interleave(768, dim=0).unsqueeze(0)
    
    generated_frames = []
    tokens_per_frame = 768  # 24x32
    
    # 2. Frame-by-Frame loop with progress bar
    pbar_frames = tqdm(range(dream_steps), desc="Dreaming frames", unit="frame")
    
    for step in pbar_frames:
        
        # Generate 768 tokens for one frame (Token-by-Token loop)
        new_frame_tokens = []
        
        # Action for future frame (keep last action or use zero)
        last_action = current_actions[-1:].unsqueeze(0)  # [1, 1, Action_dim]
        
        # 3. Token-by-Token generation with progress bar
        pbar_tokens = tqdm(range(tokens_per_frame), desc=f"  Frame {step+1} tokens", leave=False, unit="tok")
        
        for i in pbar_tokens:
            # Trim context if too long
            if context.size(1) > memory_model.block_size:
                context_cond = context[:, -memory_model.block_size:]
                actions_cond = actions_expanded[:, -memory_model.block_size:, :]
            else:
                context_cond = context
                actions_cond = actions_expanded
            
            # Forward pass
            logits = memory_model(context_cond, actions_cond)['logits']
            
            # Get logits for last token only
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]
            
            # Optional: Top-K filtering to remove very low probability tokens
            if use_top_k:
                top_k = 20
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # GREEDY: Always pick highest probability token (argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]
            
            # Update context immediately! (CRITICAL!)
            context = torch.cat((context, next_token), dim=1)
            new_frame_tokens.append(next_token)
            
            # Trim context if it exceeds block_size
            if context.size(1) > memory_model.block_size:
                context = context[:, -memory_model.block_size:]
                actions_expanded = actions_expanded[:, -memory_model.block_size:, :]
            
            # Update action context for next token
            actions_expanded = torch.cat((actions_expanded, last_action), dim=1)
        
        pbar_tokens.close()
        
        # 4. After collecting 768 tokens -> Decode to image
        frame_indices = torch.cat(new_frame_tokens, dim=1)  # [1, 768]
        
        # Decode VQ-VAE (keep on GPU)
        token_grid = frame_indices.view(1, 24, 32)
        embeddings = vision_model.model.vq_layer.embeddings(token_grid)  # [1, 24, 32, 64]
        embeddings = embeddings.permute(0, 3, 1, 2)  # [1, 64, 24, 32]
        decoded_frame = vision_model.model.decoder(embeddings)  # [1, 3, 192, 256]
        
        # Keep on GPU, only move to CPU at the end
        generated_frames.append(decoded_frame[0])
        
        pbar_frames.set_postfix({'tokens': context.size(1), 'frames': len(generated_frames)})
    
    # Move all frames to CPU at once
    return torch.stack([f.cpu() for f in generated_frames])


def save_video(frames, output_path, fps=20):
    """Save frames as video."""
    frames = frames.cpu().numpy()
    
    # Normalize: Handle both [0,1] and [-1,1] range
    if frames.min() < 0:
        # [-1, 1] -> [0, 1]
        frames = (frames + 1) / 2
    
    # Clip and convert to uint8
    frames = (frames * 255).clip(0, 255).astype(np.uint8)
    frames = frames.transpose(0, 2, 3, 1)  # [T, H, W, C]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()


def test_memory_dreaming(
    config_path='config.yaml',
    memory_checkpoint='checkpoints/memory/best_memory_model',
    vision_checkpoint='checkpoints/vision/best_model',
    data_path='../recorded/',
    num_samples=3,
    dream_steps=100,
    output_dir='dreams',
    device='cuda'
):
    """
    Test memory model's dreaming capability.
    
    Args:
        config_path: Path to config.yaml
        memory_checkpoint: Path to memory model checkpoint
        vision_checkpoint: Path to vision model checkpoint
        data_path: Path to gameplay data
        num_samples: Number of dream sequences
        dream_steps: Frames to dream forward
        output_dir: Output directory
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print("Memory Model Dreaming Test")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config(config_path)
    general_config = get_general_config(config)
    vision_config = get_vision_config(config)
    memory_config = get_memory_config(config)
    
    # Load models
    print("[1/3] Loading models...")
    memory_model, _ = load_memory_model(memory_checkpoint, device)
    vision_model, _ = load_vision_model(vision_checkpoint, device)
    print("✓ Models loaded\n")
    
    # Load dataset
    print("[2/3] Loading dataset...")
    dataset = MemoryDataset(
        data_dir=data_path,
        checkpoint_path=vision_checkpoint,
        max_files=general_config.get('total_play'),
        sequence_length=memory_config.get('sequence_length', 5),
        stride=memory_config.get('stride', 5),
        device=device
    )
    print(f"✓ Dataset loaded ({len(dataset)} sequences)\n")
    
    # Generate dreams
    print(f"[3/3] Generating dreams...")
    print(f"⚠️  Warning: Autoregressive generation is SLOW (768 forward passes per frame)")
    print(f"    Estimated time: ~5-10 minutes per frame on GPU")
    print(f"    Consider reducing dream_steps to 5-10 for testing\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}:")
        sample = dataset[i * 100]  # Sample every 100 sequences
        
        seed_tokens = sample['input_tokens']
        seed_actions = sample['actions']
        
        # Generate with greedy decoding
        generated = dream_sequence(
            memory_model, vision_model,
            seed_tokens, seed_actions,
            dream_steps, use_top_k=True, device=device
        )
        
        video_path = output_path / f"dream_{i+1}_greedy.mp4"
        save_video(generated, video_path, fps=general_config.get('frame_rate', 20))
        print(f"\n✓ Saved to {video_path}")

    
    print(f"\n{'='*60}")
    print(f"Dreaming complete! Videos saved to {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_memory_dreaming()
