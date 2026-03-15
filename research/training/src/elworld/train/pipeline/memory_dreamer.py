import torch
import numpy as np
from pathlib import Path
import json
import cv2

from elworld.model.memory import MemoryModel
from elworld.model.vision import VisionModel


class MemoryDreamer:
    """Dream (autoregressive generation) with trained memory model."""
    
    def __init__(self, memory_checkpoint_path, vision_checkpoint_path, device='cuda'):
        self.device = device
        
        # Load memory model
        print(f"\n[1/2] Loading Memory Model from {memory_checkpoint_path}...")
        with open(Path(memory_checkpoint_path) / "config.json", 'r') as f:
            memory_config = json.load(f)
        
        self.memory_model = MemoryModel(
            vocab_size=memory_config['vocab_size'],
            block_size=memory_config['block_size'],
            n_emb=memory_config['n_emb'],
            num_layers=memory_config['num_layers'],
            num_heads=memory_config['num_heads'],
            dropout=0.0,  # No dropout for inference
            action_dim=memory_config['action_dim']
        ).to(device)
        
        self.memory_model.load_state_dict(
            torch.load(Path(memory_checkpoint_path) / "model.pth", map_location=device)
        )
        self.memory_model.eval()
        print(f"✓ Memory model loaded")
        
        # Load vision model for decoding
        print(f"\n[2/2] Loading Vision Model from {vision_checkpoint_path}...")
        with open(Path(vision_checkpoint_path) / "config.json", 'r') as f:
            vision_config = json.load(f)
        
        self.vision_model = VisionModel(
            num_hidden=vision_config['num_hidden'],
            res_layer=vision_config['res_layer'],
            res_hidden=vision_config['res_hidden'],
            input_channels=vision_config['input_channels'],
            num_embedding=vision_config['num_embedding'],
            embedding_dim=vision_config['latent_dim'],
            commitment_cost=vision_config['commitment_cost']
        ).to(device)
        
        self.vision_model.load_state_dict(
            torch.load(Path(vision_checkpoint_path) / "model.pth", map_location=device)
        )
        self.vision_model.eval()
        print(f"✓ Vision model loaded\n")
        
        self.memory_config = memory_config
        self.vision_config = vision_config
    
    @torch.no_grad()
    def dream(self, seed_frames, seed_actions, dream_steps=50, temperature=1.0):
        """
        Autoregressively generate future frames.
        
        Args:
            seed_frames: [T, C, H, W] - Initial frames to condition on
            seed_actions: [T, action_dim] - Initial actions
            dream_steps: Number of future frames to generate
            temperature: Sampling temperature (1.0=default, <1.0=more deterministic)
            
        Returns:
            generated_frames: [dream_steps, C, H, W] - Generated frames
        """
        print(f"Starting dream sequence...")
        print(f"  Seed frames: {len(seed_frames)}")
        print(f"  Dream steps: {dream_steps}")
        print(f"  Temperature: {temperature}\n")
        
        # Encode seed frames to tokens
        seed_frames = seed_frames.to(self.device)
        _, _, _, _, encoding_indices = self.vision_model(seed_frames)
        
        # Flatten: [T, 24, 32] -> [T, 768]
        current_tokens = encoding_indices.view(len(seed_frames), -1)  # [T, 768]
        current_actions = seed_actions.to(self.device)  # [T, action_dim]
        
        generated_frames = []
        
        for step in range(dream_steps):
            # Predict next frame tokens
            logits = self.memory_model(current_tokens, current_actions)  # [B, T, vocab_size]
            
            # Get prediction for last timestep
            next_token_logits = logits[0, -1, :] / temperature  # [vocab_size]
            
            # Sample next tokens (768 tokens per frame)
            next_frame_tokens = torch.zeros(768, dtype=torch.long, device=self.device)
            for i in range(768):
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                next_frame_tokens[i] = next_token
            
            # Decode tokens to frame
            token_grid = next_frame_tokens.view(1, 24, 32)  # [1, 24, 32]
            decoded_frame = self.vision_model.decode_from_indices(token_grid)  # [1, 3, 192, 256]
            generated_frames.append(decoded_frame[0])
            
            # Update context (sliding window)
            next_frame_tokens = next_frame_tokens.unsqueeze(0)  # [1, 768]
            current_tokens = torch.cat([current_tokens[1:], next_frame_tokens], dim=0)  # Keep window size
            
            # Use zero action for dreaming (or could use last action)
            zero_action = torch.zeros(1, self.memory_config['action_dim'], device=self.device)
            current_actions = torch.cat([current_actions[1:], zero_action], dim=0)
            
            if (step + 1) % 10 == 0:
                print(f"  Generated {step + 1}/{dream_steps} frames...")
        
        print(f"✓ Dream completed!\n")
        return torch.stack(generated_frames)  # [dream_steps, 3, 192, 256]
    
    def save_dream_video(self, frames, output_path, fps=20):
        """Save generated frames as video."""
        frames = frames.cpu().numpy()
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
        print(f"✓ Dream video saved to {output_path}")
