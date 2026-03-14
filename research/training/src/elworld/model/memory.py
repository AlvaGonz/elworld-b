import torch
import torch.nn as nn

from elworld.model.architectures.memory.parallel_memory import ParallelMemoryModel


class MemoryModel(nn.Module):
    """
    Memory Model wrapper for MinGPT transformer.
    Predicts next frame's visual tokens given current frame + action.
    
    Architecture:
        Input: visual_tokens [B, T] + actions [B, T, action_dim]
        Output: next_token_predictions [B, T, vocab_size]
    
    Usage similar to VisionModel for consistency.
    """
    
    def __init__(
        self,
        vocab_size=512,      # VQ-VAE codebook size
        context_frames=4,    # Number of past frames to look at
        embed_dim=64,        # Embedding dimension
        hidden_dim=256,      # ResNet hidden channels
        num_res_blocks=6,    # Number of ResNet blocks
        action_dim=22,       # Action dimension
        grid_h=24,           # Grid height
        grid_w=32            # Grid width
    ):
        """
        Args:
            vocab_size: Size of VQ-VAE codebook (default 512)
            context_frames: Frames of context (default 4)
            embed_dim: Token embedding dimension (default 64)
            hidden_dim: ResNet internal channels (default 256)
            num_res_blocks: Number of resblocks (default 6)
            action_dim: Action vector dimension (default 22)
            grid_h: Latent height
            grid_w: Latent width
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_frames = context_frames
        self.action_dim = action_dim
        
        # Parallel Spatial CNN
        self.model = ParallelMemoryModel(
            vocab_size=vocab_size,
            context_frames=context_frames,
            embed_dim=embed_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            grid_h=grid_h,
            grid_w=grid_w
        )
    
    def forward(self, visual_tokens, actions=None, targets=None):
        """
        Forward pass through memory model.
        
        Args:
            visual_tokens: Context frames [B, context_frames, H, W]
            actions: Action vectors [B, action_dim] for current step
            targets: Target block [B, H, W] for supervised learning
        
        Returns:
            Dictionary containing:
                - 'logits': [B, vocab_size, H, W] - Predicted token distributions
                - 'loss': Scalar loss (only if targets provided)
                - 'predictions': [B, H, W] - Argmax predictions
        """
        # Forward through CNN
        if targets is not None:
            logits, loss = self.model(visual_tokens, actions=actions, targets=targets)
            predictions = torch.argmax(logits, dim=1) # dim 1 is vocab_size channel

            return {
                'logits': logits,
                'loss': loss,
                'predictions': predictions
            }
        else:
            logits = self.model(visual_tokens, actions=actions)
            predictions = torch.argmax(logits, dim=1)
            return {
                'logits': logits,
                'predictions': predictions
            }

    def generate(self, start_tokens, actions=None, num_steps=768, temperature=1.0, top_k=None):
        """
        Generate EXACTLY one next frame in parallel.
        
        Args:
            start_tokens: Context frames [B, context_frames, H, W]
            actions: Action vector [B, action_dim]
            num_steps: Ignored for parallel model (always generates 1 full frame)
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            Generated token frame [B, 1, H, W]
        """
        return self.model.generate(
            context=start_tokens,
            action=actions,
            temperature=temperature,
            top_k=top_k
        )
    
    def get_num_params(self):
        """Return number of parameters."""
        return self.model.get_num_params()
        
    def prepare_qat(self):
        """Prepare the underlying MinGPT model for Quantization-Aware Training."""
        self.model.prepare_qat()
