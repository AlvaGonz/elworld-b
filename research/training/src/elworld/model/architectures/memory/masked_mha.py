import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedMultiHeadAttention(nn.Module):
    """
    Masked Multi-Head Attention for autoregressive generation.
    Uses causal mask to prevent attending to future tokens.
    """
    
    def __init__(self, n_emb, num_heads, block_size, dropout=0.1):
        """
        Args:
            n_emb: Embedding dimension (e.g., 64)
            num_heads: Number of attention heads (e.g., 4)
            block_size: Maximum sequence length (e.g., 1537)
            dropout: Dropout rate
        """
        super().__init__()
        assert n_emb % num_heads == 0, f"n_emb ({n_emb}) must be divisible by num_heads ({num_heads})"
        
        self.n_emb = n_emb
        self.num_heads = num_heads
        self.head_dim = n_emb // num_heads
        self.block_size = block_size
        
        # Key, Query, Value projections for all heads in batch
        self.qkv = nn.Linear(n_emb, 3 * n_emb)
        
        # Output projection
        self.proj = nn.Linear(n_emb, n_emb)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure attention only attends to earlier positions
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C] where T <= block_size
        
        Returns:
            Output tensor [B, T, C] after attention
        """
        B, T, C = x.shape  # Batch, sequence length, embedding dim
        
        # Calculate Q, K, V for all heads in batch
        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_emb, dim=2)  # Each: [B, T, C]
        
        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch native scaled_dot_product_attention (Flash Attention / Memory Efficient Attention)
        # It natively handles causality and dropout, ignoring the manual `self.mask` buffer.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=True
        )

        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        
        # Output projection
        y = self.proj_dropout(self.proj(y))
        
        return y
