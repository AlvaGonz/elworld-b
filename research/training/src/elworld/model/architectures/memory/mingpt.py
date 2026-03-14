import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quant

from elworld.model.architectures.memory.transformer_block import TransformerBlock
from elworld.model.architectures.memory.ff_norm import LayerNorm


class MinGPT(nn.Module):
    """
    MinGPT: Minimal GPT-style transformer for sequence modeling.
    
    Architecture:
        1. Token Embedding + Position Embedding
        2. N x Transformer Blocks (Attention + FeedForward)
        3. Layer Norm
        4. Linear head to predict next token
    
    For World Model:
        - Input: Visual tokens from VQ-VAE [B, T] where T = 768 (24x32)
        - Output: Predicted next frame tokens [B, T, vocab_size]
    """
    
    def __init__(
        self,
        vocab_size=512,      # VQ-VAE codebook size
        block_size=1537,     # Max sequence length (context window)
        n_emb=64,           # Embedding dimension
        num_layers=6,       # Number of transformer blocks
        num_heads=4,        # Number of attention heads
        dropout=0.1,        # Dropout rate
        action_dim=None     # Optional: action conditioning
    ):
        """
        Args:
            vocab_size: Size of vocabulary (512 for VQ-VAE)
            block_size: Maximum sequence length
            n_emb: Token embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            dropout: Dropout probability
            action_dim: If not None, add action conditioning
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_emb = n_emb
        self.action_dim = action_dim
        
        # Token embeddings: map token IDs to vectors
        self.tok_emb = nn.Embedding(vocab_size, n_emb)
        
        # Positional embeddings: learned position encodings
        self.pos_emb = nn.Embedding(block_size, n_emb)
        
        # Action conditioning (optional)
        if action_dim is not None:
            self.action_proj = nn.Linear(action_dim, n_emb)
        
        # Input dropout
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_emb, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_emb)
        
        # Output projection head to vocabulary size
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        
        # Tie the weights between embedding and output head (saves parameters, often improves performance)
        self.tok_emb.weight = self.head.weight

        # Quantization stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"MinGPT initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm) or isinstance(module, nn.LayerNorm): # Added nn.LayerNorm for consistency
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx, actions=None, targets=None):
        """
        Forward pass through transformer.
        
        Args:
            idx: Token indices [B, T] where T <= block_size
            actions: Optional action tensor [B, T, action_dim] for conditioning
            targets: Optional target tokens [B, T] for computing loss
        
        Returns:
            If targets is None:
                logits: [B, T, vocab_size] - predicted token distributions
            If targets is not None:
                (logits, loss): prediction logits and cross-entropy loss
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        
        # Token embeddings: [B, T] -> [B, T, n_emb]
        tok_emb = self.tok_emb(idx)
        
        # Position embeddings: [T] -> [T, n_emb] -> [1, T, n_emb]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos).unsqueeze(0)  # [1, T, C]
        
        # Combine embeddings
        x = tok_emb + pos_emb
        
        # Add action conditioning if provided
        if actions is not None and self.action_dim is not None:
            # Actions shape can be:
            # - [B, action_dim]: single action per sequence -> expand to all positions
            # - [B, T, action_dim]: action per token position
            if actions.dim() == 2:  # [B, action_dim]
                act_emb = self.action_proj(actions)  # [B, n_emb]
                act_emb = act_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, n_emb]
            else:  # [B, T, action_dim]
                act_emb = self.action_proj(actions)  # [B, T, n_emb]
            x = x + act_emb

        # QAT: Quantize inputs before transformer blocks
        x = self.quant(x)
            
        x = self.drop(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)

        # Output linear head
        logits = self.head(x)  # [B, T, vocab_size]

        # QAT: Dequantize outputs back to fp32/16
        logits = self.dequant(logits)
        
        # Calculate loss if targets are provided
        if targets is not None:
            # Reshape for cross-entropy: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens if any
            )
            return logits, loss
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, actions=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If not None, only sample from top k tokens
            actions: Optional actions [B, max_new_tokens, action_dim]
        
        Returns:
            Generated token sequence [B, T + max_new_tokens]
        """
        for i in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get action for current step if provided
            action_cond = None
            if actions is not None and self.action_dim is not None:
                # If actions are provided for the full generation length, slice them
                # Otherwise, assume actions are for the current context or single action
                if actions.dim() == 3: # [B, total_gen_len, action_dim]
                    action_cond = actions[:, :idx_cond.size(1)]
                elif actions.dim() == 2: # [B, action_dim]
                    action_cond = actions # Will be expanded in forward
            
            # Forward pass
            logits = self(idx_cond, actions=action_cond)
            
            # Focus only on last time step
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # [B, T+1]
        
        return idx
    
    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
        
    def prepare_qat(self):
        """Prepare the model for Quantization-Aware Training (QAT)."""
        self.train()
        # fbgemm is standard for server/x86 INT8 inference.
        # qnnpack is for ARM/mobile.
        self.qconfig = quant.get_default_qat_qconfig('fbgemm')
        quant.prepare_qat(self, inplace=True)
        print("  [OK] Model prepared for QAT (Quantization-Aware Training).")
