import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quant

class ResBlock(nn.Module):
    """Simple robust Pre-Activation Residual Block"""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    
    def forward(self, x):
        return x + self.net(x)


class ParallelMemoryModel(nn.Module):
    """
    Parallel Spatial Dynamics Model (O(1) generation)
    
    Predicts the entire NEXT FRAME of VQ-VAE tokens simultaneously 
    given a sequence of past frames + an action.
    """
    def __init__(
        self,
        vocab_size=512,
        context_frames=4,
        embed_dim=64,
        action_dim=22,
        hidden_dim=256,
        num_res_blocks=6,
        grid_h=24,
        grid_w=32
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_frames = context_frames
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # Continuous embedding for categorical tokens
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        
        # The input has channels: (context_frames * embed_dim) + action_dim
        in_channels = (context_frames * embed_dim) + action_dim
        
        # Spatial CNN Backbone
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        self.res_blocks = nn.Sequential(*[
            ResBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Classifier Head -> maps back to 512 vocab classes per spatial coordinate
        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, vocab_size, kernel_size=1)
        )
        
        # Quantization stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        
        print(f"ParallelMemoryModel initialized with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x, actions=None, targets=None):
        """
        x: [B, context_frames, H, W] -> sequence of past token grids
        actions: [B, action_dim] -> action taken at current step
        targets: [B, H, W] -> real next frame tokens (optional)
        """
        B, Seq, H, W = x.shape
        assert Seq == self.context_frames, f"Expected {self.context_frames} context frames, got {Seq}"
        assert H == self.grid_h and W == self.grid_w, f"Grid mismatch"
        
        # Embed tokens to continuous space: [B, Seq, H, W] -> [B, Seq, H, W, embed_dim]
        emb = self.token_emb(x)
        
        # Rearrange to Channel-first for CNNs: [B, Seq, embed_dim, H, W]
        emb = emb.permute(0, 1, 4, 2, 3) 
        
        # Flatten time into channels: [B, Seq * embed_dim, H, W]
        emb = emb.reshape(B, Seq * self.embed_dim, H, W)
        
        # Inject actions spatially
        if actions is not None:
            # Broadcast action to match spatial dimensions: [B, action_dim, H, W]
            act_emb = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            # Concatenate
            features = torch.cat([emb, act_emb], dim=1)
        else:
            # Failsafe zeros if action_dim > 0 but no actions provided
            act_emb = torch.zeros((B, self.action_dim, H, W), device=x.device, dtype=emb.dtype)
            features = torch.cat([emb, act_emb], dim=1)
            
        # QAT stubs surround the compute-heavy CNN spine
        features = self.quant(features)
        
        # Push through CNN spine
        h = self.conv_in(features)
        h = self.res_blocks(h)
        logits = self.conv_out(h) # [B, vocab_size, H, W]
        
        logits = self.dequant(logits)
        
        if targets is not None:
            # targets is [B, H, W]
            # cross_entropy handles spatial multijoint classification seamlessly when form is [B, C, H, W]
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            return logits, loss
            
        return logits
        
    @torch.no_grad()
    def generate(self, context, action=None, temperature=1.0, top_k=None):
        """
        Generates the EXACT next frame in completely O(1) time.
        context: [B, context_frames, H, W]
        action: [B, action_dim]
        
        Returns: [B, 1, H, W] -> The generated frame
        """
        logits = self(context, action) # [B, vocab_size, H, W]
        
        # Pre-process for sampling: shift channels to the end
        logits = logits.permute(0, 2, 3, 1) # [B, H, W, vocab_size]
        logits = logits / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # logits: [B, H, W, V], v: [B, H, W, K]. We broadcast min condition.
            logits[logits < v[..., [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1) # [B, H, W, vocab_size]
        
        # Flatten spatial dims to sample easily
        B, H, W, V = probs.shape
        probs_flat = probs.reshape(-1, V) # [B * H * W, vocab_size]
        
        # Sample concurrently across all 768 pixels simultaneously
        next_tokens_flat = torch.multinomial(probs_flat, num_samples=1) # [B * H * W, 1]
        next_tokens = next_tokens_flat.reshape(B, 1, H, W) # [B, 1, H, W]
        
        return next_tokens

    def prepare_qat(self):
        """Prepare the model for Quantization-Aware Training (QAT)."""
        self.train()
        self.qconfig = quant.get_default_qat_qconfig('fbgemm')
        # token_emb is categorically tricky to quantize, usually we quantize the Conv blocks
        # but PyTorch QAT handles Embedding gracefully if configured
        quant.prepare_qat(self, inplace=True)
        print("  [OK] ParallelMemoryModel prepared for QAT (Quantization-Aware Training).")

    def get_num_params(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
