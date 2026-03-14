import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Improved Vector Quantizer with EMA (Exponential Moving Average) update.
    """
    def __init__(self, num_embedding=512, embedding_dim=64, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.embeddings.weight.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embedding))
        self.ema_w = nn.Parameter(torch.Tensor(num_embedding, embedding_dim))
        self.ema_w.data.normal_()
        
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, x):
        # Convert inputs from [B, C, H, W] to [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape
        
        # Flatten input
        flat_x = x.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
                    
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embedding, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Use EMA to update the codebook
        if self.training:
            self._ema_cluster_size.data.mul_(self.decay).add_(
                torch.sum(encodings, 0), alpha=(1 - self.decay)
            )
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (
                (self._ema_cluster_size.data + self.epsilon)
                / (n + self.num_embedding * self.epsilon) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_x)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))
            
            self.embeddings.weight.data.copy_(self.ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from [B, H, W, C] to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return {
            'quantized': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encoding_indices': encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])
        }