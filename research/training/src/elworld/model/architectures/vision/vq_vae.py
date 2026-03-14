import torch
import torch.nn as nn

from elworld.model.architectures.vision.encoder import VisionEncoder
from elworld.model.architectures.vision.vector_quantizer import VectorQuantizer
from elworld.model.architectures.vision.decoder import VisionDecoder


from elworld.model.architectures.vision.layers import get_activation

class VQ_VAE(nn.Module):
    def __init__(
        self, num_hidden=128, res_layer=2, res_hidden=32, input_channels=3,
        num_embedding=512, embedding_dim=64, commitment_cost=0.25,
        decay=0.99, epsilon=1e-5
    ):
        super(VQ_VAE, self).__init__()
        self.encoder = VisionEncoder(
            input_channels=input_channels, num_hidden=num_hidden,
            res_layer=res_layer, res_hidden=res_hidden
        )
        
        self.activation = get_activation()
        
        # Pointwise convolution to projector latent space
        self.pre_vq_conv = nn.Conv2d(num_hidden, embedding_dim, kernel_size=1, stride=1)
        
        self.vq_layer = VectorQuantizer(
            num_embedding=num_embedding, embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost, decay=decay, epsilon=epsilon
        )
        
        self.decoder = VisionDecoder(
            input_channels=embedding_dim, num_hidden=num_hidden,
            res_layer=res_layer, res_hidden=res_hidden, output_channels=input_channels
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        z = self.activation(z) # Add activation before quantization
        vq_output = self.vq_layer(z)
        vq_loss = vq_output['loss']
        quantized = vq_output['quantized']
        perplexity = vq_output['perplexity']
        x_recon = self.decoder(quantized)

        return {
            'x_recon': x_recon, # [B, 3, 192, 256]: The reconstructed image
            'vq_loss': vq_loss, # VQ-VAE loss
            'perplexity': perplexity, # Measure of codebook utilization, range [1, num_embedding]
            'encoding_indices': vq_output['encoding_indices'] # [B, 24, 32]: Indices of the embeddings used
        }