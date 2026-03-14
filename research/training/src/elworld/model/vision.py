import torch
import torch.nn as nn

from elworld.model.architectures.vision.vq_vae import VQ_VAE


class VisionModel(nn.Module):
    def __init__(
        self, num_hidden=128, res_layer=2, res_hidden=32, input_channels=3,
        num_embedding=512, embedding_dim=64, commitment_cost=0.25,
        decay=0.99, epsilon=1e-5
    ):
        super(VisionModel, self).__init__()
        self.model = VQ_VAE(
            num_hidden=num_hidden, res_layer=res_layer, res_hidden=res_hidden, input_channels=input_channels,
            num_embedding=num_embedding, embedding_dim=embedding_dim, commitment_cost=commitment_cost,
            decay=decay, epsilon=epsilon
        )

    def forward(self, x):
        x = self.model(x)
        return {
            'x_recon': x['x_recon'],
            'vq_loss': x['vq_loss'],
            'perplexity': x['perplexity'],
            'encoding_indices': x['encoding_indices']
        }