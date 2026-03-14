import torch
import torch.nn as nn
from elworld.model.architectures.vision.layers import get_activation

class ResidualBlock(nn.Module):
    def __init__(self, input_channels=128, num_hidden=128, res_hidden=32):
        super(ResidualBlock, self).__init__()
        self.activation = get_activation()
        self.block = nn.Sequential(
            self.activation,
            nn.Conv2d(input_channels, res_hidden, kernel_size=3, stride=1, padding=1, bias=False),
            self.activation,
            nn.Conv2d(res_hidden, num_hidden, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, input_channels=3, num_hidden=128, res_layer=2, res_hidden=32,
    ):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                ResidualBlock(input_channels, num_hidden, res_hidden)
                for _ in range(res_layer)
            ]
        )
        self.activation = get_activation()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.activation(x)