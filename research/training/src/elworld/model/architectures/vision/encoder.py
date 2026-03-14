import torch
import torch.nn as nn

from elworld.model.architectures.vision.layers import DepthwiseSeparableConv, get_activation
from elworld.model.architectures.vision.residual_stack import ResidualStack

class VisionEncoder(nn.Module):
    def __init__(
        self, input_channels=3, num_hidden=128, res_layer=2, res_hidden=32,
    ):
        super(VisionEncoder, self).__init__()
        # Initial convolution to increase channels
        self.conv1 = nn.Conv2d(input_channels, num_hidden//2, kernel_size=4, stride=2, padding=1)
        self.ln1 = nn.GroupNorm(8, num_hidden//2)
        
        # Depthwise Separable convolutions for speed
        self.conv2 = DepthwiseSeparableConv(num_hidden//2, num_hidden, kernel_size=4, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv(num_hidden, num_hidden, kernel_size=4, stride=2, padding=1)

        self.res_stack = ResidualStack(
            input_channels=num_hidden, num_hidden=num_hidden, 
            res_layer=res_layer, res_hidden=res_hidden
        )
        self.activation = get_activation()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        
        return self.res_stack(x)