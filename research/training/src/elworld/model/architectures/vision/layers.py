import torch
import torch.nn as nn

def get_activation():
    return nn.SiLU(inplace=True)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.activation = get_activation()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        return x
