import torch
import torch.nn as nn

from elworld.model.architectures.vision.layers import DepthwiseSeparableConv, get_activation
from elworld.model.architectures.vision.residual_stack import ResidualStack

class VisionDecoder(nn.Module):
    def __init__(
        self, input_channels=3, num_hidden=128, res_layer=2, res_hidden=32, output_channels=3
    ):
        super(VisionDecoder, self).__init__()
        self.activation = get_activation()
        
        self.conv1 = nn.Conv2d(input_channels, num_hidden, kernel_size=3, stride=1, padding=1)
        self.res_stack = ResidualStack(
            input_channels=num_hidden, num_hidden=num_hidden, 
            res_layer=res_layer, res_hidden=res_hidden
        )
        
        # Upsampling 1: 24x32 -> 48x64
        self.upsample1_conv = DepthwiseSeparableConv(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.upsample1_ps = nn.PixelShuffle(2) # [B, 256, 24, 32] -> [B, 64, 48, 64] if input is 256. 
        # Wait, if num_hidden=256, ps(2) -> 64. 
        # Let's use a standard conv to prepare for PixelShuffle
        self.pre_shuffle1 = nn.Conv2d(num_hidden, (num_hidden//2) * 4, kernel_size=3, padding=1)
        
        # Upsampling 2: 48x64 -> 96x128
        self.pre_shuffle2 = nn.Conv2d(num_hidden//2, (num_hidden//4) * 4, kernel_size=3, padding=1)
        
        # Upsampling 3: 96x128 -> 192x256
        self.pre_shuffle3 = nn.Conv2d(num_hidden//4, output_channels * 4, kernel_size=3, padding=1)
        
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.res_stack(x)
        
        # Layer 1
        x = self.pre_shuffle1(x)
        x = self.ps(x)
        x = self.activation(x)
        
        # Layer 2
        x = self.pre_shuffle2(x)
        x = self.ps(x)
        x = self.activation(x)
        
        # Layer 3
        x = self.pre_shuffle3(x)
        x = self.ps(x)
        
        return x