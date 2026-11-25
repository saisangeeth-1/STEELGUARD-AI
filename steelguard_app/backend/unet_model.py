"""
STEELGUARD AI — Production-Grade Lightweight U-Net Architecture
Implements a 4-stage encoder-decoder U-Net with skip connections tailored
for high-aspect-ratio (256x1600) industrial surface defect segmentation.
Five output channels: Background (0), Class 1 (1), Class 2 (2), Class 3 (3), Class 4 (4).
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive Conv3x3 -> BatchNorm2d -> ReLU blocks.
    Uses padding=1 to preserve spatial dimensions across convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool2d(2) followed by DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class UpBlock(nn.Module):
    """
    Upscaling with ConvTranspose2d followed by skip connection concatenation
    and DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: from deeper layer, x2: skip connection from encoder
        x1 = self.up(x1)

        # Handle any possible spatial dimension differences due to non-power-of-two shapes
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LightweightUNet(nn.Module):
    """
    Lightweight 4-stage U-Net for multi-class surface defect segmentation.
    
    Args:
        in_channels: Input image channels (default: 3 for RGB).
        num_classes: Number of target segmentation classes (default: 5: 0=bg, 1..4=defects).
        base_channels: Starting channel width for first encoder stage (default: 16).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        base_channels: int = 16
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        c = base_channels

        # 4 Encoder / Downsampling stages
        self.inc = DoubleConv(in_channels, c)               # (B, c, H, W)
        self.down1 = DownBlock(c, c * 2)                    # (B, 2c, H/2, W/2)
        self.down2 = DownBlock(c * 2, c * 4)                # (B, 4c, H/4, W/4)
        self.down3 = DownBlock(c * 4, c * 8)                # (B, 8c, H/8, W/8)
        self.down4 = DownBlock(c * 8, c * 16)               # (B, 16c, H/16, W/16) - Bottleneck

        # 4 Decoder / Upsampling stages with skip connections
        self.up1 = UpBlock(c * 16, c * 8)                   # (B, 8c, H/8, W/8)
        self.up2 = UpBlock(c * 8, c * 4)                    # (B, 4c, H/4, W/4)
        self.up3 = UpBlock(c * 4, c * 2)                    # (B, 2c, H/2, W/2)
        self.up4 = UpBlock(c * 2, c)                        # (B, c, H, W)

        # Final 1x1 classification head outputting original resolution logits
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers, ones/zeros for batch norm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
        Returns:
            Logits tensor of shape (B, num_classes, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_summary(self) -> Dict[str, Any]:
        """Returns model structural summary and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_architecture": "LightweightUNet",
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "encoder_stages": 4,
            "decoder_stages": 4,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameters_millions": round(total_params / 1e6, 3)
        }
