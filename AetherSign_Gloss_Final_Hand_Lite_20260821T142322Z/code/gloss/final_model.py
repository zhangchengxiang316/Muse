from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    """A1-friendly block: depthwise 3x3 + pointwise 1x1 + optional residual."""

    def __init__(self, in_channels: int, out_channels: int, residual: bool = True) -> None:
        super().__init__()
        self.depthwise = ConvBNReLU(in_channels, in_channels, kernel_size=3, groups=in_channels)
        self.pointwise = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.use_residual = residual and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise(self.depthwise(x))
        if self.use_residual:
            y = y + x
        return y


class FinalGlossTranslatorNet(nn.Module):
    """Heavier final-round Gloss Translator while keeping ONNX/A1 operators simple.

    Expected input:  NCHW = [B, 4, 54, 64]
    Expected output: NCHW = [B, num_classes, 1, 1]
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 4,
        widths: tuple[int, int, int, int] = (64, 128, 192, 256),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = widths
        self.stem = ConvBNReLU(in_channels, c1, kernel_size=1)

        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(c1, c1),
            DepthwiseSeparableBlock(c1, c1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(c1, c2, residual=False),
            DepthwiseSeparableBlock(c2, c2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(c2, c3, residual=False),
            DepthwiseSeparableBlock(c3, c3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(c3, c4, residual=False),
            DepthwiseSeparableBlock(c4, c4),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(c4, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(num_classes: int = 6) -> FinalGlossTranslatorNet:
    return FinalGlossTranslatorNet(num_classes=num_classes)

