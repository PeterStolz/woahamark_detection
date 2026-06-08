"""WatermarkCNN: MobileNetV3-small backbone (ImageNet-pretrained) + small head.

V1/V2 used a from-scratch ~1.5M-param CNN; it plateaued at mean F1 ≈ 0.63 on test
because the small dataset + heavy augmentation is not enough to learn good features
from scratch. MobileNetV3-small starts from ImageNet weights (~1.5M backbone params,
~17M mults) and only needs to learn the head — converges far faster on this kind of
small fine-tuning task.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .config import CLASSES


class WatermarkCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        # MobileNetV3-small's feature extractor outputs 576 channels at the GAP point.
        self.features = backbone.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# Note: ImageNet normalization is what the pretrained backbone expects.
# We override the (mean=0.5, std=0.5) used in v1/v2 with the standard
# ImageNet stats; data.py reads NORM_MEAN / NORM_STD for the transform.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = WatermarkCNN()
    x = torch.randn(2, 3, 192, 192)
    y = m(x)
    print("output shape:", y.shape)
    print(f"params: {count_params(m):,}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = WatermarkCNN()
    x = torch.randn(2, 3, 192, 192)
    y = m(x)
    print("output shape:", y.shape)
    print(f"params: {count_params(m):,}")
