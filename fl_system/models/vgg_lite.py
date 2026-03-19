# @PATH: fl_system/models/vgg_lite.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# VGG-lite CNN for CIFAR-10/100 (Table 2).

"""
VGG-lite CNN for CIFAR-10/100.
Input: 3x32x32, Output: K (K=10 or 100).
"""

import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv 3x3/1/1 + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGGLite(nn.Module):
    """
    VGG-lite CNN for CIFAR-10/100.
    Block 1: 3→32→32, MaxPool → 32×16×16
    Block 2: 32→64→64, MaxPool → 64×8×8
    Block 3: 64→128→128, MaxPool → 128×4×4
    Global Avg Pool → FC → K
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            _conv_bn_relu(3, 32),
            _conv_bn_relu(32, 32),
            nn.MaxPool2d(2, 2),
            # Block 2
            _conv_bn_relu(32, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(2, 2),
            # Block 3
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
