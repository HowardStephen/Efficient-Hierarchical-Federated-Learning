# @PATH: fl_system/models/lenet.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# LeNet-style CNN for FEMNIST (Table 3).

"""
LeNet-style CNN for FEMNIST.
Input: 1x28x28, Output: K (typically K=62).
"""

import torch.nn as nn


class FEMNISTLeNet(nn.Module):
    """
    LeNet-style CNN for FEMNIST.
    Conv 1→32 (5×5) + ReLU → MaxPool → 32×12×12
    Conv 32→64 (5×5) + ReLU → MaxPool → 64×4×4
    Flatten → FC 1024→512 + ReLU → FC 512→K
    """

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: Conv 1→32 (5×5/1/0) + ReLU → 32×24×24
            nn.Conv2d(1, 32, 5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # Stage 2: MaxPool 2×2/2 → 32×12×12
            nn.MaxPool2d(2, 2),
            # Stage 3: Conv 32→64 (5×5/1/0) + ReLU → 64×8×8
            nn.Conv2d(32, 64, 5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # Stage 4: MaxPool 2×2/2 → 64×4×4
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
