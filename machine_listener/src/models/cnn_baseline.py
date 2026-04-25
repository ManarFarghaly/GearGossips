"""
Phase 1 — 2-D CNN on Mel-Spectrogram.

Input : (B, 1, 128, 84)
Output: (B, 6)   logits — pass through CrossEntropyLoss, no softmax here

extract_features(x) → (B, 256)   ← used by Phase 2 to re-use learned weights
forward(x)          → (B, 6)     ← used for standalone Phase 1 training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelCNN(nn.Module):

    def __init__(self, num_classes: int = 6):
        super().__init__()
        # MaxPool halves the spatial size; AdaptiveAvgPool forces exact (4,4).

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,1,128,84)->(B,32,128,84)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # (B,32,64,42)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # (B,64,32,21)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # (B,128,16,10)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                # →(B,256,4,4)
        )

        self.fc1     = nn.Linear(256 * 4 * 4, 256)   # 4096 → 256
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)    # 256  → 6

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        added for Phase 2 calls this so it can plug the mel-stream into a larger model.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)       # (B, 256*4*4) = (B, 4096)
        x = F.relu(self.fc1(x))       # (B, 256)
        x = self.dropout(x)
        return x                       # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)  # (B, 256)
        return self.fc2(x)             # (B, 6)
