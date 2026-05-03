"""
Phase 2 — MFCC sub-stream (used inside MelMFCCCNN, not a standalone classifier).

Input : (B, 3, 40, 84)   — 3 channels: mfcc, delta, delta2
Output: (B, 128)          — feature vector, concatenated with mel stream in MelMFCCCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFCCStream(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B,3,40,84)→(B,32,40,84)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # →(B,32,20,42)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # →(B,64,10,21)

            nn.AdaptiveAvgPool2d((4, 4)),                 # →(B,64,4,4)
        )

        self.fc = nn.Linear(64 * 4 * 4, 128)             # 1024 → 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                # (B, 64, 4, 4)
        x = torch.flatten(x, 1)             # (B, 1024)
        x = F.relu(self.fc(x))              # (B, 128)
        return x
