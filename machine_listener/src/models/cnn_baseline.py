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


class MelCNNHier(nn.Module):
    """
    Phase 1 v3 — same backbone as MelCNN but with three output heads.

    head_main    → (B, num_classes)   primary 6-class prediction
    head_machine → (B, 3)             auxiliary: which machine?
    head_fault   → (B, 1)             auxiliary: normal=0 / abnormal=1 (binary logit)

    forward() returns (main, machine, fault) — use head_main for final prediction.
    extract_features() → (B, 256) shared embedding used by Phase 2b v2 as mel_stream.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc1     = nn.Linear(256 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)

        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x, 1))))   # (B, 256)

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)
        return (self.head_main(feat),     # (B, num_classes)
                self.head_machine(feat),  # (B, 3)
                self.head_fault(feat))    # (B, 1)
