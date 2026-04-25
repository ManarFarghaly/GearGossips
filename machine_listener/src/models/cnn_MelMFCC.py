"""
Phase 2 — Dual-stream CNN: Mel-Spectrogram + MFCC.

Inputs : mel  (B, 1, 128, 84)   mfcc  (B, 3, 40, 84)
Output : (B, 6) logits

Architecture:
  Stream 1 (mel_stream)  → extract_features() → (B, 256)    ← weights from Phase 1
  Stream 2 (mfcc_stream) → forward()          → (B, 128)    ← trained from scratch
  Concatenate                                 → (B, 384)
  FC(384→256) + ReLU + Dropout(0.4)
  FC(256→6)

Why dual-stream instead of stacking channels?
  Mel-spec has 128 freq bins; MFCC has 40 bins.
  Stacking them as channels would force the same conv filters to work on
  geometrically different grids.  Separate streams let each branch learn
  its own representations independently, then merge them at the FC level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from machine_listener.src.models.cnn_baseline import MelCNN
from machine_listener.src.models.cnn_mfcc import MFCCStream


class MelMFCCCNN(nn.Module):

    def __init__(self, num_classes: int = 6):
        super().__init__()

        self.mel_stream  = MelCNN(num_classes=num_classes)   # full model; we use extract_features()
        self.mfcc_stream = MFCCStream()

        # Fusion head — receives concatenated features from both streams
        self.fc1     = nn.Linear(256 + 128, 256)   # 384 → 256
        self.dropout = nn.Dropout(0.4)
        self.fc2     = nn.Linear(256, num_classes)  # 256 → 6

    def forward(self, mel: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel  : (B, 1, 128, 84)
            mfcc : (B, 3, 40,  84)
        Returns:
            logits (B, 6)
        """
        f_mel  = self.mel_stream.extract_features(mel)   # (B, 256)
        f_mfcc = self.mfcc_stream(mfcc)                  # (B, 128)

        x = torch.cat([f_mel, f_mfcc], dim=1)            # (B, 384)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                                   # (B, 6)
        return x
