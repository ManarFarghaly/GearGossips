"""
Phase 2b — Mel-Spectrogram CNN + Statistical features branch.

A lightweight alternative to Phase 2 (MelMFCCCNN):
  • Keeps the mel_stream from Phase 1 (transfer learning)
  • Adds a tiny FC branch for global statistical descriptors
  • Skips the MFCC CNN stream  →  ~same inference cost as Phase 1

Why this might work well:
  • Statistical features (RMS, ZCR, centroid, rolloff, BW) directly encode
    machine health indicators that CNNs need many layers to learn implicitly.
  • Adding them for "free" (< 0.001 s per sample) can close accuracy gaps
    without the 2× inference cost of the MFCC stream.

Architecture:
  mel_stream  → 256-d  (MelCNN.extract_features — loaded from Phase 1)
  stat_branch →  32-d  (Linear(stat_dim→64)→ReLU→Linear(64→32)→ReLU)
  concat      → 288-d  → fc1(288→256) → Dropout(0.4) → fc2(256→6)

NOTE: stat features must be StandardScaler-normalised before passing in.
      The scaler is fitted on training data and saved to stat_scaler_2b.pkl.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from machine_listener.src.models.cnn_baseline import MelCNN


class MelStatCNN(nn.Module):
    """Mel CNN stream + Statistical feature branch, fused before the classifier head."""

    def __init__(self, num_classes: int = 6, stat_dim: int = 5):
        super().__init__()
        self.mel_stream  = MelCNN(num_classes=num_classes)

        # Tiny FC network for the stat vector — very fast, no convolutions
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Late fusion: concat mel (256) + stat (32) = 288 → classify
        self.fc1     = nn.Linear(256 + 32, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, mel: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        """
        mel  : (B, 1, 128, 84)  — mel-spectrogram tensor
        stat : (B, stat_dim)    — StandardScaler-normalised stat features
        Returns (B, num_classes) logits.
        """
        f_mel  = self.mel_stream.extract_features(mel)   # (B, 256)
        f_stat = self.stat_branch(stat)                   # (B,  32)
        x = torch.cat([f_mel, f_stat], dim=1)            # (B, 288)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
