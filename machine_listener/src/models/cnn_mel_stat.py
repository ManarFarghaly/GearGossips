"""
Phase 2b — Mel-Spectrogram CNN + Statistical features branch.

Architecture:
  mel_stream  → 256-d  (MelCNN.extract_features — loaded from Phase 1)
  stat_branch →  32-d  (Linear(stat_dim→64) → ReLU → Linear(64→32) → ReLU)
  concat      → 288-d  → fc1(288→256) → Dropout(0.4) → fc2(256→6)

Stat features must be StandardScaler-normalised before passing in.
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


class MelStatCNNHier(nn.Module):
    """
    Phase 2b v2 — Mel CNN stream + statistical branch with hierarchical output heads.

    Backbone (mel_stream) is a MelCNNHier loaded from Phase 1 v3 checkpoint.
    Stat branch is a small FC network: Linear(stat_dim→64) → BN → ReLU → Linear(64→32) → ReLU.
    Fusion: cat(mel_feat 256, stat_feat 32) → fc1(288→256) → Dropout(0.4).

    Three output heads (same as Phase 1 v3 for architectural consistency):
      head_main    → (B, num_classes)
      head_machine → (B, 3)
      head_fault   → (B, 1)

    Differential LRs at training time:
      mel_stream : low LR  (already trained in Phase 1 v3)
      everything else: higher LR
    """

    def __init__(self, num_classes: int = 6, stat_dim: int = 5):
        super().__init__()
        from machine_listener.src.models.cnn_baseline import MelCNNHier
        self.mel_stream = MelCNNHier(num_classes=num_classes)

        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.fc1     = nn.Linear(256 + 32, 256)
        self.dropout = nn.Dropout(0.4)

        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def forward(self, mel: torch.Tensor, stat: torch.Tensor):
        """
        mel  : (B, 1, 128, 84)
        stat : (B, stat_dim)  — globally normalised
        Returns (main_logits, machine_logits, fault_logits).
        """
        f_mel  = self.mel_stream.extract_features(mel)            # (B, 256)
        f_stat = self.stat_branch(stat)                           # (B, 32)
        fused  = self.dropout(F.relu(self.fc1(
            torch.cat([f_mel, f_stat], dim=1))))                  # (B, 256)
        return (self.head_main(fused),
                self.head_machine(fused),
                self.head_fault(fused))
