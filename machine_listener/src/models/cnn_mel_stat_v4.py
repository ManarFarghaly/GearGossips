"""
Phase 2b V4 model — Mel-spectrogram CNN + Statistical features (19-d).

Architecture:
  MelCNNAttn   : same 4-block CNN as Phase 1 but block4 uses AttentionPool2d
                 instead of AdaptiveAvgPool2d.  Three output heads.
  MelStatCNNV4 : MelCNNAttn (256-d) + stat branch (64-d) → fusion → three heads.

This is the best-performing local model.  The Kaggle checkpoint for this
architecture uses class names MelCNN / MelStatCNN (without the V4 suffix)
so infer.py defines those names inline to match the saved state dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool2d(nn.Module):
    """
    Learnable spatial pooling that replaces AdaptiveAvgPool2d.
    The network learns which time-frequency regions to emphasize before pooling.
    """

    def __init__(self, in_channels: int, out_size: tuple = (4, 4)):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x).flatten(2), dim=-1)
        return self.pool(x * w.view(x.shape[0], 1, x.shape[2], x.shape[3]))


class MelCNNAttn(nn.Module):
    """
    Phase 1 V3 backbone with attention pooling in block 4.
    Three output heads: main (6-class), machine (3-class), fault (binary).
    extract_features() returns the shared 256-d embedding used by MelStatCNNV4.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            AttentionPool2d(256, (4, 4)))
        self.fc1          = nn.Linear(256 * 4 * 4, 256)
        self.dropout      = nn.Dropout(0.5)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x, 1))))

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)
        return self.head_main(feat), self.head_machine(feat), self.head_fault(feat)


class MelStatCNNV4(nn.Module):
    """
    Phase 2b V4: MelCNNAttn (256-d) + statistical branch (64-d) fused into
    three output heads (main, machine, fault).

    Stat branch: Linear(stat_dim→128) → BN → ReLU → Dropout(stat_dropout) → Linear(128→64) → ReLU
    Fusion:      concat(256 + 64) → fc1(320→256) → Dropout(0.5)

    stat_dim = 19 features: rms, zcr, rolloff, bandwidth, spectral_flux, kurtosis, mfcc_1..13

    Stats must be per-machine normalised before passing in.
    Scalers are fitted on training data only and stored inside the checkpoint.
    """

    def __init__(self, num_classes: int = 6, stat_dim: int = 19, stat_dropout: float = 0.5):
        super().__init__()
        self.mel_stream  = MelCNNAttn(num_classes)
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(stat_dropout),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.fc1          = nn.Linear(256 + 64, 256)
        self.dropout      = nn.Dropout(0.5)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def forward(self, mel: torch.Tensor, stat: torch.Tensor):
        """
        mel  : (B, 1, 128, 84)   — mel-spectrogram
        stat : (B, stat_dim)     — per-machine normalised stat features
        Returns (main_logits, machine_logits, fault_logits).
        """
        f_mel  = self.mel_stream.extract_features(mel)
        f_stat = self.stat_branch(stat)
        fused  = self.dropout(F.relu(self.fc1(torch.cat([f_mel, f_stat], dim=1))))
        return self.head_main(fused), self.head_machine(fused), self.head_fault(fused)
