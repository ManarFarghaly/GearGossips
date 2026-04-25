"""
Phase 3 — Mel + MFCC + Statistical feature CNN.

Inputs : mel  (B, 1, 128, 84)
         mfcc (B, 3, 40,  84)
         stat (B, N)            where N = number of statistical features (1–5)
Output : (B, 6) logits

Architecture:
  CNN branch (from Phase 2)  → (B, 384)    ← weights from Phase 2
  Statistical branch         → (B, 32)     ← small FC, trained from scratch
  Concatenate                → (B, 416)
  FC(416→256) + ReLU + Dropout(0.4)
  FC(256→6)

Ablation design:
  stat_dim controls how many statistical features are fed in (1–5).
  For the ablation study the CNN branch is FROZEN and only the statistical
  branch + head are trained (fast, ~10 epochs).
  After picking the best stat_dim the whole model is fine-tuned end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from machine_listener.src.models.cnn_MelMFCC import MelMFCCCNN


class MelMFCCStatCNN(nn.Module):

    def __init__(self, num_classes: int = 6, stat_dim: int = 5):
        """
        Args:
            num_classes : number of output classes (6)
            stat_dim    : how many statistical features to accept  (1–5)
                          Controls the ablation study.
        """
        super().__init__()

        # Re-use the Phase 2 model as a feature extractor
        self._phase2 = MelMFCCCNN(num_classes=num_classes)

        # Small FC branch for statistical features
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fusion head
        # 256 (mel) + 128 (mfcc) = 384 from phase2's fc1 output,
        # but we access it before phase2's fc2, so we tap into the 384-dim concat.
        # Total: 384 + 32 = 416
        self.fc1     = nn.Linear(384 + 32, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2     = nn.Linear(256, num_classes)

    def freeze_cnn(self):
        """Freeze the CNN branch during ablation (only train stat branch + head)."""
        for param in self._phase2.mel_stream.parameters():
            param.requires_grad = False
        for param in self._phase2.mfcc_stream.parameters():
            param.requires_grad = False
        # Keep phase2's fc1/fc2 frozen too
        for param in self._phase2.fc1.parameters():
            param.requires_grad = False
        for param in self._phase2.fc2.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        """Unfreeze everything for end-to-end fine-tuning after ablation."""
        for param in self._phase2.parameters():
            param.requires_grad = True

    def forward(
        self,
        mel: torch.Tensor,
        mfcc: torch.Tensor,
        stat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mel  : (B, 1, 128, 84)
            mfcc : (B, 3, 40,  84)
            stat : (B, stat_dim)    — already StandardScaler-normalized
        Returns:
            logits (B, 6)
        """
        # Get 384-dim fused CNN features (before phase2's final classifier)
        f_mel  = self._phase2.mel_stream.extract_features(mel)    # (B, 256)
        f_mfcc = self._phase2.mfcc_stream(mfcc)                   # (B, 128)
        f_cnn  = torch.cat([f_mel, f_mfcc], dim=1)                # (B, 384)

        # Statistical branch
        f_stat = self.stat_branch(stat)                            # (B, 32)

        # Fuse
        x = torch.cat([f_cnn, f_stat], dim=1)                     # (B, 416)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                                            # (B, 6)
        return x
