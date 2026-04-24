import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_baseline import MelCNN
from cnn_MFCC import MFCC

# Stream 1 (mel-spec):  Input (B, 1, 128, 84)
#   → same Block 1-4 from MelCNN  → AdaptiveAvgPool → Flatten → (B, 4096) → FC(4096→256)
# Stream 2 (MFCC+deltas):  Input (B, 3, 40, 84)
#   → Block A: Conv2d(3→32, 3×3) + BN + ReLU + MaxPool  → (B, 32, 20, 42)
#   → Block B: Conv2d(32→64, 3×3) + BN + ReLU + MaxPool → (B, 64, 10, 21)
#   → AdaptiveAvgPool(4,4) → Flatten → (B, 1024) → FC(1024→128)
# Concatenate: (B, 384)
# FC(384→256) + ReLU + Dropout(0.4)
# FC(256→num_classes)

class MelMFCCCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.mel_stream = MelCNN()
        self.mfcc_stream = MFCC()

        self.fc1 = nn.Linear(256 + 128, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, mel, mfcc):
        m1 = self.mel_stream(mel)    # (B,256)
        m2 = self.mfcc_stream(mfcc)  # (B,128)

        x = torch.cat([m1, m2], dim=1)  # (B,384)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
