import torch
import torch.nn as nn
import torch.nn.functional as F

class MFCC(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (40→20)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (20→10)

            nn.AdaptiveAvgPool2d((4,4))
        )

        self.fc = nn.Linear(64*4*4, 128)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x   
    
