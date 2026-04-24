import torch
import torch.nn as nn
import torch.nn.functional as F

class MelCNN(nn.Module):
    def __init__(self,num_classes=6):
        super(MelCNN,self).__init__()
        # Conv blocks = feature extraction (understanding image)
        self.block1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), # we recieve a 1 channel image (mel spectrogram) and output 32 channels( like we are adding more snapshots), kernel size is 3x3, padding is 1 to maintain the same spatial dimensions
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # (128,84)->(64,42)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # (64,42)->(32,21)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # (32,21)->(16,10)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((4,4))  # (16,10)->(4,4)
        )
    # FC layers = decision making (understanding what the image is)
    def forward(self):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x,1) # flatten all dimensions except the batch dimension # (B,256,4,4) -> (B,256*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = self.fc2(x) # 4096 → fc1 → 512 → fc2 → 6

        return x