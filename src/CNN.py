import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1 & 2 -> Input: (64, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Layer 3 & 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # CHANGED: 64 * 4 * 4 = 1024 to support 28x28 inputs
            nn.Linear(64 * 4 * 4, 256, bias=False), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Layer 8 & 9
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Layer 10 & 11
            nn.Linear(128, 84, bias=False),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            # Output Layer
            nn.Linear(84, 10) 
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)