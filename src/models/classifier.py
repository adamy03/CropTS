import torch
import torch.nn as nn
import torch.nn.functional as F

class CropTypeClassifier(nn.Module):
    def __init__(self, n_classes, input_dim=1024, hidden_dim=512, dropout=0.5):
        super(CropTypeClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, x):
        return self.linear(x)
    