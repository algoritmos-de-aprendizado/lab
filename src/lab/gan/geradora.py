import torch.nn as nn

class Geradora(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, n_features), nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
