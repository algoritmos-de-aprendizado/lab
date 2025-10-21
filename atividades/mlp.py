import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, n_features, n_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes)  # n_classes para dígitos
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(X, y, n_features, n_classes=10, epochs=5, batch_size=128):
    mlp = MLP(n_features, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mlp.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            out = mlp(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    mlp.eval()
    return mlp
