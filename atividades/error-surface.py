import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.lab.dataset import mnist

X, y = mnist(1800)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        return self.net(x)


model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

params = list(model.parameters())
tracked_params = [
    (params[0], (0, 0), params[0], (0, 1), "Layer 1, Neuron 0, Weights 0 & 1"),
    (params[0], (1, 2), params[0], (1, 3), "Layer 1, Neuron 1, Weights 2 & 3"),
    (params[0], (2, 4), params[0], (2, 5), "Layer 1, Neuron 2, Weights 4 & 5"),
    (params[2], (0, 0), params[2], (0, 1), "Layer 2, Neuron 0, Weights 0 & 1"),
    (params[2], (1, 2), params[2], (1, 3), "Layer 2, Neuron 1, Weights 2 & 3"),
    (params[2], (2, 4), params[2], (2, 5), "Layer 2, Neuron 2, Weights 4 & 5"),
]

grid_range = np.linspace(-5, 5, 30)
W1, W2 = np.meshgrid(grid_range, grid_range)
num_epochs = 1000

fig, axes = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(18, 10))
axes = axes.flatten()

for epoch in range(num_epochs):
    for data, target in train_loader:
        outputs = model(data)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for idx, (w1, idx1, w2, idx2, caption) in enumerate(tracked_params):
        Z = np.zeros_like(W1)
        base_val1 = w1.data[idx1].item()
        base_val2 = w2.data[idx2].item()
        with torch.no_grad():
            for p, q in np.ndindex(W1.shape):
                w1.data[idx1] = W1[p, q]
                w2.data[idx2] = W2[p, q]
                outputs = model(X_test)
                Z[p, q] = criterion(outputs, y_test).item()
            w1.data[idx1] = base_val1
            w2.data[idx2] = base_val2

        ax = axes[idx]
        ax.clear()
        ax.plot_wireframe(W1, W2, Z, alpha=0.4)
        ax.set_title(f"{caption}\nEpoch {epoch + 1}")
        ax.set_xlabel("Weight 1")
        ax.set_ylabel("Weight 2")
        ax.set_zlabel("Loss")

    plt.pause(0.1)

    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Test Loss: {test_loss:.4f}")

plt.show()
