import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from lab.dataset import mnist


def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    with torch.no_grad():
        logits = model(grid_tensor)
        Z = (logits > 0.5).int().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.title(title)
    plt.show()


# Load MNIST and reduce to 2D with PCA
X, y = mnist(1800)
# Binary classification: 0 vs not 0
y_bin = (y == 0).astype(np.float32)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=0)
X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 3), nn.Tanh(),
            nn.Linear(3, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = Classifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(train_loader):
        print(epoch, i)
        outputs = model(data)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plot_decision_boundary(X_train.numpy(), y_train.numpy().flatten(), model, "Tanh")
