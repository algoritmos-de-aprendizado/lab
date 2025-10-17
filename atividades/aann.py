import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from lab.dataset import mnist

X, y = mnist(1000)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.FloatTensor(X_scaled)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class RNAAutoAssociativa(nn.Module):
    """
    A simple autoencoder model with an encoder and decoder.
    """

    def __init__(self):
        super(RNAAutoAssociativa, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 60), nn.ReLU(),
            nn.Linear(60, 30), nn.ReLU(),
            nn.Linear(30, 2)  # Gargalo
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 30), nn.ReLU(),
            nn.Linear(30, 60), nn.ReLU(),
            nn.Linear(60, 64)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


aann = RNAAutoAssociativa()
criterion = nn.MSELoss()
optimizer = optim.Adam(aann.parameters(), lr=0.005)

num_epochs = 100
for epoch in range(num_epochs):
    for data, target in dataloader:
        # Forward
        encoded, decoded = aann(data)
        loss = criterion(decoded, target)

        # Backward e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 3 == 0:
        print(f'Época [{epoch + 1}/{num_epochs}], Valor da função de erro: {loss.item():.4f}')

with torch.no_grad():
    encoded_imgs, _ = aann(X_tensor)
    embeddings = encoded_imgs.numpy()

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit Class')
plt.title('Autoencoder 2D Embedding of MNIST Sample')
plt.show()

# Plot PCA
pca = PCA()
embeddings = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit Class')
plt.title('PCA 2D Embedding of MNIST Sample')
plt.show()

# Plot TSNE
tsne = TSNE()
embeddings = tsne.fit_transform(X_scaled)
plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit Class')
plt.title('TSNE 2D Embedding of MNIST Sample')
plt.show()
