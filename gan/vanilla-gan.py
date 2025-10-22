import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from lab.dataset import mnist
from lab.gan.discriminadora import Discriminadora
from lab.gan.geradora import Geradora

X, y = mnist(500)

# Filtra para apenas certas classes
mask = (y == 4) | (y == 7)
X = X[mask]
y = y[mask]

# Obtém dados
data = DataLoader(X, batch_size=10, shuffle=True)

# Prepara G, D e otimizadores
n_features = X.shape[1]
G, D = Geradora(n_features), Discriminadora(n_features)
loss = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0001)
opt_D = optim.Adam(D.parameters(), lr=0.0001)

# Treinamento
for epoch in range(10000):
    for x in data:
        x = x.view(-1, 64)
        b = x.size(0)
        real = torch.ones(b, 1)
        fake = torch.zeros(b, 1)

        # Discriminador
        z = torch.randn(b, 100)
        x_fake = G(z).detach()
        D_loss = loss(D(x), real) + loss(D(x_fake), fake)
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # Gerador
        z = torch.randn(b, 100)
        G_loss = loss(D(G(z)), real)
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    if epoch % 10 == 0:
        print(f"Época {epoch + 1}: Loss D={D_loss.item():.3f}, G={G_loss.item():.3f}")
