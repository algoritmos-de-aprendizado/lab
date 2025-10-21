import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from atividades.mlp import train_mlp

def get_mnist_data(l=14, N=4000, batch_size=100):
    transform = transforms.Compose([
        transforms.Resize((l, l)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    # Filtra apenas dígitos 1,2,3
    idx = (mnist.targets >= 1) & (mnist.targets <= 2)
    X_all = torch.stack([mnist[i][0].view(-1) for i in range(len(mnist)) if idx[i]])
    y_all = mnist.targets[idx] - 1  # Remapeia: 1->0, 2->1, 3->2, 4->3
    # Seleciona N amostras
    N = min(N, len(X_all))
    indices = torch.randperm(len(X_all))[:N]
    X_all = X_all[indices]
    y_all = y_all[indices]
    X = X_all[:N // 2]
    y = y_all[:N // 2]
    X_teste = X_all[N // 2:]
    y_teste = y_all[N // 2:]
    n_features = X.shape[1]
    data = DataLoader(X, batch_size=batch_size, shuffle=True)
    return X, y, X_teste, y_teste, n_features, data

def train_and_eval_mlp(X, y, X_teste, y_teste, n_features):
    mlp = train_mlp(X, y, n_features, n_classes=4, epochs=5, batch_size=256)
    with torch.no_grad():
        preds_teste = mlp(X_teste)
        pred_digits = torch.argmax(preds_teste, dim=1)
        acc_teste = (pred_digits == y_teste).float().mean().item()
    return mlp, acc_teste

import matplotlib.pyplot as plt

def init_plot():
    plt.ion()
    fig, (ax_img, ax_loss) = plt.subplots(1, 2, figsize=(8, 6))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15)
    return fig, ax_img, ax_loss

def close_plot():
    plt.ioff()
    plt.close()

def connect_key_handler(fig, on_key):
    fig.canvas.mpl_connect('key_press_event', on_key)
