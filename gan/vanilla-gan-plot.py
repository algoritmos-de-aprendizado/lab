import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from atividades.data_utils import get_mnist_data, train_and_eval_mlp, init_plot, close_plot, connect_key_handler
from atividades.plot_utils import update_plot
from atividades.seed_config import set_seeds
from lab.gan.discriminadora import Discriminadora
from lab.gan.geradora import Geradora

window_size = 50

# Configuração de seeds para comportamento determinístico
set_seeds(42)

l = 14
N = 4000
batch_size = 200
X, y, X_teste, y_teste, n_features, data = get_mnist_data(l, N, batch_size)
G, D = Geradora(n_features), Discriminadora(n_features)
loss = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0001)

mlp, acc_teste = train_and_eval_mlp(X, y, X_teste, y_teste, n_features)
print(f'Acurácia de teste da MLP: {acc_teste:.4f}')

fig, ax_img, ax_loss = init_plot()
ax_right = ax_loss.twinx()  # Cria apenas uma vez
losses_D = []
losses_G = []

paused = [False]  # Mutable for handler

def on_key(event):
    if event.key == ' ':  # Espaço
        paused[0] = not paused[0]
        print('Pausado' if paused[0] else 'Retomado')

connect_key_handler(fig, on_key)

for epoch in range(10000):
    for x in data:
        b = x.size(0)
        real = torch.ones(b, 1)
        fake = torch.zeros(b, 1)
        z = torch.randn(b, 100)
        x_fake = G(z).detach()
        D_loss = loss(D(x), real) + loss(D(x_fake), fake)
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()
        z = torch.randn(b, 100)
        G_loss = loss(D(G(z)), real)
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    # Sliding window para losses
    losses_D.append(D_loss.item())
    if len(losses_D) > window_size:
        losses_D = losses_D[-window_size:]
    losses_G.append(G_loss.item())
    if len(losses_G) > window_size:
        losses_G = losses_G[-window_size:]

    z = torch.randn(1, 100)
    sample = G(z).detach()
    pred = D(sample).item()
    logits = mlp(sample)
    pred_digit = torch.argmax(logits, dim=1).item()
    mlp_digit = str(pred_digit + 1)
    prob_pred = torch.softmax(logits, dim=1)[0, pred_digit].item()
    update_plot(ax_img, ax_loss, ax_right, sample, l, pred, mlp_digit, prob_pred, losses_D, losses_G, epoch)
    while paused[0]:
        plt.pause(0.1)
close_plot()
