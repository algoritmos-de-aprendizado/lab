import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lab.dataset import mnist

torch.manual_seed(0)
X, y = mnist(100)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
input_size = X.shape[1]


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        out, hn = self.rnn(x, h0)
        return self.fc(out), hn


def train_with_live_plot(hidden_size, lr, clip=None, steps=300, delay=0.05, case_title=""):
    model = SimpleRNN(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses, grad_norms, loss_changes = [], [], []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
    plt.ion()
    show = True
    previous_loss = None

    for step in range(steps):
        optimizer.zero_grad()
        x_input = X.unsqueeze(0)
        y_pred, _ = model(x_input)
        loss = criterion(y_pred.squeeze(0), y.unsqueeze(1))
        loss.backward()

        grad_norm = sum((p.grad.norm() ** 2 for p in model.parameters())) ** 0.5
        grad_norms.append(grad_norm.item())
        losses.append(loss.item())

        # Calculate loss change (convergence rate)
        if previous_loss is not None:
            loss_change = abs(loss.item() - previous_loss)
            loss_changes.append(loss_change)
        previous_loss = loss.item()

        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if step % 5 == 0:
            ax1.cla()
            ax2.cla()
            ax3.cla()

            # Plot 1
            ax1.plot(grad_norms, label="Norma do gradiente", color='blue', linewidth=2)
            ax1.axhline(y=1e-5, color='red', linestyle='--', label='Limite vanishing (1e-5)')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.set_title(f"{case_title} - Norma do Gradiente")
            ax1.set_ylabel("Norma do Gradiente (log)")
            ax1.grid(True, alpha=0.3)

            # Plot 2
            ax2.plot(losses, label="Perda (MSE)", color='green', linewidth=2)
            if len(losses) > 10:
                recent_min = min(losses[-len(losses) // 10:])
                ax2.axhline(y=recent_min, color='red', linestyle='--',
                            label=f'Melhor recente: {recent_min:.4f}')
            ax2.legend()
            ax2.set_title(f"{case_title} - Evolução da Perda")
            ax2.set_ylabel("Loss (MSE)")
            ax2.grid(True, alpha=0.3)

            # Plot 3
            if len(loss_changes) > 1:
                ax3.plot(range(1, len(loss_changes) + 1), loss_changes,
                         label="Taxa de mudança da loss", color='orange', linewidth=2)
                ax3.axhline(y=1e-6, color='red', linestyle='--',
                            label='Limite estagnação (1e-6)')
                ax3.set_yscale('log')
                ax3.legend()
                ax3.set_title("Taxa de Convergência")
                ax3.set_xlabel("Steps de Treinamento")
                ax3.set_ylabel("Δ Loss (log)")
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(delay)
            if show:
                plt.show()
                show = False

    final_loss = losses[-1]
    final_grad_norm = grad_norms[-1]
    avg_final_10_loss = np.mean(losses[-10:])
    avg_initial_10_loss = np.mean(losses[:10])

    improvement_ratio = (avg_initial_10_loss - avg_final_10_loss) / avg_initial_10_loss

    print(f"\n=== ANÁLISE: {case_title} ===")
    print(f"Loss final: {final_loss:.6f}")
    print(f"Norma do gradiente final: {final_grad_norm:.2e}")
    print(f"Melhoria na acurácia do início ao fim: {improvement_ratio * 100:.1f}%")

    plt.ioff()
    plt.show(block=True)


print("Caso 1: Explosão de gradiente (lr=0.1, sem clipping)")
train_with_live_plot(hidden_size=64, lr=0.1, clip=None, steps=20, case_title="Caso 1: Explosão de gradiente")

print("\n" + "=" * 60)
print("Caso 2: Dissipação de gradiente (lr=0.001, sem clipping)")
train_with_live_plot(hidden_size=64, lr=0.001, clip=None, steps=500, case_title="Caso 2: Dissipação de gradiente")

print("\n" + "=" * 60)
print("Caso 3: Controle com gradient clipping (lr=0.1, clip=1.0)")
train_with_live_plot(hidden_size=64, lr=0.1, clip=1.0, steps=500, case_title="Caso 3: Com gradient clipping")
