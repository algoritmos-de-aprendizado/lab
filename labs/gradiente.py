import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Função alvo: erro = (x - 3)^2 + 2
def funcao_a_otimizar(x_):
    return (x_ - 3) ** 2 + 2


# Inicialização
x = torch.tensor([0.0], requires_grad=True)
taxa_de_aprendizado = 0.1
passos = 100
coordenadas = []

# Gradiente descendente
for i in range(passos):
    loss = funcao_a_otimizar(x)
    loss.backward()
    coordenadas.append((x.item(), loss.item(), loss))
    with torch.no_grad():
        x -= taxa_de_aprendizado * x.grad
        x.grad.zero_()

# Dados para plot
x_vals = torch.linspace(-1, 10, 200)
y_vals = funcao_a_otimizar(x_vals)

# Plot
fig, ax = plt.subplots()
line, = ax.plot(x_vals, y_vals, 'b-', label='Erro')
point, = ax.plot([], [], 'ro', label='Iteração')
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('Erro')
ax.set_title('Gradiente Descendente')
ax.legend()
ax.grid(True)


# Animação
def update(frame):
    x_val, y_val, loss = coordenadas[frame]
    print(f"{frame=:.3f} {x_val=:.3f} {y_val=:.3f} {loss=}")
    point.set_data([x_val], [y_val])
    text.set_text(f'Iteração {frame}\nx={x_val:.2f}\nErro={y_val:.2f}')
    return point, text


ani = FuncAnimation(fig, update, frames=len(coordenadas), interval=1000, blit=True)
plt.show()
