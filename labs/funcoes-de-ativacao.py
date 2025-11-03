import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Entrada de exemplo
x = torch.linspace(-10, 10, 100)

# Funções de ativação
linear = x
degrau = (x >= 0).float()
sigmoide = torch.sigmoid(x)
tanh = torch.tanh(x)
relu = F.relu(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, linear, label='Linear')
plt.plot(x, degrau, label='Degrau')
plt.plot(x, sigmoide, label='Sigmoide Logística')
plt.plot(x, tanh, label='Tangente Hiperbólica')
plt.plot(x, relu, label='ReLU (Linear Retificada)')
plt.legend()
plt.grid(True)
plt.title('Funções de Ativação')
plt.show()
