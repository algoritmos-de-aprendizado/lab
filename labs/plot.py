import torch
from matplotlib import pyplot as plt

pontos2d = torch.randn(500, 2)  # 2D
plt.scatter(pontos2d[:, 0], pontos2d[:, 1])
plt.show()

pontos2d = pontos2d.sort(dim=0)[0]
plt.plot(pontos2d[:, 0], pontos2d[:, 1])
plt.show()
