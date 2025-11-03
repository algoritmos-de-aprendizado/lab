import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

# Gerar rocambole
n_samples = 2000
X, _ = make_swiss_roll(n_samples)
X = torch.tensor(X, dtype=torch.float32)

# Visualização da variedade
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', s=1)
plt.title("Rocambole (variedade 2D imersa em R³)")
plt.show()


# Estimador simples de dimensionalidade intrínseca baseado em PCA local
def estimate_intrinsic_dim(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    local_dims = []
    for idx in indices:
        neighbors = X[idx[1:]]  # ignorar o próprio ponto
        neighbors -= neighbors.mean(dim=0)
        cov = neighbors.T @ neighbors / k
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals.flip(0)
        eigvals /= eigvals.sum()
        dim = (eigvals ** 2).sum().reciprocal()
        local_dims.append(dim.item())
    return local_dims


# Estimar dimensionalidade local
dims = estimate_intrinsic_dim(X, k=20)

# Visualizar distribuição das estimativas
plt.hist(dims, bins=50, density=True)
plt.xlabel("Dimensionalidade intrínseca estimada")
plt.ylabel("Densidade")
plt.title("Distribuição da dimensionalidade local (PCA)")
plt.show()

# Média e desvio-padrão
mean_dim = np.mean(dims)
std_dim = np.std(dims)
print(f"Dimensionalidade intrínseca média: {mean_dim:.2f} ± {std_dim:.2f}")
