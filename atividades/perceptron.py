import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import StandardScaler

from lab.perceptron import Perceptron, tangente_hiperbolica


def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predizer(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.title(title)
    plt.show()

# Dua luas
# X_nonlin, y_nonlin = make_moons(n_samples=300, noise=0.2, random_state=0)

# Iris
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
mask = y < 2
X_ = X[mask]
y_ = y[mask]

scaler_lin = StandardScaler()
X_ = scaler_lin.fit_transform(X_)

modelo = Perceptron(dimensionalidade=2, af=tangente_hiperbolica)
modelo.aprender(X_, y_)
plot_decision_boundary(X_, y_, modelo, "Problema Linearmente Separável (Iris)")
