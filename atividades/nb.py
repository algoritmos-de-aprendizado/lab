import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.title(title)
    plt.show()


# Iris
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
mask = y < 2
X_lin = X[mask]
y_lin = y[mask]

scaler_lin = StandardScaler()
X_lin = scaler_lin.fit_transform(X_lin)

model_lin = GaussianNB()
model_lin.fit(X_lin, y_lin)
plot_decision_boundary(X_lin, y_lin, model_lin, "Naive Bayes – Problema Linearmente Separável (Iris)")

# Moons
X_nonlin, y_nonlin = make_moons(n_samples=300, noise=0.2, random_state=0)

scaler_nonlin = StandardScaler()
X_nonlin = scaler_nonlin.fit_transform(X_nonlin)

model_nonlin = GaussianNB()
model_nonlin.fit(X_nonlin, y_nonlin)
plot_decision_boundary(X_nonlin, y_nonlin, model_nonlin, "Naive Bayes – Problema Não Linearmente Separável (Moons)")

# XOR
rnd = np.random.default_rng(0)
n_points = 100
X_xor = rnd.integers(0, 2, (n_points, 2))
y_xor = np.logical_xor(X_xor[:, 0], X_xor[:, 1]).astype(int)
model_xor = GaussianNB()
model_xor.fit(X_xor, y_xor)
plot_decision_boundary(X_xor, y_xor, model_xor, "Naive Bayes – Problema XOR")

