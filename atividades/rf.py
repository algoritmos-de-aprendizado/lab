from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# deterministic RNG used by augmentation and subsampling
rng = np.random.RandomState(0)

digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
ests = np.arange(1, 100, 1)
acc_tr = []
acc_te = []

# Real-time evolving plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))
line_tr, = ax.plot([], [], label='treino', color='tab:blue')
line_te, = ax.plot([], [], label='teste', color='tab:orange')
ax.set_xlabel('n_estimators')
ax.set_ylabel('acurácia')
ax.set_title('Acurácia vs Número de Árvores (sklearn.datasets.load_digits, 15% label noise)')
ax.set_xlim(ests[0], ests[-1])
ax.set_ylim(0.0, 1.05)
ax.legend()

for i, n in enumerate(ests):
    m = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    acc_tr.append(accuracy_score(ytr, m.predict(Xtr)))
    acc_te.append(accuracy_score(yte, m.predict(Xte)))
    line_tr.set_data(ests[: i + 1], acc_tr)
    line_te.set_data(ests[: i + 1], acc_te)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)

plt.ioff()
plt.show()
if acc_tr and acc_te:
    print(f'Final treino acc: {acc_tr[-1]:.4f}, teste acc: {acc_te[-1]:.4f}')
plt.close()
