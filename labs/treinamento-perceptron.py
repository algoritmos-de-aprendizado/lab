import numpy as np

from src.labrna.perceptron import Perceptron

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
ys = ["0 0 0 1", "0 1 1 1", "0 1 1 0", "1 0 0 1"]
for y_str in ys:
    p = Perceptron(dimensionalidade=2)
    p.aprender(X, np.fromstring(y_str, dtype=int, sep=' '))
    print(p.predizer(X))
