import numpy as np


def sorteia_de_lista(lst, rnd=np.random.default_rng(0)):
    print(rnd.choice(lst, size=1))
    return rnd.choice(lst, size=1)[0]


def sorteia_coords(grade, rnd=np.random.default_rng(0)):
    linha = rnd.integers(1, grade.nlinhas)
    coluna = rnd.integers(1, grade.ncolunas)
    return linha, coluna
