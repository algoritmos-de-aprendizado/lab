import numpy as np

from src.lab.busca.alvo import Alvo


def sorteia_de_lista(lst, rnd):
    return rnd.choice(lst, size=1)[0]


def embaralha(lst, rnd):
    rnd.shuffle(lst)


def sorteia_coords(grade, rnd=np.random.default_rng(0)):
    linha = rnd.integers(1, grade.nlinhas + 1)
    coluna = rnd.integers(1, grade.ncolunas + 1)
    return linha, coluna
