import turtle
from collections import deque

import numpy as np

from lab.busca import sorteia_coords, embaralha
from lab.busca.agente import Agente
from lab.busca.alvo import Alvo
from lab.busca.grade import Grade

rnd = np.random.default_rng(5)
grade = Grade(fps=5)
agente = Agente(grade, linha=10, coluna=10)
alvo = Alvo(grade, *sorteia_coords(grade, rnd))
visitados = set()
sucessores = deque([agente.posicao])
while agente != alvo and sucessores:
    embaralha(sucessores, rnd)
    proximo = sucessores.pop()
    sucessores.clear()
    agente.move(*proximo)
    visitados.add(proximo)
    for sucessor in agente.sucessores:
        if sucessor not in visitados:
            grade.pinta(*sucessor, cor="lightgreen")
            sucessores.append(sucessor)
    grade.pinta(*agente.posicao, cor="blue")
    grade.desenha()

grade.pinta(*agente.posicao, cor="green" if agente == alvo else "black")
grade.desenha()
turtle.done()
