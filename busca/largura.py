import turtle
from collections import deque

import numpy as np

from src.lab.busca import sorteia_coords
from src.lab.busca.agente import Agente
from src.lab.busca.alvo import Alvo
from src.lab.busca.grade import Grade

rnd = np.random.default_rng(23)
grade = Grade(fps=5)
agente = Agente(grade, 8, 8)
alvo = Alvo(grade, *sorteia_coords(grade, rnd))
visitados = set()
fronteira = deque([agente.posicao])
while agente != alvo and fronteira:
    proximo = fronteira.pop()  # Retira da lista o elemento mais à direita.
    agente.move(*proximo)
    grade.pinta(*proximo, cor="blue")
    visitados.add(proximo)
    for sucessor in agente.sucessores:
        if sucessor not in visitados and sucessor not in fronteira:
            grade.pinta(*sucessor, cor="lightgreen")
            fronteira.appendleft(sucessor)  # Insere elemento à esquerda da fila.
    grade.desenha()

grade.pinta(*agente.posicao, cor="green" if agente == alvo else "black")
grade.desenha()
turtle.done()
