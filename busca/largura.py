import turtle
import numpy as np

from src.lab.busca import sorteia_de_lista, sorteia_coords
from src.lab.busca.agente import Agente
from src.lab.busca.alvo import Alvo
from src.lab.busca.grade import Grade

rnd = np.random.default_rng(1650)
grade = Grade()
agente = Agente(grade, linha=-19, coluna=-19)
alvo = Alvo(grade, *sorteia_coords(grade, rnd))
direcoes_possiveis = ["norte", "sul", "leste", "oeste"]
visitados = set()
while agente != alvo:
    for direcao in direcoes_possiveis:
        posicao_candidata = agente.move(direcao, simulado=True)
        if posicao_candidata not in visitados:
            agente.move(direcao)
            visitados.add(agente.posicao)
        # fim do IF
    # fim do FOR

turtle.done()
