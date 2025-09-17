import numpy as np

from lab.busca import embaralha
from lab.busca.grafo import Grafo

adj = {
    'A': {'B': 1, 'C': 2},
    'B': {'A': 1, 'D': 3, 'E': 2},
    'C': {'A': 2, 'F': 4, 'G': 2},
    'D': {'B': 3, 'H': 5},
    'E': {'B': 2, 'I': 3},
    'F': {'C': 4, 'J': 1},
    'G': {'C': 2},
    'H': {'D': 5},
    'I': {'E': 3},
    'J': {'F': 1}
}

rnd = np.random.default_rng(5)
grafo = Grafo(adj)
visitados = set()
proximo = "A"
caminho = [proximo]
custo_total = 0
grafo.desenha_grafo(visitados, caminho, proximo)
while proximo != "J":
    sucessores = grafo.sucessores(proximo)
    if not sucessores:
        print("sem sucessores")
        break
    embaralha(sucessores, rnd)
    print(f"sucessores de {proximo}: {sucessores}")
    while sucessores:
        proximo_, custo = sucessores.pop()
        if proximo_ not in visitados:
            proximo = proximo_
            break
    else:
        print("sem sucessores")
        break
    visitados.add(proximo)
    caminho.append(proximo)
    custo_total += custo
    print(proximo, custo_total)
    grafo.desenha_grafo(visitados, caminho, proximo)

grafo.desenha_grafo(visitados, caminho, proximo, pause=True)
