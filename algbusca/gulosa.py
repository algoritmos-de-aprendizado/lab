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


from lab.busca.grafo import Grafo


def gulosa(grafo, inicio, objetivo, animar=True):
    """
    Greedy best-first search on a graph.
    Returns path from inicio to objetivo, or None if not found.
    If animar=True, animates the search in real time.
    """
    from heapq import heappush, heappop
    visitados = set()
    heap = []
    heappush(heap, (grafo.h(inicio), inicio, [inicio]))
    while heap:
        ...

heur = {'A': 2, 'B': 1, 'C': 1, 'D': 0}
grafo = Grafo(adj)
caminho = gulosa(grafo, 'A', 'F', animar=True)
print("Caminho guloso:", caminho)
