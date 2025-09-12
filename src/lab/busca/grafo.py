import matplotlib.pyplot as plt
import networkx as nx


class Grafo:
    def __init__(self, adjacencia, heuristica=None):
        self.adjacencia = adjacencia
        self.heuristica = heuristica or (lambda n: 0)
        self.G = nx.DiGraph()
        for no, sucessores in adjacencia.items():
            for sucessor, custo in sucessores.items():
                self.G.add_edge(no, sucessor, weight=custo)
        self.pos = nx.spring_layout(self.G, seed=42)
        self.fig, self.ax = None, None

    def sucessores(self, no):
        return list(self.adjacencia.get(no, {}).items())

    def custo(self, no, sucessor):
        return self.adjacencia.get(no, {}).get(sucessor, float('inf'))

    def h(self, no):
        if callable(self.heuristica):
            return self.heuristica(no)
        return self.heuristica.get(no, 0)

    def ajusta_layout_arvore(self):
        """
        Arrange nodes in a tree layout: origin ('A') at the top, target (last letter alphabetically) at the bottom,
        other nodes distributed by BFS level, siblings spread horizontally.
        """
        nodes = list(self.G.nodes)
        if not nodes:
            return
        origin = 'A' if 'A' in nodes else nodes[0]
        # Find target as last letter alphabetically
        target = sorted(nodes)[-1]
        # BFS to assign levels
        from collections import deque, defaultdict
        level = {origin: 0}
        queue = deque([origin])
        while queue:
            node = queue.popleft()
            for succ in self.G.successors(node):
                if succ not in level:
                    level[succ] = level[node] + 1
                    queue.append(succ)
        # Group nodes by level
        levels = defaultdict(list)
        for node, l in level.items():
            levels[l].append(node)
        max_level = max(levels.keys()) if levels else 1
        # Assign positions
        pos = {}
        for l in range(max_level + 1):
            nodes_at_level = levels[l]
            n = len(nodes_at_level)
            for i, node in enumerate(sorted(nodes_at_level)):
                # Spread nodes horizontally in [-1, 1]
                x = (i - (n - 1) / 2) / max(n, 1)
                # y: origin at top, target at bottom
                if node == origin:
                    y = 1
                elif node == target:
                    y = -1
                else:
                    y = 1 - 2 * l / max_level if max_level > 0 else 0
                pos[node] = (x, y)
        # For nodes not reached from origin, put at bottom
        for node in nodes:
            if node not in pos:
                pos[node] = (0, -2)
        self.pos = pos

    def desenha_grafo(self, visitados=None, caminho=None, atual=None, pause=False):
        self.ajusta_layout_arvore()
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.clear()
        visitados = set(visitados or [])
        caminho = caminho or []
        cores_nos = []
        for n in self.G.nodes:
            if n == atual:
                cores_nos.append('blue')
            elif n in visitados:
                cores_nos.append('lightgreen')
            else:
                cores_nos.append('lightgray')
        cores_arestas = []
        percorridas = set(zip(caminho, caminho[1:]))
        for e in self.G.edges:
            if e in percorridas:
                cores_arestas.append('green')
            else:
                cores_arestas.append('gray')
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True,
                node_color=cores_nos, edge_color=cores_arestas,
                node_size=600, arrowsize=20, font_size=10)
        rotulos = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=rotulos, ax=self.ax)
        plt.pause(0.3)
        if pause:

            def on_key(event):
                if event.key == 'escape':
                    plt.close(self.fig)

            self.fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()

    def resetar_plot(self):
        self.fig, self.ax = None, None
        plt.close('all')
