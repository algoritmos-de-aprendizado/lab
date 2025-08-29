import turtle


class Agente:
    def __init__(self, grade, linha, coluna, cor="blue", forma="turtle", vel=2):
        self.moves = {"norte": (0, 1),
                      "sul": (0, -1),
                      "oeste": (1, 0),
                      "leste": (-1, 0)}
        self.grade = grade
        self.linha = linha
        self.coluna = coluna
        self.t = turtle.Turtle(shape=forma)
        self.t.color(cor)
        self.t.speed(vel)
        self.t.penup()
        self.desenha()

    def move(self, direcao, simulado=False):
        self.linha += self.moves[direcao][0]
        self.coluna += self.moves[direcao][1]
        if self.linha < 1:
            self.linha = 1
        if self.coluna < 1:
            self.coluna = 1
        if self.linha > self.grade.nlinhas:
            self.linha = self.grade.nlinhas
        if self.coluna > self.grade.ncolunas:
            self.coluna = self.grade.ncolunas
        if not simulado:
            self.desenha()

    def desenha(self):
        self.t.penup()
        self.t.goto(*self.grade(self.linha, self.coluna))
        self.t.pendown()

    def __eq__(self, other):
        return self.linha == other.linha and self.coluna == other.coluna
