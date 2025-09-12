import turtle


class Alvo:
    def __init__(self, grade, linha, coluna, size=12, cor="red"):
        self.linha = linha
        self.coluna = coluna
        self.grade = grade
        self.posicao = linha, coluna
        self.cor = cor
        self.grade.alvo = self
        t = turtle.Turtle()
        t.hideturtle()
        t.speed(0)
        t.penup()

    def recolore(self, size=16, cor=None):
        if cor is None:
            cor = self.cor
        m = turtle.Turtle()
        m.hideturtle()
        m.penup()
        m.goto(*self.grade(self.linha, self.coluna))
        m.dot(size, cor)

    def __repr__(self):
        return f"Alvo({self.linha}, {self.coluna})"

    def __eq__(self, other):
        return (self.linha, self.coluna) == (other.linha, other.coluna)
