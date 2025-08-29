import turtle


class Alvo:
    def __init__(self, grade, linha, coluna, size=12, color="red"):
        self.linha = linha
        self.coluna = coluna
        self.grade = grade
        t = turtle.Turtle()
        t.hideturtle()
        t.speed(0)
        t.penup()
        t.goto(*grade(linha, coluna))
        t.dot(size, color)

    def recolorir(self, size=16, color="green"):
        m = turtle.Turtle()
        m.hideturtle()
        m.penup()
        m.goto(*self.grade(self.linha, self.coluna))
        m.dot(size, color)
