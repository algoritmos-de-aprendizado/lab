import turtle
from time import sleep, time

from lab.busca import Alvo


class Grade:
    alvo: Alvo

    def __init__(self, nlinhas=15, ncolunas=15, tamanho_do_no=30, fps=10):
        self.tamanho_do_no, self.nlinhas, self.ncolunas = tamanho_do_no, nlinhas, ncolunas
        width = ncolunas * tamanho_do_no + 100
        height = nlinhas * tamanho_do_no + 100
        self.screen = turtle.Screen()
        self.screen.setup(width, height)
        self.screen.tracer(0, 0)
        self.fps = fps
        self.grid = turtle.Turtle()
        self.grid.hideturtle()
        self.grid.speed(0)
        self.grid.color("lightgray")
        self.inicio = time()

        self.xi = - (ncolunas * tamanho_do_no) // 2
        self.yi = - (nlinhas * tamanho_do_no) // 2
        xf = self.xi + ncolunas * tamanho_do_no
        yf = self.yi + nlinhas * tamanho_do_no

        for x in range(self.xi, xf + 1, tamanho_do_no):
            self.grid.penup()
            self.grid.goto(x, self.yi)
            self.grid.pendown()
            self.grid.goto(x, yf)

        for y in range(self.yi, yf + 1, tamanho_do_no):
            self.grid.penup()
            self.grid.goto(self.xi, y)
            self.grid.pendown()
            self.grid.goto(xf, y)

        self.pincel = turtle.Turtle()
        self.pincel.hideturtle()
        self.pincel.speed(0)
        self.screen.update()

    def desenha(self):
        espera = 1 / self.fps - (time() - self.inicio)
        if espera > 0:
            sleep(espera)
        self.inicio = time()
        self.alvo.recolore()
        self.screen.update()

    def pinta(self, l, c, cor):
        self.pincel.penup()
        self.pincel.goto(self(l + 0.5, c - 0.5))
        self.pincel.pendown()
        self.pincel.fillcolor(cor)
        self.pincel.begin_fill()
        for _ in range(4):
            self.pincel.forward(self.tamanho_do_no)
            self.pincel.left(90)
        self.pincel.end_fill()

    def __call__(self, linha, coluna):
        x = self.xi + (coluna - 0.5) * self.tamanho_do_no
        y = self.yi + (self.nlinhas - linha + 0.5) * self.tamanho_do_no
        return x, y
