import turtle


class Grade:
    def __init__(self, nlinhas=20, ncolunas=20, tamanho_do_no=10):
        self.cell_size, self.nlinhas, self.ncolunas = tamanho_do_no, nlinhas, ncolunas
        self.offset = 40
        width = ncolunas * tamanho_do_no + 2 * self.offset
        height = nlinhas * tamanho_do_no + 2 * self.offset
        screen = turtle.Screen()
        screen.setup(width, height)
        grid = turtle.Turtle()
        grid.hideturtle()
        grid.speed(0)
        grid.color("lightgray")
        xi = -width // 2 + self.offset
        xf = width // 2 - self.offset
        yi = -height // 2 + self.offset
        yf = height // 2 - self.offset
        # TODO: arrumar grade sobrando
        for x in range(xi, xf, tamanho_do_no):
            grid.penup()
            grid.goto(x, yi)
            grid.pendown()
            grid.goto(x, yf)
        for y in range(yi, yf, tamanho_do_no):
            grid.penup()
            grid.goto(xi, y)
            grid.pendown()
            grid.goto(xf, y)

    def __call__(self, linha, coluna):
        x = self.offset + (1 + coluna - self.ncolunas // 2) * self.cell_size
        y = self.offset + (1 + linha - self.nlinhas // 2) * self.cell_size
        return x, y
