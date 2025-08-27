import turtle
import time

import numpy as np

rnd = np.random.default_rng(0)
WIDTH, HEIGHT = 400, 400
CELL = 20

COLS = WIDTH // CELL
ROWS = HEIGHT // CELL

screen = turtle.Screen()
screen.setup(WIDTH + 40, HEIGHT + 40)

backtrack = []

# desenha grid
grid = turtle.Turtle()
grid.hideturtle()
grid.speed(0)
grid.color("lightgray")
for x in range(-WIDTH // 2, WIDTH // 2 + 1, CELL):
    grid.penup()
    grid.goto(x, -HEIGHT // 2)
    grid.pendown()
    grid.goto(x, HEIGHT // 2)
for y in range(-HEIGHT // 2, HEIGHT // 2 + 1, CELL):
    grid.penup()
    grid.goto(-WIDTH // 2, y)
    grid.pendown()
    grid.goto(WIDTH // 2, y)

# alvo
TX = rnd.integers(-COLS // 2 + 1, COLS // 2 - 1) * CELL
TY = rnd.integers(-ROWS // 2 + 1, ROWS // 2 - 1) * CELL
target = turtle.Turtle()
target.hideturtle()
target.speed(0)
target.penup()
target.goto(TX, TY)
target.dot(12, "red")

# agente
t = turtle.Turtle()
t.shape("turtle")
t.color("blue")
t.speed(2)
t.penup()
t.goto(0, 0)
t.pendown()

moves = [(CELL, 0), (-CELL, 0), (0, CELL), (0, -CELL)]
visited = set()
visited.add((t.xcor(), t.ycor()))

while True:
    if t.xcor() == TX and t.ycor() == TY:
        m = turtle.Turtle()
        m.hideturtle()
        m.penup()
        m.goto(TX, TY)
        m.dot(16, "green")
        break

    while True:
        possible_moves = []
        for dx, dy in moves:
            nx = t.xcor() + dx
            ny = t.ycor() + dy
            if -WIDTH / 2 <= nx <= WIDTH / 2 and -HEIGHT / 2 <= ny <= HEIGHT / 2:
                if (nx, ny) not in visited:
                    possible_moves.append((dx, dy))
        if possible_moves:
            break
        dx, dy = backtrack.pop()
        nx = t.xcor() + dx
        ny = t.ycor() + dy
        t.goto(nx, ny)

    print(possible_moves)
    dx, dy = rnd.choice(possible_moves, size=1)[0]
    nx = t.xcor() + dx
    ny = t.ycor() + dy
    t.goto(nx, ny)
    visited.add((nx, ny))
    backtrack.append((-dx, -dy))
    # time.sleep(0.1)

turtle.done()
