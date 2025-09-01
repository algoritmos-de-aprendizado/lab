import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.8
gorilla_x, gorilla_y = 0, 0

enemy_xs = np.array([15, 22, 30, 35, 40])
enemy_ys = np.zeros_like(enemy_xs)

# Tabular data: angles (degrees) and targets (x)
angles = np.array([30, 40, 50, 60, 70])
targets = enemy_xs

# For each (angle, target), compute required force (v0)
def required_force(angle_deg, target_x):
    angle_rad = np.deg2rad(angle_deg)
    # x = v0^2 * sin(2*theta) / g  =>  v0 = sqrt(x * g / sin(2*theta))
    sin2theta = np.sin(2 * angle_rad)
    if sin2theta == 0:
        return np.nan
    v0 = np.sqrt(target_x * g / sin2theta)
    return v0

# Example: animate for a specific (angle, target)
target_idx = 2
angle = 45
target_x = targets[target_idx]
v0 = required_force(angle, target_x)

def ballistic_trajectory(angle_deg, v0, t):
    angle_rad = np.deg2rad(angle_deg)
    x = v0 * np.cos(angle_rad) * t
    y = v0 * np.sin(angle_rad) * t - 0.5 * g * t ** 2
    return x, y

def landing_time(angle_deg, v0):
    angle_rad = np.deg2rad(angle_deg)
    return 2 * v0 * np.sin(angle_rad) / g

T_land = landing_time(angle, v0)
t_vals = np.linspace(0, T_land, 100)
xs, ys = ballistic_trajectory(angle, v0, t_vals)
predicted_x = xs[-1]

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(-5, 45)
ax.set_ylim(-2, 10)
ax.set_aspect('equal')
ax.set_title('Gorilla throws a projectile!')

thrower, = ax.plot(gorilla_x, gorilla_y, 'o', color='brown', markersize=15, label='Thrower')
enemies, = ax.plot(enemy_xs, enemy_ys, 'o', color='red', markersize=15, label='Enemies')
projectile, = ax.plot([], [], 'ko', markersize=8, label='Projectile')
target_marker, = ax.plot([target_x], [0], 'gx', markersize=12, label='Target (enemy)')
pred_marker, = ax.plot([predicted_x], [0], 'bs', markersize=10, label='Predicted landing')

ax.legend(loc='upper right')

def init():
    projectile.set_data([], [])
    return projectile, target_marker, pred_marker

def animate(i):
    projectile.set_data([xs[i]], [ys[i]])
    return projectile, target_marker, pred_marker

ani = FuncAnimation(fig, animate, frames=len(t_vals), init_func=init, blit=True, interval=30, repeat=False)
plt.show()
