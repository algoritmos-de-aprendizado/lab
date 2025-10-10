####################################
import torch
# import torch.nn.functional as F
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import threading

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['text.antialiased'] = True

torch.manual_seed(0)
np.random.seed(0)

# Lista de instâncias de entrada (add bias column of 1s)
x_list = [
    torch.tensor([[1.0, 0.0, 0.0]]),  # [bias, x1, x2]
    torch.tensor([[1.0, 1.0, 0.0]]),
    torch.tensor([[1.0, 0.0, 1.0]]),
    torch.tensor([[1.0, 1.0, 1.0]])
]
y_list = [torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[0.0]])]
current_instance = [0]

x = torch.tensor([[1.0, 1.0, 0.0]])  # [bias, x1, x2]
y = torch.tensor([[1.0]])

# Improved initial weights for faster XOR learning
W1 = torch.tensor([[0.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], requires_grad=True)  # 3x2
W2 = torch.tensor([[-1.0], [1.5], [1.5]], requires_grad=True)  # 3x1 (includes bias)

def forward(x):
    z1 = x @ W1
    a1_raw = torch.sigmoid(z1)
    # Add bias column to hidden layer
    a1 = torch.cat([torch.ones(a1_raw.shape[0], 1), a1_raw], dim=1)
    z2 = a1 @ W2
    a2 = torch.sigmoid(z2)
    return a1_raw, a2

a1, a2 = forward(x)
loss = 0.5 * (y - a2) ** 2
loss.backward()

lr = 0.1
new_W1 = (W1 - lr * W1.grad).detach().numpy()
new_W2 = (W2 - lr * W2.grad).detach().numpy()

nodes = {
    'x1': (-3, 2.4), 'x2': (-3, 0.8),
    'h1': (0, 3.0), 'h2': (0, 0.2),
    'y':  (3, 1.6)
}
edges = [
    ('x1','h1'), ('x2','h1'),
    ('x1','h2'), ('x2','h2'),
    ('h1','y'), ('h2','y')
]
weights = {
    ('x1','h1'): W1[1,0].item(),
    ('x2','h1'): W1[2,0].item(),
    ('x1','h2'): W1[1,1].item(),
    ('x2','h2'): W1[2,1].item(),
    ('h1','y'):  W2[1,0].item(),
    ('h2','y'):  W2[2,0].item()
}

acts = {'x1': x[0,1].item(), 'x2': x[0,2].item(), 'h1': 0, 'h2': 0, 'y': 0}
target_acts = {'h1': a1[0,0].item(), 'h2': a1[0,1].item(), 'y': a2.item()}

fig, (ax, ax_boundary) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [2, 1]})
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 4)
ax.axis('off')
ax.set_title('Rede Neural')

ax_boundary.set_xlim(-0.2, 1.2)
ax_boundary.set_ylim(-0.2, 1.2)
ax_boundary.set_title('Decision Boundary')
ax_boundary.set_xlabel('x1')
ax_boundary.set_ylabel('x2')

# Plot training points
for i, xpt in enumerate(x_list):
    ax_boundary.scatter(xpt[0,1], xpt[0,2], c='C1' if y_list[i].item() > 0.5 else 'C0', s=120, edgecolor='k', linewidth=2, marker='o', zorder=3)

# Draw weight connections (edges) between nodes
for (u, v) in edges:
    x0, y0 = nodes[u]
    x1, y1 = nodes[v]
    ax.plot([x0, x1], [y0, y1], color='gray', lw=3, zorder=1, alpha=0.7)

# Create node text objects
node_texts = {}
for n, (x, y) in nodes.items():
    act_val = acts[n] if n in acts else 0
    node_texts[n] = ax.text(x, y, f"{n}\n{act_val:.3f}", fontsize=18, ha='center', va='center', fontweight='bold', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.6'))

# Create edge text objects
edge_texts = {}
for (u, v) in edges:
    x0, y0 = nodes[u]
    x1, y1 = nodes[v]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    val = weights[(u, v)]
    edge_texts[(u, v)] = ax.text(xm, ym, f"{val:.3f}", fontsize=16, ha='center', va='center', color='black', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.6'))

# Add bias annotations to hidden and output nodes
bias_texts = {}
for i, n in enumerate(['h1', 'h2']):
    x, y = nodes[n]
    bias_texts[n] = ax.text(x, y-0.5, f"b={W1[0,i].item():.3f}", fontsize=14, ha='center', va='center', color='purple', bbox=dict(facecolor='white', edgecolor='purple', boxstyle='round,pad=0.8'))
x, y = nodes['y']
bias_texts['y'] = ax.text(x, y-0.5, f"b={W2[0,0].item():.3f}", fontsize=14, ha='center', va='center', color='purple', bbox=dict(facecolor='white', edgecolor='purple', boxstyle='round,pad=0.8'))

boundary_img = None

def plot_decision_boundary(W1, W2):
    global boundary_img
    xx, yy = torch.meshgrid(torch.linspace(0, 1, 200), torch.linspace(0, 1, 200), indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    # Add bias column
    grid_with_bias = torch.cat([torch.ones(grid.shape[0], 1), grid], dim=1)
    with torch.no_grad():
        # Ensure W1 and W2 are tensors
        if not isinstance(W1, torch.Tensor):
            W1 = torch.tensor(W1)
        if not isinstance(W2, torch.Tensor):
            W2 = torch.tensor(W2)

        z1 = grid_with_bias @ W1
        a1_raw = torch.sigmoid(z1)
        a1 = torch.cat([torch.ones(a1_raw.shape[0], 1), a1_raw], dim=1)
        z2 = a1 @ W2
        a2 = torch.sigmoid(z2).reshape(xx.shape)

    # Remove previous hidden neuron lines and legend FIRST
    if hasattr(ax_boundary, '_hidden_lines'):
        for l in ax_boundary._hidden_lines:
            l.remove()
    ax_boundary._hidden_lines = []

    if boundary_img is not None:
        boundary_img.remove()

    # Create a custom color map: light blue for class 0, light orange for class 1
    cmap = mpl.colors.ListedColormap(['#add8e6', '#ffe4b5'])
    # Use the predicted class (0 or 1) to color the background
    predicted_classes = (a2 > 0.5).long().reshape(xx.shape)
    boundary_img = ax_boundary.imshow(predicted_classes.numpy(), extent=[0,1,0,1], origin='lower', cmap=cmap, alpha=0.7, aspect='auto', zorder=1)
    ax_boundary.contour(xx.numpy(), yy.numpy(), a2.numpy(), levels=[0.5], colors='k', linewidths=2, linestyles='--', antialiased=True, zorder=2)

    # Plot hidden neuron boundaries AFTER the heatmap
    colors = ['orange', 'purple']
    lines_plotted = 0
    for i in range(W1.shape[1]):
        # W1 has structure [bias; w1; w2] for each hidden neuron
        if isinstance(W1, torch.Tensor):
            w1 = W1[1,i].detach().item()  # weight for x1
            w2 = W1[2,i].detach().item()  # weight for x2
            b = W1[0,i].detach().item()   # bias
        else:
            w1 = W1[1,i]
            w2 = W1[2,i]
            b = W1[0,i]
        # Plot line: w1*x1 + w2*x2 + b = 0
        if abs(w2) > 1e-6:
            x_vals = np.array([-0.2, 1.2])
            y_vals = -(w1*x_vals + b)/w2
            line, = ax_boundary.plot(x_vals, y_vals, color=colors[i%2], lw=4, ls='-',
                                    label=f'h{i+1} boundary', alpha=1.0, zorder=10,
                                    marker='o', markersize=8)
            ax_boundary._hidden_lines.append(line)
            lines_plotted += 1
        elif abs(w1) > 1e-6:
            x_val = -b/w1
            if -0.2 <= x_val <= 1.2:
                y_vals = np.array([-0.2, 1.2])
                x_vals = np.full_like(y_vals, x_val)
                line, = ax_boundary.plot(x_vals, y_vals, color=colors[i%2], lw=4, ls='-',
                                        label=f'h{i+1} boundary', alpha=1.0, zorder=10,
                                        marker='o', markersize=8)
                ax_boundary._hidden_lines.append(line)
                lines_plotted += 1

    # print(f"Total lines plotted: {lines_plotted}")

    if hasattr(ax_boundary, '_legend') and ax_boundary._legend:
        ax_boundary._legend.remove()
    ax_boundary._legend = ax_boundary.legend(loc='upper right', fontsize=12)
    ax_boundary.figure.canvas.draw_idle()

fluid_lines = {e: ax.plot([], [], color='C0', lw=10, alpha=0.8)[0] for e in edges}

timeline = [
    ('edge','x1','h1',0,20), ('edge','x2','h1',0,20),
    ('activate','h1',20),
    ('edge','x1','h2',20,40), ('edge','x2','h2',20,40),
    ('activate','h2',40),
    ('edge','h1','y',40,60), ('edge','h2','y',40,60),
    ('activate','y',60),
    # Backpropagation: error from y to h1/h2
    ('edge_back','y','h1',60,80), ('edge_back','y','h2',60,80),
    ('activate_err','h1',80), ('activate_err','h2',80),
    # Error from h1/h2 to x1/x2 (new)
    ('edge_back','h1','x1',80,100), ('edge_back','h1','x2',80,100),
    ('edge_back','h2','x1',80,100), ('edge_back','h2','x2',80,100),
    ('activate_err','x1',100), ('activate_err','x2',100),
    # Update weights (all)
    ('update_w',100)
]

def lerp(a,b,t): return a + (b-a)*t

# --- Individual weight update after back fluid passes ---
# Map each backprop edge (from, to) to the forward edge that needs updating and the frame
backprop_to_forward_edge = {
    ('y','h1'): ('h1','y'),
    ('y','h2'): ('h2','y'),
    ('h1','x1'): ('x1','h1'),
    ('h1','x2'): ('x2','h1'),
    ('h2','x1'): ('x1','h2'),
    ('h2','x2'): ('x2','h2'),
}

# Store which weights have been updated
updated_weights = set()

def frame(t):
    for (u, v), line in fluid_lines.items():
        line.set_data([], [])
    updated_this_frame = set()
    # Get current input for this instance
    x = x_list[current_instance[0] % len(x_list)]
    acts['x1'] = x[0,1].item()
    acts['x2'] = x[0,2].item()
    # Use latest weights for forward pass
    a1, a2 = forward(x)
    # Draw fluid and update activations when fluid reaches node
    for ev in timeline:
        typ, *args = ev
        if typ == 'edge':
            u, v, t0, t1 = args
            if t0 <= t < t1:
                frac = (t-t0)/(t1-t0)
                x0, y0 = nodes[u]; x1, y1 = nodes[v]
                xn, yn = lerp(x0, x1, frac), lerp(y0, y1, frac)
                fluid_lines[(u, v)].set_data([x0, xn], [y0, yn])
            if t >= t1 - 1:
                # Update activation only when fluid reaches the node
                if v == 'h1':
                    acts['h1'] = a1[0,0].item()
                    node_texts['h1'].set_text(f"h1\n{acts['h1']:.3f}")
                elif v == 'h2':
                    acts['h2'] = a1[0,1].item()
                    node_texts['h2'].set_text(f"h2\n{acts['h2']:.3f}")
                elif v == 'y':
                    acts['y'] = a2.item()
                    node_texts['y'].set_text(f"y\n{acts['y']:.3f}")
    # Always update node text for x1, x2
    node_texts['x1'].set_text(f"x1\n{acts['x1']:.3f}")
    node_texts['x2'].set_text(f"x2\n{acts['x2']:.3f}")
    plot_decision_boundary(W1, W2)
    return list(fluid_lines.values()) + list(node_texts.values()) + list(edge_texts.values())

# --- Compute pause frames (activation or weight update) ---
pause_frames = []
for ev in timeline:
    typ, *args = ev
    if typ == 'edge':
        u,v,t0,t1 = args
        pause_frames.append(t1-1)  # activation update
    elif typ == 'edge_back':
        u,v,t0,t1 = args
        if (u,v) in backprop_to_forward_edge:
            pause_frames.append(t1-1)  # weight update
pause_frames = sorted(set(pause_frames))

current_pause_idx = [0]
current_frame = [0]

# Função para atualizar pesos, ativação e textos para nova instância
def update_for_new_instance():
    global W1, W2, acts, target_acts, new_W1, new_W2
    x = x_list[current_instance[0] % len(x_list)]
    y = y_list[current_instance[0] % len(y_list)]
    W1 = torch.tensor(new_W1, requires_grad=True)
    W2 = torch.tensor(new_W2, requires_grad=True)
    a1, a2 = forward(x)
    loss = 0.5 * (y - a2) ** 2
    loss.backward()
    lr = 0.5
    new_W1 = (W1 - lr * W1.grad).detach().numpy()
    new_W2 = (W2 - lr * W2.grad).detach().numpy()
    # acts = {'x1': x[0,1].item(), 'x2': x[0,2].item(), 'h1': 0, 'h2': 0, 'y': 0}
    acts.update({'x1': x[0,1].item(), 'x2': x[0,2].item()})
    target_acts = {'h1': a1[0,0].item(), 'h2': a1[0,1].item(), 'y': a2.item()}
    for n in node_texts:
        node_texts[n].set_color('black')
    node_texts['x1'].set_text(f"x1\n{acts['x1']:.3f}")
    node_texts['x2'].set_text(f"x2\n{acts['x2']:.3f}")
    node_texts['h1'].set_text(f"h1\n{acts['h1']:.3f}")
    node_texts['h2'].set_text(f"h2\n{acts['h2']:.3f}")
    node_texts['y'].set_text(f"y\n{acts['y']:.3f}")
    for i, n in enumerate(['h1', 'h2']):
        bias_texts[n].set_text(f"b={W1[0,i].item():.3f}")
    bias_texts['y'].set_text(f"b={W2[0,0].item():.3f}")
    for (u,v) in edges:
        val = W1[1,0].item() if v=='h1' and u=='x1' else \
            W1[2,0].item() if v=='h1' and u=='x2' else \
                W1[1,1].item() if v=='h2' and u=='x1' else \
                    W1[2,1].item() if v=='h2' and u=='x2' else \
                        W2[1,0].item() if v=='y' and u=='h1' else W2[2,0].item()
        edge_texts[(u,v)].set_text(f"{val:.3f}")
        edge_texts[(u,v)].set_color('black')
    for e in fluid_lines:
        fluid_lines[e].set_color('C0')
        fluid_lines[e].set_data([], [])

# Create node text objects
node_texts = {}
for n, (x, y) in nodes.items():
    act_val = acts[n] if n in acts else 0
    node_texts[n] = ax.text(x, y, f"{n}\n{act_val:.3f}", fontsize=18, ha='center', va='center', fontweight='bold', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.8'))

# Create edge text objects
edge_texts = {}
for (u, v) in edges:
    x0, y0 = nodes[u]
    x1, y1 = nodes[v]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    val = weights[(u, v)]
    edge_texts[(u, v)] = ax.text(xm, ym, f"{val:.3f}", fontsize=16, ha='center', va='center', color='black', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.8'))

is_playing = [False]
frame_step = 4  # Advance 4 frames per timer tick for faster fluid movement
play_interval = 0  # milliseconds between frame updates (set to 0 for fastest animation)
play_timer = [None]

# Function to advance frames while playing
def play_animation():
    if is_playing[0]:
        current_frame[0] += frame_step
        max_frame = timeline[-1][-1] if timeline else 30
        if current_frame[0] > max_frame:
            current_frame[0] = 0
            updated_weights.clear()  # Reset for repeat
            current_instance[0] += 1
            update_for_new_instance()
        frame(current_frame[0])
        fig.canvas.draw_idle()
        # Schedule next frame
        play_timer[0] = fig.canvas.new_timer(interval=play_interval)
        play_timer[0].add_callback(play_animation)
        play_timer[0].start()

# Key event toggles play/pause

def on_key(event):
    if event.key:
        is_playing[0] = not is_playing[0]
        if is_playing[0]:
            # Start playing
            play_animation()
        else:
            # Pause: stop timer
            if play_timer[0] is not None:
                play_timer[0].stop()
                play_timer[0] = None

# Inicializa primeira instância
update_for_new_instance()

fig.canvas.mpl_connect('key_press_event', on_key)
frame(0)
plt.tight_layout()
plt.show()
# --- Remove FuncAnimation ---
