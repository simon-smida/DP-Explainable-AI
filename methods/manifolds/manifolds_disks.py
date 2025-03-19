"""
Neural Network Disk Classification - Visualizing Decision Boundaries and Manifold Transformations

Overview:
This project demonstrates how a neural network learns to separate a dataset composed of an inner disk (class 0)
and an outer ring (class 1) by progressively transforming the input space into a more separable representation.
It provides visualizations of both the training process (showing the evolving decision boundary) and the 
layer-by-layer transformations in either 2D or 3D hidden spaces.

Main Components:
1. Disk Dataset Generation: produces a simple dataset of two classes (inner disk vs. outer ring).
2. Neural Network Architecture: a small feedforward network with tanh activations.
3. Training Visualization: continuously updates the decision boundary as the network learns.
4. Feature Space Transformation: interpolates activations across layers to illustrate how the network
   “unwraps” or deforms the data into a more linearly separable form.

What It Demonstrates:
- Manifold Hypothesis: highlights how neural networks can reshape non-linear data distributions.
- Progressive Layer Transformations: each hidden layer incrementally warps the data.
- Decision Boundaries: exhibits how the classification boundary evolves over training.

Useful Resources:
- Theory: https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
- Code Inspiration: https://igreat.github.io/blog/manifold_hypothesis/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D


def generate_disk_dataset(n_points=2000, inner_radius=0.5, outer_radius_range=(1.0, 2.0)):
    """Generates a 2D dataset forming an inner disk (class 0) and an outer ring (class 1)."""
    # Split total points between inner and outer regions
    n_inner = n_points // 2
    n_outer = n_points - n_inner

    # Inner disk
    theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    r_inner = np.random.uniform(0, inner_radius, n_inner)
    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)

    # Outer ring
    theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
    r_outer = np.random.uniform(outer_radius_range[0], outer_radius_range[1], n_outer)
    x_outer = r_outer * np.cos(theta_outer)
    y_outer = r_outer * np.sin(theta_outer)

    # Combine, label, and shuffle
    X = np.vstack([np.column_stack((x_inner, y_inner)),
                   np.column_stack((x_outer, y_outer))])
    y = np.vstack([np.zeros((n_inner, 1)), np.ones((n_outer, 1))])
    indices = np.random.permutation(n_points)
    return X[indices], y[indices]

class DiskClassifier(nn.Module):
    """Simple two-layer NN for binary classification of the disk dataset."""
    def __init__(self, hidden_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Apply tanh to hidden layer; output is a single logit
        hidden = torch.tanh(self.fc1(x))
        return self.fc2(hidden)


def plot_decision_boundary(model, ax, X, y):
    """Plots the model's decision boundary along with the dataset points."""
    margin = 1
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    # Create a grid for evaluating predictions across the region
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predicted probabilities
    with torch.no_grad():
        logits = model(torch.tensor(grid, dtype=torch.float32))
        probs = torch.sigmoid(logits).reshape(xx.shape)

    # Decision boundary at probability=0.5
    ax.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=["blue", "red"])
    ax.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], color='blue', label='Inner Disk', s=20, alpha=0.5)
    ax.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], color='red', label='Outer Ring', s=20, alpha=0.5)
    ax.legend(loc='upper right')

def train_model(model, X_data, y_data, epochs=300, lr=0.01, visualize=True):
    """
    Trains the given model on X_data, y_data using BCEWithLogitsLoss.
    Optionally visualizes the decision boundary during training.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Update plot every 10 epochs
        if visualize and epoch % 10 == 0:
            ax.clear()
            plot_decision_boundary(model, ax,
                                   X_data.detach().numpy(),
                                   y_data.detach().numpy())
            ax.set_xlim([-2.5, 2.5])
            ax.set_ylim([-2.5, 2.5])
            ax.set_title(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            plt.draw()
            plt.pause(0.05)

    if visualize:
        plt.ioff()
        plt.figure()
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    return model, losses

def get_activations(model, x):
    """
    Returns a list of activations from each 'layer':
    [input, hidden_layer_output, output_layer_output].
    If the output is 1D, it's padded so that dimension=2 for visualization.
    """
    acts = []
    # Input
    acts.append(x.detach().numpy())
    # Hidden layer
    hidden = torch.tanh(model.fc1(x))
    acts.append(hidden.detach().numpy())
    # Output layer (pad to 2D if it's just 1D)
    out = model.fc2(hidden)
    if out.shape[1] == 1:
        out = torch.cat([out, torch.zeros_like(out)], dim=1)
    acts.append(out.detach().numpy())
    return acts

def create_grid_mesh(x_min, x_max, y_min, y_max, n_grid=15):
    """
    Creates a set of horizontal and vertical lines (h_lines, v_lines) and
    concentric circles for transformation visualization (of the feature space).
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.linspace(y_min, y_max, n_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Horizontal lines
    h_lines = [np.column_stack((xx[i], yy[i])) for i in range(xx.shape[0])]
    # Vertical lines
    v_lines = [np.column_stack((xx[:, j], yy[:, j])) for j in range(xx.shape[1])]

    # Set of circles of increasing radius
    circles = []
    for radius in np.linspace(0.2, 2.2, 6):
        theta = np.linspace(0, 2 * np.pi, 100)
        circles.append(np.column_stack((radius * np.cos(theta), radius * np.sin(theta))))

    return h_lines, v_lines, circles

def visualize_2d_transformations(model, X_inner, X_outer):
    """Show how points transform from input space to output through the hidden layer."""
    # Convert data to tensors and get activations for inner and outer points
    inner_tensor = torch.tensor(X_inner, dtype=torch.float32)
    outer_tensor = torch.tensor(X_outer, dtype=torch.float32)
    inner_acts = get_activations(model, inner_tensor)
    outer_acts = get_activations(model, outer_tensor)

    # Determine plot boundaries
    margin = 0.5
    x_min = min(X_inner[:, 0].min(), X_outer[:, 0].min()) - margin
    x_max = max(X_inner[:, 0].max(), X_outer[:, 0].max()) + margin
    y_min = min(X_inner[:, 1].min(), X_outer[:, 1].min()) - margin
    y_max = max(X_inner[:, 1].max(), X_outer[:, 1].max()) + margin

    # Prepare lines/circles
    h_lines, v_lines, circles = create_grid_mesh(x_min, x_max, y_min, y_max, n_grid=15)
    h_lines_tensors = [torch.tensor(line, dtype=torch.float32) for line in h_lines]
    v_lines_tensors = [torch.tensor(line, dtype=torch.float32) for line in v_lines]
    circle_tensors = [torch.tensor(circle, dtype=torch.float32) for circle in circles]

    # Get activations for each grid/circle line
    h_lines_acts = [get_activations(model, hl) for hl in h_lines_tensors]
    v_lines_acts = [get_activations(model, vl) for vl in v_lines_tensors]
    circle_acts = [get_activations(model, ct) for ct in circle_tensors]

    layer_names = ["Input", "Hidden Layer", "Output"]
    n_steps = 30  # number of steps per transition (interpolation)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(True, alpha=0.5)
    title_text = fig.suptitle("", fontsize=14)

    # Legend for data points (blue / red)
    legend_ax = fig.add_axes([0.81, 0.025, 0.15, 0.1])
    legend_ax.axis('off')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Inner Disk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outer Ring')
    ]
    legend_ax.legend(handles=legend_handles, loc='center')

    # Animate interpolation from one layer's activations to the next
    for i in range(len(layer_names) - 1):
        for alpha in np.linspace(0, 1, n_steps):
            ax.clear()
            ax.axis('off')

            # Interpolate data points between layer i and i+1
            inter_inner = (1 - alpha) * inner_acts[i] + alpha * inner_acts[i + 1]
            inter_outer = (1 - alpha) * outer_acts[i] + alpha * outer_acts[i + 1]

            # Interpolate grid lines
            for hl_acts in h_lines_acts:
                inter_h = (1 - alpha) * hl_acts[i] + alpha * hl_acts[i + 1]
                ax.plot(inter_h[:, 0], inter_h[:, 1], 'gray', alpha=0.1)
            for vl_acts in v_lines_acts:
                inter_v = (1 - alpha) * vl_acts[i] + alpha * vl_acts[i + 1]
                ax.plot(inter_v[:, 0], inter_v[:, 1], 'gray', alpha=0.1)

            # Interpolate circles
            for c_idx, c_acts in enumerate(circle_acts):
                inter_c = (1 - alpha) * c_acts[i] + alpha * c_acts[i + 1]
                ax.plot(inter_c[:, 0], inter_c[:, 1], 'gray', alpha=0.1)

            # Plot inner/outer points
            ax.scatter(inter_inner[:, 0], inter_inner[:, 1], color="blue", s=20, alpha=0.5)
            ax.scatter(inter_outer[:, 0], inter_outer[:, 1], color="red", s=20, alpha=0.5)

            # Auto scale
            all_x = np.concatenate([inter_inner[:, 0], inter_outer[:, 0]])
            all_y = np.concatenate([inter_inner[:, 1], inter_outer[:, 1]])
            x_lim = (np.min(all_x) - margin, np.max(all_x) + margin)
            y_lim = (np.min(all_y) - margin, np.max(all_y) + margin)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            title_text.set_text(f"Transition: {layer_names[i]} → {layer_names[i+1]} ({alpha:.2f})")
            ax.set_aspect('equal')
            plt.draw()
            plt.pause(0.03)

    plt.ioff()
    plt.show()

def visualize_3d_transformations(model, X_inner, X_outer):
    """Show how points transform from input space to output through the hidden layer (now 3 neurons, not 2)."""
    # Convert data to tensors
    inner_tensor = torch.tensor(X_inner, dtype=torch.float32)
    outer_tensor = torch.tensor(X_outer, dtype=torch.float32)
    inner_acts = get_activations(model, inner_tensor)
    outer_acts = get_activations(model, outer_tensor)

    # Create a few circles to watch them deform in 3D
    circles, circle_acts = [], []
    for radius in np.linspace(0.2, 2.2, 4):
        theta = np.linspace(0, 2 * np.pi, 50)
        circle = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        circles.append(circle)
        circle_acts.append(get_activations(model, torch.tensor(circle, dtype=torch.float32)))

    layer_names = ["Input", "Hidden Layer", "Output"]
    n_steps = 30
    total_steps = (len(layer_names) - 1) * n_steps
    initial_azim, final_azim = 0, 35  # camera rotation angles

    plt.ion()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    title_text = fig.suptitle("", fontsize=14)

    # Legend
    legend_ax = fig.add_axes([0.81, 0.025, 0.15, 0.1])
    legend_ax.axis('off')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Inner Disk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outer Ring')
    ]
    legend_ax.legend(handles=legend_handles, loc='center')

    # Transition through layers
    for i in range(len(layer_names) - 1):
        start_in, end_in = inner_acts[i], inner_acts[i + 1]
        start_out, end_out = outer_acts[i], outer_acts[i + 1]

        # Pad if needed so everything is at least 3D
        max_dim = max(start_in.shape[1], end_in.shape[1])
        if start_in.shape[1] < max_dim:
            start_in = np.pad(start_in, ((0, 0), (0, max_dim - start_in.shape[1])), 'constant')
            start_out = np.pad(start_out, ((0, 0), (0, max_dim - start_out.shape[1])), 'constant')
            for ci in range(len(circle_acts)):
                circle_acts[ci][i] = np.pad(circle_acts[ci][i], ((0, 0), (0, max_dim - circle_acts[ci][i].shape[1])), 'constant')

        if end_in.shape[1] < max_dim:
            end_in = np.pad(end_in, ((0, 0), (0, max_dim - end_in.shape[1])), 'constant')
            end_out = np.pad(end_out, ((0, 0), (0, max_dim - end_out.shape[1])), 'constant')
            for ci in range(len(circle_acts)):
                circle_acts[ci][i + 1] = np.pad(circle_acts[ci][i + 1], ((0, 0), (0, max_dim - circle_acts[ci][i + 1].shape[1])), 'constant')

        # Animate interpolation
        for step, alpha in enumerate(np.linspace(0, 1, n_steps)):
            inter_in = (1 - alpha) * start_in + alpha * end_in
            inter_out = (1 - alpha) * start_out + alpha * end_out

            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # Hide default 3D pane lines
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.xaxis.line.set_color((0, 0, 0, 0))
            ax.yaxis.line.set_color((0, 0, 0, 0))
            ax.zaxis.line.set_color((0, 0, 0, 0))
            ax.grid(False)

            # Interpolate circles
            for ci, cacts in enumerate(circle_acts):
                inter_c = (1 - alpha) * cacts[i] + alpha * cacts[i + 1]
                if abs(circles[ci][0, 0] - 0.5) < 0.1:  # highlight a circle near radius=0.5
                    ax.plot(inter_c[:, 0], inter_c[:, 1], inter_c[:, 2], 'green', alpha=0.2)
                else:
                    ax.plot(inter_c[:, 0], inter_c[:, 1], inter_c[:, 2], 'gray', alpha=0.2)

            # Plot inner / outer data
            ax.scatter(inter_in[:, 0], inter_in[:, 1], inter_in[:, 2], color="blue", s=20, alpha=0.5)
            ax.scatter(inter_out[:, 0], inter_out[:, 1], inter_out[:, 2], color="red", s=20, alpha=0.5)

            global_step = i * n_steps + step
            azim = initial_azim + (final_azim - initial_azim) * (global_step / total_steps)
            ax.view_init(elev=20, azim=azim)  # rotate camera
            title_text.set_text(f"Transition: {layer_names[i]} → {layer_names[i+1]} ({alpha:.2f})")

            # Set axes to have equal range around data
            all_pts = np.vstack([inter_in, inter_out])
            min_x, max_x = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
            min_y, max_y = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
            min_z, max_z = np.min(all_pts[:, 2]), np.max(all_pts[:, 2])
            mid_x = (max_x + min_x) / 2
            mid_y = (max_y + min_y) / 2
            mid_z = (max_z + min_z) / 2
            max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.draw()
            plt.pause(0.05)

    plt.ioff()
    plt.show()


def main():
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Generate dataset ---
    X, y = generate_disk_dataset(n_points=500, outer_radius_range=(1.0, 2.0))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Separate points by class for visualization
    inner_mask = y.flatten() == 0
    outer_mask = y.flatten() == 1
    X_inner, X_outer = X[inner_mask], X[outer_mask]

    # --- Train model with 2D hidden layer ---
    print("Training model with 2D hidden layer...")
    model_2d = DiskClassifier(hidden_dim=2)
    model_2d, losses2d = train_model(model_2d, X_tensor, y_tensor, epochs=300)
    
    print(f"Final loss for 2D model: {losses2d[-1]:.6f}")

    # 2D transformations visualization
    print("Visualizing 2D model layer transitions...")
    visualize_2d_transformations(model_2d, X_inner, X_outer)

    # --- Train model with 3D hidden layer ---
    print("Training model with 3D hidden layer...")
    model_3d = DiskClassifier(hidden_dim=3)
    model_3d, losses3d = train_model(model_3d, X_tensor, y_tensor, epochs=300)

    print(f"Final loss for 3D model: {losses3d[-1]:.6f}")

    # 3D transformations visualization
    print("Visualizing 3D model layer transitions...")
    visualize_3d_transformations(model_3d, X_inner, X_outer)

if __name__ == "__main__":
    main()