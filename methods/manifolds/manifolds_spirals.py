"""
Neural Network Manifold Learning - Visualizing Decision Boundaries and Feature Transformations

Overview:
This project demonstrates how a neural network learns to separate a complex dataset (two interwoven spirals)
by progressively transforming the input space into a more separable representation. It visualizes both
the *training process* and the *feature space transformations* at each layer.

Main Components:
1. Spiral Dataset Generation: creates two interwoven spirals as a challenging classification problem.
2. Neural Network Model: a fully connected feedforward network with tanh activations.
3. Training Process Animation: shows the evolving decision boundary as the model learns.
4. Feature Space Transformation Animation: smoothly interpolates activations across layers to 
   visualize how the network "unfolds" the spirals into a separable space.

What It Demonstrates:
- Manifold Hypothesis: neural networks reshape complex data into linearly separable forms.
- Layer Transformations: each layer applies nonlinear warping to the data.
- Decision Boundaries: how neural networks refine classification boundaries over time.

Useful Resources:
- Theory: https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
- Code Inspiration: https://igreat.github.io/blog/manifold_hypothesis/
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


# --- Generate Spiral Dataset ---
t = np.arange(1, 5, 0.05)

red_spiral = (np.array([np.cos(t), np.sin(t)]) * t).T * 0.5
blue_spiral = (np.array([np.cos(t + np.pi), np.sin(t + np.pi)]) * t).T * 0.5

red_labels = np.ones((red_spiral.shape[0], 1))
blue_labels = np.zeros((blue_spiral.shape[0], 1))

data_red = np.hstack((red_spiral, red_labels))
data_blue = np.hstack((blue_spiral, blue_labels))

data = np.vstack((data_red, data_blue))

X_data = torch.tensor(data[:, :2], dtype=torch.float32)
y_data = torch.tensor(data[:, 2:], dtype=torch.float32)


# --- Define the Neural Network ---
class SpiralsClassifier2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.fc4 = nn.Linear(2, 2)
        self.fc5 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return self.fc5(x)


model = SpiralsClassifier2D()

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()


# --- Decision Boundary During Training ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))


def plot_decision_boundary(model, ax):
    """Plot filled contour map over a 2D grid."""
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        outputs = model(grid_tensor)
        probs = torch.sigmoid(outputs).reshape(xx.shape)

    # Plot the decision boundary contour
    contour = ax.contourf(
        xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=["blue", "red", "white"]
    )


# --- Model training ---
epochs = 300

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_data)
    loss = criterion(outputs, y_data)
    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        ax.clear()
        plot_decision_boundary(model, ax)
        ax.scatter(red_spiral[:, 0], red_spiral[:, 1], color="r", label="Class 1")
        ax.scatter(blue_spiral[:, 0], blue_spiral[:, 1], color="b", label="Class 0")
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_title(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        ax.legend()
        plt.draw()
        plt.pause(0.05)

plt.ioff()
plt.show()


# --- Compute Intermediate Activations ---
def get_activations(model, x):
    """Compute intermediate activations for each layer of the model."""
    activations = []
    activations.append(x.numpy())  # Input

    a = x
    a = torch.tanh(model.fc1(a))
    activations.append(a.detach().numpy())

    a = torch.tanh(model.fc2(a))
    activations.append(a.detach().numpy())

    a = torch.tanh(model.fc3(a))
    activations.append(a.detach().numpy())

    a = torch.tanh(model.fc4(a))
    activations.append(a.detach().numpy())

    a = model.fc5(a)  # Final linear output (1D)
    a = a.detach().numpy()
    a_padded = np.concatenate((a, np.zeros_like(a)), axis=1)
    activations.append(a_padded)

    return activations


# Prepare tensors for red and blue spirals
red_tensor = torch.tensor(red_spiral, dtype=torch.float32)
blue_tensor = torch.tensor(blue_spiral, dtype=torch.float32)

# Create a grid of points (for visualizing the warping of the entire space)
x_min, x_max, y_min, y_max = -3, 3, -3, 3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

red_activations = get_activations(model, red_tensor)
blue_activations = get_activations(model, blue_tensor)
grid_activations = get_activations(model, grid_tensor)

layers = ["Input", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Output"]


# --- Animate the Transformation Between Layers ---
plt.ion()
fig2, ax2 = plt.subplots(figsize=(6, 6))
title_text = fig2.suptitle("", fontsize=14)
n_steps = 30  # Number of steps between layers (interpolation)

for i in range(len(layers) - 1):
    start_red, end_red = red_activations[i], red_activations[i + 1]  # Red spiral
    start_blue, end_blue = blue_activations[i], blue_activations[i + 1]  # Blue spiral
    start_grid, end_grid = grid_activations[i], grid_activations[i + 1]

    for alpha in np.linspace(0, 1, n_steps):
        inter_red = (1 - alpha) * start_red + alpha * end_red
        inter_blue = (1 - alpha) * start_blue + alpha * end_blue
        inter_grid = (1 - alpha) * start_grid + alpha * end_grid

        ax2.clear()
        ax2.scatter(inter_red[:, 0], inter_red[:, 1], color="r", s=30, alpha=0.8)
        ax2.scatter(inter_blue[:, 0], inter_blue[:, 1], color="b", s=30, alpha=0.8)
        ax2.scatter(inter_grid[:, 0], inter_grid[:, 1], s=5, alpha=0.2)

        # Set fixed limits if preferred
        # ax2.set_xlim(-3, 3)
        # ax2.set_ylim(-3, 3)

        title_text.set_text(f"Transition: {layers[i]} → {layers[i+1]} ({alpha:.2f})")

        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.set_aspect("equal")

        if i + 1 == len(layers) - 1 and alpha == 1:
            ax2.axvline(x=0, color="black", linestyle="-", label="Decision Boundary")
            ax2.set_xlim(-5, 5)
            ax2.set_ylim(-5, 5)
            ax2.legend()

        plt.draw()
        plt.pause(0.05)

plt.ioff()
plt.show()
