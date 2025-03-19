"""
MNIST Neural Network Visualization - Manifold Transformations Through Network Layers

Overview:
This project visualizes how a neural network transforms MNIST digit data through its layers,
demonstrating the *manifold hypothesis* in action. It shows how the network progressively
transforms the 784-dimensional input space (28x28 images) into more separable representations
through hidden layers until reaching the final 10-dimensional output space.

Main Components:
1. MNIST Dataset Processing: Uses the standard MNIST digits dataset.
2. Neural Network Architecture: A simple feedforward network with tanh activations.
3. Layer-wise Visualization: Interpolates between successive layer activations to show
   progressive transformation of the data manifold.
4. PCA Dimensionality Reduction: Projects high-dimensional activations to 3D for visualization.

What It Demonstrates:
- Manifold Hypothesis: Shows how neural networks reshape complex data distributions.
- Progressive Transformations: Illustrates how each layer contributes to separating classes.
- Feature Learning: Demonstrates how the network learns to disentangle the digit classes.

Useful Resources:
- Theory: https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
- Code Inspiration: https://igreat.github.io/blog/manifold_hypothesis/

## TODO 
- try t-SNE
- embed actual digits instead of points
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class MnistClassifier(nn.Module):
    """Simple NN for MNIST digit classification."""
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28x28 images
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        out = self.fc3(x)
        return out

def get_activations(model, x):
    """Extract activations from each layer of the network."""
    activations = []
    x_flat = x.view(x.size(0), -1)
    activations.append(x_flat.detach().cpu().numpy())  # Input flattened (784 dims)
    a1 = torch.tanh(model.fc1(x_flat))
    activations.append(a1.detach().cpu().numpy())      # After fc1 (128 dims)
    a2 = torch.tanh(model.fc2(a1))
    activations.append(a2.detach().cpu().numpy())      # After fc2 (64 dims)
    a3 = model.fc3(a2)
    activations.append(a3.detach().cpu().numpy())      # Output logits (10 dims)
    return activations

def pad_activations(activations):
    """Pad all activation arrays to the same number of columns."""
    max_dims = max(act.shape[1] for act in activations)
    padded_activations = []
    
    for act in activations:
        if act.shape[1] < max_dims:
            padded = np.pad(act, ((0, 0), (0, max_dims - act.shape[1])), mode='constant')
            padded_activations.append(padded)
        else:
            padded_activations.append(act)
            
    return padded_activations

def visualize_continuous_manifold_transition(activations, labels, layer_names, n_steps=30):
    """Visualize continuous transition through all layers using a single PCA projection."""
    # Pad activations to same dimensionality
    padded_activations = pad_activations(activations)
    
    # Combine all activations for a single PCA
    combined = np.vstack(padded_activations)
    
    # Apply PCA to get 3D projection
    pca = PCA(n_components=3)
    combined_3d = pca.fit_transform(combined)
    
    # Split back into separate layers
    sample_count = padded_activations[0].shape[0]
    layer_points_3d = []
    for i in range(len(padded_activations)):
        start_idx = i * sample_count
        end_idx = (i + 1) * sample_count
        layer_points_3d.append(combined_3d[start_idx:end_idx])
    
    # Setup visualization
    plt.ion()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)
    
    # Create legend elements
    legend_elements = []
    for i in range(10):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=plt.cm.tab10(i), markersize=10, 
                              label=f' {i}'))
    
    # Create a separate legend axis outside the main plot area
    legend_ax = fig.add_axes([0.85, 0.8, 0.15, 0.2])  # [left, bottom, width, height]
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_elements, title="MNIST Digits")
    
    colors = np.array([plt.cm.tab10(int(l)) for l in labels])
    
    # Animate transitions between consecutive layers
    for i in range(len(layer_points_3d) - 1):
        act1_3d = layer_points_3d[i]
        act2_3d = layer_points_3d[i+1]
        
        for alpha in np.linspace(0, 1, n_steps):
            inter = (1 - alpha) * act1_3d + alpha * act2_3d
            ax.clear()
            ax.scatter(inter[:, 0], inter[:, 1], inter[:, 2], c=colors, s=20, depthshade=True, alpha=0.6)
            ax.set_title(f"Transition: {layer_names[i]} → {layer_names[i+1]}\nInterpolation: {alpha:.2f}")
            
            plt.draw()
            plt.pause(0.01)
        plt.waitforbuttonpress()
    plt.ioff()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model
    model = MnistClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training the model...")
    model.train()
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    print("Preparing data for visualization...")
    indices = np.random.choice(len(test_dataset), size=500, replace=False)
    subset = Subset(test_dataset, indices)
    subset_loader = DataLoader(subset, batch_size=500, shuffle=False)
    data_vis, labels = next(iter(subset_loader))
    data_vis, labels = data_vis.to(device), labels.to(device)
    
    print("Extracting activations and visualizing transitions...")
    model.eval()
    activations = get_activations(model, data_vis)
    layer_names = ["Input (flattened)", "Layer 1", "Layer 2", "Output (logits)"]
    
    # Visualize continuous manifold transformation across all layers
    print("Visualizing continuous manifold transformation...")
    visualize_continuous_manifold_transition(activations, labels.cpu().numpy(), layer_names, n_steps=30)

if __name__ == "__main__":
    main()