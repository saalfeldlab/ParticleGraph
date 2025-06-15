import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from mpl_toolkits.mplot3d import Axes3D


def analyze_mlp_edge_synaptic(model, n_neurons=300, voltage_range=(0, 10), resolution=50, n_sample_pairs=200, device=None):
    """
    Analyze the learned MLP edge function with statistical sampling
    Creates 2D heatmaps and 3D surface plots: mean and std dev across many neuron pairs
    """

    embedding = model.a  # Shape: (300, 2)

    # Create voltage grid
    u_vals = torch.linspace(voltage_range[0], voltage_range[1], resolution, device=device)
    u_i_grid, u_j_grid = torch.meshgrid(u_vals, u_vals, indexing='ij')

    # Flatten for batch processing
    u_i_flat = u_i_grid.flatten().unsqueeze(1)  # (resolution^2, 1)
    u_j_flat = u_j_grid.flatten().unsqueeze(1)  # (resolution^2, 1)
    n_grid_points = len(u_i_flat)

    print(f"sampling {n_sample_pairs} random neuron pairs across {resolution}x{resolution} voltage grid...")

    # Sample random neuron pairs
    np.random.seed(42)  # For reproducibility
    neuron_indices = np.random.choice(n_neurons, size=(n_sample_pairs, 2), replace=True)

    # Store all outputs for statistics
    all_outputs = torch.zeros(n_sample_pairs, resolution, resolution, device=device)

    # Process in batches to manage memory
    batch_size = 10
    for batch_start in trange(0, n_sample_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_sample_pairs)
        batch_size_actual = batch_end - batch_start
        batch_inputs = []
        for batch_idx in range(batch_size_actual):
            pair_idx = batch_start + batch_idx
            i, j = neuron_indices[pair_idx]

            embedding_i = embedding[i].unsqueeze(0).repeat(n_grid_points, 1)  # (n_grid_points, 2)
            embedding_j = embedding[j].unsqueeze(0).repeat(n_grid_points, 1)  # (n_grid_points, 2)

            in_features = torch.cat([u_i_flat, u_j_flat, embedding_i, embedding_j], dim=1)
            batch_inputs.append(in_features)

        batch_features = torch.stack(batch_inputs, dim=0)  # (batch_size, n_grid_points, 6)
        batch_features = batch_features.reshape(-1, 6)  # (batch_size * n_grid_points, 6)

        with torch.no_grad():
            lin_edge = model.lin_edge(batch_features)
            if model.lin_edge_positive:
                lin_edge = lin_edge ** 2

        lin_edge = lin_edge.reshape(batch_size_actual, n_grid_points, -1).squeeze(-1)
        lin_edge = lin_edge.reshape(batch_size_actual, resolution, resolution)
        all_outputs[batch_start:batch_end] = lin_edge

    mean_output = torch.mean(all_outputs, dim=0).cpu().numpy()
    std_output = torch.std(all_outputs, dim=0).cpu().numpy()

    # Create 2D heatmap plots
    fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(mean_output, extent=[voltage_range[0], voltage_range[1],
                                          voltage_range[0], voltage_range[1]],
                     origin='lower', cmap='viridis', aspect='equal')
    ax1.set_title(f'mean edge function\n(over {n_sample_pairs} random pairs)')
    ax1.set_xlabel('u_i (voltage)')
    ax1.set_ylabel('u_j (voltage)')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('mean output')

    im2 = ax2.imshow(std_output, extent=[voltage_range[0], voltage_range[1],
                                         voltage_range[0], voltage_range[1]],
                     origin='lower', cmap='plasma', aspect='equal')
    ax2.set_title(f'std dev of edge function\n(over {n_sample_pairs} random pairs)')
    ax2.set_xlabel('u_i (voltage)')
    ax2.set_ylabel('u_j (voltage)')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('std dev')

    plt.tight_layout()

    # Create 3D surface plots
    fig_3d = plt.figure(figsize=(18, 8))

    # Prepare meshgrid for 3D plotting
    u_vals_np = np.linspace(voltage_range[0], voltage_range[1], resolution)
    U_i, U_j = np.meshgrid(u_vals_np, u_vals_np, indexing='ij')

    # 3D plot for mean
    ax3d_1 = fig_3d.add_subplot(121, projection='3d')
    surf1 = ax3d_1.plot_surface(U_i, U_j, mean_output, cmap='viridis',
                                alpha=0.8, linewidth=0, antialiased=True)
    ax3d_1.set_xlabel('u_i (voltage)')
    ax3d_1.set_ylabel('u_j (voltage)')
    ax3d_1.set_zlabel('mean edge output')
    ax3d_1.set_title(f'3d mean edge function\n(over {n_sample_pairs} pairs)')
    ax3d_1.view_init(elev=30, azim=45)

    # Add contour lines at the bottom
    ax3d_1.contour(U_i, U_j, mean_output, zdir='z',
                   offset=mean_output.min(), cmap='viridis', alpha=0.5)

    # 3D plot for std dev
    ax3d_2 = fig_3d.add_subplot(122, projection='3d')
    surf2 = ax3d_2.plot_surface(U_i, U_j, std_output, cmap='plasma',
                                alpha=0.8, linewidth=0, antialiased=True)
    ax3d_2.set_xlabel('u_i (voltage)')
    ax3d_2.set_ylabel('u_j (voltage)')
    ax3d_2.set_zlabel('std dev edge output')
    ax3d_2.set_title(f'3d std dev edge function\n(over {n_sample_pairs} pairs)')
    ax3d_2.view_init(elev=30, azim=45)

    # Add contour lines at the bottom
    ax3d_2.contour(U_i, U_j, std_output, zdir='z',
                   offset=std_output.min(), cmap='plasma', alpha=0.5)

    # Add colorbars for 3D plots
    fig_3d.colorbar(surf1, ax=ax3d_1, shrink=0.6, aspect=20, label='mean output')
    fig_3d.colorbar(surf2, ax=ax3d_2, shrink=0.6, aspect=20, label='std dev')

    plt.tight_layout()

    # # Print some statistics
    # print(f"\nstatistics over {n_sample_pairs} neuron pairs:")
    # print(f"mean output range: [{mean_output.min():.4f}, {mean_output.max():.4f}]")
    # print(f"std output range: [{std_output.min():.4f}, {std_output.max():.4f}]")
    # print(f"average std across voltage space: {std_output.mean():.4f}")
    #
    # # Find peaks and valleys
    # mean_max_idx = np.unravel_index(np.argmax(mean_output), mean_output.shape)
    # mean_min_idx = np.unravel_index(np.argmin(mean_output), mean_output.shape)
    # std_max_idx = np.unravel_index(np.argmax(std_output), std_output.shape)
    #
    # print(f"mean peak at u_i={u_vals_np[mean_max_idx[0]]:.2f}, u_j={u_vals_np[mean_max_idx[1]]:.2f}")
    # print(f"mean valley at u_i={u_vals_np[mean_min_idx[0]]:.2f}, u_j={u_vals_np[mean_min_idx[1]]:.2f}")
    # print(f"highest variability at u_i={u_vals_np[std_max_idx[0]]:.2f}, u_j={u_vals_np[std_max_idx[1]]:.2f}")

    return fig_2d, fig_3d



def analyze_embedding_space(model, n_neurons=300):
    """Analyze the learned embedding space"""

    embedding = model.a.detach().cpu().numpy()  # (300, 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Embedding scatter plot
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1],
                              c=np.arange(n_neurons), cmap='tab10', alpha=0.7)
    axes[0].set_xlabel('Embedding Dimension 1')
    axes[0].set_ylabel('Embedding Dimension 2')
    axes[0].set_title('Learned Neuron Embeddings')
    axes[0].grid(True, alpha=0.3)

    # 2. Embedding distribution
    axes[1].hist(embedding[:, 0], bins=30, alpha=0.7, label='Dim 1')
    axes[1].hist(embedding[:, 1], bins=30, alpha=0.7, label='Dim 2')
    axes[1].set_xlabel('Embedding Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Embedding Value Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Distance matrix between embeddings
    distances = np.linalg.norm(embedding[:, None] - embedding[None, :], axis=2)
    im = axes[2].imshow(distances, cmap='viridis')
    axes[2].set_title('Pairwise Embedding Distances')
    axes[2].set_xlabel('Neuron Index')
    axes[2].set_ylabel('Neuron Index')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return embedding, distances