
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from mpl_toolkits.mplot3d import Axes3D


def analyze_mlp_edge_synaptic(model, n_neurons=300, voltage_range=(0, 10), resolution=50, n_sample_pairs=200,
                              device=None):
    """
    Analyze the learned MLP edge function with statistical sampling
    Creates 2D heatmaps, 3D surface plots, and scatter plot of edge function vs voltage difference
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

    # Store data for scatter plot (voltage differences and corresponding outputs)
    voltage_diffs = []
    edge_outputs = []

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

        # Collect data for scatter plot
        for batch_idx in range(batch_size_actual):
            # Calculate voltage differences for this batch (u_j - u_i)
            u_diff = (u_j_flat - u_i_flat).squeeze().cpu().numpy()  # (n_grid_points,)
            edge_vals = lin_edge[batch_idx].flatten().cpu().numpy()  # (n_grid_points,)

            voltage_diffs.extend(u_diff)
            edge_outputs.extend(edge_vals)

    mean_output = torch.mean(all_outputs, dim=0).cpu().numpy()
    std_output = torch.std(all_outputs, dim=0).cpu().numpy()

    # Convert scatter plot data to numpy arrays
    voltage_diffs = np.array(voltage_diffs)
    edge_outputs = np.array(edge_outputs)

    # Create 2D heatmap plots
    fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(np.flipud(mean_output), extent=[voltage_range[0], voltage_range[1],
                                          voltage_range[0], voltage_range[1]],
                     origin='lower', cmap='viridis', aspect='equal')
    ax1.set_title(f'mean edge function\n(over {n_sample_pairs} random pairs)')
    ax1.set_xlabel('u_i (voltage)')
    ax1.set_ylabel('u_j (voltage)')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('mean output')

    im2 = ax2.imshow(np.flipud(std_output), extent=[voltage_range[0], voltage_range[1],
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
    surf1 = ax3d_1.plot_surface(U_i, U_j, np.flipud(mean_output), cmap='viridis',
                                alpha=0.8, linewidth=0, antialiased=True)
    ax3d_1.set_xlabel('u_i (voltage)')
    ax3d_1.set_ylabel('u_j (voltage)')
    ax3d_1.set_zlabel('mean edge output')
    ax3d_1.set_title(f'3d mean edge function\n(over {n_sample_pairs} pairs)')
    ax3d_1.view_init(elev=30, azim=45)

    # Add contour lines at the bottom
    ax3d_1.contour(U_i, U_j, np.flipud(mean_output), zdir='z',
                   offset=mean_output.min(), cmap='viridis', alpha=0.5)

    # 3D plot for std dev
    ax3d_2 = fig_3d.add_subplot(122, projection='3d')
    surf2 = ax3d_2.plot_surface(U_i, U_j, np.flipud(std_output), cmap='plasma',
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

    # Create scatter plot of edge function vs voltage difference
    fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(10, 6))

    # Subsample for visualization if too many points
    max_points = 10000
    if len(voltage_diffs) > max_points:
        indices = np.random.choice(len(voltage_diffs), max_points, replace=False)
        voltage_diffs_plot = voltage_diffs[indices]
        edge_outputs_plot = edge_outputs[indices]
        alpha_val = 0.3
    else:
        voltage_diffs_plot = voltage_diffs
        edge_outputs_plot = edge_outputs
        alpha_val = 0.5

    # Create scatter plot
    scatter = ax_scatter.scatter(voltage_diffs_plot, edge_outputs_plot,
                                 alpha=alpha_val, s=1, c=edge_outputs_plot,
                                 cmap='viridis', rasterized=True)

    ax_scatter.set_xlabel('u_j - u_i (voltage difference)')
    ax_scatter.set_ylabel('edge function output')
    ax_scatter.set_title(f'edge function vs voltage difference\n(sampled from {n_sample_pairs} neuron pairs)')
    ax_scatter.grid(True, alpha=0.3)

    # Add colorbar
    cbar_scatter = plt.colorbar(scatter, ax=ax_scatter)
    cbar_scatter.set_label('edge output')

    plt.tight_layout()

    return fig_2d, fig_3d, fig_scatter

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

def analyze_mlp_phi_synaptic(model, n_neurons=300, voltage_range=(0, 10), resolution=50, n_sample_pairs=200,
                             device=None):
    """
    Analyze the learned MLP phi function with statistical sampling
    Creates 2D plots: mean with std band + all individual line plots

    For generic_excitation update type:
    - u: voltage (varied)
    - embedding: neuron embedding (sampled from different neurons)
    - msg: set to zeros (no message passing)
    - field: set to ones
    - excitation: set to zeros
    """

    embedding = model.a  # Shape: (300, 2)

    # Get excitation dimension from model
    excitation_dim = getattr(model, 'excitation_dim', 0)

    # Create voltage grid (1D since we're analyzing voltage vs embedding effects)
    u_vals = torch.linspace(voltage_range[0], voltage_range[1], resolution, device=device)

    print(f"sampling {n_sample_pairs} random neurons across {resolution} voltage points...")
    print(f"excitation_dim: {excitation_dim}")

    # Sample random neurons
    np.random.seed(42)  # For reproducibility
    neuron_indices = np.random.choice(n_neurons, size=n_sample_pairs, replace=True)

    # Store all outputs for statistics
    all_outputs = torch.zeros(n_sample_pairs, resolution, device=device)

    # Process in batches to manage memory
    batch_size = 50
    for batch_start in trange(0, n_sample_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_sample_pairs)
        batch_size_actual = batch_end - batch_start

        batch_inputs = []
        for batch_idx in range(batch_size_actual):
            neuron_idx = neuron_indices[batch_start + batch_idx]

            # Get embedding for this neuron
            neuron_embedding = embedding[neuron_idx].unsqueeze(0).repeat(resolution, 1)  # (resolution, 2)

            # Create voltage array
            u_batch = u_vals.unsqueeze(1)  # (resolution, 1)

            # Create fixed components
            msg = torch.zeros(resolution, 1, device=device)  # Message set to zeros
            field = torch.ones(resolution, 1, device=device)  # Field set to ones
            excitation = torch.zeros(resolution, excitation_dim, device=device)  # Excitation set to zeros

            # Concatenate input features: [u, embedding, msg, field, excitation]
            in_features = torch.cat([u_batch, neuron_embedding, msg, field, excitation], dim=1)
            batch_inputs.append(in_features)

        # Stack batch inputs
        batch_features = torch.stack(batch_inputs, dim=0)  # (batch_size, resolution, input_dim)
        batch_features = batch_features.reshape(-1, batch_features.shape[-1])  # (batch_size * resolution, input_dim)

        # Forward pass through MLP
        with torch.no_grad():
            phi_output = model.lin_phi(batch_features)

        # Reshape back to batch format
        phi_output = phi_output.reshape(batch_size_actual, resolution, -1).squeeze(-1)

        # Store results
        all_outputs[batch_start:batch_end] = phi_output

    # Compute statistics across all sampled neurons
    mean_output = torch.mean(all_outputs, dim=0).cpu().numpy()  # (resolution,)
    std_output = torch.std(all_outputs, dim=0).cpu().numpy()  # (resolution,)
    all_outputs_np = all_outputs.cpu().numpy()  # (n_sample_pairs, resolution)

    u_vals_np = u_vals.cpu().numpy()

    print(f"statistics computed over {n_sample_pairs} neurons")
    print(f"mean output range: [{mean_output.min():.4f}, {mean_output.max():.4f}]")
    print(f"std output range: [{std_output.min():.4f}, {std_output.max():.4f}]")

    # Create 2D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: Mean plot with std band
    ax1.plot(u_vals_np, mean_output, 'b-', linewidth=3, label='mean', zorder=10)
    ax1.fill_between(u_vals_np, mean_output - std_output, mean_output + std_output,
                     alpha=0.3, color='blue', label='±1 std')
    ax1.set_xlabel('voltage (u)')
    ax1.set_ylabel('phi output')
    ax1.set_title(f'mean phi function\n(over {n_sample_pairs} random neurons)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right panel: All individual line plots
    # Use alpha to make individual lines semi-transparent
    alpha_val = min(0.8, max(0.1, 50.0 / n_sample_pairs))  # Adaptive alpha based on number of lines

    for i in range(n_sample_pairs):
        ax2.plot(u_vals_np, all_outputs_np[i], '-', alpha=alpha_val, linewidth=0.5, color='gray')

    # Overlay the mean on top
    ax2.plot(u_vals_np, mean_output, 'r-', linewidth=2, label='mean', zorder=10)

    ax2.set_xlabel('voltage (u)')
    ax2.set_ylabel('phi output')
    ax2.set_title(f'all individual phi functions\n({n_sample_pairs} neurons)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    return fig


def analyze_mlp_phi_embedding(model, n_neurons=300, voltage_range=(0, 10), resolution=50, n_sample_pairs=200,
                                 device=None):
    """
    Analyze MLP phi function across voltage and embedding space
    Creates 2D heatmaps showing how phi varies with voltage and embedding dimensions
    """

    embedding = model.a  # Shape: (300, 2)
    excitation_dim = getattr(model, 'excitation_dim', 0)

    # Create voltage grid
    u_vals = torch.linspace(voltage_range[0], voltage_range[1], resolution, device=device)

    print(f"analyzing phi function across voltage and embedding space...")
    print(f"resolution: {resolution}x{resolution}, excitation_dim: {excitation_dim}")

    # Sample random neurons for embedding analysis
    np.random.seed(42)
    neuron_indices = np.random.choice(n_neurons, size=n_sample_pairs, replace=True)

    # Store outputs for each embedding dimension
    all_outputs_emb1 = torch.zeros(n_sample_pairs, resolution, device=device)
    all_outputs_emb2 = torch.zeros(n_sample_pairs, resolution, device=device)

    # Process in batches
    batch_size = 50
    for batch_start in trange(0, n_sample_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_sample_pairs)
        batch_size_actual = batch_end - batch_start

        batch_inputs = []
        for batch_idx in range(batch_size_actual):
            neuron_idx = neuron_indices[batch_start + batch_idx]

            # Get embedding for this neuron
            neuron_embedding = embedding[neuron_idx].unsqueeze(0).repeat(resolution, 1)

            # Create voltage array
            u_batch = u_vals.unsqueeze(1)

            # Fixed components
            msg = torch.zeros(resolution, 1, device=device)
            field = torch.ones(resolution, 1, device=device)
            excitation = torch.zeros(resolution, excitation_dim, device=device)

            # Input features
            in_features = torch.cat([u_batch, neuron_embedding, msg, field, excitation], dim=1)
            batch_inputs.append(in_features)

        # Process batch
        batch_features = torch.stack(batch_inputs, dim=0)
        batch_features = batch_features.reshape(-1, batch_features.shape[-1])

        with torch.no_grad():
            phi_output = model.lin_phi(batch_features)

        phi_output = phi_output.reshape(batch_size_actual, resolution, -1).squeeze(-1)

        # Store results
        all_outputs_emb1[batch_start:batch_end] = phi_output

    # Now create 2D grid: voltage vs embedding dimension
    # We'll vary embedding dimension 1 and keep dimension 2 at mean value
    emb_vals = torch.linspace(embedding[:, 0].min(), embedding[:, 0].max(), resolution, device=device)
    emb_mean_dim2 = embedding[:, 1].mean()

    # Create 2D output grid
    output_grid = torch.zeros(resolution, resolution, device=device)  # (emb_dim1, voltage)

    print("creating 2D grid: embedding dim 1 vs voltage...")
    for i, emb1_val in enumerate(trange(len(emb_vals))):
        emb1_val = emb_vals[i]

        # Create embedding with varying dim1 and fixed dim2
        neuron_embedding = torch.stack([
            emb1_val.repeat(resolution),
            emb_mean_dim2.repeat(resolution)
        ], dim=1)

        u_batch = u_vals.unsqueeze(1)
        msg = torch.zeros(resolution, 1, device=device)
        field = torch.ones(resolution, 1, device=device)
        excitation = torch.zeros(resolution, excitation_dim, device=device)

        in_features = torch.cat([u_batch, neuron_embedding, msg, field, excitation], dim=1)

        with torch.no_grad():
            phi_output = model.lin_phi(in_features)

        output_grid[i, :] = phi_output.squeeze()

    output_grid_np = output_grid.cpu().numpy()
    u_vals_np = u_vals.cpu().numpy()
    emb_vals_np = emb_vals.cpu().numpy()

    # Create 2D heatmap
    fig_2d, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(output_grid_np, extent=[voltage_range[0], voltage_range[1],
                                           emb_vals_np.min(), emb_vals_np.max()],
                   origin='lower', cmap='viridis', aspect='auto')
    ax.set_xlabel('voltage (u)')
    ax.set_ylabel('embedding dimension 1')
    ax.set_title(f'phi function: voltage vs embedding\n(dim 2 fixed at {emb_mean_dim2:.3f})')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('phi output')

    plt.tight_layout()

    return fig_2d, output_grid_np

# Example usage:
# fig_2d_voltage, mean_out, std_out = analyze_mlp_phi_function(model, n_sample_pairs=1000, resolution=100, device=device)
# fig_2d_heatmap, grid_out = analyze_mlp_phi_embedding(model, n_sample_pairs=1000, resolution=50, device=device)
#
# fig_2d_voltage.savefig(f"./{log_dir}/results/phi_function_voltage.png", dpi=300, bbox_inches='tight')
# fig_2d_heatmap.savefig(f"./{log_dir}/results/phi_function_2d.png", dpi=300, bbox_inches='tight')
# plt.close(fig_2d_voltage)
# plt.close(fig_2d_heatmap)