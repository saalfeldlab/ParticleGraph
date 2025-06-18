from tqdm import trange
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
from collections import defaultdict
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

def get_neuron_index(neuron_name, activity_neuron_list):
    """
    Returns the index of the neuron_name in activity_neuron_list.
    Raises ValueError if not found.
    """
    try:
        return activity_neuron_list.index(neuron_name)
    except ValueError:
        raise ValueError(f"Neuron '{neuron_name}' not found in activity_neuron_list.")


def get_neuron_index(neuron_name, activity_neuron_list):
    """
    Returns the index of the neuron_name in activity_neuron_list.
    Raises ValueError if not found.
    """
    try:
        return activity_neuron_list.index(neuron_name)
    except ValueError:
        raise ValueError(f"Neuron '{neuron_name}' not found in activity_neuron_list.")


def analyze_mlp_edge_lines(model, neuron_list, all_neuron_list, adjacency_matrix, signal_range=(0, 10), resolution=100,
                           device=None):
    """
    Create line plots showing edge function vs signal difference for neuron pairs
    Uses adjacency matrix to find all connected neurons for each neuron of interest
    Plots mean and standard deviation across all connections

    Args:
        model: The trained model with embeddings and lin_edge
        neuron_list: List of neuron names of interest (1-5 neurons)
        all_neuron_list: Complete list of all 300 neuron names
        adjacency_matrix: 2D array (300x300) where adjacency_matrix[i,j] = 1 if i->j connection exists
        signal_range: Tuple of (min_signal, max_signal)
        resolution: Number of points for signal difference sampling
        device: PyTorch device

    Returns:
        fig_lines: Figure with line plots showing mean ± std for each neuron of interest
    """

    embedding = model.a  # Shape: (300, 2)

    print(f"generating line plots for {len(neuron_list)} neurons using adjacency matrix connections...")

    # Get indices of the neurons of interest
    neuron_indices_of_interest = []
    for neuron_name in neuron_list:
        try:
            neuron_idx = get_neuron_index(neuron_name, all_neuron_list)
            neuron_indices_of_interest.append(neuron_idx)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

    if len(neuron_indices_of_interest) == 0:
        raise ValueError("No valid neurons found in neuron_list")

    # Create signal difference array for line plots
    u_diff_line = torch.linspace(-signal_range[1], signal_range[1], resolution * 2 - 1, device=device)

    # For each neuron of interest, find all its connections and compute statistics
    neuron_stats = {}

    for neuron_idx, neuron_id in enumerate(neuron_indices_of_interest):
        neuron_name = neuron_list[neuron_idx]
        receiver_embedding = embedding[neuron_id]  # This neuron as receiver (embedding_i)

        # Find all connected senders (where adjacency_matrix[receiver, sender] = 1)
        connected_senders = np.where(adjacency_matrix[neuron_id, :] == 1)[0]

        if len(connected_senders) == 0:
            print(f"Warning: No incoming connections found for {neuron_name}")
            continue

        # print(f"Found {len(connected_senders)} incoming connections for {neuron_name}")
        # Store outputs for all connections to this receiver
        connection_outputs = torch.zeros(len(connected_senders), len(u_diff_line), device=device)

        for conn_idx, sender_id in enumerate(connected_senders):
            sender_embedding = embedding[sender_id]  # Connected neuron as sender (embedding_j)

            line_inputs = []
            for diff_idx, diff in enumerate(u_diff_line):
                # Create signal pairs that span the valid range
                u_center = (signal_range[0] + signal_range[1]) / 2
                u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
                u_j = torch.clamp(u_center + diff / 2, signal_range[0], signal_range[1])

                # Ensure the actual difference matches what we want
                actual_diff = u_j - u_i
                if abs(actual_diff - diff) > 1e-6:
                    # Adjust to get the exact difference we want
                    u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
                    u_j = u_i + diff
                    if u_j > signal_range[1]:
                        u_j = torch.tensor(signal_range[1], device=device)
                        u_i = u_j - diff
                    elif u_j < signal_range[0]:
                        u_j = torch.tensor(signal_range[0], device=device)
                        u_i = u_j - diff

                # Create input feature vector: [u_i, u_j, embedding_i, embedding_j]
                in_features = torch.cat([
                    u_i.unsqueeze(0),  # u_i as (1,)
                    u_j.unsqueeze(0),  # u_j as (1,)
                    receiver_embedding,  # embedding_i (receiver) as (2,)
                    sender_embedding  # embedding_j (sender) as (2,)
                ], dim=0)  # Final shape: (6,)
                line_inputs.append(in_features)

            line_features = torch.stack(line_inputs, dim=0)  # (len(u_diff_line), 6)

            with torch.no_grad():
                lin_edge = model.lin_edge(line_features)
                if model.lin_edge_positive:
                    lin_edge = lin_edge ** 2

            connection_outputs[conn_idx] = lin_edge.squeeze(-1)

        # Compute mean and std across all connections to this receiver
        mean_output = torch.mean(connection_outputs, dim=0).cpu().numpy()
        std_output = torch.std(connection_outputs, dim=0).cpu().numpy()

        neuron_stats[neuron_name] = {
            'mean': mean_output,
            'std': std_output,
            'n_connections': len(connected_senders)
        }

    # Create line plot figure
    fig_lines, ax_lines = plt.subplots(1, 1, figsize=(14, 8))

    # Generate colors for each neuron of interest
    colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_stats)))
    u_diff_line_np = u_diff_line.cpu().numpy()

    for neuron_idx, (neuron_name, stats) in enumerate(neuron_stats.items()):
        color = colors[neuron_idx]
        mean_vals = stats['mean']
        std_vals = stats['std']
        n_conn = stats['n_connections']

        # Plot mean line
        ax_lines.plot(u_diff_line_np, mean_vals,
                      color=color, linewidth=2,
                      label=f'{neuron_name} (n={n_conn})')

        # Plot standard deviation as shaded area
        ax_lines.fill_between(u_diff_line_np,
                              mean_vals - std_vals,
                              mean_vals + std_vals,
                              color=color, alpha=0.2)

    ax_lines.set_xlabel('u_j - u_i (signal difference)')
    ax_lines.set_ylabel('edge function output')
    ax_lines.set_title(f'edge function vs signal difference\n(mean ± std across incoming connections)')
    # grid(True, alpha=0.3)

    # Adaptive legend placement based on number of neurons
    n_neurons = len(neuron_stats)
    if n_neurons <= 20:
        # For few neurons, use right side
        ax_lines.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig_lines


def analyze_mlp_edge_lines_weighted_with_max(model, neuron_name, all_neuron_list, adjacency_matrix, weight_matrix,
                                             signal_range=(0, 10), resolution=100, device=None):
    """
    Create line plots showing weighted edge function vs signal difference for a single neuron of interest
    Uses adjacency matrix to find connections and weight matrix to scale the outputs
    Plots individual lines for each incoming connection
    Returns the connection with maximum response in signal difference range [8, 10]

    Args:
        model: The trained model with embeddings and lin_edge
        neuron_name: Single neuron name of interest
        all_neuron_list: Complete list of all 300 neuron names
        adjacency_matrix: 2D array (300x300) where adjacency_matrix[i,j] = 1 if i->j connection exists
        weight_matrix: 2D array (300x300) with connection weights to scale edge function output
        signal_range: Tuple of (min_signal, max_signal) for DF/F0 measurements
        resolution: Number of points for signal difference sampling
        device: PyTorch device

    Returns:
        fig_lines: Figure with individual weighted line plots for each connection
        max_response_data: Dict with info about the connection with maximum response in [8,10] range
    """

    embedding = model.a  # Shape: (300, 2)

    # print(f"generating weighted line plots for {neuron_name} using adjacency and weight matrices...")

    # Get index of the neuron of interest
    try:
        neuron_id = get_neuron_index(neuron_name, all_neuron_list)
    except ValueError as e:
        raise ValueError(f"Neuron '{neuron_name}' not found: {e}")

    receiver_embedding = embedding[neuron_id]  # This neuron as receiver (embedding_i)

    # Find all connected senders (where adjacency_matrix[receiver, sender] = 1)
    connected_senders = np.where(adjacency_matrix[neuron_id, :] == 1)[0]
    #
    # if len(connected_senders) == 0:
    #     print(f"No incoming connections found for {neuron_name}")
    #     return None, None

    # print(f"Found {len(connected_senders)} incoming connections for {neuron_name}")

    # Create signal difference array for line plots
    u_diff_line = torch.linspace(-signal_range[1], signal_range[1], resolution * 2 - 1, device=device)
    u_diff_line_np = u_diff_line.cpu().numpy()

    # Find indices corresponding to signal difference range [8, 10]
    target_range_mask = (u_diff_line_np >= 8.0) & (u_diff_line_np <= 10.0)
    target_indices = np.where(target_range_mask)[0]

    # Store outputs and metadata for all connections
    connection_data = []
    max_response = -float('inf')
    max_response_data = None

    for sender_id in connected_senders:
        sender_name = all_neuron_list[sender_id]
        sender_embedding = embedding[sender_id]  # Connected neuron as sender (embedding_j)
        connection_weight = weight_matrix[neuron_id, sender_id]  # Weight for this connection

        line_inputs = []
        for diff_idx, diff in enumerate(u_diff_line):
            # Create signal pairs that span the valid range
            u_center = (signal_range[0] + signal_range[1]) / 2
            u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
            u_j = torch.clamp(u_center + diff / 2, signal_range[0], signal_range[1])

            # Ensure the actual difference matches what we want
            actual_diff = u_j - u_i
            if abs(actual_diff - diff) > 1e-6:
                # Adjust to get the exact difference we want
                u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
                u_j = u_i + diff
                if u_j > signal_range[1]:
                    u_j = torch.tensor(signal_range[1], device=device)
                    u_i = u_j - diff
                elif u_j < signal_range[0]:
                    u_j = torch.tensor(signal_range[0], device=device)
                    u_i = u_j - diff

            # Create input feature vector: [u_i, u_j, embedding_i, embedding_j]
            in_features = torch.cat([
                u_i.unsqueeze(0),  # u_i as (1,)
                u_j.unsqueeze(0),  # u_j as (1,)
                receiver_embedding,  # embedding_i (receiver) as (2,)
                sender_embedding  # embedding_j (sender) as (2,)
            ], dim=0)  # Final shape: (6,)
            line_inputs.append(in_features)

        line_features = torch.stack(line_inputs, dim=0)  # (len(u_diff_line), 6)

        with torch.no_grad():
            lin_edge = model.lin_edge(line_features)
            if model.lin_edge_positive:
                lin_edge = lin_edge ** 2

        # Apply weight scaling
        edge_output = lin_edge.squeeze(-1).cpu().numpy()
        weighted_output = edge_output * connection_weight

        # Find maximum response in target range [8, 10]
        if len(target_indices) > 0:
            max_in_range = np.max(weighted_output[target_indices])
            if max_in_range > max_response:
                max_response = max_in_range
                max_response_data = {
                    'receiver_name': neuron_name,
                    'sender_name': sender_name,
                    'receiver_id': neuron_id,
                    'sender_id': sender_id,
                    'weight': connection_weight,
                    'max_response': max_response,
                    'signal_diff_range': [8.0, 10.0]
                }

        connection_data.append({
            'sender_name': sender_name,
            'sender_id': sender_id,
            'weight': connection_weight,
            'output': weighted_output,
            'unweighted_output': edge_output
        })

    # Sort connections by weight magnitude for better visualization
    connection_data.sort(key=lambda x: abs(x['weight']), reverse=True)

    # Create line plot figure
    fig_lines, ax_lines = plt.subplots(1, 1, figsize=(14, 10))

    # Generate colors using a colormap that handles many lines well
    if len(connection_data) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(connection_data)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(connection_data)))

    # Plot each connection
    for conn_idx, conn_data in enumerate(connection_data):
        color = colors[conn_idx]
        sender_name = conn_data['sender_name']
        weight = conn_data['weight']
        weighted_output = conn_data['output']

        # Line style based on weight sign
        line_style = '-' if weight >= 0 else '--'

        # Calculate line width with safe division
        max_weight = np.max(np.abs([c['weight'] for c in connection_data]))
        if max_weight > 0:
            line_width = 1.5 + min(2.0, abs(weight) / max_weight)
        else:
            line_width = 1.5  # Default width if all weights are zero

        ax_lines.plot(u_diff_line_np, weighted_output,
                      color=color, linewidth=line_width, linestyle=line_style,
                      label=f'{sender_name} (w={weight:.3f})')

    # Highlight the target range [8, 10]
    ax_lines.axvspan(8.0, 10.0, alpha=0.2, color='red', label='Target range [8,10]')

    ax_lines.set_xlabel('u_j - u_i (signal difference)')
    ax_lines.set_ylabel('weighted edge function output')
    ax_lines.set_title(
        f'weighted edge function vs signal difference\n(receiver: {neuron_name}, all incoming connections)')
    ax_lines.grid(True, alpha=0.3)

    # Adaptive legend placement based on number of connections
    n_connections = len(connection_data)
    if n_connections <= 5:
        # For few connections, use right side
        ax_lines.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    elif n_connections <= 15:
        # For medium number, use multiple columns on right
        ax_lines.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                        fontsize='x-small', ncol=1)
    else:
        # For many connections, use multiple columns below plot
        ncol = min(4, n_connections // 5 + 1)
        ax_lines.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center',
                        ncol=ncol, fontsize='x-small', framealpha=0.9)
        # Add more space at bottom for legend
        plt.subplots_adjust(bottom=0.25)

    plt.tight_layout()

    return fig_lines, max_response_data


def find_top_responding_pairs(model, all_neuron_list, adjacency_matrix, weight_matrix,
                              signal_range=(0, 10), resolution=100, device=None, top_k=10):
    """
    Find the top K receiver-sender pairs with largest response in signal difference range [8, 10]
    by analyzing all neurons as receivers

    Args:
        model: The trained model with embeddings and lin_edge
        all_neuron_list: Complete list of all 300 neuron names
        adjacency_matrix: 2D array (300x300) where adjacency_matrix[i,j] = 1 if i->j connection exists
        weight_matrix: 2D array (300x300) with connection weights
        signal_range: Tuple of (min_signal, max_signal) for DF/F0 measurements
        resolution: Number of points for signal difference sampling
        device: PyTorch device
        top_k: Number of top pairs to return

    Returns:
        top_pairs: List of top K pairs sorted by response magnitude
        top_figures: List of figures for the top pairs
    """

    # print(f"Analyzing all {len(all_neuron_list)} neurons to find top {top_k} responding pairs...")

    all_responses = []

    # Analyze each neuron as receiver
    for neuron_idx, neuron_name in enumerate(all_neuron_list):
        try:
            fig , max_response_data = analyze_mlp_edge_lines_weighted_with_max(
                model, neuron_name, all_neuron_list, adjacency_matrix, weight_matrix,
                signal_range, resolution, device
            )

            plt.close(fig)

            if max_response_data is not None:
                all_responses.append(max_response_data)

        except Exception as e:
            print(f"Error processing {neuron_name}: {e}")
            continue

    # Sort by response magnitude and get top K
    all_responses.sort(key=lambda x: x['max_response'], reverse=True)
    top_pairs = all_responses[:top_k]
    for i, pair in enumerate(top_pairs):
        print(f"{i + 1:2d}. {pair['receiver_name']} ← {pair['sender_name']}:  ({pair['max_response']:.4f})")

    return top_pairs    # , top_figures

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

def analyze_mlp_phi_synaptic(model, n_neurons=300, signal_range=(0, 10), resolution=50, n_sample_pairs=200,
                             device=None):
    """
    Analyze the learned MLP phi function with statistical sampling
    Creates 2D plots: mean with std band + all individual line plots

    For generic_excitation update type:
    - u: signal (varied)
    - embedding: neuron embedding (sampled from different neurons)
    - msg: set to zeros (no message passing)
    - field: set to ones
    - excitation: set to zeros
    """

    embedding = model.a  # Shape: (300, 2)

    # Get excitation dimension from model
    excitation_dim = getattr(model, 'excitation_dim', 0)

    # Create signal grid (1D since we're analyzing signal vs embedding effects)
    u_vals = torch.linspace(signal_range[0], signal_range[1], resolution, device=device)

    print(f"sampling {n_sample_pairs} random neurons across {resolution} signal points...")
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

            # Create signal array
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
    ax1.set_xlabel('signal (u)')
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

    ax2.set_xlabel('signal (u)')
    ax2.set_ylabel('phi output')
    ax2.set_title(f'all individual phi functions\n({n_sample_pairs} neurons)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    return fig


def analyze_mlp_phi_embedding(model, n_neurons=300, signal_range=(0, 10), resolution=50, n_sample_pairs=200,
                                 device=None):
    """
    Analyze MLP phi function across signal and embedding space
    Creates 2D heatmaps showing how phi varies with signal and embedding dimensions
    """

    embedding = model.a  # Shape: (300, 2)
    excitation_dim = getattr(model, 'excitation_dim', 0)

    # Create signal grid
    u_vals = torch.linspace(signal_range[0], signal_range[1], resolution, device=device)

    print(f"analyzing phi function across signal and embedding space...")
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

            # Create signal array
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

    # Now create 2D grid: signal vs embedding dimension
    # We'll vary embedding dimension 1 and keep dimension 2 at mean value
    emb_vals = torch.linspace(embedding[:, 0].min(), embedding[:, 0].max(), resolution, device=device)
    emb_mean_dim2 = embedding[:, 1].mean()

    # Create 2D output grid
    output_grid = torch.zeros(resolution, resolution, device=device)  # (emb_dim1, signal)

    print("creating 2D grid: embedding dim 1 vs signal...")
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

    im = ax.imshow(output_grid_np, extent=[signal_range[0], signal_range[1],
                                           emb_vals_np.min(), emb_vals_np.max()],
                   origin='lower', cmap='viridis', aspect='auto')
    ax.set_xlabel('signal (u)')
    ax.set_ylabel('embedding dimension 1')
    ax.set_title(f'phi function: signal vs embedding\n(dim 2 fixed at {emb_mean_dim2:.3f})')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('phi output')

    plt.tight_layout()

    return fig_2d, output_grid_np

# Example usage:
# fig_2d_signal, mean_out, std_out = analyze_mlp_phi_function(model, n_sample_pairs=1000, resolution=100, device=device)
# fig_2d_heatmap, grid_out = analyze_mlp_phi_embedding(model, n_sample_pairs=1000, resolution=50, device=device)
#
# fig_2d_signal.savefig(f"./{log_dir}/results/phi_function_signal.png", dpi=300, bbox_inches='tight')
# fig_2d_heatmap.savefig(f"./{log_dir}/results/phi_function_2d.png", dpi=300, bbox_inches='tight')
# plt.close(fig_2d_signal)
# plt.close(fig_2d_heatmap)




def compute_separation_index(connectivity_neurons, odor_responsive_neurons):
    """
    Compute functional separation between high connectivity and high odor-responsive neurons

    Args:
        connectivity_neurons: List of neuron names with high connectivity
        odor_responsive_neurons: List of neuron names with high odor responses

    Returns:
        separation_metrics: Dict with separation statistics
    """
    connectivity_set = set(connectivity_neurons)
    odor_set = set(odor_responsive_neurons)

    # Find overlap
    overlap = connectivity_set.intersection(odor_set)

    # Compute separation metrics
    total_unique = len(connectivity_set.union(odor_set))
    overlap_count = len(overlap)
    min_set_size = min(len(connectivity_set), len(odor_set))

    # Separation index: 1 - (overlap / min_set_size)
    separation_index = 1.0 - (overlap_count / min_set_size) if min_set_size > 0 else 1.0

    # Additional metrics
    connectivity_purity = 1.0 - (overlap_count / len(connectivity_set)) if len(connectivity_set) > 0 else 1.0
    odor_purity = 1.0 - (overlap_count / len(odor_set)) if len(odor_set) > 0 else 1.0

    return {
        'separation_index': separation_index,
        'overlap_count': overlap_count,
        'connectivity_purity': connectivity_purity,
        'odor_purity': odor_purity,
        'total_connectivity_neurons': len(connectivity_set),
        'total_odor_neurons': len(odor_set),
        'overlapping_neurons': list(overlap)
    }


def classify_neural_architecture(separation_index, specialist_threshold=0.95, adapter_threshold=0.70):
    """
    Classify neural architecture based on separation index

    Args:
        separation_index: Float between 0 and 1
        specialist_threshold: Threshold for specialist classification
        adapter_threshold: Threshold for adapter classification

    Returns:
        architecture_type: String classification
    """
    if separation_index >= specialist_threshold:
        return 'specialist'
    elif separation_index >= adapter_threshold:
        return 'adapter'
    else:
        return 'generalist'


def analyze_individual_architectures(top_pairs_by_run, odor_responses_by_run, all_neuron_list,
                                     specialist_threshold=0.95, adapter_threshold=0.70):
    """
    Analyze neural architectures across all individual worms

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        odor_responses_by_run: Dict with run_id -> odor response data
        all_neuron_list: List of all neuron names
        specialist_threshold: Threshold for specialist classification
        adapter_threshold: Threshold for adapter classification

    Returns:
        architecture_analysis: Dict with comprehensive analysis results
    """

    architecture_data = []
    separation_details = {}

    print("=== INDIVIDUAL NEURAL ARCHITECTURE ANALYSIS ===")

    for run_id in top_pairs_by_run.keys():
        # Extract high connectivity neurons
        connectivity_neurons = []
        for pair in top_pairs_by_run[run_id]:
            connectivity_neurons.extend([pair['sender_name'], pair['receiver_name']])
        connectivity_neurons = list(set(connectivity_neurons))  # Remove duplicates

        # Extract high odor-responsive neurons from all odors
        odor_responsive_neurons = set()
        if run_id in odor_responses_by_run:
            for odor in ['butanone', 'pentanedione', 'NaCL']:
                if odor in odor_responses_by_run[run_id]:
                    odor_responsive_neurons.update(odor_responses_by_run[run_id][odor]['names'])
        odor_responsive_neurons = list(odor_responsive_neurons)

        # Compute separation metrics
        separation_metrics = compute_separation_index(connectivity_neurons, odor_responsive_neurons)

        # Classify architecture
        architecture_type = classify_neural_architecture(
            separation_metrics['separation_index'],
            specialist_threshold,
            adapter_threshold
        )

        # Store data
        architecture_data.append({
            'run_id': run_id,
            'architecture_type': architecture_type,
            'separation_index': separation_metrics['separation_index'],
            'overlap_count': separation_metrics['overlap_count'],
            'connectivity_purity': separation_metrics['connectivity_purity'],
            'odor_purity': separation_metrics['odor_purity'],
            'n_connectivity_neurons': separation_metrics['total_connectivity_neurons'],
            'n_odor_neurons': separation_metrics['total_odor_neurons']
        })

        separation_details[run_id] = separation_metrics

        print(f"Run {run_id}: {architecture_type.upper()} "
              f"(separation: {separation_metrics['separation_index']:.3f}, "
              f"overlap: {separation_metrics['overlap_count']})")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(architecture_data)

    # Summary statistics by architecture type
    type_summary = df.groupby('architecture_type').agg({
        'separation_index': ['count', 'mean', 'std', 'min', 'max'],
        'overlap_count': ['mean', 'std'],
        'n_connectivity_neurons': ['mean', 'std'],
        'n_odor_neurons': ['mean', 'std']
    }).round(3)

    print("\n=== ARCHITECTURE TYPE SUMMARY ===")
    print(type_summary)

    return {
        'architecture_data': df,
        'separation_details': separation_details,
        'type_summary': type_summary,
        'classification_thresholds': {
            'specialist': specialist_threshold,
            'adapter': adapter_threshold
        }
    }


def identify_hub_neurons_by_type(top_pairs_by_run, architecture_analysis, min_frequency=0.5):
    """
    Identify hub neurons for each architecture type

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        architecture_analysis: Results from analyze_individual_architectures
        min_frequency: Minimum frequency to be considered a hub

    Returns:
        hub_analysis: Dict with hub neuron analysis by architecture type
    """

    # Get architecture types for each run
    run_to_type = dict(zip(architecture_analysis['architecture_data']['run_id'],
                           architecture_analysis['architecture_data']['architecture_type']))

    # Group by architecture type
    hubs_by_type = defaultdict(lambda: defaultdict(int))
    connection_counts_by_type = defaultdict(int)

    print("=== HUB NEURON ANALYSIS BY ARCHITECTURE TYPE ===")

    for run_id, pairs in top_pairs_by_run.items():
        arch_type = run_to_type[run_id]
        connection_counts_by_type[arch_type] += 1

        # Count neuron appearances in this run
        neuron_counts = defaultdict(int)
        for pair in pairs:
            neuron_counts[pair['sender_name']] += 1
            neuron_counts[pair['receiver_name']] += 1

        # Add to architecture type totals
        for neuron, count in neuron_counts.items():
            hubs_by_type[arch_type][neuron] += count

    # Analyze hubs for each architecture type
    hub_analysis = {}

    for arch_type in hubs_by_type.keys():
        n_runs = connection_counts_by_type[arch_type]

        # Calculate frequencies and identify hubs
        neuron_frequencies = {}
        for neuron, total_count in hubs_by_type[arch_type].items():
            # Frequency = appearances / total possible appearances
            max_possible = n_runs * 40  # Max appearances if in all top-20 pairs as both sender and receiver
            frequency = total_count / max_possible
            neuron_frequencies[neuron] = {
                'total_count': total_count,
                'frequency': frequency,
                'n_runs': n_runs
            }

        # Sort by frequency
        sorted_neurons = sorted(neuron_frequencies.items(),
                                key=lambda x: x[1]['frequency'], reverse=True)

        # Identify hubs above threshold
        hubs = [(neuron, stats) for neuron, stats in sorted_neurons
                if stats['frequency'] >= min_frequency]

        hub_analysis[arch_type] = {
            'all_neurons': dict(sorted_neurons),
            'hub_neurons': hubs,
            'n_runs': n_runs,
            'top_10_neurons': sorted_neurons[:10]
        }

        print(f"\n{arch_type.upper()} ARCHITECTURE ({n_runs} individuals):")
        print("Top 10 hub neurons:")
        for i, (neuron, stats) in enumerate(sorted_neurons[:10]):
            print(f"  {i + 1:2d}. {neuron}: {stats['frequency']:.3f} "
                  f"({stats['total_count']} total appearances)")

    return hub_analysis


def compare_pathway_organization(top_pairs_by_run, architecture_analysis, all_neuron_list):
    """
    Compare pathway organization across architecture types

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        architecture_analysis: Results from analyze_individual_architectures
        all_neuron_list: List of all neuron names

    Returns:
        pathway_analysis: Dict with pathway comparison results
    """

    # Define functional neuron classes
    neuron_classes = {
        'chemosensory': ['ADLR', 'ADLL', 'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR',
                         'ASKL', 'ASKR', 'ASHL', 'ASHR', 'ASJL', 'ASJR'],
        'command': ['AVAR', 'AVAL', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVKL', 'AVKR',
                    'AVHL', 'AVHR', 'AVJL', 'AVJR'],
        'ring_integration': ['RID', 'RIS', 'RIML', 'RIMR', 'RIBL', 'RIBR', 'RIAR', 'RIAL',
                             'RICL', 'RICR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR'],
        'motor': ['SMDVL', 'SMDVR', 'SMDDL', 'SMDDR', 'SMBVL', 'SMBVR', 'SMBDL', 'SMBDR',
                  'VB01', 'VB02', 'VB03', 'VB04', 'VB05', 'VB06', 'VB07', 'VB08', 'VB09', 'VB10', 'VB11',
                  'DB01', 'DB02', 'DB03', 'DB04', 'DB05', 'DB06', 'DB07'],
        'head_sensory': ['CEPVL', 'CEPVR', 'CEPDL', 'CEPDR', 'OLQVL', 'OLQVR', 'OLQDL', 'OLQDR',
                         'OLLR', 'OLLL'],
        'muscle': ['M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5', 'MI', 'I1L', 'I1R', 'I2L', 'I2R',
                   'I3', 'I4', 'I5', 'I6']
    }

    # Get architecture types for each run
    run_to_type = dict(zip(architecture_analysis['architecture_data']['run_id'],
                           architecture_analysis['architecture_data']['architecture_type']))

    # Analyze pathway patterns by architecture type
    pathway_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    connection_types = defaultdict(lambda: defaultdict(int))

    print("=== PATHWAY ORGANIZATION ANALYSIS ===")

    for run_id, pairs in top_pairs_by_run.items():
        arch_type = run_to_type[run_id]

        for pair in pairs:
            sender = pair['sender_name']
            receiver = pair['receiver_name']
            weight = pair.get('max_response', 0)  # Use weight if available

            # Classify sender and receiver
            sender_class = 'other'
            receiver_class = 'other'

            for class_name, neurons in neuron_classes.items():
                if sender in neurons:
                    sender_class = class_name
                if receiver in neurons:
                    receiver_class = class_name

            # Record connection type
            connection_type = f"{sender_class}→{receiver_class}"
            connection_types[arch_type][connection_type] += 1

            # Record pathway statistics
            pathway_stats[arch_type][sender_class]['out_degree'] += 1
            pathway_stats[arch_type][receiver_class]['in_degree'] += 1
            pathway_stats[arch_type][sender_class]['out_weight'] += weight
            pathway_stats[arch_type][receiver_class]['in_weight'] += weight

    # Normalize by number of runs of each type
    type_counts = architecture_analysis['architecture_data']['architecture_type'].value_counts()

    normalized_connections = {}
    for arch_type in connection_types.keys():
        n_runs = type_counts[arch_type]
        normalized_connections[arch_type] = {
            conn_type: count / n_runs
            for conn_type, count in connection_types[arch_type].items()
        }

    # Print results
    for arch_type in ['specialist', 'adapter', 'generalist']:
        if arch_type in normalized_connections:
            print(f"\n{arch_type.upper()} PATHWAY PATTERNS (avg per individual):")

            # Sort connection types by frequency
            sorted_connections = sorted(normalized_connections[arch_type].items(),
                                        key=lambda x: x[1], reverse=True)

            for conn_type, avg_count in sorted_connections[:10]:
                print(f"  {conn_type}: {avg_count:.2f}")

    return {
        'pathway_stats': dict(pathway_stats),
        'connection_types': dict(connection_types),
        'normalized_connections': normalized_connections,
        'neuron_classes': neuron_classes
    }


def normalize_edge_function_amplitudes(edge_functions_by_run, method='z_score'):
    """
    Normalize edge function amplitudes across runs to account for different scales

    Args:
        edge_functions_by_run: Dict with run_id -> edge function data
        method: Normalization method ('z_score', 'min_max', 'robust')

    Returns:
        normalized_functions: Dict with normalized edge function data
    """

    normalized_functions = {}

    for run_id, edge_data in edge_functions_by_run.items():
        if method == 'z_score':
            # Z-score normalization
            mean_val = np.mean(edge_data)
            std_val = np.std(edge_data)
            normalized = (edge_data - mean_val) / std_val if std_val > 0 else edge_data

        elif method == 'min_max':
            # Min-max normalization to [0, 1]
            min_val = np.min(edge_data)
            max_val = np.max(edge_data)
            normalized = (edge_data - min_val) / (max_val - min_val) if max_val > min_val else edge_data

        elif method == 'robust':
            # Robust normalization using median and IQR
            median_val = np.median(edge_data)
            q75 = np.percentile(edge_data, 75)
            q25 = np.percentile(edge_data, 25)
            iqr = q75 - q25
            normalized = (edge_data - median_val) / iqr if iqr > 0 else edge_data

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized_functions[run_id] = normalized

    return normalized_functions


def plot_architecture_analysis_summary(architecture_analysis, hub_analysis, pathway_analysis):
    """
    Create comprehensive visualization of architecture analysis results

    Args:
        architecture_analysis: Results from analyze_individual_architectures
        hub_analysis: Results from identify_hub_neurons_by_type
        pathway_analysis: Results from compare_pathway_organization

    Returns:
        fig: matplotlib figure with summary plots
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Distribution of architecture types
    df = architecture_analysis['architecture_data']
    type_counts = df['architecture_type'].value_counts()

    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Distribution of Architecture Types')

    # 2. Separation index distribution
    for arch_type in df['architecture_type'].unique():
        subset = df[df['architecture_type'] == arch_type]
        axes[0, 1].hist(subset['separation_index'], alpha=0.7, label=arch_type, bins=10)

    axes[0, 1].set_xlabel('Separation Index')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Separation Index by Architecture Type')
    axes[0, 1].legend()

    # 3. Overlap count vs separation index
    colors = {'specialist': 'blue', 'adapter': 'green', 'generalist': 'red'}
    for arch_type in df['architecture_type'].unique():
        subset = df[df['architecture_type'] == arch_type]
        axes[0, 2].scatter(subset['separation_index'], subset['overlap_count'],
                           c=colors.get(arch_type, 'gray'), label=arch_type, alpha=0.7)

    axes[0, 2].set_xlabel('Separation Index')
    axes[0, 2].set_ylabel('Overlap Count')
    axes[0, 2].set_title('Separation vs Overlap')
    axes[0, 2].legend()

    # 4. Hub neuron frequency comparison
    if len(hub_analysis) >= 2:
        arch_types = list(hub_analysis.keys())[:2]  # Compare first two types

        # Get top neurons for each type
        neurons_type1 = [item[0] for item in hub_analysis[arch_types[0]]['top_10_neurons']]
        freq_type1 = [item[1]['frequency'] for item in hub_analysis[arch_types[0]]['top_10_neurons']]

        neurons_type2 = [item[0] for item in hub_analysis[arch_types[1]]['top_10_neurons']]
        freq_type2 = [item[1]['frequency'] for item in hub_analysis[arch_types[1]]['top_10_neurons']]

        x_pos = np.arange(len(neurons_type1))
        width = 0.35

        axes[1, 0].bar(x_pos - width / 2, freq_type1, width, label=arch_types[0])
        axes[1, 0].bar(x_pos + width / 2, freq_type2, width, label=arch_types[1])

        axes[1, 0].set_xlabel('Neurons')
        axes[1, 0].set_ylabel('Hub Frequency')
        axes[1, 0].set_title(f'Top Hub Neurons: {arch_types[0]} vs {arch_types[1]}')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(neurons_type1, rotation=45)
        axes[1, 0].legend()

    # 5. Connection type heatmap
    if 'normalized_connections' in pathway_analysis:
        # Create matrix of connection types vs architecture types
        all_arch_types = list(pathway_analysis['normalized_connections'].keys())
        all_conn_types = set()
        for arch_conns in pathway_analysis['normalized_connections'].values():
            all_conn_types.update(arch_conns.keys())
        all_conn_types = sorted(list(all_conn_types))

        matrix = np.zeros((len(all_conn_types), len(all_arch_types)))
        for i, conn_type in enumerate(all_conn_types):
            for j, arch_type in enumerate(all_arch_types):
                matrix[i, j] = pathway_analysis['normalized_connections'][arch_type].get(conn_type, 0)

        im = axes[1, 1].imshow(matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_xticks(range(len(all_arch_types)))
        axes[1, 1].set_xticklabels(all_arch_types)
        axes[1, 1].set_yticks(range(len(all_conn_types)))
        axes[1, 1].set_yticklabels(all_conn_types, fontsize=8)
        axes[1, 1].set_title('Connection Types by Architecture')
        plt.colorbar(im, ax=axes[1, 1])

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    SUMMARY STATISTICS

    Total Individuals: {len(df)}

    Architecture Types:
    """

    for arch_type in df['architecture_type'].unique():
        count = sum(df['architecture_type'] == arch_type)
        mean_sep = df[df['architecture_type'] == arch_type]['separation_index'].mean()
        summary_text += f"  {arch_type}: {count} ({count / len(df) * 100:.1f}%)\n"
        summary_text += f"    Mean separation: {mean_sep:.3f}\n"

    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    return fig


def run_neural_architecture_pipeline(top_pairs_by_run, odor_responses_by_run, all_neuron_list,
                                     specialist_threshold=0.95, adapter_threshold=0.70):
    """
    Run the complete neural architecture analysis pipeline

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        odor_responses_by_run: Dict with run_id -> odor response data
        all_neuron_list: List of all neuron names
        specialist_threshold: Threshold for specialist classification
        adapter_threshold: Threshold for adapter classification

    Returns:
        complete_analysis: Dict with all analysis results
    """

    print("RUNNING NEURAL ARCHITECTURE ANALYSIS PIPELINE")
    print("=" * 60)

    # Phase 1: Individual Architecture Classification
    architecture_analysis = analyze_individual_architectures(
        top_pairs_by_run, odor_responses_by_run, all_neuron_list,
        specialist_threshold, adapter_threshold
    )

    # Phase 2: Hub Neuron Analysis
    hub_analysis = identify_hub_neurons_by_type(
        top_pairs_by_run, architecture_analysis
    )

    # Phase 2: Pathway Organization Comparison
    pathway_analysis = compare_pathway_organization(
        top_pairs_by_run, architecture_analysis, all_neuron_list
    )

    # Create summary visualization
    summary_fig = plot_architecture_analysis_summary(
        architecture_analysis, hub_analysis, pathway_analysis
    )

    return {
        'architecture_analysis': architecture_analysis,
        'hub_analysis': hub_analysis,
        'pathway_analysis': pathway_analysis,
        'summary_figure': summary_fig
    }


