
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from mpl_toolkits.mplot3d import Axes3D


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

        # Find all connected senders (where adjacency_matrix[sender, receiver] = 1)
        connected_senders = np.where(adjacency_matrix[:, neuron_id] == 1)[0]

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
    ax_lines.grid(True, alpha=0.3)

    # Adaptive legend placement based on number of neurons
    n_neurons = len(neuron_stats)
    if n_neurons <= 50:
        # For few neurons, use right side
        ax_lines.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig_lines


def analyze_mlp_edge_lines_weighted(model, neuron_name, all_neuron_list, adjacency_matrix, weight_matrix,
                                    signal_range=(0, 10), resolution=100, device=None):
    """
    Create line plots showing weighted edge function vs signal difference for a single neuron of interest
    Uses adjacency matrix to find connections and weight matrix to scale the outputs
    Plots individual lines for each incoming connection

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
    """

    embedding = model.a  # Shape: (300, 2)

    print(f"generating weighted line plots for {neuron_name} using adjacency and weight matrices...")

    # Get index of the neuron of interest
    try:
        neuron_id = get_neuron_index(neuron_name, all_neuron_list)
    except ValueError as e:
        raise ValueError(f"Neuron '{neuron_name}' not found: {e}")

    receiver_embedding = embedding[neuron_id]  # This neuron as receiver (embedding_i)

    # Find all connected senders (where adjacency_matrix[sender, receiver] = 1)
    connected_senders = np.where(adjacency_matrix[:, neuron_id] == 1)[0]

    if len(connected_senders) == 0:
        raise ValueError(f"No incoming connections found for {neuron_name}")

    print(f"Found {len(connected_senders)} incoming connections for {neuron_name}")

    # Create signal difference array for line plots
    u_diff_line = torch.linspace(-signal_range[1], signal_range[1], resolution * 2 - 1, device=device)

    # Store outputs and metadata for all connections
    connection_data = []

    for sender_id in connected_senders:
        sender_name = all_neuron_list[sender_id]
        sender_embedding = embedding[sender_id]  # Connected neuron as sender (embedding_j)
        connection_weight = weight_matrix[sender_id, neuron_id]  # Weight for this connection

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

    u_diff_line_np = u_diff_line.cpu().numpy()

    # Plot each connection
    for conn_idx, conn_data in enumerate(connection_data):
        color = colors[conn_idx]
        sender_name = conn_data['sender_name']
        weight = conn_data['weight']
        weighted_output = conn_data['output']

        # Line style based on weight sign
        line_style = '-' if weight >= 0 else '--'
        line_width = 1.5 + min(2.0, abs(weight) / np.max(
            np.abs([c['weight'] for c in connection_data])))  # Thicker for stronger weights

        ax_lines.plot(u_diff_line_np, weighted_output,
                      color=color, linewidth=line_width, linestyle=line_style,
                      label=f'{sender_name} (w={weight:.3f})')

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

    # Add text annotation explaining line styles
    ax_lines.text(0.02, 0.98, 'Line style: solid (w≥0), dashed (w<0)\nLine width ∝ |weight|',
                  transform=ax_lines.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  fontsize='small')

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

    # Find all connected senders (where adjacency_matrix[sender, receiver] = 1)
    connected_senders = np.where(adjacency_matrix[:, neuron_id] == 1)[0]

    if len(connected_senders) == 0:
        print(f"No incoming connections found for {neuron_name}")
        return None, None

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
        connection_weight = weight_matrix[sender_id, neuron_id]  # Weight for this connection

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
        line_width = 1.5 + min(2.0, abs(weight) / np.max(
            np.abs([c['weight'] for c in connection_data])))  # Thicker for stronger weights

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

    # Add text annotation explaining line styles
    ax_lines.text(0.02, 0.98, 'Line style: solid (w≥0), dashed (w<0)\nLine width ∝ |weight|',
                  transform=ax_lines.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  fontsize='small')

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

    # # Generate plots for top pairs
    # top_figures = []
    # for i, pair in enumerate(top_pairs):
    #     print(f"\nGenerating plot {i + 1}/{top_k} for {pair['receiver_name']} ← {pair['sender_name']}")
    #
    #     fig, _ = analyze_mlp_edge_lines_weighted_with_max(
    #         model, pair['receiver_name'], all_neuron_list, adjacency_matrix, weight_matrix,
    #         signal_range, resolution, device
    #     )
    #
    #     if fig is not None:
    #         # Update title to indicate this is a top pair
    #         fig.suptitle(f"Top #{i + 1} responding pair: {pair['receiver_name']} ← {pair['sender_name']}\n"
    #                      f"Max response: {pair['max_response']:.4f}", fontsize=14)
    #         top_figures.append(fig)

    return top_pairs    # , top_figures


# Usage example:
# top_pairs, top_figures = find_top_responding_pairs(
#     model, all_neuron_list, adjacency_matrix, weight_matrix,
#     signal_range=(0, 10), resolution=100, device=device, top_k=10
# )
#
# # Save the top figures
# for i, fig in enumerate(top_figures):
#     fig.savefig(f"top_pair_{i+1}.png", dpi=300, bbox_inches='tight')
#     plt.close(fig)



def analyze_mlp_edge_synaptic(model, n_neurons=300, signal_range=(0, 10), resolution=50, n_sample_pairs=200,
                              device=None):
    """
    Analyze the learned MLP edge function with statistical sampling
    Creates 2D heatmaps and scatter plot of edge function vs signal difference
    (Line plots are now handled by separate analyze_mlp_edge_lines function)
    """

    embedding = model.a  # Shape: (300, 2)

    # Create signal grid
    u_vals = torch.linspace(signal_range[0], signal_range[1], resolution, device=device)
    u_i_grid, u_j_grid = torch.meshgrid(u_vals, u_vals, indexing='ij')

    # Flatten for batch processing
    u_i_flat = u_i_grid.flatten().unsqueeze(1)  # (resolution^2, 1)
    u_j_flat = u_j_grid.flatten().unsqueeze(1)  # (resolution^2, 1)
    n_grid_points = len(u_i_flat)

    print(f"sampling {n_sample_pairs} random neuron pairs across {resolution}x{resolution} signal grid...")

    # Sample random neuron pairs
    np.random.seed(42)  # For reproducibility
    neuron_indices = np.random.choice(n_neurons, size=(n_sample_pairs, 2), replace=True)

    # Store all outputs for statistics
    all_outputs = torch.zeros(n_sample_pairs, resolution, resolution, device=device)

    # Store data for scatter plot (signal differences and corresponding outputs)
    signal_diffs = []
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
            # Calculate signal differences for this batch (u_j - u_i)
            u_diff = (u_j_flat - u_i_flat).squeeze().cpu().numpy()  # (n_grid_points,)
            edge_vals = lin_edge[batch_idx].flatten().cpu().numpy()  # (n_grid_points,)

            signal_diffs.extend(u_diff)
            edge_outputs.extend(edge_vals)

    mean_output = torch.mean(all_outputs, dim=0).cpu().numpy()
    std_output = torch.std(all_outputs, dim=0).cpu().numpy()

    # Convert scatter plot data to numpy arrays
    signal_diffs = np.array(signal_diffs)
    edge_outputs = np.array(edge_outputs)

    # Create 2D heatmap plots
    fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(np.flipud(mean_output), extent=[signal_range[0], signal_range[1],
                                                     signal_range[0], signal_range[1]],
                     origin='lower', cmap='viridis', aspect='equal')
    ax1.set_title(f'mean edge function\n(over {n_sample_pairs} random pairs)')
    ax1.set_xlabel('u_i (signal)')
    ax1.set_ylabel('u_j (signal)')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('mean output')

    im2 = ax2.imshow(np.flipud(std_output), extent=[signal_range[0], signal_range[1],
                                                    signal_range[0], signal_range[1]],
                     origin='lower', cmap='plasma', aspect='equal')
    ax2.set_title(f'std dev of edge function\n(over {n_sample_pairs} random pairs)')
    ax2.set_xlabel('u_i (signal)')
    ax2.set_ylabel('u_j (signal)')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('std dev')

    plt.tight_layout()

    # Create scatter plot of edge function vs signal difference
    fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(10, 6))

    # Subsample for visualization if too many points
    max_points = 10000
    if len(signal_diffs) > max_points:
        indices = np.random.choice(len(signal_diffs), max_points, replace=False)
        signal_diffs_plot = signal_diffs[indices]
        edge_outputs_plot = edge_outputs[indices]
        alpha_val = 0.3
    else:
        signal_diffs_plot = signal_diffs
        edge_outputs_plot = edge_outputs
        alpha_val = 0.5

    # Create scatter plot
    scatter = ax_scatter.scatter(signal_diffs_plot, edge_outputs_plot,
                                 alpha=alpha_val, s=1, c=edge_outputs_plot,
                                 cmap='viridis', rasterized=True)

    ax_scatter.set_xlabel('u_j - u_i (signal difference)')
    ax_scatter.set_ylabel('edge function output')
    ax_scatter.set_title(f'edge function vs signal difference\n(sampled from {n_sample_pairs} neuron pairs)')
    ax_scatter.grid(True, alpha=0.3)

    # Add colorbar
    cbar_scatter = plt.colorbar(scatter, ax=ax_scatter)
    cbar_scatter.set_label('edge output')

    plt.tight_layout()

    return fig_2d, fig_scatter


def analyze_mlp_edge_synaptic(model, n_neurons=300, signal_range=(0, 10), resolution=50, n_sample_pairs=200,
                              device=None):
    """
    Analyze the learned MLP edge function with statistical sampling
    Creates 2D heatmaps and scatter plot of edge function vs signal difference
    (Line plots are now handled by separate analyze_mlp_edge_lines function)
    """

    embedding = model.a  # Shape: (300, 2)

    # Create signal grid
    u_vals = torch.linspace(signal_range[0], signal_range[1], resolution, device=device)
    u_i_grid, u_j_grid = torch.meshgrid(u_vals, u_vals, indexing='ij')

    # Flatten for batch processing
    u_i_flat = u_i_grid.flatten().unsqueeze(1)  # (resolution^2, 1)
    u_j_flat = u_j_grid.flatten().unsqueeze(1)  # (resolution^2, 1)
    n_grid_points = len(u_i_flat)

    print(f"sampling {n_sample_pairs} random neuron pairs across {resolution}x{resolution} signal grid...")

    # Sample random neuron pairs
    np.random.seed(42)  # For reproducibility
    neuron_indices = np.random.choice(n_neurons, size=(n_sample_pairs, 2), replace=True)

    # Store all outputs for statistics
    all_outputs = torch.zeros(n_sample_pairs, resolution, resolution, device=device)

    # Store data for scatter plot (signal differences and corresponding outputs)
    signal_diffs = []
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
            # Calculate signal differences for this batch (u_j - u_i)
            u_diff = (u_j_flat - u_i_flat).squeeze().cpu().numpy()  # (n_grid_points,)
            edge_vals = lin_edge[batch_idx].flatten().cpu().numpy()  # (n_grid_points,)

            signal_diffs.extend(u_diff)
            edge_outputs.extend(edge_vals)

    mean_output = torch.mean(all_outputs, dim=0).cpu().numpy()
    std_output = torch.std(all_outputs, dim=0).cpu().numpy()

    # Convert scatter plot data to numpy arrays
    signal_diffs = np.array(signal_diffs)
    edge_outputs = np.array(edge_outputs)

    # Create 2D heatmap plots
    fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(np.flipud(mean_output), extent=[signal_range[0], signal_range[1],
                                                     signal_range[0], signal_range[1]],
                     origin='lower', cmap='viridis', aspect='equal')
    ax1.set_title(f'mean edge function\n(over {n_sample_pairs} random pairs)')
    ax1.set_xlabel('u_i (signal)')
    ax1.set_ylabel('u_j (signal)')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('mean output')

    im2 = ax2.imshow(np.flipud(std_output), extent=[signal_range[0], signal_range[1],
                                                    signal_range[0], signal_range[1]],
                     origin='lower', cmap='plasma', aspect='equal')
    ax2.set_title(f'std dev of edge function\n(over {n_sample_pairs} random pairs)')
    ax2.set_xlabel('u_i (signal)')
    ax2.set_ylabel('u_j (signal)')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('std dev')

    plt.tight_layout()

    # Create scatter plot of edge function vs signal difference
    fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(10, 6))

    # Subsample for visualization if too many points
    max_points = 10000
    if len(signal_diffs) > max_points:
        indices = np.random.choice(len(signal_diffs), max_points, replace=False)
        signal_diffs_plot = signal_diffs[indices]
        edge_outputs_plot = edge_outputs[indices]
        alpha_val = 0.3
    else:
        signal_diffs_plot = signal_diffs
        edge_outputs_plot = edge_outputs
        alpha_val = 0.5

    # Create scatter plot
    scatter = ax_scatter.scatter(signal_diffs_plot, edge_outputs_plot,
                                 alpha=alpha_val, s=1, c=edge_outputs_plot,
                                 cmap='viridis', rasterized=True)

    ax_scatter.set_xlabel('u_j - u_i (signal difference)')
    ax_scatter.set_ylabel('edge function output')
    ax_scatter.set_title(f'edge function vs signal difference\n(sampled from {n_sample_pairs} neuron pairs)')
    ax_scatter.grid(True, alpha=0.3)

    # Add colorbar
    cbar_scatter = plt.colorbar(scatter, ax=ax_scatter)
    cbar_scatter.set_label('edge output')

    plt.tight_layout()

    return fig_2d, fig_scatter

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