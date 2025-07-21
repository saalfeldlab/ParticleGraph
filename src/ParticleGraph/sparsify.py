import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.cluster.hierarchy as hcluster

from ParticleGraph.utils import *
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, silhouette_score
from scipy.optimize import linear_sum_assignment

class EmbeddingCluster:
    def __init__(self, config):
        self.cluster_connectivity = config.training.cluster_connectivity    # 'single' (default) or 'average'

    def get(self, data, method, thresh=2.5):

        match method:
            case 'kmeans_auto':
                silhouette_avg_list = []
                silhouette_max = 0
                n_clusters = None
                for n in range(2, 10):
                    clusterer = KMeans(n_clusters=n, random_state=10, n_init='auto')
                    cluster_labels = clusterer.fit_predict(data)
                    if (np.unique(cluster_labels) == [0]):
                        n_clusters = 1
                    else:
                        silhouette_avg = silhouette_score(data, cluster_labels)
                        silhouette_avg_list.append(silhouette_avg)
                        if silhouette_avg > silhouette_max:
                            silhouette_max = silhouette_avg
                            n_clusters = n
                kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto')
                k = kmeans.fit(data)
                clusters = k.labels_
            case 'distance':
                clusters = hcluster.fclusterdata(data, thresh, criterion="distance", method=self.cluster_connectivity) - 1
                n_clusters = len(np.unique(clusters))
            case 'inconsistent':
                clusters = hcluster.fclusterdata(data, thresh, criterion="inconsistent", method=self.cluster_connectivity) - 1
                n_clusters = len(np.unique(clusters))

            case _:
                raise ValueError(f'Unknown method {method}')

        return clusters, n_clusters


def sparsify_cluster(cluster_method, proj_interaction, embedding, cluster_distance_threshold, type_list, n_particle_types, embedding_cluster):

    # normalization of projection because UMAP output is not normalized
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction)+1e-10)
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)+1e-10)

    match cluster_method:
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance', thresh=cluster_distance_threshold)
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'inconsistent_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'inconsistent', thresh=cluster_distance_threshold)
        case 'inconsistent_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'inconsistent', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance', thresh=cluster_distance_threshold)

    label_list = []
    for n in range(n_particle_types):
        pos = torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            tmp = labels[pos[:,0]]
            label_list.append(np.round(np.median(tmp)))
            pos_ = np.argwhere(labels == np.median(tmp))

    label_list = np.array(label_list)

    fig,ax = fig_init()
    for n in label_list:
        pos = np.argwhere(labels == n)
        # print(len(pos))
        if len(pos) > 0:
            ax.scatter(embedding[pos, 0], embedding[pos, 1], s=5)
    plt.close()

    new_labels = np.ones_like(labels) * n_particle_types
    for n in range(n_particle_types):
        if n < len(label_list):
            new_labels[labels == label_list[n]] = n

    fig,ax = fig_init()
    ax.scatter(proj_interaction[:, 0], proj_interaction[:, 1], c=new_labels, s=5, cmap='tab20')
    plt.close()

    return labels, n_clusters, new_labels

def sparsify_cluster_state(cluster_method, proj_interaction, embedding, cluster_distance_threshold, true_type_list, n_particle_types, embedding_cluster):

    # normalization of projection because UMAP output is not normalized
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction)+1e-10)
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)+1e-10)

    start_time = time.time()
    match cluster_method:
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance', thresh=cluster_distance_threshold)
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'inconsistent_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'inconsistent', thresh=cluster_distance_threshold)
        case 'inconsistent_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'inconsistent', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance', thresh=cluster_distance_threshold)

    computation_time = time.time() - start_time
    print(f"clustering computation time is {computation_time:0.2f} seconds.")

    label_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(true_type_list == n).squeeze().astype(int)
        if len(pos)>0:
            tmp = labels[pos]
            label_list.append(np.round(np.median(tmp)))
        else:
            label_list.append(0)
    label_list = np.array(label_list)
    new_labels = np.ones_like(labels) * n_particle_types
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n

    return labels, n_clusters, new_labels

def evaluate_embedding_clustering(model, type_list, n_types=64):
    """
    Cluster model.a embeddings and evaluate against true neuron types
    """

    # Extract embeddings
    embeddings = to_numpy(model.a)
    true_labels = to_numpy(type_list).flatten()  # Fix: add .flatten()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_types, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Calculate metrics that don't require label alignment
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

    # Calculate accuracy with optimal label mapping (Hungarian algorithm)
    def find_optimal_mapping(true_labels, cluster_labels, n_clusters):
        # Create confusion matrix
        confusion_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(len(true_labels)):
            confusion_matrix[int(true_labels[i]), int(cluster_labels[i])] += 1  # Add int() for safety

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # Create mapping dictionary
        mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}

        # Map cluster labels to true labels
        mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels])

        return mapped_labels

    # Get optimally mapped labels and calculate accuracy
    mapped_labels = find_optimal_mapping(true_labels, cluster_labels, n_types)
    accuracy = accuracy_score(true_labels, mapped_labels)

    return {
        'ari': ari_score,
        'nmi': nmi_score,
        'accuracy': accuracy,
        'cluster_labels': cluster_labels,
        'mapped_labels': mapped_labels
    }

def clustering_evaluation(model, type_list, eps=0.5):
    """
    Blind clustering using DBSCAN (doesn't require number of clusters)
    Parameter: eps - maximum distance between points in same cluster
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
    from scipy.optimize import linear_sum_assignment

    # Extract embeddings
    embeddings = to_numpy(model.a)
    true_labels = to_numpy(type_list).flatten()

    # Perform DBSCAN clustering (automatically finds number of clusters)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    cluster_labels = dbscan.fit_predict(embeddings)

    # Count found clusters (excluding noise points labeled as -1)
    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_points = list(cluster_labels).count(-1)

    # Handle noise points for metrics (assign them to separate cluster)
    cluster_labels_clean = cluster_labels.copy()
    cluster_labels_clean[cluster_labels_clean == -1] = n_clusters_found  # Assign noise to separate cluster

    # Calculate clustering metrics
    ari_score = adjusted_rand_score(true_labels, cluster_labels_clean)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels_clean)

    # Calculate accuracy with optimal label mapping (Hungarian algorithm)
    def find_optimal_mapping_blind(true_labels, cluster_labels):
        n_true_clusters = len(np.unique(true_labels))
        n_found_clusters = len(np.unique(cluster_labels))

        # Create confusion matrix
        confusion_matrix = np.zeros((n_true_clusters, n_found_clusters))
        for i in range(len(true_labels)):
            true_idx = int(true_labels[i])
            cluster_idx = int(cluster_labels[i])
            if 0 <= true_idx < n_true_clusters and 0 <= cluster_idx < n_found_clusters:
                confusion_matrix[true_idx, cluster_idx] += 1

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # Create mapping dictionary
        mapping = {}
        for i in range(len(col_ind)):
            if i < len(row_ind):
                mapping[col_ind[i]] = row_ind[i]

        # Map cluster labels to true labels
        mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels])

        return mapped_labels

    # Calculate accuracy
    mapped_labels = find_optimal_mapping_blind(true_labels, cluster_labels_clean)
    accuracy = accuracy_score(true_labels, mapped_labels)

    # Calculate silhouette score (clustering quality)
    from sklearn.metrics import silhouette_score
    if n_clusters_found > 1:
        silhouette = silhouette_score(embeddings, cluster_labels_clean)
    else:
        silhouette = 0.0

    return {
        'n_clusters_found': n_clusters_found,
        'n_noise_points': n_noise_points,
        'eps_used': eps,
        'ari': ari_score,
        'nmi': nmi_score,
        'accuracy': accuracy,
        'silhouette': silhouette,
        'cluster_labels': cluster_labels,
        'mapped_labels': mapped_labels,
        'total_points': len(embeddings)
    }

def functional_clustering_evaluation(func_list, type_list, eps=0.5, min_samples=5, normalize=True):
    """
    Cluster neurons based on their phi function responses instead of embeddings

    Parameters:
    - func_list: List of torch tensors, each containing phi function output for one neuron
    - type_list: True neuron type labels
    - eps: DBSCAN epsilon parameter
    - min_samples: DBSCAN min_samples parameter
    - normalize: Whether to normalize function responses
    """

    if isinstance(func_list, torch.Tensor):
        # func_list is already a tensor
        func_features = to_numpy(func_list)
    elif isinstance(func_list, list) and len(func_list) > 0:
        if isinstance(func_list[0], torch.Tensor):
            # Stack all functions into single array
            func_array = torch.stack(func_list).squeeze()  # Shape: (n_neurons, n_points)
            func_features = to_numpy(func_array)
            print("Stacked list of tensors")
        else:
            func_features = np.array(func_list)
            print("Converted list to numpy array")
    else:
        raise ValueError(f"Unexpected func_list type: {type(func_list)}")

    # Handle different function output shapes
    if len(func_features.shape) == 3:
        # If functions are (n_neurons, n_points, 1), flatten last dimension
        func_features = func_features.squeeze(-1)


    # Normalize functions if requested
    if normalize:
        scaler = StandardScaler()
        func_features_processed = scaler.fit_transform(func_features)
    else:
        func_features_processed = func_features

    # Extract true labels
    true_labels = to_numpy(type_list).flatten()

    # Perform DBSCAN clustering on functional responses
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(func_features_processed)

    # Count clusters and noise points
    unique_clusters = np.unique(cluster_labels)
    n_clusters_found = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_noise_points = np.sum(cluster_labels == -1)

    # Handle noise points for metrics (assign to separate cluster)
    cluster_labels_clean = cluster_labels.copy()
    cluster_labels_clean[cluster_labels_clean == -1] = n_clusters_found  # Assign noise to separate cluster

    # Calculate clustering metrics
    ari_score = adjusted_rand_score(true_labels, cluster_labels_clean)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels_clean)

    # Calculate silhouette score (functional clustering quality)
    if n_clusters_found > 1 and n_noise_points < len(cluster_labels):
        # Use original cluster labels for silhouette (excluding noise points)
        valid_indices = cluster_labels != -1
        if np.sum(valid_indices) > 1:
            silhouette = silhouette_score(
                func_features_processed[valid_indices],
                cluster_labels[valid_indices]
            )
        else:
            silhouette = 0.0
    else:
        silhouette = 0.0

    # Calculate accuracy with optimal mapping (Hungarian algorithm)
    def find_optimal_functional_mapping(true_labels, cluster_labels):
        n_true_clusters = len(np.unique(true_labels))
        n_found_clusters = len(np.unique(cluster_labels))

        # Create confusion matrix
        confusion_matrix = np.zeros((n_true_clusters, n_found_clusters))
        for i in range(len(true_labels)):
            true_idx = int(true_labels[i])
            cluster_idx = int(cluster_labels[i])
            if 0 <= true_idx < n_true_clusters and 0 <= cluster_idx < n_found_clusters:
                confusion_matrix[true_idx, cluster_idx] += 1

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # Create mapping dictionary
        mapping = {}
        for i in range(len(col_ind)):
            if i < len(row_ind):
                mapping[col_ind[i]] = row_ind[i]

        # Map cluster labels to true labels
        mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels])

        return mapped_labels, confusion_matrix

    # Get optimal mapping and calculate accuracy
    mapped_labels, confusion_matrix = find_optimal_functional_mapping(true_labels, cluster_labels_clean)
    accuracy = accuracy_score(true_labels, mapped_labels)

    # Calculate additional functional clustering metrics
    def calculate_functional_purity():
        """Calculate how 'pure' each functional cluster is in terms of true types"""
        cluster_purities = []
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue
            cluster_mask = cluster_labels == cluster_id
            cluster_true_types = true_labels[cluster_mask]
            if len(cluster_true_types) > 0:
                # Purity = fraction of most common true type in this cluster
                unique_types, counts = np.unique(cluster_true_types, return_counts=True)
                purity = np.max(counts) / len(cluster_true_types)
                cluster_purities.append(purity)

        return np.mean(cluster_purities) if cluster_purities else 0.0

    def calculate_functional_completeness():
        """Calculate how completely each true type is captured by functional clusters"""
        type_completeness = []
        for true_type in np.unique(true_labels):
            type_mask = true_labels == true_type
            type_clusters = cluster_labels[type_mask]
            type_clusters_clean = type_clusters[type_clusters != -1]  # Exclude noise

            if len(type_clusters_clean) > 0:
                # Completeness = fraction in largest cluster for this type
                unique_clusters_for_type, counts = np.unique(type_clusters_clean, return_counts=True)
                completeness = np.max(counts) / len(type_clusters_clean)
                type_completeness.append(completeness)

        return np.mean(type_completeness) if type_completeness else 0.0

    purity = calculate_functional_purity()
    completeness = calculate_functional_completeness()

    # Calculate function diversity within vs between clusters
    def calculate_functional_separation():
        """Calculate how well separated functional clusters are"""
        if n_clusters_found <= 1:
            return 0.0, 0.0, 0.0

        within_cluster_distances = []
        between_cluster_distances = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_functions = func_features_processed[cluster_mask]

            if len(cluster_functions) > 1:
                # Calculate pairwise distances within cluster
                from sklearn.metrics.pairwise import euclidean_distances
                within_distances = euclidean_distances(cluster_functions)
                # Take upper triangle (avoid diagonal and duplicates)
                upper_triangle = np.triu_indices_from(within_distances, k=1)
                within_cluster_distances.extend(within_distances[upper_triangle])

            # Calculate distances to other clusters
            other_cluster_mask = (cluster_labels != cluster_id) & (cluster_labels != -1)
            if np.any(other_cluster_mask):
                other_functions = func_features_processed[other_cluster_mask]
                between_distances = euclidean_distances(cluster_functions, other_functions)
                between_cluster_distances.extend(between_distances.flatten())

        avg_within = np.mean(within_cluster_distances) if within_cluster_distances else 0.0
        avg_between = np.mean(between_cluster_distances) if between_cluster_distances else 0.0
        separation_ratio = avg_between / avg_within if avg_within > 0 else 0.0

        return avg_within, avg_between, separation_ratio

    avg_within_dist, avg_between_dist, separation_ratio = calculate_functional_separation()

    return {
        'n_clusters_found': n_clusters_found,
        'n_noise_points': n_noise_points,
        'eps_used': eps,
        'min_samples_used': min_samples,
        'ari': ari_score,
        'nmi': nmi_score,
        'accuracy': accuracy,
        'silhouette': silhouette,
        'purity': purity,
        'completeness': completeness,
        'avg_within_cluster_distance': avg_within_dist,
        'avg_between_cluster_distance': avg_between_dist,
        'separation_ratio': separation_ratio,
        'cluster_labels': cluster_labels,
        'mapped_labels': mapped_labels,
        'confusion_matrix': confusion_matrix,
        'total_points': len(func_features),
        'normalization_applied': normalize,
        'function_features_shape': func_features.shape
    }


# Usage example:
# After running your phi function plotting code:
# results = functional_clustering_evaluation(func_list, type_list, eps=0.2)
# comparison = compare_embedding_vs_functional_clustering(model, type_list, func_list)
if __name__ == '__main__':
    # generate 3 clusters of each around 100 points and one orphan point

    N = 100
    data = np.random.randn(3 * N, 2)
    data[:N] += 5
    data[-N:] += 10
    data[-1:] -= 20

    # clustering
    thresh = 1.5
    clusters, n_clusters = embedding_cluster.get(data, method="distance")

    # plotting
    plt.scatter(*np.transpose(data), c=clusters, s=5)
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (thresh, n_clusters)
    plt.title(title)
    plt.show()
