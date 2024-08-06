"""
This script demonstrates how to load a large C. elegans data set with cell positions and gene expression data and how
to access the data.
"""
import os.path
import torch
from GNN_particles_Ntype import *
from ParticleGraph.utils import set_size
from tqdm import trange
import matplotlib.pyplot as plt
from ParticleGraph.generators.cell_utils import *
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import tifffile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from ParticleGraph.models.utils import *
# from ParticleGraph.utils import *
from ParticleGraph.data_loaders import load_celegans_gene_data
#from ParticleGraph.generators.graph_data_generator import *


def choose_boundary_values(bc_name):
    def identity(x):
        return x

    def periodic(x):
        return torch.remainder(x, 1.0)  # in [0, 1)

    def periodic_special(x):
        return torch.remainder(x, 1.0) + (x > 10) * 10  # to discard dead cells set at x=10

    def shifted_periodic(x):
        return torch.remainder(x - 0.5, 1.0) - 0.5  # in [-0.5, 0.5)

    def shifted_periodic_special(x):
        return torch.remainder(x - 0.5, 1.0) - 0.5 + (x > 10) * 10  # to discard dead cells set at x=10

    match bc_name:
        case 'no':
            return identity, identity
        case 'periodic':
            return periodic, shifted_periodic
        case 'periodic_special':
            return periodic, shifted_periodic
        case _:
            raise ValueError(f'Unknown boundary condition {bc_name}')
        
bc_pos, bc_dpos = choose_boundary_values("no")

def place_cells(dimension, n_particles, device="cuda0"):
    pos = torch.rand(1, dimension, device=device)
    pos[:, 0] = pos[:, 0] * 20 - 10
    pos[:, 1] = pos[:, 1] * 20 - 10
    pos[:, 2] *= 150
    count = 1
    intermediate_count = 0
    distance_threshold = 0.025
    while count < n_particles:
        new_pos = torch.rand(1, dimension, device=device)
        pos[:, 0] = pos[:, 0] * 20 - 10
        pos[:, 1] = pos[:, 1] * 20 - 10
        pos[:, 2] *= 150
        distance = torch.sum(pos[:, None, :] - new_pos[None, :, :] ** 2, dim=2)
        if torch.all(distance > distance_threshold**2):
            pos = torch.cat((pos, new_pos), 0)
            count += 1
        intermediate_count += 1
        if intermediate_count > 100:
            distance_threshold = distance_threshold * 0.99
            intermediate_count = 0

    return pos


# Load the data set (insert the path to the data set on your system)
# The data consists of a list of time points, each containing a :py:class:`torch_geometric.data.Data` object and
# a :py:class:`pandas.DataFrame` object containing information about the cells
root = "/groups/saalfeld/home/allierc/signaling/Celegans"
path = os.path.join(root, "position_genes.h5")
time_series, cell_info = load_celegans_gene_data(path, device="cpu")

# The info about the cells is stored in a pandas DataFrame
print(f"Number of cells: {len(cell_info)}")
print(cell_info.describe())

cell_name = cell_info.index[5]
cell_type = cell_info['type'].iloc[5]
print(f"First few gene names: {cell_info.index[:6]}")
print(f"Cell type for {cell_name}: {cell_type}")

# Since the location and gene data are acquired at different time intervals, the time series only contains
# the time in which both data are available (gene data is linearly interpolated)
print(f"Number of time points: {len(time_series)}")
print(f"Time points: {time_series.time}")

# The data objects contain the fields 'pos' and 'gene_cpm' and their derivatives ('velocity' and 'd_gene_cpm')
time_point = time_series[0]
print(f"Time point fields: {time_point.node_attrs()}")
print(f"Number of cells in time point 0: {time_point.num_nodes}")

print(f"Position shape: {time_point.pos.shape}")
print(f"Velocity shape: {time_point.velocity.shape}")
print(f"Gene expression shape: {time_point.gene_cpm.shape}")
print(f"Gene expression derivative shape: {time_point.d_gene_cpm.shape}")

num_background = 1000
background = torch.tensor(np.zeros((num_background)))
pos = place_cells(dimension=3, n_particles=num_background, device="cpu")
all_pos = torch.concatenate((pos, time_point.pos))

# num_background = 0
# all_pos = torch.tensor(time_point.pos)

print(all_pos.shape)

type_array = np.array(cell_info["type"])

unique_types, unique_indices = np.unique(type_array, return_inverse=True)

T1 = torch.concat((background, torch.tensor(unique_indices + 1)))
# T1 = torch.tensor(unique_indices)

color_map = plt.colormaps.get_cmap("tab20")
def get_color(n):
    return color_map(n % 20)

for it in trange(len(time_series)):
    n_particles = len(all_pos)
    # print(f'frame {it}, {n_particles} particles, {edge_index.shape[1]} edges')
    # vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=all_pos, device="cpu")

    # cells = [[] for i in range(n_particles)]
    # for (l, r), vertices in vor.ridge_dict.items():
    #     if l < n_particles:
    #         cells[l].append(vor.vertices[vertices])
    #     elif r < n_particles:
    #         cells[r].append(vor.vertices[vertices])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', box_aspect=[1, 1, 15])
    # for n, poly in enumerate(cells):
    #     polygon = Poly3DCollection(poly, alpha=0.5,
    #                                 facecolors=get_color(to_numpy(T1[n]).astype(int)),
    #                                 linewidths=0.1, edgecolors='black')
    #     ax.add_collection3d(polygon)

    ax.scatter(to_numpy(all_pos[num_background:, 0]), to_numpy(all_pos[num_background:, 1]), to_numpy(all_pos[num_background:, 2]), c="k", marker="o")
    ax.scatter(to_numpy(all_pos[:num_background, 0]), to_numpy(all_pos[:num_background, 1]), to_numpy(all_pos[:num_background, 2]), c="r", marker="o")
    # ax.auto_scale_xyz(to_numpy(all_pos[:, 0]), to_numpy(all_pos[:, 1]), to_numpy(all_pos[:, 2]))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 150)

    plt.tight_layout()
    plt.show()

    num = f"{it:06}"
    plt.savefig(f"graphs_data/celegens/Fig/Fig_0_{num}.tif", dpi=85.35)
    plt.close()
