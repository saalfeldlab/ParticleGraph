from time import sleep

import numpy as np
import torch
from scipy.spatial import Delaunay
from tifffile import imread
from torch_geometric.utils import get_mesh_laplacian
from tqdm import trange

from ParticleGraph.utils import to_numpy


def init_particles(config, device, cycle_length=None):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types

    dpos_init = simulation_config.dpos_init

    if cycle_length == None:
        cycle_length = torch.clamp(torch.abs(torch.ones(n_particle_types, 1, device=device) * 400 + torch.randn(n_particle_types, 1, device=device) * 50), min=100, max=700)

    if simulation_config.boundary == 'periodic':
        pos = torch.rand(n_particles, 2, device=device)
    else:
        pos = torch.randn(n_particles, 2, device=device) * 0.5
    dpos = dpos_init * torch.randn((n_particles, 2), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    type = torch.zeros(int(n_particles / n_particle_types), device=device)
    for n in range(1, n_particle_types):
        type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
    if (simulation_config.params == 'continuous') | (config.simulation.non_discrete_level > 0):  # TODO: params is a list[list[float]]; this can never happen?
        type = torch.tensor(np.arange(n_particles), device=device)
    features = torch.cat((torch.rand((n_particles, 1), device=device) , 0.1 * torch.randn((n_particles, 1), device=device)), 1)
    cycle_length_distrib = cycle_length[to_numpy(type)].squeeze() * (torch.ones(n_particles, device=device) + 0.05 * torch.randn(n_particles, device=device))
    type = type[:, None]
    cycle_duration = torch.rand(n_particles, device=device)
    cycle_duration = cycle_duration * cycle_length[to_numpy(type)].squeeze()
    cycle_duration = cycle_duration[:, None]
    particle_id = torch.arange(n_particles, device=device)
    particle_id = particle_id[:, None]

    return pos, dpos, type, features, cycle_duration, particle_id, cycle_length, cycle_length_distrib


def init_mesh(config, device):
    simulation_config = config.simulation
    n_nodes = simulation_config.n_nodes
    n_particles = simulation_config.n_particles
    node_value_map = simulation_config.node_value_map
    node_type_map = simulation_config.node_type_map

    n_nodes_per_axis = int(np.sqrt(n_nodes))
    xs = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    ys = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_nodes_per_axis ** 2, 1))
    y_mesh = torch.reshape(y_mesh, (n_nodes_per_axis ** 2, 1))
    mesh_size = 1 / n_nodes_per_axis
    pos_mesh = torch.zeros((n_nodes, 2), device=device)
    pos_mesh[0:n_nodes, 0:1] = x_mesh[0:n_nodes]
    pos_mesh[0:n_nodes, 1:2] = y_mesh[0:n_nodes]

    i0 = imread(f'graphs_data/{node_value_map}')
    values = i0[(to_numpy(pos_mesh[:, 0]) * 255).astype(int), (to_numpy(pos_mesh[:, 1]) * 255).astype(int)]

    mask_mesh = (x_mesh > torch.min(x_mesh) + 0.02) & (x_mesh < torch.max(x_mesh) - 0.02) & (y_mesh > torch.min(y_mesh) + 0.02) & (y_mesh < torch.max(y_mesh) - 0.02)

    pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 8

    match config.graph_model.mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            features_mesh = torch.zeros((n_nodes, 2), device=device)
            features_mesh[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            features_mesh[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case 'RD_FitzHugh_Nagumo_Mesh':
            features_mesh = torch.zeros((n_nodes, 2), device=device) + torch.rand((n_nodes, 2), device=device) * 0.1
        case 'RD_RPS_Mesh':
            features_mesh = torch.rand((n_nodes, 3), device=device)
            s = torch.sum(features_mesh, dim=1)
            for k in range(3):
                features_mesh[:, k] = features_mesh[:, k] / s
        case 'DiffMesh' | 'WaveMesh' | 'Chemotaxism_Mesh':
            features_mesh = torch.zeros((n_nodes, 2), device=device)
            features_mesh[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case 'PDE_O_Mesh':
            features_mesh = torch.zeros((n_particles, 5), device=device)
            features_mesh[0:n_particles, 0:1] = x_mesh[0:n_particles]
            features_mesh[0:n_particles, 1:2] = y_mesh[0:n_particles]
            features_mesh[0:n_particles, 2:3] = torch.randn(n_particles, 1, device=device) * 2 * np.pi  # theta
            features_mesh[0:n_particles, 3:4] = torch.ones(n_particles, 1, device=device) * np.pi / 200  # d_theta
            features_mesh[0:n_particles, 4:5] = features_mesh[0:n_particles, 3:4]  # d_theta0
            pos_mesh[:, 0] = features_mesh[:, 0] + (3 / 8) * mesh_size * torch.cos(features_mesh[:, 2])
            pos_mesh[:, 1] = features_mesh[:, 1] + (3 / 8) * mesh_size * torch.sin(features_mesh[:, 2])

    i0 = imread(f'graphs_data/{node_type_map}')
    values = i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(y_mesh[:, 0]) * 255).astype(int)]
    type_mesh = torch.tensor(values, device=device)
    type_mesh = type_mesh[:, None]

    node_id_mesh = torch.arange(n_nodes, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((n_nodes, 2), device=device)

    x_mesh = torch.concatenate((node_id_mesh.clone().detach(), pos_mesh.clone().detach(), dpos_mesh.clone().detach(),
                                type_mesh.clone().detach(), features_mesh.clone().detach()), 1)

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    print('Removal of skinny faces ...')
    sleep(0.5)
    for k in trange(face.shape[0]):
        # compute edge distances
        x1 = pos[face[k, 0], :]
        x2 = pos[face[k, 1], :]
        x3 = pos[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
    edge_index_mesh, edge_weight_mesh = get_mesh_laplacian(pos=pos_3d, face=face, normalization="None")

    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}
    return pos_mesh, dpos_mesh, type_mesh, features_mesh, node_id_mesh, mesh_data
