from time import sleep

import numpy as np
import torch
from scipy.spatial import Delaunay
from tifffile import imread
from torch_geometric.utils import get_mesh_laplacian
from tqdm import trange

from ParticleGraph.utils import to_numpy


def init_particles(model_config, device):
    nparticles = model_config['nparticles']
    nparticle_types = model_config['nparticle_types']
    v_init = model_config['v_init']

    cycle_length = torch.clamp(torch.abs(
        torch.ones(nparticle_types, 1, device=device) * 400 + torch.randn(nparticle_types, 1, device=device) * 150),
                               min=100, max=700)

    if model_config['boundary'] == 'periodic':
        pos = torch.rand(nparticles, 2, device=device)
    else:
        pos = torch.randn(nparticles, 2, device=device) * 0.5
    dpos = v_init * torch.randn((nparticles, 2), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    type = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        type = torch.cat((type, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    type = type[:, None]
    if model_config['p'] == 'continuous':
        type = torch.tensor(np.arange(nparticles), device=device)
        type = type[:, None]
    features = torch.zeros((nparticles, 2), device=device)
    cycle_length_distrib = cycle_length[to_numpy(type[:, 0]).astype(int)]
    cycle_duration = torch.rand(nparticles, device=device)
    cycle_duration = cycle_duration[:, None]
    cycle_duration = cycle_duration * cycle_length_distrib
    particle_id = torch.arange(nparticles, device=device)
    particle_id = particle_id[:, None]

    return pos, dpos, type, features, cycle_duration, cycle_length_distrib, particle_id


def init_mesh(model_config, device):
    nnodes = model_config['nnodes']
    nparticles = model_config['nparticles']
    node_value_map = model_config['node_value_map']
    node_type_map = model_config['node_type_map']

    n_nodes_per_axis = int(np.sqrt(nnodes))
    xs = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    ys = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_nodes_per_axis ** 2, 1))
    y_mesh = torch.reshape(y_mesh, (n_nodes_per_axis ** 2, 1))
    mesh_size = 1 / n_nodes_per_axis
    pos_mesh = torch.zeros((nnodes, 2), device=device)
    pos_mesh[0:nnodes, 0:1] = x_mesh[0:nnodes]
    pos_mesh[0:nnodes, 1:2] = y_mesh[0:nnodes]

    mask_mesh = (x_mesh > torch.min(x_mesh)) & (x_mesh < torch.max(x_mesh)) & (y_mesh > torch.min(y_mesh)) & (
                y_mesh < torch.max(y_mesh))
    pos_mesh = pos_mesh + torch.randn(nnodes, 2, device=device) * mesh_size

    i0 = imread(f'graphs_data/{node_value_map}')
    values = i0[(to_numpy(pos_mesh[:, 0]) * 255).astype(int), (to_numpy(pos_mesh[:, 1]) * 255).astype(int)]

    if model_config['model'] == 'RD_Gray_Scott_Mesh':
        features_mesh = torch.zeros((nnodes, 2), device=device)
        features_mesh[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
        features_mesh[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
    elif model_config['model'] == 'RD_FitzHugh_Nagumo_Mesh':
        features_mesh = torch.zeros((nnodes, 2), device=device) + torch.rand((nnodes, 2), device=device) * 0.1
    elif model_config['model'] == 'RD_RPS_Mesh':
        features_mesh = torch.rand((nnodes, 3), device=device)
        s = torch.sum(features_mesh, axis=1)
        for k in range(3):
            features_mesh[:, k] = features_mesh[:, k] / s
    elif (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh') | (
            model_config['model'] == 'Maze'):
        features_mesh = torch.zeros((nnodes, 2), device=device)
        features_mesh[:, 0] = torch.tensor(values / 255 * 5000, device=device)
    if model_config['model'] == 'PDE_O':
        features_mesh = torch.zeros((nparticles, 5), device=device)
        features_mesh[0:nparticles, 0:1] = x_mesh[0:nparticles]
        features_mesh[0:nparticles, 1:2] = y_mesh[0:nparticles]
        features_mesh[0:nparticles, 2:3] = torch.randn(nparticles, 1, device=device) * 2 * np.pi  # theta
        features_mesh[0:nparticles, 3:4] = torch.ones(nparticles, 1, device=device) * np.pi / 200  # d_theta
        features_mesh[0:nparticles, 4:5] = features_mesh[0:nparticles, 3:4]  # d_theta0
        pos_mesh[:, 0] = features_mesh[:, 0] + (3 / 8) * mesh_size * torch.cos(features_mesh[:, 2])
        pos_mesh[:, 1] = features_mesh[:, 1] + (3 / 8) * mesh_size * torch.sin(features_mesh[:, 2])

    i0 = imread(f'graphs_data/{node_type_map}')
    values = i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(y_mesh[:, 0]) * 255).astype(int)]
    type_mesh = torch.tensor(values, device=device)
    type_mesh = type_mesh[:, None]

    node_id_mesh = torch.arange(nnodes, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((nnodes, 2), device=device)

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
