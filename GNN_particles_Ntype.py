import glob
import json
import logging
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import numpy as np
import torch_geometric.data as data
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import umap
import yaml  # need to install pyyaml
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from sklearn import metrics
from sklearn.cluster import KMeans
from tifffile import imread
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils.convert import to_networkx
from tqdm import trange
from matplotlib import rc
import os
import scipy.spatial

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.data_loaders import *
from ParticleGraph.config_manager import create_config_manager, ConfigManager
from ParticleGraph.utils import to_numpy, cc
from ParticleGraph.generators.PDE_A import PDE_A
from ParticleGraph.generators.PDE_B import PDE_B
from ParticleGraph.generators.PDE_E import PDE_E
from ParticleGraph.generators.PDE_G import PDE_G
from ParticleGraph.generators.PDE_O import PDE_O
from ParticleGraph.generators.Laplacian_A import Laplacian_A
from ParticleGraph.generators.RD_FitzHugh_Nagumo import RD_FitzHugh_Nagumo
from ParticleGraph.generators.RD_Gray_Scott import RD_Gray_Scott
from ParticleGraph.generators.RD_RPS import RD_RPS

from ParticleGraph.models.Interaction_CElegans import Interaction_CElegans
from ParticleGraph.models.Interaction_Particles import Interaction_Particles
from ParticleGraph.models.Mesh_Laplacian import Mesh_Laplacian
from ParticleGraph.models.Mesh_RPS import Mesh_RPS
from ParticleGraph.models.PDE_embedding import PDE_embedding
from ParticleGraph.embedding_cluster import *


def func_pow(x, a, b):
    return a / (x ** b)


def func_lin(x, a, b):
    return a * x + b


def normalize99(Y, lower=1, upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01 + 1e-10)
    return x01, x99


def norm_velocity(xx, device):
    mvx = torch.mean(xx[:, 3])
    mvy = torch.mean(xx[:, 4])
    vx = torch.std(xx[:, 3])
    vy = torch.std(xx[:, 4])
    nvx = np.array(xx[:, 3].detach().cpu())
    vx01, vx99 = normalize99(nvx)
    nvy = np.array(xx[:, 4].detach().cpu())
    vy01, vy99 = normalize99(nvy)

    return torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)


def norm_acceleration(yy, device):
    max = torch.mean(yy[:, 0])
    may = torch.mean(yy[:, 1])
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = normalize99(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = normalize99(nay)

    return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)


def data_generate(model_config, bVisu=True, bStyle='color', bErase=False, step=5, alpha=0.2, ratio=1, scenario='none', device=[]):
    print('')
    print('Generating data ...')

    # create output folder, empty it if bErase=True, copy files into it
    dataset_name = model_config['dataset']
    folder = f'./graphs_data/graphs_particles_{dataset_name}/'
    if bErase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-8:] != 'tmp_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_particles_{dataset_name}/tmp_data/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_particles_{dataset_name}/tmp_data/*')
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))


    # load model parameters and create local varibales
    model_config['nparticles'] = model_config['nparticles'] * ratio
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    noise_level = model_config['noise_level']
    v_init = model_config['v_init']
    bMesh = 'Mesh' in model_config['model']
    bDivision = 'division_cycle' in model_config
    delta_t = model_config['delta_t']
    aggr_type = model_config['aggr_type']
    nnode_types = model_config['nnode_types']
    nnodes = model_config['nnodes']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    # create boundary functions for position and velocity respectively
    if model_config['boundary'] == 'no':  # change this for usual BC
        def bc_pos(X):
            return X

        def bc_diff(D):
            return D
    else:
        def bc_pos(X):
            return torch.remainder(X, 1.0)

        def bc_diff(D):
            return torch.remainder(D - .5, 1.0) - .5

    cycle_length = torch.clamp(torch.abs(torch.ones(nparticle_types, 1, device=device) * 400 + torch.randn(nparticle_types, 1, device=device) * 150),min=100, max=700)
    if bDivision:
        for n in range(model_config['nparticle_types']):
            print(f'cell cycle duration: {to_numpy(cycle_length[n])}')
        torch.save(torch.squeeze(cycle_length), f'graphs_data/graphs_particles_{dataset_name}/cycle_length.pt')

    rr = torch.tensor(np.linspace(0, radius * 2, 1000), device=device)
    if bMesh | (model_config['model'] == 'PDE_O') | (model_config['model'] == 'Maze'):
        node_value_map = model_config['node_value_map']
        node_type_map = model_config['node_type_map']


    if model_config['model'] == 'PDE_A':
        print(f'Generate PDE_A')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        if nparticle_types == 1:
            model = PDE_A(aggr_type=aggr_type, p=p, sigma=model_config['sigma'], bc_diff=bc_diff)
        else:
            model = PDE_A(aggr_type=aggr_type, p=torch.squeeze(p), sigma=model_config['sigma'], bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
        torch.save(torch.squeeze(p), f'graphs_data/graphs_particles_{dataset_name}/p.pt')
    if model_config['model'] == 'PDE_B':
        print(f'Generate PDE_B')
        p = torch.rand(nparticle_types, 3, device=device) * 100  # comprised between 10 and 50
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        if nparticle_types == 1:
            model = PDE_A(aggr_type=aggr_type, p=p, bc_diff=bc_diff)
        else:
            model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
        torch.save(torch.squeeze(p), f'graphs_data/graphs_particles_{dataset_name}/p.pt')
    if model_config['model'] == 'PDE_G':
        if model_config['p'][0] == -1:
            p = np.linspace(0.5, 5, nparticle_types)
            p = torch.tensor(p, device=device)
        if len(model_config['p']) > 1:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p), clamp=model_config['clamp'],
                      pred_limit=model_config['pred_limit'], bc_diff=bc_diff)
        psi_output = []
        for n in range(len(p)):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
        torch.save(torch.squeeze(p), f'graphs_data/graphs_particles_{dataset_name}/p.pt')
    if model_config['model'] == 'PDE_E':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
                print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
                torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = PDE_E(aggr_type=aggr_type, p=torch.squeeze(p),
                      clamp=model_config['clamp'], pred_limit=model_config['pred_limit'],
                      prediction=model_config['prediction'], bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            for m in range(nparticle_types):
                psi_output.append(model.psi(rr, torch.squeeze(p[n]), torch.squeeze(p[m])))
        torch.save(torch.squeeze(p), f'graphs_data/graphs_particles_{dataset_name}/p.pt')
    if bMesh:
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p),
                      clamp=model_config['clamp'], pred_limit=model_config['pred_limit'], bc_diff=bc_diff)
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])

        if (model_config['model'] == 'RD_Gray_Scott_Mesh'):
            model_mesh = RD_Gray_Scott(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                       bc_diff=bc_diff)
        elif (model_config['model'] == 'RD_FitzHugh_Nagumo_Mesh'):
            model_mesh = RD_FitzHugh_Nagumo(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                            bc_diff=bc_diff)
        elif (model_config['model'] == 'RD_RPS_Mesh'):
            model_mesh = RD_RPS(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'], bc_diff=bc_diff)
        elif (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
            model_mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                     bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if model_config['model'] == 'PDE_O':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = PDE_O(aggr_type=aggr_type, p=torch.squeeze(p), bc_diff=bc_diff, beta=model_config['beta'])
    if model_config['model'] == 'Maze':
        print(f'Generate PDE_B')
        p = torch.rand(nparticle_types, 3, device=device) * 100  # comprised between 10 and 50
        for n in range(nparticle_types):
            p[n] = torch.tensor(model_config['p'][n])
        model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
        torch.save(torch.squeeze(p), f'graphs_data/graphs_particles_{dataset_name}/p.pt')
        c = torch.ones(nnode_types, 1, device=device) + torch.rand(nnode_types, 1, device=device)
        for n in range(nnode_types):
            c[n] = torch.tensor(model_config['c'][n])

        model_mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                 bc_diff=bc_diff)

    torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    for run in range(model_config['nrun']):

        x_list = []
        y_list = []
        x_mesh_list = []
        y_mesh_list = []

        # initialize particle and graph states
        if (model_config['boundary'] == 'periodic'):
            X1 = torch.rand(nparticles, 2, device=device)
        else:
            X1 = torch.randn(nparticles, 2, device=device) * 0.5
        V1 = v_init * torch.randn((nparticles, 2), device=device)
        V1 = torch.clamp(V1, min=-torch.std(V1), max=+torch.std(V1))
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = T1[:, None]
        if model_config['p'] == 'continuous':
            T1 = torch.tensor(np.arange(nparticles), device=device)
            T1 = T1[:, None]
        H1 = torch.zeros((nparticles, 2), device=device)
        cycle_length_distrib = cycle_length[to_numpy(T1[:, 0]).astype(int)]
        A1 = torch.rand(nparticles, device=device)
        A1 = A1[:, None]
        A1 = A1 * cycle_length_distrib
        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:, None]

        # create different initial conditions
        if scenario == 'scenario A':
            X1[:, 0] = X1[:, 0] / nparticle_types
            for n in range(nparticle_types):
                X1[index_particles[n], 0] = X1[index_particles[n], 0] + n / nparticle_types

        # scenario C
        # i0 = imread('graphs_data/pattern_1.tif')
        # pos = np.argwhere(i0 == 255)
        # l = np.arange(pos.shape[0])
        # l = np.random.permutation(l)
        # X1[index_particles[0],:] = torch.tensor(pos[l[index_particles[0]],:]/255,dtype=torch.float32,device=device)
        # pos = np.argwhere(i0 == 0)
        # l = np.arange(pos.shape[0])
        # l = np.random.permutation(l)
        # X1[index_particles[1],:] = torch.tensor(pos[l[index_particles[0]],:]/255,dtype=torch.float32,device=device)

        # scenario D
        # i0 = imread('graphs_data/pattern_2.tif')
        # pos = np.argwhere(i0 == 255)
        # l = np.arange(pos.shape[0])
        # l = np.random.permutation(l)
        # X1[index_particles[0],:] = torch.tensor(pos[l[0:1000],:]/255,dtype=torch.float32,device=device)
        # pos = np.argwhere(i0 == 128)
        # l = np.arange(pos.shape[0])
        # l = np.random.permutation(l)
        # X1[index_particles[1],:] = torch.tensor(pos[l[0:1000],:]/255,dtype=torch.float32,device=device)
        # pos = np.argwhere(i0 == 0)
        # l = np.arange(pos.shape[0])
        # l = np.random.permutation(l)
        # X1[index_particles[2],:] = torch.tensor(pos[l[0:1000],:]/255,dtype=torch.float32,device=device)

        if (bMesh) | (model_config['model'] == 'PDE_O') | (model_config['model'] == 'Maze'):
            x_width = int(np.sqrt(nnodes))
            xs = torch.linspace(1 / x_width / 2, 1 - 1 / x_width / 2, steps=x_width)
            ys = torch.linspace(1 / x_width / 2, 1 - 1 / x_width / 2, steps=x_width)
            x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
            x_mesh = torch.reshape(x_mesh, (x_width ** 2, 1))
            y_mesh = torch.reshape(y_mesh, (x_width ** 2, 1))
            x_width = 1 / x_width / 8
            X1_mesh = torch.zeros((nnodes, 2), device=device)
            X1_mesh[0:nnodes, 0:1] = x_mesh[0:nnodes]
            X1_mesh[0:nnodes, 1:2] = y_mesh[0:nnodes]

            mask_mesh = (x_mesh>torch.min(x_mesh)) & (x_mesh<torch.max(x_mesh)) & (y_mesh>torch.min(y_mesh)) & (y_mesh<torch.max(y_mesh))
            X1_mesh = X1_mesh + torch.randn(nnodes, 2, device=device) * x_width
            
            i0 = imread(f'graphs_data/{node_value_map}')
            values = i0[(to_numpy(X1_mesh[:, 0]) * 255).astype(int), (to_numpy(X1_mesh[:, 1]) * 255).astype(int)]

            if (model_config['model'] == 'RD_Gray_Scott_Mesh'):
                H1_mesh = torch.zeros((nnodes, 2), device=device)
                H1_mesh[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
                H1_mesh[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
            elif (model_config['model'] == 'RD_FitzHugh_Nagumo_Mesh'):
                H1_mesh = torch.zeros((nnodes, 2), device=device) + torch.rand((nnodes, 2), device=device) * 0.1
            elif (model_config['model'] == 'RD_RPS_Mesh'):
                H1_mesh = torch.rand((nnodes, 3), device=device)
                s = torch.sum(H1_mesh, axis=1)
                for k in range(3):
                    H1_mesh[:, k] = H1_mesh[:, k] / s
            elif (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh') | (model_config['model'] == 'Maze'):
                H1_mesh = torch.rand((nnodes, 1), device=device)
                H1_mesh[:, 0] = torch.tensor(values / 255 * 5000, device=device)
            if model_config['model'] == 'PDE_O':
                H1_mesh = torch.zeros((nparticles, 5), device=device)
                H1_mesh[0:nparticles, 0:1] = x[0:nparticles]
                H1_mesh[0:nparticles, 1:2] = y[0:nparticles]
                H1_mesh[0:nparticles, 2:3] = torch.randn(nparticles, 1, device=device) * 2 * np.pi  # theta
                H1_mesh[0:nparticles, 3:4] = torch.ones(nparticles, 1, device=device) * np.pi / 200  # d_theta
                H1_mesh[0:nparticles, 4:5] = H1_mesh[0:nparticles, 3:4]  # d_theta0
                X1_mesh[:, 0] = H1_mesh[:, 0] + 3 * x_width * torch.cos(H1_mesh[:, 2])
                X1_mesh[:, 1] = H1_mesh[:, 1] + 3 * x_width * torch.sin(H1_mesh[:, 2])

            i0 = imread(f'graphs_data/{node_type_map}')
            values = i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(y_mesh[:, 0]) * 255).astype(int)]
            T1_mesh = torch.tensor(values, device=device)
            T1_mesh = T1_mesh[:, None]

            N1_mesh = torch.arange(nnodes, device=device)
            N1_mesh = N1_mesh[:, None]
            V1_mesh = torch.zeros((nnodes, 2), device=device)
            #
            # plt.ion()
            # plt.scatter(to_numpy(X1_mesh[:, 0]), to_numpy(X1_mesh[:, 1]), s=10, c=to_numpy(T1_mesh[:, 0]))

            x_mesh = torch.concatenate((N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(), T1_mesh.clone().detach(), H1_mesh.clone().detach()), 1)

            # dataset = data.Data(x=x, pos=x[:, 1:3])
            # transform_0 = T.Compose([T.Delaunay()])
            # dataset_face = transform_0(dataset).face
            # mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
            # edge_index_mesh, edge_weight_mesh = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face, normalization="None")
            # plt.ion()
            # pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
            # dataset = data.Data(x=x, edge_index=edge_index_mesh)
            # vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            # nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=10, with_labels=False, alpha=1)

            pos = to_numpy(x_mesh[:, 1:3])
            tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
            face = torch.from_numpy(tri.simplices)
            face_longest_edge = np.zeros((face.shape[0], 1))

            print('Removal of skinny faces ...')
            time.sleep(0.5)
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
            face = face.to(device,torch.long)

            mesh_pos = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
            edge_index_mesh, edge_weight_mesh = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=face, normalization="None")

            torch.save({'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh, 'mask_mesh': mask_mesh,'mesh_pos': mesh_pos }, f'graphs_data/graphs_particles_{dataset_name}/mesh_data_{run}.pt')

            if model_config['model'] != 'Maze':
                X1 = X1_mesh
                H1 = H1_mesh


            # pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
            # dataset = data.Data(x=x, edge_index=edge_index_mesh)
            # vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            # nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=10, with_labels=False, alpha=1, edge_color='r')

            # deg = pyg_utils.degree(edge_index_mesh[0], mesh_pos.shape[0])
            # plt.ion()
            # plt.hist(to_numpy(deg),100)
            # plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=10, c=to_numpy(deg))

        time.sleep(0.5)
        for it in trange(model_config['start_frame'], nframes + 1):

            # calculate graph states at itme t and t+1 
            if (it > 0) & bDivision & (nparticles < 20000):
                cycle_test = (torch.ones(nparticles, device=device) + 0.05 * torch.randn(nparticles, device=device))
                pos = torch.argwhere(A1 > cycle_test[:, None] * cycle_length_distrib)
                # cell division
                if len(pos) > 1:
                    n_add_nodes = len(pos)
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    nparticles = nparticles + n_add_nodes
                    N1 = torch.arange(nparticles, device=device)
                    N1 = N1[:, None]

                    separation = 1E-3 * torch.randn((n_add_nodes, 2), device=device)
                    X1 = torch.cat((X1, X1[pos, :] + separation), axis=0)
                    X1[pos, :] = X1[pos, :] - separation

                    phi = torch.randn(n_add_nodes, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                    cos_phi = torch.cos(phi)
                    sin_phi = torch.sin(phi)
                    new_x = cos_phi * V1[pos, 0] + sin_phi * V1[pos, 1]
                    new_y = -sin_phi * V1[pos, 0] + cos_phi * V1[pos, 1]
                    V1[pos, 0] = new_x
                    V1[pos, 1] = new_y
                    V1 = torch.cat((V1, -V1[pos, :]), axis=0)

                    T1 = torch.cat((T1, T1[pos, :]), axis=0)
                    H1 = torch.cat((H1, H1[pos, :]), axis=0)
                    A1[pos, :] = 0
                    A1 = torch.cat((A1, A1[pos, :]), axis=0)

                    index_particles = []
                    for n in range(nparticles):
                        pos = torch.argwhere(T1 == n)
                        pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                        index_particles.append(pos)

            # append x_list
            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach()), 1)

            # create mesh dataset
            if bMesh | (model_config['model'] == 'Maze'):
                x_mesh = torch.concatenate((N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
                                           T1_mesh.clone().detach(),
                                           H1_mesh.clone().detach()), 1)
                dataset_mesh = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
            # compute connectivity rule
            distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            # model prediction
            with torch.no_grad():
                y = model(dataset)
            # append y_list
            if (it >= 0):
                x_list.append(x.clone().detach())
                y_list.append(y.clone().detach())

            # Euler integration update
            if not (bMesh):

                if model_config['model'] == 'PDE_O':
                    H1[:, 2] = H1[:, 2] + y.squeeze() * delta_t
                    # pos= torch.argwhere(H1[:, 0] > 0.9)
                    # H1[pos, 2] = 0
                    X1[:, 0] = H1[:, 0] + 3 * x_width * torch.cos(H1[:, 2])
                    X1[:, 1] = H1[:, 1] + 3 * x_width * torch.sin(H1[:, 2])
                    X1 = bc_pos(X1)
                else:
                    if model_config['prediction'] == '2nd_derivative':
                        V1 += y[:, 0:2] * delta_t
                    else:
                        V1 = y[:, 0:2]
                    X1 = bc_pos(X1 + V1 * delta_t)

                A1 = A1 + 1
            # append y_mesh_list
            # Euler integration update for mesh

            if model_config['model'] == 'DiffMesh':
                mask = to_numpy(
                    torch.argwhere((X1[:, 0] > 0.1) & (X1[:, 0] < 0.9) & (X1[:, 1] > 0.1) & (X1[:, 1] < 0.9))).astype(
                    int)
                mask = mask[:, 0:1]
                with torch.no_grad():
                    pred = model_mesh(dataset_mesh)
                    H1[mask, 1:2] = pred[mask]
                H1[mask, 0:1] += H1[mask, 1:2] * delta_t
                new_pred = torch.zeros_like(pred)
                new_pred[mask] = pred[mask]
                y_mesh_list.append(new_pred)

            if model_config['model']=='Maze':
                x_mesh_list.append(x_mesh.clone().detach())
                with torch.no_grad():
                    pred = model_mesh(dataset_mesh)
                    H1_mesh[mask_mesh.squeeze(), :] += pred[mask_mesh.squeeze(), :] * delta_t


                    distance = torch.sum(bc_diff(x[:, None, 1:3] - x_mesh[None, :, 1:3]) ** 2, axis=2)
                    distance = distance < 0.001
                    distance = torch.sum(distance, axis=0)
                    H1_mesh = torch.abs(H1_mesh*1.025 - 10*distance[:,None])
                    H1_mesh = torch.clamp(H1_mesh, min=0, max=5000)

                    # fig = plt.figure(figsize=(12, 12))
                    # plt.ion()
                    # H1_IM = torch.reshape(H1_mesh, (100, 100))
                    # plt.ion()
                    # plt.imshow(H1_IM.detach().cpu().numpy()*5)



                y_mesh_list.append(pred)


            if model_config['model'] == 'WaveMesh':
                with torch.no_grad():
                    pred = model_mesh(dataset_mesh)
                    H1_mesh[:, 1:2] += pred[:] * delta_t
                H1_mesh[:, 0:1] += H1_mesh[:, 1:2] * delta_t
                H1 = H1_mesh
                y_mesh_list.append(pred)
            if (model_config['model'] == 'RD_Gray_Scott_Mesh') | (
                    model_config['model'] == 'RD_FitzHugh_Nagumo_Mesh') | (model_config['model'] == 'RD_RPS_Mesh'):
                # mask = to_numpy(torch.argwhere(
                #     (X1[:, 0] > 0.02) & (X1[:, 0] < 0.98) & (X1[:, 1] > 0.02) & (X1[:, 1] < 0.98))).astype(int)
                # mask = mask[:, 0:1]
                with torch.no_grad():
                    pred = model_mesh(dataset_mesh)
                    H1_mesh[mask_mesh.squeeze(),:] += pred[mask_mesh.squeeze(),:] * delta_t
                    H1 = H1_mesh
                    # fig = plt.figure(figsize=(12, 12))
                    # plt.ion()
                    # H1_IM = torch.reshape(pred, (100, 100, 3))
                    # plt.ion()
                    # plt.imshow(H1_IM.detach().cpu().numpy()*5)


                y_mesh_list.append(pred)

            # output plots
            if bVisu & (run == 0) & (it % step == 0) & (it >= 0):

                if 'graph' in bStyle:
                    fig = plt.figure(figsize=(10, 10))
                    # plt.ion()

                    distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                    adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                    edge_index2 = adj_t2.nonzero().t().contiguous()
                    dataset2 = data.Data(x=x, edge_index=edge_index2)
                    pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                    vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=alpha)

                    if model_config['model'] == 'PDE_G':
                        for n in range(nparticle_types):
                            g = p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 7.5
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=40, color=cmap.color(n))
                    elif bMesh:
                        pts = x[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x[tri.simplices, 6], axis=1) / 3.0
                        if model_config['model'] == 'WaveMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                          vmax=2500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=2500)
                    elif model_config['model'] == 'PDE_E':
                        for n in range(nparticle_types):
                            g = 40
                            if model_config['p'][n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                    else:
                        for n in range(nparticle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n),
                                        alpha=0.5)
                    if bMesh | (model_config['boundary'] == 'periodic'):
                        # plt.text(0, 1.08, f'frame: {it}')
                        # plt.text(0, 1.03, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    else:
                        # plt.text(-1.25, 1.5, f'frame: {it}')
                        # plt.text(-1.25, 1.4, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                        plt.xlim([-0.5, 0.5])
                        plt.ylim([-0.5, 0.5])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_particles_{dataset_name}/tmp_data/Fig_g_color_{it}.tif", dpi=300)
                    plt.close()

                if 'color' in bStyle:

                    # plt.ion()
                    if model_config['model'] == 'PDE_O':
                        fig = plt.figure(figsize=(12, 12))
                        plt.style.use('dark_background')
                        plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=500,
                                    c=np.sin(to_numpy(H1[:, 2])), vmin=-1, vmax=1, cmap='viridis')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_particles_{dataset_name}/tmp_data/Lut_Fig_{it}.jpg", dpi=75)
                        plt.close()
                        fig = plt.figure(figsize=(12, 12))
                        plt.style.use('default')
                        plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=1, c='b')
                        plt.scatter(X1[:, 0].detach().cpu().numpy(), X1[:, 1].detach().cpu().numpy(), s=10, c='r',
                                    alpha=0.75)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_particles_{dataset_name}/tmp_data/Rot_Fig{it}.jpg", dpi=75)
                        plt.close()

                    elif model_config['model'] == 'Maze':

                        fig = plt.figure(figsize=(12, 12))
                        pts = x_mesh[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x_mesh[tri.simplices, 6], axis=1) / 3.0
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=5000)
                        for n in range(nparticle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=10, color='w')
                            
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_particles_{dataset_name}/tmp_data/Mesh_{it}.jpg", dpi=100)
                        plt.close()

                    else:

                        fig = plt.figure(figsize=(12, 12))
                        if bMesh:
                            pts = x[:, 1:3].detach().cpu().numpy()
                            tri = Delaunay(pts)
                            colors = torch.sum(x[tri.simplices, 6], axis=1) / 3.0
                            if model_config['model'] == 'DiffMesh':
                                plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                              facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                            if model_config['model'] == 'WaveMesh':
                                plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                              facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                            if (model_config['model'] == 'RD_Gray_Scott_Mesh'):
                                fig = plt.figure(figsize=(12, 6))
                                ax = fig.add_subplot(1, 2, 1)
                                colors = torch.sum(x[tri.simplices, 6], axis=1) / 3.0
                                plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                              facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                                plt.xticks([])
                                plt.yticks([])
                                plt.axis('off')
                                ax = fig.add_subplot(1, 2, 2)
                                colors = torch.sum(x[tri.simplices, 7], axis=1) / 3.0
                                plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                              facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                                plt.xticks([])
                                plt.yticks([])
                                plt.axis('off')
                            if (model_config['model'] == 'RD_RPS_Mesh'):
                                fig = plt.figure(figsize=(12, 12))
                                H1_IM = torch.reshape(H1, (100, 100, 3))
                                plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                                plt.xticks([])
                                plt.yticks([])
                                plt.axis('off')
                        else:
                            for n in range(nparticle_types):
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n))

                        if bMesh | (model_config['boundary'] == 'periodic'):
                            g = 1
                        else:
                            plt.xlim([-4, 4])
                            plt.ylim([-4, 4])

                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_particles_{dataset_name}/tmp_data/Fig_{it}.jpg", dpi=100)
                        plt.close()





                if 'bw' in bStyle:
                    fig = plt.figure(figsize=(12, 12))
                    if bMesh:
                        pts = x[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x[tri.simplices, 6], axis=1) / 3.0
                        if model_config['model'] == 'WaveMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors='w', edgecolors='k', vmin=-2500, vmax=2500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=2500)
                    else:
                        for n in range(nparticle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=25, color='k')
                    if bMesh | (model_config['boundary'] == 'periodic'):
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    else:
                        plt.xlim([-1.3, 1.3])
                        plt.ylim([-1.3, 1.3])
                    plt.xticks([])
                    plt.yticks([])

                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_particles_{dataset_name}/tmp_data/Fig_bw_{it}.tif", dpi=300)
                    plt.close()

        torch.save(x_list, f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
        if model_config['model'] != 'Maze':
            torch.save(x_mesh_list, f'graphs_data/graphs_particles_{dataset_name}/x_mesh_list_{run}.pt')
        torch.save(y_mesh_list, f'graphs_data/graphs_particles_{dataset_name}/y_mesh_list_{run}.pt')

    model_config['nparticles'] = int(model_config['nparticles'] / ratio)


def data_train(model_config):
    print('')

    model = []
    Nepochs = model_config['Nepochs']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
    batch_size = model_config['batch_size']
    bMesh = 'Mesh' in model_config['model']
    bReplace = 'replace' in model_config['sparsity']
    aggr_type = model_config['aggr_type']
    bVisuEmbedding = False

    embedding_cluster = EmbeddingCluster(model_config)

    if model_config['boundary'] == 'no':  # change this for usual BC
        def bc_pos(X):
            return X

        def bc_diff(D):
            return D
    else:
        def bc_pos(X):
            return torch.remainder(X, 1.0)

        def bc_diff(D):
            return torch.remainder(D - .5, 1.0) - .5

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(model_config)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs - 1}')
    logger.info(f'Graph files N: {NGraphs - 1}')

    x_list = []
    y_list = []
    print('Load data ...')
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))
    x = torch.stack(x_list)
    x = torch.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    y = torch.stack(y_list)
    y = torch.reshape(y, (y.shape[0] * y.shape[1] * y.shape[2], y.shape[3]))
    vnorm = norm_velocity(x, device)
    ynorm = norm_acceleration(y, device)
    vnorm = vnorm[4]
    ynorm = ynorm[4]
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    if bMesh:
        y_mesh_list = []
        for run in trange(NGraphs):
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
            y_mesh_list.append(torch.stack(h))
        h = torch.stack(y_mesh_list)
        h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
        hnorm = torch.std(h)
        time.sleep(0.5)
        print(f'hnorm: {to_numpy(hnorm)}')
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        logger.info(f'hnorm : {to_numpy(hnorm)}')

        batch_size = 1

        mesh_data = torch.load(f'graphs_data/graphs_particles_{dataset_name}/mesh_data_1.pt',map_location=device)

        mask_mesh = mesh_data['mask_mesh']
        # mesh_pos = mesh_data['mesh_pos']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']
        # face = mesh_data['face']

    print('')

    if model_config['model'] == 'PDE_G':
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if model_config['model'] == 'PDE_E':
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'DiffMesh'):
        model = Mesh_Laplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'WaveMesh'):
        model = Mesh_Laplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'RD_RPS_Mesh'):
        model = Mesh_RPS(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

    # net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_6.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lra = 1E-3
    lr = 1E-3
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    it = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if it == 0:
            optimizer = torch.optim.Adam([model.a], lr=lra)
        else:
            optimizer.add_param_group({'params': parameter, 'lr': lr})
        it += 1
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    logger.info(f'Learning rates: {lr}, {lra}')

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {Nepochs}')
    logger.info(f'batch_size: {batch_size}')

    x = x_list[1][0].clone().detach()
    T1 = x[:, 5:6].clone().detach()
    index_particles = []
    for n in range(model_config['nparticle_types']):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    # plt.ion()
    # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=10, c=to_numpy(mask_mesh)*2)
    # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=10, c=to_numpy(T1) )

    # optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    data_augmentation_loop = 200
    print("Start training ...")
    print(f'{nframes * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{nframes * data_augmentation_loop // batch_size} iterations per epoch')

    model.train()

    list_loss = []
    time.sleep(0.5)

    for epoch in range(Nepochs + 1):

        if epoch == 0:
            min_radius = 0.002
        elif epoch == 1:
            min_radius = model_config['min_radius']
            logger.info(f'min_radius: {min_radius}')
        elif (epoch == 2) & (batch_size == 1):
            batch_size = 8
            print(f'batch_size: {batch_size}')
            logger.info(f'batch_size: {batch_size}')
        elif epoch == 3 * Nepochs // 4 + 2:
            lra = 1E-3
            lr = 5E-4
            it = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if it == 0:
                    optimizer = torch.optim.Adam([model.a], lr=lra)
                else:
                    optimizer.add_param_group({'params': parameter, 'lr': lr})
                it += 1
            print(f'Learning rates: {lr}, {lra}')
            logger.info(f'Learning rates: {lr}, {lra}')

        error_weight = torch.ones((batch_size * nparticles, 1),device=device, requires_grad=False)

        total_loss = 0

        Niter = nframes * data_augmentation_loop // batch_size
        if (bMesh) & (batch_size == 1):
            Niter = Niter // 4

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []

            for batch in range(batch_size):

                k = np.random.randint(nframes - 1)
                x = x_list[run][k].clone().detach()

                if bMesh:
                    dataset = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
                    dataset_batch.append(dataset)
                    y = y_mesh_list[run][k].clone().detach() / hnorm
                    if batch == 0:
                        y_batch = y
                    else:
                        y_batch = torch.cat((y_batch, y), axis=0)
                else:
                    distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                    adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    t = torch.Tensor([radius ** 2])
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)
                    dataset_batch.append(dataset)
                    y = y_list[run][k].clone().detach()
                    if model_config['prediction'] == '2nd_derivative':
                        y = y / ynorm
                    else:
                        y = y / vnorm
                    if data_augmentation:
                        new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                        new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                        y[:, 0] = new_x
                        y[:, 1] = new_y
                    if batch == 0:
                        y_batch = y
                    else:
                        y_batch = torch.cat((y_batch, y), axis=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                if bMesh:
                    pred = model(batch, data_id=run - 1)
                else:
                    pred = model(batch, data_id=run - 1, training=True, vnorm=vnorm, phi=phi)

            if model_config['model'] == 'RD_RPS_Mesh':
                loss = ((pred - y_batch) * error_weight * mask_mesh).norm(2)
            else:
                loss = ((pred - y_batch) * error_weight).norm(2)

            if model_config['loss_weight'] & (epoch > 1 * Nepochs // 4):
                with torch.no_grad():
                    error_weight = torch.abs(pred - y_batch).reshape((batch_size, nparticles, 1))
                    error_weight = torch.mean(error_weight, axis=0)
                    error_weight = 1 + error_weight / torch.std(error_weight)
                    error_weight = error_weight.repeat(batch_size, 1, 1).reshape((batch_size * nparticles, 1)).clone().detach()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if bVisuEmbedding:
                fig = plt.figure(figsize=(8, 8))
                embedding = []
                for n in range(model.a.shape[0]):
                    embedding.append(model.a[n])
                embedding = to_numpy(torch.stack(embedding))
                embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
                embedding_ = embedding
                embedding_particle = []
                for m in range(model.a.shape[0]):
                    for n in range(nparticle_types):
                        embedding_particle.append(embedding[index_particles[n] + m * nparticles, :])
                if (embedding.shape[1] > 2):
                    ax = fig.add_subplot(2, 4, 2, projection='3d')
                    for n in range(nparticle_types):
                        ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1],
                                   embedding_particle[n][:, 2],
                                   color=cmap.color(n), s=1)
                else:
                    if (embedding.shape[1] > 1):
                        for m in range(model.a.shape[0]):
                            for n in range(nparticle_types):
                                plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                                            embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=3)
                        plt.xlabel('Embedding 0', fontsize=12)
                        plt.ylabel('Embedding 1', fontsize=12)
                    else:
                        for n in range(nparticle_types):
                            plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5, color=cmap.color(n))
                plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{N}.tif")


        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / nparticles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / nparticles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        list_loss.append(total_loss / (N + 1) / nparticles / batch_size)

        fig = plt.figure(figsize=(22, 4))
        plt.ion()
        ax = fig.add_subplot(1, 6, 1)
        plt.plot(list_loss, color='k')
        plt.ylim([0, 0.010])
        plt.xlim([0, Nepochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding = []
        for n in range(model.a.shape[0]):
            embedding.append(model.a[n])
        embedding = to_numpy(torch.stack(embedding))
        embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
        embedding_ = embedding
        embedding_particle = []
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                embedding_particle.append(embedding[index_particles[n] + m * nparticles, :])
        if (embedding.shape[1] > 2):
            ax = fig.add_subplot(2, 4, 2, projection='3d')
            for n in range(nparticle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],
                           color=cmap.color(n), s=1)
        else:
            if (embedding.shape[1] > 1):
                for m in range(model.a.shape[0]):
                    for n in range(nparticle_types):
                        plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                                    embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=3)
                plt.xlabel('Embedding 0', fontsize=12)
                plt.ylabel('Embedding 1', fontsize=12)
            else:
                for n in range(nparticle_types):
                    plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5, color=cmap.color(n))

        ax = fig.add_subplot(1, 6, 3)
        if model_config['ninteractions'] < 100:  # cluster embedding
            if model_config['model'] == 'PDE_E':
                acc_list = []
                for m in range(model.a.shape[0]):
                    for k in range(nparticle_types):
                        for n in index_particles[k]:
                            rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                            embedding0 = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                            embedding1 = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                            in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                                     rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                                     0 * rr[:, None],
                                                     0 * rr[:, None], 0 * rr[:, None], embedding0, embedding1), dim=1)
                            acc = model.lin_edge(in_features.float())
                            acc = acc[:, 0]
                            acc_list.append(acc)
                            if n % 5 == 0:
                                plt.plot(to_numpy(rr),
                                         to_numpy(acc) * to_numpy(ynorm) / model_config['delta_t'],
                                         linewidth=1,
                                         color=cmap.color(k), alpha=0.25)
                acc_list = torch.stack(acc_list)
                plt.xlim([0, 0.05])
                plt.xlabel('Distance [a.u]', fontsize=12)
                plt.ylabel('MLP [a.u]', fontsize=12)
                coeff_norm = to_numpy(acc_list)
                trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                                  n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
                proj_interaction = trans.transform(coeff_norm)
            elif model_config['model'] == 'PDE_G':
                acc_list = []
                for n in range(nparticles):
                    rr = torch.tensor(np.linspace(0, radius * 1.3, 1000)).to(device)
                    embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                    in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                    acc = model.lin_edge(in_features.float())
                    acc = acc[:, 0]
                    acc_list.append(acc)

                    plt.plot(rr.detach().cpu().numpy(),
                             acc.detach().cpu().numpy() * ynorm.detach().cpu().numpy() / model_config['delta_t'],
                             color=cmap.color(x[n, 5].detach().cpu().numpy()), linewidth=1, alpha=0.25)
                acc_list = torch.stack(acc_list)
                plt.yscale('log')
                plt.xscale('log')
                plt.xlim([1E-3, 0.2])
                plt.xlabel('Distance [a.u]', fontsize=12)
                plt.ylabel('MLP [a.u]', fontsize=12)
                coeff_norm = to_numpy(acc_list)
                trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                                  n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
                proj_interaction = trans.transform(coeff_norm)
            elif (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
                acc_list = []
                for n in range(nparticles):
                    rr = torch.tensor(np.linspace(0, radius, 200)).to(device)
                    embedding = model.a[0, n, :] * torch.ones((200, model_config['embedding']), device=device)
                    if ((model_config['model'] == 'PDE_A')):
                        in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                                 rr[:, None] / model_config['radius'], embedding), dim=1)
                    else:
                        in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                                 rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                    acc = model.lin_edge(in_features.float())
                    acc = acc[:, 0]
                    acc_list.append(acc)
                    if n % 5 == 0:
                        plt.plot(to_numpy(rr),
                                 to_numpy(acc) * to_numpy(ynorm) / model_config['delta_t'],
                                 color=cmap.color(to_numpy(x[n, 5])), linewidth=1, alpha=0.25)
                plt.xlabel('Distance [a.u]', fontsize=12)
                plt.ylabel('MLP [a.u]', fontsize=12)
                acc_list = torch.stack(acc_list)
                coeff_norm = to_numpy(acc_list)
                new_index = np.random.permutation(coeff_norm.shape[0])
                new_index = new_index[0:min(1000, coeff_norm.shape[0])]
                trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                                  n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm[new_index])
                proj_interaction = trans.transform(coeff_norm)
            elif bMesh:
                f_list = []
                popt_list = []
                for n in range(nparticles):
                    embedding = model.a[0, n, :] * torch.ones((100, model_config['embedding']), device=device)
                    if model_config['model'] == 'RD_RPS_Mesh':
                        embedding = model.a[0, n, :] * torch.ones((100, model_config['embedding']), device=device)
                        u = torch.tensor(np.linspace(0, 1, 100)).to(device)
                        u = u[:, None]
                        r = u
                        in_features = torch.cat((u, u, u, u, u, u, embedding), dim=1)
                        h = model.lin_phi(in_features.float())
                        h = h[:, 0]
                    else:
                        r = torch.tensor(np.linspace(-150, 150, 100)).to(device)
                        in_features = torch.cat((r[:, None], embedding), dim=1)
                        h = model.lin_phi(in_features.float())
                        popt, pcov = curve_fit(func_lin, to_numpy(r.squeeze()), to_numpy(h.squeeze()))
                        popt_list.append(popt)
                        h = h[:, 0]
                    f_list.append(h)
                    if (n % 24) & (mask_mesh[n]) == 0:
                        plt.plot(to_numpy(r),
                                 to_numpy(h) * to_numpy(hnorm), linewidth=1,
                                 color='k', alpha=0.05)
                f_list = torch.stack(f_list)
                coeff_norm = to_numpy(f_list)
                popt_list = np.array(popt_list)

                if model_config['model'] == 'RD_RPS_Mesh':
                    trans = umap.UMAP(n_neighbors=500,
                                      n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
                    proj_interaction = trans.transform(coeff_norm)
                else:
                    proj_interaction = popt_list
                    proj_interaction[:, 1] = proj_interaction[:, 0]
            # save projections
            np.save(f'./{log_dir}/tmp_training/umap_projection_{epoch}.npy', proj_interaction)

            ax = fig.add_subplot(1, 6, 4)
            if model_config['cluster_method'] =='kmeans_auto':
                labels, nclusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
            if model_config['cluster_method'] == 'distance_plot':
                labels, nclusters = embedding_cluster.get(proj_interaction, 'distance')
            if model_config['cluster_method'] == 'distance_embedding':
                labels, nclusters = embedding_cluster.get(embedding_, 'distance', thresh=1.5)
            if model_config['cluster_method'] == 'distance_both':
                new_projection = np.concatenate((proj_interaction, embedding_), axis=-1)
                labels, nclusters = embedding_cluster.get(new_projection, 'distance')

            for n in range(nclusters):
                pos = np.argwhere(labels == n)
                pos = np.array(pos)
                if pos.size > 0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=5)
            label_list = []
            for n in range(nparticle_types):
                tmp = labels[index_particles[n]]
                label_list.append(np.round(np.median(tmp)))
            label_list = np.array(label_list)

            plt.xlabel('proj 0', fontsize=12)
            plt.ylabel('proj 1', fontsize=12)
            plt.text(0., 1.1, f'Nclusters: {nclusters}', ha='left', va='top', transform=ax.transAxes)

            ax = fig.add_subplot(1, 6, 5)
            new_labels = labels.copy()
            for n in range(nparticle_types):
                new_labels[labels == label_list[n]] = n
                pos = np.argwhere(labels == label_list[n])
                pos = np.array(pos)
                if pos.size>0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                                color=cmap.color(n), s=0.1)
            Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
            plt.text(0, 1.1, f'Accuracy: {np.round(Accuracy, 3)}', ha='left', va='top', transform=ax.transAxes,
                     fontsize=10)
            print(f'Accuracy: {np.round(Accuracy, 3)}')
            logger.info(f'Accuracy: {np.round(Accuracy, 3)}')

            ax = fig.add_subplot(1, 6, 6)
            model_a_ = model.a.clone().detach()
            model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
            for n in range(nclusters):
                pos = np.argwhere(labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    median_center = model_a_[pos, :]
                    median_center = torch.median(median_center, axis=0).values
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=20, c='r')
                    model_a_[pos, :] = median_center
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=20, c='k')
            model_a_ = torch.reshape(model_a_, (model.a.shape[0], model.a.shape[1], model.a.shape[2]))
            for n in np.unique(new_labels):
                pos = np.argwhere(new_labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size>0:
                    plt.scatter(to_numpy(model_a_[0, pos, 0]), to_numpy(model_a_[0, pos, 1]), color='k', s=6)
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)
            plt.xticks(fontsize=10.0)
            plt.yticks(fontsize=10.0)

            if (bReplace) & ((epoch == 1 * Nepochs // 4) | (epoch == 2 * Nepochs // 4) | (epoch == 3 * Nepochs // 4)):
                # Constrain embedding
                with torch.no_grad():
                    for n in range(model.a.shape[0]):
                        model.a[n] = model_a_[0].clone().detach()
                print(f'regul_embedding: replaced')
                logger.info(f'regul_embedding: replaced')
                plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes,
                         fontsize=10)
                if model_config['fix_cluster_embedding']:
                    lra = 0
                    lr = 1E-3
                    it = 0
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            continue
                        if it == 0:
                            optimizer = torch.optim.Adam([model.a], lr=lra)
                        else:
                            optimizer.add_param_group({'params': parameter, 'lr': lr})
                        it += 1
                    print(f'Learning rates: {lr}, {lra}')
                    logger.info(f'Learning rates: {lr}, {lra}')
            else:
                lra = 1E-3
                lr = 1E-3
                it = 0
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if it == 0:
                        optimizer = torch.optim.Adam([model.a], lr=lra)
                    else:
                        optimizer.add_param_group({'params': parameter, 'lr': lr})
                    it += 1
                print(f'Learning rates: {lr}, {lra}')
                logger.info(f'Learning rates: {lr}, {lra}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()


def data_test(model_config, bVisu=False, bPrint=True, bDetails=False, index_particles=0, prev_nparticles=0, new_nparticles=0, prev_index_particles=0, best_model=0, step=5, bTest='', folder_out='tmp_recons', initial_map='', forced_embedding=[], forced_color=0, ratio=1):
    print('')
    print('Plot validation inference ... ')

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = 'Mesh' in model_config['model']
    delta_t = model_config['delta_t']
    aggr_type = model_config['aggr_type']

    if model_config['boundary'] == 'no':  # change this for usual BC
        def bc_pos(X):
            return X

        def bc_diff(D):
            return D
    else:
        def bc_pos(X):
            return torch.remainder(X, 1.0)

        def bc_diff(D):
            return torch.remainder(D - .5, 1.0) - .5

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if model_config['model'] == 'PDE_G':
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
        p_mass = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        p_mass = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'PDE_E':
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
        p_elec = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_elec[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if bMesh:
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p),
                      clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])
        if (model_config['model'] == 'WaveMesh'):
            model_mesh = Mesh_Laplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
        if (model_config['model'] == 'RD_RPS_Mesh'):
            model_mesh = Mesh_RPS(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))
    if best_model == -1:
        net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    else:
        net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"
    print('Graph files N: ', NGraphs - 1)
    print(f'network: {net}')

    if bMesh:
        state_dict = torch.load(net, map_location=device)
        model_mesh.load_state_dict(state_dict['model_state_dict'])
        model_mesh.eval()
    else:
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

    if len(forced_embedding) > 0:
        with torch.no_grad():
            model.a[0] = torch.tensor(forced_embedding, device=device).repeat(nparticles, 1)

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
    else:
        # labels = T1
        print('Use ground truth labels')

    # nparticles larger than initially
    if ratio > 1:  # nparticles larger than initially

        prev_index_particles = index_particles

        new_nparticles = nparticles * ratio
        prev_nparticles = nparticles

        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio}')
        print('')

        embedding = model.a[0].data.clone().detach()
        new_embedding = []
        new_labels = []

        for n in range(nparticle_types):
            for m in range(ratio):
                if (n == 0) & (m == 0):
                    new_embedding = embedding[prev_index_particles[n].astype(int), :]
                    new_labels = labels[prev_index_particles[n].astype(int)]
                else:
                    new_embedding = torch.cat((new_embedding, embedding[prev_index_particles[n].astype(int), :]),
                                              axis=0)
                    new_labels = torch.cat((new_labels, labels[prev_index_particles[n].astype(int)]), axis=0)

        model.a = nn.Parameter(
            torch.tensor(np.ones((NGraphs - 1, int(prev_nparticles) * ratio, 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        model.a.requires_grad = False
        model.a[0] = new_embedding
        labels = new_labels
        nparticles = new_nparticles
        model_config['nparticles'] = new_nparticles

        index_particles = []
        np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
        for n in range(model_config['nparticle_types']):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)
    if bMesh:
        hnorm = torch.load(f'./log/try_{dataset_name}/hnorm.pt', map_location=device).to(device)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if bPrint:
        print(table)
        print(f"Total Trainable Params: {total_params}")

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    x_list.append(torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_0.pt', map_location=device))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    if bMesh:
        index_particles = []
        T1 = []
        for n in range(model_config['nparticle_types']):
            index = np.argwhere(to_numpy(x[:, 5]) == n)
            index_particles.append(index.squeeze())

    if bPrint:
        print('')
        print(f'x: {x.shape}')
        print(f'index_particles: {index_particles[0].shape}')
        print('')
    time.sleep(0.5)

    rmserr_list = []
    discrepency_list = []

    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    time.sleep(1)
    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        if bMesh:
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset = data.Data(x=x, pos=x[:, 1:3])
            transform_0 = T.Compose([T.Delaunay()])
            dataset_face = transform_0(dataset).face
            mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
            edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)
            dataset_mesh = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)

        if model_config['model'] == 'DiffMesh':
            with torch.no_grad():
                pred = model_mesh(dataset_mesh, data_id=0, )
            x[:, 6:7] += pred * hnorm * delta_t
        elif model_config['model'] == 'WaveMesh':
            with torch.no_grad():
                pred = model_mesh(dataset_mesh, data_id=0)
            x[:, 7:8] += pred * hnorm * delta_t
            x[:, 6:7] += x[:, 7:8] * delta_t
        elif (model_config['model'] == 'RD_RPS_Mesh'):
            mask = to_numpy(
                torch.argwhere((x[:, 1] > 0.02) & (x[:, 1] < 0.98) & (x[:, 2] > 0.02) & (x[:, 2] < 0.98))).astype(int)
            mask = mask[:, 0:1]
            with torch.no_grad():
                pred = model_mesh(dataset_mesh, data_id=0)
                x[mask, 6:9] += pred[mask] * hnorm * delta_t
        else:
            distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

            edge_index = adj_t.nonzero().t().contiguous()

            dataset = data.Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset, data_id=0, training=False, vnorm=vnorm,
                          phi=torch.zeros(1, device=device))  # acceleration estimation

            if model_config['prediction'] == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm
                x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t)  # position update

            x_recons.append(x.clone().detach())
            y_recons.append(y.clone().detach())

        if bMesh:
            mask = to_numpy(
                torch.argwhere((x[:, 1] < 0.025) | (x[:, 1] > 0.975) | (x[:, 2] < 0.025) | (x[:, 2] > 0.975))).astype(
                int)
            mask = mask[:, 0:1]
            if model_config['model'] == 'WaveMesh':
                x[mask, 6:8] = 0
            rmserr = torch.sqrt(torch.mean(torch.sum((x[:, 6:7] - x0_next[:, 6:7]) ** 2, axis=1)))
            rmserr_list.append(rmserr.item())
        else:
            rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
            rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0) & bVisu:

            if True:  # 'color' in bStyle:

                sc = 80

                fig = plt.figure(figsize=(12, 12))
                # plt.ion()
                if bMesh:
                    pts = x[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = torch.sum(x[tri.simplices, 6], axis=1) / 3.0
                    if model_config['model'] == 'DiffMesh':
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                    if model_config['model'] == 'WaveMesh':
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                    if (model_config['model'] == 'RD_Gray_Scott_Mesh'):
                        fig = plt.figure(figsize=(12, 6))
                        ax = fig.add_subplot(1, 2, 1)
                        colors = torch.sum(x[tri.simplices, 6], axis=1) / 3.0
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('off')
                        ax = fig.add_subplot(1, 2, 2)
                        colors = torch.sum(x[tri.simplices, 7], axis=1) / 3.0
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('off')
                    if (model_config['model'] == 'RD_RPS_Mesh'):
                        fig = plt.figure(figsize=(12, 12))
                        H1_IM = torch.reshape(x0[:, 6:9], (100, 100, 3))
                        plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('off')
                else:
                    for n in range(nparticle_types):
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                    x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n))

                if bMesh | (model_config['boundary'] == 'periodic'):
                    # plt.text(0.08, 0.92, f'frame: {it}',fontsize=8,color='w')
                    gg = 0
                    # plt.text(0, 1.03, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                # else:
                # plt.text(-1.25, 1.5, f'frame: {it}')
                # plt.text(-1.25, 1.4, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                # plt.xlim([-0.5, 0.5])
                # plt.ylim([-0.5, 0.5])

                # plt.xlim([0, 1])
                # plt.ylim([0, 1])

                # plt.xticks([])
                # plt.yticks([])

                # plt.xlim([-4, 4])
                # plt.ylim([-4, 4])

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{dataset_name}_{it}.tif", dpi=300)
                plt.close()

            if False:
                if bMesh:
                    dataset2 = dataset_mesh
                else:
                    distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                    adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                    edge_index2 = adj_t2.nonzero().t().contiguous()
                    dataset2 = data.Data(x=x, edge_index=edge_index2)

                fig = plt.figure(figsize=(16, 7.2))
                plt.ion()

                for k in range(5):
                    if k == 0:
                        ax = fig.add_subplot(2, 4, 1)
                        x_ = x00
                        sc = 2
                    elif k == 1:
                        ax = fig.add_subplot(2, 4, 2)
                        x_ = x0
                        sc = 2
                    elif k == 2:
                        ax = fig.add_subplot(2, 4, 6)
                        x_ = x
                        sc = 2
                    elif k == 3:
                        ax = fig.add_subplot(2, 4, 3)
                        x_ = x0
                        sc = 5
                    elif k == 4:
                        ax = fig.add_subplot(2, 4, 7)
                        x_ = x
                        sc = 5

                    if (k == 0) & (bMesh):
                        plt.scatter(to_numpy(x0_next[:, 6]), to_numpy(x[:, 6]), s=1, alpha=0.25, c='k')
                        plt.xlabel('True [a.u.]', fontsize="14")
                        plt.ylabel('Model [a.u]', fontsize="14")
                    elif model_config['model'] == 'PDE_G':
                        for n in range(nparticle_types):
                            g = to_numpy(p_mass[to_numpy(T1[index_particles[n], 0])]) * 10 * sc
                            plt.scatter(x_[index_particles[n], 1].detach().cpu(),
                                        x_[index_particles[n], 2].detach().cpu(),
                                        s=g, alpha=0.75, color=cmap.color(n))  # , facecolors='none', edgecolors='k')
                    elif model_config['model'] == 'PDE_E':
                        for n in range(nparticle_types):
                            g = np.abs(to_numpy(p_elec[to_numpy(T1[index_particles[n], 0])]) * 20) * sc
                            if model_config['p'][n][0] <= 0:
                                plt.scatter(to_numpy(x_[index_particles[n], 1]),
                                            to_numpy(x_[index_particles[n], 2]), s=g,
                                            c='r', alpha=0.5)  # , facecolors='none', edgecolors='k')
                            else:
                                plt.scatter(to_numpy(x_[index_particles[n], 1]),
                                            to_numpy(x_[index_particles[n], 2]), s=g,
                                            c='b', alpha=0.5)  # , facecolors='none', edgecolors='k')
                    elif bMesh:
                        pts = to_numpy(x_[:, 1:3])
                        tri = Delaunay(pts)
                        colors = torch.sum(x_[tri.simplices, 6], axis=1) / 3.0
                        if model_config['model'] == 'WaveMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=to_numpy(colors), vmin=-1500, vmax=1500)
                        elif model_config['model'] == 'RD_RPS_Mesh':
                            H1_IM = torch.reshape(x_[:, 6:9], (100, 100, 3))
                            plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                        elif model_config['model'] == 'DiffMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=to_numpy(colors), vmin=0, vmax=5000)
                    else:
                        if ((k == 2) | (k == 4)) & (len(forced_embedding) > 0):
                            for n in range(nparticle_types):
                                plt.scatter(x_[index_particles[n], 1].detach().cpu(),
                                            x_[index_particles[n], 2].detach().cpu(),
                                            s=sc, color=cmap.color(forced_color))
                        else:
                            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(),
                                        s=sc, color=cmap.color(to_numpy(labels)))

                    if (k > 2) & (bMesh == False):
                        for n in range(nparticles):
                            plt.arrow(x=x_[n, 1].detach().cpu().item(), y=x_[n, 2].detach().cpu().item(),
                                      dx=x_[n, 3].detach().cpu().item() * model_config['arrow_length'],
                                      dy=x_[n, 4].detach().cpu().item() * model_config['arrow_length'], color='k')
                    if k < 3:
                        if (k == 0) & (bMesh):
                            plt.xlim([-1.3, 1.3])
                            plt.ylim([-1.3, 1.3])
                        elif (model_config['boundary'] == 'no'):
                            plt.xlim([-1.3, 1.3])
                            plt.ylim([-1.3, 1.3])
                        elif not (model_config['model'] == 'RD_RPS_Mesh'):
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                    else:
                        if model_config['model'] == 'RD_RPS_Mesh':
                            plt.xlim([40, 60])
                            plt.ylim([40, 60])
                        elif bMesh | ('Boids' in model_config['description']) | (
                                model_config['boundary'] == 'periodic'):
                            plt.xlim([0.3, 0.7])
                            plt.ylim([0.3, 0.7])
                        elif not (model_config['model'] == 'RD_RPS_Mesh'):
                            plt.xlim([-0.25, 0.25])
                            plt.ylim([-0.25, 0.25])
                    plt.xticks([])
                    plt.yticks([])

                if True:
                    ax = fig.add_subplot(2, 4, 4)
                    plt.plot(np.arange(len(rmserr_list)), rmserr_list, label='RMSE', c='k')
                    plt.ylim([0, 0.1])
                    plt.xlim([0, nframes])
                    plt.tick_params(axis='both', which='major', labelsize=10)
                    plt.xlabel('Frame [a.u]', fontsize="14")
                    ax.set_ylabel('RMSE [a.u]', fontsize="14", color='k')
                    if bMesh:
                        plt.ylim([0, 5000])

                    if bDetails:
                        ax = fig.add_subplot(2, 5, 6)
                        pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                        vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                        nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.2)
                        if model_config['boundary'] == 'no':
                            plt.xlim([-1.3, 1.3])
                            plt.ylim([-1.3, 1.3])
                        else:
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])

                    ax = fig.add_subplot(2, 4, 8)
                    if not (bMesh):
                        temp1 = torch.cat((x, x0_next), 0)
                        temp2 = torch.tensor(np.arange(nparticles), device=device)
                        temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
                        temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
                        temp4 = torch.t(temp4)
                        distance3 = torch.sqrt(torch.sum((x[:, 1:3] - x0_next[:, 1:3]) ** 2, 1))
                        p = torch.argwhere(distance3 < 0.3)
                        pos = dict(enumerate(np.array((temp1[:, 1:3]).detach().cpu()), 0))
                        dataset = data.Data(x=temp1[:, 1:3], edge_index=torch.squeeze(temp4[:, p]))
                        vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                        nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False)
                        if model_config['boundary'] == 'no':
                            plt.xlim([-1.3, 1.3])
                            plt.ylim([-1.3, 1.3])
                        else:
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])

                plt.tight_layout()

                if len(forced_embedding) > 0:
                    plt.savefig(f"./{log_dir}/tmp_recons/Fig_{dataset_name}_{forced_color}_{it}.tif", dpi=300)
                else:
                    plt.savefig(f"./{log_dir}/tmp_recons/Fig_{dataset_name}_{it}.tif", dpi=300)

                plt.close()

    print(f'RMSE: {np.round(rmserr.item(), 4)}')
    if bPrint:
        print(f'dataset_name: {dataset_name}')

    torch.save(x_recons, f'{log_dir}/x_list.pt')
    torch.save(y_recons, f'{log_dir}/y_list.pt')



if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    # config_manager = create_config_manager(config_type='simulation')

    config_manager = ConfigManager(config_schema='./config_schemas/config_schema_simulation.yaml')
    config_list = ['config_maze'] # ['config_RD_RPS2c'] # ['config_wave_HR3d'] #


    for config in config_list:

        # Load parameters from config file
        # model_config = load_model_config(id=config)
        model_config = config_manager.load_and_validate_config(f'./config/{config}.yaml')
        model_config['dataset'] = config[7:]

        for key, value in model_config.items():
            print(key, ":", value)
            if ('E-' in str(value)) | ('E+' in str(value)):
                value = float(value)
                model_config[key] = value

        cmap = cc(model_config=model_config)  # create colormap for given model_config

        data_generate(model_config, device=device, bVisu=True, bStyle='color', alpha=1, bErase=True, step=model_config['nframes']//200)
        # data_train(model_config)
        # data_plot(model_config, epoch=-1, bPrint=True, best_model=4, cluster_method=model_config['cluster_method'])
        # data_test(model_config, bVisu=True, bPrint=True, best_model=20, bDetails=False, step = model_config['nframes']//50, ratio=1)

        # data_train_shrofflab_celegans(model_config)
        # data_test_shrofflab_celegans(model_config)

        # x, rmserr_list = data_test(model_config, bVisu=True, bPrint=True, best_model=-1, step=10, bTest='',initial_map='', forced_embedding=[1.265,0.636], forced_color=0)

