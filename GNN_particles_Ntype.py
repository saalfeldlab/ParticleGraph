import glob
import json
import logging
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import networkx as nx
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

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.data_loaders import *
from ParticleGraph.config_manager import create_config_manager, ConfigManager
from ParticleGraph.utils import to_numpy
from ParticleGraph.generators.PDE_A import PDE_A
from ParticleGraph.generators.PDE_B import PDE_B
from ParticleGraph.generators.PDE_E import PDE_E
from ParticleGraph.generators.PDE_G import PDE_G
from ParticleGraph.generators.Laplacian_A import Laplacian_A
from ParticleGraph.generators.RD_FitzHugh_Nagumo import RD_FitzHugh_Nagumo
from ParticleGraph.generators.RD_Gray_Scott import RD_Gray_Scott
from ParticleGraph.generators.RD_RPS import RD_RPS

from ParticleGraph.models.ElecParticles import ElecParticles
from ParticleGraph.models.GravityParticles import GravityParticles
from ParticleGraph.models.InteractionCElegans import InteractionCElegans
from ParticleGraph.models.InteractionParticles import InteractionParticles
from ParticleGraph.models.MeshLaplacian import MeshLaplacian
from ParticleGraph.models.Mesh_RPS import Mesh_RPS
from ParticleGraph.models.PDE_embedding import PDE_embedding
from ParticleGraph.embedding_cluster import *

def func_pow(x, a, b):
    return a / (x**b)


def func_lin(x, a, b):
    return a * x + b


def normalize99(Y, lower=1, upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
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


class cc:

    def __init__(self, model_config):
        self.model_config = model_config
        self.model = model_config['model']
        if model_config['cmap'] == 'tab10':
            self.nmap = 8
        else:
            self.nmap = model_config['nparticle_types']

        self.bMesh = 'Mesh' in model_config['model']

    def color(self, index):

        if self.model == 'ElecParticles':

            if index == 0:
                index = (0, 0, 1)
            elif index == 1:
                index = (1, 0, 0)
            elif index == 2:
                index = (0, 0.5, 0.75)
            elif index == 3:
                index = (0.75, 0, 0)
            return (index)
        elif self.bMesh:
            if index == 0:
                index = (0, 0, 0)
            else:
                color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
                index = color_map(index / self.nmap)
                
        else:
            # color_map = plt.cm.get_cmap(self.model_config['cmap'])
            color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
            index = color_map(index / self.nmap)

        return index


def data_generate(model_config, bVisu=True, bStyle='color', bErase=False, bLoad_p=False, step=5, alpha=0.2, ratio=1,scenario='none', device=[]):
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
    json_ = json.dumps(model_config)
    f = open(f"{folder}/model_config.json", "w")
    f.write(json_)
    f.close()

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

    cycle_length = torch.clamp(torch.abs(
        torch.ones(nparticle_types, 1, device=device) * 400 + torch.randn(nparticle_types, 1, device=device) * 150),
        min=100, max=700)
    if bDivision:
        for n in range(model_config['nparticle_types']):
            print(f'cell cycle duration: {to_numpy(cycle_length[n])}')
        torch.save(torch.squeeze(cycle_length), f'graphs_data/graphs_particles_{dataset_name}/cycle_length.pt')

    rr = torch.tensor(np.linspace(0, radius * 2, 1000))
    rr = rr.to(device)
    if bMesh:
        particle_value_map = model_config['particle_value_map']
        particle_type_map = model_config['particle_type_map']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if model_config['model'] == 'PDE_A':
        print(f'Generate PDE_A')
        if bLoad_p:
            p = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt')
        else:
            p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
            if len(model_config['p']) > 0:
                for n in range(nparticle_types):
                    p[n] = torch.tensor(model_config['p'][n])
        if nparticle_types == 1:
            model = PDE_A(aggr_type=aggr_type, p=p, delta_t=model_config['delta_t'], sigma=model_config['sigma'], bc_diff=bc_diff)
        else:
            model = PDE_A(aggr_type=aggr_type, p=torch.squeeze(p), delta_t=model_config['delta_t'], sigma=model_config['sigma'], bc_diff=bc_diff)
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
            model = PDE_A(aggr_type=aggr_type, p=p, delta_t=model_config['delta_t'], bc_diff=bc_diff)
        else:
            model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), delta_t=model_config['delta_t'], bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
        torch.save(torch.squeeze(p), f'graphs_data/graphs_particles_{dataset_name}/p.pt')
    if model_config['model'] == 'GravityParticles':
        if model_config['p'] == 'continuous':
            p = np.linspace(0.5, 5, nparticles)
            p = torch.tensor(p, device=device)
            print ('p: continous ')
        else:
            p = np.linspace(0.5, 5, nparticle_types)
            p = torch.tensor(p, device=device)
            if len(model_config['p']) > 0:
                for n in range(nparticle_types):
                    p[n] = torch.tensor(model_config['p'][n])
        model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p), delta_t=model_config['delta_t'],
                      clamp=model_config['clamp'], pred_limit=model_config['pred_limit'], bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
        torch.save(torch.squeeze(p), f'graphs_data/graphs_particles_{dataset_name}/p.pt')
    if model_config['model'] == 'ElecParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
                print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
                torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = PDE_E(aggr_type=aggr_type, p=torch.squeeze(p), delta_t=model_config['delta_t'],
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
        model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p), delta_t=model_config['delta_t'],
                      clamp=model_config['clamp'], pred_limit=model_config['pred_limit'], bc_diff=bc_diff)
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])

        if (model_config['model'] == 'RD_Gray_Scott_Mesh'):
            model_mesh = RD_Gray_Scott(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'], bc_diff=bc_diff)
        elif (model_config['model'] == 'RD_FitzHugh_Nagumo_Mesh'):
            model_mesh = RD_FitzHugh_Nagumo(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'], bc_diff=bc_diff)
        elif (model_config['model'] == 'RD_RPS_Mesh'):
            model_mesh = RD_RPS(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'], bc_diff=bc_diff)
        elif (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
            model_mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'], bc_diff=bc_diff)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')

    torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    for run in range(model_config['nrun']):

        x_list = []
        y_list = []
        h_list = []

        # initialize particle and graph states
        if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
            X1 = torch.rand(nparticles, 2, device=device)
        else:
            X1 = torch.randn(nparticles, 2, device=device) * 0.5
        V1 = v_init * torch.randn((nparticles, 2), device=device)
        V1 = torch.clamp(V1, min=-torch.std(V1), max=+torch.std(V1))
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = T1[:, None]
        cycle_length_distrib = cycle_length[to_numpy(T1[:, 0]).astype(int)]
        if model_config['p'] == 'continuous':
            T1 = torch.tensor(np.arange(nparticles), device=device)
            T1 = T1[:, None]
        H1 = torch.zeros((nparticles, 2), device=device)
        A1 = torch.rand(nparticles, device=device)
        A1 = A1[:, None]
        A1 = A1 * cycle_length_distrib
        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:, None]

        # create differnet initial conditions
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

        if bMesh:
            x_width = int(np.sqrt(nparticles))
            xs = torch.linspace(0, 1, steps=x_width)
            ys = torch.linspace(0, 1, steps=x_width)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            x = torch.reshape(x, (x_width ** 2, 1))
            y = torch.reshape(y, (x_width ** 2, 1))
            x_width = 1 / x_width / 8
            X1[0:nparticles, 0:1] = x[0:nparticles]
            X1[0:nparticles, 1:2] = y[0:nparticles]
            X1 = X1 + torch.randn(nparticles, 2, device=device) * x_width
            X1_ = torch.clamp(X1, min=0, max=1)

            i0 = imread(f'graphs_data/{particle_value_map}')
            values = i0[(to_numpy(X1_[:, 0]) * 255).astype(int), (to_numpy(X1_[:, 1]) * 255).astype(int)]

            if (model_config['model'] == 'RD_Gray_Scott_Mesh'):
                H1[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
                H1[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
            elif (model_config['model'] == 'RD_FitzHugh_Nagumo_Mesh'):
                H1 = torch.zeros((nparticles, 2), device=device) + torch.rand((nparticles, 2),device=device) * 0.1
            elif (model_config['model'] == 'RD_RPS_Mesh'):
                H1 = torch.rand((nparticles, 3),device=device)
                s = torch.sum(H1,axis=1)
                for k in range(3):
                    H1[:,k]=H1[:,k]/s
            elif (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                H1[:, 0] = torch.tensor(values / 255 * 5000, device=device)

            i0 = imread(f'graphs_data/{particle_type_map}')
            values = i0[
                (to_numpy(X1_[:, 0]) * 255).astype(int), (to_numpy(X1_[:, 1]) * 255).astype(
                    int)]
            T1 = torch.tensor(values, device=device)
            T1 = T1[:, None]
            # plt.scatter(X1[:, 0].detach().cpu().numpy(), X1[:, 1].detach().cpu().numpy(), s=10,c=T1[:, 0].detach().cpu().numpy())

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach()), 1)

            dataset = data.Data(x=x, pos=x[:, 1:3])
            transform_0 = T.Compose([T.Delaunay()])
            dataset_face = transform_0(dataset).face
            mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
            edge_index_mesh, edge_weight_mesh = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face,normalization="None")  # "None", "sym", "rw"

        time.sleep(0.5)
        for it in trange(model_config['start_frame'], nframes+1):

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
            if (it >= 0):
                x_list.append(x.clone().detach())
            # create mesh dataset
            if bMesh:
                dataset_mesh = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
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
                y_list.append(y.clone().detach())

            # Euler integration update
            if not (bMesh):
                if model_config['prediction'] == '2nd_derivative':
                    V1 += y[:, 0:2] * delta_t
                else:
                    V1 = y[:, 0:2]
                X1 = bc_pos(X1 + V1 * delta_t)
                A1 = A1 + 1
            # append h_list
            # Euler integration update for mesh
            if it >= 0:
                if model_config['model'] == 'DiffMesh':
                        mask = to_numpy(torch.argwhere((X1[:, 0] > 0.1) & (X1[:, 0] < 0.9) & (X1[:, 1] > 0.1) & (X1[:, 1] < 0.9))).astype(int)
                        mask = mask[:, 0:1]
                        with torch.no_grad():
                            pred = model_mesh(dataset_mesh)
                            H1[mask, 1:2] = pred[mask]
                        H1[mask, 0:1] += H1[mask, 1:2] * delta_t
                        h_list.append(pred)
                if model_config['model'] == 'WaveMesh':
                        with torch.no_grad():
                            pred = model_mesh(dataset_mesh)
                            H1[:, 1:2] += pred[:] * delta_t
                        H1[:, 0:1] += H1[:, 1:2] * delta_t
                        h_list.append(pred)
                if (model_config['model'] == 'RD_Gray_Scott_Mesh') | (model_config['model'] == 'RD_FitzHugh_Nagumo_Mesh') | (model_config['model'] == 'RD_RPS_Mesh'):
                        mask = to_numpy(torch.argwhere((X1[:, 0] > 0.02) & (X1[:, 0] < 0.98) & (X1[:, 1] > 0.02) & (X1[:, 1] < 0.98))).astype(int)
                        mask = mask[:, 0:1]
                        with torch.no_grad():
                            pred = model_mesh(dataset_mesh)
                            H1[mask] += pred[mask] * delta_t
                        h_list.append(pred)

            # output plots
            if bVisu & (run == 0) & (it % step == 0) & (it >= 0) :

                if 'graph' in bStyle:
                    fig = plt.figure(figsize=(10, 10))
                    # plt.ion()

                    distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                    adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                    edge_index2 = adj_t2.nonzero().t().contiguous()
                    dataset2 = data.Data(x=x, edge_index=edge_index2)
                    pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                    vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False,alpha=alpha)

                    if model_config['model'] == 'GravityParticles':
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
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500, vmax=2500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=2500)
                    elif model_config['model'] == 'ElecParticles':
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
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n),alpha=0.5)
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

                    sc=80

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
                                          facecolors=colors.detach().cpu().numpy(),vmin=0,vmax=1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.axis('off')
                            ax = fig.add_subplot(1, 2, 2)
                            colors = torch.sum(x[tri.simplices, 7], axis=1) / 3.0
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(),vmin=0,vmax=1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.axis('off')
                        if (model_config['model'] == 'RD_RPS_Mesh'):
                            fig = plt.figure(figsize=(12, 12))
                            H1_IM=torch.reshape(H1,(100,100,3))
                            plt.imshow(H1_IM.detach().cpu().numpy(),vmin=0,vmax=1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.axis('off')
                    else:
                        for n in range(nparticle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n))

                    if bMesh | (model_config['boundary'] == 'periodic'):
                        # plt.text(0.08, 0.92, f'frame: {it}',fontsize=8,color='w')
                        gg=0
                        # plt.text(0, 1.03, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                        # plt.xlim([0, 1])
                        # plt.ylim([0, 1])
                    else:
                        # plt.text(-1.25, 1.5, f'frame: {it}')
                        # plt.text(-1.25, 1.4, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                        plt.xlim([-0.5, 0.5])
                        plt.ylim([-0.5, 0.5])

                    plt.xticks([])
                    plt.yticks([])
                    # plt.xlim([0, 1])
                    # plt.ylim([0, 1])

                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_particles_{dataset_name}/tmp_data/Fig_color_{it}.jpg", dpi=100)
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

                        # plt.scatter(x[:, 1].detach().cpu().numpy(),x[:, 2].detach().cpu().numpy(), s=10, alpha=0.75,
                        #                 c=x[:, 6].detach().cpu().numpy(), cmap='gist_gray',vmin=-5000,vmax=5000)
                        # ax.set_facecolor([0.5,0.5,0.5])
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
        torch.save(h_list, f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt')

    model_config['nparticles'] = int(model_config['nparticles'] / ratio)


def data_train(model_config, bSparse=False):
    print('')

    model = []
    Nepochs = model_config['Nepochs']
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
    embedding = model_config['embedding']
    batch_size = model_config['batch_size']
    bMesh = 'Mesh' in model_config['model']
    bRegul = 'regul' in model_config['sparsity']
    bReplace = 'replace' in model_config['sparsity']
    kmeans_input = model_config['kmeans_input']
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

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

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
    for run in trange(NGraphs):   ##############to be changed
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
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    print(to_numpy(vnorm), to_numpy(ynorm))
    logger.info(f'vnorm ynorm: {to_numpy(vnorm[4])} {to_numpy(ynorm[4])}')
    if bMesh:
        h_list = []
        for run in trange(NGraphs):
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt', map_location=device)
            h_list.append(torch.stack(h))
        h = torch.stack(h_list)
        h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(torch.mean(h), torch.std(h))
        logger.info(f'hnorm : {to_numpy(hnorm)}')

    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = InteractionParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'DiffMesh'):
        model = MeshLaplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'WaveMesh'):
        model = MeshLaplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'RD_RPS_Mesh'):
        model = Mesh_RPS(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

    # net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_17.pt"
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
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {Nepochs}')
    print(f'batch_size: {batch_size}')
    logger.info(f'batch_size: {batch_size}')
    print('')
    min_radius = 0.002
    model.train()

    model.train()
    best_loss = np.inf
    list_loss = []

    if 'data_augmentation_loop' in model_config:
        data_augmentation_loop = model_config['data_augmentation_loop']
    else:
        data_augmentation_loop = 200
    print(f'data_augmentation_loop: {data_augmentation_loop}')
    logger.info(f'data_augmentation_loop: {data_augmentation_loop}')


    if bMesh:
        h_list=[]
        for run in trange(0, NGraphs):
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt',map_location=device)
            h_list.append(torch.stack(h))
        x = x_list[0][0].clone().detach()
        index_particles = []
        for n in range(model_config['nparticle_types']):
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
            index_particles.append(index.squeeze())
        logger.info(hnorm)
        batch_size = 1

    x = x_list[1][0].clone().detach()

    if bMesh:
        dataset = data.Data(x=x, pos=x[:, 1:3])
        transform_0 = T.Compose([T.Delaunay()])
        dataset_face = transform_0(dataset).face
        mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
        edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face,
                                                               normalization="None")  # "None", "sym", "rw"

    print('Start training ...')
    print(f'   {nframes * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info("Start training ...")
    time.sleep(0.5)

    for epoch in range(Nepochs + 1):

        if epoch == 1:
            min_radius = model_config['min_radius']
            logger.info(f'min_radius: {min_radius}')
        if (epoch == 2) & (batch_size==1):
            batch_size = 8
            print(f'batch_size: {batch_size}')
            logger.info(f'batch_size: {batch_size}')
        if epoch == 3 * Nepochs // 4:
            lra = 1E-3
            lr = 5E-4
            table = PrettyTable(["Modules", "Parameters"])
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

        total_loss = 0

        for N in trange(0, nframes * data_augmentation_loop // batch_size):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            loss_embedding = torch.zeros(1, dtype=torch.float32, device=device)

            for batch in range(batch_size):

                k = np.random.randint(nframes - 1)
                x = x_list[run][k].clone().detach()

                if bMesh:
                    dataset = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)
                    dataset_batch.append(dataset)
                    y = h_list[run][k].clone().detach() / hnorm
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
                        y = y / ynorm[4]
                    else:
                        y = y / vnorm[4]
                    if data_augmentation:
                        new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                        new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                        y[:, 0] = new_x
                        y[:, 1] = new_y
                    if batch == 0:
                        y_batch = y
                    else:
                        y_batch = torch.cat((y_batch, y), axis=0)

                    if bRegul & (epoch >= Nepochs // 4) & (epoch <= 3 * Nepochs // 4):
                        embedding = []
                        for n in range(model.a.shape[0]):
                            embedding.append(model.a[n])
                        embedding = torch.stack(embedding).squeeze()

                        if model.a.shape[0] > 2:
                            embedding = torch.reshape(embedding,
                                                      [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
                        radius_embedding = torch.std(embedding) / 2

                        distance = torch.sum((embedding[:, None, :] - embedding[None, :, :]) ** 2, axis=2)
                        adj_t = (distance < radius_embedding ** 2).float() * 1
                        t = torch.Tensor([radius_embedding ** 2])
                        edges_embedding = adj_t.nonzero().t().contiguous()
                        dataset_embedding = data.Data(x=embedding, edge_index=edges_embedding)
                        pred = model_embedding(dataset_embedding)
                        loss_embedding += pred.norm(2)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                if bMesh:
                    pred = model(batch, data_id=run - 1)
                else:
                    pred = model(batch, data_id=run - 1, step=1, vnorm=vnorm, cos_phi=cos_phi, sin_phi=sin_phi)

            loss = (pred - y_batch).norm(2) + loss_embedding

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
                plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{N}.tif")

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        if (total_loss / nparticles / batch_size / (N + 1) < best_loss):
            best_loss = total_loss / (N + 1) / nparticles / batch_size
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs.pt'))
            print("Epoch {}. Loss: {:.6f} saving model  ".format(epoch, total_loss / (N + 1) / nparticles / batch_size))
            logger.info(
                "Epoch {}. Loss: {:.6f} saving model  ".format(epoch, total_loss / (N + 1) / nparticles / batch_size))
        else:
            print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / nparticles / batch_size))
            logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / nparticles / batch_size))

        list_loss.append(total_loss / (N + 1) / nparticles / batch_size)

        fig = plt.figure(figsize=(16, 4))
        plt.ion()
        ax = fig.add_subplot(1, 4, 1)
        plt.plot(list_loss, color='k')
        plt.ylim([0, 0.010])
        plt.xlim([0, Nepochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 4, 2)
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

        ax = fig.add_subplot(1, 4, 3)
        if model_config['model'] == 'ElecParticles':
            acc_list = []
            for m in range(model.a.shape[0]):
                for k in range(nparticle_types):
                    for n in index_particles[k]:
                        rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                        embedding0 = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                        embedding1 = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                        in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                                 rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding0, embedding1), dim=1)
                        acc = model.lin_edge(in_features.float())
                        acc = acc[:, 0]
                        acc_list.append(acc)
                        if n % 5 == 0:
                            plt.plot(to_numpy(rr),
                                     to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
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
        elif model_config['model'] == 'GravityParticles':
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
                         acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['delta_t'],
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
                             to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
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
            for n in range(nparticles):
                embedding = model.a[0, n, :] * torch.ones((100, model_config['embedding']), device=device)
                if model_config['model'] == 'RD_RPS_Mesh':
                    u = torch.tensor(np.linspace(0, 1, 100)).to(device)
                    u = u[:, None]
                    in_features = torch.cat((u,u,u,u,u,u, embedding), dim=1)
                    r = u
                else:
                    r = torch.tensor(np.linspace(-250, 250, 100)).to(device)
                    in_features = torch.cat((r[:,None], embedding), dim=1)
                h=model.lin_phi(in_features.float())
                h = h[:, 0]
                f_list.append(h)
                if n % 100 == 0:
                    plt.plot(to_numpy(r),
                             to_numpy(h) * to_numpy(hnorm), linewidth=1,
                             color='k', alpha=0.05)
            f_list = torch.stack(f_list)
            coeff_norm = to_numpy(f_list)
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
            ax = fig.add_subplot(2, 4, 4)
            for n in range(nparticle_types):
                plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1], s=5)
            plt.xlabel('UMAP 0', fontsize=12)
            plt.ylabel('UMAP 1', fontsize=12)
            kmeans = KMeans(init="random", n_clusters=nparticle_types, n_init=1000, max_iter=10000, random_state=13)
            kmeans.fit(proj_interaction)
            for n in range(nparticle_types):
                plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)
                pos = np.argwhere(kmeans.labels_ == n).squeeze().astype(int)

        # save UMAP projection
        np.save(f'./{log_dir}/models/umap_projection_{epoch}.npy', proj_interaction)

        ax = fig.add_subplot(1, 4, 4)
        for n in range(nparticle_types):
            plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                        color=cmap.color(n), s=5, alpha=0.75)
        plt.xlabel('UMAP 0', fontsize=12)
        plt.ylabel('UMAP 1', fontsize=12)
        kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=5000, max_iter=10000,
                        random_state=13)

        if kmeans_input == 'plot':
            kmeans.fit(proj_interaction)
        if kmeans_input == 'embedding':
            kmeans.fit(embedding_)

        print(f'kmeans.inertia_: {np.round(kmeans.inertia_, 3)}')

        for n in range(nparticle_types):
            tmp = kmeans.labels_[index_particles[n]]
            sub_group = np.round(np.median(tmp))
            accuracy = len(np.argwhere(tmp == sub_group)) / len(tmp) * 100
            print(f'Sub-group {n} accuracy: {np.round(accuracy, 3)}')
        for n in range(model_config['ninteractions']):
            plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()

        if (epoch == 1 * Nepochs // 4) | (epoch == 2 * Nepochs // 4) | (epoch == 3 * Nepochs // 4):

            model_a_ = model.a.clone().detach()
            model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
            embedding_center = []
            for k in range(model_config['ninteractions']):
                pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, axis=0).values
                embedding_center.append(median_center.clone().detach())
                model_a_[pos, :] = torch.median(median_center, axis=0).values
            model_a_ = torch.reshape(model_a_, (model.a.shape[0], model.a.shape[1], model.a.shape[2]))

            # Constrain embedding with UMAP of plots clustering
            if bReplace:
                with torch.no_grad():
                    for n in range(model.a.shape[0]):
                        model.a[n] = model_a_[0].clone().detach()
                print(f'regul_embedding: replaced')
                logger.info(f'regul_embedding: replaced')


def data_test(model_config, bVisu=False, bPrint=True, bDetails=False, index_particles=0, prev_nparticles=0, new_nparticles=0,
              prev_index_particles=0, best_model=0, step=5, bTest='', folder_out='tmp_recons', initial_map='',forced_embedding=[], forced_color=0,ratio=1):

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
        model = InteractionParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
        p_mass = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        p_mass = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
        p_elec = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_elec[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if bMesh:
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p), delta_t=model_config['delta_t'],
                      clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])
        if (model_config['model'] == 'WaveMesh'):
            model_mesh = MeshLaplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
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
    if bPrint:
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
        labels = T1
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
            mask = to_numpy(torch.argwhere((x[:, 1] > 0.02) & (x[:, 1] < 0.98) & (x[:, 2] > 0.02) & (x[:, 2] < 0.98))).astype(int)
            mask = mask[:, 0:1]
            with torch.no_grad():
                pred = model_mesh(dataset_mesh,data_id=0)
                x[mask,6:9] += pred[mask] * hnorm * delta_t
        else:
            distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

            edge_index = adj_t.nonzero().t().contiguous()

            dataset = data.Data(x=x, edge_index=edge_index)


            with torch.no_grad():
                y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

            if model_config['prediction'] == '2nd_derivative':
                y = y * ynorm[4] * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm[4]
                x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

            x_recons.append(x.clone().detach())
            y_recons.append(y.clone().detach())

        if bMesh:
            mask = to_numpy(torch.argwhere((x[:, 1] < 0.025) | (x[:, 1] > 0.975) | (x[:, 2] < 0.025) | (x[:, 2] > 0.975))).astype(int)
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
                        H1_IM = torch.reshape(x0[:,6:9], (100, 100, 3))
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
                    # plt.xlim([0, 1])
                    # plt.ylim([0, 1])
                else:
                    # plt.text(-1.25, 1.5, f'frame: {it}')
                    # plt.text(-1.25, 1.4, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([-0.5, 0.5])
                    plt.ylim([-0.5, 0.5])

                # plt.xlim([0, 1])
                # plt.ylim([0, 1])

                plt.xticks([])
                plt.yticks([])

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
                    elif model_config['model'] == 'GravityParticles':
                        for n in range(nparticle_types):
                            g = to_numpy(p_mass[to_numpy(T1[index_particles[n], 0])]) * 10 * sc
                            plt.scatter(x_[index_particles[n], 1].detach().cpu(), x_[index_particles[n], 2].detach().cpu(),
                                        s=g, alpha=0.75, color=cmap.color(n))  # , facecolors='none', edgecolors='k')
                    elif model_config['model'] == 'ElecParticles':
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
                            H1_IM = torch.reshape(x_[:,6:9], (100, 100, 3))
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
                        elif not(model_config['model']=='RD_RPS_Mesh'):
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                    else:
                        if model_config['model'] == 'RD_RPS_Mesh':
                            plt.xlim([40, 60])
                            plt.ylim([40, 60])
                        elif bMesh | ('Boids' in model_config['description']) | (model_config['boundary'] == 'periodic'):
                            plt.xlim([0.3, 0.7])
                            plt.ylim([0.3, 0.7])
                        elif not(model_config['model']=='RD_RPS_Mesh'):
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


def data_plot(model_config, epoch, bPrint, best_model=0, kmeans_input='plot'):

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    nrun = model_config['nrun']
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

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    # arr = np.arange(0, NGraphs)
    # x_list=[]
    # y_list=[]
    # for run in arr:
    #     x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
    #     y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
    #     x_list.append(torch.stack(x))
    #     y_list.append(torch.stack(y))
    # x = torch.stack(x_list)
    # x = torch.reshape(x,(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    # y = torch.stack(y_list)
    # y = torch.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3]))
    # vnorm = norm_velocity(x, device)
    # ynorm = norm_acceleration(y, device)
    # torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    # torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    # print (vnorm,ynorm)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    print('Load normalizations ...')
    time.sleep(1)

    if False:  # analyse tmp_recons
        x = torch.load(f'{log_dir}/x_list.pt')
        y = torch.load(f'{log_dir}/y_list.pt')
        for k in np.arange(0, len(x) - 1, 4):
            distance = torch.sum(bc_diff(x[k][:, None, 1:3] - x[k][None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, edge_index=edge_index)
            distance = np.sqrt(to_numpy(distance[edge_index[0, :], edge_index[1, :]]))
            deg = degree(dataset.edge_index[0], dataset.num_nodes)
            deg_list.append(to_numpy(deg))
            distance_list.append([np.mean(distance), np.std(distance)])
            x_stat.append(to_numpy(torch.concatenate((torch.mean(x[k][:, 3:5], axis=0), torch.std(x[k][:, 3:5], axis=0)), axis=-1)))
            y_stat.append(to_numpy(torch.concatenate((torch.mean(y[k], axis=0), torch.std(y[k], axis=0)), axis=-1)))
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))
    else:
        for run in trange(NGraphs):
            x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
            if run == 0:
                for k in np.arange(0, len(x) - 1, 4):
                    distance = torch.sum(bc_diff(x[k][:, None, 1:3] - x[k][None, :, 1:3]) ** 2, axis=2)
                    t = torch.Tensor([radius ** 2])  # threshold
                    adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x, edge_index=edge_index)
                    distance = np.sqrt(to_numpy(distance[edge_index[0, :], edge_index[1, :]]))
                    deg = degree(dataset.edge_index[0], dataset.num_nodes)
                    deg_list.append(to_numpy(deg))
                    distance_list.append([np.mean(distance), np.std(distance)])
                    x_stat.append(to_numpy(torch.concatenate((torch.mean(x[k][:, 3:5], axis=0), torch.std(x[k][:, 3:5], axis=0)),
                                                    axis=-1)))
                    y_stat.append(to_numpy(torch.concatenate((torch.mean(y[k], axis=0), torch.std(y[k], axis=0)),
                                                    axis=-1)))
            x_list.append(torch.stack(x))
            y_list.append(torch.stack(y))

    x = torch.stack(x_list)
    x = torch.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    y = torch.stack(y_list)
    y = torch.reshape(y, (y.shape[0] * y.shape[1] * y.shape[2], y.shape[3]))
    vnorm = norm_velocity(x, device)
    ynorm = norm_acceleration(y, device)
    print(vnorm, ynorm)
    print(vnorm[4], ynorm[4])

    x_stat = np.array(x_stat)
    y_stat = np.array(y_stat)

    # fig = plt.figure(figsize=(20, 5))
    # plt.ion()
    # ax = fig.add_subplot(1, 5, 4)
    #
    # deg_list = np.array(deg_list)
    # distance_list = np.array(distance_list)
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0] + deg_list[:, 1], c='k')
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0], c='r')
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0] - deg_list[:, 1], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Degree [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 5, 1)
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0] + distance_list[:, 1], c='k')
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0], c='r')
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0] - distance_list[:, 1], c='k')
    # plt.ylim([0, model_config['radius']])
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Distance [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 5, 2)
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0] + x_stat[:, 2], c='k')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0], c='r')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0] - x_stat[:, 2], c='k')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1] + x_stat[:, 3], c='k')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1], c='r')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1] - x_stat[:, 3], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Velocity [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 5, 3)
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0] + y_stat[:, 2], c='k')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0], c='r')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0] - y_stat[:, 2], c='k')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1] + y_stat[:, 3], c='k')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1], c='r')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1] - y_stat[:, 3], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Acceleration [a.u]', fontsize="14")
    # plt.tight_layout()
    # plt.show()

    if bMesh:
        h_list = []
        for run in trange(NGraphs):
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt', map_location=device)
            h_list.append(torch.stack(h))
        h = torch.stack(h_list)
        h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(hnorm)
        model = MeshLaplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = InteractionParticles(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)
        print(f'Training InteractionParticles')

    # if best_model == -1:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    # else:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

    net = f"./log/try_{dataset_name}/models/best_model_with_{nrun - 1}_graphs_{best_model}.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

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
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f'Learning rates: {lr}, {lra}')
    print('')
    print(f'network: {net}')
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=weight_decay)
    model.eval()
    best_loss = np.inf

    print('')
    time.sleep(0.5)
    print('Plotting ...')


    if bMesh:
        x = x_list[0][0].clone().detach()
        index_particles = []
        for n in range(model_config['nparticle_types']):
            index = np.argwhere(to_numpy(x[:, 5]) == n)
            index_particles.append(index.squeeze())
    rr = torch.tensor(np.linspace(min_radius, radius, 1000)).to(device)
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

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    # cm = 1 / 2.54 * 3 / 2.3
    # plt.subplots(frameon=False)
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })
    # fig = plt.figure(figsize=(3*cm, 3*cm))

    if bMesh:
        X1 = torch.rand(nparticles, 2, device=device)
        x_width = int(np.sqrt(nparticles))
        xs = torch.linspace(0, 1, steps=x_width)
        ys = torch.linspace(0, 1, steps=x_width)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        x = torch.reshape(x, (x_width ** 2, 1))
        y = torch.reshape(y, (x_width ** 2, 1))
        x_width = 1 / x_width / 8
        X1[0:nparticles, 0:1] = x[0:nparticles]
        X1[0:nparticles, 1:2] = y[0:nparticles]
        X1 = X1 + torch.randn(nparticles, 2, device=device) * x_width
        X1_ = torch.clamp(X1, min=0, max=1)

        particle_type_map = model_config['particle_type_map']
        i0 = imread(f'graphs_data/{particle_type_map}')

        values = i0[(to_numpy(X1_[:, 0]) * 255).astype(int), (to_numpy(X1_[:, 1]) * 255).astype(int)]
        T1 = torch.tensor(values, device=device)
        T1 = T1[:, None]


    fig = plt.figure(figsize=(16, 7.6))
    plt.ion()
    ax = fig.add_subplot(2, 4, 1)
    if (embedding.shape[1] > 2):
        ax = fig.add_subplot(2, 4, 1, projection='3d')
        for n in range(nparticle_types):
            ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],color=cmap.color(n), s=1)
    else:
        if (embedding.shape[1] > 1):
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types):
                    plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                                embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=1)
            plt.xlabel(r'Embedding 0',fontsize=14)
            plt.ylabel(r'Embedding 1',fontsize=14)
        else:
            for n in range(nparticle_types):
                plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5, color=cmap.color(n))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(2, 4, 2)
    with torch.no_grad():
        if model_config['model'] == 'ElecParticles':
            acc_list = []
            for m in range(model.a.shape[0]):
                for k in range(nparticle_types):
                    for n in index_particles[k]:
                        embedding = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                        in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                                 rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding, embedding), dim=1)
                        with torch.no_grad():
                            acc = model.lin_edge(in_features.float())
                        acc = acc[:, 0]
                        acc_list.append(acc)
                        if n % 5 == 0:
                            plt.plot(to_numpy(rr),
                                     to_numpy(acc) * to_numpy(ynorm[4]),
                                     linewidth=1,
                                     color=cmap.color(k), alpha=0.25)
            acc_list = torch.stack(acc_list)
            plt.xlim([0, 0.02])
            plt.ylim([-0.5E6, 0.5E6])
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            coeff_norm = to_numpy(acc_list)
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                              random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
            proj_interaction = np.squeeze(proj_interaction)
        elif model_config['model'] == 'GravityParticles':
            acc_list = []
            for n in range(nparticles):
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                with torch.no_grad():
                    acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                acc_list.append(acc)
                plt.plot(to_numpy(rr),
                         to_numpy(acc) * to_numpy(ynorm[4]),
                         color=cmap.color(to_numpy(x[n, 5])), linewidth=1, alpha=0.25)
            acc_list = torch.stack(acc_list)
            # plt.yscale('log')
            # plt.xscale('log')
            plt.xlim([0, 0.02])
            plt.ylim([0, 0.5E6])
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            coeff_norm = to_numpy(acc_list)
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                              random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
            proj_interaction = np.squeeze(proj_interaction)
        elif (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
            acc_list = []
            for n in range(nparticles):
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                if model_config['prediction'] == '2nd_derivative':
                    in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                else:
                    in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], embedding), dim=1)
                with torch.no_grad():
                    acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                acc_list.append(acc)
                if n % 5 == 0:
                    plt.plot(to_numpy(rr),
                             to_numpy(acc) * to_numpy(ynorm[4]),
                             color=cmap.color(to_numpy(x[n, 5])), linewidth=0.1, alpha=0.25)
            acc_list = torch.stack(acc_list)
            coeff_norm = to_numpy(acc_list)
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                              random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
            proj_interaction = np.squeeze(proj_interaction)
            plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij}) [a.u.]$', fontsize=14)
        elif bMesh:
            f_list = []
            for n in range(nparticles):
                r = torch.tensor(np.linspace(-250, 250, 1000)).to(device)
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((r[:, None], embedding), dim=1)
                h = model.lin_phi(in_features.float())
                h = h[:, 0]
                f_list.append(h)
                if n % 100 == 0:
                    plt.plot(to_numpy(r),
                             to_numpy(h) * to_numpy(hnorm), linewidth=1,
                             color='k', alpha=0.05)
            f_list = torch.stack(f_list)
            coeff_norm = to_numpy(f_list)
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                              random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
            proj_interaction = np.squeeze(proj_interaction)
        if (model_config['model'] == 'PDE_B'):
            plt.xlim([0, 0.02])
            plt.ylim([-0.001, 0.00025])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    ax = fig.add_subplot(2, 4, 3)
    for n in range(nparticle_types):
        if proj_interaction.ndim == 1:
            plt.hist(proj_interaction[index_particles[n]], width=0.01, alpha=0.5, color=cmap.color(n))
        if proj_interaction.ndim == 2:
            plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                        color=cmap.color(n), s=0.1)
            plt.xlabel(r'UMAP 0', fontsize=14)
            plt.ylabel(r'UMAP 1', fontsize=14)
    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,random_state=13)
    if kmeans_input == 'plot':
        kmeans.fit(proj_interaction)
    if kmeans_input == 'embedding':
        kmeans.fit(embedding_)
    label_list = []
    for n in range(nparticle_types):
        tmp = kmeans.labels_[index_particles[n]]
        sub_group = np.round(np.median(tmp))
        label_list.append(sub_group)
        accuracy = len(np.argwhere(tmp == sub_group)) / len(tmp) * 100
        print(f'Sub-group {n} accuracy: {np.round(accuracy, 3)}')
    label_list = np.array(label_list)
    new_labels = 0* kmeans.labels_.copy()
    for n in range(nparticle_types):
        new_labels[kmeans.labels_ == label_list[n]] = n
    Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
    print(' ')
    print (f'Accuracy: {np.round(Accuracy,3)}')
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_{best_model}.pt'))
    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    t = []
    for k in range(model_config['ninteractions']):
        pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        print(torch.median(temp, axis=0).values)
        model_a_[pos, :] = torch.median(temp, axis=0).values.repeat((len(pos), 1))
        t.append(torch.median(temp, axis=0).values)
    model_a_ = torch.reshape(model_a_, (model.a.shape[0], model.a.shape[1], model.a.shape[2]))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_[0]
    embedding = []
    for n in range(model.a.shape[0]):
        embedding.append(model.a[n])
    embedding = to_numpy(torch.stack(embedding))
    embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
    embedding_particle = []
    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            embedding_particle.append(embedding[index_particles[n] + m * nparticles, :])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    ax = fig.add_subplot(2, 4, 5)
    if (embedding.shape[1] > 2):
        ax = fig.add_subplot(2, 4, 5, projection='3d')
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                ax.scatter(to_numpy(model.a[m][index_particles[n], 0]),
                           to_numpy(model.a[m][index_particles[n], 1]),
                           to_numpy(model.a[m][index_particles[n], 1]),
                           color=cmap.color(new_labels[n]), s=20)
    else:
        if (embedding.shape[1] > 1):
            for m in range(model.a.shape[0]):
                for n in range(model.a.shape[1]):
                    plt.scatter(to_numpy(model.a[m][n, 0]),
                                to_numpy(model.a[m][n, 1]),
                                color=cmap.color(new_labels[n]), s=1)
            plt.xlabel(r'Embedding 0', fontsize=14)
            plt.ylabel(r'Embedding 1', fontsize=14)
        else:
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types - 1, -1, -1):
                    plt.hist(to_numpy(model.a[m][index_particles[n], 0]), width=0.01, alpha=0.5,
                             color=cmap.color(n))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(2, 4, 6)
    if model_config['model'] == 'ElecParticles':
        t = to_numpy(model.a)
        tmean = np.ones((model_config['nparticle_types'], model_config['embedding']))
        for n in range(model_config['nparticle_types']):
            tmean[n] = np.mean(t[:, index_particles[n], :], axis=(0, 1))
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                embedding0 = torch.tensor(tmean[m], device=device) * torch.ones((1000, model_config['embedding']),
                                                                                device=device)
                embedding1 = torch.tensor(tmean[n], device=device) * torch.ones((1000, model_config['embedding']),
                                                                                device=device)
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding0, embedding1), dim=1)
                acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                plt.plot(to_numpy(rr),
                         to_numpy(acc) * to_numpy(ynorm[4]),
                         linewidth=1)
        plt.xlim([0, 0.02])
        plt.ylim([-0.5E6, 0.5E6])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    elif model_config['model'] == 'GravityParticles':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            plt.plot(to_numpy(rr),
                     to_numpy(acc) * to_numpy(ynorm[4]) ,
                     color=cmap.color(to_numpy(x[n, 5])), linewidth=1, alpha=0.25)
        acc_list = torch.stack(acc_list)
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        # plt.xlim([1E-3, 0.2])
        # plt.ylim([1, 1E7])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    elif (model_config['model'] == 'PDE_A'):
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            if model_config['prediction'] == '2nd_derivative':
                in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            else:
                in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], embedding), dim=1)
            with torch.no_grad():
                acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            if n % 5 == 0:
                plt.plot(to_numpy(rr),
                         to_numpy(acc) * to_numpy(ynorm[4]) ,
                         color=cmap.color(to_numpy(x[n, 5])), linewidth=0.1, alpha=0.25)
        plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij}) [a.u.]$', fontsize=14)
    elif (model_config['model'] == 'PDE_B'):
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            with torch.no_grad():
                acc = model.lin_edge(in_features.float())
            update_features = torch.cat((acc, acc * 0, embedding), dim=1)
            with torch.no_grad():
                acc = model.lin_update(update_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            if n % 5 == 0:
                plt.plot(to_numpy(rr),
                         to_numpy(acc) * to_numpy(ynorm[4]) ,
                         color=cmap.color(to_numpy(x[n, 5])), linewidth=1, alpha=0.25)
    elif bMesh:
        for n in range(nparticles):
            r0 = torch.tensor(np.linspace(4, 5, 1000)).to(device)
            r1 = torch.tensor(np.linspace(-100, 100, 1000)).to(device)
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((r0[:, None], r1[:, None], embedding), dim=1)
            h = model.lin_edge(in_features.float())
            h = h[:, 0]
            if n % 5 == 0:
                plt.plot(to_numpy(r1), to_numpy(h) * to_numpy(hnorm),
                         linewidth=1, color='k', alpha=0.05)
    if (model_config['model'] == 'PDE_B'):
        plt.xlim([0, 0.02])
        plt.ylim([-5E-5, 1E-5])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(2, 4, 8)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        p = model_config['p']
        if len(p) > 0:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt')

        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, p[n]))
        for n in range(nparticle_types - 1, -1, -1):
            plt.plot(to_numpy(rr), np.array(psi_output[n].cpu()), color=cmap.color(n), linewidth=1)
        plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij}) [a.u.]$', fontsize=14)
    if model_config['model'] == 'GravityParticles':
        p = model_config['p']
        if len(p) > 0:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt')
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, p[n]))
        for n in range(nparticle_types - 1, -1, -1):
            plt.plot(to_numpy(rr), np.array(psi_output[n].cpu()), linewidth=1, color=cmap.color(n))
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    if model_config['model'] == 'ElecParticles':
        p = model_config['p']
        if len(p) > 0:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt')
        psi_output = []
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                temp = model.psi(rr, p[n], p[m])
                plt.plot(to_numpy(rr), np.array(temp.cpu()), linewidth=1)
        plt.xlim([0, 0.02])
        plt.ylim([-0.5E6, 0.5E6])
    if bMesh:
        for n in range(nparticle_types):
            plt.scatter(to_numpy(x[index_particles[n]]),
                        to_numpy(y[index_particles[n]]), s=10)
    if (model_config['model'] == 'PDE_B'):
        plt.xlim([0, 0.02])
        plt.ylim([-5E-5, 1E-5])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    ax = fig.add_subplot(2, 4, 4)
    confusion_matrix = metrics.confusion_matrix(to_numpy(T1), new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if nparticle_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d')
    Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
    Precision = metrics.precision_score(to_numpy(T1), new_labels, average='micro')
    Recall = metrics.recall_score(to_numpy(T1), new_labels, average='micro')
    F1 = metrics.f1_score(to_numpy(T1), new_labels, average='micro')
    plt.text(0, -0.5, r"F1: {:.3f}".format(F1), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(log_dir, 'result.png'), dpi=300)


    # Post analysis of interaction function plots

    if model_config['model'] == 'ElecParticles':

        plot_list_pairwise = []
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                embedding0 = torch.tensor(tmean[m], device=device) * torch.ones((1000, model_config['embedding']),
                                                                                device=device)
                embedding1 = torch.tensor(tmean[n], device=device) * torch.ones((1000, model_config['embedding']),
                                                                                device=device)
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding0, embedding1), dim=1)
                with torch.no_grad():
                    pred = model.lin_edge(in_features.float())
                pred = pred[:, 0]
                plot_list_pairwise.append(pred * ynorm[4])

        p = [2, 1, -1]
        popt_list = []
        ptrue_list = []
        nn = 0
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                if plot_list_pairwise[nn][10] < 0:
                    popt, pocv = curve_fit(func_pow, to_numpy(rr),
                                           -to_numpy(plot_list_pairwise[nn]), bounds=([0, 1.5], [5., 2.5]))
                    popt[0] = -popt[0]
                else:
                    popt, pocv = curve_fit(func_pow, to_numpy(rr),
                                           to_numpy(plot_list_pairwise[nn]), bounds=([0, 1.5], [5., 2.5]))
                nn += 1
                popt_list.append(popt)
                ptrue_list.append(-p[n] * p[m])
        popt_list = -np.array(popt_list)
        ptrue_list = -np.array(ptrue_list)

        fig = plt.figure(figsize=(16, 4))

        ax = fig.add_subplot(1, 4, 2)
        plt.scatter(ptrue_list, popt_list[:, 0], color='k')
        x_data = ptrue_list
        y_data = popt_list[:, 0]
        lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
        plt.plot(ptrue_list, func_lin(x_data, lin_fit[0], lin_fit[1]), color='r')
        plt.xlabel('True q_i.q_j [a.u.]', fontsize=12)
        plt.ylabel('Predicted q_i.q_j [a.u.]', fontsize=12)
        plt.text(-2, 4, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
        residuals = y_data - func_lin(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.text(-2, 3.5, f"R2: {np.round(r_squared, 3)}", fontsize=10)
        ax = fig.add_subplot(1, 4, 3)
        plt.scatter(ptrue_list, -popt_list[:, 1], color='k')
        plt.ylim([0, 4])
        plt.xlabel('True q_i.q_j [a.u.]', fontsize=12)
        plt.ylabel('Power fit [a.u.]', fontsize=12)
        plt.text(-2, 3.5, f"{np.round(np.mean(popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
                 fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(log_dir, 'electrostatic_result.png'), dpi=300)
        plt.close()

    elif not(bMesh):
        plot_list = []
        for n in range(nparticle_types):
            embedding = t[int(label_list[n])] * torch.ones((1000, model_config['embedding']), device=device)
            if model_config['prediction'] == '2nd_derivative':
                in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            else:
                in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], embedding), dim=1)
            with torch.no_grad():
                pred = model.lin_edge(in_features.float())
            pred = pred[:, 0]
            plot_list.append(pred * ynorm[4] )

    if model_config['model'] == 'GravityParticles':
        p = np.linspace(0.5, 5, nparticle_types)
        popt_list = []
        for n in range(nparticle_types):
            popt, pcov = curve_fit(func_pow, to_numpy(rr), to_numpy(plot_list[n]))
            popt_list.append(popt)
        popt_list = np.array(popt_list)

        plot_list_2 = []
        vv = torch.tensor(np.linspace(0, 2, 100)).to(device)
        r_list = np.linspace(0.002, 0.01, 5)
        for r_ in r_list:
            rr_ = r_ * torch.tensor(np.ones((vv.shape[0], 1)), device=device)
            embedding = t[int(label_list[5])] * torch.ones((100, model_config['embedding']), device=device)
            in_features = torch.cat((rr_ / model_config['radius'], 0 * rr_,
                                     rr_ / model_config['radius'], vv[:, None], vv[:, None], vv[:, None], vv[:, None],
                                     embedding), dim=1)
            with torch.no_grad():
                pred = model.lin_edge(in_features.float())
            pred = pred[:, 0]
            plot_list_2.append(pred * ynorm[4] )

        fig = plt.figure(figsize=(16, 4))
        plt.ion()
        ax = fig.add_subplot(1, 4, 1)
        for n in range(len(r_list)):
            plt.plot(to_numpy(vv), to_numpy(plot_list_2[n]), linewidth=1, color=cmap.color(n),
                     label=f'r={r_list[n]}')
        plt.xlabel('Normalized Velocity [a.u.]', fontsize=12)
        plt.ylabel('MLP [a.u.]', fontsize=12)
        plt.xlim([0, 2])
        plt.legend()

        ax = fig.add_subplot(1, 4, 2)
        plt.scatter(p, popt_list[:, 0], color='k')
        x_data = p
        y_data = popt_list[:, 0]
        lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
        plt.plot(p, func_lin(x_data, lin_fit[0], lin_fit[1]), color='r')
        plt.xlabel('True mass [a.u.]', fontsize=12)
        plt.ylabel('Predicted mass [a.u.]', fontsize=12)
        plt.xlim([0, 5.5])
        plt.ylim([0, 5.5])
        plt.text(0.5, 5, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
        residuals = y_data - func_lin(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.text(0.5, 4.5, f"R2: {np.round(r_squared, 2)}", fontsize=10)
        ax = fig.add_subplot(1, 4, 3)
        plt.scatter(p, popt_list[:, 1], color='k')
        plt.xlim([0, 5.5])
        plt.ylim([0, 4])
        plt.xlabel('True mass [a.u.]', fontsize=12)
        plt.ylabel('Power fit [a.u.]', fontsize=12)
        plt.text(0.5, 3.5, f"{np.round(np.mean(popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
                 fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(log_dir, 'gravity_result.png'), dpi=300)
        plt.close()

    if model_config['model'] != 'ElecParticles':
        rmserr_list = []
        for n in range(nparticle_types):
            min_norm = torch.min(plot_list[n])
            max_norm = torch.max(plot_list[n])
            if torch.min(plot_list[n]) < min_norm:
                min_norm = torch.min(plot_list[n])
            if torch.max(psi_output[n]) > max_norm:
                max_norm = torch.max(psi_output[n])
            plot_list[n] = (plot_list[n] - min_norm) / (max_norm - min_norm)
            psi_output[n] = (psi_output[n] - min_norm) / (max_norm - min_norm)
            rmserr = torch.sqrt(torch.mean((plot_list[n] - torch.squeeze(psi_output[n])) ** 2))
            rmserr_list.append(rmserr.item())
            print(f'sub-group {n}: RMSE: {rmserr.item()}')

        print(f'RMSE: {np.mean(rmserr_list)}+\-{np.std(rmserr_list)} ')


def data_train_shrofflab_celegans(model_config):
    print('')

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
    embedding = model_config['embedding']
    batch_size = model_config['batch_size']
    batch_size = 1
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    bRegul = 'regul' in model_config['sparsity']
    bReplace = 'replace' in model_config['sparsity']
    Nepochs = model_config['Nepochs']
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

    # training file management ###

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

    files = glob.glob(f"{log_dir}/tmp_training/*")
    for f in files:
        os.remove(f)
    files = glob.glob(f"{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(model_config)

    # load dataset ###

    print('load dataset ...')

    x_list = []
    y_list = []
    dataset, time_points, cell_names = load_shrofflab_celegans(model_config['input_dataset'], device=device)
    x_list.append(dataset)
    y = []
    for t in range(time_points.shape[0] - 1):
        x_prev = dataset[t]
        x_next = dataset[t + 1]
        id_prev = x_prev[:, 0]
        id_next = x_next[:, 0]
        y_ = []
        for id in id_prev:
            if id in id_next:
                y_.append(x_next[id_next == id, :] - x_prev[id_prev == id, :])
            else:
                y_.append(torch.nan(x_prev.shape[0], device=device))
        y.append(torch.stack(y_).squeeze())
    y_list.append(y)

    NGraphs = len(x_list)
    print(f'Graph files N: {NGraphs}')
    logger.info(f'Graph files N: {NGraphs}')
    model_config['ndataset'] = NGraphs

    # normalization

    print('normalization ...')

    t = []
    Ncells = 0
    nframes = np.zeros(NGraphs)
    for n in range(NGraphs):
        for k in trange(len(x_list[n])):
            nframes[n] = int(len(x_list[n]))
            t_ = x_list[n][k]
            n_actual_cells = to_numpy(torch.max(t_[:, 0]))
            if n_actual_cells > Ncells:
                Ncells = n_actual_cells
                model_config['nparticles'] = int(Ncells)
            if t == []:
                t = t_
            else:
                t = torch.concatenate((t, t_), axis=0)
    nframes = nframes.astype(int)
    t = torch.nan_to_num(t, nan=0)
    xnorm = torch.max(torch.abs(t[:, 1:4]))
    vnorm = torch.std(torch.abs(t[:, 4:7]))
    t = []
    for n in range(NGraphs):
        for k in trange(len(y_list[n])):
            t_ = y_list[n][k]
            if t == []:
                t = t_
            else:
                t = torch.concatenate((t, t_), axis=0)
    t = torch.nan_to_num(t, nan=0)
    ynorm = torch.std(torch.abs(t[:, 4:7]))

    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    print(to_numpy(xnorm), to_numpy(vnorm), to_numpy(ynorm))
    logger.info(
        f'xnorm vnorm ynorm: {to_numpy(xnorm), to_numpy(vnorm), to_numpy(ynorm)}')

    model = InteractionCElegans(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs}_graphs.pt"
    print(f'network: {net}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {Nepochs}')
    print('')

    model.train()
    best_loss = np.inf
    list_loss = []
    embedding_center = []
    regul_embedding = 0

    print('Start training ...')
    logger.info("Start training ...")
    time.sleep(0.5)

    data_augmentation = False

    for epoch in range(Nepochs + 1):

        if epoch == 0:
            batch_size = model_config['batch_size']
            print(f'batch_size: {batch_size}')
            logger.info(f'batch_size: {batch_size}')
        if epoch == 0:
            if data_augmentation:
                data_augmentation_loop = 200
                print(f'data_augmentation_loop: {data_augmentation_loop}')
                logger.info(f'data_augmentation_loop: {data_augmentation_loop}')
        if epoch == 3 * Nepochs // 4:
            lra = 1E-3
            lr = 5E-4
            table = PrettyTable(["Modules", "Parameters"])
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
        if epoch == Nepochs - 2:
            print('not training embedding ...')
            logger.info('not training embedding ...')
            model.a.requires_grad = False
            regul_embedding = 0

        total_loss = 0

        for N in trange(0, nframes[0] // batch_size * 10):

            run = np.random.randint(NGraphs)

            dataset_batch = []
            mask_batch = []
            time_batch = []

            for batch in range(batch_size):

                k = np.random.randint(nframes[run] - 2)
                x = x_list[run][k].clone().detach()
                x = torch.nan_to_num(x, nan=0)

                x[:, 1:4] = x[:, 1:4] / xnorm

                distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
                t = torch.Tensor([radius ** 2])
                edges = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)
                y = y_list[run][k].clone().detach()

                mask = torch.isnan(y[:, 0])

                mask = 1 - mask.long()

                if batch == 0:
                    mask_batch = mask
                    time_batch = torch.tensor(k, device=device)

                else:
                    mask_batch = torch.cat((mask_batch, mask), axis=0)
                    time_batch = torch.cat((time_batch, torch.tensor(k, device=device)), axis=0)

                y = torch.nan_to_num(y, nan=0)
                if model_config['prediction'] == '2nd_derivative':
                    y = y[:, 4:7] / ynorm
                else:
                    y = y[:, 1:4] / vnorm
                if batch == 0:
                    y_batch = y
                else:
                    y_batch = torch.cat((y_batch, y), axis=0)

            if dataset_batch != []:

                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                optimizer.zero_grad()

                for k, batch in enumerate(batch_loader):
                    pred = model(batch, data_id=run, time=time_batch[k])

                mask_batch = mask_batch[:, None].repeat(1, 3)
                loss = (mask_batch * (pred - y_batch)).norm(2)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # optimizer.zero_grad()
            # t = torch.sum(model.a[run])
            # loss = (pred - y_batch).norm(2) + t
            # loss.backward()
            # optimizer.step()
            # total_loss += loss.item()

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        if (total_loss / batch_size / (N + 1) < best_loss):
            best_loss = total_loss / (N + 1) / batch_size
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs.pt'))
            print("Epoch {}. Loss: {:.6f} saving model  ".format(epoch, total_loss / (N + 1) / batch_size))
            logger.info("Epoch {}. Loss: {:.6f} saving model  ".format(epoch, total_loss / (N + 1) / batch_size))
        else:
            print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / batch_size))
            logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / batch_size))

        list_loss.append(total_loss / (N + 1) / nparticles / batch_size)

        fig = plt.figure(figsize=(16, 4))
        # plt.ion()

        ax = fig.add_subplot(1, 4, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, Nepochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        embedding = []
        for n in range(model.a.shape[0]):
            embedding.append(model.a[n])
        embedding = to_numpy(torch.stack(embedding))
        embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])

        ax = fig.add_subplot(1, 4, 2)
        if (embedding.shape[1] > 2):
            ax = fig.add_subplot(2, 4, 2, projection='3d')
            ax.scatter(embedding[:, 0], embedding[n][:, 1], embedding[n][:, 2], color='k', s=1)
        else:
            if (embedding.shape[1] > 1):
                for m in range(model.a.shape[0]):
                    plt.scatter(embedding[:, 0], embedding[:, 1], color='k', s=3)
                plt.xlabel('Embedding 0', fontsize=12)
                plt.ylabel('Embedding 1', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif", dpi=300)
        plt.close()


def data_test_shrofflab_celegans(model_config):
    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
    embedding = model_config['embedding']
    batch_size = model_config['batch_size']
    batch_size = 1
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    bRegul = 'regul' in model_config['sparsity']
    bReplace = 'replace' in model_config['sparsity']
    Nepochs = model_config['Nepochs']
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

    # load dataset ###

    print('load dataset ...')

    x_list = []
    y_list = []
    dataset, time_points, cell_names = load_shrofflab_celegans(model_config['input_dataset'], device=device)
    x_list.append(dataset)
    y = []
    for t in trange(time_points.shape[0] - 1):
        x_prev = dataset[t]
        x_next = dataset[t + 1]
        id_prev = x_prev[:, 0]
        id_next = x_next[:, 0]
        y_ = []
        for id in id_prev:
            if id in id_next:
                y_.append(x_next[id_next == id, :] - x_prev[id_prev == id, :])
            else:
                y_.append(torch.nan(x_prev.shape[0], device=device))
        y.append(torch.stack(y_).squeeze())
    y_list.append(y)

    xnorm = torch.load(f'./log/try_{dataset_name}/xnorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)
    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)

    NGraphs = len(x_list)
    print(f'Graph files N: {NGraphs}')
    model_config['ndataset'] = NGraphs

    t = []
    Ncells = 0
    nframes = np.zeros(NGraphs)
    for n in range(NGraphs):
        for k in trange(len(x_list[n])):
            nframes[n] = int(len(x_list[n]))
            t_ = x_list[n][k]
            if to_numpy(torch.max(t_[:, 0])) > Ncells:
                Ncells = to_numpy(torch.max(t_[:, 0]))
                model_config['nparticles'] = int(Ncells)
    nframes = nframes.astype(int)
    t = []

    # set up model ###

    print('set up model ...')

    model = InteractionCElegans(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    run = 0
    mina = to_numpy(torch.min(model.a))
    maxa = to_numpy(torch.max(model.a))
    error_list = []

    for k in trange(nframes[run] - 2):
        x = x_list[run][k].clone().detach()
        x = torch.nan_to_num(x, nan=0)

        x[:, 1:4] = x[:, 1:4] / xnorm
        embedding = model.a[run, to_numpy(x[:, 0]).astype(int), :]

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
        t = torch.Tensor([radius ** 2])
        edges = adj_t.nonzero().t().contiguous()
        dataset = data.Data(x=x[:, :], edge_index=edges)
        y = y_list[run][k].clone().detach()
        mask = torch.isnan(y[:, 0])
        mask = 1 - mask.long()
        y = torch.nan_to_num(y, nan=0)
        if model_config['prediction'] == '2nd_derivative':
            y = y[:, 4:7] / ynorm
        else:
            y = y[:, 1:4] / vnorm
        pred = model(dataset, data_id=run, time=torch.tensor(k, device=device))

        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 3]), to_numpy(x[:, 2]),
                   c=to_numpy(embedding[:, 1]), alpha=1, vmin=mina, vmax=maxa)
        ax.set_aspect('equal')
        # remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.title(f'Time: {np.round(time_points[k])}     LUT=Network embedding', fontsize=10)
        plt.tight_layout()
        # ax = fig.add_subplot(1, 3, 2, projection='3d')
        # ax.scatter(x[:, 1].detach().cpu().numpy(), x[:, 3].detach().cpu().numpy(), x[:, 2].detach().cpu().numpy(), c=embedding[:,0].detach().cpu().numpy() ,alpha=1,vmin=mina,vmax=maxa)
        # ax.set_aspect('equal')
        # plt.tight_layout()
        ax = fig.add_subplot(1, 3, 2)
        plt.plot(to_numpy(y[:, 0]), to_numpy(pred[:, 0]), 'o', color='b', markersize=1)
        plt.plot(to_numpy(y[:, 1]), to_numpy(pred[:, 1]), 'o', color='g', markersize=1)
        if model_config['prediction'] == '1st_derivative':
            plt.xlabel('True velocity [a.u.]', fontsize=12)
            plt.ylabel('Predicted velocity [a.u.]', fontsize=12)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
        else:
            plt.xlabel('True acceleration [a.u.]', fontsize=12)
            plt.ylabel('Predicted acceleration [a.u.]', fontsize=12)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
        ax = fig.add_subplot(1, 3, 3)
        error_list.append(100 * to_numpy(torch.sqrt(torch.mean((pred - y) ** 2))))
        plt.plot(time_points[0:len(error_list)], error_list, color='k')
        plt.xlim([time_points[0], time_points[-1]])
        plt.ylim([0, 10])
        plt.xlabel('Time [a.u.]', fontsize=12)
        plt.ylabel('Error/S.D. [%]', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"./log/try_{dataset_name}/tmp_recons/Fig_{dataset_name}_{k}.tif", dpi=300)
        plt.close()

    # pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
    # distance = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
    # adj_t = (distance < radius ** 2).float() * 1
    # edge_index = adj_t.nonzero().t().contiguous()
    # dataset = data.Data(x=x, edge_index=edge_index)
    # vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
    # nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.005


if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    # config_manager = create_config_manager(config_type='simulation')

    config_manager = ConfigManager(config_schema='./config_schemas/config_schema_simulation.yaml')

    # config_list=['config_CElegans_32']

    # config_list = ['config_arbitrary_3'] #,'config_gravity_16','config_arbitrary_16']
    # config_list = ['config_arbitrary_16_HR','config_gravity_16_001']
    # config_list = ['config_gravity_16_001_HR','config_gravity_16_001']
    # config_list = ['config_Coulomb_3_HR']
    # config_list = ['config_boids_16_HR']
    # config_list = ['config_wave_testA']

    # Test plotting figures paper
    config_list = ['config_arbitrary_3_test'] # ,'config_boids_16_HR8','config_boids_16_HR9']# ['config_boids_16_HR7','config_boids_16_HR8','config_boids_16_HR9']
    
    # Load a graph neural network model used to sparsify the particle embedding during training
    model_config_embedding = config_manager.load_and_validate_config('./config/config_embedding.yaml')
    p = torch.ones(1, 4, device=device)
    p[0] = torch.tensor(model_config_embedding['p'][0])
    model_embedding = PDE_embedding(aggr_type='mean', p=p, delta_t=model_config_embedding['delta_t'],
                                    sigma=model_config_embedding['sigma'],
                                    prediction=model_config_embedding['prediction'], device=device)
    model_embedding.eval()

    for config in config_list:

        # Load parameters from config file
        # model_config = load_model_config(id=config)
        model_config = config_manager.load_and_validate_config(f'./config/{config}.yaml')
        model_config['dataset']=config[7:]

        for key, value in model_config.items():
            print(key, ":", value)
            if ('E-' in str(value)) | ('E+' in str(value)):
                value = float(value)
                model_config[key] = value

        cmap = cc(model_config=model_config)

        data_generate(model_config, device=device, bVisu=True, bStyle='bw', alpha=0.2, bErase=True, bLoad_p=False, step=model_config['nframes']//20, ratio=1, scenario='none' )
        data_train(model_config,model_embedding)
        # data_plot(model_config, epoch=-1, bPrint=True, best_model=4, kmeans_input=model_config['kmeans_input'])
        # data_test(model_config, bVisu=True, bPrint=True, best_model=20, bDetails=False, step = model_config['nframes']//200, ratio=1)

        # data_train_shrofflab_celegans(model_config)
        # data_test_shrofflab_celegans(model_config)

        # x, rmserr_list = data_test(model_config, bVisu=True, bPrint=True, best_model=-1, step=10, bTest='',initial_map='', forced_embedding=[1.265,0.636], forced_color=0)

