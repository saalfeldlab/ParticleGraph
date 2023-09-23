
# use of https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb

import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import networkx as nx
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import glob
import torch_geometric as pyg
import torch_geometric.data as data
import math
import torch_geometric.utils as pyg_utils
import torch.nn as nn
from torch.nn import functional as F
import time
from shutil import copyfile
from prettytable import PrettyTable

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import csv
import json
from geomloss import SamplesLoss
from tifffile import imread


def distmat_square(X, Y):
    return torch.sum(bc_diff(X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

def kernel(X, Y):
    return -torch.sqrt(distmat_square(X, Y))

def MMD(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.sum(kernel(X, X)) / n ** 2 + \
        torch.sum(kernel(Y, Y)) / m ** 2 - \
        2 * torch.sum(kernel(X, Y)) / (n * m)
    return a.item()

def psi(r, p):
    return -p[2] * torch.exp(-r ** p[0] / (2 * sigma ** 2)) + p[3] * torch.exp(-r ** p[1] / (2 * sigma ** 2))

def normalize99(Y, lower=1, upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return x01, x99

def norm_velocity(xx, device):
    mvx = torch.mean(xx[:, 2])
    mvy = torch.mean(xx[:, 3])
    vx = torch.std(xx[:, 2])
    vy = torch.std(xx[:, 3])
    nvx = np.array(xx[:, 2].detach().cpu())
    vx01, vx99 = normalize99(nvx)
    nvy = np.array(xx[:, 3].detach().cpu())
    vy01, vy99 = normalize99(nvy)

    # print(f'v_x={mvx} +/- {vx}')
    # print(f'v_y={mvy} +/- {vy}')
    # print(f'vx01={vx01} vx99={vx99}')
    # print(f'vy01={vy01} vy99={vy99}')

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

    # print(f'acc_x={max} +/- {ax}')
    # print(f'acc_y={may} +/- {ay}')
    # print(f'ax01={ax01} ax99={ax99}')
    # print(f'ay01={ay01} ay99={ay99}')

    return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)
def norm_velocity3D(xx, device):
    mvx = torch.mean(xx[:, 2])
    mvy = torch.mean(xx[:, 3])
    vx = torch.std(xx[:, 2])
    vy = torch.std(xx[:, 3])
    vz = torch.std(xx[:, 4])

    nvx = np.array(xx[:, 2].detach().cpu())
    vx01, vx99 = normalize99(nvx)
    nvy = np.array(xx[:, 3].detach().cpu())
    vy01, vy99 = normalize99(nvy)

    # print(f'v_x={mvx} +/- {vx}')
    # print(f'v_y={mvy} +/- {vy}')
    # print(f'vx01={vx01} vx99={vx99}')
    # print(f'vy01={vy01} vy99={vy99}')

    return torch.tensor([vx01, vx99, vy01, vy99, vx, vy, vz], device=device)
def norm_acceleration3D(yy, device):
    max = torch.mean(yy[:, 0])
    may = torch.mean(yy[:, 1])
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    az = torch.std(yy[:, 2])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = normalize99(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = normalize99(nay)

    # print(f'acc_x={max} +/- {ax}')
    # print(f'acc_y={may} +/- {ay}')
    # print(f'ax01={ax01} ax99={ax99}')
    # print(f'ay01={ay01} ay99={ay99}')

    return torch.tensor([ax01, ax99, ay01, ay99, ax, ay, az], device=device)

class Embedding_freq(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding_freq, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
class InteractionParticles_0(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_0, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.tau * self.propagate(edge_index, x=(x, x))
        oldv = x[:, 2:4]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)  # squared distance

        # psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))
        pp = self.p[x_i[:, 5].detach().cpu().numpy(),:]
        psi = -pp[:,2] * torch.exp(-r ** pp[:,0] / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** pp[:,1] / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2])
class InteractionParticles_1(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_1, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.tau * self.propagate(edge_index, x=(x, x))
        oldv = x[:, 2:4]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)  # squared distance

        # psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))

        pp = torch.squeeze(self.p[x_i[:, 5].detach().cpu().numpy(),x_j[:, 5].detach().cpu().numpy(),:])
        psi = -pp[:,2] * torch.exp(-r ** pp[:,0] / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** pp[:,1] / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2])
class InteractionParticles_2(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_2, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.tau * self.propagate(edge_index, x=(x, x))
        oldv = x[:, 3:6]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:, 0:3] - x_j[:, 0:3]) ** 2, axis=1)  # squared distance

        # psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))

        pp = self.p[x_i[:, 6].detach().cpu().numpy(),:]
        psi = -pp[:,2] * torch.exp(-r ** pp[:,0] / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** pp[:,1] / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:3] - x_j[:, 0:3])
class InteractionParticles_3(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_3, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.tau * self.propagate(edge_index, x=(x, x))
        oldv = x[:, 3:6]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:, 0:3] - x_j[:, 0:3]) ** 2, axis=1)  # squared distance

        # psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))

        pp = torch.squeeze(self.p[x_i[:, 6].detach().cpu().numpy(),x_j[:, 6].detach().cpu().numpy(),:])
        psi = -pp[:,2] * torch.exp(-r ** pp[:,0] / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** pp[:,1] / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:3] - x_j[:, 0:3])
class MLP(nn.Module):

    def __init__(self, input_size, output_size, nlayers, hidden_size, device):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(InteractionParticles, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.particle_embedding = model_config['particle_embedding']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.noise_type = model_config['noise_type']
        self.embedding_type = model_config['embedding_type']
        self.embedding = model_config['embedding']
        num_t_freq = 2
        self.embedding_freq = Embedding_freq(2, num_t_freq)
        self.upgrade_type = model_config['upgrade_type']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)
        self.lin_acc = MLP(input_size=4+self.embedding, output_size=self.output_size, nlayers=3,
                            hidden_size=self.hidden_size, device=self.device)

        if self.embedding_type == 'none':
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), self.embedding)), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 2)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        self.a_bf_kmean = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device='cuda:0', requires_grad=False))


    def forward(self, data, step, vnorm, cos_phi, sin_phi):

        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        acc = self.propagate(edge_index, x=(x, x))

        if self.upgrade_type == 1:
            embedding = self.a[x[:, 6].detach().cpu().numpy(), :]
            x_vx = x[:, 2:3] / self.vnorm[4]
            x_vy = x[:, 3:4] / self.vnorm[5]
            if (self.data_augmentation) & (self.step == 1):
                new_vx = self.cos_phi * x_vx + self.sin_phi * x_vy
                new_vy = -self.sin_phi * x_vx + self.cos_phi * x_vy
                x_vx = new_vx
                x_vy = new_vy
            acc = self.lin_acc(torch.cat((acc, x_vx, x_vy, embedding), dim=-1))

        if step == 2:
            deg = pyg_utils.degree(edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)
            return deg * acc

        else:

            return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) / self.radius
        x_i_vx = x_i[:, 2:3] / self.vnorm[4]
        x_i_vy = x_i[:, 3:4] / self.vnorm[5]
        x_i_type = x_i[:, 4:6]
        x_j_vx = x_j[:, 2:3] / self.vnorm[4]
        x_j_vy = x_j[:, 3:4] / self.vnorm[5]

        if (self.data_augmentation) & (self.step==1):

            new_x = self.cos_phi * delta_pos[:,0] + self.sin_phi * delta_pos[:,1]
            new_y = -self.sin_phi * delta_pos[:,0] + self.cos_phi * delta_pos[:,1]
            delta_pos[:,0] = new_x
            delta_pos[:,1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        if self.particle_embedding > 0:
            if self.embedding_type=='none':
                embedding = self.a[x_i[:, 6].detach().cpu().numpy(), :]
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding),dim=-1)
            if self.embedding_type=='repeat':
                x_i_type_0 = x_i[:, 4]
                x_i_type_1 = x_i[:, 5]
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type_0[:, None].repeat(1, 4), x_i_type_1[:, None].repeat(1, 4)),dim=-1)
            if self.embedding_type=='frequency':
                embedding=self.embedding_freq(x_i[:, 4:6])
                embedding=embedding[:,0:8]
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding),dim=-1)

        else :
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type),dim=-1)


        return self.lin_edge(in_features)


    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)
class MixInteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(MixInteractionParticles, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.particle_embedding = model_config['particle_embedding']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.noise_type = model_config['noise_type']
        self.embedding_type = model_config['embedding_type']
        self.embedding = model_config['embedding']
        num_t_freq = 2
        self.embedding_freq = Embedding_freq(2, num_t_freq)
        self.upgrade_type = model_config['upgrade_type']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)
        self.lin_acc = MLP(input_size=4+self.embedding, output_size=self.output_size, nlayers=3,
                            hidden_size=16, device=self.device)

        if self.embedding_type == 'none':
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), self.embedding)), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 2)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))


    def forward(self, data, step, vnorm, cos_phi, sin_phi):

        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        if self.upgrade_type == 1:
            embedding = self.a[x[:, 6].detach().cpu().numpy(), :]
            x_vx = x[:, 2:3] / self.vnorm[4]
            x_vy = x[:, 3:4] / self.vnorm[5]
            if (self.data_augmentation) & (self.step == 1):
                new_vx = self.cos_phi * x_vx + self.sin_phi * x_vy
                new_vy = -self.sin_phi * x_vx + self.cos_phi * x_vy
                x_vx = new_vx
                x_vy = new_vy
            acc = self.lin_acc(torch.cat((acc, x_vx, x_vy, embedding), dim=-1))

        if step == 2:
            deg = pyg_utils.degree(edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)

            return deg * acc

        else:

            return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) / self.radius
        x_i_vx = x_i[:, 2:3] / self.vnorm[4]
        x_i_vy = x_i[:, 3:4] / self.vnorm[5]
        x_i_type = x_i[:, 4:6]
        x_j_vx = x_j[:, 2:3] / self.vnorm[4]
        x_j_vy = x_j[:, 3:4] / self.vnorm[5]

        if (self.data_augmentation) & (self.step==1):

            new_x = self.cos_phi * delta_pos[:,0] + self.sin_phi * delta_pos[:,1]
            new_y = -self.sin_phi * delta_pos[:,0] + self.cos_phi * delta_pos[:,1]
            delta_pos[:,0] = new_x
            delta_pos[:,1] = new_y
            new_x = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_y = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_x
            x_i_vy = new_y
            new_x = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_y = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_x
            x_j_vy = new_y

        if self.particle_embedding>0:

            if self.embedding_type=='none':
                embedding0 = self.a[x_i[:, 6].detach().cpu().numpy(), :]
                embedding1 = self.a[x_j[:, 6].detach().cpu().numpy(), :]
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding0,embedding1),dim=-1)
            if self.embedding_type=='repeat':
                x_i_type_0 = self.a[x_i[:, 6].detach().cpu().numpy(), 4]
                x_i_type_1 = self.a[x_j[:, 6].detach().cpu().numpy(), 5]
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type_0[:, None].repeat(1, 4), x_i_type_1[:, None].repeat(1, 4), x_j_type_0[:, None].repeat(1, 4), x_j_type_1[:, None].repeat(1, 4)),dim=-1)
            if self.embedding_type=='frequency':
                embedding=self.embedding_freq(self.a[x_i[:, 6].detach().cpu().numpy(), 0:2])
                embedding=embedding[:,0:8]
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding),dim=-1)

        else :

            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type, x_j_type),dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)
class InteractionParticles3D(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(InteractionParticles3D, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.particle_embedding = model_config['particle_embedding']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.noise_type = model_config['noise_type']
        self.embedding_type = model_config['embedding_type']
        num_t_freq = 2
        self.embedding_freq = Embedding_freq(2, num_t_freq)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)
        if self.embedding_type == 'none':
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 3)), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 2)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.a_bf_kmean = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 2)), device=self.device, requires_grad=False))
        self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))

    def forward(self, data, step, vnorm, cos_phi, sin_phi):

        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        if step == 2:
            deg = pyg_utils.degree(edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)

            return deg * acc

        else:
            return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:3] - x_j[:, 0:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 0:3] - x_j[:, 0:3]) / self.radius

        x_i_vx = x_i[:, 2:3] / self.vnorm[4]
        x_i_vy = x_i[:, 3:4] / self.vnorm[5]
        x_i_vz = x_i[:, 4:5] / self.vnorm[6]
        x_j_vx = x_j[:, 2:3] / self.vnorm[4]
        x_j_vy = x_j[:, 3:4] / self.vnorm[5]
        x_j_vz = x_j[:, 4:5] / self.vnorm[6]

        if (self.data_augmentation) & (self.step==1):

            new_x = self.cos_phi * delta_pos[:,0] + self.sin_phi * delta_pos[:,1]
            new_y = -self.sin_phi * delta_pos[:,0] + self.cos_phi * delta_pos[:,1]
            delta_pos[:,0] = new_x
            delta_pos[:,1] = new_y
            new_x = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_y = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_x
            x_i_vy = new_y
            new_x = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_y = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_x
            x_j_vy = new_y

        if self.particle_embedding>0:

            embedding0 = self.a[x_i[:, 8].detach().cpu().numpy(), :]
            embedding1 = self.a[x_j[:, 8].detach().cpu().numpy(), :]
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding0, embedding1), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)
class InteractionParticlesLoop(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(InteractionParticlesLoop, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']

        self.noise_level = model_config['noise_level']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        # self.particle_emb = MLP(input_size=2, hidden_size=8, output_size=8, nlayers=3, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device=self.device, requires_grad=True))

        self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))

    def forward(self, data, nframes):

        x, edge_index = data.x, data.edge_index
        x[:, 4:6] = self.a[x[:, 6].detach().cpu().numpy(), 0:2]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)


        for loop in range(nframes):

            distance = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)

            acc = self.propagate(edge_index, x=(x, x))

            acc[:, 0] = acc[:, 0] * ynorm[4]
            acc[:, 1] = acc[:, 1] * ynorm[5]

            x[:, 2:4] = x[:, 2:4] + acc
            x[:, 0:2] = x[:, 0:2] + x[:, 2:4]

        return x[:,0:4]

        plt.ion()
        for n in range(nparticle_types):
            plt.scatter(x[index_particles[n], 0].detach().cpu(), x[index_particles[n], 1].detach().cpu(), s=3)

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) / radius
        x_i_vx = x_i[:, 2:3] / vnorm[4]
        x_i_vy = x_i[:, 3:4] / vnorm[5]
        x_i_type = x_i[:, 4:6]
        x_j_vx = x_j[:, 2:3] / vnorm[4]
        x_j_vy = x_j[:, 3:4] / vnorm[5]

        if particle_embedding:

            x_i_type_0 = x_i[:, 4]
            x_i_type_1 = x_i[:, 5]
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type_0[:, None].repeat(1, 4), x_i_type_1[:, None].repeat(1, 4)),dim=-1)

        else :

            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type),dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)
class EdgeNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self):
        super().__init__(aggr=aggr_type)  # "mean" aggregation.

    def forward(self, x, edge_index):
        aggr = self.propagate(edge_index, x=(x, x))

        return self.new_edges

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum((x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos = (x_i[:, 0:2] - x_j[:, 0:2]) / radius
        x_i_vx = x_i[:, 2:3] / vnorm[4]
        x_i_vy = x_i[:, 3:4] / vnorm[5]
        x_i_type = x_i[:, 4]
        x_j_vx = x_j[:, 2:3] / vnorm[4]
        x_j_vy = x_j[:, 3:4] / vnorm[5]

        d = r

        self.new_edges = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:, None].repeat(1, 4)),dim=-1)

        return d
class InteractionNetworkEmb(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, nlayers, embedding, device):
        super().__init__(aggr=aggr_type)  # "mean" aggregation.

        self.nlayers = nlayers
        self.device = device
        self.embedding = embedding

        self.lin_edge = MLP(input_size=3 * self.embedding, hidden_size=3 * self.embedding, output_size=self.embedding,
                            nlayers=self.nlayers, device=self.device)
        self.lin_node = MLP(input_size=2 * self.embedding, hidden_size=2 * self.embedding, output_size=self.embedding,
                            nlayers=self.nlayers, device=self.device)


    def forward(self, x, edge_index, edge_feature):
        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        node_out = x + node_out
        edge_out = edge_feature + self.new_edges

        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((edge_feature, x_i, x_j), dim=-1)

        x = self.lin_edge(x)
        self.new_edges = x

        return x
class ResNetGNN(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""

    def __init__(self, model_config, device):
        super().__init__()

        self.hidden_size = model_config['hidden_size']
        self.embedding = model_config['embedding']
        self.nlayers = model_config['n_mp_layers']
        self.device = device
        self.noise_level = model_config['noise_level']

        self.edge_init = EdgeNetwork()

        # self.layer = torch.nn.ModuleList(
        #     [InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device) for _ in
        #      range(self.nlayers)])

        self.layer = InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device)

        self.node_out = MLP(input_size=self.embedding, hidden_size=self.hidden_size, output_size=2, nlayers=3,
                            device=self.device)

        self.embedding_node = MLP(input_size=12, hidden_size=self.embedding, output_size=self.embedding, nlayers=3,
                                  device=self.device)
        self.embedding_edges = MLP(input_size=11, hidden_size=self.embedding, output_size=self.embedding, nlayers=3,
                                   device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device=self.device, requires_grad=True))


    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x[:, 4:6] = self.a[x[:, 6].detach().cpu().numpy(), 0:2]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        node_feature = torch.cat((x[:, 0:4], x[:, 4:5].repeat(1, 4),x[:, 5:6].repeat(1, 4)), dim=-1)

        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]), requires_grad=False,
                            device=self.device) * self.noise_level
        node_feature = node_feature + noise
        edge_feature = self.edge_init(node_feature, edge_index)

        node_feature = self.embedding_node(node_feature)
        edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer(node_feature, edge_index, edge_feature=edge_feature)

        pred = self.node_out(node_feature)

        return pred

def data_generate(model_config, index_particles):

    if model_config['model']=='InteractionParticles3D' :
        data_generate_3D(model_config, index_particles)
    else:
        data_generate_2D(model_config, index_particles)

def data_generate_2D(model_config, index_particles):

    print('')
    print('Generating data ...')

    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_data/*")
    # for f in files:
    #     os.remove(f)

    files = glob.glob(f"{folder}/*")
    for f in files:
        os.remove(f)

    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    json_ = json.dumps(model_config)
    f = open(f"{folder}/model_config.json", "w")
    f.write(json_)
    f.close()

    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']

    if model_config['model'] == 'MixInteractionParticles':
        print(f'Generate MixInteractionParticles')

        p = torch.ones(nparticle_types, nparticle_types, 4, device=device) + torch.rand(nparticle_types,nparticle_types, 4, device=device)

        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)

        # read previous data
        # for n in range(nparticle_types):
        #     p[n] = torch.load(f'graphs_data/graphs_particles_230902_30/p_{n}.pt')
        #     print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
        # p[2]=p[1]*0.975
        #
        # p[0,0] = torch.tensor([1.0696,1.8843,1.322,1.252])
        # p[0, 1] = torch.tensor([1.7112,1.7178,1.108,1.471])
        # p[0, 2] = torch.tensor([1.8224,1.4711,1.7202,1.2569])
        # p[1, 1] = torch.tensor([1.078,1.3741,1.053,1.0633])
        # p[1, 2] = torch.tensor([1.0395,1.8933,1.5266,1.5097])
        # p[2, 2] = torch.tensor([1.0833,1.2819,1.6062,1.0675])

        for n in range(nparticle_types):
            for m in range(n, nparticle_types):
                p[m, n] = p[n, m]
                psi_output.append(psi(rr, torch.squeeze(p[n, m])))
                print(f'p{n, m}: {np.round(torch.squeeze(p[n, m]).detach().cpu().numpy(), 4)}')
                torch.save(torch.squeeze(p[n, m]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}_{m}.pt')

        model = InteractionParticles_1(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    else:

        print(f'Generate InteractionParticles')

        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)

        # read previous data
        # for n in range(nparticle_types):
        #     p[n] = torch.load(f'graphs_data/graphs_particles_230902_30/p_{n}.pt')
        #     print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
        # p[2]=p[1]*0.975
        # p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        # p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        # p[2] = torch.tensor([1.785, 1.8579,1.7226, 1.0584])


        for n in range(nparticle_types):
            # model.append(InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p[n]), tau=tau))
            # torch.save({'model_state_dict': model[n].state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model_{n}.pt')
            psi_output.append(psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')

        model = InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p), tau=tau)
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    time.sleep(0.5)


    for run in range(2):

        X1 = torch.rand(nparticles, 2, device=device)
        X1t = torch.zeros((nparticles, 2, nframes))  # to store all the intermediate time

        V1 = torch.zeros((nparticles, 2), device=device)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:, None]

        time.sleep(0.5)

        for it in tqdm(range(nframes)):

            X1t[:, :, it] = X1.clone().detach()  # for later display

            X1 = bc_pos(X1 + V1)

            distance = torch.sum(bc_diff(X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            x = torch.concatenate(
                (X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), N1.clone().detach()), 1)
            torch.save(x, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')

            dataset = data.Data(x=x, edge_index=edge_index)

            y = 0
            with torch.no_grad():
                y = model(dataset)

            torch.save(y, f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{it}.pt')

            V1 += y

            if (run == 0) & (it % 5 == 0):

                # distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                # adj_t2 = ((distance < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                # edge_index2 = adj_t2.nonzero().t().contiguous()
                # dataset2 = data.Data(x=x, edge_index=edge_index2)

                fig = plt.figure(figsize=(14, 7 * 0.95))
                # plt.ion()
                # ax = fig.add_subplot(1, 2, 2)
                # pos = dict(enumerate(x[:, 0:2].detach().cpu().numpy(), 0))
                # vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                # nx.draw_networkx(vis, pos=pos, node_size=10, linewidths=0, edge_color='b', with_labels=False)
                # plt.xlim([-0.3, 1.3])
                # plt.ylim([-0.3, 1.3])

                ax = fig.add_subplot(1, 2, 1)

                for n in range(nparticle_types):
                    plt.scatter(X1t[index_particles[n], 0, it], X1t[index_particles[n], 1, it], s=3)
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.xlim([-0.3, 1.3])
                plt.ylim([-0.3, 1.3])
                plt.text(-0.25, 1.38, f'frame: {it}')
                plt.text(-0.25, 1.33, f'Graph    {x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)

                if model_config['model'] == 'MixInteractionParticles':
                    N=0
                    for n in range(nparticle_types):
                        for m in range(n, nparticle_types):
                            plt.text(-0.25, 1.25 - N * 0.05, f'p{n}: {np.round(p[n,m].detach().cpu().numpy(), 4)}',color='k')
                            N+=1
                else:
                    plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                    for n in range(nparticle_types):
                        plt.text(-0.25, 1.25 - n * 0.05, f'p{n}: {np.round(p[n].detach().cpu().numpy(), 4)}',
                                 color='k')

                ax = fig.add_subplot(5, 5, 21)

                if model_config['model'] == 'MixInteractionParticles':
                    N = 0
                    for n in range(nparticle_types):
                        for m in range(n, nparticle_types):
                            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[N].cpu()), linewidth=1)
                            plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0,
                                     color=[0, 0, 0], linewidth=0.5)
                            N += 1
                else:
                    for n in range(nparticle_types):
                        plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                        plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],linewidth=0.5)

                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

def data_generate_3D(model_config, index_particles):

    print('')
    print('Generating data ...')

    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_data/*")
    # for f in files:
    #     os.remove(f)

    files = glob.glob(f"{folder}/*")
    for f in files:
        os.remove(f)

    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    json_ = json.dumps(model_config)
    f = open(f"{folder}/model_config.json", "w")
    f.write(json_)
    f.close()

    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']

    if True:
        print(f'Generate MixInteractionParticles')

        p = torch.ones(nparticle_types, nparticle_types, 4, device=device) + torch.rand(nparticle_types,nparticle_types, 4, device=device)

        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)

        # read previous data
        # for n in range(nparticle_types):
        #     p[n] = torch.load(f'graphs_data/graphs_particles_230902_30/p_{n}.pt')
        #     print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
        # p[2]=p[1]*0.975

        # p[0,0] = torch.tensor([1.0696,1.8843,1.322,1.252])
        # p[0, 1] = torch.tensor([1.7112,1.7178,1.108,1.471])
        # p[0, 2] = torch.tensor([1.8224,1.4711,1.7202,1.2569])
        # p[1, 1] = torch.tensor([1.078,1.3741,1.053,1.0633])
        # p[1, 2] = torch.tensor([1.0395,1.8933,1.5266,1.5097])
        # p[2, 2] = torch.tensor([1.0833,1.2819,1.6062,1.0675])

        for n in range(nparticle_types):
            for m in range(n, nparticle_types):
                p[m, n] = p[n, m]
                psi_output.append(psi(rr, torch.squeeze(p[n, m])))
                print(f'p{n, m}: {np.round(torch.squeeze(p[n, m]).detach().cpu().numpy(), 4)}')
                torch.save(torch.squeeze(p[n, m]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}_{m}.pt')

        model = InteractionParticles_3(aggr_type=aggr_type, p=torch.squeeze(p), tau=tau)
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    else:

        print(f'Generate InteractionParticles')

        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)

        # read previous data
        # for n in range(nparticle_types):
        #     p[n] = torch.load(f'graphs_data/graphs_particles_230902_30/p_{n}.pt')
        #     print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
        # p[2]=p[1]*0.975
        # p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        # p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        # p[2] = torch.tensor([1.785, 1.8579,1.7226, 1.0584])


        for n in range(nparticle_types):
            # model.append(InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p[n]), tau=tau))
            # torch.save({'model_state_dict': model[n].state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model_{n}.pt')
            psi_output.append(psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')

        model = InteractionParticles_2(aggr_type=aggr_type, p=torch.squeeze(p), tau=tau)
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    time.sleep(0.5)

    for run in range(2):

        X1 = torch.rand(nparticles, 3, device=device)

        for n in range(nparticles):
            flag=True
            while(flag):
                temp = torch.rand(1, 3, device=device) - torch.tensor([0.5,0.5,0.5], device=device)
                if temp.norm(2)>0.001:
                    X1[n,:] = temp / temp.norm(2)
                    flag=False

        X1t = torch.zeros((nparticles, 3, nframes))  # to store all the intermediate time

        V1 = torch.zeros((nparticles, 3), device=device)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:, None]

        time.sleep(0.5)
        for it in tqdm(range(nframes)):

            X1t[:, :, it] = X1.clone().detach()  # for later display

            X1 = bc_pos(X1 + V1)
            norm = X1[:,0:3].norm(2,dim=1)
            X1 [:,0:3] = X1[:,0:3] / torch.cat((norm[:,None],norm[:,None],norm[:,None],),axis=1)

            distance = torch.sum(bc_diff(X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            x = torch.concatenate(
                (X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), N1.clone().detach()), 1)
            torch.save(x, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')

            dataset = data.Data(x=x, edge_index=edge_index)

            y = 0
            with torch.no_grad():
                y = model(dataset)

            torch.save(y, f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{it}.pt')

            V1 += y

            if (run == 0) & (it % 5 == 0):

                fig = plt.figure(figsize=(8, 10))
                # plt.ion()
                ax = fig.add_subplot(1, 1, 1,projection='3d')

                for n in range(nparticle_types):
                    ax.scatter(X1t[index_particles[n], 0, it], X1t[index_particles[n], 1, it], X1t[index_particles[n], 2, it], s=2)
                ax = plt.gca()

                ax = fig.add_subplot(5, 5, 21)

                if True:
                    N = 0
                    for n in range(nparticle_types):
                        for m in range(n, nparticle_types):
                            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[N].cpu()), linewidth=1)
                            plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0,
                                     color=[0, 0, 0], linewidth=0.5)
                            N += 1
                else:
                    for n in range(nparticle_types):
                        plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                        plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],linewidth=0.5)

                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

def data_train(model_config, index_particles):

    print('')
    print('Training loop ...')

    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
    noise_type = model_config['noise_type']
    embedding_type = model_config['embedding_type']
    embedding = model_config['embedding']

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_*")
    NGraphs = int(len(graph_files) / nframes)
    print('Graph files N: ', NGraphs-1)
    time.sleep(0.5)

    arr = np.arange(0, NGraphs - 1, 2)
    for run in arr:
        kr = np.arange(0, nframes - 1, 4)
        for k in kr:
            x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k}.pt')
            y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{k}.pt')
            if (run == 0) & (k == 0):
                xx = x
                yy = y
            else:
                xx = torch.concatenate((x, xx))
                yy = torch.concatenate((y, yy))
    if model_config['model'] == 'InteractionParticles3D':
        vnorm = norm_velocity3D(xx, device)
        ynorm = norm_acceleration3D(yy, device)
    else:
        vnorm = norm_velocity(xx, device)
        ynorm = norm_acceleration(yy, device)

    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

    if model_config['model'] == 'InteractionParticles':
        model = InteractionParticles(model_config, device)
        print(f'Training InteractionParticles')
    if model_config['model'] == 'InteractionParticles3D':
        model = InteractionParticles3D(model_config, device)
        print(f'Training InteractionParticles3d')
    if model_config['model'] == 'MixInteractionParticles':
        model = MixInteractionParticles(model_config, device)
        print(f'Training MixInteractionParticles')
    if model_config['model'] == 'ResNetGNN':
        model = ResNetGNN(model_config, device)
        print(f'Training ResNetGNN')
    # state_dict = torch.load(net)
    # model.load_state_dict(state_dict['model_state_dict'])

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")

    print('')
    net = f"./log/try_{ntry}/models/best_model_with_{NGraphs-1}_graphs.pt"
    print(f'network: {net}')
    print('')

    time.sleep(0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)  # , weight_decay=5e-4)

    model.train()
    best_loss = np.inf
    stp = 1

    if data_augmentation:
        data_augmentation_loop = 20
        print(f'data_augmentation_loop: {data_augmentation_loop}')
    else:
        data_augmentation_loop = 1
        print('no data augmentation ...')
    print('')
    time.sleep(0.5)

    list_loss = []
    list_gap = []
    embedding_list=[]
    D_nm = torch.zeros((60,nparticle_types, nparticle_types))

    for epoch in range(40):

        if epoch == 30:
            optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)  # , weight_decay=5e-4)

        total_loss = 0

        for N in range(1, nframes * data_augmentation_loop, stp):

            run = 1 + np.random.randint(NGraphs - 1)
            k = np.random.randint(nframes - 1)

            x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k}.pt')
            x = x.to(device)

            if (noise_type > 0):

                noise = torch.randn((x.shape[0], 4), device=device) * noise_level
                if (noise_type == 1) | (noise_type == 3):
                    x[:, 0:2] = x[:, 0:2] + noise[:, 0:2] * radius
                    x[:, 2:4] = x[:, 2:4] + noise[:, 2:4] * torch.std(x[:, 2:4])

            if data_augmentation:
                phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)

            distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            adj_t = (distance < radius ** 2).float() * 1
            t = torch.Tensor([radius ** 2])
            edges = adj_t.nonzero().t().contiguous()
            y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{k}.pt')
            y = y.to(device)
            # y.requires_grad = False
            y[:, 0] = y[:, 0] / ynorm[4]
            y[:, 1] = y[:, 1] / ynorm[5]
            if model_config['model'] == 'InteractionParticles3D':
                y[:, 2] = y[:, 2] / ynorm[6]

            if (noise_type >1):
                noise = torch.randn((y.shape[0], 2), device=device) * noise_level
                y[:, 0:2] = y[:, 0:2] + noise[:, 0:2] * torch.std(y[:, 0:2])

            if data_augmentation:
                new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                y[:, 0] = new_x
                y[:, 1] = new_y

            dataset = data.Data(x=x[:, :], edge_index=edges)

            optimizer.zero_grad()
            pred = model(dataset, step = 1, vnorm=vnorm, cos_phi=cos_phi, sin_phi=sin_phi)

            loss = (pred - y).norm(2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.a.data = torch.clamp(model.a.data, min=-4, max=4)
        embedding = model.a.detach().cpu().numpy()
        embedding = scaler.fit_transform(embedding)
        embedding_particle = []
        for n in range(nparticle_types):
            embedding_particle.append(embedding[index_particles[n], :])
        kmeans = KMeans(init="random", n_clusters=nparticle_types, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(embedding)
        gap = kmeans.inertia_
        kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
        sse = []
        for k in range(1, 11):
            kmeans_ = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans_.fit(embedding)
            sse.append(kmeans_.inertia_)
        kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")

        for n in range(nparticle_types-1):
            for m in range(n+1,nparticle_types):
                D_nm[epoch,n,m] = S_e(torch.tensor(embedding_particle[n]), torch.tensor(embedding_particle[m]))

        torch.save(D_nm, f"./tmp_training/D_nm_{ntry}.pt")

        if (total_loss / nframes / data_augmentation_loop / nparticles < best_loss):
            best_loss = total_loss / nframes / data_augmentation_loop / nparticles
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_with_{NGraphs-1}_graphs.pt'))
            print("Epoch {}. Loss: {:.6f} geomloss {:.2f} saving model  ".format(epoch,total_loss / N / nparticles,torch.sum(D_nm[epoch]).item()))
        else:
            print(
                "Epoch {}. Loss: {:.6f} geomloss {:.2f} ".format(epoch, total_loss / N / nparticles,torch.sum(D_nm[epoch]).item()))

        if epoch == 29:
            if data_augmentation:
                data_augmentation_loop = 200
                print(f'data_augmentation_loop: {data_augmentation_loop}')

        if epoch == 49:
            print('training MLP only ...')
            model.a.requires_grad = False
            new_a = kmeans.cluster_centers_[kmeans.labels_, :]

            # if gap < 100:
            #     model.a.data = torch.tensor(new_a, device=device)

            embedding = model.a.detach().cpu().numpy()
            embedding = scaler.fit_transform(embedding)
            embedding_list.append(torch.tensor(embedding,device=device))
            torch.save(embedding_list,f"./tmp_training/Embedding_{ntry}.pt")
            embedding_particle = []
            for n in range(nparticle_types):
                embedding_particle.append(embedding[index_particles[n], :])
            best_loss = np.inf

        list_loss.append(total_loss / N / nparticles)
        list_gap.append(gap)


        fig = plt.figure(figsize=(13, 8))
        # plt.ion()
        ax = fig.add_subplot(2, 3, 1,projection='3d')

        if (embedding_type == 'none') & (embedding.shape[1]>2):
            for n in range(nparticle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],s=1)
        else :
            for n in range(nparticle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 1]*0,s=1)

        ax = fig.add_subplot(2, 3, 2)
        for n in range(nparticle_types):
            plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], s=3)
        plt.xlim([-4.1, 4.1])
        plt.ylim([-4.1, 4.1])
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        plt.text(-3.9, 3.6, f'kmeans.inertia: {np.round(gap, 2)}   kl.elbow: {kl.elbow}', fontsize=10)
        ax = fig.add_subplot(2, 3, 3)
        plt.plot(list_loss, color='k')
        plt.xlim([0, 60])
        plt.ylim([0, 0.02])
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epochs', fontsize=10)

        ax = fig.add_subplot(2, 3, 4)
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters", fontsize=12)
        plt.ylabel("SSE", fontsize=12)

        ax = fig.add_subplot(2, 3, 5)
        for n in range(nparticle_types - 1):
            for m in range(n + 1, nparticle_types):
                plt.plot(D_nm[0:epoch, n, m])

        plt.xlim([0, 60])
        plt.ylabel('Geomloss', fontsize=10)
        plt.xlabel('Epochs', fontsize=10)

        plt.tight_layout()

        plt.savefig(f"./tmp_training/Fig_{ntry}_{epoch}.tif")
        plt.close()

def data_test(model_config, index_particles, prev_nparticles, new_nparticles, prev_index_particles, bVisu):
    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_recons/*")
    # for f in files:
    #     os.remove(f)

    print('')
    print('Plot validation test ... ')

    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']

    if model_config['model'] == 'InteractionParticles':
        model = InteractionParticles(model_config, device)
    if model_config['model'] == 'MixInteractionParticles':
        model = MixInteractionParticles(model_config, device)
    if model_config['model'] == 'ResNetGNN':
        model = ResNetGNN(model_config, device)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_*")
    NGraphs = int(len(graph_files) / nframes)
    print('Graph files N: ', NGraphs-1)

    net = f"./log/try_{ntry}/models/best_model_with_{NGraphs-1}_graphs.pt"
    print(f'network: {net}')
    state_dict = torch.load(net)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    if new_nparticles>0:  # nparticles larger than initially

        ratio_particles = int(new_nparticles / prev_nparticles)
        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio_particles}')
        print('')

        embedding = model.a.data
        new_embedding = []

        for n in range(nparticle_types):
            for m in range(ratio_particles):
                if (n == 0) & (m == 0):
                    new_embedding = embedding[prev_index_particles[n]]
                else:
                    new_embedding = torch.cat((new_embedding, embedding[prev_index_particles[n]]), axis=0)

        model.a = nn.Parameter(torch.tensor(np.ones((int(prev_nparticles) * ratio_particles, 2)), device=device, requires_grad=False))
        model.a.data = new_embedding
        nparticles = new_nparticles
        model_config['nparticles'] = new_nparticles

    ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt')
    vnorm = torch.load(f'./log/try_{ntry}/vnorm.pt')
    ynorm = ynorm.to(device)
    v = vnorm.to(device)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")

    x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_0.pt')
    x00 = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_0.pt')
    y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_0_0.pt')
    x = x.to(device)
    x00 = x00.to(device)
    y = y.to(device)
    print('')
    print(f'x: {x.shape}')
    print(f'index_particles: {index_particles[0].shape}')
    print('')

    rmserr_list = []
    discrepency_list = []
    Sxy_list = []

    for it in tqdm(range(nframes - 1)):

        x0 = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_{min(it + 1, nframes - 2)}.pt')
        x0 = x0.to(device)

        distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = (distance < radius ** 2).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, step=2, vnorm=v, cos_phi=0, sin_phi=0)  # acceleration estimation

        y[:, 0] = y[:, 0] * ynorm[4]
        y[:, 1] = y[:, 1] * ynorm[5]

        x[:, 2:4] = x[:, 2:4] + y  # speed update

        x[:, 0:2] = bc_pos(x[:, 0:2] + x[:, 2:4])  # position update

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 0:2] - x0[:, 0:2]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        discrepency = MMD(x[:, 0:2], x0[:, 0:2])
        discrepency_list.append(discrepency)

        # Sxy = S_e(x[:, 0:2], x0[:, 0:2])
        # Sxy_list.append(Sxy.item())

        stp = 5
        if (it % stp == 0) & bVisu:

            distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
            edge_index2 = adj_t2.nonzero().t().contiguous()
            dataset2 = data.Data(x=x, edge_index=edge_index2)

            fig = plt.figure(figsize=(25, 16))
            # plt.ion()
            ax = fig.add_subplot(2, 3, 1)
            for n in range(nparticle_types):
                plt.scatter(x00[index_particles[n], 0].detach().cpu(), x00[index_particles[n], 1].detach().cpu(), s=3)

            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, f't0 {nparticles} particles', fontsize=10)

            ax = fig.add_subplot(2, 3, 2)
            for n in range(nparticle_types):
                plt.scatter(x0[index_particles[n], 0].detach().cpu(), x0[index_particles[n], 1].detach().cpu(), s=3)
            ax = plt.gca()
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'True', fontsize=30)

            ax = fig.add_subplot(2, 3, 3)
            plt.plot(np.arange(len(rmserr_list)), rmserr_list, label='RMSE', c='r')
            plt.ylim([0, 0.1])
            plt.xlim([0, nframes])
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.xlabel('Frame [a.u]', fontsize="14")
            ax.set_ylabel('RMSE [a.u]', fontsize="14", color='r')
            ax2 = ax.twinx()
            plt.plot(np.arange(len(discrepency_list)), discrepency_list,
                     label='Maximum Mean Discrepencies', c='b')
            ax2.set_ylabel('MMD [a.u]', fontsize="14", color='b')
            ax2.set_ylim([0, 2E-3])

            ax = fig.add_subplot(2, 3, 4)
            pos = dict(enumerate(np.array(x[:, 0:2].detach().cpu()), 0))
            vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=10, linewidths=0, with_labels=False)
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, f'Frame: {it}')
            plt.text(-0.25, 1.33, f'Graph: {x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)

            ax = fig.add_subplot(2, 3, 5)
            for n in range(nparticle_types):
                plt.scatter(x[index_particles[n], 0].detach().cpu(), x[index_particles[n], 1].detach().cpu(), s=3)
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'Model', fontsize=30)

            ax = fig.add_subplot(2, 3, 6)
            temp1 = torch.cat((x, x0), 0)
            temp2 = torch.tensor(np.arange(nparticles), device=device)
            temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
            temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
            temp4 = torch.t(temp4)

            distance3 = torch.sqrt(torch.sum((x[:, 0:2] - x0[:, 0:2]) ** 2, 1))
            p = torch.argwhere(distance3 < 0.3)

            pos = dict(enumerate(np.array((temp1[:, 0:2]).detach().cpu()), 0))
            dataset = data.Data(x=temp1[:, 0:2], edge_index=torch.squeeze(temp4[:, p]))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False)
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.18, f'Frame: {it}')
            plt.text(-0.25, 1.13, 'Prediction RMSE: {:.4f}'.format(rmserr.detach()), fontsize=10)

            ax = fig.add_subplot(8, 10, 54)
            embedding = model.a.detach().cpu().numpy()
            embedding = scaler.fit_transform(embedding)
            embedding_particle = []
            for n in range(nparticle_types):
                embedding_particle.append(embedding[index_particles[n], :])
                plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], s=3)
            plt.xlim([-4.1, 4.1])
            plt.ylim([-4.1, 4.1])
            plt.xlabel('Embedding 0', fontsize=8)
            plt.ylabel('Embedding 1', fontsize=8)

            plt.savefig(f"./tmp_recons/Fig_{ntry}_{it}.tif")

            plt.close()
    print('')
    print(f'ntry: {ntry}')
    print(f'Final RMSE: {rmserr.item()}')
    print(f'Final MMD: {discrepency}')
    # print(f'Final Sxy: {Sxy.item()}')

def data_test_generate(model_config, index_particles):

    print('')
    print('Generating test data ...')

    nframes = 200
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']

    if model_config['model'] == 'MixInteractionParticles':
        print(f'Generate MixInteractionParticles')

        p = torch.ones(nparticle_types, nparticle_types, 4, device=device) + torch.rand(nparticle_types,nparticle_types, 4, device=device)

        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)

        # read previous data
        # for n in range(nparticle_types):
        #     p[n] = torch.load(f'graphs_data/graphs_particles_230902_30/p_{n}.pt')
        #     print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
        # p[2]=p[1]*0.975

        for n in range(nparticle_types):
            for m in range(n, nparticle_types):
                p[n,m] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}_{m}.pt')
                p[m, n] = p[n, m]
                psi_output.append(psi(rr, torch.squeeze(p[n, m])))
                print(f'p{n, m}: {np.round(torch.squeeze(p[n, m]).detach().cpu().numpy(), 4)}')

        model = InteractionParticles_1(aggr_type=aggr_type, p=torch.squeeze(p), tau=tau)

    else:
        print(f'Generate InteractionParticles')

        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)
        for n in range(nparticle_types):
            p[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
            # model.append(InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p[n]), tau=tau))
            psi_output.append(psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')

        model = InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p), tau=tau)


    prev_nparticles = nparticles
    prev_index_particles = index_particles

    new_nparticles = 3000
    nparticles = new_nparticles

    index_particles = []
    for n in range(nparticle_types):
        index_particles.append(np.arange(int(nparticles / nparticle_types) * n,
                                         int(nparticles / nparticle_types) * (n + 1)))

    X1 = torch.rand(nparticles, 2, device=device)

    # scenario A
    # X1[:, 0] = X1[:, 0] / nparticle_types
    # for n in range(nparticle_types):
    #     X1[index_particles[n], 0] = X1[index_particles[n], 0] + n / nparticle_types

    # scenario B
    # X1[index_particles[0], :] = X1[index_particles[0], :]/2 + 1/4

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

    X1t = torch.zeros((nparticles, 2, nframes))  # to store all the intermediate time

    V1 = torch.zeros((nparticles, 2), device=device)
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    N1 = torch.arange(nparticles, device=device)
    N1 = N1[:, None]

    for it in tqdm(range(nframes)):

        X1t[:, :, it] = X1.clone().detach()  # for later display

        X1 = bc_pos(X1 + V1)

        distance = torch.sum(bc_diff(X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = (distance < radius ** 2).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()

        x = torch.concatenate(
            (X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), N1.clone().detach()), 1)

        torch.save(x, f'graphs_data/graphs_particles_{dataset_name}/x_{0}_{it}.pt')

        dataset = data.Data(x=x, edge_index=edge_index)

        y = 0
        with torch.no_grad():
            y = model(dataset)

        torch.save(y, f'graphs_data/graphs_particles_{dataset_name}/y_{0}_{it}.pt')

        V1 += y

        if (it % 5 == 0):

            distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            adj_t2 = ((distance < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
            edge_index2 = adj_t2.nonzero().t().contiguous()
            dataset2 = data.Data(x=x, edge_index=edge_index2)

            fig = plt.figure(figsize=(14, 7 * 0.95))
            # plt.ion()

            ax = fig.add_subplot(1, 2, 1)

            for n in range(nparticle_types):
                plt.scatter(X1t[index_particles[n], 0, it], X1t[index_particles[n], 1, it], s=3)

            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            plt.text(-0.25, 1.38, f'frame: {it}')
            plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
            if model_config['model'] == 'MixInteractionParticles':
                N = 0
                for n in range(nparticle_types):
                    for m in range(n, nparticle_types):
                        plt.text(-0.25, 1.25 - N * 0.05, f'p{n}: {np.round(p[n, m].detach().cpu().numpy(), 4)}',
                                 color='k')
                        N += 1
            else:
                for n in range(nparticle_types):
                    plt.text(-0.25, 1.25 - n * 0.05, f'p{n}: {np.round(p[n].detach().cpu().numpy(), 4)}', color='k')

            ax = fig.add_subplot(5, 5, 21)

            if model_config['model'] == 'MixInteractionParticles':
                N = 0
                for n in range(nparticle_types):
                    for m in range(n, nparticle_types):
                        plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[N].cpu()), linewidth=1)
                        plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0,
                                 color=[0, 0, 0], linewidth=0.5)
                        N += 1
            else:
                for n in range(nparticle_types):
                    plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                    plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                             linewidth=0.5)

            plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
            plt.close()

    return prev_nparticles, new_nparticles, prev_index_particles, index_particles

def data_train_generate(model_config, index_particles, arrow, prev_folder):

    print('')
    print('Generating training data ...')

    nframes = 200
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']

    p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
    model = []
    psi_output = []
    rr = torch.tensor(np.linspace(0, 0.015, 100))
    rr = rr.to(device)

    # read previous data
    for n in range(nparticle_types):
        p[n] = torch.load(f'{prev_folder}/p_{n}.pt')
        print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')

    for n in range(nparticle_types):
        model.append(InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p[n]), tau=model_config['tau']))
        torch.save({'model_state_dict': model[n].state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model_{n}.pt')
        psi_output.append(psi(rr, torch.squeeze(p[n])))
        print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')

    for n in range(nparticle_types):
        torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')

    prev_nparticles = nparticles
    prev_index_particles = index_particles

    new_nparticles = 4000
    nparticles = new_nparticles

    index_particles = []
    for n in range(nparticle_types):
        index_particles.append(np.arange(int(nparticles / nparticle_types) * n,
                                         int(nparticles / nparticle_types) * (n + 1)))

    X1 = torch.rand(nparticles, 2, device=device)

    # scenario A
    # X1[:, 0] = X1[:, 0] / nparticle_types
    # for n in range(nparticle_types):
    #     X1[index_particles[n], 0] = X1[index_particles[n], 0] + n / nparticle_types

    # scenario B
    # X1[index_particles[0], :] = X1[index_particles[0], :]/2 + 1/4

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

    X1t = torch.zeros((nparticles, 2, nframes))  # to store all the intermediate time

    V1 = torch.zeros((nparticles, 2), device=device)
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    N1 = torch.arange(nparticles, device=device)
    N1 = N1[:, None]

    if arrow == 'forward':
        for it in tqdm(range(nframes)):

            X1t[:, :, it] = X1.clone().detach()  # for later display

            X1 = bc_pos(X1 + V1)

            distance = torch.sum(bc_diff(X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            x = torch.concatenate(
                (X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), N1.clone().detach()), 1)

            torch.save(x, f'graphs_data/graphs_particles_{dataset_name}/x_{0}_{it}.pt')

            dataset = data.Data(x=x, edge_index=edge_index)

            y = 0
            with torch.no_grad():
                for n in range(nparticle_types):
                    y += model[n](dataset) * (x[:, 4:6] == n)

            torch.save(y, f'graphs_data/graphs_particles_{dataset_name}/y_{0}_{it}.pt')

            V1 += y

            if (it % 5 == 0):

                distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                adj_t2 = ((distance < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                edge_index2 = adj_t2.nonzero().t().contiguous()
                dataset2 = data.Data(x=x, edge_index=edge_index2)

                fig = plt.figure(figsize=(14, 7 * 0.95))
                # plt.ion()

                ax = fig.add_subplot(1, 2, 1)

                for n in range(nparticle_types):
                    plt.scatter(X1t[index_particles[n], 0, it], X1t[index_particles[n], 1, it], s=3)

                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.xlim([-0.3, 1.3])
                plt.ylim([-0.3, 1.3])
                plt.text(-0.25, 1.38, f'frame: {it}')
                plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                for n in range(nparticle_types):
                    plt.text(-0.25, 1.25 - n * 0.05, f'p{n}: {np.round(model[n].p.detach().cpu().numpy(), 4)}', color='k')

                ax = fig.add_subplot(5, 5, 21)

                for n in range(nparticle_types):
                    plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                    plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                             linewidth=0.5)

                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

    if arrow == 'backward':

        for run in tqdm(range(2)):
            N=0

            x = torch.concatenate(
                (X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), N1.clone().detach()), 1)

            for it in range(nframes-3, -1,-1):

                x_current = torch.load(f'{prev_folder}/x_{run}_{it}.pt')
                x_current = x_current.to(device)
                x_prev = torch.load(f'{prev_folder}/x_{run}_{it+1}.pt')
                x_prev = x_prev.to(device)
                x_prev_prev = torch.load(f'{prev_folder}/x_{run}_{it+2}.pt')
                x_prev_prev = x_prev_prev.to(device)

                x[:,0:2] = x_current[:,0:2]
                x[:,2:4] = x_current[:,0:2] - x_prev[:,0:2]
                y = x_current[:,0:2] - 2*x_prev[:,0:2] + x_prev_prev[:,0:2]

                torch.save(x.detach(), f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{N}.pt')
                torch.save(y.detach(), f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{N}.pt')

                if (it % 5 == 0) & (run==0):
                    fig = plt.figure(figsize=(14, 7 * 0.95))
                    # plt.ion()
                    ax = fig.add_subplot(1, 2, 1)
                    for n in range(nparticle_types):
                        plt.scatter(x[index_particles[n], 0].detach().cpu().numpy(), x[index_particles[n], 1].detach().cpu().numpy(), s=3)
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    plt.text(-0.25, 1.38, f'frame: {N}')
                    plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                    for n in range(nparticle_types):
                        plt.text(-0.25, 1.25 - n * 0.05, f'p{n}: {np.round(model[n].p.detach().cpu().numpy(), 4)}', color='k')
                    ax = fig.add_subplot(5, 5, 21)
                    for n in range(nparticle_types):
                        plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                        plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                                 linewidth=0.5)
                    plt.savefig(f"./tmp_data/Fig_{ntry}_{N}.tif")
                    plt.close()

                N += 1
            torch.save(x.detach(), f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{N}.pt')
            torch.save(y.detach(), f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{N}.pt')
            N += 1
            torch.save(x.detach(), f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{N}.pt')
            torch.save(y.detach(), f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{N}.pt')

    if arrow == 'geomloss':

        for run in tqdm(range(2)):

            N=0

            x0 = torch.rand(nparticles, 2, device=device)
            x1 = torch.rand(nparticles, 2, device=device)
            x2 = torch.rand(nparticles, 2, device=device)
            x3 = torch.rand(nparticles, 2, device=device)
            x199 = torch.rand(nparticles, 2, device=device)


            i0 = imread('graphs_data/pattern_1.tif')
            pos = np.argwhere(i0 == 255)
            l = np.arange(pos.shape[0])
            l = np.random.permutation(l)
            x199[index_particles[0],0:2] = torch.tensor(pos[l[index_particles[0]],:]/255,dtype=torch.float32,device=device)
            pos = np.argwhere(i0 == 0)
            l = np.arange(pos.shape[0])
            l = np.random.permutation(l)
            x199[index_particles[1],0:2] = torch.tensor(pos[l[index_particles[0]],:]/255,dtype=torch.float32,device=device)

            Sxy_0_199 = S_e(x0[:, 0:2], x199[:, 0:2])

            x = torch.tensor(np.zeros((int(nparticles), 2)), dtype=torch.float32, device=device, requires_grad=True)
            x.data[:, 0:2] = x199[:, 0:2]
            optimizer = torch.optim.Adam([x], lr=1E-3)  # , weight_decay=5e-4)

            for it in range(nframes):

                alpha = (it/nframes) ** 3

                optimizer.zero_grad()

                loss=0
                for n in range(nparticle_types):
                    loss += 1E4 * alpha * (S_e(x[index_particles[n],0:2], x0[index_particles[n], 0:2]) +  S_e(x[index_particles[n],0:2], x1[index_particles[n], 0:2]) + S_e(x[index_particles[n],0:2], x2[index_particles[n], 0:2]) + S_e(x[index_particles[n],0:2], x3[index_particles[n], 0:2])) +  1E4 * (1-alpha) * S_e(x[index_particles[n],0:2], x199[index_particles[n], 0:2]) #- alpha torch.log(S_e(x[index_particles[n],0:2], x199[index_particles[n], 0:2])+1E-8)
                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    x.data =  bc_pos(x.data)

                torch.save(x.detach(), f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')

                if (it % 5 == 0) & (run==0):
                    fig = plt.figure(figsize=(14, 7 * 0.95))
                    # plt.ion()
                    ax = fig.add_subplot(1, 2, 1)
                    for n in range(nparticle_types):
                        plt.scatter(x[index_particles[n], 0].detach().cpu().numpy(), x[index_particles[n], 1].detach().cpu().numpy(), s=3)
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    plt.text(-0.25, 1.38, f'frame: {N}')
                    plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                    for n in range(nparticle_types):
                        plt.text(-0.25, 1.25 - n * 0.05, f'p{n}: {np.round(model[n].p.detach().cpu().numpy(), 4)}', color='k')
                    ax = fig.add_subplot(5, 5, 21)
                    for n in range(nparticle_types):
                        plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                        plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                                 linewidth=0.5)
                    ax = fig.add_subplot(1, 2, 2)
                    for n in range(nparticle_types):
                        plt.scatter(x0[index_particles[n], 0].detach().cpu().numpy(), x0[index_particles[n], 1].detach().cpu().numpy(), s=3)
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                    plt.close()

def print_model_config (model_config):

    print('')
    ntry = model_config['ntry']
    print(f'ntry: {ntry}')
    dataset_name = model_config['dataset']
    print(f'dataset_name: {dataset_name}')
    nparticles = model_config['nparticles']  # number of particles
    print(f'nparticles: {nparticles}')
    nparticle_types = model_config['nparticle_types']  # number of particles
    print(f'nparticle_types: {nparticle_types}')
    nframes = model_config['nframes']
    print(f'nframes: {nframes}')
    radius = model_config['radius']
    print(f'radius: {radius}')
    sigma = model_config['sigma']
    print(f'sigma: {sigma}')
    tau = model_config['tau']
    print(f'tau: {tau}')
    aggr_type = model_config['aggr_type']
    print(f'aggr_type: {aggr_type}')
    particle_embedding = model_config['particle_embedding']
    print(f'particle_embedding: {particle_embedding}')
    boundary = model_config['boundary']
    print(f'boundary: {boundary}')
    input_size = model_config['input_size']
    print(f'input_size: {input_size}')
    hidden_size = model_config['hidden_size']
    print(f'hidden_size: {hidden_size}')
    output_size = model_config['output_size']
    print(f'output_size: {output_size}')
    noise_level = model_config['noise_level']
    print(f'noise_level: {noise_level}')
    noise_type = model_config['noise_type']
    print(f'noise_type: {noise_type}')
    embedding_type = model_config['embedding_type']
    print(f'embedding_type: {embedding_type}')
    embedding = model_config['embedding']
    print(f'embedding: {embedding}')
    if model_config['upgrade_type']==0:
        print ('Acc = aggr(message)')
    if model_config['upgrade_type']==1:
        print ('Acc = MLP(aggr(message),velocity,embedding')


if __name__ == '__main__':

    print('')
    print('version 1.2 230923')
    print('')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    # model_config = {'ntry': 700,
    #                 'input_size': 15,
    #                 'output_size': 2,
    #                 'hidden_size': 64,
    #                 'n_mp_layers': 5,
    #                 'noise_level': 0,
    #                 'noise_type': 0,
    #                 'radius': 0.075,
    #                 'dataset': '230902_700',
    #                 'nparticles': 5000,
    #                 'nparticle_types': 10,
    #                 'nframes': 200,
    #                 'sigma': .005,
    #                 'tau': 0.1,
    #                 'aggr_type' : 'mean',
    #                 'particle_embedding': True,
    #                 'boundary': 'periodic',  # periodic   'no'  # no boundary condition
    #                 'data_augmentation' : True,
    #                 'embedding_type': 'repeat',
    #                 'model': 'InteractionParticles',
    #                     'upgrade_type':0}

    # model_config = {'ntry': 30,
    #                 'input_size': 15,
    #                 'output_size': 2,
    #                 'hidden_size': 64,
    #                 'n_mp_layers': 5,
    #                 'noise_level': 0,
    #                 'noise_type': 0,
    #                 'radius': 0.075,
    #                 'dataset': '230902_30',
    #                 'nparticles': 3000,
    #                 'nparticle_types': 3,
    #                 'nframes': 200,
    #                 'sigma': .005,
    #                 'tau': 0.1,
    #                 'aggr_type' : 'mean',
    #                 'particle_embedding': True,
    #                 'boundary': 'periodic',  # periodic   'no'  # no boundary condition
    #                 'data_augmentation' : True,
    #                 'embedding_type': 'repeat',
    #                 'model': 'InteractionParticles',
    #                 'upgrade_type':0}
    #
    # model_config = {'ntry': 37,
    #                 'input_size': 9,
    #                 'output_size': 2,
    #                 'hidden_size': 64,
    #                 'n_mp_layers': 5,
    #                 'noise_level': 0,
    #                 'noise_type': 0,
    #                 'radius': 0.075,
    #                 'dataset': '230902_30',
    #                 'nparticles': 3000,
    #                 'nparticle_types': 3,
    #                 'nframes': 200,
    #                 'sigma': .005,
    #                 'tau': 0.1,
    #                 'aggr_type' : 'mean',
    #                 'particle_embedding': True,
    #                 'boundary': 'periodic',  # periodic   'no'  # no boundary condition
    #                 'data_augmentation' : True,
    #                 'embedding_type': 'none',
    #                 'model': 'InteractionParticles',
    #                 'upgrade_type':0}

    # model_config = {'ntry': 39,
    #                 'input_size': 9,
    #                 'output_size': 2,
    #                 'hidden_size': 64,
    #                 'n_mp_layers': 5,
    #                 'noise_level': 0,
    #                 'noise_type': 0,
    #                 'radius': 0.075,
    #                 'dataset': '230902_39',
    #                 'nparticles': 3000,
    #                 'nparticle_types': 3,
    #                 'nframes': 200,
    #                 'sigma': .005,
    #                 'tau': 0.1,
    #                 'aggr_type' : 'mean',
    #                 'particle_embedding': True,
    #                 'boundary': 'periodic',  # periodic   'no'  # no boundary condition
    #                 'data_augmentation' : True,
    #                 'embedding_type': 'none',
    #                 'embedding': 2,
    #                 'model': 'InteractionParticles',
    #                 'upgrade_type':0}

    # model_config = {'ntry': 40,
    #                 'input_size': 9,
    #                 'output_size': 2,
    #                 'hidden_size': 64,
    #                 'n_mp_layers': 5,
    #                 'noise_level': 0,
    #                 'noise_type': 0,
    #                 'radius': 0.075,
    #                 'dataset': '230902_40',
    #                 'nparticles': 3000,
    #                 'nparticle_types': 3,
    #                 'nframes': 200,
    #                 'sigma': .005,
    #                 'tau': 0.1,
    #                 'aggr_type' : 'mean',
    #                 'particle_embedding': True,
    #                 'boundary': 'periodic',  # periodic   'no'  # no boundary condition
    #                 'data_augmentation' : True,
    #                 'embedding_type': 'none',
    #                 'model': 'InteractionParticles',
    #                 'upgrade_type':0}

    model_config = {'ntry': 41,
                    'input_size': 13,
                    'output_size': 2,
                    'hidden_size': 64,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.075,
                    'dataset': '230902_41',
                    'nparticles': 3000,
                    'nparticle_types': 3,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'embedding_type': 'none',
                    'embedding': 3,
                    'model': 'MixInteractionParticles',
                    'upgrade_type':0}

    model_config = {'ntry': 42,
                    'input_size': 10,
                    'output_size': 2,
                    'hidden_size': 64,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.075,
                    'dataset': '230902_39',
                    'nparticles': 3000,
                    'nparticle_types': 3,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'embedding_type': 'none',
                    'embedding': 3,
                    'model': 'InteractionParticles',
                    'upgrade_type':0}

    model_config = {'ntry': 43,
                    'input_size': 13,
                    'output_size': 2,
                    'hidden_size': 128,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.075,
                    'dataset': '230902_43',
                    'nparticles': 4800,
                    'nparticle_types': 3,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'embedding_type': 'none',
                    'embedding': 3,
                    'model': 'MixInteractionParticles',
                    'upgrade_type':0}

    model_config = {'ntry': 44,
                    'input_size': 13,
                    'output_size': 2,
                    'hidden_size': 64,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.075,
                    'dataset': '230902_41',
                    'nparticles': 3000,
                    'nparticle_types': 3,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'embedding_type': 'none',
                    'embedding': 3,
                    'model': 'MixInteractionParticles',
                    'upgrade_type':0}

    model_config = {'ntry': 45,
                    'input_size': 23,
                    'output_size': 2,
                    'hidden_size': 128,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.075,
                    'dataset': '230902_43',
                    'nparticles': 4800,
                    'nparticle_types': 3,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'embedding_type': 'none',
                    'embedding': 8,
                    'model': 'MixInteractionParticles',
                    'upgrade_type':0}

    model_config = {'ntry': 46,
                    'input_size': 23,
                    'output_size': 2,
                    'hidden_size': 128,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.075,
                    'dataset': '230902_46',
                    'nparticles': 9600,
                    'nparticle_types': 3,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'embedding_type': 'none',
                    'embedding': 8,
                    'model': 'MixInteractionParticles',
                    'upgrade_type':0}

    # model_config = {'ntry': 48,
    #                 'input_size': 13,
    #                 'output_size': 3,
    #                 'hidden_size': 64,
    #                 'n_mp_layers': 5,
    #                 'noise_level': 0,
    #                 'noise_type': 0,
    #                 'radius': 0.125,
    #                 'dataset': '230902_48',
    #                 'nparticles': 9600,
    #                 'nparticle_types': 3,
    #                 'nframes': 200,
    #                 'sigma': .005,
    #                 'tau': 0.25,
    #                 'aggr_type' : 'mean',
    #                 'particle_embedding': True,
    #                 'boundary': 'no',  # periodic   'no'  # no boundary condition
    #                 'data_augmentation' : True,
    #                 'embedding_type': 'none',
    #                 'model': 'InteractionParticles3D',
    #                 'embedding': 8,
    #                 'upgrade_type':0}

    model_config = {'ntry': 49,
                    'input_size': 13,
                    'output_size': 2,
                    'hidden_size': 64,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.075,
                    'dataset': '230902_49',
                    'nparticles': 4800,
                    'nparticle_types': 3,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'embedding_type': 'none',
                    'embedding': 3,
                    'model': 'MixInteractionParticles',
                    'upgrade_type':1}


    gtest_list=[10,11,15]

    for gtest in range(10):

        ntry = 59 + gtest
        model_config['ntry'] = ntry
        # model_config['nparticles'] = 3000
        # model_config['noise_level'] =  gtest_list[gtest%4] / 100
        # model_config['noise_type'] = 1 + gtest // 4
        # ntry = model_config['ntry']
        # model_config['input_size'] = gtest_list[gtest]
        # model_config['ntry'] = ntry
        # model_config['hidden_size'] = gtest_list[gtest]
        # dataset_name = model_config['dataset']
        dataset_name = '230902_' + str(49 + gtest)
        model_config['dataset'] = dataset_name
        # model_config['nparticles'] = gtest_list[gtest]

        folder = f'./graphs_data/graphs_particles_{dataset_name}/'
        os.makedirs(folder, exist_ok=True)

        sigma = model_config['sigma']
        aggr_type = model_config['aggr_type']

        scaler = StandardScaler()
        S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

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
        np_i =int(model_config['nparticles'] / model_config['nparticle_types'])
        for n in range(model_config['nparticle_types']):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

        time.sleep(0.5)

        print_model_config(model_config)
        # data_generate(model_config, index_particles)
        # data_train(model_config, index_particles)
        data_test(model_config, index_particles, prev_nparticles=0, new_nparticles=0, prev_index_particles=0, bVisu = True)

        # prev_nparticles, new_nparticles, prev_index_particles, index_particles = data_test_generate(model_config, index_particles)
        # data_test(model_config, index_particles, prev_nparticles, new_nparticles, prev_index_particles, bVisu = True)

        # data_train_generate(model_config, index_particles, 'geomloss', f'./graphs_data/graphs_particles_230902_43/')
        # data_train_generate(model_config, index_particles, 'backward', f'./graphs_data/graphs_particles_230902_43/')





