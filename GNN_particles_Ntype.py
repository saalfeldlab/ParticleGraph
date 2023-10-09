
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
from torch_geometric.loader import DataLoader
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
from matplotlib import cm
import torch_geometric.transforms as T

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
def psi_A(r, p):
    return r*(-p[2] * torch.exp(-r ** (2*p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(-r ** (2*p[1]) / (2 * sigma ** 2)))
def psi_B(r, p):
    return (-p[2] * torch.exp(-r ** (2*p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(-r ** (2*p[1]) / (2 * sigma ** 2)))
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
from torch_geometric.utils import degree

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
class InteractionParticles_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

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
        # pp = self.p[None,:]
        psi = - pp[:,2] * torch.exp(-r ** pp[:,0] / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** pp[:,1] / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2])
class InteractionParticles_B(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_B, self).__init__(aggr=aggr_type)  # "mean" aggregation.

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
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1))# distance
        r = torch.clamp(r, min=0.00005)

        pp = self.p[x_i[:, 5].detach().cpu().numpy(),:]
        # pp = self.p[None,:]
        psi = - pp[:,2] * torch.exp(-r ** (2*pp[:,0]) / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** (2*pp[:,1]) / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) / r[:,None]
class InteractionParticles_C(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_C, self).__init__(aggr=aggr_type)  # "mean" aggregation.

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
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1))  # distance
        r = torch.clamp(r, min=0.00005)

        # psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))

        pp = torch.squeeze(self.p[x_i[:, 5].detach().cpu().numpy(),x_j[:, 5].detach().cpu().numpy(),:])
        psi = - pp[:,2] * torch.exp(-r ** (2*pp[:,0]) / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** (2*pp[:,1]) / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) / r[:, None]
class InteractionParticles_D(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_D, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1))
        r = torch.clamp(r,min=0.005)
        r = torch.concatenate((r[:,None],r[:,None]),-1)


        p = self.p[x_j[:, 4].detach().cpu().numpy()]
        p = p.squeeze()
        p = torch.concatenate((p[:, None], p[:, None]), -1)

        acc = p * bc_diff(x_j[:, 0:2] - x_i[:, 0:2]) / r**3

        return acc
class InteractionParticles_E(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_E, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1))
        r = torch.clamp(r,min=0.005)
        r = torch.concatenate((r[:,None],r[:,None]),-1)


        p1 = self.p[x_i[:, 4].detach().cpu().numpy()]
        p1 = p1.squeeze()
        p1 = torch.concatenate((p1[:, None], p1[:, None]), -1)

        p2 = self.p[x_j[:, 4].detach().cpu().numpy()]
        p2 = p2.squeeze()
        p2 = torch.concatenate((p2[:, None], p2[:, None]), -1)

        acc = - p1 * p2 * bc_diff(x_j[:, 0:2] - x_i[:, 0:2]) / r**3

        return acc
class InteractionParticles_F(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_F, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)  # squared distance
        r = torch.sqrt(r)

        # psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))
        pp = self.p[x_i[:, 5].detach().cpu().numpy(),:]
        # pp = self.p[None,:]
        psi = - pp[:,2] * torch.exp(-r ** (2*pp[:,0]) / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** (2*pp[:,1]) / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2])
class InteractionParticles_G(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[]):
        super(InteractionParticles_G, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1))  # distance
        r = torch.clamp(r, min=0.00005)

        # psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))

        pp = torch.squeeze(self.p[x_i[:, 5].detach().cpu().numpy(),x_j[:, 5].detach().cpu().numpy(),:])
        psi = - pp[:,2] * torch.exp(-r ** (2*pp[:,0]) / (2 * sigma ** 2)) + pp[:,3] * torch.exp(-r ** (2*pp[:,1]) / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) / r[:, None]

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

        # self.lin_update = MLP(input_size=3, output_size=2, nlayers=2,hidden_size=8, device=self.device)

        # self.lin_acc = MLP(input_size=4+self.embedding, output_size=self.output_size, nlayers=3,
        #                     hidden_size=self.hidden_size, device=self.device)

        if self.embedding_type == 'none':
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), self.embedding)), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 2)), device=self.device, requires_grad=True, dtype=torch.float32))

        # self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        # self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        # self.a_bf_kmean = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device='cuda:0', requires_grad=False))


    def forward(self, data, step, vnorm, cos_phi, sin_phi):

        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        acc = self.propagate(edge_index, x=(x, x))

        deg = pyg_utils.degree(edge_index[0], data.num_nodes)
        deg = (deg > 0)
        deg = (deg > 0).type(torch.float32)


        if step == 2:
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)# test, if degree = 0 acc =0
            return deg * acc
        else:
            # acc = self.lin_update(torch.cat((acc, deg[:,None]), dim=-1))
            return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 0:2] - x_j[:, 0:2]) / self.radius
        x_i_vx = x_i[:, 2:3] / self.vnorm[4]
        x_i_vy = x_i[:, 3:4] / self.vnorm[5]
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
            x_i_type = x_i[:, 4:6]
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type),dim=-1)

        return self.lin_edge(in_features)

        t = self.a.detach().cpu().numpy()

        t = [-1.7, 0.67, 1.22, 3.96 ]
        X = np.arange(-1, 1, 0.02) * 0.075
        VX = (np.arange(-1, 1, 0.02) - 0.5) / 0.5 * 2
        X, VX = np.meshgrid(X, VX)
        X_ = X.reshape(10000, 1)
        VX_ = VX.reshape(10000, 1)
        X_ = torch.tensor(X_, device=self.device)
        VX_ = torch.tensor(VX_, device=self.device)
        fig = plt.figure(figsize=(16, 8))
        plt.ion()
        for k,emb in enumerate (t):
            embedding = torch.tensor(emb, device=self.device) * torch.ones((10000,1), device=self.device)
            in_features = torch.cat((X_ , 0*X_, X_ , 0*VX_, 0*VX_, VX_, 0*VX_, embedding),dim=1)   # VX, 0*VX, 3.96*
            acc_mess = self.lin_edge(in_features.float())
            acc_mess = acc_mess.detach().cpu().numpy()
            acc_messx = acc_mess[:, 0:1].reshape(100, 100)
            ax = fig.add_subplot(2, 4, k+1, projection='3d')
            surf = ax.plot_surface(X, VX, acc_messx, cmap=cm.coolwarm, linewidth=0, antialiased=True, vmin=-5,vmax=5)
            ax.set_xlabel('Distance',fontsize=14)
            ax.set_ylabel('Velocity',fontsize=14)
            ax.set_zlabel('Acceleration',fontsize=14)
            ax.set_zlim(-10, 10)
            in_features = torch.cat((X_, 0*X_, X_ , 0*VX_, 0*VX_, 0*VX_, VX_, embedding),dim=1)   # VX, 0*VX, 3.96*
            acc_mess = self.lin_edge(in_features.float())
            acc_mess = acc_mess.detach().cpu().numpy()
            acc_messx = acc_mess[:, 1:2].reshape(100, 100)
            ax = fig.add_subplot(2, 4, 4 + k + 1, projection='3d')
            surf = ax.plot_surface(X, VX, acc_messx,cmap=cm.coolwarm, linewidth=0, antialiased=True, vmin=-5,vmax=5)
            ax.set_xlabel('Distance',fontsize=14)
            ax.set_ylabel('Velocity',fontsize=14)
            ax.set_zlabel('Acceleration',fontsize=14)
            ax.set_zlim(-2, 2)


    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)
class GravityParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(GravityParticles, self).__init__(aggr='add')  # "Add" aggregation.

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
        # self.lin_acc = MLP(input_size=4+self.embedding, output_size=self.output_size, nlayers=3,
        #                     hidden_size=self.hidden_size, device=self.device)

        if self.embedding_type == 'none':
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), self.embedding)), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 2)), device=self.device, requires_grad=True, dtype=torch.float32))

        # self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        # self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        # self.a_bf_kmean = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device='cuda:0', requires_grad=False))


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
                embedding = self.a[x_j[:, 6].detach().cpu().numpy(), :]     # depends on other
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
            x_i_type = x_i[:, 4:6]
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type),dim=-1)

        return self.lin_edge(in_features)

        t = self.a.detach().cpu().numpy()


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
        # self.lin_acc = MLP(input_size=4+self.embedding, output_size=self.output_size, nlayers=3,
        #                     hidden_size=self.hidden_size, device=self.device)

        if self.embedding_type == 'none':
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), self.embedding)), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(np.ones((int(self.nparticles), 2)), device=self.device, requires_grad=True, dtype=torch.float32))

        # self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        # self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        # self.a_bf_kmean = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device='cuda:0', requires_grad=False))


    def forward(self, data, step, nloop, vnorm, ynorm, cos_phi, sin_phi):

        self.vnorm = vnorm
        self.ynorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        for loop in range(nloop):
            distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([self.radius ** 2])  # threshold
            adj_t = (distance < self.radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)

            acc = self.propagate(edge_index, x=(x, x))
            if (self.data_augmentation):
                new_acc = cos_phi * acc[:, 0] - sin_phi * acc[:, 1]
                new_acc = sin_phi * acc[:, 0] + cos_phi * acc[:, 1]
                acc[:, 0] = new_acc
                acc[:, 1] = new_acc
            acc_norm = torch.sqrt(acc[:,0]**2+acc[:,1]**2)
            t = torch.min(acc_norm,ynorm[4]*3)/acc_norm
            t = torch.permute(t.repeat(2, 1),(1,0))
            acc = acc*t

            x[:, 2:4] = x[:, 2:4] + acc
            x[:, 0:2] = bc_pos(x[:, 0:2] + x[:, 2:4])

        return x[:, 0:4]

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
                embedding = torch.clamp(self.a[x_i[:, 6].detach().cpu().numpy(), :],min=-4,max=4)
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

        # self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        # self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))


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
                # embedding = torch.abs(self.a[x_i[:, 6].detach().cpu().numpy(), :] - self.a[x_j[:, 6].detach().cpu().numpy(), :])
                # in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)
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

class EdgeNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self):
        super().__init__(aggr=aggr_type)  # "mean" aggregation.

    def forward(self, x, edge_index, radius, vnorm):

        self.radius = radius
        self.vnorm = vnorm
        aggr = self.propagate(edge_index, x=(x, x))

        return self.new_edges

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum((x_i[:, 0:2] - x_j[:, 0:2]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = (x_i[:, 0:2] - x_j[:, 0:2]) / self.radius
        x_i_vx = x_i[:, 2:3] / self.vnorm[4]
        x_i_vy = x_i[:, 3:4] / self.vnorm[5]
        x_i_type = x_i[:, 4]
        x_j_vx = x_j[:, 2:3] / self.vnorm[4]
        x_j_vy = x_j[:, 3:4] / self.vnorm[5]

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
        self.nparticles = model_config['nparticles']
        self.edge_init = EdgeNetwork()
        self.radius = model_config['radius']

        # self.layer = torch.nn.ModuleList(
        #     [InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device) for _ in
        #      range(self.nlayers)])

        self.layer = InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device)

        self.node_out = MLP(input_size=self.embedding, hidden_size=self.hidden_size, output_size=2, nlayers=3,
                            device=self.device)

        self.embedding_node = MLP(input_size=4+self.embedding, hidden_size=self.embedding, output_size=self.embedding, nlayers=3,
                                  device=self.device)
        self.embedding_edges = MLP(input_size=11, hidden_size=self.embedding, output_size=self.embedding, nlayers=3,
                                   device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((self.nparticles, self.embedding)), device=self.device, requires_grad=True))


    def forward(self, data, vnorm):

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        x[:, 2:4] = x[:, 2:4] / vnorm[4].float()

        node_feature = torch.cat((x[:, 0:4], self.a[x[:, 6].detach().cpu().numpy(), 0:2].float()), dim=-1)

        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]), requires_grad=False,
                            device=self.device) * self.noise_level
        node_feature = node_feature + noise
        edge_feature = self.edge_init(node_feature, edge_index, self.radius, vnorm)

        node_feature = self.embedding_node(node_feature)
        edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer(node_feature, edge_index, edge_feature=edge_feature)

        pred = self.node_out(node_feature)

        return pred

class MixResNetGNN(torch.nn.Module):
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

def data_generate(model_config):

    print('')
    print('Generating data ...')

    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_data/*")
    # for f in files:
    #     os.remove(f)

    dataset_name = model_config['dataset']
    folder = f'./graphs_data/graphs_particles_{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(f"{folder}/*")
    for f in files:
        os.remove(f)

    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    json_ = json.dumps(model_config)
    f = open(f"{folder}/model_config.json", "w")
    f.write(json_)
    f.close()

    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    v_init = model_config['v_init']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if model_config['model'] == 'InteractionParticles_A':
        print(f'Generate InteractionParticles_A')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius*2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        p[2] = torch.tensor([1.785, 1.8579, 1.7226, 1.0584])
        for n in range(nparticle_types):
            psi_output.append(psi_A(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_A(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'InteractionParticles_B':
        print(f'Generate InteractionParticles_B')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius*2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        p[2] = torch.tensor([1.785, 1.8579, 1.7226, 1.0584])
        for n in range(nparticle_types):
            psi_output.append(psi_B(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_B(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'MixInteractionParticles':
        print(f'Generate MixInteractionParticles')

        p = torch.ones(nparticle_types, nparticle_types, 4, device=device) + torch.rand(nparticle_types,nparticle_types, 4, device=device)

        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.075, 100))
        rr = rr.to(device)

        # read previous data
        # for n in range(nparticle_types):
        #     p[n] = torch.load(f'graphs_data/graphs_particles_230902_30/p_{n}.pt')
        #     print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
        # p[2]=p[1]*0.975
        #

        p[0 , 0] = torch.tensor([1.0696,1.8843,1.322,1.252])
        p[0, 1] = torch.tensor([1.7112,1.7178,1.108,1.471])
        p[0, 2] = torch.tensor([1.8224,1.4711,1.7202,1.2569])
        p[1, 1] = torch.tensor([1.078,1.3741,1.053,1.0633])
        p[1, 2] = torch.tensor([1.0395,1.8933,1.5266,1.5097])
        p[2, 2] = torch.tensor([1.0833,1.2819,1.6062,1.0675])

        for n in range(nparticle_types):
            for m in range(nparticle_types):
                # p[m,n] = p[n,m]
                psi_output.append(psi_A(rr, torch.squeeze(p[n, m])))
                print(f'p{n, m}: {np.round(torch.squeeze(p[n, m]).detach().cpu().numpy(), 4)}')
                torch.save(torch.squeeze(p[n, m]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}_{m}.pt')

        model = InteractionParticles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    elif model_config['model'] == 'GravityParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius * 1.2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([5])
        p[1] = torch.tensor([1])
        p[2] = torch.tensor([0.2])
        print(p)
        for n in range(nparticle_types):
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_D(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'ElecParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius * 1.2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([-1])
        p[1] = torch.tensor([1])
        p[2] = torch.tensor([2])
        print(p)
        for n in range(nparticle_types):
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_E(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'InteractionParticles_F':
        print(f'Generate InteractionParticles_F')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius*2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        p[2] = torch.tensor([1.785, 1.8579, 1.7226, 1.0584])
        for n in range(nparticle_types):
            psi_output.append(psi_A(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_F(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    else:
        print('Pb model unknown')

    if False:

        print(f'Generate InteractionParticles')

        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius*2, 100))
        rr = rr.to(device)

        p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        p[2] = torch.tensor([1.9548, 1.0472, 1.994, 1.2743])
        for n in range(nparticle_types):
            # model.append(InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p[n]), tau=tau))
            # torch.save({'model_state_dict': model[n].state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model_{n}.pt')
            psi_output.append(psi_A(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            # torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        plt.ion()
        for n in range(nparticle_types):
            #plt.plot(rr.detach().cpu().numpy(), rr.detach().cpu().numpy() *  np.array(psi_output[n].cpu()), linewidth=1)
            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
            plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                     linewidth=0.5)

    time.sleep(0.5)

    for run in range(10):

        X1 = torch.rand(nparticles, 2, device=device)
        X1t = torch.zeros((nparticles, 2, nframes))  # to store all the intermediate time

        V1 = v_init * torch.randn((nparticles, 2), device=device)

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
                #plt.ion()
                ax = fig.add_subplot(1, 2, 1)
                if model_config['model'] == 'GravityParticles':
                    for n in range(nparticle_types):
                        g=p[T1[index_particles[n],0].detach().cpu().numpy()].detach().cpu().numpy()*10
                        plt.scatter(X1t[index_particles[n], 0, it], X1t[index_particles[n], 1, it], s=g) #, facecolors='none', edgecolors='k')
                elif model_config['model'] == 'ElecParticles':
                    for n in range(nparticle_types):
                        g=np.abs(p[T1[index_particles[n],0].detach().cpu().numpy()].detach().cpu().numpy()*20)
                        if n==0:
                            plt.scatter(X1t[index_particles[n], 0, it], X1t[index_particles[n], 1, it], s=g, c='r') #, facecolors='none', edgecolors='k')
                        else:
                            plt.scatter(X1t[index_particles[n], 0, it], X1t[index_particles[n], 1, it], s=g, c='b') #, facecolors='none', edgecolors='k')
                else:
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
                        for m in range(nparticle_types):
                            plt.text(-0.25, 1.25 - N * 0.05, f'p{n}{m}: {np.round(p[n,m].detach().cpu().numpy(), 4)}',color='k')
                            N+=1
                else:
                    for n in range(nparticle_types):
                        plt.text(-0.25, 1.25 - n * 0.05, f'p{n}: {np.round(p[n].detach().cpu().numpy(), 4)}',
                                 color='k')

                ax = fig.add_subplot(1, 2, 2)
                for n in range(nparticle_types):
                    plt.scatter(X1t[:, 0, it], X1t[:, 1, it], s=1, color='k')
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.xlim([-0.3, 1.3])
                plt.ylim([-0.3, 1.3])

                if model_config['model'] != 'GravityParticles':
                    ax = fig.add_subplot(5, 5, 21)
                if model_config['model'] == 'MixInteractionParticles':
                    N = 0
                    for n in range(nparticle_types):
                        for m in range(nparticle_types):
                            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[N].cpu()), linewidth=1)
                            plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0,
                                     color=[0, 0, 0], linewidth=0.5)
                            N += 1
                if (model_config['model'] == 'InteractionParticles_A') | (model_config['model'] == 'InteractionParticles_B') | (model_config['model'] == 'InteractionParticles_F'):
                    for n in range(nparticle_types):
                        plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                        plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],linewidth=0.5)

                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

def data_train(model_config,gtest):

    print('')
    print('Training loop ...')

    model=[]
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
    noise_type = model_config['noise_type']
    embedding_type = model_config['embedding_type']
    embedding = model_config['embedding']
    batch_size = model_config['batch_size']
    batch_size = 1

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

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

    vnorm = norm_velocity(xx, device)
    ynorm = norm_acceleration(yy, device)

    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if (model_config['model'] == 'InteractionParticles_A') | (model_config['model'] == 'InteractionParticles_B') | (model_config['model'] == 'InteractionParticles_F'):
        if training_mode == 'regressive_loop':
            model = InteractionParticlesLoop(model_config, device)
            print(f'Training InteractionParticles regressive loop')
        else:
            model = InteractionParticles(model_config, device)
            print(f'Training InteractionParticles')
    if (model_config['model'] == 'MixInteractionParticles') | (model_config['model'] == 'ElecParticles'):
        model = MixInteractionParticles(model_config, device)
        print(f'Training MixInteractionParticles')
    if model_config['model'] == 'ResNetGNN':
        model = ResNetGNN(model_config, device)
        print(f'Training ResNetGNN')
    if model_config['model'] == 'MixResNetGNN':
        model = MixResNetGNN(model_config, device)
        print(f'Training MixResnet')

    # net = f"./log/try_{ntry}/models/best_model_with_{NGraphs-1}_graphs.pt"
    # state_dict = torch.load(net)
    # model.load_state_dict(state_dict['model_state_dict'])

    lra=1E-2
    lr=1E-3

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    it=0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if it==0:
            optimizer = torch.optim.Adam([model.a], lr=lra)
        else:
            optimizer.add_param_group({'params': parameter, 'lr': lr})
        it+=1
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f'Learning rates: {lr}, {lra}')
    print('')
    net = f"./log/try_{ntry}/models/best_model_with_{NGraphs-1}_graphs.pt"
    print(f'network: {net}')
    print('')

    # model.a_bf_kmean = nn.Parameter(
    # torch.tensor(np.ones((int(model.nparticles), 2)), device=model.device, requires_grad=False))
    # model.p0 = nn.Parameter(torch.tensor(np.ones(4), device=model.device, requires_grad=False))
    # model.p1 = nn.Parameter(torch.tensor(np.ones(4), device=model.device, requires_grad=False))
    # delattr(model, "lin_acc")
    # net = f"./log/try_39/models/best_model_with_{NGraphs - 1}_graphs.pt"
    # state_dict = torch.load(net)
    # model.load_state_dict(state_dict['model_state_dict'])

    time.sleep(0.5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=weight_decay)
    Nepochs=60
    print(f'N epochs: {Nepochs}')
    model.train()
    best_loss = np.inf

    # optimizer = torch.optim.Adam([model.a], lr=0.01)
    # optimizer.add_param_group({'params': model.lin_edge.layers[0].weight, 'lr': lr})
    # optimizer.add_param_group({'params': model.lin_edge.layers[0].bias, 'lr': lr})

    if data_augmentation:
        data_augmentation_loop = 20
        print(f'data_augmentation_loop: {data_augmentation_loop}')
    else:
        data_augmentation_loop = 1
        print('no data augmentation ...')

    list_loss = []
    list_gap = []
    embedding_list=[]
    D_nm = torch.zeros((Nepochs+1,nparticle_types, nparticle_types))

    print('')
    time.sleep(0.5)
    for epoch in range(Nepochs+1):

        if epoch == 10:
            batch_size = model_config['batch_size']
            print(f'batch_size: {batch_size}')
        if epoch == 20:
            lra = 1E-3
            lr = 2E-4
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
            if data_augmentation:
                data_augmentation_loop = 200
                print(f'data_augmentation_loop: {data_augmentation_loop}')
        if epoch == 30:
            print('training MLP only ...')
            model.a.requires_grad = False
            # new_a = kmeans.cluster_centers_[kmeans.labels_, :]
            # if gap < 100:
            #     model.a.data = torch.tensor(new_a, device=device)

        total_loss = 0

        if training_mode == 't+1':
            for N in range(1, nframes * data_augmentation_loop // batch_size ):

                phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)

                run = 1 + np.random.randint(NGraphs - 1)

                batch_size = 8

                dataset_batch = []
                for batch in range(batch_size):

                    k = np.random.randint(nframes - 1)
                    x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k}.pt').to(device)
                    distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                    adj_t = (distance < radius ** 2).float() * 1
                    t = torch.Tensor([radius ** 2])
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)
                    dataset_batch.append(dataset)

                    y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{k}.pt')
                    y = y.to(device)
                    y[:, 0] = y[:, 0] / ynorm[4]
                    y[:, 1] = y[:, 1] / ynorm[5]
                    if model_config['model'] == 'InteractionParticles3D':
                        y[:, 2] = y[:, 2] / ynorm[6]
                    if data_augmentation:
                        new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                        new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                        y[:, 0] = new_x
                        y[:, 1] = new_y
                    if batch==0:
                        y_batch=y
                    else:
                        y_batch=torch.cat((y_batch, y), axis=0)

                batch_loader = DataLoader(dataset_batch, batch_size=8, shuffle=False)
                optimizer.zero_grad()

                for batch in batch_loader:
                    if model_config['model'] == 'ResNetGNN':
                        pred = model(batch, vnorm=vnorm)
                    else:
                        pred = model(batch, step = 1, vnorm=vnorm, cos_phi=cos_phi, sin_phi=sin_phi)

                loss = (pred - y_batch).norm(2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        elif training_mode == 'regressive':

            regressive_step = 5

            for N in range(1, nframes * data_augmentation_loop // regressive_step ):

                k = np.random.randint(nframes - 1-  regressive_step)
                run = 1 + np.random.randint(NGraphs - 1)
                x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k}.pt')
                x = x.to(device)

                for regressive_loop in range(regressive_step):

                    phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                    cos_phi = torch.cos(phi)
                    sin_phi = torch.sin(phi)

                    distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                    adj_t = (distance < radius ** 2).float() * 1
                    t = torch.Tensor([radius ** 2])
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)

                    y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{k+regressive_loop}.pt')
                    y = y.to(device)
                    y[:, 0] = y[:, 0] / ynorm[4]
                    y[:, 1] = y[:, 1] / ynorm[5]

                    if data_augmentation:
                        new_yx = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                        new_yy = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                        y[:, 0] = new_yx
                        y[:, 1] = new_yy

                    optimizer.zero_grad()
                    pred = model(dataset, step = 1, vnorm=vnorm, cos_phi=cos_phi, sin_phi=sin_phi)

                    loss = (pred - y).norm(2)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    y[:, 0] = y[:, 0] * ynorm[4]
                    y[:, 1] = y[:, 1] * ynorm[5]
                    x[:, 2:4] = x[:, 2:4] + y  # speed update
                    x[:, 0:2] = bc_pos(x[:, 0:2] + x[:, 2:4])  # position update

        elif training_mode == 'regressive_loop':

            regressive_step = 5
            for N in range(1, nframes // regressive_step * data_augmentation_loop):

                phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)

                k = np.random.randint(nframes - 1-  regressive_step)
                run = 1 + np.random.randint(NGraphs - 1)
                x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k}.pt')
                x = x.to(device)

                distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                adj_t = (distance < radius ** 2).float() * 1
                t = torch.Tensor([radius ** 2])
                edges = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x[:, :], edge_index=edges)

                optimizer.zero_grad()
                pred = model(dataset, step = 1, nloop=regressive_step, vnorm=vnorm, ynorm=ynorm, cos_phi=cos_phi, sin_phi=sin_phi)

                y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k+regressive_step}.pt')
                y = y.to(device)

                # loss = bc_diff(pred[:,0:2] - y[:,0:2]).norm(2)

                loss = 1E5 * S_e(pred[:,0:2],y[:,0:2])

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        else:
            print('Pb training mode')

        model.a.data = torch.clamp(model.a.data, min=-4, max=4)
        embedding = model.a.detach().cpu().numpy()
        embedding = scaler.fit_transform(embedding)
        embedding_particle = []
        for n in range(nparticle_types):
            embedding_particle.append(embedding[index_particles[n], :])

        # kmeans = KMeans(init="random", n_clusters=nparticle_types, n_init=10, max_iter=300, random_state=42)
        # kmeans.fit(embedding)
        # gap = kmeans.inertia_
        kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
        sse = []
        # for k in range(1, 11):
        #     kmeans_ = KMeans(n_clusters=k, **kmeans_kwargs)
        #     kmeans_.fit(embedding)
        #     sse.append(kmeans_.inertia_)
        # kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
        # list_gap.append(gap)

        for n in range(nparticle_types-1):
            for m in range(n+1,nparticle_types):
                D_nm[epoch,n,m] = S_e(torch.tensor(embedding_particle[n]), torch.tensor(embedding_particle[m]))

        torch.save(D_nm, f"./tmp_training/D_nm_{ntry}.pt")

        S_geomD = torch.sum(D_nm[epoch]).item()
        # print(f'total_loss / S_geomD: {total_loss / S_geomD}  best_loss {best_loss}')

        if (total_loss / nparticles / batch_size / N < best_loss):
            best_loss = total_loss / N / nparticles / batch_size
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_with_{NGraphs-1}_graphs.pt'))
            print("Epoch {}. Loss: {:.6f} geomloss {:.2f} saving model  ".format(epoch, total_loss / N / nparticles / batch_size , S_geomD))
        else:
            print("Epoch {}. Loss: {:.6f} geomloss {:.2f} ".format(epoch, total_loss / N / nparticles / batch_size, S_geomD))

        list_loss.append(total_loss / N / nparticles / batch_size)

        fig = plt.figure(figsize=(8, 8))
        # plt.ion()

        if (embedding.shape[1]>2):
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            for n in range(nparticle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],s=1)
        else:
            ax = fig.add_subplot(2, 2, 1)
            if (embedding.shape[1] > 1):
                for n in range(nparticle_types):
                    plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], s=3)
                    plt.xlim([-2.1, 2.1])
                    plt.ylim([-2.1, 2.1])
                    plt.xlabel('Embedding 0', fontsize=12)
                    plt.ylabel('Embedding 1', fontsize=12)
                plt.xlim([-5.1, 5.1])
                plt.ylim([-5.1, 5.1])
            else:
                for n in range(nparticle_types):
                    plt.hist(embedding_particle[n][:, 0],100, alpha=0.5)
                plt.xlim([-5.1, 5.1])

        ax = fig.add_subplot(2, 2, 2)
        plt.plot(list_loss, color='k')
        plt.xlim([0, 100])
        plt.ylim([0, 0.02])
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epochs', fontsize=10)

        if (epoch%10==0) & (epoch>0):
            best_loss = total_loss / N / nparticles / batch_size
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},os.path.join(log_dir, 'models', f'best_model_with_{NGraphs-1}_graphs.pt'))
            xx, rmserr_list = data_test(model_config, bVisu=True, bPrint=False)
            model.train()

        if (epoch>9):

            ax = fig.add_subplot(2, 2, 3)
            for n in range(nparticle_types):
                plt.scatter(xx[index_particles[n], 0], xx[index_particles[n], 1], s=1)
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')

            ax = fig.add_subplot(2, 2, 4)
            plt.plot(np.arange(len(rmserr_list)), rmserr_list, label='RMSE', c='r')
            plt.ylim([0, 0.1])
            plt.xlim([0, nframes])
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.xlabel('Frame [a.u]', fontsize="14")
            ax.set_ylabel('RMSE [a.u]', fontsize="14", color='r')

        plt.tight_layout()
        plt.savefig(f"./tmp_training/Fig_{ntry}_{epoch}.tif")
        plt.close()
def data_test(model_config, bVisu=False, bPrint=True, index_particles=0, prev_nparticles=0, new_nparticles=0, prev_index_particles=0):
    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_recons/*")
    # for f in files:
    #     os.remove(f)
    if bPrint:
        print('')
        print('Plot validation test ... ')

    model = []
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']

    if index_particles == 0:
        index_particles = []
        np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
        for n in range(model_config['nparticle_types']):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if (model_config['model'] == 'InteractionParticles_A') | (model_config['model'] == 'InteractionParticles_B') | (model_config['model'] == 'InteractionParticles_F'):
        model = InteractionParticles(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
        p_mass = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_mass[n]=torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        print(p_mass)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'MixInteractionParticles':
        model = MixInteractionParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = MixInteractionParticles(model_config, device)
        p_elec = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_elec[n]=torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        print(p_elec)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'ResNetGNN':
        model = ResNetGNN(model_config, device)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_*")
    NGraphs = int(len(graph_files) / nframes)
    if bPrint:
        print('Graph files N: ', NGraphs-1)

    net = f"./log/try_{ntry}/models/best_model_with_{NGraphs-1}_graphs.pt"
    if bPrint:
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

    # arr = np.arange(0, NGraphs - 1, 2)
    # distance_list=[]
    # x_list=[]
    # y_list=[]
    # deg_list=[]
    # for run in arr:
    #     kr = np.arange(0, nframes - 1, 4)
    #     for k in kr:
    #         x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k}.pt')
    #         y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{k}.pt')
    #         x_list.append(torch.concatenate((torch.mean(x[:,2:4],axis=0),torch.std(x[:,2:4],axis=0)),axis=-1).detach().cpu().numpy())
    #         y_list.append(torch.concatenate((torch.mean(y,axis=0),torch.std(y,axis=0)),axis=-1).detach().cpu().numpy())
    #
    #         distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
    #         t = torch.Tensor([radius ** 2])  # threshold
    #         adj_t = (distance < radius ** 2).float() * 1
    #         edge_index = adj_t.nonzero().t().contiguous()
    #         dataset = data.Data(x=x, edge_index=edge_index)
    #         distance=np.sqrt(distance[edge_index[0, :],edge_index[1,:]].detach().cpu().numpy())
    #         deg = degree(dataset.edge_index[0], dataset.num_nodes)
    #         deg_list.append(deg.detach().cpu().numpy())
    #         distance_list.append([np.mean(distance),np.std(distance)])
    #
    # x_list=np.array(x_list)
    # y_list=np.array(y_list)
    # deg_list=np.array(deg_list)
    # distance_list=np.array(distance_list)
    # fig = plt.figure(figsize=(15, 5))
    # plt.ion()
    # ax = fig.add_subplot(1, 4, 4)
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0]+deg_list[:, 1], c='k')
    # plt.plot(np.arange(deg_list.shape[0])*4,deg_list[:,0],c='r')
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0]-deg_list[:, 1], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Degree [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 4, 1)
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0]+distance_list[:, 1], c='k')
    # plt.plot(np.arange(distance_list.shape[0])*4,distance_list[:,0],c='r')
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0]-distance_list[:, 1], c='k')
    # plt.ylim([0, model.radius])
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Distance [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 4, 2)
    # plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 0]+x_list[:, 2], c='k')
    # plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 0], c='r')
    # plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 0]-x_list[:, 2], c='k')
    # plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 1]+x_list[:, 3], c='k')
    # plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 1], c='r')
    # plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 1]-x_list[:, 3], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Velocity [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 4, 3)
    # plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 0]+y_list[:, 2], c='k')
    # plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 0], c='r')
    # plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 0]-y_list[:, 2], c='k')
    # plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 1]+y_list[:, 3], c='k')
    # plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 1], c='r')
    # plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 1]-y_list[:, 3], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Velocity [a.u]', fontsize="14")
    # plt.tight_layout()

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
    if bPrint:
        print(table)
        print(f"Total Trainable Params: {total_params}")

    x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_0.pt')
    x00 = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_0.pt')
    y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_0_0.pt')
    x = x.to(device)
    x00 = x00.to(device)
    y = y.to(device)

    if bPrint:
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
            if model_config['model'] == 'ResNetGNN':
                y = model(dataset, vnorm=v)
            else:
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

        if (it % 5 == 0) & bVisu:

            distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
            edge_index2 = adj_t2.nonzero().t().contiguous()
            dataset2 = data.Data(x=x, edge_index=edge_index2)

            fig = plt.figure(figsize=(25, 16))
            # plt.ion()
            ax = fig.add_subplot(2, 3, 1)

            if model_config['model'] == 'GravityParticles':
                for n in range(nparticle_types):
                    g=p_mass[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 10
                    plt.scatter(x00[index_particles[n], 0].detach().cpu(), x00[index_particles[n], 1].detach().cpu(),s=g)  # , facecolors='none', edgecolors='k')
            elif model_config['model'] == 'ElecParticles':
                for n in range(nparticle_types):
                    g = np.abs(p_elec[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)
                    if n == 0:
                        plt.scatter(x00[index_particles[n], 0].detach().cpu().numpy(), x00[index_particles[n], 1].detach().cpu().numpy(), s=g,c='r')  # , facecolors='none', edgecolors='k')
                    else:
                        plt.scatter(x00[index_particles[n], 0].detach().cpu().numpy(), x00[index_particles[n], 1].detach().cpu().numpy(), s=g,c='b')  # , facecolors='none', edgecolors='k')
            else:
                for n in range(nparticle_types):
                    plt.scatter(x00[index_particles[n], 0].detach().cpu(), x00[index_particles[n], 1].detach().cpu(), s=3)

            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, f't0 {nparticles} particles', fontsize=10)

            ax = fig.add_subplot(2, 3, 2)
            if model_config['model'] == 'GravityParticles':
                for n in range(nparticle_types):
                    g=p_mass[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 10
                    plt.scatter(x0[index_particles[n], 0].detach().cpu(), x0[index_particles[n], 1].detach().cpu(),s=g)  # , facecolors='none', edgecolors='k')
            elif model_config['model'] == 'ElecParticles':
                for n in range(nparticle_types):
                    g = np.abs(p_elec[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)
                    if n == 0:
                        plt.scatter(x0[index_particles[n], 0].detach().cpu().numpy(), x0[index_particles[n], 1].detach().cpu().numpy(), s=g,c='r')  # , facecolors='none', edgecolors='k')
                    else:
                        plt.scatter(x0[index_particles[n], 0].detach().cpu().numpy(), x0[index_particles[n], 1].detach().cpu().numpy(), s=g,c='b')  # , facecolors='none', edgecolors='k')
            else:
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
            if model_config['model'] == 'GravityParticles':
                for n in range(nparticle_types):
                    g=p_mass[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 10
                    plt.scatter(x[index_particles[n], 0].detach().cpu(), x[index_particles[n], 1].detach().cpu(),s=g)  # , facecolors='none', edgecolors='k')
            elif model_config['model'] == 'ElecParticles':
                for n in range(nparticle_types):
                    g = np.abs(p_elec[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)
                    if n == 0:
                        plt.scatter(x[index_particles[n], 0].detach().cpu().numpy(), x[index_particles[n], 1].detach().cpu().numpy(), s=g,c='r')  # , facecolors='none', edgecolors='k')
                    else:
                        plt.scatter(x[index_particles[n], 0].detach().cpu().numpy(), x[index_particles[n], 1].detach().cpu().numpy(), s=g,c='b')  # , facecolors='none', edgecolors='k')
            else:
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

            if (embedding.shape[1] > 1):

                for n in range(nparticle_types):
                    embedding_particle.append(embedding[index_particles[n], :])
                    plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], s=3)
                plt.xlim([-4.1, 4.1])
                plt.ylim([-4.1, 4.1])
                plt.xlabel('Embedding 0', fontsize=8)
                plt.ylabel('Embedding 1', fontsize=8)

            else:
                for n in range(nparticle_types):
                    embedding_particle.append(embedding[index_particles[n], :])
                    plt.hist(embedding_particle[n][:, 0], 50, alpha=0.5)
                    plt.xlabel('Embedding', fontsize=14)

            plt.savefig(f"./tmp_recons/Fig_{ntry}_{it}.tif")

            plt.close()

    if bPrint:
        print('')
        print(f'ntry: {ntry}')
    print(f'RMSE: {np.round(rmserr.item(),4)}')
    if bPrint:
        print(f'MMD: {np.round(discrepency,4)}')
    # print(f'Final Sxy: {Sxy.item()}')

    return x.detach().cpu().numpy(), rmserr_list
def data_test_generate(model_config):

    print('')
    print('Generating test data ...')


    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    tau = model_config['tau']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if model_config['model'] == 'InteractionParticles_A':
        print(f'Generate InteractionParticles_A')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius*2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        p[2] = torch.tensor([1.785, 1.8579, 1.7226, 1.0584])
        for n in range(nparticle_types):
            psi_output.append(psi_A(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_A(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'InteractionParticles_B':
        print(f'Generate InteractionParticles_B')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius*2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
        p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
        p[2] = torch.tensor([1.785, 1.8579, 1.7226, 1.0584])
        for n in range(nparticle_types):
            psi_output.append(psi_B(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_B(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'GravityParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, radius * 1.2, 100))
        rr = rr.to(device)
        p[0] = torch.tensor([5])
        p[1] = torch.tensor([1])
        p[2] = torch.tensor([0.2])
        print(p)
        for n in range(nparticle_types):
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_2(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'MixInteractionParticles':
        print(f'Generate MixInteractionParticles')

        p = torch.ones(nparticle_types, nparticle_types, 4, device=device) + torch.rand(nparticle_types,nparticle_types, 4, device=device)

        model = []
        psi_output = []
        rr = torch.tensor(np.linspace(0, 0.075, 100))
        rr = rr.to(device)

        # read previous data
        # for n in range(nparticle_types):
        #     p[n] = torch.load(f'graphs_data/graphs_particles_230902_30/p_{n}.pt')
        #     print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
        # p[2]=p[1]*0.975
        #
        # p[0 , 0] = torch.tensor([1.0696,1.8843,1.322,1.252])
        # p[0, 1] = torch.tensor([1.7112,1.7178,1.108,1.471])
        # p[0, 2] = torch.tensor([1.8224,1.4711,1.7202,1.2569])
        # p[1, 1] = torch.tensor([1.078,1.3741,1.053,1.0633])
        # p[1, 2] = torch.tensor([1.0395,1.8933,1.5266,1.5097])
        # p[2, 2] = torch.tensor([1.0833,1.2819,1.6062,1.0675])

        for n in range(nparticle_types):
            for m in range(nparticle_types):
                # p[m,n] = p[n,m]
                psi_output.append(psi(rr, torch.squeeze(p[n, m])))
                print(f'p{n, m}: {np.round(torch.squeeze(p[n, m]).detach().cpu().numpy(), 4)}')
                torch.save(torch.squeeze(p[n, m]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}_{m}.pt')

        model = InteractionParticles_1(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    else:
        print('Pb model unknown')

    ratio = 1
    prev_nparticles = nparticles
    prev_index_particles = index_particles

    new_nparticles = prev_nparticles * ratio
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

            ax = fig.add_subplot(1, 2, 2)
            for n in range(nparticle_types):
                plt.scatter(X1t[:, 0, it], X1t[:, 1, it], s=3, color='k')
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])

            plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
            plt.close()

    return prev_nparticles, new_nparticles, prev_index_particles, index_particles
def data_train_generate(model_config, prev_folder):

    print('')
    print('Generating training data ...')

    dataset_name = model_config['dataset']
    folder = f'./graphs_data/graphs_particles_{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(f"{folder}/*")
    for f in files:
        os.remove(f)

    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    noise_level = model_config['noise_level']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
    model = []
    psi_output = []
    rr = torch.tensor(np.linspace(0, 0.015, 100))
    rr = rr.to(device)

    # read previous data
    for n in range(nparticle_types):
        p[n] = torch.load(f'{prev_folder}/p_{n}.pt')
        print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')

    if model_config['model'] == 'InteractionParticles_A':
        print(f'Generate InteractionParticles_A')
        for n in range(nparticle_types):
            psi_output.append(psi_A(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_A(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    elif model_config['model'] == 'InteractionParticles_B':
        print(f'Generate InteractionParticles_B')
        for n in range(nparticle_types):
            psi_output.append(psi_B(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = InteractionParticles_B(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'])
        torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')
    else:
        print('Pb model unknown')

    torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    for run in range(2):
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

        noise_current = 0 * torch.randn((nparticles, 2), device=device)
        noise_prev = 0 * torch.randn((nparticles, 2), device=device)
        noise_prev_prev = 0 * torch.randn((nparticles, 2), device=device)

        for it in tqdm(range(nframes)):

            noise_prev_prev = noise_prev_prev
            noise_prev = noise_current
            noise_current = 0 * torch.randn((nparticles, 2), device=device) * noise_level * radius

            X1t[:, :, it] = X1.clone().detach()  # for later display

            X1 = bc_pos(X1 + V1)

            distance = torch.sum(bc_diff(X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            x = torch.concatenate(
                (X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), N1.clone().detach()), 1)

            x_noise = x
            x_noise[:,0:2] = x[:,0:2] + noise_current
            x_noise[:, 2:4] = x[:, 2:4] + noise_current - noise_prev

            torch.save(x_noise, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')

            dataset = data.Data(x=x, edge_index=edge_index)

            y = 0
            with torch.no_grad():
                y = model(dataset)
            y_noise = y + noise_current - 2 * noise_prev + noise_prev_prev

            torch.save(y_noise, f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{it}.pt')

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
                    plt.text(-0.25, 1.25 - n * 0.05, f'p{n}: {np.round(p[n].detach().cpu().numpy(), 4)}', color='k')

                ax = fig.add_subplot(5, 5, 21)

                for n in range(nparticle_types):
                    plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                    plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                             linewidth=0.5)

                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()
def data_plot(model_config):

    model = []
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))


    if (model_config['model'] == 'InteractionParticles_A') | (model_config['model'] == 'InteractionParticles_B') | (model_config['model'] == 'InteractionParticles_F'):
        model = InteractionParticles(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
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

    arr = np.arange(0, NGraphs - 1, 2)
    distance_list=[]
    x_list=[]
    y_list=[]
    deg_list=[]
    for run in arr:
        kr = np.arange(0, nframes - 1, 4)
        for k in kr:
            x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{k}.pt')
            y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{k}.pt')
            x_list.append(torch.concatenate((torch.mean(x[:,2:4],axis=0),torch.std(x[:,2:4],axis=0)),axis=-1).detach().cpu().numpy())
            y_list.append(torch.concatenate((torch.mean(y,axis=0),torch.std(y,axis=0)),axis=-1).detach().cpu().numpy())

            distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, edge_index=edge_index)
            distance=np.sqrt(distance[edge_index[0, :],edge_index[1,:]].detach().cpu().numpy())
            deg = degree(dataset.edge_index[0], dataset.num_nodes)
            deg_list.append(deg.detach().cpu().numpy())
            distance_list.append([np.mean(distance),np.std(distance)])

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 4, 4)
    x_list=np.array(x_list)
    y_list=np.array(y_list)
    deg_list=np.array(deg_list)
    distance_list=np.array(distance_list)

    plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0]+deg_list[:, 1], c='k')
    plt.plot(np.arange(deg_list.shape[0])*4,deg_list[:,0],c='r')
    plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0]-deg_list[:, 1], c='k')
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Degree [a.u]', fontsize="14")
    ax = fig.add_subplot(1, 4, 1)
    plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0]+distance_list[:, 1], c='k')
    plt.plot(np.arange(distance_list.shape[0])*4,distance_list[:,0],c='r')
    plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0]-distance_list[:, 1], c='k')
    plt.ylim([0, model.radius])
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Distance [a.u]', fontsize="14")
    ax = fig.add_subplot(1, 4, 2)
    plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 0]+x_list[:, 2], c='k')
    plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 0], c='r')
    plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 0]-x_list[:, 2], c='k')
    plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 1]+x_list[:, 3], c='k')
    plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 1], c='r')
    plt.plot(np.arange(x_list.shape[0]) * 4, x_list[:, 1]-x_list[:, 3], c='k')
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Velocity [a.u]', fontsize="14")
    ax = fig.add_subplot(1, 4, 3)
    plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 0]+y_list[:, 2], c='k')
    plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 0], c='r')
    plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 0]-y_list[:, 2], c='k')
    plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 1]+y_list[:, 3], c='k')
    plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 1], c='r')
    plt.plot(np.arange(y_list.shape[0]) * 4, y_list[:, 1]-y_list[:, 3], c='k')
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Acceleration [a.u]', fontsize="14")
    plt.tight_layout()
    plt.show()

    p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
    psi_output = []
    rr = torch.tensor(np.linspace(0, radius * 2, 100))
    rr = rr.to(device)
    p[0] = torch.tensor([1.0413, 1.5615, 1.6233, 1.6012])
    p[1] = torch.tensor([1.8308, 1.9055, 1.7667, 1.0855])
    p[2] = torch.tensor([1.785, 1.8579, 1.7226, 1.0584])
    for n in range(nparticle_types):
        psi_output.append(psi_A(rr, torch.squeeze(p[n])))
        print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')

    types = [1,2,3]


    fig = plt.figure(figsize=(24, 6))
    plt.ion()
    ax = fig.add_subplot(1, 4, 1)
    t = model.a.detach().cpu().numpy()
    tmean = np.ones(3)
    tstd = np.ones(3)
    for n in range(model_config['nparticle_types']):
        plt.hist(t[index_particles[n]])
        tmean[n]=np.round(np.mean(t[index_particles[n]])*1000)/1000
        tstd[n]=np.round(np.std(t[index_particles[n]])*1000)/1000
        plt.text(tmean[n], 80, f'{tmean[n]}')
        plt.text(tmean[n], 75, f'+/- {tstd[n]}')
    plt.xlabel('Embedding [a.u]', fontsize="14")
    plt.ylabel('Counts [a.u]', fontsize="14")

    ax = fig.add_subplot(1, 4, 2)
    plt.scatter(types,tmean,s=30,color='k')
    plt.xlabel('Types [a.u]', fontsize="14")
    plt.ylabel('Embedding [a.u]', fontsize="14")

    ax = fig.add_subplot(1, 4, 3)
    for n in range(nparticle_types):
        plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
        plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                 linewidth=0.5)

    tau = model_config['tau']
    ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt')
    ynorm = ynorm[4].detach().cpu().numpy()

    ax = fig.add_subplot(1, 4, 4)
    for k in range(3):
        embedding = torch.tensor(tmean[k], device=device) * torch.ones((100), device=device)
        in_features = torch.cat((rr[:,None], 0 * rr[:,None], rr[:,None], 0 * rr[:,None], 0 * rr[:,None], 0 * rr[:,None], 0 * rr[:,None], embedding[:,None]), dim=1)
        acc = model.lin_edge(in_features.float())
        acc = acc[:,0]
        plt.plot(rr.detach().cpu().numpy(), acc.detach().cpu().numpy()*ynorm/model_config['tau'])
    plt.plot(rr.detach().cpu().numpy(), 0*acc.detach().cpu().numpy() * ynorm / model_config['tau'],c='k')
    # plt.xlim([0, 0.075*2])
    plt.xlabel('Distance [a.u]', fontsize="14")
    plt.ylabel('Acceleration [a.u]', fontsize="14")

    # t = [-1.7, 0.67, 1.22, 3.96 ]
    # X = np.arange(-1, 1, 0.02) * 0.075
    # VX = (np.arange(-1, 1, 0.02) - 0.5) / 0.5 * 2
    # X, VX = np.meshgrid(X, VX)
    # X_ = X.reshape(10000, 1)
    # VX_ = VX.reshape(10000, 1)
    # X_ = torch.tensor(X_, device=device)
    # VX_ = torch.tensor(VX_, device=device)
    # fig = plt.figure(figsize=(16, 8))
    # plt.ion()
    # for k,emb in enumerate (t):
    #     embedding = torch.tensor(emb, device=device) * torch.ones((10000,1), device=device)
    #     in_features = torch.cat((X_ , 0*X_, X_ , 0*VX_, 0*VX_, VX_, 0*VX_, embedding),dim=1)   # VX, 0*VX, 3.96*
    #     acc_mess = model.lin_edge(in_features.float())
    #     acc_mess = acc_mess.detach().cpu().numpy()
    #     acc_messx = acc_mess[:, 0:1].reshape(100, 100)
    #     ax = fig.add_subplot(2, 4, k+1, projection='3d')
    #     surf = ax.plot_surface(X, VX, acc_messx, cmap=cm.coolwarm, linewidth=0, antialiased=True, vmin=-5,vmax=5)
    #     ax.set_xlabel('Distance',fontsize=14)
    #     ax.set_ylabel('Velocity',fontsize=14)
    #     ax.set_zlabel('Acceleration',fontsize=14)
    #     ax.set_zlim(-10, 10)
    #     in_features = torch.cat((X_, 0*X_, X_ , 0*VX_, 0*VX_, 0*VX_, VX_, embedding),dim=1)   # VX, 0*VX, 3.96*
    #     acc_mess = model.lin_edge(in_features.float())
    #     acc_mess = acc_mess.detach().cpu().numpy()
    #     acc_messx = acc_mess[:, 1:2].reshape(100, 100)
    #     ax = fig.add_subplot(2, 4, 4 + k + 1, projection='3d')
    #     surf = ax.plot_surface(X, VX, acc_messx,cmap=cm.coolwarm, linewidth=0, antialiased=True, vmin=-5,vmax=5)
    #     ax.set_xlabel('Distance',fontsize=14)
    #     ax.set_ylabel('Velocity',fontsize=14)
    #     ax.set_zlabel('Acceleration',fontsize=14)
    #     ax.set_zlim(-2, 2)

    plt.tight_layout()
    plt.show()
def load_model_config (id=48):

    model_config_test = []

# gravity
    if id==68:
        model_config_test = {'ntry': id,
                    'input_size': 8,
                    'output_size': 2,
                    'hidden_size': 64,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.15,
                    'dataset': '230902_68',
                    'nparticles': 960,
                    'nparticle_types': 3,
                    'nframes': 1000,
                    'sigma': .005,
                    'tau': 5E-9,
                    'v_init': 1E-4,
                    'aggr_type' : 'add',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'batch_size': 4,
                    'embedding_type': 'none',
                    'embedding': 1,
                    'model': 'GravityParticles',
                    'upgrade_type':0}

# particles
    if id==72:
        model_config_test = {'ntry': id,
                        'input_size': 8,
                        'output_size': 2,
                        'hidden_size': 64,
                        'n_mp_layers': 5,
                        'noise_level': 0,
                        'noise_type': 0,
                        'radius': 0.075,
                        'dataset': '231001_72',
                        'nparticles': 4800,
                        'nparticle_types': 3,
                        'nframes': 200,
                        'sigma': .005,
                        'tau': 0.01,
                        'v_init': 0,
                        'aggr_type' : 'mean',
                        'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                        'data_augmentation' : True,
                        'batch_size' :4,
                        'particle_embedding': True,
                        'embedding_type': 'none',
                        'embedding': 1,
                        'model': 'InteractionParticles_B',
                        'upgrade_type':0}
    if id==73:
        model_config_test = {'ntry': id,
                        'input_size': 8,
                        'output_size': 2,
                        'hidden_size': 128,
                        'n_mp_layers': 5,
                        'noise_level': 0,
                        'noise_type': 0,
                        'radius': 0.075,
                        'dataset': '231001_72',
                        'nparticles': 4800,
                        'nparticle_types': 3,
                        'nframes': 200,
                        'sigma': .005,
                        'tau': 0.01,
                        'v_init': 0,
                        'aggr_type' : 'mean',
                        'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                        'data_augmentation' : True,
                        'batch_size' :4,
                        'particle_embedding': True,
                        'embedding_type': 'none',
                        'embedding': 1,
                        'model': 'InteractionParticles_B',
                        'upgrade_type':0}
    if id==74:
        model_config_test = {'ntry': id,
                        'input_size': 8,
                        'output_size': 2,
                        'hidden_size': 64,
                        'n_mp_layers': 8,
                        'noise_level': 0,
                        'noise_type': 0,
                        'radius': 0.075,
                        'dataset': '231001_72',
                        'nparticles': 4800,
                        'nparticle_types': 3,
                        'nframes': 200,
                        'sigma': .005,
                        'tau': 0.01,
                        'v_init': 0,
                        'aggr_type' : 'mean',
                        'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                        'data_augmentation' : True,
                        'batch_size' :4,
                        'particle_embedding': True,
                        'embedding_type': 'none',
                        'embedding': 1,
                        'model': 'InteractionParticles_B',
                        'upgrade_type':0}
    if id==75:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'noise_type': 0,
                             'radius': 0.075,
                             'dataset': '231001_75',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 4,
                             'particle_embedding': True,
                             'embedding_type': 'none',
                             'embedding': 1,
                             'model': 'InteractionParticles_A',
                             'upgrade_type': 0}
    if id==76:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'noise_type': 0,
                             'radius': 0.075,
                             'dataset': '231001_76',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.005,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 4,
                             'particle_embedding': True,
                             'embedding_type': 'none',
                             'embedding': 1,
                             'model': 'InteractionParticles_F',
                             'upgrade_type': 0}
    if id==77:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'noise_type': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.01,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 4,
                             'particle_embedding': True,
                             'embedding_type': 'none',
                             'embedding': 1,
                             'model': 'MixInteractionParticles',
                             'upgrade_type': 0}
    if id==78:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'noise_type': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.005,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 4,
                             'particle_embedding': True,
                             'embedding_type': 'none',
                             'embedding': 1,
                             'model': 'InteractionParticles_F',
                             'upgrade_type': 0}

# elctrostatic
    if id==80:
        model_config_test = {'ntry': id,
                    'input_size': 11,
                    'output_size': 2,
                    'hidden_size': 64,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.15,
                    'dataset': '230902_80',
                    'nparticles': 960,
                    'nparticle_types': 3,
                    'nframes': 1000,
                    'sigma': .005,
                    'tau': 5E-9,
                    'v_init': 1E-4,
                    'aggr_type' : 'add',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'batch_size': 4,
                    'embedding_type': 'none',
                    'embedding': 2,
                    'model': 'ElecParticles',
                    'upgrade_type':0}
    if id==81:
        model_config_test = {'ntry': id,
                    'input_size': 11,
                    'output_size': 2,
                    'hidden_size': 64,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'noise_type': 0,
                    'radius': 0.15,
                    'dataset': '230902_81',
                    'nparticles': 960,
                    'nparticle_types': 3,
                    'nframes': 1000,
                    'sigma': .005,
                    'tau': 5E-9,
                    'v_init': 1E-4,
                    'aggr_type' : 'add',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'data_augmentation' : True,
                    'batch_size': 4,
                    'embedding_type': 'none',
                    'embedding': 2,
                    'model': 'ElecParticles',
                    'upgrade_type':0}

    return model_config_test

if __name__ == '__main__':

    print('')
    print('version 1.4 231008')
    print('use of https://github.com/gpeyre/.../ml_10_particle_system.ipynb')
    print('')

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    scaler = StandardScaler()
    S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    model_config = load_model_config(id=72)

    sigma = model_config['sigma']
    aggr_type = model_config['aggr_type']
    print('')
    training_mode='t+1'   # 't+1' 'regressive' 'regressive_loop'
    print(f'training_mode: {training_mode}')

    for gtest in range(78,76,-1):

        model_config = load_model_config(id=gtest)

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

        # ntry = gtest
        # model_config['ntry'] = ntry
        # dataset_name = '231001_'+str(ntry)
        # model_config['dataset'] = dataset_name
        # model_config['model']= 'MixInteractionParticles'

        for key, value in model_config.items():
            print(key, ":", value)
        # data_generate(model_config)
        data_train(model_config,gtest)
        # data_plot(model_config)

        # x, rmserr_list = data_test(model_config, bVisu=True, bPrint=True)



        # prev_nparticles, new_nparticles, prev_index_particles, index_particles = data_test_generate(model_config)
        # x, rmserr_list = data_test(model_config, bVisu = True, bPrint=True, index_particles=index_particles, prev_nparticles=prev_nparticles, new_nparticles=new_nparticles, prev_index_particles=prev_index_particles)
        # data_train_generate(model_config, f'./graphs_data/graphs_particles_230902_72/')

