import glob
import json
import logging
import time
from decimal import Decimal
from math import *
from shutil import copyfile

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.data as data
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import umap
import yaml  # need to install pyyaml
from geomloss import SamplesLoss
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tifffile import imread
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils.convert import to_networkx
from tqdm import trange
from matplotlib import rc
import os

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from data_loaders import *
from src.utils import to_numpy

def func_pow(x, a, b):
    return a / (x**b)


def func_lin(x, a, b):
    return a * x + b


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
        else:
            # color_map = plt.cm.get_cmap(self.model_config['cmap'])
            color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
            index = color_map(index / self.nmap)

        return index


class Laplacian_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], c=[], beta=[], clamp=[]):
        super(Laplacian_A, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.clamp = clamp

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        c = self.c[to_numpy(x[:, 5])]
        c = c[:, None]

        laplacian = self.beta * c * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        return laplacian

    def message(self, x_i, x_j, edge_attr):
        L = edge_attr * x_j[:, 6]

        return L[:, None]

    def psi(self, I, p):

        return I


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


class PDE_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], delta_t=[], sigma=[]):
        super(PDE_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.sigma = sigma
        self.delta_t = delta_t

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        delta_x = self.delta_t * self.propagate(edge_index, x=(x, x))

        return delta_x

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)  # squared distance
        pp = self.p[to_numpy(x_i[:, 5]), :]
        psi = pp[:, 0] * torch.exp(-r ** pp[:, 1] / (2 * self.sigma ** 2)) - pp[:, 2] * torch.exp(
            -r ** pp[:, 3] / (2 * self.sigma ** 2))
        return psi[:, None] * bc_diff(x_j[:, 1:3] - x_i[:, 1:3])

    def psi(self, r, p):
        return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2)) - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * self.sigma ** 2)))


class PDE_embedding(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], delta_t=[], prediction=[], sigma=[]):
        super(PDE_embedding, self).__init__(aggr='mean')  # "mean" aggregation.

        self.p = p
        self.delta_t = delta_t
        self.prediction = prediction
        self.sigma = torch.tensor([sigma], device=device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.delta_t * self.propagate(edge_index, x=(x, x))

        return newv

    def message(self, x_i, x_j):
        r = torch.sum((x_i[:, :] - x_j[:, :]) ** 2, axis=1)  # squared distance
        pp = self.p[0].repeat(x_i.shape[0], 1)
        ssigma = self.sigma[0].repeat(x_i.shape[0], 1)
        psi = - pp[:, 2] * torch.exp(-r ** pp[:, 0] / (2 * ssigma[:, 0] ** 2)) + pp[:, 3] * torch.exp(
            -r ** pp[:, 1] / (2 * ssigma[:, 0] ** 2))
        return psi[:, None] * (x_i - x_j)

    def psi(self, r, p):
        return r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * self.sigma ** 2)) + p[3] * torch.exp(
            -r ** (2 * p[1]) / (2 * self.sigma ** 2)))


class PDE_B(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], delta_t=[]):
        super(PDE_B, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.delta_t = delta_t

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.delta_t * self.propagate(edge_index, x=(x, x))

        oldv = x[:, 3:5]
        newv = oldv + acc
        p = self.p[to_numpy(x[:, 5]), :]
        oldv_norm = torch.norm(oldv, dim=1)
        newv_norm = torch.norm(newv, dim=1)
        factor = (oldv_norm + p[:, 1] / 5E2 * (newv_norm - oldv_norm)) / newv_norm
        newv *= factor[:, None].repeat(1, 2)
        acc = newv - oldv

        return acc

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)  # distance squared

        pp = self.p[to_numpy(x_i[:, 5]), :]

        cohesion = pp[:, 0:1].repeat(1, 2) * 0.5E-5 * bc_diff(x_j[:, 1:3] - x_i[:, 1:3])

        alignment = pp[:, 1:2].repeat(1, 2) * 5E-4 * bc_diff(x_j[:, 3:5] - x_i[:, 3:5])

        separation = pp[:, 2:3].repeat(1, 2) * 1E-8 * bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / (r[:, None].repeat(1, 2))

        return (separation + alignment + cohesion)

    def psi(self, r, p):
        cohesion = p[0] * 0.5E-5 * r
        separation = -p[2] * 1E-8 / r
        return (cohesion + separation)  # 5E-4 alignement


class PDE_E(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], delta_t=[], clamp=[], pred_limit=[], prediction=[]):
        super(PDE_E, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.delta_t = delta_t
        self.clamp = clamp
        self.pred_limit = pred_limit
        self.prediction = prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.delta_t * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1))
        # r = torch.clamp(r, min=self.clamp)
        r = torch.concatenate((r[:, None], r[:, None]), -1)

        p1 = self.p[to_numpy(x_i[:, 5])]
        p1 = p1.squeeze()
        p1 = torch.concatenate((p1[:, None], p1[:, None]), -1)

        p2 = self.p[to_numpy(x_j[:, 5])]
        p2 = p2.squeeze()
        p2 = torch.concatenate((p2[:, None], p2[:, None]), -1)

        acc = p1 * p2 * bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / r ** 3
        # acc = torch.clamp(acc, max=self.pred_limit)

        return acc

    def psi(self, r, p1, p2):
        acc = p1 * p2 / r ** 2
        return -acc  # Elec particles


class PDE_G(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], delta_t=[], clamp=[], pred_limit=[]):
        super(PDE_G, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.delta_t = delta_t
        self.clamp = clamp
        self.pred_limit = pred_limit

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.delta_t * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1))
        r = torch.clamp(r, min=self.clamp)
        r = torch.concatenate((r[:, None], r[:, None]), -1)

        p = self.p[to_numpy(x_j[:, 5])]
        p = p.squeeze()
        p = torch.concatenate((p[:, None], p[:, None]), -1)

        acc = p * bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) / r ** 3

        return torch.clamp(acc, max=self.pred_limit)

    def psi(self, r, p):
        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

        return psi[:, None]


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
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.ndataset = model_config['nrun'] - 1
        self.upgrade_type = model_config['upgrade_type']
        self.prediction = model_config['prediction']
        self.upgrade_type = model_config['upgrade_type']
        self.nlayers_update = model_config['nlayers_update']
        self.hidden_size_update = model_config['hidden_size_update']
        self.sigma = model_config['sigma']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

        if self.upgrade_type != 'none':
            self.lin_update = MLP(input_size=self.output_size + self.embedding + 2, output_size=self.output_size,
                                  nlayers=self.nlayers_update, hidden_size=self.hidden_size_update, device=self.device)

    def forward(self, data, data_id, step, vnorm, cos_phi, sin_phi):

        self.data_id = data_id
        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pred = self.propagate(edge_index, x=(x, x))

        if self.upgrade_type == 'linear':
            embedding = self.a[self.data_id, to_numpy(x[:, 0]), :]
            pred = self.lin_update(torch.cat((pred, x[:, 3:5], embedding), dim=-1))

        if step == 2:  # test, if degree = 0 acc =0
            deg = pyg_utils.degree(edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)
            return deg * pred
        else:
            return pred

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) / self.radius
        x_i_vx = x_i[:, 3:4] / self.vnorm[4]
        x_i_vy = x_i[:, 4:5] / self.vnorm[4]
        x_j_vx = x_j[:, 3:4] / self.vnorm[4]
        x_j_vy = x_j[:, 4:5] / self.vnorm[4]

        if (self.data_augmentation) & (self.step == 1):
            new_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_x
            delta_pos[:, 1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        embedding = self.a[self.data_id, to_numpy(x_i[:, 0]), :]

        if self.prediction == '2nd_derivative':
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)
        else:
            if self.prediction == 'first_derivative_L':
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)
            if self.prediction == 'first_derivative':
                in_features = torch.cat((delta_pos, r, embedding), dim=-1)

        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        if (len(p) == 3):  # PDE_B
            cohesion = p[0] * 0.5E-5 * r
            separation = -p[2] * 1E-8 / r
            return (cohesion + separation) * p[1] / 500  #
        else: # PDE_A
            return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2)) - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * self.sigma ** 2)))


class InteractionCElegans(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(InteractionCElegans, self).__init__(aggr='mean')  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.ndataset = model_config['nrun'] - 1
        self.upgrade_type = model_config['upgrade_type']
        self.prediction = model_config['prediction']
        self.upgrade_type = model_config['upgrade_type']
        self.nlayers_update = model_config['nlayers_update']
        self.hidden_size_update = model_config['hidden_size_update']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles + 1), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float64))

        if self.upgrade_type == 'linear':
            self.lin_update = MLP(input_size=self.output_size + self.embedding + 2, output_size=self.output_size,
                                  nlayers=self.nlayers_update, hidden_size=self.hidden_size_update, device=self.device)

        self.to(device=self.device)
        self.to(torch.float64)

    def forward(self, data, data_id, time):

        self.data_id = data_id

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pred = self.propagate(edge_index, x=(x, x), time=time)

        if self.upgrade_type == 'linear':
            embedding = self.a[self.data_id, to_numpy(x[:, 0]), :]
            pred = self.lin_update(torch.cat((pred, x[:, 3:5], embedding), dim=-1))

        return pred

    def message(self, x_i, x_j, time):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:4] - x_j[:, 1:4]) ** 2, axis=1))  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 1:4] - x_j[:, 1:4])
        embedding = self.a[self.data_id, to_numpy(x_i[:, 0]).astype(int), :]
        in_features = torch.cat((delta_pos, r, x_i[:, 4:7], x_j[:, 4:7], embedding, time[:, None]), dim=-1)

        out = self.lin_edge(in_features)

        return out

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
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.upgrade_type = model_config['upgrade_type']
        self.ndataset = model_config['nrun'] - 1
        self.clamp = model_config['clamp']
        self.pred_limit = model_config['pred_limit']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id, step, vnorm, cos_phi, sin_phi):

        self.data_id = data_id
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

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) / self.radius
        x_i_vx = x_i[:, 3:4] / self.vnorm[4]
        x_i_vy = x_i[:, 4:5] / self.vnorm[5]
        x_j_vx = x_j[:, 3:4] / self.vnorm[4]
        x_j_vy = x_j[:, 4:5] / self.vnorm[5]

        if (self.data_augmentation) & (self.step == 1):
            new_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_x
            delta_pos[:, 1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        embedding = self.a[self.data_id, to_numpy(x_j[:, 0]), :]  # depends on other
        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        psi = p / r ** 2
        return psi[:, None]


class ElecParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(ElecParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.ndataset = model_config['nrun'] - 1
        self.upgrade_type = model_config['upgrade_type']
        self.clamp = model_config['clamp']
        self.pred_limit = model_config['pred_limit']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id, step, vnorm, cos_phi, sin_phi):

        self.data_id = data_id
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

        r = torch.sqrt(torch.sum(bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / self.radius
        x_i_vx = x_i[:, 3:4] / self.vnorm[4]
        x_i_vy = x_i[:, 4:5] / self.vnorm[5]
        x_j_vx = x_j[:, 3:4] / self.vnorm[4]
        x_j_vy = x_j[:, 4:5] / self.vnorm[5]

        if (self.data_augmentation) & (self.step == 1):
            new_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_x
            delta_pos[:, 1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        embedding0 = self.a[self.data_id, to_numpy(x_i[:, 0]), :]
        embedding1 = self.a[self.data_id, to_numpy(x_j[:, 0]), :]  # depends on other
        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding0, embedding1), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p1, p2):
        acc = p1 * p2 / r ** 2
        return -acc  # Elec particles


class MeshLaplacian(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):
        super(MeshLaplacian, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.ndataset = model_config['nrun']-1

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.ndataset), int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        height = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        return height

    def message(self, x_i, x_j, edge_attr):
        embedding = self.a[self.data_id, to_numpy(x_i[:, 0]), :]

        in_features = torch.cat((edge_attr[:, None], x_j[:, 6:7] - x_i[:, 6:7], embedding), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return r


def data_plot_FIG2():

    config = 'config_arbitrary_3'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    for key, value in model_config.items():
        print(key, ":", value)
        if ('E-' in str(value)) | ('E+' in str(value)):
            value = float(value)
            model_config[key] = value

    cmap = cc(model_config=model_config)
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
    ratio = 1


    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    nrun = model_config['nrun']
    kmeans_input = model_config['kmeans_input']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

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
        model = MeshLaplacian(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = InteractionParticles(model_config, device)
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


    net = f"./log/try_{dataset_name}/models/best_model_with_{nrun - 1}_graphs_0.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

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

    cm = 1 / 2.54 * 3 / 2.3

    # plt.subplots(frameon=False)
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    # fig = plt.figure(figsize=(3*cm, 3*cm))

    fig = plt.figure(figsize=(13, 9.6))
    ax = fig.add_subplot(3, 4, 1)
    print('1')
    if (embedding.shape[1] > 1):
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                            embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=0.1)
        plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$',fontsize=14)
        plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 4, 2)
    print('2')
    acc_list = []
    for n in range(nparticles):
        embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
        in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                 rr[:, None] / model_config['radius'], embedding), dim=1)
        with torch.no_grad():
            acc = model.lin_edge(in_features.float())
        acc = acc[:, 0]
        acc_list.append(acc)
        if (n % (nparticles // 50) == 0):
            plt.plot(to_numpy(rr),
                     to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
                     color=cmap.color(to_numpy(x[n, 5])), linewidth=1)
    acc_list = torch.stack(acc_list)
    coeff_norm = to_numpy(acc_list)
    trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                      random_state=42, transform_queue_size=0).fit(coeff_norm)
    proj_interaction = trans.transform(coeff_norm)
    proj_interaction = np.squeeze(proj_interaction)
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim([-0.04,0.03])

    ax = fig.add_subplot(3, 4, 3)
    print('3')
    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,
                    random_state=13)
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
    new_labels = kmeans.labels_.copy()
    for n in range(nparticle_types):
        new_labels[kmeans.labels_ == label_list[n]] = n
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=0.1)
        plt.xlabel(r'UMAP 0', fontsize=14)
        plt.ylabel(r'UMAP 1', fontsize=14)
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

    ax = fig.add_subplot(3, 4, 4)
    print('4')
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]
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
    plt.text(0, -0.75, r"F1-score: {:.3f}".format(F1), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ####

    net = f"./log/try_{dataset_name}/models/best_model_with_{nrun - 1}_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

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

    ax = fig.add_subplot(3, 4, 5)
    print('5')
    if (embedding.shape[1] > 1):
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                            embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=0.1)
        plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$',fontsize=14)
        plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 4, 6)
    print('6')
    acc_list = []
    for n in range(nparticles):
        embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
        in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                 rr[:, None] / model_config['radius'], embedding), dim=1)
        with torch.no_grad():
            acc = model.lin_edge(in_features.float())
        acc = acc[:, 0]
        acc_list.append(acc)
        if (n % (nparticles // 50) == 0):
            plt.plot(to_numpy(rr),
                     to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
                     color=cmap.color(to_numpy(x[n, 5])), linewidth=1)
    acc_list = torch.stack(acc_list)
    coeff_norm = to_numpy(acc_list)
    trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                      random_state=42, transform_queue_size=0).fit(coeff_norm)
    proj_interaction = trans.transform(coeff_norm)
    proj_interaction = np.squeeze(proj_interaction)
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim([-0.04,0.03])

    ax = fig.add_subplot(3, 4, 7)
    print('7')
    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,
                    random_state=13)
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
    new_labels = kmeans.labels_.copy()
    for n in range(nparticle_types):
        new_labels[kmeans.labels_ == label_list[n]] = n
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=0.1)
        plt.xlabel(r'UMAP 0', fontsize=14)
        plt.ylabel(r'UMAP 1', fontsize=14)
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

    ax = fig.add_subplot(3, 4, 8)
    print('8')
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]
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
    plt.text(0, -0.75, r"F1-score: {:.3f}".format(F1), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 4, 9)
    print('9')
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=1)
    plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$', fontsize=14)
    plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 4, 10)
    print('10')
    acc_list = []
    for n in range(nparticles):
        embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
        in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                 rr[:, None] / model_config['radius'], embedding), dim=1)
        with torch.no_grad():
            acc = model.lin_edge(in_features.float())
        acc = acc[:, 0]
        acc_list.append(acc)
        if (n % (nparticles // 50) == 0):
            plt.plot(to_numpy(rr),
                     to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
                     color=cmap.color(to_numpy(x[n, 5])), linewidth=1)
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0,0.02,r'Model', fontsize=14)
    plt.ylim([-0.04,0.03])

    ax = fig.add_subplot(3,4,11)
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
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim([-0.04,0.03])
    plt.text(0,0.02,r'True', fontsize=14)

    plt.tight_layout()
    plt.savefig('Fig2.pdf', format="pdf", dpi=300)
    plt.close()

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
        plot_list.append(pred * ynorm[4] / torch.tensor(model_config['delta_t'], device=device))

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

    print(f'RMSE: {np.round(np.mean(rmserr_list),3)}+\-{np.round(np.std(rmserr_list),3)} ')

def data_plot_FIG3():


    config = 'config_gravity_16'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    for key, value in model_config.items():
        print(key, ":", value)
        if ('E-' in str(value)) | ('E+' in str(value)):
            value = float(value)
            model_config[key] = value

    cmap = cc(model_config=model_config)
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
    ratio = 1

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    nrun = model_config['nrun']
    kmeans_input = model_config['kmeans_input']


    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

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
        model = MeshLaplacian(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = InteractionParticles(model_config, device)
        print(f'Training InteractionParticles')

    # if best_model == -1:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    # else:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

    net = f"./log/try_{dataset_name}/models/best_model_with_{nrun - 1}_graphs_20.pt"
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

    cm = 1 / 2.54 * 3 / 2.3

    # plt.subplots(frameon=False)
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    # fig = plt.figure(figsize=(3*cm, 3*cm))

    fig = plt.figure(figsize=(10, 9))
    # plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    print('1')
    if (embedding.shape[1] > 1):
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                            embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=0.1)
        plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$',fontsize=14)
        plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # ax = fig.add_subplot(3, 4, 2)
    # print('2')
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
        # plt.plot(to_numpy(rr),
        #          to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
        #          color=cmap.color(to_numpy(x[n, 5])), linewidth=1, alpha=0.25)
    acc_list = torch.stack(acc_list)
    # plt.xlim([0, 0.02])
    # plt.ylim([0, 0.5E6])
    coeff_norm = to_numpy(acc_list)
    trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                      random_state=42, transform_queue_size=0).fit(coeff_norm)
    proj_interaction = trans.transform(coeff_norm)
    proj_interaction = np.squeeze(proj_interaction)

    # plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_j, r_{ij}) [a.u.]$', fontsize=14)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.ylim([0, 0.5E6])


    ax = fig.add_subplot(3, 3, 2)
    print('2')
    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,
                    random_state=13)
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
    new_labels = kmeans.labels_.copy()
    for n in range(nparticle_types):
        new_labels[kmeans.labels_ == label_list[n]] = n
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=0.1)
        plt.xlabel(r'UMAP 0', fontsize=14)
        plt.ylabel(r'UMAP 1', fontsize=14)

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

    ax = fig.add_subplot(3, 3, 3)
    print('3')
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]
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
    plt.text(0, -0.75, r"F1-score: {:.3f}".format(F1), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 4)
    print('4')
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=1)
    plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$', fontsize=14)
    plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    print('5')
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
                 to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
                 color=cmap.color(to_numpy(x[n, 5])), linewidth=1, alpha=0.25)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_j, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0.0075,0.4E6,r'Model', fontsize=14)

    ax = fig.add_subplot(3,3,6)
    print('6')
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
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_j, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0.0075,0.4E6,r'True', fontsize=14)

    plot_list = []
    for n in range(nparticle_types):
        embedding = t[int(label_list[n])] * torch.ones((1000, model_config['embedding']), device=device)
        in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                 rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                 0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        with torch.no_grad():
            pred = model.lin_edge(in_features.float())
        pred = pred[:, 0]
        plot_list.append(pred * ynorm[4] / torch.tensor(model_config['delta_t'], device=device))
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

    print(f'RMSE: {np.round(np.mean(rmserr_list),4)}+\-{np.round(np.std(rmserr_list),4)} ')

    #############

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
        plot_list.append(pred * ynorm[4] / torch.tensor(model_config['delta_t'],device=device))

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
        plot_list_2.append(pred * ynorm[4] / torch.tensor(model_config['delta_t'], device=device))

    ax = fig.add_subplot(3, 3, 9)
    print('9')
    for n in range(len(r_list)):
        plt.plot(to_numpy(vv), to_numpy(plot_list_2[n]), linewidth=1, color=cmap.color(n),
                 label=f'$r$={r_list[n]}')
    plt.xlabel(r'Normalized velocity $[a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_j, r_{ij}, \ensuremath{\mathbf{\dot{x}}}_i,) [a.u.]$', fontsize=14)
    plt.xlim([0, 2])
    plt.legend()

    ax = fig.add_subplot(3, 3, 7)
    print('7')
    plt.scatter(p, popt_list[:, 0], color='k')
    x_data = p
    y_data = popt_list[:, 0]
    lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
    plt.plot(p, func_lin(x_data, lin_fit[0], lin_fit[1]), color='r')
    plt.xlabel(r'True mass $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted mass $[a.u.]$', fontsize=14)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.text(0.5, 5, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - func_lin(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(0.5, 4.5, f"$R^2$: {np.round(r_squared, 2)}", fontsize=10)
    ax = fig.add_subplot(3, 3, 8)
    print('8')
    plt.scatter(p, popt_list[:, 1], color='k')
    plt.xlim([0, 5.5])
    plt.ylim([0, 4])
    plt.xlabel(r'True mass $[a.u.]$', fontsize=14)
    plt.ylabel(r'Exponential fit $[a.u.]$', fontsize=14)
    plt.text(0.5, 3.5, f"{np.round(np.mean(popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)
    plt.tight_layout()

    plt.savefig('Fig3.pdf', format="pdf", dpi=300)
    plt.close()

def data_plot_FIG4():


    config = 'config_Coulomb_3'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    for key, value in model_config.items():
        print(key, ":", value)
        if ('E-' in str(value)) | ('E+' in str(value)):
            value = float(value)
            model_config[key] = value

    cmap = cc(model_config=model_config)
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
    ratio = 1

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    nrun = model_config['nrun']
    kmeans_input = model_config['kmeans_input']


    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

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
        model = MeshLaplacian(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = InteractionParticles(model_config, device)
        print(f'Training InteractionParticles')

    # if best_model == -1:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    # else:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

    net = f"./log/try_{dataset_name}/models/best_model_with_{nrun - 1}_graphs_20.pt"
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

    cm = 1 / 2.54 * 3 / 2.3

    # plt.subplots(frameon=False)
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    # fig = plt.figure(figsize=(3*cm, 3*cm))

    fig = plt.figure(figsize=(10, 9))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    print('1')
    if (embedding.shape[1] > 1):
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                            embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=0.1)
        plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$',fontsize=14)
        plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # ax = fig.add_subplot(3, 4, 2)
    # print('2')
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
                # if n % 5 == 0:
                #     plt.plot(to_numpy(rr),
                #              to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
                #              linewidth=1,
                #              color=cmap.color(k), alpha=0.25)
    acc_list = torch.stack(acc_list)
    # plt.xlim([0, 0.02])
    # plt.ylim([-0.5E6, 0.5E6])
    # plt.xlabel('Distance [a.u]', fontsize=12)
    # plt.ylabel('MLP [a.u]', fontsize=12)
    coeff_norm = to_numpy(acc_list)
    trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                      random_state=42, transform_queue_size=0).fit(coeff_norm)
    proj_interaction = trans.transform(coeff_norm)
    proj_interaction = np.squeeze(proj_interaction)

    ax = fig.add_subplot(3, 3, 2)
    print('2')
    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,
                    random_state=13)
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
    new_labels = kmeans.labels_.copy()
    for n in range(nparticle_types):
        new_labels[kmeans.labels_ == label_list[n]] = n
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=0.1)
        plt.xlabel(r'UMAP 0', fontsize=14)
        plt.ylabel(r'UMAP 1', fontsize=14)

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

    ax = fig.add_subplot(3, 3, 3)
    print('3')
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]
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
    plt.text(0, -0.75, r"F1-score: {:.3f}".format(F1), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 4)
    print('4')
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=1)
    plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$', fontsize=14)
    plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    print('5')
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
                     to_numpy(acc) * to_numpy(ynorm[4]) / model_config['delta_t'],
                     linewidth=1)
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0.0075,0.35E6,r'Model', fontsize=14)
    plt.xlim([0, 0.02])
    plt.ylim([-0.5E6, 0.5E6])

    ax = fig.add_subplot(3,3,6)
    print('6')
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
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0.0075,0.35E6,r'True', fontsize=14)


    #############

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
            plot_list_pairwise.append(pred * ynorm[4] / torch.tensor(model_config['delta_t'], device=device))

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

    ax = fig.add_subplot(3, 3, 8)

    plt.scatter(ptrue_list, popt_list[:, 0], color='k')
    x_data = ptrue_list
    y_data = popt_list[:, 0]
    lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
    plt.plot(ptrue_list, func_lin(x_data, lin_fit[0], lin_fit[1]), color='r')
    plt.xlabel(r'True $q_i q_j [a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted $q_i q_j [a.u.]$', fontsize=14)
    plt.text(-2, 4, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - func_lin(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(-2, 3, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)

    ax = fig.add_subplot(3,3,9)
    plt.scatter(ptrue_list, -popt_list[:, 1], color='k')
    plt.ylim([0, 4])
    plt.xlabel(r'True $q_i q_j [a.u.]$', fontsize=14)
    plt.ylabel(r'Exponential fit $[a.u.]$', fontsize=14)
    plt.text(-2, 3.5, f"{np.round(-np.mean(popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)
    plt.tight_layout()

    plt.savefig('Fig4.pdf', format="pdf", dpi=300)
    plt.close()





if __name__ == '__main__':

    print('')
    print('version 1.9 240103')
    print('use of https://github.com/gpeyre/.../ml_10_particle_system.ipynb')
    print('')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    scaler = StandardScaler()
    S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    # def bc_pos(X):
    #     return torch.remainder(X, 1.0)
    # def bc_diff(D):
    #     return torch.remainder(D - .5, 1.0) - .5
    # aggr_type = 'mean'
    # data_plot_FIG2()

    # def bc_pos(X):
    #     return X
    # def bc_diff(D):
    #     return D
    # aggr_type= 'add'
    # data_plot_FIG3()

    def bc_pos(X):
        return torch.remainder(X, 1.0)
    def bc_diff(D):
        return torch.remainder(D - .5, 1.0) - .5
    aggr_type = 'add'
    data_plot_FIG4()





