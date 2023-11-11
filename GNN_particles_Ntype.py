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
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import time
from shutil import copyfile
from prettytable import PrettyTable
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from geomloss import SamplesLoss
import torch_geometric.transforms as T
import pandas
import trackpy
from numpy import vstack
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from torch_geometric.utils import degree
import umap
from tifffile import imwrite, imread

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
def histogram(xs, bins, min, max):
    # Like torch.histogram, but works with cuda
    # min, max = xs.min(), xs.max()
    counts = torch.histc(xs, bins, min=min, max=max)
    boundaries = torch.linspace(min, max, bins + 1)
    return counts

class cc:

    def __init__(self, model_config):
        self.model_config = model_config
        self.model = model_config['model']
        if model_config['cmap'] == 'tab10':
            self.nmap = 8
        else:
            self.nmap = model_config['nparticle_types']

    def color(self,index):
        if self.model=='ElecParticles':
            if index == 0:
                index = (0, 0, 1)
            elif index== 1:
                index = (0, 0.5, 0.75)
            elif index == 2:
                index = (1, 0, 0)
            return (index)
        else:
            color_map = plt.cm.get_cmap(self.model_config['cmap'])
            #color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
            index = color_map(index/self.nmap)

        return index

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden, out, device):
        super(FeedForwardNN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(input_dim, hidden, device=device)
        self.layer2 = nn.Linear(hidden, hidden, device=device)
        self.layer3 = nn.Linear(hidden, out, device=device)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x
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
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)

            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

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
                out += [func(freq * x)]

        return torch.cat(out, -1)
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

        # ee = edge_index.detach().cpu().numpy()
        # pos = np.argwhere(ee[0, :] == 2243)
        # pos = pos.squeeze().astype(int)
        #
        # with torch.no_grad():
        #     h=x.detach()
        #     sum_weight = edge_attr[pos] * h[ee[1,pos],5]

        heat_flow = self.beta * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        heat_flow = torch.clamp(heat_flow, min=-self.clamp, max=self.clamp)
        return heat_flow

    def message(self, x_i, x_j, edge_attr):
        c = self.c[x_i[:, 5].detach().cpu().numpy()]
        c = c.squeeze()
        heat = c * edge_attr * x_j[:, 6]

        return heat[:, None]

    def psi(self, r, p):
        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

        return psi[:, None]
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

class Particles_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[], prediction=[]):
        super(Particles_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau
        self.prediction = prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.tau * self.propagate(edge_index, x=(x, x))


        if self.prediction == '2nd_derivative':
            oldv = x[:, 3:5]
            acc = newv - oldv
            return acc
        else:
            return newv

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)  # squared distance
        pp = self.p[x_i[:, 5].detach().cpu().numpy(), :]
        psi = - pp[:, 2] * torch.exp(-r ** pp[:, 0] / (2 * sigma ** 2)) + pp[:, 3] * torch.exp(-r ** pp[:, 1] / (2 * sigma ** 2))
        return psi[:, None] * bc_diff(x_i[:, 1:3] - x_j[:, 1:3])

    def psi(self, r, p):
        return r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(
            -r ** (2 * p[1]) / (2 * sigma ** 2)))
class Particles_E(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[], clamp=[], pred_limit=[], prediction=[]):
        super(Particles_E, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.tau = tau
        self.clamp = clamp
        self.pred_limit = pred_limit
        self.prediction = prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1))
        r = torch.clamp(r, min=self.clamp)
        r = torch.concatenate((r[:, None], r[:, None]), -1)

        p1 = self.p[x_i[:, 5].detach().cpu().numpy()]
        p1 = p1.squeeze()
        p1 = torch.concatenate((p1[:, None], p1[:, None]), -1)

        p2 = self.p[x_j[:, 5].detach().cpu().numpy()]
        p2 = p2.squeeze()
        p2 = torch.concatenate((p2[:, None], p2[:, None]), -1)

        acc = p1 * p2 * bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / r ** 3
        acc = torch.clamp(acc, max=self.pred_limit)

        return acc

    def psi(self, r, p1, p2):
        r_ = torch.clamp(r, min=self.clamp)
        acc = p1 * p2 * r / r_ ** 2
        acc = torch.clamp(acc, max=self.pred_limit)
        return acc  # Elec particles
class Particles_G(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[], clamp=[], pred_limit=[]):
        super(Particles_G, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.tau = tau
        self.clamp = clamp
        self.pred_limit = pred_limit

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1))
        r = torch.clamp(r, min=self.clamp)
        r = torch.concatenate((r[:, None], r[:, None]), -1)

        p = self.p[x_j[:, 5].detach().cpu().numpy()]
        p = p.squeeze()
        p = torch.concatenate((p[:, None], p[:, None]), -1)

        acc = p * bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) / r ** 3

        return torch.clamp(acc, max=self.pred_limit)

    def psi(self, r, p):
        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

        return psi[:, None]
class Particles_H(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[], beta=[], clamp=[], pred_limit=[], prediction=[]):
        super(Particles_H, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.tau = tau
        self.beta = beta
        self.clamp = clamp
        self.pred_limit = pred_limit
        self.prediction = prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=(x, x))

        acc = self.tau * out[:, 0:2]
        heat = -self.beta * out[:, 2]

        return torch.cat((acc, heat[:, None]), axis=1)

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1))
        r = torch.clamp(r, min=self.clamp)
        r = torch.concatenate((r[:, None], r[:, None]), -1)

        p = self.p[x_j[:, 5].detach().cpu().numpy()]
        p = p.squeeze()
        p = torch.concatenate((p[:, None], p[:, None]), -1)

        acc = p * bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) / r ** 3

        p = self.p[x_i[:, 5].detach().cpu().numpy()]
        p = p.squeeze()

        heat = p * x_i[:, 0] - x_j[:, 0]
        heat = heat[:, None] / r[:, 0:1]
        heat = torch.nan_to_num(heat, nan=1, posinf=1, neginf=1)

        acc = torch.clamp(acc, max=self.pred_limit)

        return torch.cat((acc, heat), axis=1)

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

        pred = self.propagate(edge_index, x=(x, x))
        ## TO BE CHANGED ##
        deg = pyg_utils.degree(edge_index[0], data.num_nodes)
        deg = (deg > 0)
        deg = (deg > 0).type(torch.float32)
        if step == 2:
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)  # test, if degree = 0 acc =0
            return deg * pred
        else:
            return pred

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
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

        embedding = self.a[self.data_id, x_i[:, 0].detach().cpu().numpy(), :]

        if self.prediction == '2nd_derivative':
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)
        else:
            in_features = torch.cat((delta_pos, r, embedding), dim=-1)

        try:
            out = self.lin_edge(in_features)
        except:
            a=1

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        return -(r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(-r ** (2 * p[1]) / (2 * sigma ** 2))))
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

        embedding = self.a[self.data_id, x_j[:, 0].detach().cpu().numpy(), :]  # depends on other
        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

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

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
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

        embedding0 = self.a[self.data_id, x_i[:, 0].detach().cpu().numpy(), :]
        embedding1 = self.a[self.data_id, x_j[:, 0].detach().cpu().numpy(), :]  # depends on other
        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding0, embedding1), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p1, p2):
        r_ = torch.clamp(r, min=self.clamp)
        acc = p1 * p2 * r / r_ ** 3
        acc = torch.clamp(acc, max=self.pred_limit)
        return -acc  # Elec particles
class MeshDiffusion(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(MeshDiffusion, self).__init__(aggr=aggr_type)  # "Add" aggregation.

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

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):

        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        heat = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        return heat

    def message(self, x_i, x_j, edge_attr):

        embedding = self.a[self.data_id, x_i[:, 0].detach().cpu().numpy(), :]

        in_features = torch.cat((edge_attr[:,None], x_j[:, 6:7]-x_i[:, 6:7], embedding), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        return r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(
            -r ** (2 * p[1]) / (2 * sigma ** 2)))

def data_generate(model_config,bVisu=True, bDetails=False, bSave=True, step=5):
    print('')
    print('Generating data ...')

    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_data/*")
    # for f in files:
    #     os.remove(f)


    dataset_name = model_config['dataset']
    folder = f'./graphs_data/graphs_particles_{dataset_name}/'
    os.makedirs(folder, exist_ok=True)

    if bSave:
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
    noise_level = model_config['noise_level']
    v_init = model_config['v_init']
    rr = torch.tensor(np.linspace(0, radius * 2, 1000))
    rr = rr.to(device)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if model_config['model'] == 'Particles_A':
        print(f'Generate Particles_A')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        if nparticle_types == 1:
            model = Particles_A(aggr_type=aggr_type, p=p, tau=model_config['tau'],
                                prediction=model_config['prediction'])
        else:
            model = Particles_A(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                                prediction=model_config['prediction'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if model_config['model'] == 'GravityParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if model_config['model'] == 'HeatParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_H(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            beta=model_config['beta'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])
        model_mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],clamp=model_config['clamp'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if model_config['model'] == 'ElecParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
                print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
                torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = Particles_E(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'],
                            prediction=model_config['prediction'])
        psi_output = []
        for n in range(nparticle_types):
            for m in range(nparticle_types):
                psi_output.append(model.psi(rr, torch.squeeze(p[n]), torch.squeeze(p[m])))

    torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    for run in range(model_config['nrun']):

        x_list=[]
        y_list=[]
        h_list=[]

        if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
            X1 = torch.rand(nparticles, 2, device=device)
        else:
            X1 = torch.randn(nparticles, 2, device=device) * 0.5
        V1 = v_init * torch.randn((nparticles, 2), device=device)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = T1[:, None]
        h = torch.zeros((nparticles, 1), device=device)
        H1 = torch.ones((nparticles, 1), device=device) + torch.randn((nparticles, 1), device=device) / 2
        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:, None]

        time.sleep(0.5)

        noise_current = 0 * torch.randn((nparticles, 2), device=device)
        noise_prev_prev = 0 * torch.randn((nparticles, 2), device=device)

        for it in tqdm(range(-int(nframes * model_config['start_frame']), nframes)):

            noise_prev_prev = noise_prev_prev
            noise_prev = noise_current
            noise_current = 0 * torch.randn((nparticles, 2), device=device) * noise_level * radius

            ### TO BE CHANGED ###
            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), h.clone().detach()), 1)
            if (it >= 0) & (noise_level == 0):
                x_list.append(x)
                # torch.save(x, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')
            if (it >= 0) & (noise_level > 0):
                x_noise = x
                x_noise[:, 1:3] = x[:, 1:3] + noise_current
                x_noise[:, 3:5] = x[:, 3:5] + noise_current - noise_prev
                x_list.append(x_noise)
                # torch.save(x_noise, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')

            if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                dataset = data.Data(x=x, pos=x[:, 1:3])
                transform_0 = T.Compose([T.Delaunay()])
                dataset_face = transform_0(dataset).face
                mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
                edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)
                dataset_mesh = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)

            distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset)
            if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                y = y*0
            if (it >= 0) & (noise_level == 0):
                y_list.append(y)
                # torch.save(y, f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{it}.pt')
            if (it >= 0) & (noise_level > 0):
                y_noise = y[:, 0:2] + noise_current - 2 * noise_prev + noise_prev_prev
                y_list.append(y_noise)
                # torch.save(y_noise, f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{it}.pt')

            if model_config['prediction'] == '2nd_derivative':
                V1 += y[:, 0:2]
            else:
                V1 = y[:, 0:2]

            X1 = bc_pos(X1 + V1)

            if model_config['model'] == 'DiffMesh':
                if it >= 0:
                    with torch.no_grad():
                        h = model_mesh(dataset_mesh)
                    H1 += h
                    h_list.append(h)
            if model_config['model'] == 'WaveMesh':
                if it >= 0:
                    with torch.no_grad():
                        pred = model_mesh(dataset_mesh)
                        h += pred
                    H1 += h
                    h_list.append(pred)

            if (run == 0) & (it % step == 0) & (it >= 0) & bVisu:

                fig = plt.figure(figsize=(11.8, 12))
                # plt.ion()
                ax = fig.add_subplot(2, 2, 1)
                if model_config['model'] == 'GravityParticles':
                    for n in range(nparticle_types):
                        g = p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 7.5
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                    x[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                    alpha=0.75,color=cc(n))  # , facecolors='none', edgecolors='k')
                elif (model_config['model'] == 'HeatParticles') | (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                    plt.scatter(x[:, 1].detach().cpu().numpy(),x[:, 2].detach().cpu().numpy(), s=10, alpha=0.75,
                                    c=x[:, 6].detach().cpu().numpy(), cmap='inferno', vmin=0, vmax=2)
                elif model_config['model'] == 'ElecParticles':
                    for n in range(nparticle_types):
                        g = np.abs(p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)
                        if model_config['p'][n][0]<=0:
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='r', alpha=0.5)
                        else:
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='b', alpha=0.5)
                else:
                    for n in range(nparticle_types):
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(), x[index_particles[n], 2].detach().cpu().numpy(), s=3, color=cmap.color(n))
                if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                    plt.text(0, 1.08, f'frame: {it}')
                    plt.text(0, 1.03, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                else:
                    plt.text(-1.25, 1.5, f'frame: {it}')
                    plt.text(-1.25, 1.4, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])

                ax = fig.add_subplot(2, 2, 2)
                plt.scatter(x[:, 1].detach().cpu().numpy(), x[:, 2].detach().cpu().numpy(), s=1, color='k',alpha=0.75)
                if bDetails: # model_config['radius']<0.01:
                    pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                        vis = to_networkx(dataset_mesh, remove_self_loops=True, to_undirected=True)
                    else:
                        distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                        adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                        edge_index2 = adj_t2.nonzero().t().contiguous()
                        dataset2 = data.Data(x=x, edge_index=edge_index2)
                        vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False,alpha=0.3)
                if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                else:
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])

                if bDetails:

                    ax = fig.add_subplot(2, 2, 3)
                    if model_config['model'] == 'GravityParticles':
                        for n in range(nparticle_types):
                            g = p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 7.5 * 4
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                        alpha=0.75,
                                        color=cmap.color(n))  # , facecolors='none', edgecolors='k')
                    elif (model_config['model'] == 'HeatParticles') | (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                        plt.scatter(x[:, 1].detach().cpu().numpy(), x[:, 2].detach().cpu().numpy(), s=60, alpha=0.75,
                                    c=x[:, 6].detach().cpu().numpy(), cmap='inferno', vmin=0, vmax=2)
                    elif model_config['model'] == 'ElecParticles':
                        for n in range(nparticle_types):
                            g = np.abs(p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20) * 4
                            if model_config['p'][n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='r', alpha=0.5)
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='b', alpha=0.5)
                    elif model_config['model'] == 'Particles_A':
                        for n in range(nparticle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=50, alpha=0.75,
                                        color=cmap.color(n))

                    if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                        plt.xlim([0.3, 0.7])
                        plt.ylim([0.3, 0.7])
                    else:
                        plt.xlim([-0.25, 0.25])
                        plt.ylim([-0.25, 0.25])

                    if (model_config['model'] != 'HeatParticles') & (model_config['model'] != 'DiffMesh') & (model_config['model'] != 'WaveMesh'):
                        for k in range(nparticles):
                            plt.arrow(x=x[k, 1].detach().cpu().item(),y=x[k, 2].detach().cpu().item(),
                                      dx=x[k, 3].detach().cpu().item()*model_config['arrow_length'], dy=x[k, 4].detach().cpu().item()*model_config['arrow_length'],color='k')

                    ax = fig.add_subplot(2, 2, 4)
                    if (model_config['model'] == 'HeatParticles') | (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                        N1 = torch.arange(nparticles, device=device)
                        for n in range(nparticle_types):
                            plt.scatter(N1[index_particles[n]].detach().cpu().numpy(),
                                        H1[index_particles[n]].detach().cpu().numpy(), color=cmap.color(n), s=5,
                                        alpha=0.5)
                        plt.ylim([-0.5, 2.5])
                        plt.ylabel('Temperature [a.u]', fontsize="14")
                    else:
                        if len(x_list)>30:
                            x_all =torch.stack(x_list)
                            for k in range(nparticles):
                                xc = x_all[-30:-1, k, 1].detach().cpu().numpy().squeeze()
                                yc = x_all[-30:-1, k, 2].detach().cpu().numpy().squeeze()
                                plt.scatter(xc,yc,s=0.05, color='k',alpha=0.75)
                        elif len(x_list)>6:
                            x_all =torch.stack(x_list)
                            for k in range(nparticles):
                                xc = x_all[:, k, 1].detach().cpu().numpy().squeeze()
                                yc = x_all[:, k, 2].detach().cpu().numpy().squeeze()
                                plt.scatter(xc,yc,s=0.05, color='k',alpha=0.75)
                        if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                            plt.xlim([0,1])
                            plt.ylim([0,1])
                        else:
                            plt.xlim([-1.3, 1.3])
                            plt.ylim([-1.3, 1.3])

                plt.tight_layout()
                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
            torch.save(y_list, f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
            torch.save(h_list, f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt')
def data_train(model_config, bTest=False):
    print('')

    # for loop in range(25):
    #     print(f'Loop: {loop}')

    model = []
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
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

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    arr = np.arange(0, NGraphs)
    x_list=[]
    y_list=[]
    for run in arr:
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt',map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt',map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))
    x = torch.stack(x_list)
    x = torch.reshape(x,(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    y = torch.stack(y_list)
    y = torch.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3]))
    vnorm = norm_velocity(x, device)
    ynorm = norm_acceleration(y, device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    print (vnorm,ynorm)
    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
        h_list=[]
        for run in arr:
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt',map_location=device)
            h_list.append(torch.stack(h))
        h = torch.stack(h_list)
        h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(hnorm)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
    if (model_config['model'] == 'Particles_A'):
        model = InteractionParticles(model_config, device)
        print(f'Training InteractionParticles')
    if (model_config['model'] == 'DiffMesh'):
        model = MeshDiffusion(model_config, device)
        print(f'Training MeshDiffusion for diffusion')
    if (model_config['model'] == 'WaveMesh'):
        model = MeshDiffusion(model_config, device)
        print(f'Training MeshDiffusion for waves')

    # net = f"./log/try_{ntry}/models/best_model_with_1_graphs.pt"
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
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f'Learning rates: {lr}, {lra}')
    print('')
    net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    Nepochs = 22  ######################## 40
    print(f'N epochs: {Nepochs}')
    print('')

    time.sleep(0.5)
    model.train()
    best_loss = np.inf
    list_loss = []
    data_augmentation_loop = 20
    print(f'data_augmentation_loop: {data_augmentation_loop}')

    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
        x = x_list[0][0].clone().detach()
        dataset = data.Data(x=x, pos=x[:, 1:3])
        transform_0 = T.Compose([T.Delaunay()])
        dataset_face = transform_0(dataset).face
        mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
        edge_index_mesh, edge_weight_mesh = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)

    print('')
    time.sleep(0.5)
    for epoch in range(Nepochs + 1):

        if epoch == 1:
            batch_size = model_config['batch_size']
            print(f'batch_size: {batch_size}')
        if epoch == 5:
            if data_augmentation:
                data_augmentation_loop = 200
                print(f'data_augmentation_loop: {data_augmentation_loop}')
        if epoch == 10:
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
        if epoch == 20:
            print('training MLP only ...')
            model.a.requires_grad = False
            # new_a = kmeans.cluster_centers_[kmeans.labels_, :]
            # if gap < 100:
            #     model.a.data = torch.tensor(new_a, device=device)

        total_loss = 0

        for N in range(1, nframes * data_augmentation_loop // batch_size):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            # if ((N - 1) % 1000 == 0) & (epoch < 6):
            #
            #     fig = plt.figure(figsize=(8, 8))
            #     # plt.ion()
            #     embedding = model.a.detach().cpu().numpy()
            #     if embedding.ndim>2:
            #         embedding=embedding[0]
            #     if model_config['embedding'] == 1:
            #         embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
            #         embedding_particle = []
            #         for n in range(nparticle_types):
            #             embedding_particle.append(embedding[index_particles[n], :])
            #         for n in range(nparticle_types):
            #             plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5)
            #     if model_config['embedding'] == 2:
            #         embedding_particle = []
            #         for n in range(nparticle_types):
            #             embedding_particle.append(embedding[index_particles[n], :])
            #         for n in range(nparticle_types):
            #             plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], s=10, alpha=0.75, color=cmap.color(n))
            #
            #     plt.savefig(f"./tmp/Fig_{ntry}_{epoch * 20000 + N - 1}.tif")
            #     plt.close()

            dataset_batch = []
            for batch in range(batch_size):

                k = np.random.randint(nframes - 1)
                x = x_list[run][k].clone().detach()

                if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                    dataset = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
                    dataset_batch.append(dataset)
                    y = h_list[run][k].clone().detach()/hnorm
                    if batch == 0:
                        try:
                            y_batch = y
                        except:
                            a=1
                    else:
                        try:
                            y_batch = torch.cat((y_batch, y), axis=0)
                        except:
                            a=1
                else:
                    distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                    adj_t = (distance < radius ** 2).float() * 1
                    t = torch.Tensor([radius ** 2])
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)
                    dataset_batch.append(dataset)
                    # y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{k}.pt').to(device)
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

                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                optimizer.zero_grad()

                for batch in batch_loader:
                    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                        pred = model(batch, data_id=run - 1)
                    else:
                        pred = model(batch, data_id=run - 1, step=1, vnorm=vnorm, cos_phi=cos_phi, sin_phi=sin_phi)

            loss = (pred - y_batch).norm(2)

            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        sparsity_index = torch.sum((histogram(model.a, 50, -4, 4) > nparticles / 100))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        if (total_loss / nparticles / batch_size / N < best_loss):
            best_loss = total_loss / N / nparticles / batch_size
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs.pt'))
            print("Epoch {}. Loss: {:.6f} sparsity_index {:.3f}  saving model  ".format(epoch,total_loss / N / nparticles / batch_size,sparsity_index.item()))
        else:
            print("Epoch {}. Loss: {:.6f} sparsity_index {:.3f} ".format(epoch,total_loss / N / nparticles / batch_size,sparsity_index.item()))

        list_loss.append(total_loss / N / nparticles / batch_size)

        fig = plt.figure(figsize=(16, 8))
        # plt.ion()

        ax = fig.add_subplot(2, 4, 1)
        plt.plot(list_loss, color='k')
        plt.ylim([0, 0.003])
        plt.xlim([0, 50])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        embedding = []
        for n in range(model.a.shape[0]):
            embedding.append(model.a[n])
        embedding = torch.stack(embedding).detach().cpu().numpy()
        embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])

        embedding_particle = []
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                embedding_particle.append(embedding[index_particles[n]+m*nparticles, :])

        ax = fig.add_subplot(2, 4, 2)
        if (embedding.shape[1] > 2):
            ax = fig.add_subplot(2, 4, 2, projection='3d')
            for n in range(nparticle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2], color=cmap.color(n), s=1)
        else:
            if (embedding.shape[1] > 1):
                for m in range(model.a.shape[0]):
                    for n in range(nparticle_types):
                        plt.scatter(embedding_particle[n+m*nparticle_types][:, 0], embedding_particle[n+m*nparticle_types][:, 1], color=cmap.color(n), s=3)
                plt.xlabel('Embedding 0', fontsize=12)
                plt.ylabel('Embedding 1', fontsize=12)
            else:
                for n in range(nparticle_types):
                    plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5,color=cmap.color(n))

        ax = fig.add_subplot(2, 4, 3)
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
                            plt.plot(rr.detach().cpu().numpy(),
                                     acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'], linewidth=1,
                                     color=cmap.color(k),alpha=0.25)
            acc_list = torch.stack(acc_list)
            plt.xlim([0, 0.05])
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            coeff_norm = acc_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
        elif (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):

            f_list = []
            for n in range(nparticles):
                r0 = torch.tensor(np.ones(1000)).to(device)
                r1 = torch.tensor(np.linspace(0, 2, 1000)).to(device)
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((r0[:, None], r1[:, None], embedding), dim=1)
                h = model.lin_edge(in_features.float())
                h = h[:, 0]
                f_list.append(h)
                plt.plot(r1.detach().cpu().numpy(),
                         h.detach().cpu().numpy() * hnorm.detach().cpu().numpy(), linewidth=1,
                         color='k',alpha=0.05)
            f_list = torch.stack(f_list)
            coeff_norm = f_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
            particle_types = x_list[0][0, :, 5].clone().detach().cpu().numpy()
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
        elif model_config['model'] == 'GravityParticles':
            acc_list = []
            for n in range(nparticles):
                rr = torch.tensor(np.linspace(0, radius * 1.3, 1000)).to(device)
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                acc_list.append(acc)
                plt.plot(rr.detach().cpu().numpy(),acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],color=cmap.color(x[n,5].detach().cpu().numpy()), linewidth=1,alpha=0.25)
            acc_list = torch.stack(acc_list)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim([1E-3, 0.2])
            plt.ylim([1, 1E7])
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            coeff_norm = acc_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
        elif model_config['model'] == 'Particles_A':
            acc_list = []
            for n in range(nparticles):
                rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                if model_config['prediction'] == '2nd_derivative':
                    in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                else:
                    in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], embedding), dim=1)

                acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                acc_list.append(acc)
                if n%5==0:
                    plt.plot(rr.detach().cpu().numpy(),acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],color=cmap.color(x[n,5].detach().cpu().numpy()), linewidth=1,alpha=0.25)
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            acc_list = torch.stack(acc_list)
            coeff_norm = acc_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)

        # Constrain embedding with UMAP of plots clustering
        ax = fig.add_subplot(2, 4, 4)
        if (nparticles<2000) | (epoch==14) | (epoch==19):
            for n in range(nparticle_types):
                plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                            color=cmap.color(n), s=5)
            plt.xlabel('UMAP 0', fontsize=12)
            plt.ylabel('UMAP 1', fontsize=12)
            kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,random_state=13)
            kmeans.fit(proj_interaction)
            for n in range(nparticle_types):
                tmp = kmeans.labels_[index_particles[n]]
                sub_group = np.round(np.median(tmp))
                accuracy = len(np.argwhere(tmp == sub_group)) / len(tmp) * 100
                print(f'Sub-group {n} accuracy: {np.round(accuracy, 3)}')
            for n in range(model_config['ninteractions']):
                plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)
        if (epoch==14) | (epoch==19):
            if (epoch == 19):
                kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=50000, random_state=13)
                kmeans.fit(proj_interaction)

            model_a_=model.a.clone().detach()
            model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
            for k in range(model_config['ninteractions']):
                pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
                temp = model_a_[pos, :].clone().detach()
                print(torch.median(temp, axis=0).values)
                model_a_[pos, :] = torch.median(temp, axis=0).values
            model_a_ = torch.reshape(model_a_, (model.a.shape[0],model.a.shape[1], model.a.shape[2]))
            with torch.no_grad():
                for n in range(model.a.shape[0]):
                    model.a[n]=model_a_[n]
        if bTest:
            if (epoch % 10 == 0) & (epoch > 0):
                best_loss = total_loss / N / nparticles / batch_size
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs.pt'))
                xx, rmserr_list = data_test(model_config, bVisu=True, bPrint=False, step=int(nframes//20))
                model.train()
            if (epoch > 9):
                ax = fig.add_subplot(2, 4, 5)
                for n in range(nparticle_types):
                    plt.scatter(xx[index_particles[n], 1], xx[index_particles[n], 2], s=1,color='k')
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.xlim([0,1])
                plt.ylim([0,1])
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                plt.axis('off')
                ax = fig.add_subplot(2, 4, 6)
                plt.plot(np.arange(len(rmserr_list)), rmserr_list, label='RMSE', color='r')
                plt.ylim([0, 0.1])
                plt.xlim([0, nframes])
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.xlabel('Frame [a.u]', fontsize=14)
                ax.set_ylabel('RMSE [a.u]', fontsize=14, color='r')

        plt.tight_layout()
        plt.savefig(f"./tmp_training/Fig_{ntry}_{epoch}.tif")
        plt.close()
def data_test(model_config, bVisu=False, bPrint=True, index_particles=0, prev_nparticles=0, new_nparticles=0,prev_index_particles=0,best_model=0,step=5):
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

    if (model_config['model'] == 'Particles_A'):
        model = InteractionParticles(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
        p_mass = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_mass[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
        p_elec = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_elec[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])
        model = MeshDiffusion(model_config, device)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))
    if best_model == -1:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    else:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"
    if bPrint:
        print('Graph files N: ', NGraphs - 1)
        print(f'network: {net}')
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    if new_nparticles > 0:  # nparticles larger than initially

        ratio_particles = int(new_nparticles / prev_nparticles)
        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio_particles}')
        print('')

        if ratio_particles>1:
            embedding = model.a.data.clone().detach()
            new_embedding = []

            for n in range(nparticle_types):
                for m in range(ratio_particles):
                    if (n == 0) & (m == 0):
                        new_embedding = embedding[0,prev_index_particles[n]]
                    else:
                        new_embedding = torch.cat((new_embedding, embedding[0,prev_index_particles[n]]), axis=0)

            model.a = nn.Parameter(
                torch.tensor(np.ones((model.ndataset,int(prev_nparticles) * ratio_particles, model_config['embedding'])), device=device, requires_grad=False, dtype=torch.float32))
            model.a.data[0] = new_embedding.float()
            nparticles = new_nparticles
            model_config['nparticles'] = new_nparticles

    ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{ntry}/vnorm.pt', map_location=device).to(device)
    v = vnorm
    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
        hnorm = torch.load(f'./log/try_{ntry}/hnorm.pt', map_location=device).to(device)

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

    x_list=[]
    y_list=[]
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt',map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt',map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()

    if bPrint:
        print('')
        print(f'x: {x.shape}')
        print(f'index_particles: {index_particles[0].shape}')
        print('')
    time.sleep(0.5)

    rmserr_list = []
    discrepency_list = []

    for it in tqdm(range(nframes - 1)):

        x0 = x_list[0][min(it, nframes - 2)].clone().detach()
        x0_next = x_list[0][min(it+1, nframes - 2)].clone().detach()

        if model_config['model'] == 'DiffMesh':
            x[:,1:5]=x0[:,1:5].clone().detach()
            dataset = data.Data(x=x, pos=x[:, 1:3])
            transform_0 = T.Compose([T.Delaunay()])
            dataset_face = transform_0(dataset).face
            mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
            edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)
            dataset_mesh = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)
            with torch.no_grad():
                h = model(dataset_mesh, data_id=0,)
            x[:,6:7] += h * hnorm

        if model_config['model'] == 'WaveMesh':
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset = data.Data(x=x, pos=x[:, 1:3])
            transform_0 = T.Compose([T.Delaunay()])
            dataset_face = transform_0(dataset).face
            mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
            edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)
            dataset_mesh = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)
            with torch.no_grad():
                h = model(dataset_mesh, data_id=0, )
            x[:, 7:8] += h * hnorm
            x[:, 6:7] += x[:, 7:8]



            rmserr = torch.sqrt(torch.mean(torch.sum((x[:,6:7]-x0_next[:,6:7]) ** 2, axis=1)))
            rmserr_list.append(rmserr.item())
        else:
            distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            dataset = data.Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset, data_id=0, step=2, vnorm=v, cos_phi=0, sin_phi=0)  # acceleration estimation
            if model_config['prediction'] == '2nd_derivative':
                y = y * ynorm[4]
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm[4]
                x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5])  # position update

            rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0[:, 1:3]) ** 2, axis=1)))
            rmserr_list.append(rmserr.item())

        discrepency = MMD(x[:, 1:3], x0[:, 1:3])
        discrepency_list.append(discrepency)

        # Sxy = S_e(x[:, 0:2], x0[:, 0:2])
        # Sxy_list.append(Sxy.item())

        if (it % step == 0) & (it>=0) &  bVisu:

            if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                dataset2 = dataset_mesh
            else:
                distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                edge_index2 = adj_t2.nonzero().t().contiguous()
                dataset2 = data.Data(x=x, edge_index=edge_index2)

            fig = plt.figure(figsize=(25, 12))
            # plt.ion()

            for k in range(5):
                if k==0:
                    ax = fig.add_subplot(2, 4, 1)
                    x_ = x00
                    sc = 1
                elif k == 1:
                    ax = fig.add_subplot(2, 4, 2)
                    x_ = x0
                    sc = 1
                elif k == 2:
                    ax = fig.add_subplot(2, 4, 6)
                    x_ = x
                    sc = 1
                elif k == 3:
                    ax = fig.add_subplot(2, 4, 3)
                    x_ = x0
                    sc = 5
                elif k == 4:
                    ax = fig.add_subplot(2, 4, 7)
                    x_ = x
                    sc = 5

                if (k==0) & ((model_config['model'] == 'DiffMesh')|(model_config['model'] == 'WaveMesh')|(model_config['model'] == 'HeatParticles')):
                    plt.scatter(x0_next[:, 6].detach().cpu().numpy(),x[:, 6].detach().cpu().numpy(),s=1, alpha=0.25, cmap='inferno', vmin=0, vmax=2, c='k')
                    plt.xlabel('True temperature [a.u.]', fontsize="14")
                    plt.ylabel('Model temperature [a.u]', fontsize="14")
                elif model_config['model'] == 'GravityParticles':
                    for n in range(nparticle_types):
                        g = p_mass[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 10 * sc
                        plt.scatter(x_[index_particles[n], 1].detach().cpu(), x_[index_particles[n], 2].detach().cpu(), s=g, alpha=0.75,color=cmap.color(n))  # , facecolors='none', edgecolors='k')
                elif model_config['model'] == 'ElecParticles':
                    for n in range(nparticle_types):
                        g = np.abs(p_elec[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)*sc
                        if model_config['p'][n][0]<=0:
                            plt.scatter(x_[index_particles[n], 1].detach().cpu().numpy(),
                                        x_[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                        c='r',alpha=0.5)  # , facecolors='none', edgecolors='k')
                        else:
                            plt.scatter(x_[index_particles[n], 1].detach().cpu().numpy(),
                                        x_[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                        c='b',alpha=0.5)  # , facecolors='none', edgecolors='k')
                elif (model_config['model'] == 'DiffMesh')|(model_config['model'] == 'WaveMesh')|(model_config['model'] == 'HeatParticles'):
                    plt.scatter(x_[:, 1].detach().cpu().numpy(),x_[:, 2].detach().cpu().numpy(), s=20*sc, alpha=0.75,
                                    c=x_[:, 6].detach().cpu().numpy(), cmap='inferno', vmin=0, vmax=2)
                else:
                    for n in range(nparticle_types):

                        plt.scatter(x_[index_particles[n], 1].detach().cpu(), x_[index_particles[n], 2].detach().cpu(),s=sc,color=cmap.color(n))

                if (k > 2) & (model_config['model'] != 'HeatParticles') & (model_config['model'] != 'DiffMesh') & (model_config['model'] != 'WaveMesh'):
                    for n in range(nparticles):
                        plt.arrow(x=x_[n, 1].detach().cpu().item(),y=x_[n, 2].detach().cpu().item(),
                                  dx=x_[n, 3].detach().cpu().item()*model_config['arrow_length'], dy=x_[n, 4].detach().cpu().item()*model_config['arrow_length'],color='k')

                if k<3:
                    if (k == 0) & ((model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh') | (model_config['model'] == 'HeatParticles')):
                        plt.xlim([0, 2])
                        plt.ylim([0, 2])
                    elif (model_config['boundary'] == 'no'):
                        plt.xlim([-1.3, 1.3])
                        plt.ylim([-1.3, 1.3])
                    else:
                        plt.xlim([0,1])
                        plt.ylim([0,1])
                else:
                    if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                        plt.xlim([0.3, 0.7])
                        plt.ylim([0.3, 0.7])
                    else:
                        plt.xlim([-0.25, 0.25])
                        plt.ylim([-0.25, 0.25])

            ax = fig.add_subplot(2, 4, 4)
            plt.plot(np.arange(len(rmserr_list)), rmserr_list, label='RMSE', c='k')
            plt.ylim([0, 0.1])
            plt.xlim([0, nframes])
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.xlabel('Frame [a.u]', fontsize="14")
            ax.set_ylabel('RMSE [a.u]', fontsize="14", color='k')
            if model_config['model'] != 'DiffMesh':
                ax2 = ax.twinx()
                plt.plot(np.arange(len(discrepency_list)), discrepency_list,
                         label='Maximum Mean Discrepencies', c='b')
                ax2.set_ylabel('MMD [a.u]', fontsize="14", color='b')
                ax2.set_ylim([0, 2E-3])

            ax = fig.add_subplot(2, 4, 5)
            pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
            vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.03)
            if model_config['boundary'] == 'no':
                plt.xlim([-1.3, 1.3])
                plt.ylim([-1.3, 1.3])
            else:
                plt.xlim([0,1])
                plt.ylim([0,1])

            ax = fig.add_subplot(2, 4, 8)
            if(model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                # plt.scatter(x0[:, 1].detach().cpu(), x0[:, 2].detach().cpu(), s=10, alpha=0.75,
                #             c=np.abs(x0_next[:, 6].detach().cpu().numpy()-x[:, 6].detach().cpu().numpy()), cmap='inferno', vmin=0, vmax=0.1)
                for n in range(nparticle_types):
                    plt.scatter(N1[index_particles[n]].detach().cpu().numpy(),
                                x[index_particles[n],6].detach().cpu().numpy(), color=cmap.color(n), s=5,
                                alpha=0.5)
                plt.ylim([-0.5, 2.5])
                plt.ylabel('Temperature [a.u]', fontsize="14")


            else:
                temp1 = torch.cat((x, x0), 0)
                temp2 = torch.tensor(np.arange(nparticles), device=device)
                temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
                temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
                temp4 = torch.t(temp4)
                distance3 = torch.sqrt(torch.sum((x[:, 1:3] - x0[:, 1:3]) ** 2, 1))
                p = torch.argwhere(distance3 < 0.3)
                pos = dict(enumerate(np.array((temp1[:, 1:3]).detach().cpu()), 0))
                dataset = data.Data(x=temp1[:, 1:3], edge_index=torch.squeeze(temp4[:, p]))
                vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False)
                if model_config['boundary'] == 'no':
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])
                else:
                    plt.xlim([0,1])
                    plt.ylim([0,1])

            plt.tight_layout()

            plt.savefig(f"./tmp_recons/Fig_{ntry}_{it}.tif")

            plt.close()

    print(f'RMSE: {np.round(rmserr.item(), 4)}')
    if bPrint:
        print(f'ntry: {ntry}')
        print(f'MMD: {np.round(discrepency, 4)}')

    return x.detach().cpu().numpy(), rmserr_list
def data_test_tracking(model_config, bVisu=False, bPrint=True, index_particles=0, prev_nparticles=0, new_nparticles=0,prev_index_particles=0):
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

    if (model_config['model'] == 'Particles_A'):
        model = InteractionParticles(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
        p_mass = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_mass[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
        p_elec = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_elec[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        print(p_elec)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_*")
    NGraphs = int(len(graph_files) / nframes)
    net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    if bPrint:
        print('Graph files N: ', NGraphs - 1)
        print(f'network: {net}')
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    if new_nparticles > 0:  # nparticles larger than initially

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

        model.a = nn.Parameter(
            torch.tensor(np.ones((int(prev_nparticles) * ratio_particles, 2)), device=device, requires_grad=False))
        model.a.data = new_embedding
        nparticles = new_nparticles
        model_config['nparticles'] = new_nparticles

    ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt', map_location=device)
    vnorm = torch.load(f'./log/try_{ntry}/vnorm.pt', map_location=device)
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
    tracking_match = []
    tracking_unmatch = []
    error_tracking = 0

    track = np.arange(nparticles).astype(int)

    for it in tqdm(range(nframes - 2)):

        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_{min(it, nframes - 2)}.pt',
                       map_location=device)
        x = x.to(device)

        x00 = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_{min(it, nframes - 2)}.pt',
                         map_location=device)
        x00 = x00.to(device)

        x0 = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_0_{min(it + 1, nframes - 2)}.pt',
                        map_location=device)
        x0 = x0.to(device)

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = (distance < radius ** 2).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            if model_config['model'] == 'ResNetGNN':
                y = model(dataset, vnorm=v)
            else:
                y = model(dataset, data_id=0, step=2, vnorm=v, cos_phi=0, sin_phi=0)  # acceleration estimation

        y[:, 0] = y[:, 0] * ynorm[4]
        y[:, 1] = y[:, 1] * ynorm[4]

        if model_config['prediction'] == '2nd_derivative':
            x[:, 3:4] = x[:, 3:4] + y  # speed update
        else:
            x[:, 3:4] = y
        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:4])  # position update

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        discrepency = MMD(x[:, 1:3], x0[:, 1:3])
        discrepency_list.append(discrepency)

        # Sxy = S_e(x[:, 1:3], x0[:, 1:3])
        # Sxy_list.append(Sxy.item())

        fx0 = pandas.DataFrame(
            dict(x=x0[:, 1].detach().cpu().numpy().flatten(), y=x0[:, 2].detach().cpu().numpy().flatten(),
                 frame=np.zeros(nparticles)))
        # g = np.random.permutation(np.arange(nparticles)).astype(int)
        g = np.arange(nparticles).astype(int)
        fx = pandas.DataFrame(
            dict(x=x[g, 1].detach().cpu().numpy().flatten(), y=x[g, 2].detach().cpu().numpy().flatten(),
                 frame=np.ones(nparticles)))
        tr = pandas.concat(trackpy.link_df_iter((fx0, fx), 100 * rmserr.detach().cpu().numpy()))
        error_tracking += np.sum((tr.particle.to_numpy()[nparticles:2 * nparticles] - g) != 0)

        if bVisu:

            fig = plt.figure(figsize=(14, 7 * 0.95))
            # plt.ion()

            ax = fig.add_subplot(1, 2, 1)
            tmp = tr.particle.to_numpy()[nparticles:2 * nparticles].astype(int)
            pos = np.argwhere(tmp >= nparticles)
            if len(pos) > 0:
                tmp[pos] = 0
            track = track[tmp]
            plt.scatter(x[:, 1].detach().cpu(), x[:, 2].detach().cpu(), s=5, c=track[:], cmap='prism')
            plt.xlim([0,1])
            plt.ylim([0,1])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.text(-0.25, 1.38, f'frame: {it}')
            plt.text(-0.25, 1.33, f'Graph    {x.shape[0]} nodes ', fontsize=10)

            ax = fig.add_subplot(1, 2, 2)
            plt.scatter(x[:, 1].detach().cpu(), x[:, 2].detach().cpu(), s=5, c='k')
            plt.xlim([0,1])
            plt.ylim([0,1])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.savefig(f"./tmp_recons/Fig_{ntry}_{it}.tif")

            plt.close()

    print(f'RMSE: {np.round(rmserr.item(), 4)}')
    print(f'Tracking error: {error_tracking}')
    if bPrint:
        print('')
        print(f'ntry: {ntry}')
        print(f'MMD: {np.round(discrepency, 4)}')

    return x.detach().cpu().numpy(), rmserr_list
def data_test_generate(model_config, bVisu=True, bDetails=False, step=5):
    print('')
    print('Generating test data ...')

    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    tau = model_config['tau']
    noise_level = model_config['noise_level']
    v_init = model_config['v_init']
    rr = torch.tensor(np.linspace(0, radius * 2, 1000))
    rr = rr.to(device)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if model_config['model'] == 'Particles_A':
        print(f'Generate Particles_A')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        if nparticle_types == 1:
            model = Particles_A(aggr_type=aggr_type, p=p, tau=model_config['tau'],
                                prediction=model_config['prediction'])
        else:
            model = Particles_A(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                                prediction=model_config['prediction'])
    if model_config['model'] == 'GravityParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
    if model_config['model'] == 'HeatParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_H(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            beta=model_config['beta'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])
        model_mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],clamp=model_config['clamp'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if model_config['model'] == 'ElecParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
                print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
                torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = Particles_E(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'],
                            prediction=model_config['prediction'])

    ratio = 1
    prev_nparticles = nparticles
    prev_index_particles = index_particles

    new_nparticles = prev_nparticles * ratio
    nparticles = new_nparticles

    index_particles = []
    for n in range(nparticle_types):
        index_particles.append(np.arange(int(nparticles / nparticle_types) * n,
                                         int(nparticles / nparticle_types) * (n + 1)))

    V1 = v_init * torch.randn((nparticles, 2), device=device)
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]
    h = torch.zeros((nparticles, 1), device=device)
    H1 = torch.ones((nparticles, 1), device=device) + torch.randn((nparticles, 1), device=device) / 4
    N1 = torch.arange(nparticles, device=device)
    N1 = N1[:, None]

    if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
        X1 = torch.rand(nparticles, 2, device=device)
    else:
        X1 = torch.randn(nparticles, 2, device=device) * 0.5


    temp = X1.detach().cpu().numpy()
    sub_group = np.argwhere(temp[:, 0] > 0.93).astype(int)

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
    # i0 = imread('graphs_data/pattern_3.tif')
    # pos = np.argwhere(i0 == 255)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[0],:] = torch.tensor(pos[l[0:1600*3],:]/255,dtype=torch.float32,device=device)
    # pos = np.argwhere(i0 == 128)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[1],:] = torch.tensor(pos[l[0:1600*3],:]/255,dtype=torch.float32,device=device)
    # pos = np.argwhere(i0 == 0)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[2],:] = torch.tensor(pos[l[0:1600*3],:]/255,dtype=torch.float32,device=device)

    # scenario E
    i0 = imread('graphs_data/pattern_5.tif')
    values=i0[(X1[:,0].detach().cpu().numpy()*256).astype(int),(X1[:,1].detach().cpu().numpy()*256).astype(int)]
    H1 = torch.tensor(values/255*1.5,device=device)
    H1 = H1[:,None]

    for run in range(1):

        x_list=[]
        y_list=[]
        h_list=[]

        time.sleep(0.5)

        noise_current = 0 * torch.randn((nparticles, 2), device=device)
        noise_prev_prev = 0 * torch.randn((nparticles, 2), device=device)

        for it in tqdm(range(-int(nframes * model_config['start_frame']), nframes)):

            noise_prev_prev = noise_prev_prev
            noise_prev = noise_current
            noise_current = 0 * torch.randn((nparticles, 2), device=device) * noise_level * radius

            ### TO BE CHANGED ###
            # H1[sub_group] = 2
            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach()), 1)


            if (it >= 0) & (noise_level == 0):
                x_list.append(x)
                # torch.save(x, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')
            if (it >= 0) & (noise_level > 0):
                x_noise = x
                x_noise[:, 1:3] = x[:, 1:3] + noise_current
                x_noise[:, 3:5] = x[:, 3:5] + noise_current - noise_prev
                x_list.append(x_noise)
                # torch.save(x_noise, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')

            if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                dataset = data.Data(x=x, pos=x[:, 1:3])
                transform_0 = T.Compose([T.Delaunay()])
                dataset_face = transform_0(dataset).face
                mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
                edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)
                dataset_mesh = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)

            distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset)
            if model_config['model'] == 'WaveMesh':
                y = y*0
            if (it >= 0) & (noise_level == 0):
                y_list.append(y)
                # torch.save(y, f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{it}.pt')
            if (it >= 0) & (noise_level > 0):
                y_noise = y[:, 0:2] + noise_current - 2 * noise_prev + noise_prev_prev
                y_list.append(y_noise)
                # torch.save(y_noise, f'graphs_data/graphs_particles_{dataset_name}/y_{run}_{it}.pt')

            if model_config['prediction'] == '2nd_derivative':
                V1 += y[:, 0:2]
            else:
                V1 = y[:, 0:2]

            X1 = bc_pos(X1 + V1)

            if model_config['model'] == 'DiffMesh':
                if it >= 0:
                    with torch.no_grad():
                        h = model_mesh(dataset_mesh)
                    H1 += h
                    h_list.append(h)
            if model_config['model'] == 'WaveMesh':
                if it >= 0:
                    with torch.no_grad():
                        pred = model_mesh(dataset_mesh)
                        h += pred
                    H1 += h
                    h_list.append(pred)

            if (run == 0) & (it % step == 0) & (it >= 0) & bVisu:

                fig = plt.figure(figsize=(11.8, 12))
                # plt.ion()
                ax = fig.add_subplot(2, 2, 1)
                if model_config['model'] == 'GravityParticles':
                    for n in range(nparticle_types):
                        g = p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 7.5
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                    x[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                    alpha=0.75,color=cc(n))  # , facecolors='none', edgecolors='k')
                elif (model_config['model'] == 'HeatParticles') | (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                    plt.scatter(x[:, 1].detach().cpu().numpy(),x[:, 2].detach().cpu().numpy(), s=10, alpha=0.75,
                                    c=x[:, 6].detach().cpu().numpy(), cmap='inferno', vmin=0, vmax=2)
                elif model_config['model'] == 'ElecParticles':
                    for n in range(nparticle_types):
                        g = np.abs(p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)
                        if model_config['p'][n][0]<=0:
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='r', alpha=0.5)
                        else:
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='b', alpha=0.5)
                else:
                    for n in range(nparticle_types):
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(), x[index_particles[n], 2].detach().cpu().numpy(), s=3, color=cmap.color(n))
                if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                    plt.text(0, 1.08, f'frame: {it}')
                    plt.text(0, 1.03, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                else:
                    plt.text(-1.25, 1.5, f'frame: {it}')
                    plt.text(-1.25, 1.4, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])

                ax = fig.add_subplot(2, 2, 2)
                plt.scatter(x[:, 1].detach().cpu().numpy(), x[:, 2].detach().cpu().numpy(), s=1, color='k',alpha=0.75)
                if bDetails: # model_config['radius']<0.01:
                    pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                    if (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                        vis = to_networkx(dataset_mesh, remove_self_loops=True, to_undirected=True)
                    else:
                        distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                        adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                        edge_index2 = adj_t2.nonzero().t().contiguous()
                        dataset2 = data.Data(x=x, edge_index=edge_index2)
                        vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False,alpha=0.3)
                if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                else:
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])

                if bDetails:

                    ax = fig.add_subplot(2, 2, 3)
                    if model_config['model'] == 'GravityParticles':
                        for n in range(nparticle_types):
                            g = p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 7.5 * 4
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                        alpha=0.75,
                                        color=cmap.color(n))  # , facecolors='none', edgecolors='k')
                    elif (model_config['model'] == 'HeatParticles') | (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                        plt.scatter(x[:, 1].detach().cpu().numpy(), x[:, 2].detach().cpu().numpy(), s=60, alpha=0.75,
                                    c=x[:, 6].detach().cpu().numpy(), cmap='inferno', vmin=0, vmax=2)
                    elif model_config['model'] == 'ElecParticles':
                        for n in range(nparticle_types):
                            g = np.abs(p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20) * 4
                            if model_config['p'][n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='r', alpha=0.5)
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='b', alpha=0.5)
                    elif model_config['model'] == 'Particles_A':
                        for n in range(nparticle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=50, alpha=0.75,
                                        color=cmap.color(n))

                    if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                        plt.xlim([0.3, 0.7])
                        plt.ylim([0.3, 0.7])
                    else:
                        plt.xlim([-0.25, 0.25])
                        plt.ylim([-0.25, 0.25])

                    if (model_config['model'] != 'HeatParticles') & (model_config['model'] != 'DiffMesh') & (model_config['model'] != 'WaveMesh'):
                        for k in range(nparticles):
                            plt.arrow(x=x[k, 1].detach().cpu().item(),y=x[k, 2].detach().cpu().item(),
                                      dx=x[k, 3].detach().cpu().item()*model_config['arrow_length'], dy=x[k, 4].detach().cpu().item()*model_config['arrow_length'],color='k')

                    ax = fig.add_subplot(2, 2, 4)
                    if (model_config['model'] == 'HeatParticles') | (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh'):
                        for n in range(nparticle_types):
                            plt.scatter(N1[index_particles[n]].detach().cpu().numpy(),
                                        H1[index_particles[n]].detach().cpu().numpy(), color=cmap.color(n), s=5,
                                        alpha=0.5)
                        plt.ylim([-0.5, 2.5])
                        plt.ylabel('Temperature [a.u]', fontsize="14")
                    else:
                        if len(x_list)>30:
                            x_all =torch.stack(x_list)
                            for k in range(nparticles):
                                xc = x_all[-30:-1, k, 1].detach().cpu().numpy().squeeze()
                                yc = x_all[-30:-1, k, 2].detach().cpu().numpy().squeeze()
                                plt.scatter(xc,yc,s=0.05, color='k',alpha=0.75)
                        elif len(x_list)>6:
                            x_all =torch.stack(x_list)
                            for k in range(nparticles):
                                xc = x_all[:, k, 1].detach().cpu().numpy().squeeze()
                                yc = x_all[:, k, 2].detach().cpu().numpy().squeeze()
                                plt.scatter(xc,yc,s=0.05, color='k',alpha=0.75)
                        if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
                            plt.xlim([0,1])
                            plt.ylim([0,1])
                        else:
                            plt.xlim([-1.3, 1.3])
                            plt.ylim([-1.3, 1.3])

                plt.tight_layout()
                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

        torch.save(x_list, f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
        torch.save(h_list, f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt')

    return prev_nparticles, new_nparticles, prev_index_particles, index_particles
def data_plot(model_config, epoch, bPrint, best_model=0):
    print('')

    # for loop in range(25):
    #     print(f'Loop: {loop}')

    model = []
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']

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

    arr = np.arange(0, NGraphs)
    x_list=[]
    y_list=[]
    for run in tqdm(arr):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))
    x = torch.stack(x_list)
    x = torch.reshape(x,(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    y = torch.stack(y_list)
    y = torch.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3]))
    vnorm = norm_velocity(x, device)
    ynorm = norm_acceleration(y, device)
    print(vnorm, ynorm)

    if model_config['model'] == 'DiffMesh':
        h_list=[]
        for run in arr:
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt')
            h_list.append(torch.stack(h))
        h = torch.stack(h_list)
        h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(hnorm)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
    if (model_config['model'] == 'Particles_A'):
        model = InteractionParticles(model_config, device)
        print(f'Training InteractionParticles')
    if (model_config['model'] == 'DiffMesh'):
        model = MeshDiffusion(model_config, device)
        print(f'Training MeshDiffusion')

    if best_model==-1:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs-1}_graphs.pt"
    else:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"
    state_dict = torch.load(net,map_location=device)
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

    time.sleep(0.5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=weight_decay)
    model.train()
    best_loss = np.inf

    print('')
    time.sleep(0.5)
    print('Plotting ...')

    fig = plt.figure(figsize=(16, 8))
    plt.ion()

    embedding = []
    for n in range(model.a.shape[0]):
        embedding.append(model.a[n])
    embedding = torch.stack(embedding).detach().cpu().numpy()
    embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
    embedding_particle = []
    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            embedding_particle.append(embedding[index_particles[n] + m * nparticles, :])

    ax = fig.add_subplot(2, 4, 2)
    if (embedding.shape[1] > 2):
        ax = fig.add_subplot(2, 4, 2, projection='3d')
        for n in range(nparticle_types):
            ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],
                       color=cmap.color(n), s=1)   #
    else:
        if (embedding.shape[1] > 1):
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types):
                    plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=3)
            plt.xlabel('Embedding 0', fontsize=12)
            plt.ylabel('Embedding 1', fontsize=12)
        else:
            for n in range(nparticle_types):
                plt.hist(embedding_particle[n][:, 0], 100, alpha=0.5, color=cmap.color(n))


    rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
    ax = fig.add_subplot(2, 4, 3)
    if model_config['model'] == 'ElecParticles':
        acc_list = []
        for m in range(model.a.shape[0]):
            for k in range(nparticle_types):
                for n in index_particles[k]:
                    embedding = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                    in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding, embedding), dim=1)
                    acc = model.lin_edge(in_features.float())
                    acc = acc[:, 0]
                    acc_list.append(acc)
                    if n % 5 == 0:
                        plt.plot(rr.detach().cpu().numpy(),
                                 acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                                 linewidth=1,
                                 color=cmap.color(k ), alpha=0.25)
        acc_list = torch.stack(acc_list)
        plt.xlim([0, 0.05])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
        coeff_norm = acc_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                          random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)
    elif model_config['model'] == 'DiffMesh':
        h_list = []
        for n in range(nparticles):
            r0 = torch.tensor(np.ones(1000)).to(device)
            r1 = torch.tensor(np.linspace(0, 2, 1000)).to(device)
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((r0[:, None], r1[:, None], embedding), dim=1)
            h = model.lin_edge(in_features.float())
            h = h[:, 0]
            h_list.append(h)
            plt.plot(r1.detach().cpu().numpy(),
                     h.detach().cpu().numpy() * hnorm.detach().cpu().numpy(), linewidth=1,
                     color='k', alpha=0.05)
        h_list = torch.stack(h_list)
        coeff_norm = h_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                          random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)
        particle_types = x_list[0][0, :, 5].clone().detach().cpu().numpy()
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
    elif model_config['model'] == 'GravityParticles':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            plt.plot(rr.detach().cpu().numpy(),
                     acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                     color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
        acc_list = torch.stack(acc_list)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1E-3, 0.2])
        plt.ylim([1, 1E7])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
        coeff_norm = acc_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                          random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)
    elif model_config['model'] == 'Particles_A':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            if model_config['prediction'] == '2nd_derivative':
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            else:
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            if n % 5 == 0:
                plt.plot(rr.detach().cpu().numpy(),
                         acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                         color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
        acc_list = torch.stack(acc_list)
        coeff_norm = acc_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                          random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)

    ax = fig.add_subplot(2, 4, 4)

    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,random_state=13)
    kmeans.fit(proj_interaction)
    for n in range(nparticle_types):
        tmp=kmeans.labels_[index_particles[n]]
        sub_group = np.round(np.median(tmp))
        accuracy=len(np.argwhere(tmp==sub_group))/len(tmp)*100
        print(f'Sub-group {n} accuracy: {np.round(accuracy,3)}')

    for n in range(nparticle_types):
        if proj_interaction.ndim == 1:
            plt.hist(proj_interaction[index_particles[n]], width=0.01, alpha=0.5, color=cmap.color(n))
        if proj_interaction.ndim==2:
            plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                        color=cmap.color(n), s=5)
            plt.xlabel('UMAP 0', fontsize=12)
            plt.ylabel('UMAP 1', fontsize=12)

    for n in range(model_config['ninteractions']):
        plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)

    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    t=[]
    for k in range(model_config['ninteractions']):
        pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        # print(torch.median(temp, axis=0).values)
        model_a_[pos, :] = torch.median(temp, axis=0).values
        t.append(torch.median(temp, axis=0).values)
    model_a_ = torch.reshape(model_a_, (model.a.shape[0], model.a.shape[1], model.a.shape[2]))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_[n]
    embedding = []
    for n in range(model.a.shape[0]):
        embedding.append(model.a[n])
    embedding = torch.stack(embedding).detach().cpu().numpy()
    embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
    embedding_particle = []
    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            embedding_particle.append(embedding[index_particles[n] + m * nparticles, :])

    ax = fig.add_subplot(2, 4, 6)
    if (embedding.shape[1] > 2):
        ax = fig.add_subplot(2, 4, 6, projection='3d')
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                ax.scatter(model.a[m][index_particles[n], 0].detach().cpu().numpy(), model.a[m][index_particles[n], 1].detach().cpu().numpy(), model.a[m][index_particles[n], 1].detach().cpu().numpy(),
                           color=cmap.color(n), s=20)
    else:
        if (embedding.shape[1] > 1):
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types-1,-1,-1):
                    plt.scatter(model.a[m][index_particles[n], 0].detach().cpu().numpy(),model.a[m][index_particles[n], 1].detach().cpu().numpy(),
                                color=cmap.color(n), s=20)
            plt.xlabel('Embedding 0', fontsize=12)
            plt.ylabel('Embedding 1', fontsize=12)
        else:
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types-1,-1,-1):
                    plt.hist(model.a[m][index_particles[n], 0].detach().cpu().numpy(), width=0.01, alpha=0.5, color=cmap.color(n))

    ax = fig.add_subplot(2, 4, 7)
    if model_config['model'] == 'ElecParticles':
        t = model.a.detach().cpu().numpy()
        tmean = np.ones((model_config['nparticle_types'],model_config['embedding']))
        for n in range(model_config['nparticle_types']):
            tmean[n] = np.mean(t[:,index_particles[n],:],axis=(0,1))
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                embedding0 = torch.tensor(tmean[m],device=device) * torch.ones((1000, model_config['embedding']), device=device)
                embedding1 = torch.tensor(tmean[n],device=device) * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding0, embedding1), dim=1)
                acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                plt.plot(rr.detach().cpu().numpy(),acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],linewidth=1,color='k')
        plt.xlim([0,0.02])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    elif model_config['model'] == 'GravityParticles':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            plt.plot(rr.detach().cpu().numpy(),
                     acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                     color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
        acc_list = torch.stack(acc_list)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1E-3, 0.2])
        plt.ylim([1, 1E7])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    elif model_config['model'] == 'Particles_A':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            if model_config['prediction'] == '2nd_derivative':
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            else:
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            if n % 5 == 0:
                plt.plot(rr.detach().cpu().numpy(),
                         acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                         color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
    plt.xlabel('Distance [a.u]', fontsize=12)
    plt.ylabel('MLP [a.u]', fontsize=12)

    ax = fig.add_subplot(2, 4, 8)
    if model_config['model'] == 'Particles_A':
        p = model_config['p']
        p = torch.tensor(p,device=device)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, p[n]))
        for n in range(nparticle_types-1,-1,-1):
            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    if model_config['model'] == 'GravityParticles':
        p = model_config['p']
        p = torch.tensor(p,device=device)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, p[n]))
        for n in range(nparticle_types-1,-1,-1):
            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1, color=cmap.color(n))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1E-3, 0.2])
        plt.ylim([1, 1E7])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    if model_config['model'] == 'ElecParticles':
        p = model_config['p']
        p = torch.tensor(p,device=device)
        psi_output = []
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                temp=model.psi(rr, p[n],p[m])
                plt.plot(rr.detach().cpu().numpy(), np.array(temp.cpu()), linewidth=1,c='k')
        plt.xlim([0, 0.02])

    plt.tight_layout()
    plt.show()

def load_model_config(id=48):
    model_config_test = []

    # gravity
    # 4 types N=960 dim 128 no boundary   GOOD
    if id == 43:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-6,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 2,
                             'clamp': 1E-6,
                             'pred_limit': 1E7,
                             'start_frame': 1,
                             'arrow_length':10,
                             'cmap':'tab20c',
                             'arrow_length':10}
    # 4 types N=960 dim 128 no boundary
    if id == 44:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': 1,
                             'arrow_length':10,
                             'cmap':'tab20c',
                             'arrow_length':10}
    # gravity 4 types N=960 dim 128 no boundary   GOOD
    if id == 45:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 2,
                             'clamp': 0.001,
                             'pred_limit': 1E9,
                             'start_frame': 1,
                             'arrow_length':10,
                             'cmap':'tab20c',
                             'arrow_length':10}

    if id == 46:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': 1,
                             'arrow_length':10,
                             'cmap':'tab20c',
                             'arrow_length':10}
    # 8 types N=960 dim 128 no boundary
    if id == 47:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 8,
                             'ninteractions': 8,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 8).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': 1,
                             'arrow_length':20,
                             'cmap':'tab20c',
                             'arrow_length':10}
    # 16 types N=960 dim 128 no boundary
    if id == 48:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 16,
                             'ninteractions': 16,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 16).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': 1,
                             'arrow_length':20,
                             'cmap':'tab20c',
                             'arrow_length':10}
    # 24 types N=960 dim 128 no boundary
    if id == 49:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 24,
                             'ninteractions': 24,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 24).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': 1,
                             'arrow_length':20,
                             'cmap':'tab20c',
                             'arrow_length':10}

    # particles
    if id == 74:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': 'first_derivative',
                             'upgrade_type': 0,
                             'p': [[1.0413, 1.5615, 1.6233, 1.6012], [1.8308, 1.9055, 1.7667, 1.0855],
                                   [1.785, 1.8579, 1.7226, 1.0584]],
                             'nrun': 2,
                             'start_frame': 0.1,
                             'arrow_length':20,
                             'cmap':'tab10'}
    if id == 75:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': [[1.0413, 1.5615, 1.6233, 1.6012], [1.8308, 1.9055, 1.7667, 1.0855],
                                   [1.785, 1.8579, 1.7226, 1.0584]],
                             'nrun': 2,
                             'start_frame': 0.1,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description': 'pred=first derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14'}
    if id == 76:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': [[1.0413, 1.5615, 1.6233, 1.6012], [1.8308, 1.9055, 1.7667, 1.0855],
                                   [1.785, 1.8579, 1.7226, 1.0584]],
                             'nrun': 2,
                             'start_frame': 0.1,
                             'cmap':'tab10',
                             'arrow_length':20,
                             'description': 'pred=second derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14'}
    if id == 77:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 8,
                             'ninteractions': 8,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': 'first_derivative',
                             'upgrade_type': 0,
                             'p': [[1.2425, 1.3355, 1.3397, 1.3929], [1.629, 1.4932, 1.5311, 1.8677],[1.9852, 1.1892, 1.1544, 1.993],[1.6898, 1.1336, 1.4869, 1.7767],[1.8847, 1.5448, 1.8063, 1.3873],[1.496, 1.4064, 1.9045, 1.733],[1.5108, 1.9904, 1.1665, 1.6975],[1.6153, 1.8557, 1.2758, 1.0684]],
                             'nrun': 2,
                             'start_frame': 0,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description': '8 types pred=first derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14'}
    if id == 78:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 8,
                             'ninteractions': 8,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': 'first_derivative',
                             'upgrade_type': 0,
                             'p': [],
                             'nrun': 2,
                             'start_frame': 0,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description': '8 types pred=first derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14'}

    # elctrostatic
    if id == 84:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 2000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 0,
                             'nrun': 10,
                             'clamp': 0.005,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':40,
                             'cmap':'tab20c',
                             'arrow_length':10}
    if id == 85:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 5E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 0,
                             'nrun': 10,
                             'clamp': 0.005,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':10,
                             'cmap':'tab20b'}
    if id == 86:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 2000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 0,
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':40,
                             'cmap':'tab10',
                             'arrow_length':10}
    if id == 87:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 5E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 0,
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':10,
                             'cmap':'tab10',
                             'description':'Periodic Particles_E is a second derivative simulation, acceleration is function of electrostatic law qiqj/r2 interaction is type-type dependent best_model:22'}

    # heat
    # 8 types boundary periodic N=960
    if id == 120:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 3840,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 2000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'beta': 1E-5,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'HeatParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 8).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': 0.25,
                             'cmap':'tab20b',
                             'arrow_length':10
        }
    # 4 types boundary periodic N=960 mesh diffusion
    if id == 121:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 16,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 3840,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'DiffMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'c': np.linspace(1, 12, 4).tolist(),
                             'beta': 1E-4,
                             'nrun': 2,
                             'clamp': 0.01,
                             'pred_limit': 1E9,
                             'start_frame': 0.3,
                             'cmap':'tab20b',
                             'arrow_length':10
                             }
    # 4 types boundary periodic N=960 mesh wave
    if id == 122:
        model_config_test = {'ntry': id,
                             'input_size': 3,
                             'output_size': 1,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 6000,
                             'nparticle_types': 4,
                             'ninteractions': 1,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(1, 1, 4).tolist(),
                             'c': [0.1, 0.2, 0.5, 1],
                             'beta': 1E-5,
                             'nrun': 2,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'description':'Wave equation fixed particles 4 beta coefficients'
                             }
    if id == 123:
        model_config_test = {'ntry': id,
                             'input_size': 3,
                             'output_size': 1,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 6000,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'DiffMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'c': np.linspace(1, 12, 4).tolist(),
                             'beta': 1E-4,
                             'nrun': 2,
                             'clamp': 0.01,
                             'pred_limit': 1E9,
                             'start_frame': 0.3,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'description':'Heat equation fixed particles 4 conductivities'
                             }
    if id == 124:
        model_config_test = {'ntry': id,
                             'input_size': 3,
                             'output_size': 1,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 6000,
                             'nparticle_types': 4,
                             'ninteractions': 1,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(1, 1, 4).tolist(),
                             'c': [0.1, 0.2, 0.5, 1],
                             'beta': 1E-5,
                             'nrun': 10,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'description':'Wave equation fixed particles 4 beta coefficients'
                             }
    if id == 125:
        model_config_test = {'ntry': id,
                             'input_size': 3,
                             'output_size': 1,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 6000,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'DiffMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'c': np.linspace(1, 12, 4).tolist(),
                             'beta': 1E-4,
                             'nrun': 10,
                             'clamp': 0.01,
                             'pred_limit': 1E9,
                             'start_frame': 0.3,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'description':'Heat equation fixed particles 4 conductivities'
                             }
    if id == 126:
        model_config_test = {'ntry': id,
                             'input_size': 3,
                             'output_size': 1,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 6000,
                             'nparticle_types': 4,
                             'ninteractions': 1,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 0,
                             'p': np.linspace(1, 1, 4).tolist(),
                             'c': [2, 2, 2, 2],
                             'beta': 1E-5,
                             'nrun': 2,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'description':'Wave equation fixed particles 4 beta coefficients'
                             }

    for key, value in model_config_test.items():
        print(key, ":", value)

    if model_config_test['model'] == 'Particles_G':
        print(
            'Particles_G is a second derivative simulation, acceleration is function of gravity law mj/r2 interaction is type dependent')

    return model_config_test

if __name__ == '__main__':

    print('')
    print('version 1.5 231024')
    print('use of https://github.com/gpeyre/.../ml_10_particle_system.ipynb')
    print('')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    scaler = StandardScaler()
    S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


    gtestlist = [122,123,124,125] # [75,84,85] #[121, 84, 85, 46, 47, 48] # [121, 84, 85, 46] #[85, 75 ,84] #,75,,84] #[46, 47, 48, 121, 75, 84]

    for gtest in gtestlist:

        model_config = load_model_config(id=gtest)

        # if (gtest>=0) and (gtest<10):
        #     model_config = load_model_config(id=44)
        # if (gtest>=10) and (gtest<20):
        #     model_config = load_model_config(id=45)
        # if (gtest>=20) and (gtest<30):
        #     model_config = load_model_config(id=46)
        # model_config['ntry']=gtest
        cmap = cc(model_config=model_config)

        sigma = model_config['sigma']
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

        data_generate(model_config, bVisu=False, bDetails=True, bSave=True, step=10)
        data_train(model_config, gtest)
        # data_plot(model_config, epoch=-1, bPrint=True, best_model=20)
        # x, rmserr_list = data_test(model_config, bVisu=True, bPrint=True, best_model=-1, step=100)
        # prev_nparticles, new_nparticles, prev_index_particles, index_particles = data_test_generate(model_config, bVisu=True, bDetails=True, step=10)
        # x, rmserr_list = data_test(model_config, bVisu = True, bPrint=True, index_particles=index_particles, prev_nparticles=prev_nparticles, new_nparticles=new_nparticles, prev_index_particles=prev_index_particles, best_model=-1, step=100)


