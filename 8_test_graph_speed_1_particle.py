import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from tensorboardX import SummaryWriter
import os
from shutil import copyfile
import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
from torch_geometric.nn import GATConv
import glob
from matplotlib.animation import FuncAnimation


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, x



class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 32
        self.in_head = 4
        self.out_head = 1

        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, 2, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        emb = x
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return emb, x

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Transform node feature matrix.
        self_x = self.lin_self(x)
        #x = self.lin(x)

        return self_x + self.propagate(edge_index, size=(x.size(0), x.size(0)), x=self.lin(x))

    def message(self, x_i, x_j, edge_index, size):
        # Compute messages
        # x_j has shape [E, out_channels]

        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


def test(model):

    model.eval()

    kr = np.arange(98)
    np.random.shuffle(kr)

    run = 45
    total_loss= 0

    for k in kr:
        x = torch.load(f'graphs_1_particle_{datum}/X_{run}_{k}.pt')
        edges = torch.load(f'graphs_1_particle_{datum}/edge_index_{run}_{k}.pt')
        ys = torch.load(f'graphs_1_particle_{datum}/label_{run}_{k}.pt')
        ys = ys.long()
        dataset = data.Data(x=x, edge_index=edges, y=ys, num_classes=2)

        with torch.no_grad():
            embedding, pred = model(dataset)

        label = torch.load(f'graphs_1_particle_{datum}/X_{run}_{k + 1}.pt')
        label = label[:, 2:4]
        total_loss += torch.sqrt((pred - label).norm(2))

    return total_loss/98


def distmat_square(X,Y):
    return torch.sum( (X[:,None,:] - Y[None,:,:])**2, axis=2 )

def distmat_square2(X, Y):
    X_sq = (X ** 2).sum(axis=-1)
    Y_sq = (Y ** 2).sum(axis=-1)
    cross_term = X.matmul(Y.T)
    return X_sq[:, None] + Y_sq[None, :] - 2 * cross_term


if __name__ == "__main__":

    flist = ['temp']
    for folder in flist:
        files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/Graph/{folder}/*")
        for f in files:
            os.remove(f)

    datum='230403'
    k=0
    niter=100


    x=torch.load(f'graphs_1_particle_{datum}/X_0_{k}.pt')
    edges=torch.load(f'graphs_1_particle_{datum}/edge_index_0_{k}.pt')
    ys=torch.load(f'graphs_1_particle_{datum}/label_0_{k}.pt')
    ys=ys.long()
    dataset = data.Data(x=x, edge_index=edges, y=ys, num_classes=1)
    print('num_nodes: ', x.shape[0])
    print('dataset.num_node_features: ', dataset.num_node_features)
    print('dataset.num_classes: ', dataset.num_classes)

    #model=GAT()

    model = GNNStack(max(dataset.num_node_features, 1), 32, 2)
    state_dict = torch.load(f"./log/try_10/models/best_model.pt")
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()


    graph_files = glob.glob(f"graphs_1_particle_{datum}/edge_index*")
    NGraphs=int(len(graph_files)/niter)
    print ('Graph files N: ',NGraphs)

    arr = np.arange(NGraphs-1)
    for run in tqdm(arr):
        kr = np.arange(0,niter-1,2)
        for k in kr:
            x = torch.load(f'graphs_1_particle_{datum}/X_{run}_{k}.pt')
            if run==0 & k==0:
                xx=x
            else:
                xx=torch.concatenate((x,xx))


    mx = torch.mean(xx[: , 2])
    my = torch.mean(xx[: , 3])
    sx = 3* torch.std(xx[: , 2])
    sy = 3* torch.std(xx[: , 3])


    run=0
    niter=100
    n=2000


    Zsvg1 = torch.zeros((n, 4, niter))  # to store all the intermediate time
    Zsvg2 = torch.zeros((n, 2, niter))  # to store all the intermediate time

    xt = torch.load(f'graphs_1_particle_{datum}/X_{run}_{0}.pt')
    xt[:, 2:4] = x[:, 2:4] / sx + 0.5

    x = xt

    edges = torch.load(f'graphs_1_particle_{datum}/edge_index_{run}_{k}.pt')
    ys = torch.load(f'graphs_1_particle_{datum}/label_{run}_{k}.pt')
    ys = ys.long()
    dataset = data.Data(x=x, edge_index=edges, y=ys, num_classes=2)

    for k in tqdm(range(niter)):

        xt = torch.load(f'graphs_1_particle_{datum}/X_{run}_{k}.pt')

        emb, pred = model(dataset)

        x[:, 0:2] = x[:, 0:2] + (pred-0.5)*sx
        x[:, 2:4] = pred

        distance = distmat_square(x[:, 0:2], x[:, 0:2])
        t = torch.Tensor([0.075 * 0.075])  # threshold
        adj_t = (distance < 0.075 * 0.075).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        edge_index = edge_index.detach().cpu()

        dataset = data.Data(x=x, edge_index=edge_index, y=ys, num_classes=2)

        Zsvg1[:, :, k] = x.clone().detach()  # for later display
        Zsvg2[:, :, k] = xt[:, 0:2].clone().detach()  # for later display



    c1 = np.array([220, 50, 32]) / 255
    c2 = np.array([0, 114, 178]) / 255

    def animate(t):

        fig.clf()
        plt.scatter(Zsvg1[:, 0, t], Zsvg1[:, 1, t] + 0.08, s=3, color=c1, label='GT')
        plt.scatter(Zsvg2[:, 0, t], Zsvg2[:, 1, t] + 0.08, s=3, color=c2, label='prediction')
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.axis('equal')
        plt.axis([-0.3, 1.3, -0.3, 1.3])
        # plt.tight_layout()
        plt.text(-0.25, 1.38, f'frame: {t}')


    # fig = plt.figure(figsize=(12, 12))
    #
    # ani = FuncAnimation(fig, animate, frames=98, interval=0.02)
    # plt.show()


    fig = plt.figure(figsize=(9, 9))
    for t in range(0, niter):
        animate(t)
        plt.savefig(f"./temp/Fig_{run}_{t}.tif")

