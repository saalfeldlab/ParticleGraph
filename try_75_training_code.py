
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
from shutil import copyfile
from tensorboardX import SummaryWriter
from prettytable import PrettyTable

class MLP(nn.Module):
    def __init__(self, in_feats=2, out_feats=2, num_layers=3, hidden=128):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hidden, device=device))
        if num_layers > 2:
            for i in range(1, num_layers - 1):
                layer = nn.Linear(hidden, hidden, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden, out_feats, device=device)
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
    def __init__(self, in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionParticles, self).__init__(aggr='mean')  # "Add" aggregation.

        self.lin_edge = MLP(in_feats=8, out_feats=2, num_layers=3, hidden=16)
        # self.lin_node = MLP(in_feats=4, out_feats=1, num_layers=2, hidden=16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x,x))
        return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type= x_i[:,5]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]
        # x_j_type = x_j[:, 5]
        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:,None]), dim=-1)

        return self.lin_edge(in_features)
    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)

def normalize99(Y, lower=1,upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return x01, x99


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # flist = ['ReconsGraph']
    # for folder in flist:
    #     files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/Graph/{folder}/*")
    #     for f in files:
    #         os.remove(f)

    datum = '230412'
    print(datum)

    niter=200
    nparticles=2000
    sigma = .005;
    radius=0.075

    x = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_0.pt')
    edges = torch.load(f'graphs_data/graphs_particles_{datum}/edge_index_0_0.pt')
    y = torch.load(f'graphs_data/graphs_particles_{datum}/y_0_0.pt')

    dataset = data.Data(x=x.cuda(), edge_index=edges.cuda())
    print('num_nodes: ', x.shape[0])
    print('dataset.num_node_features: ', dataset.num_node_features)

    l_dir = os.path.join('.', 'log')
    try:
        try_index = np.max([int(index.split('_')[1]) for index in os.listdir(l_dir)]) + 1
    except ValueError:
        try_index = 0

    try_index = 521

    log_dir = os.path.join(l_dir, 'try_{}'.format(try_index))
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    best_loss = np.inf

    graph_files = glob.glob(f"graphs_data/graphs_particles_{datum}/edge_index*")
    NGraphs=int(len(graph_files)/niter)
    print ('Graph files N: ',NGraphs)

    arr = np.arange(0,NGraphs-1,2)
    for run in tqdm(arr):
        kr = np.arange(0,niter-1,4)
        for k in kr:
            x=torch.load(f'graphs_data/graphs_particles_{datum}/x_{run}_{k}.pt')
            y=torch.load(f'graphs_data/graphs_particles_{datum}/y_{run}_{k}.pt')
            if (run == 0) & (k == 0):
                xx = x
                yy = y
            else:
                xx = torch.concatenate((x, xx))
                yy = torch.concatenate((y, yy))

    mvx = torch.mean(xx[:, 2])
    mvy = torch.mean(xx[:, 3])
    vx = torch.std(xx[:, 2])
    vy = torch.std(xx[:, 3])
    nvx = np.array(xx[:, 2].detach().cpu())
    vx01, vx99 = normalize99(nvx)
    nvy = np.array(xx[:, 3].detach().cpu())
    vy01, vy99 = normalize99(nvy)
    vnorm = torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))

    print(f'v_x={mvx} +/- {vx}')
    print(f'v_y={mvy} +/- {vy}')
    print(f'vx01={vx01} vx99={vx99}')
    print(f'vy01={vy01} vy99={vy99}')

    max = torch.mean(yy[:, 0])
    may = torch.mean(yy[:, 1])
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = normalize99(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = normalize99(nay)

    ynorm = torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

    print(f'acc_x={max} +/- {ax}')
    print(f'acc_y={may} +/- {ay}')
    print(f'ax01={ax01} ax99={ax99}')
    print(f'ay01={ay01} ay99={ay99}')

    model = InteractionParticles()

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

    optimizer= torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()

    print(' Training  ...')

    for epoch in range(100):

        model.train()
        total_loss = 0

        for N in range(1,NGraphs*niter):

            run = np.random.randint(NGraphs - 1)
            # if epoch<8:
            #     k = 30+np.random.randint(niter - 31)
            # else:
            k= np.random.randint(niter-1)

            x=torch.load(f'graphs_data/graphs_particles_{datum}/x_{run}_{k}.pt')
            edges=torch.load(f'graphs_data/graphs_particles_{datum}/edge_index_{run}_{k}.pt')
            y=torch.load(f'graphs_data/graphs_particles_{datum}/y_{run}_{k}.pt')

            y[:, 0] = y[:, 0] / ynorm[4]
            y[:, 1] = y[:, 1] / ynorm[5]

            dataset = data.Data(x=x, edge_index=edges)

            optimizer.zero_grad()
            pred = model(dataset)

            loss = (pred-y).norm(2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if N%1000==0:
                print("N {} Loss: {:.6f}".format(N,total_loss/N/nparticles))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss/N/nparticles))

        if (total_loss<best_loss):
            best_loss=total_loss
            torch.save({ 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', 'best_model.pt'))
            print('\t\t Saving model')


