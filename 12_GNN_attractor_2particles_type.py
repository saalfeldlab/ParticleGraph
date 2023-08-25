
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

def psi(r,p):
    return -p[2]*torch.exp(-r**p[0] / (2 * sigma ** 2)) + p[3]* torch.exp(-r**p[1] / (2 * sigma ** 2))

class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=2, hidden=128):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        layer = nn.Linear(hidden, out_feats)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(nn.Linear(in_feats, hidden))
        if num_layers > 2:
            for i in range(1, num_layers - 1):
                layer = nn.Linear(hidden, hidden)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden, out_feats)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

class InteractionParticles_0(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super(InteractionParticles_0, self).__init__(aggr='mean')  # "Add" aggregation.
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.propagate(edge_index, x=(x,x))
        oldv = x[:,2:4]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):
        r = torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)   # squared distance

        psi = -p0[2] * torch.exp(-r ** p1[0] / (2 * sigma ** 2)) + p0[3] * torch.exp(-r ** p0[1] / (2 * sigma ** 2))

        return psi[:,None] * (x_i[:,0:2] - x_j[:,0:2])


class InteractionParticles_1(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super(InteractionParticles_1, self).__init__(aggr='mean')  # "Add" aggregation.
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.propagate(edge_index, x=(x,x))
        oldv = x[:,2:4]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):
        r = torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)   # squared distance

        psi = -p1[2] * torch.exp(-r ** p1[0] / (2 * sigma ** 2)) + p1[3] * torch.exp(-r ** p1[1] / (2 * sigma ** 2))

        return psi[:,None] * (x_i[:,0:2] - x_j[:,0:2])

if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = glob.glob(f"./ReconsGraph/*")
    for f in files:
        os.remove(f)

    nparticles = 2000  # number of points per classes
    nframes = 200
    sigma = .005;
    nrun=200
    radius=0.075

    datum='230412_test_good'
    run=0
    folder=f'graphs_data/graphs_particles_{datum}'

    if not (os.path.exists(folder)):
        os.mkdir(folder)
    else:
        files = glob.glob(f"./{folder}/*")
        for f in files:
            os.remove(f)



    for run in tqdm(range(nrun)):

        # X1=torch.load("X1.pt")
        X1 = torch.rand(nparticles,2,device=device)

        t = torch.tensor(np.linspace(-1.5,1.5,1000))

        X1t = torch.zeros((nparticles,2,nframes)) # to store all the intermediate time

        p0= torch.tensor([1.27, 1.41, 0.82, 0.08])
        p1 = torch.tensor([1.82, 1.72, 0.12, 0.45])
        p0[2:4] *= 2E-1 /3
        p1[2:4] *= 2E-1

        V1 = torch.zeros((nparticles,2),device=device)
        T1 =  torch.cat( ( torch.zeros(int(nparticles/2), device=device) , torch.ones(int(nparticles/2), device=device) ),0)
        T1=T1[:,None]
        T1 = torch.concatenate((T1, T1),1)

        rr = torch.tensor(np.linspace(0, 0.015, 100),device=device)
        psi1 = psi(rr,p1)

        model0 = InteractionParticles_0(hidden_size=4, layers=5)
        model1 = InteractionParticles_1(hidden_size=4, layers=5)

        for it in range(nframes):

            X1t[:,:,it] = X1.clone().detach() # for later display

            X1 = X1 + V1

            distance=torch.sum((X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius**2]) # threshold
            adj_t = (distance < radius**2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            torch.save(edge_index,f'graphs_data/graphs_particles_{datum}/edge_index_{run}_{it}.pt')

            x=torch.concatenate((X1.clone().detach(),V1.clone().detach(),T1.clone().detach()),1)
            torch.save(x,f'graphs_data/graphs_particles_{datum}/x_{run}_{it}.pt')

            dataset = data.Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                y0 = model0(dataset) * (x[:,4:6]==0)
                y1 = model1(dataset) * (x[:,4:6]==1)

            y=y0+y1

            torch.save(y, f'graphs_data/graphs_particles_{datum}/y_{run}_{it}.pt')

            V1 += y

            if it%10 == 0:

                c1 = np.array([220, 50, 32]) / 255
                c2 = np.array([0, 114, 178]) / 255
                fig = plt.figure(figsize=(14, 7))

                ax = fig.add_subplot(1,2,2)
                pos=dict(enumerate(np.array(x[:,0:2].detach().cpu()), 0))
                vis = to_networkx(dataset,remove_self_loops=True, to_undirected=True)
                nx.draw_networkx(vis, pos=pos, node_size=10, linewidths=0, with_labels=False)
                plt.xlim([-0.3, 1.3])
                plt.ylim([-0.3, 1.3])
                plt.text(-0.25, 1.33, f'Graph    {x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)

                ax = fig.add_subplot(1,2,1)
                plt.scatter(X1t[0:int(nparticles/2), 0, it], X1t[0:int(nparticles/2), 1, it], s=3, color=c1)
                plt.scatter(X1t[int(nparticles/2):nparticles, 0, it], X1t[int(nparticles/2):nparticles, 1, it], s=3, color=c2)
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.xlim([-0.3, 1.3])
                plt.ylim([-0.3, 1.3])
                # plt.tight_layout()
                plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                plt.text(-0.1, 1.25, f'p0: {np.array(p0.cpu())}', color=c1)
                plt.text(-0.1, 1.20, f'p1: {np.array(p1.cpu())}', color=c2)
                plt.text(-0.25, 1.38, f'frame: {it}')
                plt.savefig(f"./ReconsGraph/Fig_{run}_{it}.tif")
                plt.close()




