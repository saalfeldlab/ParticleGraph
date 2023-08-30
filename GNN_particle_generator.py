
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

def psi(r,p):
    return -p[2]*torch.exp(-r**p[0] / (2 * sigma ** 2)) + p[3]* torch.exp(-r**p[1] / (2 * sigma ** 2))



class InteractionParticles_0(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self):
        super(InteractionParticles_0, self).__init__(aggr=aggr_type)  # "Add" aggregation.
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.propagate(edge_index, x=(x,x))
        oldv = x[:,2:4]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:,0:2] - x_j[:,0:2])**2,axis=1)   # squared distance

        psi = -p0[2] * torch.exp(-r ** p1[0] / (2 * sigma ** 2)) + p0[3] * torch.exp(-r ** p0[1] / (2 * sigma ** 2))

        return psi[:,None] * bc_diff(x_i[:,0:2] - x_j[:,0:2])

class InteractionParticles_1(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self):
        super(InteractionParticles_1, self).__init__(aggr=aggr_type)  # "Add" aggregation.
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.propagate(edge_index, x=(x,x))
        oldv = x[:,2:4]
        acc = newv - oldv
        return acc

    def message(self, x_i, x_j):

        r = torch.sum(bc_diff(x_i[:,0:2] - x_j[:,0:2])**2,axis=1)   # squared distance

        psi = -p1[2] * torch.exp(-r ** p1[0] / (2 * sigma ** 2)) + p1[3] * torch.exp(-r ** p1[1] / (2 * sigma ** 2))

        return psi[:,None] * bc_diff(x_i[:,0:2] - x_j[:,0:2])

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

        self.noise_level = model_config['noise_level']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers, hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device='cuda:0', requires_grad=True))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x[:, 4:6] = self.a[x[:, 6].detach().cpu().numpy(), 0:2]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x,x))
        return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=bc_diff(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type = x_i[:,4:6]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]

        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type), dim=-1)   # [:,None].repeat(1,4)

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


class EdgeNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        return self.new_edges

    def message(self, x_i, x_j, edge_feature):

        r = torch.sqrt(torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type= x_i[:,4]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]

        d = r

        self.new_edges = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:,None].repeat(1,4)), dim=-1)

        return d

class InteractionNetworkEmb(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, nlayers, embedding, device):
        super().__init__(aggr='add')  # "Add" aggregation.

        self.nlayers = nlayers
        self.device = device
        self.embedding = embedding

        self.lin_edge = MLP(input_size=3*self.embedding, hidden_size=3*self.embedding, output_size=self.embedding, nlayers= self.nlayers, device=self.device)
        self.lin_node = MLP(input_size=2*self.embedding, hidden_size=2*self.embedding, output_size=self.embedding, nlayers= self.nlayers, device=self.device)


    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        node_out = x + node_out
        edge_out = edge_feature + self.new_edges

        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):

        x = torch.cat((edge_feature, x_i, x_j ), dim=-1)

        x = self.lin_edge(x)
        self.new_edges = x

        return x

class ResNetGNN(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(self,model_config, device):
        super().__init__()

        self.hidden_size = model_config['hidden_size']
        self.embedding = model_config['embedding']
        self.nlayers = model_config['n_mp_layers']
        self.device = device
        self.noise_level = model_config['noise_level']

        self.edge_init = EdgeNetwork()

        self.layer = torch.nn.ModuleList([InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device) for _ in range(self.nlayers)])
        self.node_out = MLP(input_size=self.embedding, hidden_size=self.hidden_size, output_size=2, nlayers=3, device=self.device)

        self.embedding_node = MLP(input_size=8, hidden_size=self.embedding, output_size=self.embedding, nlayers=3, device=self.device)
        self.embedding_edges = MLP(input_size=11, hidden_size=self.embedding, output_size=self.embedding, nlayers=3, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((int(nparticles), 1)), device=self.device, requires_grad=True))

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x[:, 4] = self.a[x[:, 6].detach().cpu().numpy(), 0]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        node_feature = torch.cat((x[:,0:4],x[:,4:5].repeat(1,4)), dim=-1)

        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]),requires_grad=False, device='cuda:0') * self.noise_level
        node_feature= node_feature+noise
        edge_feature = self.edge_init(node_feature, edge_index)

        node_feature = self.embedding_node(node_feature)
        edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer[0](node_feature, data.edge_index, edge_feature=edge_feature)

        pred = self.node_out(node_feature)

        return pred

if __name__ == '__main__':

    # version 1.15 230825

    files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/ReconsGraph2/*")
    for f in files:
        os.remove(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_config = {'ntry': 515,
    #                 'input_size': 8,
    #                 'output_size': 2,
    #                 'hidden_size': 16,
    #                 'n_mp_layers': 3,
    #                 'noise_level': 0,
    #                 'radius': 0.075,
    #                 'datum': '230824',
    #                 'nparticles' : 2000,  # number of points per classes
    #                 'nframes' : 400,
    #                 'sigma' : .005,
    #                 'boundary' : 'per', # periodic   'no'  # no boundary condition
    #                 'model': 'InteractionParticles'}
    #
    # model_config = {'ntry': 516,
    #                 'embedding': 128,
    #                 'hidden_size': 32,
    #                 'n_mp_layers': 4,
    #                 'noise_level': 0,
    #                 'radius': 0.075,
    #                 'datum': '230824',
    #                 'nparticles': 2000,  # number of points per classes
    #                 'nframes': 400,
    #                 'sigma': .005,
    #                 'boundary': 'per',  # periodic   'no'  # no boundary condition
    #                 'model': 'ResNetGNN'}

    # model_config = {'ntry': 517,
    #                 'input_size': 8,
    #                 'output_size': 2,
    #                 'hidden_size': 16,
    #                 'n_mp_layers': 3,
    #                 'noise_level': 0,
    #                 'datum': '230825',
    #                 'nparticles': 2000,  # number of points per classes
    #                 'nframes': 400,
    #                 'sigma': .005,
    #                 'radius': 0.125,
    #                 'boundary' : 'per', # periodic   'no'  # no boundary condition
    #                 'model': 'InteractionParticles'}

    # model_config = {'ntry': 518,
    #                 'input_size': 8,
    #                 'output_size': 2,
    #                 'hidden_size': 16,
    #                 'n_mp_layers': 3,
    #                 'noise_level': 0,
    #                 'radius': 0.075,
    #                 'datum': '230412_test_bis',
    #                 'nparticles' : 2000,  # number of points per classes
    #                 'nframes' : 200,
    #                 'sigma' : .005,
    #                 'boundary' : 'no', # periodic   'no'  # no boundary condition
    #                 'model': 'InteractionParticles'}

    # model_config = {'ntry': 519,
    #                 'input_size': 9,
    #                 'output_size': 2,
    #                 'hidden_size': 16,
    #                 'n_mp_layers': 3,
    #                 'noise_level': 0,
    #                 'radius': 0.075,
    #                 'datum': '230412_test_bis',
    #                 'nparticles' : 2000,  # number of points per classes
    #                 'nframes' : 200,
    #                 'sigma' : .005,
    #                 'boundary' : 'no', # periodic   'no'  # no boundary condition
    #                 'model': 'InteractionParticles'}

    model_config = {'ntry': 520,
                    'input_size': 9,
                    'output_size': 2,
                    'hidden_size': 16,
                    'n_mp_layers': 3,
                    'noise_level': 0,
                    'radius': 0.075,
                    'datum': '230412_test_bis',
                    'nparticles' : 2000,  # number of points per classes
                    'nframes' : 200,
                    'sigma' : .005,
                    'aggr': 'add',
                    'boundary' : 'no', # periodic   'no'  # no boundary condition
                    'model': 'InteractionParticles'}


    nrun= 20

    print('')
    ntry = model_config['ntry']
    print(f'ntry: {ntry}')
    datum = model_config['datum']
    print(f'datum: {datum}')
    nparticles = model_config['nparticles']    # number of particles
    print(f'nparticles: {nparticles}')
    nframes = model_config['nframes']
    print(f'nframes: {nframes}')
    radius = model_config['radius']
    print(f'radius: {radius}')
    sigma = model_config['sigma']
    print(f'sigma: {sigma}')
    boundary = model_config['boundary']
    print(f'boundary: {boundary}')
    aggr_type = model_config['aggr']
    print(f'aggr_type: {aggr_type}')


    if boundary == 'no':  # change this for usual BC
        def bc_pos(X):
            return X
        def bc_diff(D):
            return D
    else:
        def bc_pos(X):
            return torch.remainder(X, 1.0)
        def bc_diff(D):
            return torch.remainder(D - .5, 1.0) - .5

    time.sleep(0.5)

    print('')
    print('Generating data ...')


    for run in tqdm(range(40)):

        X1 = torch.rand(nparticles,2,device=device)
        X1t = torch.zeros((nparticles,2,nframes)) # to store all the intermediate time

        p0 = torch.rand(1, 4)
        p0 = torch.squeeze(p0)
        p0[0] = p0[0] + 1
        p0[1] = p0[1] + 1
        p0[2:4]=p0[2:4]/10
        p1 = torch.rand(1, 4)
        p1 = torch.squeeze(p1)
        p1[0] = p1[0] + 1
        p1[1] = p1[1] + 1
        p1[2:4]=p1[2:4]/10

        V1 = torch.zeros((nparticles,2),device=device)
        T1 = torch.cat( ( torch.zeros(int(nparticles/2), device=device) , torch.ones(int(nparticles/2), device=device) ),0)
        T1 = T1[:,None]
        T1 = torch.concatenate((T1, T1),1)
        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:,None]

        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)
        psi0 = psi(rr, p0)
        psi1 = psi(rr, p1)

        model0 = InteractionParticles_0()
        model1 = InteractionParticles_1()

        for it in range(nframes):

            X1t[:,:,it] = X1.clone().detach() # for later display

            X1 = bc_pos(X1 + V1)

            distance=torch.sum(bc_diff(X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius**2]) # threshold
            adj_t = (distance < radius**2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            x=torch.concatenate((X1.clone().detach(),V1.clone().detach(),T1.clone().detach(),N1.clone().detach()),1)
            dataset = data.Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                y0 = model0(dataset) * (x[:,4:6]==0)
                y1 = model1(dataset) * (x[:,4:6]==1)

            y=y0+y1

            V1 += y

            if (run>-1) & (it%5==0):

                distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                adj_t2 = ((distance < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                edge_index2 = adj_t2.nonzero().t().contiguous()
                dataset2 = data.Data(x=x, edge_index=edge_index2)

                c1 = np.array([220, 50, 32]) / 255
                c2 = np.array([0, 114, 178]) / 255

                fig = plt.figure(figsize=(14, 6.5))
                # plt.ion()
                ax = fig.add_subplot(1,2,2)
                pos=dict(enumerate(x[:,0:2].detach().cpu().numpy(), 0))
                vis = to_networkx(dataset2,remove_self_loops=True, to_undirected=True)
                nx.draw_networkx(vis, pos=pos, node_size=10, linewidths=0, edge_color='b', with_labels=False)
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
                plt.text(-0.25, 1.38, f'frame: {it}')
                plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                plt.text(-0.25, 1.25, f'p0: {np.round(np.array(p0.cpu()),4)}', color=c1)
                plt.text(-0.25, 1.20, f'p1: {np.round(np.array(p1.cpu()),4)}', color=c2)

                ax = fig.add_subplot(5, 5, 21)
                plt.plot(rr.detach().cpu().numpy(),np.array(psi0.cpu()), color=c1, linewidth=1)
                plt.plot(rr.detach().cpu().numpy(),np.array(psi1.cpu()), color=c2, linewidth=1)
                plt.plot(rr.detach().cpu().numpy(),rr.detach().cpu().numpy()*0, color=[0, 0, 0], linewidth=0.5)

                plt.savefig(f"./ReconsGraph2/Fig_{run}_{it}.tif")
                plt.close()

