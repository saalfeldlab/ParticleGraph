
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
        super(InteractionParticles_0, self).__init__(aggr='mean')  # "Add" aggregation.
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
        super(InteractionParticles_1, self).__init__(aggr='mean')  # "Add" aggregation.
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

        super(InteractionParticles, self).__init__(aggr='add')  # "Add" aggregation.

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

    files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/ReconsGraph/*")
    for f in files:
        os.remove(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    nrun= 200

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

    model_config = {'ntry': 519,
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
                    'boundary' : 'no', # periodic   'no'  # no boundary condition
                    'model': 'InteractionParticles'}

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


    folder = f'./graphs_data/graphs_particles_{datum}/'
    os.makedirs(folder, exist_ok=True)


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

    for step in range(1,3):

        if step == 0:
            print('')
            print('Generating data ...')

            files = glob.glob(f"{folder}/*")
            for f in files:
                os.remove(f)

            for run in tqdm(range(nrun)):

                X1 = torch.rand(nparticles,2,device=device)
                X1t = torch.zeros((nparticles,2,nframes)) # to store all the intermediate time

                # p0 = torch.tensor([1.2700, 1.4100, 0.0547, 0.0053])
                # p1 = torch.tensor([1.8200, 1.5200, 0.1640, 0.0500])

                # p0 = torch.tensor([2.6011, 1.852, 0.4127/10, 1.1509/10])
                # p1 = torch.tensor([1.0445, 1.4298, 0.2948/10, 0.2112/10])

                p0 = torch.tensor([1.27, 1.41, 0.82, 0.08])
                p1 = torch.tensor([1.82, 1.72, 0.12, 0.45])
                p0[2:4] *= 2E-1 / 3
                p1[2:4] *= 2E-1

                # p0 = torch.tensor([1+np.random.randn(), 1+np.random.randn(), 0.5+0.5*np.random.randn(), 0.5+0.5*np.random.randn()])
                # p1 = torch.tensor([1+np.random.randn(), 1+np.random.randn(), 0.5+0.5*np.random.randn(), 0.5+0.5*np.random.randn()])


                V1 = torch.zeros((nparticles,2),device=device)
                T1 =  torch.cat( ( torch.zeros(int(nparticles/2), device=device) , torch.ones(int(nparticles/2), device=device) ),0)
                T1=T1[:,None]
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
                    # torch.save(edge_index,f'graphs_data/graphs_particles_{datum}/edge_index_{run}_{it}.pt')

                    x=torch.concatenate((X1.clone().detach(),V1.clone().detach(),T1.clone().detach(),N1.clone().detach()),1)
                    torch.save(x,f'graphs_data/graphs_particles_{datum}/x_{run}_{it}.pt')

                    dataset = data.Data(x=x, edge_index=edge_index)

                    with torch.no_grad():
                        y0 = model0(dataset) * (x[:,4:6]==0)
                        y1 = model1(dataset) * (x[:,4:6]==1)

                    y=y0+y1

                    torch.save(y, f'graphs_data/graphs_particles_{datum}/y_{run}_{it}.pt')

                    V1 += y

                    if (run==0) & (it%5==0):

                        distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                        adj_t2 = ((distance < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                        edge_index2 = adj_t2.nonzero().t().contiguous()
                        dataset2 = data.Data(x=x, edge_index=edge_index2)

                        c1 = np.array([220, 50, 32]) / 255
                        c2 = np.array([0, 114, 178]) / 255
                        fig = plt.figure(figsize=(14, 7))
                        # plt.ion()
                        ax = fig.add_subplot(1,2,2)
                        pos=dict(enumerate(x[:,0:2].detach().cpu().numpy(), 0))
                        vis = to_networkx(dataset2,remove_self_loops=True, to_undirected=True)
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
                        plt.text(-0.25, 1.38, f'frame: {it}')
                        plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                        plt.text(-0.25, 1.25, f'p0: {np.round(np.array(p0.cpu()),4)}', color=c1)
                        plt.text(-0.25, 1.20, f'p1: {np.round(np.array(p1.cpu()),4)}', color=c2)

                        ax = fig.add_subplot(5, 5, 21)
                        plt.plot(np.array(psi0.cpu()), color=c1, linewidth=1)
                        plt.plot(np.array(psi1.cpu()), color=c2, linewidth=1)
                        plt.plot(np.array(0 * np.linspace(0, 1, 100)), color=[0, 0, 0], linewidth=0.5)

                        plt.savefig(f"./ReconsGraph/Fig_{run}_{it}.tif")
                        plt.close()

        if step == 1:

            print('')
            print('Training loop ...')

            l_dir = os.path.join('.', 'log')
            log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
            print('log_dir: {}'.format(log_dir))

            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

            copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

            graph_files = glob.glob(f"graphs_data/graphs_particles_{datum}/x_*")
            NGraphs = int(len(graph_files) / nframes)
            print('Graph files N: ', NGraphs)
            time.sleep(0.5)

            NGraphs = 200

            arr = np.arange(0, NGraphs - 1, 2)
            for run in arr:
                kr = np.arange(0, nframes - 1, 4)
                for k in kr:
                    x = torch.load(f'graphs_data/graphs_particles_{datum}/x_{run}_{k}.pt')
                    y = torch.load(f'graphs_data/graphs_particles_{datum}/y_{run}_{k}.pt')
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

            # print(f'v_x={mvx} +/- {vx}')
            # print(f'v_y={mvy} +/- {vy}')
            # print(f'vx01={vx01} vx99={vx99}')
            # print(f'vy01={vy01} vy99={vy99}')

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

            # print(f'acc_x={max} +/- {ax}')
            # print(f'acc_y={may} +/- {ay}')
            # print(f'ax01={ax01} ax99={ax99}')
            # print(f'ay01={ay01} ay99={ay99}')

            gridsearch_list = [200]  # [20, 5, 50, 100] # [1, 5, 10, 20, 50]

            for gridsearch in gridsearch_list:

                print(f'gridsearch: {gridsearch}')

                if model_config['model'] == 'InteractionParticles':
                    model = InteractionParticles(model_config,device)
                    print(f'Training InteractionParticles')
                if model_config['model'] == 'ResNetGNN':
                    model = ResNetGNN(model_config,device)
                    print(f'Training ResNetGNN')

                best_loss = np.inf

                if gridsearch == 0:
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

                optimizer = torch.optim.Adam(model.parameters(), lr=1E-3) #, weight_decay=5e-4)

                model.train()

                stp=1

                for epoch in range(100):

                    model.train()
                    total_loss = 0

                    for N in range(1, gridsearch * nframes,stp):

                        run = 1 + np.random.randint(gridsearch-1)
                        k = np.random.randint(nframes - 1)

                        x = torch.load(f'graphs_data/graphs_particles_{datum}/x_{run}_{k}.pt')
                        distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                        adj_t = (distance < radius ** 2).float() * 1
                        t = torch.Tensor([radius ** 2])
                        edges = adj_t.nonzero().t().contiguous()
                        y = torch.load(f'graphs_data/graphs_particles_{datum}/y_{run}_{k}.pt')
                        y[:, 0] = y[:, 0] / ynorm[4]
                        y[:, 1] = y[:, 1] / ynorm[5]

                        dataset = data.Data(x=x[:, :], edge_index=edges)

                        optimizer.zero_grad()
                        pred = model(dataset)

                        loss = (pred - y).norm(2)

                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                        if N % 2000 == 0:
                            print("N {} Loss: {:.6f}".format(N, total_loss / N * stp / nparticles))

                    embedding = model.a[:,0].detach().cpu().numpy()
                    embedding0 = embedding[0:int(nparticles / 2)]
                    embedding1 = embedding[int(nparticles / 2):nparticles]
                    gap=np.abs(np.mean(embedding0) - np.mean(embedding1)) / (np.std(embedding0) + np.std(embedding1))

                    if (gap>20) & (model.a.requires_grad==True):
                        print('model.a.requires_grad=False')
                        model.a.requires_grad=False
                        model.a.data = torch.trunc(model.a.data*5)/5


                    if (total_loss < best_loss):
                        best_loss = total_loss
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()},
                                   os.path.join(log_dir, 'models', f'best_model_with_{gridsearch}_graphs.pt'))
                        print("Epoch {}. Loss: {:.6f} Gap: {:.3f}  saving model  ".format(epoch, total_loss / N / nparticles, gap))
                        fig = plt.figure(figsize=(25, 16))
                        # plt.ion()
                        plt.plot(model.a[:,0].detach().cpu().numpy(), model.a[:,1].detach().cpu().numpy(), '.', color='k')
                        plt.savefig(f"./ReconsGraph/Fig_{epoch}.tif")
                        plt.close()
                    else:
                        print("Epoch {}. Loss: {:.6f} Gap: {:.3f} ".format(epoch, total_loss / N / nparticles,gap))

        if step == 2:

            files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/ReconsGraph/*")
            for f in files:
                os.remove(f)

            print('')
            print('Testing loop ... ')

            if model_config['model'] == 'InteractionParticles':
                model = InteractionParticles(model_config, device)
            if model_config['model'] == 'ResNetGNN':
                model = ResNetGNN(model_config, device)

            net=f"./log/try_{ntry}/models/best_model_with_200_graphs.pt"
            print(f'network: {net}')
            state_dict = torch.load(net)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt')
            vnorm = torch.load(f'./log/try_{ntry}/vnorm.pt')

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

            # embedding = model.a.detach().cpu().numpy()
            # embedding0 = embedding[0:int(nparticles / 2)]
            # embedding1 = embedding[int(nparticles / 2):nparticles]
            # gap = np.abs(np.mean(embedding0) - np.mean(embedding1)) / (np.std(embedding0) + np.std(embedding1))
            #
            # print(f'gap: {np.round(gap,2)}')

            # plt.ion()
            # plt.hist(model.a.detach().cpu().numpy(),100)
            fig = plt.figure(figsize=(10, 10))
            plt.plot(model.a.detach().cpu().numpy(), '.', color='k')
            plt.show()

            x = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_0.pt')
            x00 = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_0.pt')
            y = torch.load(f'graphs_data/graphs_particles_{datum}/y_0_0.pt')

            rmserr_list = []
            rmserr_list0 = []
            rmserr_list1 = []

            c1 = np.array([220, 50, 32]) / 255
            c2 = np.array([0, 114, 178]) / 255

            for it in tqdm(range(nframes - 1)):

                x0 = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_{it + 1}.pt')

                distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                t = torch.Tensor([radius ** 2])  # threshold
                adj_t = (distance < radius ** 2).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()

                distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                adj_t2 = ((distance < radius ** 2)&(distance2<0.9**2)).float() * 1
                edge_index2 = adj_t2.nonzero().t().contiguous()

                dataset = data.Data(x=x, edge_index=edge_index)
                dataset2 = data.Data(x=x, edge_index=edge_index2)

                with torch.no_grad():
                    y = model(dataset)  # acceleration estimation

                y[:, 0] = y[:, 0] * ynorm[4]
                y[:, 1] = y[:, 1] * ynorm[5]

                x[:, 2:4] = x[:, 2:4] + y  # speed update

                if model_config['boundary']=='per':
                    x[:, 2:4] = x[:, 2:4] - torch.mean(x[:, 2:4])

                x[:, 0:2] = bc_pos(x[:, 0:2] + x[:, 2:4])  # position update

                stp = 1

                if (it % stp == 0):

                    fig = plt.figure(figsize=(25, 16))

                    ax = fig.add_subplot(2, 3, 1)
                    plt.scatter(x00[0:1000, 0].detach().cpu(), x00[0:1000, 1].detach().cpu(), s=3, color=c1)
                    plt.scatter(x00[1000:, 0].detach().cpu(), x00[1000:, 1].detach().cpu(), s=3, color=c2)
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.38, 'Distribution at t0 is 1.0x1.0')

                    ax = fig.add_subplot(2, 3, 2)
                    plt.scatter(x0[0:1000, 0].detach().cpu(), x0[0:1000, 1].detach().cpu(), s=3, color=c1)
                    plt.scatter(x0[1000:, 0].detach().cpu(), x0[1000:, 1].detach().cpu(), s=3, color=c2)
                    ax = plt.gca()
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.38, 'True', fontsize=30)
                    # plt.text(-0.25, 1.38, f'Frame: {min(nframes,it)}')
                    # plt.text(-0.25, 1.33, f'Physics simulation', fontsize=10)

                    rmserr = torch.mean(torch.sqrt(torch.sum(bc_diff(x[:,0:2] - x0[:,0:2]) ** 2, axis=1)))
                    rmserr_list.append(rmserr.item())
                    rmserr0 = torch.mean(
                        torch.sqrt(torch.sum(bc_diff(x[0:int(nparticles / 2), 0:2] - x0[0:int(nparticles / 2), 0:2]) ** 2, axis=1)))
                    rmserr_list0.append(rmserr0.item())
                    rmserr1 = torch.mean(torch.sqrt(
                        torch.sum(bc_diff(x[int(nparticles / 2):nparticles, 0:2] - x0[int(nparticles / 2):nparticles, 0:2]) ** 2,
                                  axis=1)))
                    rmserr_list1.append(rmserr1.item())

                    ax = fig.add_subplot(2, 3, 3)
                    plt.plot(np.arange(0,len(rmserr_list)*stp,stp),rmserr_list, 'k', label='RMSE')
                    plt.plot(np.arange(0,len(rmserr_list)*stp,stp),rmserr_list0, color=c1, label='RMSE0')
                    plt.plot(np.arange(0,len(rmserr_list)*stp,stp), rmserr_list1, color=c2, label='RMSE1')
                    plt.ylim([0, 0.1])
                    plt.xlim([0, nframes])
                    plt.tick_params(axis='both', which='major', labelsize=10)
                    plt.xlabel('Frame [a.u]', fontsize="10")
                    plt.ylabel('RMSE [a.u]', fontsize="10")
                    plt.legend(fontsize="10")

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
                    plt.scatter(x[0:1000, 0].detach().cpu(), x[0:1000, 1].detach().cpu(), s=3, color=c1)
                    plt.scatter(x[1000:, 0].detach().cpu(), x[1000:, 1].detach().cpu(), s=3, color=c2)
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.38, 'Model', fontsize=30)
                    # plt.text(-0.25, 1.38, f'Frame: {it}')
                    # plt.text(-0.25, 1.33, f'GNN prediction', fontsize=10)

                    ax = fig.add_subplot(2, 3, 6)
                    temp1 = torch.cat((x, x0), 0)
                    temp2 = torch.tensor(np.arange(nparticles), device=device)
                    temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
                    temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
                    temp4 = torch.t(temp4)

                    distance3 = torch.sqrt(torch.sum((x[:, 0:2] - x0[:, 0:2]) ** 2,1))
                    adj_t3 = (distance3 < 0.9).float() * 1
                    adj_t3=adj_t3[:,None]
                    adj_t3 = torch.concatenate((adj_t3, adj_t3), 1)
                    adj_t3 = torch.concatenate((adj_t3, adj_t3), 0)

                    pos = dict(enumerate(np.array((temp1[:, 0:2]*adj_t3).detach().cpu()), 0))

                    dataset = data.Data(x=temp1[:,0:2]*adj_t3, edge_index=temp4)
                    vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False)
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.18, f'Frame: {it}')
                    plt.text(-0.25, 1.13, 'Prediction RMSE: {:.4f}'.format(rmserr.detach()), fontsize=10)

                    plt.savefig(f"./ReconsGraph/Fig_{it}.tif")
                    plt.close()