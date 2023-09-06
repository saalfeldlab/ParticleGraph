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

        psi = -self.p[2] * torch.exp(-r ** self.p[0] / (2 * sigma ** 2)) + self.p[3] * torch.exp(-r ** self.p[1] / (2 * sigma ** 2))

        return psi[:, None] * bc_diff(x_i[:, 0:2] - x_j[:, 0:2])

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

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        # self.particle_emb = MLP(input_size=2, hidden_size=8, output_size=8, nlayers=3, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device=self.device, requires_grad=True))
        self.a_bf_kmean = nn.Parameter(torch.tensor(np.ones((int(nparticles), 2)), device=self.device, requires_grad=False))

        self.p0 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))
        self.p1 = nn.Parameter(torch.tensor(np.ones(4), device=self.device, requires_grad=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x[:, 4:6] = self.a[x[:, 6].detach().cpu().numpy(), 0:2]

        if step == 1:
            noise = torch.randn((x.shape[0], 4), requires_grad=False, device='cuda:0') * self.noise_level
            x[:, 0:2] = x[:, 0:2] + noise[:, 0:2]
            x[:, 2:4] = x[:, 2:4] + noise[:, 2:4] / 100

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        if step == 2:
            deg = pyg_utils.degree(data.edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)

            return deg * acc

        else:

            return acc

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
        super().__init__(aggr='mean')  # "mean" aggregation.

    def forward(self, x, edge_index, edge_feature):
        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        return self.new_edges

    def message(self, x_i, x_j, edge_feature):
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
        super().__init__(aggr='mean')  # "mean" aggregation.

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

        self.layer = torch.nn.ModuleList(
            [InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device) for _ in
             range(self.nlayers)])
        self.node_out = MLP(input_size=self.embedding, hidden_size=self.hidden_size, output_size=2, nlayers=3,
                            device=self.device)

        self.embedding_node = MLP(input_size=8, hidden_size=self.embedding, output_size=self.embedding, nlayers=3,
                                  device=self.device)
        self.embedding_edges = MLP(input_size=11, hidden_size=self.embedding, output_size=self.embedding, nlayers=3,
                                   device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((int(nparticles), 1)), device=self.device, requires_grad=True))


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x[:, 4] = self.a[x[:, 6].detach().cpu().numpy(), 0]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        node_feature = torch.cat((x[:, 0:4], x[:, 4:5].repeat(1, 4)), dim=-1)

        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]), requires_grad=False,
                            device='cuda:0') * self.noise_level
        node_feature = node_feature + noise
        edge_feature = self.edge_init(node_feature, edge_index)

        node_feature = self.embedding_node(node_feature)
        edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer[0](node_feature, data.edge_index, edge_feature=edge_feature)

        pred = self.node_out(node_feature)

        return pred


if __name__ == '__main__':

    # version 1.17 230901

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = {'ntry': 612,
                    'input_size': 15,
                    'output_size': 2,
                    'hidden_size': 32,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'radius': 0.1,
                    'datum': '230902_612',
                    'nparticles': 10000,
                    'nparticle_types': 5,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                    'model': 'InteractionParticles'}

    model_config = {'ntry': 620,
                    'input_size': 15,
                    'output_size': 2,
                    'hidden_size': 32,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'radius': 0.075,
                    'datum': '230902_620',
                    'nparticles': 2000,
                    'nparticle_types': 2,
                    'nframes': 200,
                    'sigma': .005,
                    'tau': 0.1,
                    'aggr_type' : 'mean',
                    'particle_embedding': True,
                    'boundary': 'no',  # periodic   'no'  # no boundary condition
                    'model': 'InteractionParticles'}


    gridsearch_list = [2] #, 20, 50, 100, 200]
    data_augmentation = True

    scaler = StandardScaler()

    for gtest in range(1):

            # ntry=612
            # model_config['ntry'] = ntry
            # model_config['datum']='230902_'+str(ntry)

            print('')
            ntry = model_config['ntry']
            print(f'ntry: {ntry}')
            datum = model_config['datum']
            print(f'datum: {datum}')
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

            index_particles = []
            for n in range(nparticle_types):
                index_particles.append(np.arange(int(nparticles / nparticle_types) * n, int(nparticles / nparticle_types) * (n + 1)))

            time.sleep(0.5)

            for step in range(1,3):

                if step == 0:
                    print('')
                    print('Generating data ...')

                    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/ReconsGraph2/*")
                    # for f in files:
                    #     os.remove(f)

                    files = glob.glob(f"{folder}/*")
                    for f in files:
                        os.remove(f)

                    json_ = json.dumps(model_config)
                    f = open(f"{folder}/model_config.json", "w")
                    f.write(json_)
                    f.close()

                    p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
                    model = []
                    psi_output = []
                    rr = torch.tensor(np.linspace(0, 0.015, 100))
                    rr = rr.to(device)
                    for n in range(nparticle_types):
                        model.append(InteractionParticles_0(aggr_type=aggr_type, p=torch.squeeze(p[n]), tau=tau))
                        torch.save({'model_state_dict': model[n].state_dict()},f'graphs_data/graphs_particles_{datum}/model_{n}.pt')
                        psi_output.append(psi(rr, torch.squeeze(p[n])))
                        print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
                    time.sleep(0.5)

                    for run in tqdm(range(gridsearch_list[0] + 1)):

                        X1 = torch.rand(nparticles, 2, device=device)
                        X1t = torch.zeros((nparticles, 2, nframes))  # to store all the intermediate time

                        V1 = torch.zeros((nparticles, 2), device=device)
                        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
                        for n in range(1,nparticle_types):
                            T1 = torch.cat((T1,n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
                        T1 = torch.concatenate((T1[:,None], T1[:,None]), 1)
                        N1 = torch.arange(nparticles, device=device)
                        N1 = N1[:, None]

                        for it in range(nframes):

                            X1t[:, :, it] = X1.clone().detach()  # for later display

                            X1 = bc_pos(X1 + V1)

                            distance = torch.sum(bc_diff(X1[:, None, 0:2] - X1[None, :, 0:2]) ** 2, axis=2)
                            t = torch.Tensor([radius ** 2])  # threshold
                            adj_t = (distance < radius ** 2).float() * 1
                            edge_index = adj_t.nonzero().t().contiguous()

                            x = torch.concatenate(
                                (X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), N1.clone().detach()), 1)
                            torch.save(x, f'graphs_data/graphs_particles_{datum}/x_{run}_{it}.pt')

                            dataset = data.Data(x=x, edge_index=edge_index)

                            y=0
                            with torch.no_grad():
                                for n in range(nparticle_types):
                                    y += model[n](dataset) * (x[:, 4:6] == n)

                            torch.save(y, f'graphs_data/graphs_particles_{datum}/y_{run}_{it}.pt')

                            V1 += y

                            if (run == 0) & (it % 5 == 0):

                                distance2 = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                                adj_t2 = ((distance < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                                edge_index2 = adj_t2.nonzero().t().contiguous()
                                dataset2 = data.Data(x=x, edge_index=edge_index2)

                                fig = plt.figure(figsize=(14, 7*0.95))
                                # plt.ion()
                                ax = fig.add_subplot(1, 2, 2)
                                pos = dict(enumerate(x[:, 0:2].detach().cpu().numpy(), 0))
                                vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                                nx.draw_networkx(vis, pos=pos, node_size=10, linewidths=0, edge_color='b', with_labels=False)
                                plt.xlim([-0.3, 1.3])
                                plt.ylim([-0.3, 1.3])
                                plt.text(-0.25, 1.33, f'Graph    {x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)

                                ax = fig.add_subplot(1, 2, 1)

                                for n in range(nparticle_types):
                                    plt.scatter(X1t[index_particles[n], 0, it],X1t[index_particles[n], 1, it], s=3)
                                ax = plt.gca()
                                ax.axes.xaxis.set_ticklabels([])
                                ax.axes.yaxis.set_ticklabels([])
                                plt.xlim([-0.3, 1.3])
                                plt.ylim([-0.3, 1.3])
                                plt.text(-0.25, 1.38, f'frame: {it}')
                                plt.text(-0.25, 1.33, f'sigma:{sigma} N:{nparticles} nframes:{nframes}')
                                for n in range(nparticle_types):
                                    plt.text(-0.25, 1.25-n*0.05, f'p{n}: {np.round(model[n].p.detach().cpu().numpy(), 4)}', color='k')

                                ax = fig.add_subplot(5, 5, 21)

                                for n in range(nparticle_types):
                                    plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
                                plt.plot(rr.detach().cpu().numpy(), psi_output[0].detach().cpu().numpy() * 0, color=[0, 0, 0],
                                         linewidth=0.5)

                                plt.savefig(f"./ReconsGraph2/Fig_{ntry}_{it}.tif")
                                plt.close()

                if step == 1:

                    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/ReconsGraph/*")
                    # for f in files:
                    #     os.remove(f)

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

                    vnorm = norm_velocity(xx, device)
                    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
                    ynorm = norm_acceleration(yy, device)
                    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

                    for gridsearch in gridsearch_list:

                        if model_config['model'] == 'InteractionParticles':
                            model = InteractionParticles(model_config, device)
                            print(f'Training InteractionParticles')
                            model.a_bf_kmean.requires_grad = False
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
                        net = f"./log/try_{ntry}/models/best_model_with_{gridsearch}_graphs.pt"
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

                        list_loss=[]
                        list_gap=[]

                        for epoch in range(60):

                            if epoch == 30:
                                optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)  # , weight_decay=5e-4)

                            total_loss = 0

                            for N in range(1, (gridsearch-1) * nframes * data_augmentation_loop, stp):

                                run = 1 + np.random.randint(gridsearch - 1)
                                k = np.random.randint(nframes - 1)

                                x = torch.load(f'graphs_data/graphs_particles_{datum}/x_{run}_{k}.pt')

                                if data_augmentation:

                                    phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                                    cos = torch.cos(phi)
                                    sin = torch.sin(phi)

                                    new_x = 0.5 + cos * (x[:, 0]-0.5) + sin * (x[:,1]-0.5)
                                    new_y = 0.5 + -sin * (x[:, 0]-0.5) + cos * (x[:, 1]-0.5)
                                    x[:, 0] = new_x
                                    x[:, 1] = new_y
                                    x[:,0:2] = bc_pos(x[:,0:2])

                                    new_vx = cos * x[:, 2] + sin * x[:, 3]
                                    new_vy = -sin * x[:, 2] + cos * x[:, 3]
                                    x[:, 2] = new_vx
                                    x[:, 3] = new_vy

                                # fig = plt.figure(figsize=(8, 8))
                                # plt.ion()
                                # plt.scatter(x[0:1000, 0].detach().cpu(), x[0:1000, 1].detach().cpu(), s=3)
                                # plt.scatter(x[1000:, 0].detach().cpu(), x[1000:, 1].detach().cpu(), s=3)

                                distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                                adj_t = (distance < radius ** 2).float() * 1
                                t = torch.Tensor([radius ** 2])
                                edges = adj_t.nonzero().t().contiguous()
                                y = torch.load(f'graphs_data/graphs_particles_{datum}/y_{run}_{k}.pt')
                                y[:, 0] = y[:, 0] / ynorm[4]
                                y[:, 1] = y[:, 1] / ynorm[5]

                                if data_augmentation:
                                    new_yx = cos * y[:, 0] + sin * y[:, 1]
                                    new_yy = -sin * y[:, 0] + cos * y[:, 1]
                                    y[:, 0] = new_yx
                                    y[:, 1] = new_yy

                                dataset = data.Data(x=x[:, :], edge_index=edges)

                                optimizer.zero_grad()
                                pred = model(dataset)

                                loss = (pred - y).norm(2)
                                loss.backward()
                                optimizer.step()

                                total_loss += loss.item()

                            model.a.data = torch.clamp(model.a.data,min=-4,max=4)
                            embedding = model.a.detach().cpu().numpy()
                            embedding = scaler.fit_transform(embedding)
                            embedding_particle=[]
                            for n in range(nparticle_types):
                                embedding_particle.append(embedding[index_particles[n],:])
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

                            if (total_loss / N / nparticles < best_loss):
                                best_loss = total_loss / N / nparticles
                                torch.save({'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict()},
                                           os.path.join(log_dir, 'models', f'best_model_with_{gridsearch}_graphs.pt'))
                                print("Epoch {}. Loss: {:.6f} Gap: {:.3f} kl.elbow {:d} saving model  ".format(epoch,total_loss / N / nparticles, gap, kl.elbow))
                            else:
                                print("Epoch {}. Loss: {:.6f} Gap: {:.3f} kl.elbow {:d} ".format(epoch,total_loss / N / nparticles, gap, kl.elbow))

                            if ( ((gap < 100) & (kl.elbow==nparticle_types)) | (epoch > 29)) & (model.a.requires_grad == True):

                                if data_augmentation:
                                    data_augmentation_loop = 200
                                    print(f'data_augmentation_loop: {data_augmentation_loop}')

                                print('training MLP only ...')
                                model.a.requires_grad = False
                                model.a_bf_kmean.data=model.a.data
                                new_a = kmeans.cluster_centers_[kmeans.labels_, :]

                                # model.a.data = torch.tensor(new_a, device=device)

                                embedding = model.a.detach().cpu().numpy()
                                embedding = scaler.fit_transform(embedding)
                                embedding_particle = []
                                for n in range(nparticle_types):
                                    embedding_particle.append(embedding[index_particles[n], :])
                                best_loss = np.inf

                            list_loss.append(total_loss / N / nparticles)
                            list_gap.append(gap)

                            fig = plt.figure(figsize=(15, 5))
                            # plt.ion()
                            ax = fig.add_subplot(1, 3, 1)
                            for n in range(nparticle_types):
                                plt.scatter(embedding_particle[n][:,0], embedding_particle[n][:,1], s=3)
                            plt.xlim([-4.1, 4.1])
                            plt.ylim([-4.1, 4.1])
                            plt.xlabel('Embedding 0',fontsize=12)
                            plt.ylabel('Embedding 1', fontsize=12)
                            plt.text(-3.9, 3.6, f'kmeans.inertia: {np.round(gap, 2)}   kl.elbow: {kl.elbow}', fontsize=10)
                            ax = fig.add_subplot(1, 3, 2)
                            plt.plot(range(1, 11), sse)
                            plt.xticks(range(1, 11))
                            plt.xlabel("Number of Clusters", fontsize=12)
                            plt.ylabel("SSE", fontsize=12)

                            ax = fig.add_subplot(2, 4, 4)
                            plt.plot(list_loss,color='k')
                            plt.xlim([0, 60])
                            plt.ylim([0, 0.02])
                            plt.ylabel('Loss', fontsize=10)

                            ax = fig.add_subplot(2, 4, 8)
                            plt.plot(list_gap,color='k')
                            plt.xlim([0, 50])
                            plt.xlabel('Epoch',fontsize=10)
                            plt.ylabel('Gap', fontsize=10)
                            plt.savefig(f"./ReconsGraph/Fig_{ntry}_{epoch}.tif")
                            plt.close()

                if step == 2:

                    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/ReconsGraph3/*")
                    # for f in files:
                    #     os.remove(f)

                    print('')
                    print('Testing loop ... ')

                    if model_config['model'] == 'InteractionParticles':
                        model = InteractionParticles(model_config, device)
                        model.a_bf_kmean.requires_grad = False
                    if model_config['model'] == 'ResNetGNN':
                        model = ResNetGNN(model_config, device)

                    net = f"./log/try_{ntry}/models/best_model_with_{gridsearch_list[0]}_graphs.pt"
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

                    x = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_0.pt')
                    x00 = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_0.pt')
                    y = torch.load(f'graphs_data/graphs_particles_{datum}/y_0_0.pt')

                    rmserr_list = []

                    for it in tqdm(range(nframes - 1)):

                        x0 = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_{it + 1}.pt')

                        distance = torch.sum(bc_diff(x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
                        t = torch.Tensor([radius ** 2])  # threshold
                        adj_t = (distance < radius ** 2).float() * 1
                        edge_index = adj_t.nonzero().t().contiguous()

                        dataset = data.Data(x=x, edge_index=edge_index)

                        with torch.no_grad():
                            y = model(dataset)  # acceleration estimation

                        # y = torch.clamp(y, min=-2, max=2)

                        y[:, 0] = y[:, 0] * ynorm[4]
                        y[:, 1] = y[:, 1] * ynorm[5]

                        x[:, 2:4] = x[:, 2:4] + y  # speed update

                        x[:, 0:2] = bc_pos(x[:, 0:2] + x[:, 2:4])  # position update

                        stp = 5

                        if (it % stp == 0):

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
                            plt.text(-0.25, 1.38, f'T0 {nparticles} particles', fontsize=30)

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

                            rmserr = torch.mean(torch.sqrt(torch.sum(bc_diff(x[:, 0:2] - x0[:, 0:2]) ** 2, axis=1)))
                            rmserr_list.append(rmserr.item())

                            ax = fig.add_subplot(2, 3, 3)
                            plt.plot(np.arange(0, len(rmserr_list) * stp, stp), rmserr_list, 'k', label='RMSE')
                            plt.ylim([0, 0.1])
                            plt.xlim([0, nframes])
                            plt.tick_params(axis='both', which='major', labelsize=10)
                            plt.xlabel('Frame [a.u]', fontsize="14")
                            plt.ylabel('RMSE [a.u]', fontsize="14")
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
                            adj_t3 = (distance3 < 0.9).float() * 1
                            adj_t3 = adj_t3[:, None]
                            adj_t3 = torch.concatenate((adj_t3, adj_t3), 1)
                            adj_t3 = torch.concatenate((adj_t3, adj_t3), 0)

                            pos = dict(enumerate(np.array((temp1[:, 0:2] * adj_t3).detach().cpu()), 0))
                            dataset = data.Data(x=temp1[:, 0:2] * adj_t3, edge_index=temp4)
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
                            embedding = model.a_bf_kmean.detach().cpu().numpy()
                            embedding = scaler.fit_transform(embedding)
                            embedding_particle = []
                            for n in range(nparticle_types):
                                embedding_particle.append(embedding[index_particles[n], :])
                                plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], s=3)
                            # embedding = model.a.detach().cpu().numpy()
                            # embedding = scaler.fit_transform(embedding)
                            # embedding_particle = []
                            # for n in range(nparticle_types):
                            #     embedding_particle.append(embedding[index_particles[n], :])
                            #     plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], marker='+', s=10,
                            #                 color='k')
                            plt.xlim([-2.1, 2.1])
                            plt.ylim([-2.1, 2.1])
                            plt.xlabel('Embedding 0', fontsize=8)
                            plt.ylabel('Embedding 1', fontsize=8)

                            # ax = fig.add_subplot(8, 10, 14)
                            # plt.plot(rr.detach().cpu().numpy(), np.array(psi0.cpu()), color=c1, linewidth=1)
                            # plt.plot(rr.detach().cpu().numpy(), np.array(psi1.cpu()), color=c2, linewidth=1)
                            # plt.plot(rr.detach().cpu().numpy(), rr.detach().cpu().numpy() * 0, color=[0, 0, 0],linewidth=0.5)

                            plt.savefig(f"./ReconsGraph3/Fig_{ntry}_{it}.tif")
                            plt.close()

                if step == 3:

                    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/ReconsGraph3/*")
                    # for f in files:
                    #     os.remove(f)

                    print('')
                    print('Testing loop ... ')

                    if model_config['model'] == 'InteractionParticles':
                        model = InteractionParticles(model_config, device)
                        model.a_bf_kmean.requires_grad = False
                    if model_config['model'] == 'ResNetGNN':
                        model = ResNetGNN(model_config, device)

                    net = f"./log/try_{ntry}/models/best_model_with_{gridsearch_list[0]}_graphs.pt"
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
                        parameter.requires_grad = False
                        param = parameter.numel()
                        table.add_row([name, param])
                        total_params += param
                    print(table)
                    print(f"Total Trainable Params: {total_params}")

                    x = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_{nframes - 1}.pt')
                    x00 = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_{nframes - 1}.pt')
                    y = torch.load(f'graphs_data/graphs_particles_{datum}/y_0_{nframes - 1}.pt')

                    xx = torch.tensor(np.zeros((int(nparticles), 4)), dtype=torch.float32, device=device, requires_grad=True)
                    xx.data[:,0:4] = x[:,0:4]

                    optimizer = torch.optim.Adam([xx], lr=1E-4)  # , weight_decay=5e-4)

                    rmserr_list = []

                    for it in range(nframes - 1, 0, -1):

                        x0 = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_{it - 1}.pt')

                        xx_target = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_{it}.pt')
                        xx_target[:, 0:2] = xx[:, 0:2]

                        distance = torch.sum(bc_diff(x0[:, None, 0:2] - x0[None, :, 0:2]) ** 2, axis=2)
                        t = torch.Tensor([radius ** 2])  # threshold
                        adj_t = (distance < radius ** 2).float() * 1
                        edge_index = adj_t.nonzero().t().contiguous()

                        for loop in range(6000):

                            optimizer.zero_grad()

                            # distance = torch.sum(bc_diff(xx[:, None, 0:2] - xx[None, :, 0:2]) ** 2, axis=2)
                            # t = torch.Tensor([radius ** 2])  # threshold
                            # adj_t = (distance < radius ** 2).float() * 1
                            # edge_index = adj_t.nonzero().t().contiguous()

                            dataset = data.Data(x=torch.cat((xx,x[:,4:7]),axis=1), edge_index=edge_index)

                            y = model(dataset)  # acceleration estimation

                            y[:, 0] = y[:, 0] * ynorm[4]
                            y[:, 1] = y[:, 1] * ynorm[5]

                            xx_p= 0 * xx
                            xx_p[:, 2:4] = xx[:, 2:4] + y
                            # xx_p[:, 2:4] = torch.clamp(xx_p[:, 2:4], min=-3*torch.std(xx_target[:,2:4]), max=3*torch.std(xx_target[:,2:4]) )# speed update
                            xx_p[:, 0:2] = xx[:, 0:2] + xx_p[:, 2:4]  # position update

                            loss = (bc_pos(xx_p[:,0:2]) - xx_target[:,0:2]).norm(2) + (torch.std(xx_p[:,2:4])-torch.std(xx_target[:,2:4])).norm(2) + (torch.mean(xx_p[:,2:4])-torch.mean(xx_target[:,2:4])).norm(2) + xx_p[:,2:4].norm(2)/10
                            loss.backward()

                            xx.grad = torch.nan_to_num(xx.grad, nan=0.0)

                            optimizer.step()

                        print (f"{it} {loss.item()}")

                        xx_ = bc_pos(xx[:, 0:2])

                        stp = 1

                        if (it % stp == 0):

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
                            plt.text(-0.25, 1.38, 'Distribution at t0 is 1.0x1.0')

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

                            rmserr = torch.mean(torch.sqrt(torch.sum(bc_diff(x[:, 0:2] - x0[:, 0:2]) ** 2, axis=1)))
                            rmserr_list.append(rmserr.item())

                            ax = fig.add_subplot(2, 3, 3)
                            plt.plot(np.arange(0, len(rmserr_list) * stp, stp), rmserr_list, 'k', label='RMSE')
                            plt.ylim([0, 0.1])
                            plt.xlim([0, nframes])
                            plt.tick_params(axis='both', which='major', labelsize=10)
                            plt.xlabel('Frame [a.u]', fontsize="10")
                            plt.ylabel('RMSE [a.u]', fontsize="10")
                            plt.legend(fontsize="10")

                            ax = fig.add_subplot(2, 3, 4)
                            pos = dict(enumerate(np.array(xx_[:, 0:2].detach().cpu()), 0))
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
                                plt.scatter(xx_[index_particles[n], 0].detach().cpu(), xx_[index_particles[n], 1].detach().cpu(), s=3)
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
                            temp1 = torch.cat((xx_[:,0:2], x0[:,0:2]), 0)
                            temp2 = torch.tensor(np.arange(nparticles), device=device)
                            temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
                            temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
                            temp4 = torch.t(temp4)

                            distance3 = torch.sqrt(torch.sum((xx_[:, 0:2] - x0[:, 0:2]) ** 2, 1))
                            adj_t3 = (distance3 < 0.9).float() * 1
                            adj_t3 = adj_t3[:, None]
                            adj_t3 = torch.concatenate((adj_t3, adj_t3), 1)
                            adj_t3 = torch.concatenate((adj_t3, adj_t3), 0)

                            pos = dict(enumerate(np.array((temp1[:, 0:2] * adj_t3).detach().cpu()), 0))
                            dataset = data.Data(x=temp1[:, 0:2] * adj_t3, edge_index=temp4)
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
                            embedding = model.a_bf_kmean.detach().cpu().numpy()
                            embedding = scaler.fit_transform(embedding)
                            embedding_particle = []
                            for n in range(nparticle_types):
                                embedding_particle.append(embedding[index_particles[n], :])
                                plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], s=3)
                            embedding = model.a.detach().cpu().numpy()
                            embedding = scaler.fit_transform(embedding)
                            embedding_particle = []
                            for n in range(nparticle_types):
                                embedding_particle.append(embedding[index_particles[n], :])
                                plt.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], marker='+', s=50,
                                            color='k')
                            plt.xlim([-2.1, 2.1])
                            plt.ylim([-2.1, 2.1])
                            plt.xlabel('Embedding 0', fontsize=8)
                            plt.ylabel('Embedding 1', fontsize=8)

                            # ax = fig.add_subplot(8, 10, 14)
                            # plt.plot(rr.detach().cpu().numpy(), np.array(psi0.cpu()), color=c1, linewidth=1)
                            # plt.plot(rr.detach().cpu().numpy(), np.array(psi1.cpu()), color=c2, linewidth=1)
                            # plt.plot(rr.detach().cpu().numpy(), rr.detach().cpu().numpy() * 0, color=[0, 0, 0],linewidth=0.5)

                            plt.savefig(f"./ReconsGraph3/Fig_{ntry}_{it}.tif")
                            plt.close()
