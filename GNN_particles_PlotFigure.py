import glob
import json
import logging
import time
from decimal import Decimal
from math import *
from shutil import copyfile

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmplt
import networkx as nx
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.data as data
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import umap
import yaml  # need to install pyyaml
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
from GNN_particles_Ntype import *


class InteractionParticles_extract(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device, aggr_type=[], bc_diff=[]):

        super(InteractionParticles_extract, self).__init__(aggr=aggr_type)  # "Add" aggregation.

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
        self.bc_diff = bc_diff

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

        return pred, self.in_features, self.lin_edge_out

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) / self.radius
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

        self.in_features = in_features
        self.lin_edge_out = out

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

class PDE_B_extract(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], delta_t=[], bc_diff=[]):
        super(PDE_B_extract, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.delta_t = delta_t
        self.bc_diff = bc_diff

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        sum = self.cohesion + self.alignment + self.separation

        return acc, sum, self.cohesion, self.alignment, self.separation, self.diffx, self.diffv, self.r, self.type

    def message(self, x_i, x_j):

        r = torch.sum(self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)  # distance squared

        pp = self.p[to_numpy(x_i[:, 5]), :]

        cohesion = pp[:, 0:1].repeat(1, 2) * 0.5E-5 * self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3])
        alignment = pp[:, 1:2].repeat(1, 2) * 5E-4 * self.bc_diff(x_j[:, 3:5] - x_i[:, 3:5])
        separation = pp[:, 2:3].repeat(1, 2) * 1E-8 * self.bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / (r[:, None].repeat(1, 2))

        self.cohesion = cohesion
        self.alignment = alignment
        self.separation = separation

        self.r = r
        self.diffx = self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3])
        self.diffv = self.bc_diff(x_j[:, 3:5] - x_i[:, 3:5])
        self .type = x_i[:, 5]

        return (separation + alignment + cohesion)

    def psi(self, r, p):
        cohesion = p[0] * 0.5E-5 * r
        separation = -p[2] * 1E-8 / r
        return (cohesion + separation)  # 5E-4 alignement


def func_pow(x, a, b):
    return a / (x**b)

def func_lin(x, a, b):
    return a * x + b

def func_boids(x, a, b, c):

    xdiff = x[:, 0:2]
    vdiff = x[:, 2:4]
    r = np.concatenate((x[:,4:5],x[:,4:5]),axis=1)

    sum = a * xdiff + b * vdiff - c * xdiff / r
    sum = np.sqrt(sum[:,0]**2 + sum[:,1]**2)

    return sum


def data_plot_FIG2():

    config = 'config_arbitrary_3'

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
    aggr_type = model_config['aggr_type']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    print('Load normalizations ...')
    time.sleep(1)

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

    model = InteractionParticles(model_config=model_config, device=device, aggr_type = model_config['aggr_type'], bc_diff=bc_diff)
    print(f'Training InteractionParticles')

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
    model.eval()

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
                     to_numpy(acc) * to_numpy(ynorm[4]),
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
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(F1), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))

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
    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                        embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=6)
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
                     to_numpy(acc) * to_numpy(ynorm[4]),
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
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(Accuracy), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 4, 9)
    print('9')
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=6)
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
                     to_numpy(acc) * to_numpy(ynorm[4]),
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
    plt.savefig('Fig2.jpg', dpi=300)
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
        plot_list.append(pred * ynorm[4])

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

def data_plot_FIG2sup():

    bPrint=True

    config = 'config_arbitrary_3'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    def bc_pos(X):
        return torch.remainder(X, 1.0)
    def bc_diff(D):
        return torch.remainder(D - .5, 1.0) - .5
    aggr_type = 'mean'

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

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    delta_t = model_config['delta_t']
    aggr_type = model_config['aggr_type']

    best_model = 20

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    if os.path.isfile(os.path.join(log_dir, f'labels_20.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_20.pt'))
    else:
        labels = T1
        print('Use ground truth labels')

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)


    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    cm = 1 / 2.54 * 3 / 2.3
    fig = plt.figure(figsize=(18, 31))
    plt.ion()

    #################### first set of plots

    ratio = 1
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, device=device)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    model = InteractionParticles(model_config=model_config, device=device, aggr_type=model_config['aggr_type'], bc_diff=bc_diff)


    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5


    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 1+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 6+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()


    plt.tight_layout()
    plt.savefig('Fig2_supp.pdf', format="pdf", dpi=300)
    torch.cuda.empty_cache()

    #################### second set of plots

    config = 'config_arbitrary_3'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    delta_t = model_config['delta_t']

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    if os.path.isfile(os.path.join(log_dir, f'labels_20.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_20.pt'))
    else:
        labels = T1
        print('Use ground truth labels')

    model_config['nframes'] = 500
    nframes = 500

    ratio = 1

    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4,ratio = ratio, device=device)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    model = InteractionParticles(model_config=model_config, device=device, aggr_type=model_config['aggr_type'], bc_diff=bc_diff)

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5

    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 11+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 16+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()

    plt.tight_layout()
    plt.savefig('Fig2_supp.pdf', format="pdf", dpi=300)
    torch.cuda.empty_cache()

    #################### third set of plots

    model_config['nframes'] = 250
    nframes = 250
    ratio = 2
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, device=device)
    model = InteractionParticles(model_config=model_config, device=device, aggr_type=model_config['aggr_type'], bc_diff=bc_diff)


    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    if ratio > 1:  # nparticles larger than initially

        prev_index_particles = index_particles

        new_nparticles = nparticles * ratio
        prev_nparticles = nparticles

        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio}')
        print('')

        embedding = model.a[0].data.clone().detach()
        new_embedding = []
        new_labels = []

        for n in range(nparticle_types):
            for m in range(ratio):
                if (n == 0) & (m == 0):
                    new_embedding = embedding[prev_index_particles[n].astype(int), :]
                    new_labels = labels[prev_index_particles[n].astype(int)]
                else:
                    new_embedding = torch.cat((new_embedding, embedding[prev_index_particles[n].astype(int), :]),
                                              axis=0)
                    new_labels = torch.cat((new_labels, labels[prev_index_particles[n].astype(int)]), axis=0)

        model.a = nn.Parameter(
            torch.tensor(np.ones((NGraphs - 1, int(prev_nparticles) * ratio, 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        model.a.requires_grad = False
        model.a[0] = new_embedding
        labels = new_labels
        nparticles = new_nparticles
        model_config['nparticles'] = new_nparticles

        index_particles = []
        np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
        for n in range(model_config['nparticle_types']):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5

    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 21+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 26+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()

    plt.tight_layout()
    plt.savefig('Fig2_supp.pdf', format="pdf", dpi=300)
    torch.cuda.empty_cache()

    #################### fourth set of plots

    model_config['nframes'] = 500
    model_config['nparticles'] = 4800
    nframes = 500
    ratio = 2
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, scenario = 'scenario A', device=device)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5

    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 31+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 36+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig('Fig2_supp.pdf', format="pdf", dpi=300)
    plt.savefig('Fig2_supp.jpg', dpi=300)
    plt.close()

    print(f'RMSE: {np.round(rmserr.item(), 4)}')
    if bPrint:
        print(f'dataset_name: {dataset_name}')

    model_config['nframes'] = 250
    nframes = 250
    model_config['nparticles'] = 4800
    ratio = 1
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, scenario = 'scenario A', device=device)

def data_plot_FIG3sup():

    config = 'config_arbitrary_16_HR1'
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
    aggr_type = model_config['aggr_type']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    print('Load normalizations ...')
    time.sleep(1)

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


    model = InteractionParticles(model_config=model_config, device=device, aggr_type = model_config['aggr_type'], bc_diff=bc_diff)
    print(f'Training InteractionParticles')

    # if best_model == -1:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    # else:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

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

    fig = plt.figure(figsize=(13, 9.6))
    plt.ion()
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
                     to_numpy(acc) * to_numpy(ynorm[4]),
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
    plt.ylim([-0.1,0.1])

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
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(F1), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))

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
                     to_numpy(acc) * to_numpy(ynorm[4]),
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
    plt.ylim([-0.1,0.1])

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
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(F1), fontsize=12)
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
                     to_numpy(acc) * to_numpy(ynorm[4]),
                     color=cmap.color(to_numpy(x[n, 5])), linewidth=1)
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0,0.075,r'Model', fontsize=14)
    plt.ylim([-0.1,0.1])

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
    plt.ylim([-0.1,0.1])
    plt.text(0,0.075,r'True', fontsize=14)


    plt.tight_layout()
    plt.savefig('Fig3_supp.pdf', format="pdf", dpi=300)
    plt.savefig('Fig3_supp.jpg', dpi=300)
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
        plot_list.append(pred * ynorm[4])

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

def data_plot_FIG4sup():

    bPrint=True

    config = 'config_arbitrary_16_HR1'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    def bc_pos(X):
        return torch.remainder(X, 1.0)
    def bc_diff(D):
        return torch.remainder(D - .5, 1.0) - .5
    aggr_type = 'mean'

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

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    delta_t = model_config['delta_t']
    aggr_type = model_config['aggr_type']

    best_model = 20

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    if os.path.isfile(os.path.join(log_dir, f'labels_20.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_20.pt'))
    else:
        labels = T1
        print('Use ground truth labels')

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)


    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    cm = 1 / 2.54 * 3 / 2.3
    fig = plt.figure(figsize=(18, 31))
    plt.ion()

    #################### first set of plots

    ratio = 1
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, device=device)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    model = InteractionParticles(model_config=model_config, device=device, aggr_type=model_config['aggr_type'], bc_diff=bc_diff)


    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5


    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 1+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 6+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()


    plt.tight_layout()
    plt.savefig('Fig4_supp.pdf', format="pdf", dpi=300)
    torch.cuda.empty_cache()

    #################### second set of plots

    config = 'config_arbitrary_16_HR1'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    delta_t = model_config['delta_t']

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    if os.path.isfile(os.path.join(log_dir, f'labels_20.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_20.pt'))
    else:
        labels = T1
        print('Use ground truth labels')

    model_config['nframes'] = 1000
    nframes = 1000

    ratio = 1

    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4,ratio = ratio, device=device)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    model = InteractionParticles(model_config=model_config, device=device, aggr_type=model_config['aggr_type'], bc_diff=bc_diff)

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5

    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 11+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 16+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()

    plt.tight_layout()
    plt.savefig('Fig4_supp.pdf', format="pdf", dpi=300)
    torch.cuda.empty_cache()

    #################### third set of plots

    model_config['nframes'] = 500
    nframes = 500
    ratio = 2
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, device=device)
    model = InteractionParticles(model_config=model_config, device=device, aggr_type=model_config['aggr_type'], bc_diff=bc_diff)


    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    if ratio > 1:  # nparticles larger than initially

        prev_index_particles = index_particles

        new_nparticles = nparticles * ratio
        prev_nparticles = nparticles

        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio}')
        print('')

        embedding = model.a[0].data.clone().detach()
        new_embedding = []
        new_labels = []

        for n in range(nparticle_types):
            for m in range(ratio):
                if (n == 0) & (m == 0):
                    new_embedding = embedding[prev_index_particles[n].astype(int), :]
                    new_labels = labels[prev_index_particles[n].astype(int)]
                else:
                    new_embedding = torch.cat((new_embedding, embedding[prev_index_particles[n].astype(int), :]),
                                              axis=0)
                    new_labels = torch.cat((new_labels, labels[prev_index_particles[n].astype(int)]), axis=0)

        model.a = nn.Parameter(
            torch.tensor(np.ones((NGraphs - 1, int(prev_nparticles) * ratio, 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        model.a.requires_grad = False
        model.a[0] = new_embedding
        labels = new_labels
        nparticles = new_nparticles
        model_config['nparticles'] = new_nparticles

        index_particles = []
        np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
        for n in range(model_config['nparticle_types']):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5

    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 21+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 26+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()

    plt.tight_layout()
    plt.savefig('Fig4_supp.pdf', format="pdf", dpi=300)
    torch.cuda.empty_cache()

    #################### fourth set of plots

    model_config['nframes'] = 1000
    model_config['nparticles'] = 4800
    nframes = 1000
    ratio = 2
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, scenario = 'scenario A', device=device)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5

    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            ax = fig.add_subplot(8, 5, 31+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            ax = fig.add_subplot(8, 5, 36+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig('Fig4_supp.pdf', format="pdf", dpi=300)
    plt.savefig('Fig4_supp.jpg', dpi=300)
    plt.close()

    print(f'RMSE: {np.round(rmserr.item(), 4)}')
    if bPrint:
        print(f'dataset_name: {dataset_name}')

    model_config['nframes'] = 500
    nframes = 500
    model_config['nparticles'] = 4800
    ratio = 1
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4, ratio = ratio, scenario = 'scenario A', device=device)

def data_plot_FIG3():


    config = 'config_gravity_16'

    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    def bc_pos(X):
        return X
    def bc_diff(D):
        return D

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
    aggr_type = model_config['aggr_type']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    print('Load normalizations ...')
    time.sleep(1)

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

    model = GravityParticles(model_config=model_config, device=device, bc_diff=bc_diff)

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
    acc_list = torch.stack(acc_list)
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
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))
    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    t = []
    tt = []
    for k in range(model_config['ninteractions']):
        pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        print(torch.median(temp, axis=0).values)
        model_a_[pos, :] = torch.median(temp, axis=0).values.repeat((len(pos), 1))
        t.append(torch.median(temp, axis=0).values)
        tt = np.append(tt, torch.median(temp, axis=0).values.cpu().numpy())
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
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(Accuracy), fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))

    ax = fig.add_subplot(3, 3, 4)
    print('4')
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=5)
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
                 to_numpy(acc) * to_numpy(ynorm[4]),
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
        p = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt',map_location=device)
    psi_output = []
    for n in range(nparticle_types):
        psi_output.append(model.psi(rr, p[n]))
    for n in range(nparticle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), np.array(psi_output[n].cpu()), linewidth=1, color=cmap.color(n))

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
        plot_list.append(pred * ynorm[4])
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
        plot_list.append(pred * ynorm[4])
    p = np.linspace(0.5, 5, nparticle_types)
    popt_list = []
    for n in range(nparticle_types):
        popt, pcov = curve_fit(func_pow, to_numpy(rr), to_numpy(plot_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    ax = fig.add_subplot(3, 3, 7)
    print('7')
    x_data = p
    y_data = popt_list[:, 0]
    lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
    plt.plot(p, func_lin(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    plt.scatter(p, popt_list[:, 0], color='k')

    plt.xlabel(r'True mass $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted mass $[a.u.]$', fontsize=14)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.text(0.5, 5, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - func_lin(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(0.5, 4.5, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)

    ax = fig.add_subplot(3, 3, 8)
    print('8')
    plt.scatter(p, -popt_list[:, 1], color='k')
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.xlabel(r'True mass $[a.u.]$', fontsize=14)
    plt.ylabel(r'Exponent fit $[a.u.]$', fontsize=14)
    plt.text(0.5, -0.5, f"{np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)

    plot_list_2 = []
    vv = torch.tensor(np.linspace(0, 2, 100)).to(device)
    r_list = np.linspace(0.002, 0.01, 5)
    for r_ in r_list:
        rr_ = r_ * torch.tensor(np.ones((vv.shape[0], 1)), device=device)
        for n in range(nparticle_types):
            embedding = t[int(label_list[n])] * torch.ones((100, model_config['embedding']), device=device)
            in_features = torch.cat((rr_ / model_config['radius'], 0 * rr_,
                                     rr_ / model_config['radius'], vv[:, None], vv[:, None], vv[:, None], vv[:, None],
                                     embedding), dim=1)
            with torch.no_grad():
                pred = model.lin_edge(in_features.float())
            pred = pred[:, 0]
            plot_list_2.append(pred * ynorm[4])

    ax = fig.add_subplot(3, 3, 9)
    print('9')

    t = torch.tensor(tt, device=device)
    t = torch.reshape(t, (16, 2))
    rr = torch.tensor(np.linspace(0.002, 0.005, 100)).to(device)
    vv = torch.tensor(np.linspace(-1, 1, 100)).to(device)

    tt = torch.zeros((1, 2), device=device)
    tt[0,0] = torch.mean(t[:,0])
    tt[0,1] = torch.mean(t[:,1])
    tt = tt * torch.ones((10000, model_config['embedding']), device=device)

    rr_, vv_ = torch.meshgrid(rr, vv, indexing='xy')
    rr_ = torch.reshape(rr_, (rr_.shape[0] * rr_.shape[1], 1))
    vv_ = torch.reshape(vv_, (vv_.shape[0] * vv_.shape[1], 1))

    in_features = torch.cat((rr_ / model_config['radius'], 0 * rr_, rr_ / model_config['radius'], vv_, vv_, vv_, vv_, tt), dim=1)
    with torch.no_grad():
        pred = model.lin_edge(in_features.float())
    pred = pred[:, 0]
    pred=torch.reshape(pred,(100,100))
    plt.imshow(to_numpy(pred), extent=[0.002, 0.01, -1, 1], aspect='auto', origin='lower', cmap='Blues')
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'Normalized $\dot{x}_{i} [a.u.]$', fontsize=14)
    plt.title(r'$f(\ensuremath{\bar{\mathbf{a}}}_j,\dot{x}_{i}, r_{ij}) [a.u.]$', fontsize=14)

    plt.tight_layout()

    plt.savefig('Fig3.pdf', format="pdf", dpi=300)
    plt.savefig('Fig3.jpg', dpi=300)

    plt.close()

def data_plot_FIG3_continous():


    config = 'config_gravity_16_HR_continuous'

    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    def bc_pos(X):
        return X
    def bc_diff(D):
        return D

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
    aggr_type = model_config['aggr_type']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    print('Load normalizations ...')
    time.sleep(1)

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

    model = GravityParticles(model_config=model_config, device=device, bc_diff=bc_diff)

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
    embedding_ = torch.tensor(embedding, device=device)

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

    colors = cmplt.jet(np.linspace(0, 1, nparticles))

    fig = plt.figure(figsize=(6.5, 8.2))
    plt.ion()
    ax = fig.add_subplot(3, 2, 1)
    colors = cmplt.rainbow(np.linspace(0, 1, nparticles))
    print('1')

    plt.scatter(embedding[:, 0],embedding[:, 1], s=0.1, c=colors, alpha=0.5)
    plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$',fontsize=14)
    plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

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
    acc_list = torch.stack(acc_list)
    coeff_norm = to_numpy(acc_list)
    trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                      random_state=42, transform_queue_size=0).fit(coeff_norm)
    proj_interaction = trans.transform(coeff_norm)
    proj_interaction = np.squeeze(proj_interaction)

    ax = fig.add_subplot(3, 2, 2)
    print('2')
    plt.scatter(proj_interaction[:, 0], proj_interaction[:, 1], s=0.1, c=colors, alpha=0.5)
    plt.xlabel(r'UMAP 0', fontsize=14)
    plt.ylabel(r'UMAP 1', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 2, 3)
    print('3')
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
                 to_numpy(acc) * to_numpy(ynorm[4]),
                 color=colors[n], linewidth=1, alpha=0.25)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_j, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0.0075,0.4E6,r'Model', fontsize=14)

    ax = fig.add_subplot(3,2,4)
    print('6')
    p = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p.pt',map_location=device)
    psi_output = []
    for n in range(nparticles):
        psi_output.append(model.psi(rr, p[n]))
        plt.plot(to_numpy(rr), np.array(psi_output[n].cpu()), linewidth=1, color=colors[n])
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_j, r_{ij}) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0.0075,0.4E6,r'True', fontsize=14)


    ax = fig.add_subplot(3, 2, 5)
    print('5')
    plot_list = []
    for n in range(nparticles):
        embedding = embedding_[n] * torch.ones((1000, model_config['embedding']), device=device)
        in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        with torch.no_grad():
            pred = model.lin_edge(in_features.float())
        pred = pred[:, 0]
        plot_list.append(pred * ynorm[4])
    p = np.linspace(0.5, 5, nparticles)
    popt_list = []
    for n in range(nparticles):
        popt, pcov = curve_fit(func_pow, to_numpy(rr), to_numpy(plot_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    x_data = p
    y_data = np.clip(popt_list[:, 0],0, 5)
    lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
    plt.scatter(p, popt_list[:, 0], color=colors, s=1)
    plt.xlabel(r'True mass $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted mass $[a.u.]$', fontsize=14)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.text(0.5, 4.5, f"N: {nparticles}", fontsize=10)
    plt.text(0.5, 4.0, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - func_lin(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(0.5, 3.5, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)

    ax = fig.add_subplot(3, 2, 6)
    print('8')
    plt.scatter(p, -popt_list[:, 1], color='k', s=1)
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.xlabel(r'True mass $[a.u.]$', fontsize=14)
    plt.ylabel(r'Exponent fit $[a.u.]$', fontsize=14)
    plt.text(0.5, -0.5, f"{np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)

    plt.tight_layout()

    plt.savefig('Fig3_continous.pdf', format="pdf", dpi=300)
    plt.savefig('Fig3_continuous.jpg', dpi=300)

    plt.close()

def data_plot_FIG4():

    config = 'config_Coulomb_3'

    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    def bc_pos(X):
        return torch.remainder(X, 1.0)
    def bc_diff(D):
        return torch.remainder(D - .5, 1.0) - .5

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
    aggr_type = model_config['aggr_type']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    print('Load normalizations ...')
    time.sleep(1)

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

    model = ElecParticles(model_config=model_config, device=device,bc_diff = bc_diff)

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
                #              to_numpy(acc) * to_numpy(ynorm[4]),
                #              linewidth=1,
                #              color=cmap.color(k), alpha=0.25)
    acc_list = torch.stack(acc_list)
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
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))
    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    t = []
    tt= []
    for k in range(model_config['ninteractions']):
        pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        print(torch.median(temp, axis=0).values)
        model_a_[pos, :] = torch.median(temp, axis=0).values.repeat((len(pos), 1))
        t.append(torch.median(temp, axis=0).values)
        tt = np.append(tt, torch.median(temp, axis=0).values.cpu().numpy())
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
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(Accuracy), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))

    ax = fig.add_subplot(3, 3, 4)
    print('4')
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=5)
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
                     to_numpy(acc) * to_numpy(ynorm[4]),
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
            plot_list_pairwise.append(pred * ynorm[4])
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

    ax = fig.add_subplot(3, 3, 7)

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

    ax = fig.add_subplot(3,3,8)
    plt.scatter(ptrue_list, popt_list[:, 1], color='k')
    plt.ylim([-4, 0])
    plt.xlabel(r'True $q_i q_j [a.u.]$', fontsize=14)
    plt.ylabel(r'Exponent fit $[a.u.]$', fontsize=14)
    plt.text(-2, -0.5, f"{np.round(-np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)

    ax = fig.add_subplot(3, 3, 9)
    print('9')

    t = torch.tensor(tt, device=device)
    t = torch.reshape(t, (3, 2))
    rr = torch.tensor(np.linspace(0.002, 0.005, 100)).to(device)
    vv = torch.tensor(np.linspace(-1, 1, 100)).to(device)

    tt = torch.zeros((1, 2), device=device)
    tt[0,0] = torch.mean(t[:,0])
    tt[0,1] = torch.mean(t[:,1])
    tt = tt * torch.ones((10000, model_config['embedding']), device=device)

    rr_, vv_ = torch.meshgrid(rr, vv, indexing='xy')
    rr_ = torch.reshape(rr_, (rr_.shape[0] * rr_.shape[1], 1))
    vv_ = torch.reshape(vv_, (vv_.shape[0] * vv_.shape[1], 1))

    in_features = torch.cat((rr_ / model_config['radius'], 0 * rr_, rr_ / model_config['radius'], vv_, vv_, vv_, vv_, tt, tt), dim=1)
    with torch.no_grad():
        pred = model.lin_edge(in_features.float())
    pred = pred[:, 0]
    pred=torch.reshape(pred,(100,100))
    plt.imshow(to_numpy(pred), extent=[0.002, 0.01, -1, 1], aspect='auto', origin='lower', cmap='Blues')
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'Normalized $\dot{x}_{i} [a.u.]$', fontsize=14)
    plt.title(r'$f(\ensuremath{\bar{\mathbf{a}}}_i,\ensuremath{\bar{\mathbf{a}}}_j,\dot{x}_{i}, r_{ij}) [a.u.]$', fontsize=14)

    plt.tight_layout()

    plt.savefig('Fig4.pdf', format="pdf", dpi=300)
    plt.savefig('Fig4.jpg', dpi=300)
    plt.close()

def data_plot_FIG5sup():

    bPrint=True

    config = 'config_gravity_16'

    #################### first set of plots

    def bc_pos(X):
        return X
    def bc_diff(D):
        return D


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

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    delta_t = model_config['delta_t']
    aggr_type = model_config['aggr_type']

    best_model = 20

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    if os.path.isfile(os.path.join(log_dir, f'labels_20.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_20.pt'))
    else:
        labels = T1
        print('Use ground truth labels')

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)


    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    cm = 1 / 2.54 * 3 / 2.3
    fig = plt.figure(figsize=(18, 33))
    plt.ion()


    ratio = 1
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4,ratio = ratio, bc_diff=bc_diff, bc_pos=bc_pos, aggr_type=model_config['aggr_type'])

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))

    model = GravityParticles(model_config=model_config, device=device, bc_diff=bc_diff)

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5


    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            fig.add_subplot(9, 5, 1+it // step)
            x_ = x0
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([-1.3, 1.3])
            plt.ylim([-1.3, 1.3])
            plt.tight_layout()
            fig.add_subplot(9, 5, 6+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(to_numpy(labels)))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([-1.3, 1.3])
            plt.ylim([-1.3, 1.3])
            plt.tight_layout()

            fig.add_subplot(9, 5, 11+it // step)
            temp1 = torch.cat((x, x0_next), 0)
            temp2 = torch.tensor(np.arange(nparticles), device=device)
            temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
            temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
            temp4 = torch.t(temp4)
            distance3 = torch.sqrt(torch.sum((x[:, 1:3] - x0_next[:, 1:3]) ** 2, 1))
            p = torch.argwhere(distance3 < 0.3)
            pos = dict(enumerate(np.array((temp1[:, 1:3]).detach().cpu()), 0))
            dataset = data.Data(x=temp1[:, 1:3], edge_index=torch.squeeze(temp4[:, :]))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.5)
            plt.xlim([-1.3, 1.3])
            plt.ylim([-1.3, 1.3])
            # plt.text(-1.2, 1.1, f"RMSE: {np.round(rmserr.item(), 4)}", fontsize=24, c='r')


    plt.tight_layout()
    plt.savefig('Fig3_supp.pdf', format="pdf", dpi=300)


    ######################### second set of plots

    config = 'config_Coulomb_3'


    def bc_pos(X):
        return torch.remainder(X, 1.0)
    def bc_diff(D):
        return torch.remainder(D - .5, 1.0) - .5

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

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    delta_t = model_config['delta_t']

    best_model = 20

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    if os.path.isfile(os.path.join(log_dir, f'labels_20.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_20.pt'))
    else:
        labels = T1
        print('Use ground truth labels')

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    ratio = 1
    data_generate(model_config, bVisu=False, bStyle='color', alpha=0.2, bErase=True, bLoad_p=False,step=model_config['nframes'] // 4,ratio = ratio, bc_diff=bc_diff, bc_pos=bc_pos, aggr_type=model_config['aggr_type'])

    model = ElecParticles(model_config=model_config, device=device, bc_diff=bc_diff)

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_recons = []
    y_recons = []
    x_list = []
    y_list = []
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    rmserr_list = []
    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    step=model_config['nframes']//5

    for it in trange(nframes - 1):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it + 1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

        if model_config['prediction'] == '2nd_derivative':
            y = y * ynorm[4] * delta_t
            x[:, 3:5] = x[:, 3:5] + y  # speed update
        else:
            y = y * vnorm[4]
            x[:, 3:5] = y

        x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t )  # position update

        x_recons.append(x.clone().detach())
        y_recons.append(y.clone().detach())

        rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if (it % step == 0) & (it >= 0):

            fig.add_subplot(9, 5, 16+it // step)
            x_ = x0
            sc = 4
            for n in range(nparticles):
                plt.scatter(x_[n, 1].detach().cpu().numpy(),x_[n, 2].detach().cpu().numpy(), s=4, color=cmap.color(int(to_numpy(labels[n])))  )
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            fig.add_subplot(9, 5, 21+it // step)
            x_ = x
            sc = 4
            plt.scatter(x_[:, 1].detach().cpu(), x_[:, 2].detach().cpu(), s=sc, color=cmap.color(int(to_numpy(labels[n])))  )
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()

            fig.add_subplot(9, 5, 26+it // step)
            temp1 = torch.cat((x, x0_next), 0)
            temp2 = torch.tensor(np.arange(nparticles), device=device)
            temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
            temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
            temp4 = torch.t(temp4)
            distance3 = torch.sqrt(torch.sum((x[:, 1:3] - x0_next[:, 1:3]) ** 2, 1))
            p = torch.argwhere(distance3 < 0.3)
            pos = dict(enumerate(np.array((temp1[:, 1:3]).detach().cpu()), 0))
            dataset = data.Data(x=temp1[:, 1:3], edge_index=torch.squeeze(temp4[:, :]))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.5)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            # plt.text(-1.2, 1.1, f"RMSE: {np.round(rmserr.item(), 4)}", fontsize=24, c='r


    plt.tight_layout()
    plt.savefig('Fig3_supp.pdf', format="pdf", dpi=300)
    torch.cuda.empty_cache()

def data_plot_FIG5():


    config = 'config_boids_16_HR2'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    def bc_pos(X):
        return X
    def bc_diff(D):
        return D

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
    aggr_type = model_config['aggr_type']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

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

    model = InteractionParticles_extract(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

    # if best_model == -1:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    # else:
    #     net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_20.pt"

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


    fig = plt.figure(figsize=(10, 9))
    # plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    print('1')
    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                        embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=4)
    plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$',fontsize=14)
    plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    acc_list = []
    for n in range(nparticles):
        embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
        if model_config['prediction'] == '2nd_derivative':
            in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        else:
            in_features = torch.cat((rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], embedding), dim=1)
        with torch.no_grad():
            acc = model.lin_edge(in_features.float())
        acc = acc[:, 0]
        acc_list.append(acc)
    acc_list = torch.stack(acc_list)
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
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))
    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    t = []
    tt = []
    for k in range(model_config['ninteractions']):
        pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        print(torch.median(temp, axis=0).values)
        model_a_[pos, :] = torch.median(temp, axis=0).values.repeat((len(pos), 1))
        t.append(torch.median(temp, axis=0).values)
        tt = np.append(tt, torch.median(temp, axis=0).values.cpu().numpy())
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
    plt.text(0, -1, r"Accuracy: {:.3f}".format(Accuracy), fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))

    ax = fig.add_subplot(3, 3, 4)
    print('4')
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=4)
    plt.xlabel(r'Embedding $\ensuremath{\mathbf{a}}_{i0} [a.u.]$', fontsize=14)
    plt.ylabel(r'Embedding $\ensuremath{\mathbf{a}}_{i1} [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    run = 0
    x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt', map_location=device)
    y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt', map_location=device)


    ################## comparison of predictions ########################################

    it = 300
    x0 = x_list[0][it].clone().detach()
    x0_next = x_list[0][it + 1].clone().detach()
    y0 = y_list[0][it].clone().detach()

    x = x_list[0][it].clone().detach()
    distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
    t = torch.Tensor([radius ** 2])  # threshold
    adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
    edge_index = adj_t.nonzero().t().contiguous()
    dataset = data.Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        y, in_features, lin_edge_out = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation
    y = y * ynorm[4]
    lin_edge_out = lin_edge_out * ynorm[4]

    print(f'PDE_B')
    p = torch.rand(nparticle_types, 3, device=device) * 100  # comprised between 10 and 50
    if len(model_config['p']) > 0:
        for n in range(nparticle_types):
            p[n] = torch.tensor(model_config['p'][n])
    model_B = PDE_B_extract(aggr_type=aggr_type, p=torch.squeeze(p), delta_t=model_config['delta_t'], bc_diff=bc_diff)
    psi_output = []
    for n in range(nparticle_types):
        psi_output.append(model.psi(rr, torch.squeeze(p[n])))
        print(f'p{n}: {np.round(to_numpy(torch.squeeze(p[n])), 4)}')
    with torch.no_grad():
        y_B, sum, cohesion, alignment, separation, diffx, diffv, r, type = model_B(dataset)  # acceleration estimation

    type = to_numpy(type)

    ax = fig.add_subplot(3, 3, 5)
    print('5')
    for n in range(nparticle_types):
        pos = np.argwhere(type == n)
        pos = pos[:, 0].astype(int)
        plt.scatter(to_numpy(r[pos]), to_numpy(torch.norm(lin_edge_out[pos,:], dim=1)), color=cmap.color(n), s=1)
    plt.ylim([0,5E-5])
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$\left| \left| f(\ensuremath{\mathbf{a}}_i, x_j-x_i, \dot{x}_i, \dot{x}_j, r_{ij}) \right| \right|[a.u.]$', fontsize=12)
    ax = fig.add_subplot(3, 3, 6)
    print('6')
    for n in range(nparticle_types):
        pos = np.argwhere(type == n)
        pos = pos[:, 0].astype(int)
        plt.scatter(to_numpy(r[pos]), to_numpy(torch.norm(sum[pos,:], dim=1)), color=cmap.color(n), s=1,alpha=1)
    plt.ylim([0,5E-5])
    plt.xlabel(r'$r_{ij} [a.u.]$', fontsize=14)
    plt.ylabel(r'$\left| \left| f(\ensuremath{\mathbf{a}}_i, x_j-x_i, \dot{x}_i, \dot{x}_j, r_{ij}) \right| \right|[a.u.]$', fontsize=12)

    cohesion_GT = np.zeros(nparticle_types)
    alignment_GT = np.zeros(nparticle_types)
    separation_GT = np.zeros(nparticle_types)
    cohesion_fit = np.zeros(nparticle_types)
    alignment_fit = np.zeros(nparticle_types)
    separation_fit = np.zeros(nparticle_types)

    for n in range(nparticle_types):
        pos =np.argwhere(type==n)
        pos = pos[:,0].astype(int)
        xdiff = to_numpy(diffx[pos,:])
        vdiff = to_numpy(diffv[pos,:])
        rdiff = to_numpy(r[pos])
        x_data = np.concatenate((xdiff,vdiff,rdiff[:,None]),axis=1)
        y_data = to_numpy(torch.norm(lin_edge_out[pos, :], dim=1))
        lin_fit, lin_fitv = curve_fit(func_boids, x_data, y_data, method='dogbox')
        cohesion_fit[n] = lin_fit[0]
        alignment_fit[n] = lin_fit[1]
        separation_fit[n] = lin_fit[2]

    p00 = [np.mean(cohesion_fit), np.mean(alignment_fit), np.mean(separation_fit)]

    for n in range(nparticle_types):
        pos =np.argwhere(type==n)
        pos = pos[:,0].astype(int)
        xdiff = to_numpy(diffx[pos,:])
        vdiff = to_numpy(diffv[pos,:])
        rdiff = to_numpy(r[pos])
        x_data = np.concatenate((xdiff,vdiff,rdiff[:,None]),axis=1)
        y_data = to_numpy(torch.norm(lin_edge_out[pos, :], dim=1))
        lin_fit, lin_fitv = curve_fit(func_boids, x_data, y_data, method='dogbox', p0=p00)
        cohesion_fit[n] = lin_fit[0]
        alignment_fit[n] = lin_fit[1]
        separation_fit[n] = lin_fit[2]


    ax = fig.add_subplot(3, 3, 7)
    print('6')
    x_data = np.abs(to_numpy(p[:,0])*0.5E-5)
    y_data = np.abs(cohesion_fit)
    lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
    plt.plot(x_data, func_lin(x_data, lin_fit[0],lin_fit[1]), color='r', linewidth=0.5)
    for n in range(nparticle_types):
        plt.scatter(x_data[n], y_data[n], color=cmap.color(n), s=30)
    plt.xlabel(r'True cohesion coeff. $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted cohesion coeff. $[a.u.]$', fontsize=14)
    plt.text(4E-5, 4.5E-4, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=12)
    residuals = y_data - func_lin(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(4E-5, 4E-4, f"$R^2$: {np.round(r_squared, 3)}", fontsize=12)

    ax = fig.add_subplot(3, 3, 8)
    print('6')
    x_data = np.abs(to_numpy(p[:,1])*5E-4)
    y_data = alignment_fit
    lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
    plt.plot(x_data, func_lin(alignment_fit, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    for n in range(nparticle_types):
        plt.scatter(x_data[n], y_data[n], color=cmap.color(n), s=30)
    plt.xlabel(r'True alignment coeff. $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted alignment coeff. $[a.u.]$', fontsize=14)
    plt.text(5e-3, 0.042, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=12)
    residuals = y_data - func_lin(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(5e-3, 0.038, f"$R^2$: {np.round(r_squared, 3)}", fontsize=12)

    ax = fig.add_subplot(3, 3, 9)
    x_data = np.abs(to_numpy(p[:,2])*1E-8)
    y_data = separation_fit
    lin_fit, lin_fitv = curve_fit(func_lin, x_data, y_data)
    plt.plot(x_data, func_lin(separation_fit, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    for n in range(nparticle_types):
        plt.scatter(x_data[n], y_data[n], color=cmap.color(n), s=30)
    plt.xlabel(r'True alignment coeff. $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted alignment coeff. $[a.u.]$', fontsize=14)
    plt.text(5e-8, 4.4E-7, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=12)
    residuals = y_data - func_lin(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(5e-8, 3.9E-7, f"$R^2$: {np.round(r_squared, 3)}", fontsize=12)

    #############

    plt.tight_layout()

    plt.savefig('Fig5.pdf', format="pdf", dpi=300)
    plt.savefig('Fig5.jpg', dpi=300)
    plt.close()


    if False:
        # TO BE CORRECTED ###

        fig = plt.figure(figsize=(10, 9))
        plt.ion()
        n = 14
        b_cohesion = p[n, 0:1].repeat(1, 2) * 0.5E-5 * diffx
        b_alignment = p[n, 1:2].repeat(1, 2) * 5E-4 * diffv
        b_separation = - p[n, 2:3].repeat(1, 2) * 1E-8 * diffx / (r[:, None].repeat(1, 2))
        b_sum = b_cohesion + b_alignment + b_separation
        pos = np.argwhere(type == n)
        pos = pos[:, 0].astype(int)
        plt.scatter(to_numpy(r[pos]), to_numpy(torch.norm(lin_edge_out[pos, :], dim=1)), color=cmap.color(n), s=1)
        plt.scatter(to_numpy(r[pos]), to_numpy(torch.norm(sum[pos, :], dim=1)), color='k', s=1, alpha=0.5)
        plt.scatter(to_numpy(r[pos]), to_numpy(torch.norm(b_sum[pos, :], dim=1)), color='r', s=1, alpha=0.5)
        pos = np.argwhere(type == n)
        pos = pos[:, 0].astype(int)
        xdiff = to_numpy(diffx[pos, :])
        vdiff = to_numpy(diffv[pos, :])
        rdiff = to_numpy(r[pos])
        x_data = np.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
        y_data = to_numpy(torch.norm(b_sum[pos, :], dim=1))
        lin_fit, lin_fitv = curve_fit(func_boids, x_data, y_data, method='dogbox', p0=[50*1E-5, 50*5E-4, 50*1E-8])
        fit_sum = func_boids(x_data, lin_fit[0], lin_fit[1], lin_fit[2])
        plt.scatter(to_numpy(r[pos]), fit_sum, color='b', s=2)

def data_plot_FIG6():

    config = 'config_wave_HR3'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    nrun = model_config['nrun']
    kmeans_input = model_config['kmeans_input']
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

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

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

    # fig = plt.figure(figsize=(20, 5))
    # plt.ion()
    # ax = fig.add_subplot(1, 5, 4)
    #
    # deg_list = np.array(deg_list)
    # distance_list = np.array(distance_list)
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0] + deg_list[:, 1], c='k')
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0], c='r')
    # plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0] - deg_list[:, 1], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Degree [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 5, 1)
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0] + distance_list[:, 1], c='k')
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0], c='r')
    # plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0] - distance_list[:, 1], c='k')
    # plt.ylim([0, model_config['radius']])
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Distance [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 5, 2)
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0] + x_stat[:, 2], c='k')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0], c='r')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0] - x_stat[:, 2], c='k')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1] + x_stat[:, 3], c='k')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1], c='r')
    # plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1] - x_stat[:, 3], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Velocity [a.u]', fontsize="14")
    # ax = fig.add_subplot(1, 5, 3)
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0] + y_stat[:, 2], c='k')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0], c='r')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0] - y_stat[:, 2], c='k')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1] + y_stat[:, 3], c='k')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1], c='r')
    # plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1] - y_stat[:, 3], c='k')
    # plt.xlim([0, nframes])
    # plt.xlabel('Frame [a.u]', fontsize="14")
    # plt.ylabel('Acceleration [a.u]', fontsize="14")
    # plt.tight_layout()
    # plt.show()

    h_list = []
    for run in trange(NGraphs):
        h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt', map_location=device)
        h_list.append(torch.stack(h))
    h = torch.stack(h_list)
    h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
    hnorm = torch.std(h)
    torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
    print(hnorm)
    model = MeshLaplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

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
    model.eval()

    print('')
    time.sleep(0.5)
    print('Plotting ...')

    if bMesh:
        x = x_list[0][0].clone().detach()
        index_particles = []
        for n in range(model_config['nparticle_types']):
            index = np.argwhere(to_numpy(x[:, 5]) == n)
            index_particles.append(index.squeeze())

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

    if bMesh:
        X1 = torch.rand(nparticles, 2, device=device)
        x_width = int(np.sqrt(nparticles))
        xs = torch.linspace(0, 1, steps=x_width)
        ys = torch.linspace(0, 1, steps=x_width)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        x = torch.reshape(x, (x_width ** 2, 1))
        y = torch.reshape(y, (x_width ** 2, 1))
        x_width = 1 / x_width / 8
        X1[0:nparticles, 0:1] = x[0:nparticles]
        X1[0:nparticles, 1:2] = y[0:nparticles]
        X1 = X1 + torch.randn(nparticles, 2, device=device) * x_width
        X1_ = torch.clamp(X1, min=0, max=1)

        particle_type_map = model_config['particle_type_map']
        i0 = imread(f'graphs_data/{particle_type_map}')

        values = i0[(to_numpy(X1_[:, 0]) * 255).astype(int), (to_numpy(X1_[:, 1]) * 255).astype(int)]
        T1 = torch.tensor(values, device=device)
        T1 = T1[:, None]

    cmap = cc(model_config=model_config)

    fig = plt.figure(figsize=(10, 9))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    print('1')

    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                        embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=1)
    plt.xlabel(r'Embedding 0',fontsize=14)
    plt.ylabel(r'Embedding 1',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 2)
    with torch.no_grad():
        f_list = []
        for n in trange(nparticles):
            r = torch.tensor(np.linspace(-250, 250, 1000)).to(device)
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((r[:, None], embedding), dim=1)
            h = model.lin_phi(in_features.float())
            h = h[:, 0]
            f_list.append(h)
        f_list = torch.stack(f_list)
        coeff_norm = to_numpy(f_list)

    print('UMAP ...')
    trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                      random_state=42, transform_queue_size=0).fit(coeff_norm)
    proj_interaction = trans.transform(coeff_norm)
    proj_interaction = np.squeeze(proj_interaction)

    for n in range(nparticle_types):
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=0.1)
        plt.xlabel(r'UMAP 0', fontsize=14)
        plt.ylabel(r'UMAP 1', fontsize=14)

    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,random_state=13)
    if kmeans_input == 'plot':
        kmeans.fit(proj_interaction)
    if kmeans_input == 'embedding':
        kmeans.fit(embedding_)
    label_list = []
    for n in range(nparticle_types):
        tmp = kmeans.labels_[index_particles[n]]
        sub_group = np.round(np.median(tmp))
        label_list.append(sub_group)
    label_list = np.array(label_list)
    new_labels = 0* kmeans.labels_.copy()
    for n in range(nparticle_types):
        new_labels[kmeans.labels_ == label_list[n]] = n
    Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
    print(' ')
    print (f'Accuracy: {np.round(Accuracy,3)}')
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))
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
    confusion_matrix = metrics.confusion_matrix(to_numpy(T1), new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if nparticle_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(Accuracy), fontsize=12)

    ax = fig.add_subplot(3, 3, 4)
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=5)
    plt.xlabel(r'Embedding 0', fontsize=14)
    plt.ylabel(r'Embedding 1', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    print('5')
    with torch.no_grad():
        t = to_numpy(model.a[0])
    tmean = np.ones((model_config['nparticle_types'], model_config['embedding']))
    u = torch.tensor(np.linspace(-250, 250, 100)).to(device)
    f_list = []
    for n in range(model_config['nparticle_types']):
        tmean[n] = np.mean(t[index_particles[n], :], axis=0)
        embedding = torch.tensor(tmean[n], device=device) * torch.ones((100, model_config['embedding']),device=device)
        in_features = torch.cat((u[:, None], embedding), dim=1)
        h = model.lin_phi(in_features.float())
        h = h[:, 0]
        f_list.append(h)
        plt.plot(to_numpy(u),to_numpy(h) * to_numpy(hnorm),linewidth=1)

    plt.xlabel(r'$\Delta u_{i} [a.u.]$', fontsize=14)
    plt.ylabel(r'$\Phi (\ensuremath{\mathbf{a}}_i, \Delta u_i) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(-200, 2, r'Model', fontsize=14)
    plt.xlim([-250,250])
    plt.ylim([-3, 3])

    popt_list = []
    for n in range(nparticle_types):
        popt, pcov = curve_fit(func_lin, to_numpy(u), to_numpy(f_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    ax = fig.add_subplot(3, 3, 6)
    print('6')
    c = model_config['c']
    for n in range(nparticle_types):
        plt.plot(to_numpy(u), 1E-2 * to_numpy(u*c[n]), linewidth=1, color=cmap.color(n))
    plt.xlabel(r'$\Delta u_{i} [a.u.]$', fontsize=14)
    plt.ylabel(r'$\Phi (\ensuremath{\mathbf{a}}_i, \Delta u_i) [a.u.]$', fontsize=14)
    plt.text(-200, 2, r'True', fontsize=14)
    plt.xlim([-250,250])
    plt.ylim([-3, 3])

    ax = fig.add_subplot(3, 3, 7)
    for n in range(nparticle_types):
        plt.scatter(c[n], to_numpy(hnorm) * popt_list[n, 0] * 100, color=cmap.color(n))
    plt.xlabel(r'True viscosity $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted viscosity $[a.u.]$', fontsize=14)

    ax = fig.add_subplot(3, 3, 8)
    for n in range(nparticle_types):
        plt.scatter(to_numpy(x[index_particles[n]]),
                    to_numpy(y[index_particles[n]]), s=10, color=cmap.color(n))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(r'$x_i [a.u.]$', fontsize=14)
    plt.ylabel(r'$y_i [a.u.]$', fontsize=14)

    plt.tight_layout()

    plt.savefig('Fig6.pdf', format="pdf", dpi=300)
    plt.savefig('Fig6.jpg', dpi=300)

    plt.close()

def data_plot_FIG7():

    config = 'config_RD_RPS2'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
    with open(f'./config/{config}.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]

    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = 'Mesh' in model_config['model']
    nrun = model_config['nrun']
    kmeans_input = model_config['kmeans_input']
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

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    T1 = T1[:, None]

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    print('Load normalizations ...')
    time.sleep(1)

    xlist = []
    x_list.append(torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_0.pt', map_location=device))
    ylist = []
    y_list.append(torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_0.pt', map_location=device))

    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)

    h_list = []
    h_list.append(torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_0.pt', map_location=device))
    hnorm = torch.load(os.path.join(log_dir, 'hnorm.pt'), map_location=device)

    model = MeshLaplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_diff=bc_diff)

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
    model.eval()

    print('')
    time.sleep(0.5)
    print('Plotting ...')

    x = x_list[0][0].clone().detach()
    index_particles = []
    for n in range(model_config['nparticle_types']):
        index = np.argwhere(to_numpy(x[:, 5]) == n)
        index_particles.append(index.squeeze())

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

    X1 = torch.rand(nparticles, 2, device=device)
    x_width = int(np.sqrt(nparticles))
    xs = torch.linspace(0, 1, steps=x_width)
    ys = torch.linspace(0, 1, steps=x_width)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    x = torch.reshape(x, (x_width ** 2, 1))
    y = torch.reshape(y, (x_width ** 2, 1))
    x_width = 1 / x_width / 8
    X1[0:nparticles, 0:1] = x[0:nparticles]
    X1[0:nparticles, 1:2] = y[0:nparticles]
    X1 = X1 + torch.randn(nparticles, 2, device=device) * x_width
    X1_ = torch.clamp(X1, min=0, max=1)

    particle_type_map = model_config['particle_type_map']
    i0 = imread(f'graphs_data/{particle_type_map}')

    values = i0[(to_numpy(X1_[:, 0]) * 255).astype(int), (to_numpy(X1_[:, 1]) * 255).astype(int)]
    T1 = torch.tensor(values, device=device)
    T1 = T1[:, None]

    cmap = cc(model_config=model_config)

    fig = plt.figure(figsize=(10, 9))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    print('1')

    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],
                        embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=1)
    plt.xlabel(r'Embedding 0',fontsize=14)
    plt.ylabel(r'Embedding 1',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 2)
    with torch.no_grad():
        f_list = []
        for n in trange(nparticles):
            embedding = model.a[0, n, :] * torch.ones((100, model_config['embedding']), device=device)
            if model_config['model'] == 'RD_RPS_Mesh':
                u = torch.tensor(np.linspace(0, 1, 100)).to(device)
                u = u[:, None]
                in_features = torch.cat((u, u, u, u, u, u, embedding), dim=1)
                r = u
            else:
                r = torch.tensor(np.linspace(-250, 250, 100)).to(device)
                in_features = torch.cat((r[:, None], embedding), dim=1)
            h = model.lin_phi(in_features.float())
            h = h[:, 0]
            f_list.append(h)
        f_list = torch.stack(f_list)
        coeff_norm = to_numpy(f_list)

    print('UMAP ...')
    trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                      random_state=42, transform_queue_size=0).fit(coeff_norm)
    proj_interaction = trans.transform(coeff_norm)
    proj_interaction = np.squeeze(proj_interaction)

    for n in range(nparticle_types):
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=0.1)
        plt.xlabel(r'UMAP 0', fontsize=14)
        plt.ylabel(r'UMAP 1', fontsize=14)

    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,random_state=13)
    if kmeans_input == 'plot':
        kmeans.fit(proj_interaction)
    if kmeans_input == 'embedding':
        kmeans.fit(embedding_)
    label_list = []
    for n in range(nparticle_types):
        tmp = kmeans.labels_[index_particles[n]]
        sub_group = np.round(np.median(tmp))
        label_list.append(sub_group)
    label_list = np.array(label_list)
    new_labels = 0* kmeans.labels_.copy()
    for n in range(nparticle_types):
        new_labels[kmeans.labels_ == label_list[n]] = n
    Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
    print(' ')
    print (f'Accuracy: {np.round(Accuracy,3)}')
    torch.save(torch.tensor(new_labels, device=device), os.path.join(log_dir, f'labels_20.pt'))
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
    confusion_matrix = metrics.confusion_matrix(to_numpy(T1), new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if nparticle_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(Accuracy), fontsize=12)

    ax = fig.add_subplot(3, 3, 4)
    for m in range(model.a.shape[0]):
        for n in range(model.a.shape[1]):
            plt.scatter(to_numpy(model.a[m][n, 0]),
                        to_numpy(model.a[m][n, 1]),
                        color=cmap.color(new_labels[n]), s=6)
    plt.xlabel(r'Embedding 0', fontsize=14)
    plt.ylabel(r'Embedding 1', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    print('5')
    with torch.no_grad():
        t = to_numpy(model.a[0])
    tmean = np.ones((model_config['nparticle_types'], model_config['embedding']))
    u = torch.tensor(np.linspace(-250, 250, 100)).to(device)
    f_list = []
    for n in range(model_config['nparticle_types']):
        tmean[n] = np.mean(t[index_particles[n], :], axis=0)
        embedding = torch.tensor(tmean[n], device=device) * torch.ones((100, model_config['embedding']),device=device)
        in_features = torch.cat((u[:, None], embedding), dim=1)
        h = model.lin_phi(in_features.float())
        h = h[:, 0]
        f_list.append(h)
        plt.plot(to_numpy(u),to_numpy(h) * to_numpy(hnorm),linewidth=1)

    plt.xlabel(r'$\Delta u_{i} [a.u.]$', fontsize=14)
    plt.ylabel(r'$\Phi (\ensuremath{\mathbf{a}}_i, \Delta u_i) [a.u.]$', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.text(-200, 2, r'Model', fontsize=14)
    plt.xlim([-250,250])
    plt.ylim([-3, 3])

    popt_list = []
    for n in range(nparticle_types):
        popt, pcov = curve_fit(func_lin, to_numpy(u), to_numpy(f_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    ax = fig.add_subplot(3, 3, 6)
    print('6')
    c = model_config['c']
    for n in range(nparticle_types):
        plt.plot(to_numpy(u), 1E-2 * to_numpy(u*c[n]), linewidth=1, color=cmap.color(n))
    plt.xlabel(r'$\Delta u_{i} [a.u.]$', fontsize=14)
    plt.ylabel(r'$\Phi (\ensuremath{\mathbf{a}}_i, \Delta u_i) [a.u.]$', fontsize=14)
    plt.text(-200, 2, r'True', fontsize=14)
    plt.xlim([-250,250])
    plt.ylim([-3, 3])

    ax = fig.add_subplot(3, 3, 7)
    for n in range(nparticle_types):
        plt.scatter(c[n], to_numpy(hnorm) * popt_list[n, 0] * 100, color=cmap.color(n))
    plt.xlabel(r'True viscosity $[a.u.]$', fontsize=14)
    plt.ylabel(r'Predicted viscosity $[a.u.]$', fontsize=14)

    ax = fig.add_subplot(3, 3, 8)
    for n in range(nparticle_types):
        plt.scatter(to_numpy(x[index_particles[n]]),
                    to_numpy(y[index_particles[n]]), s=10, color=cmap.color(n))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(r'$x_i [a.u.]$', fontsize=14)
    plt.ylabel(r'$y_i [a.u.]$', fontsize=14)

    plt.tight_layout()

    plt.savefig('Fig7.pdf', format="pdf", dpi=300)
    plt.savefig('Fig7.jpg', dpi=300)

    plt.close()



if __name__ == '__main__':

    print('')
    print('version 1.9 240103')
    print('use of https://github.com/gpeyre/.../ml_10_particle_system.ipynb')
    print('')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    # arbitrary_3 training
    # data_plot_FIG2()
    # print(' ')
    # print(' ')
    # arbitrary_3 inference
    # data_plot_FIG2sup()
    # print(' ')
    # print(' ')
    # arbitrary_16 training
    # data_plot_FIG3sup()
    # print(' ')
    # print(' ')
    # arbitrary_3 inference
    # data_plot_FIG4sup()

    # gravity model
    # data_plot_FIG3()
    # gravity model continuous
    # data_plot_FIG3_continous()

    # training Coloumb_3
    # data_plot_FIG4()

    # data_plot_FIG5sup()

    # boids HR2
    #data_plot_FIG5()

    data_plot_FIG6()

    # data_plot_FIG7()






