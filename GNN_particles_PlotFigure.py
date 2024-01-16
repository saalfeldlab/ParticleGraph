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

def func_pow(x, a, b):
    return a / (x**b)


def func_lin(x, a, b):
    return a * x + b


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

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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
    plt.text(0,0.02,r'Model', fontsize=14)
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
    plt.text(0,0.02,r'True', fontsize=14)


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

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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

    model = GravityParticles(model_config=model_config, device=device, bc_diff=bc_diff)

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
    Precision = metrics.precision_score(to_numpy(T1), new_labels, average='micro')
    Recall = metrics.recall_score(to_numpy(T1), new_labels, average='micro')
    F1 = metrics.f1_score(to_numpy(T1), new_labels, average='micro')
    plt.text(0, -0.75, r"Accuracy: {:.3f}".format(F1), fontsize=12)
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
    plt.text(0.5, 4.5, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)

    ax = fig.add_subplot(3, 3, 8)
    print('8')
    plt.scatter(p, -popt_list[:, 1], color='k')
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.xlabel(r'True mass $[a.u.]$', fontsize=14)
    plt.ylabel(r'Exponent fit $[a.u.]$', fontsize=14)
    plt.text(0.5, -0.5, f"{np.round(np.mean(popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
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
    for n in range(len(plot_list_2)):
        plt.plot(to_numpy(vv), to_numpy(plot_list_2[n]), linewidth=1)
    plt.xlabel(r'Normalized $\ensuremath{\mathbf{\dot{x}}}_i [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_j, r_{ij}, \ensuremath{\mathbf{\dot{x}}}_i) [a.u.]$', fontsize=14)
    plt.xlim([0, 2])
    plt.tight_layout()

    plt.savefig('Fig3.pdf', format="pdf", dpi=300)
    plt.savefig('Fig3.jpg', dpi=300)

    plt.close()


    t = torch.tensor(tt, device=device)
    t = torch.reshape(t, (16, 2))

    fig = plt.figure(figsize=(10, 10))
    plt.ion()

    ax = fig.add_subplot(2, 2, 1)
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
    plt.imshow(to_numpy(pred), extent=[0.002, 0.01, -1, 1], aspect='auto', origin='lower', cmap='jet')


    ax = fig.add_subplot(2, 2, 2)
    vv = 0.5*torch.ones((10000,1), device=device)
    tt = torch.zeros((1, 2), device=device)
    tt[0,0] = torch.mean(t[0])
    tt[0,1] = torch.mean(t[1])
    tt = tt * torch.ones((10000, model_config['embedding']), device=device)
    tt = tt.to(device)

    tl = torch.linspace(torch.min(t[:,0]), torch.max(t[:,0]), 100)
    tl = tl.to(torch.float64)
    tl = tl.to(device)
    rr_, tl_ = torch.meshgrid(rr, tl, indexing='xy')

    rr_ = torch.reshape(rr_, (rr_.shape[0] * rr_.shape[1], 1))
    tl_ = torch.reshape(tl_, (tl_.shape[0] * tl_.shape[1], 1))
    tt[:,0:1] = tl_

    in_features = torch.cat((rr_ / model_config['radius'], 0 * rr_, rr_ / model_config['radius'], vv_, vv_, vv_, vv_, tt), dim=1)
    with torch.no_grad():
        pred = model.lin_edge(in_features.float())
    pred = pred[:, 0]
    pred=torch.reshape(pred,(100,100))
    plt.imshow(to_numpy(pred), extent=[0.002, 0.01, to_numpy(torch.min(t[:,0])), to_numpy(torch.max(t[:,0]))], aspect='auto', origin='lower', cmap='jet')


    ax = fig.add_subplot(2, 2, 3)
    vv = 0.5*torch.ones((10000,1), device=device)
    tt = torch.zeros((1, 2), device=device)
    tt[0,0] = torch.mean(t[0])
    tt[0,1] = torch.mean(t[1])
    tt = tt * torch.ones((10000, model_config['embedding']), device=device)
    tt = tt.to(device)

    tl = torch.linspace(torch.min(t[:,1]), torch.max(t[:,1]), 100)
    tl = tl.to(torch.float64)
    tl = tl.to(device)
    rr_, tl_ = torch.meshgrid(rr, tl, indexing='xy')

    rr_ = torch.reshape(rr_, (rr_.shape[0] * rr_.shape[1], 1))
    tl_ = torch.reshape(tl_, (tl_.shape[0] * tl_.shape[1], 1))
    tt[:,1:2] = tl_

    in_features = torch.cat((rr_ / model_config['radius'], 0 * rr_, rr_ / model_config['radius'], vv_, vv_, vv_, vv_, tt), dim=1)
    with torch.no_grad():
        pred = model.lin_edge(in_features.float())
    pred = pred[:, 0]
    pred=torch.reshape(pred,(100,100))
    plt.imshow(to_numpy(pred), extent=[0.002, 0.01, to_numpy(torch.min(t[:,0])), to_numpy(torch.max(t[:,0]))], aspect='auto', origin='lower', cmap='jet')

    #### derivate


    in_features = torch.cat((rr_ / model_config['radius'], 0 * rr_, rr_ / model_config['radius'], vv_, vv_, vv_, vv_, tt), dim=1)

    in_features.requires_grad = True

    with torch.no_grad():
        pred = model.lin_edge(in_features.float())
    pred = pred[:, 0]
    pred=torch.reshape(pred,(100,100))
    plt.imshow(to_numpy(pred), extent=[0.002, 0.01, to_numpy(torch.min(t[:,1])), to_numpy(torch.max(t[:,1]))], aspect='auto', origin='lower', cmap='jet')


    plt.tight_layout()

def data_plot_FIG4():


    config = 'config_Coulomb_3'
    # model_config = load_model_config(id=config)

    # Load parameters from config file
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

    x_stat = np.array(x_stat)
    y_stat = np.array(y_stat)


    model = ElecParticles(model_config=model_config, device=device,bc_diff = bc_diff)


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
                #              to_numpy(acc) * to_numpy(ynorm[4]),
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
    plt.scatter(ptrue_list, -popt_list[:, 1], color='k')
    plt.ylim([0, 4])
    plt.xlabel(r'True $q_i q_j [a.u.]$', fontsize=14)
    plt.ylabel(r'Exponential fit $[a.u.]$', fontsize=14)
    plt.text(-2, 3.5, f"{np.round(-np.mean(popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)
    plt.tight_layout()


    plot_list_2 = []
    vv = torch.tensor(np.linspace(0, 2, 100)).to(device)
    r_list = np.linspace(0.002, 0.01, 5)
    for r_ in r_list:
        rr_ = r_ * torch.tensor(np.ones((vv.shape[0], 1)), device=device)
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                embedding0 = torch.tensor(tmean[m], device=device) * torch.ones((100, model_config['embedding']),
                                                                                device=device)
                embedding1 = torch.tensor(tmean[n], device=device) * torch.ones((100, model_config['embedding']),
                                                                                device=device)
                in_features = torch.cat((-rr_ / model_config['radius'], 0 * rr_,
                                         rr_ / model_config['radius'], vv[:, None], vv[:, None], vv[:, None], vv[:, None],
                                         embedding0,embedding1), dim=1)
                with torch.no_grad():
                    pred = model.lin_edge(in_features.float())
                pred = pred[:, 0]
                plot_list_2.append(pred * ynorm[4])

    ax = fig.add_subplot(3, 3, 9)
    print('9')
    for n in range(len(plot_list_2)):
        plt.plot(to_numpy(vv), to_numpy(plot_list_2[n]), linewidth=1)
    plt.xlabel(r'Normalized $\ensuremath{\mathbf{\dot{x}}}_i [a.u.]$', fontsize=14)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i,\ensuremath{\mathbf{a}}_j, r_{ij}, \ensuremath{\mathbf{\dot{x}}}_i) [a.u.]$', fontsize=14)
    plt.xlim([0, 2])

    plt.savefig('Fig4.pdf', format="pdf", dpi=300)
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

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
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

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

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
    data_plot_FIG4sup()

    # gravity model

    # data_plot_FIG3()

    # data_plot_FIG4()

    # data_plot_FIG5sup()

    # data_plot_FIG5()





