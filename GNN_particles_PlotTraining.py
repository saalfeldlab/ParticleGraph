import glob
import logging
import time
from shutil import copyfile

# import networkx as nx
import torch.nn as nn
import torch_geometric.data as data
import umap
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from tqdm import trange
import os
from sklearn import metrics
from matplotlib import rc
import matplotlib
# matplotlib.use("Qt5Agg")

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.particle_initialization import init_particles, init_mesh
from ParticleGraph.generators.utils import choose_model, choose_mesh_model
from ParticleGraph.models.utils import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.data_loaders import *
from ParticleGraph.utils import *
from ParticleGraph.fitting_models import linear_model
from ParticleGraph.embedding_cluster import *
from ParticleGraph.models import Division_Predictor
from ParticleGraph.models.utils import *

from GNN_particles_PlotFigure import plot_embedding, plot_function, plot_umap, plot_confusion_matrix, Mesh_RPS_extract

def data_plot_training(config, mode, device):
    print('')

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    n_epochs = train_config.n_epochs
    radius = simulation_config.max_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    replace_with_cluster = 'replace' in train_config.sparsity
    visualize_embedding = False
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    aggr_type = config.graph_model.aggr_type

    embedding_cluster = EmbeddingCluster(config)

    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)

    x_list = []
    y_list = []
    print('Load data ...')
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)

    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in trange(n_frames):
            if (k%10 == 0) | (n_frames<1000):
                x = torch.cat((x,x_list[run][k].clone().detach()),0)
                y = torch.cat((y,y_list[run][k].clone().detach()),0)
        print(x_list[run][k].shape)
    vnorm = norm_velocity(x, device)
    ynorm = norm_acceleration(y, device)
    vnorm = vnorm[4]
    ynorm = ynorm[4]
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    if has_mesh:
        x_mesh_list = []
        y_mesh_list = []
        for run in trange(NGraphs):
            x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
            x_mesh_list.append(x_mesh)
            h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
            y_mesh_list.append(h)
        h = y_mesh_list[0][0].clone().detach()
        for run in range(NGraphs):
            for k in range(n_frames):
                h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(f'hnorm: {to_numpy(hnorm)}')
        time.sleep(0.5)

        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)

        mask_mesh = mesh_data['mask']
        # mesh_pos = mesh_data['mesh_pos']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']
        # face = mesh_data['face']

        mask_mesh = mask_mesh.repeat(batch_size, 1)

    h=[]
    x=[]
    y=[]

    print('done ...')

    model, bc_pos, bc_dpos = choose_training_model(config, device)


    if  has_cell_division:
        model_division = Division_Predictor(config, device)
        optimizer_division, n_total_params_division = set_trainable_division_parameters(model_division, lr=1E-3)
        logger.info(f"Total Trainable Divsion Params: {n_total_params_division}")
        logger.info(f'Learning rates: 1E-3')

    x = x_list[1][0].clone().detach()
    type_list = x[:, 5:6].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    time.sleep(0.5)


    matplotlib.use("Qt5Agg")
    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0
    style = {
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "sans-serif"
    }
    matplotlib.rcParams.update(style)
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "HelveticaNeue", "Helvetica-Neue", "Helvetica", "Arial",
                                       "Liberation"]

    epoch_list = [20]
    for epoch in epoch_list:


        net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])



        n_particle_types = 3
        index_particles = []
        for n in range(n_particle_types):
            index_particles.append(
                np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=50)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.xlim([0,2])
        plt.ylim([0, 2])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()


        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        if model_config.particle_model_name == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif model_config.particle_model_name == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=True, config=config,
                                                                model_lin_edge=model.lin_edge, model_a=model.a,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                types=to_numpy(x[:, 5]),
                                                                cmap=cmap, device=device)
        plt.close()

        match train_config.cluster_method:
            case 'kmeans_auto_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
            case 'kmeans_auto_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                proj_interaction = embedding
            case 'distance_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
            case 'distance_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=1.5)
                proj_interaction = embedding
            case 'distance_both':
                new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
                labels, n_clusters = embedding_cluster.get(new_projection, 'distance')

        fig = plt.figure(figsize=(12, 12))
        for n in range(n_clusters):
            pos = np.argwhere(labels == n)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=5)
        label_list = []
        for n in range(n_particle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)

        plt.xlabel('proj 0', fontsize=12)
        plt.ylabel('proj 1', fontsize=12)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        new_labels = labels.copy()
        for n in range(n_particle_types):
            new_labels[labels == label_list[n]] = n
            pos = np.argwhere(labels == label_list[n])
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                            color=cmap.color(n), s=0.1)
        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')

        fig = plt.figure(figsize=(12, 12))
        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                model_a_[pos, :] = median_center
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
        for n in np.unique(new_labels):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), color='k', s=5)
        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)
        plt.close()

        # Constrain embedding domain
        with torch.no_grad():
            model.a[1] = model_a_.clone().detach()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)
        if n_particle_types > 1000:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                        cmap='viridis')
        else:
            for n in range(n_particle_types):
                pos = np.argwhere(new_labels == n).squeeze().astype(int)
                plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=200)

        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        func_list = []
        for n in range(n_particle_types):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            embedding = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n), linewidth=4)
        plt.xlabel(r'$r_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim([-0.04, 0.03])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        p = config.simulation.params
        if len(p) > 0:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_{dataset_name}/p.pt')
        for n in range(n_particle_types - 1, -1, -1):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=4)
        plt.xlabel(r'$r_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim([-0.04, 0.03])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{dataset_name}.tif",dpi=170.7)
        plt.close()




if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')

    config_list =['arbitrary_3_continuous']

    for config_file in config_list:

        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        cmap = CustomColorMap(config=config)  # create colormap for given model_config

        data_plot_training(config, mode='figures' , device=device)



