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
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)


    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs - 1}')
    logger.info(f'Graph files N: {NGraphs - 1}')

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
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')
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
        logger.info(f'hnorm: {to_numpy(hnorm)}')
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

    lr_embedding = train_config.learning_rate_embedding_start
    lr = train_config.learning_rate_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')

    if  has_cell_division:
        model_division = Division_Predictor(config, device)
        optimizer_division, n_total_params_division = set_trainable_division_parameters(model_division, lr=1E-3)
        logger.info(f"Total Trainable Divsion Params: {n_total_params_division}")
        logger.info(f'Learning rates: 1E-3')


    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    x = x_list[1][0].clone().detach()
    T1 = x[:, 5:6].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = [0.002002, 0.000676, 0.000481, 0.000385, 0.000334, 0.000299, 0.000144, 0.000268, 0.000249, 0.000232, 0.000218, 0.000108, 0.000203, 0.000192, 0.000184, 0.000177, 0.000088, 0.000168, 0.000086, 0.000084, 0.000082,0.000082,0.000082]
    time.sleep(0.5)

    if mode == 'figures':

        matplotlib.use("Qt5Agg")
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})

        epoch_list = [0, 1, 3, 9 , 15, 20]
        for epoch in epoch_list:
            net = f"./log/try_{dataset_name}/models/best_model_with_{epoch}_graphs.pt"
            print(f'network: {net}')

            net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_{epoch}.pt"
            state_dict = torch.load(net,map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])


            if True:
                matplotlib.rcParams['savefig.pad_inches'] = 0
                fig = plt.figure(figsize=(12, 12))
                embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)
                # for n in range(n_particle_types):
                #     plt.scatter(embedding[index_particles[n], 0],
                #                 embedding[index_particles[n], 1], color=cmap.color(n), s=20)
                for n in range(n_particle_types):
                    plt.scatter(embedding[n, 0],
                                embedding[n, 1], s=50)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif",dpi=170.7)
                plt.close()

                fig = plt.figure(figsize=(12, 12))
                rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
                func_list = []
                for n in range(n_particles):
                    embedding_ = model.a[1, n, :] * torch.ones((1000, 2), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    func_list.append(func)
                    if n % 5 == 0:
                        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), linewidth=1)
                # plt.ylim([0,3000])
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif")
                plt.close()



            else:

                fig = plt.figure(figsize=(12, 12))
                embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)
                for n in range(n_particle_types):
                    plt.scatter(embedding[index_particles[n], 0],
                                embedding[index_particles[n], 1], color=cmap.color(n), s=10)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif")
                plt.close()

                fig = plt.figure(figsize=(12.5, 9.6))
                plt.ion()
                ax = fig.add_subplot(3, 4, 1)
                embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 1, '$5.10^4$',
                                           fig, ax, cmap, device)

                ax = fig.add_subplot(3, 4, 2)
                rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
                func_list = plot_function(True, 'f)', config.graph_model.particle_model_name, model.lin_edge, model.a, 1,
                                          to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles,
                                          n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap, device)

                ax = fig.add_subplot(3, 4, 3)
                proj_interaction, new_labels, n_clusters = plot_umap('g)', func_list, log_dir, 500, index_particles,
                                                                     n_particles, n_particle_types, embedding_cluster, 20,
                                                                     '$10^6$', fig, ax, cmap,device)

                ax = fig.add_subplot(3, 4, 4)
                Accuracy = plot_confusion_matrix('h)', to_numpy(x[:, 5:6]), new_labels, n_particle_types, 1, '$5.10^$4',
                                                 fig, ax)
                plt.tight_layout()

                model_a_ = model.a[1].clone().detach()
                for k in range(n_clusters):
                    pos = np.argwhere(new_labels == k).squeeze().astype(int)
                    temp = model_a_[pos, :].clone().detach()
                    model_a_[pos, :] = torch.median(temp, dim=0).values.repeat((len(pos), 1))
                with torch.no_grad():
                    for n in range(model.a.shape[0]):
                        model.a[n] = model_a_
                embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)

                ax = fig.add_subplot(3, 4, 5)
                plt.text(-0.25, 1.1, f'i)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
                plt.title(r'Clustered particle embedding', fontsize=12)
                for n in range(n_particle_types):
                    pos = np.argwhere(new_labels == n).squeeze().astype(int)
                    plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=6)
                plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
                plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
                plt.xticks(fontsize=10.0)
                plt.yticks(fontsize=10.0)
                plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

                ax = fig.add_subplot(3, 4, 6)
                print('10')
                plt.text(-0.25, 1.1, f'j)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
                plt.title(r'Interaction functions (model)', fontsize=12)
                func_list = []
                for n in range(n_particle_types):
                    pos = np.argwhere(new_labels == n).squeeze().astype(int)
                    embedding_1 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                     device=device)
                    for m in range(n_particle_types):
                        pos = np.argwhere(new_labels == m).squeeze().astype(int)
                        embedding_2 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)

                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
                        with torch.no_grad():
                            func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        func_list.append(func)
                        plt.plot(to_numpy(rr),
                                 to_numpy(func) * to_numpy(ynorm),
                                 color=cmap.color(n), linewidth=1)
                plt.xlabel(r'$r_{ij}$', fontsize=12)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij})$', fontsize=12)
                plt.xticks(fontsize=10.0)
                plt.yticks(fontsize=10.0)
                plt.ylim([-0.15, 0.15])
                plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

                ax = fig.add_subplot(3, 4, 7)
                print('11')
                plt.text(-0.25, 1.1, f'k)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
                plt.title(r'Interaction functions (true)', fontsize=12)
                p = config.simulation.params
                p = torch.ones(n_particle_types, n_particle_types, 4, device=device)
                params = config.simulation.params
                if params[0] != [-1]:
                    for n in range(n_particle_types):
                        for m in range(n_particle_types):
                            p[n, m] = torch.tensor(params[n * 3 + m])
                for n in range(n_particle_types):
                    for m in range(n_particle_types):
                        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n, m], p[n, m])), color=cmap.color(n), linewidth=1)
                plt.xlabel(r'$r_{ij}$', fontsize=12)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, r_{ij})$', fontsize=12)
                plt.xticks(fontsize=10.0)
                plt.yticks(fontsize=10.0)
                plt.ylim([-0.15, 0.15])
                plt.tight_layout()

                plt.savefig(f"./{log_dir}/tmp_training/Fig2_{dataset_name}_{epoch}.tif")

                plt.close()


                fig = plt.figure(figsize=(12, 12))
                func_list = []
                for n in range(n_particle_types):
                    pos = np.argwhere(new_labels == n).squeeze().astype(int)
                    embedding_1 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                     device=device)
                    for m in range(n_particle_types):
                        pos = np.argwhere(new_labels == m).squeeze().astype(int)
                        embedding_2 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)

                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
                        with torch.no_grad():
                            func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        func_list.append(func)
                        plt.plot(to_numpy(rr),
                                 to_numpy(func) * to_numpy(ynorm),
                                 color=cmap.color(n), linewidth=3)
                plt.xticks([])
                plt.yticks([])
                plt.ylim([-0.15, 0.15])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif")
                plt.close()

                fig = plt.figure(figsize=(12, 12))
                for n in range(n_particle_types):
                    pos = np.argwhere(new_labels == n).squeeze().astype(int)
                    plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=100)
                plt.tight_layout()
                plt.xticks([])
                plt.yticks([])
                plt.close()
                plt.savefig(f"./{log_dir}/tmp_training/embedding_last_{dataset_name}_{epoch}.tif")


    else:

        for epoch in trange(n_epochs + 1):

            net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_{epoch}.pt"
            state_dict = torch.load(net,map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])

            fig = plt.figure(figsize=(24, 5))

            # white background
            # plt.style.use('classic')

            ax = fig.add_subplot(1, 5, 1)
            plt.plot(list_loss[0:epoch+1], color='k')
            plt.xlim([0, n_epochs])
            plt.ylim([0, 0.0025])
            plt.ylabel('loss', fontsize=14)
            plt.xlabel('epochs', fontsize=14)

            ax = fig.add_subplot(1, 5, 2)
            embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)


            if config.data_folder_name == 'graphs_data/solar_system':
                for n in range(n_particle_types):
                    plt.scatter(embedding[index_particles[n], 0],
                                embedding[index_particles[n], 1], color=cmap.color(n), s=10)
                object_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune',
                               'pluto', 'io', 'europa',
                               'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea', 'titan', 'hyperion',
                               'moon', 'phobos', 'deimos', 'charon']
                for n in range(10):
                    plt.text(embedding[index_particles[n], 0], embedding[index_particles[n], 1], object_list[n])
            else:
                for n in range(n_particle_types):
                    plt.scatter(embedding[index_particles[n], 0],
                                embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)


            ax = fig.add_subplot(1, 5, 3)
            if (simulation_config.n_interactions < 100) & (simulation_config.has_cell_division == False) :  # cluster embedding
                if model_config.particle_model_name == 'PDE_E':
                    func_list = []
                    rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                    for n in range(n_particles):
                            embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                            in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                            func = model.lin_edge(in_features.float())
                            func = func[:, 0]
                            func_list.append(func)
                            plt.plot(to_numpy(rr),
                                     to_numpy(func) * to_numpy(ynorm),
                                     linewidth=1,
                                     color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
                    func_list = torch.stack(func_list)
                    plt.xlim([0, 0.05])
                    plt.xlabel('Distance [a.u]', fontsize=14)
                    plt.ylabel('MLP [a.u]', fontsize=14)
                    coeff_norm = to_numpy(func_list)
                    trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
                    proj_interaction = trans.transform(coeff_norm)
                elif model_config.particle_model_name == 'PDE_G':
                    func_list = []
                    rr = torch.tensor(np.linspace(0, radius * 1.3, 1000)).to(device)
                    for n in range(n_particles):
                        embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        func_list.append(func)
                        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(x[n, 5]).astype(int)), linewidth=1, alpha=0.25)
                    func_list = torch.stack(func_list)
                    plt.yscale('log')
                    plt.xscale('log')
                    plt.xlim([1E-3, 0.2])
                    plt.xlabel('Distance [a.u]', fontsize=14)
                    plt.ylabel('MLP [a.u]', fontsize=14)
                    coeff_norm = to_numpy(func_list)
                    trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
                    proj_interaction = trans.transform(coeff_norm)
                elif model_config.particle_model_name == 'PDE_GS':
                    func_list = []
                    rr = torch.tensor(np.linspace(0, radius * 1.3, 1000)).to(device)
                    for n in range(n_particles):
                        embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        func_list.append(func)
                        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(x[n, 5]).astype(int)), linewidth=1)
                    func_list = torch.stack(func_list)
                    plt.xlabel('Distance [a.u]', fontsize=14)
                    plt.ylabel('MLP [a.u]', fontsize=14)
                    coeff_norm = to_numpy(func_list)
                    trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
                    proj_interaction = trans.transform(coeff_norm)
                elif (model_config.particle_model_name == 'PDE_A') | (model_config.particle_model_name == 'PDE_B'):
                    func_list = []
                    rr = torch.tensor(np.linspace(0, 0.075, 200)).to(device)
                    for n in range(n_particles):
                        embedding_ = model.a[1, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
                        if model_config.particle_model_name == 'PDE_A':
                            in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                     rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                        else:
                            in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                     rr[:, None] / simulation_config.max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        func_list.append(func)
                        if n % 5 == 0:
                            plt.plot(to_numpy(rr),
                                     to_numpy(func) * to_numpy(ynorm),
                                     color=cmap.color(to_numpy(x[n, 5]).astype(int)), linewidth=1, alpha=0.25)
                    plt.xlabel('distance', fontsize=14)
                    plt.ylabel('interaction', fontsize=14)
                    func_list = torch.stack(func_list)
                    coeff_norm = to_numpy(func_list)
                    new_index = np.random.permutation(coeff_norm.shape[0])
                    new_index = new_index[0:min(1000, coeff_norm.shape[0])]
                    trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm[new_index])
                    proj_interaction = trans.transform(coeff_norm)
                elif has_mesh:
                    func_list = []
                    popt_list = []
                    for n in range(n_particles):
                        embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                        if model_config.mesh_model_name == 'RD_RPS_Mesh':
                            embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                            u = torch.tensor(np.linspace(0, 1, 100)).to(device)
                            u = u[:, None]
                            r = u
                            in_features = torch.cat((u, u, u, u, u, u, embedding_), dim=1)
                            h = model.lin_phi(in_features.float())
                            h = h[:, 0]
                        else:
                            r = torch.tensor(np.linspace(-150, 150, 100)).to(device)
                            in_features = torch.cat((r[:, None], embedding_), dim=1)
                            h = model.lin_phi(in_features.float())
                            popt, pcov = curve_fit(linear_model, to_numpy(r.squeeze()), to_numpy(h.squeeze()))
                            popt_list.append(popt)
                            h = h[:, 0]
                        func_list.append(h)
                        if (n % 24):
                            plt.plot(to_numpy(r),
                                     to_numpy(h) * to_numpy(hnorm), linewidth=1,
                                     color='k', alpha=0.05)
                    func_list = torch.stack(func_list)
                    coeff_norm = to_numpy(func_list)
                    popt_list = np.array(popt_list)

                    if model_config.mesh_model_name == 'RD_RPS_Mesh':
                        trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm)
                        proj_interaction = trans.transform(coeff_norm)
                    else:
                        proj_interaction = popt_list
                        proj_interaction[:, 1] = proj_interaction[:, 0]
                # plt.xlim([0,0.075])
                # plt.ylim([-0.04,0.03])

                ax = fig.add_subplot(1, 5, 4)
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

                plt.xlabel('UMAP 0', fontsize=14)
                plt.ylabel('UMAP 1', fontsize=14)
                plt.xlim([-60, 60])
                plt.ylim([-60, 60])

                new_labels = labels.copy()
                for n in range(n_particle_types):
                    new_labels[labels == label_list[n]] = n
                    pos = np.argwhere(labels == label_list[n])
                    pos = np.array(pos)
                    # if pos.size>0:
                        # plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                        #             color=cmap.color(n), s=0.1)
                # Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)



                ax = fig.add_subplot(1, 5, 5)
                model_a_ = model.a.clone().detach()
                model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
                for n in range(n_clusters):
                    pos = np.argwhere(labels == n).squeeze().astype(int)
                    pos = np.array(pos)
                    if pos.size > 0:
                        median_center = model_a_[pos, :]
                        median_center = torch.median(median_center, dim=0).values
                        plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                        model_a_[pos, :] = median_center
                        plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
                model_a_ = torch.reshape(model_a_, (model.a.shape[0], model.a.shape[1], model.a.shape[2]))
                for n in np.unique(new_labels):
                    pos = np.argwhere(new_labels == n).squeeze().astype(int)
                    pos = np.array(pos)
                    if pos.size>0:
                        plt.scatter(to_numpy(model_a_[0, pos, 0]), to_numpy(model_a_[0, pos, 1]), color='k', s=5)
                plt.xlabel('dim 0', fontsize=14)
                plt.ylabel('dim 1', fontsize=14)
                plt.xticks(fontsize=10.0)
                plt.yticks(fontsize=10.0)
                plt.xlim([-1, 2])
                plt.ylim([-1, 2])


                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/new_Fig_{dataset_name}_{epoch}.tif")
                plt.close()


if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')

    config_list =['boids_16']  # ['arbitrary_3_dropout_40_pos','arbitrary_3_dropout_50_pos'] # ['arbitrary_3_3', 'arbitrary_3', 'gravity_16']

    for config_file in config_list:

        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        cmap = CustomColorMap(config=config)  # create colormap for given model_config

        data_plot_training(config, mode='figures' , device=device)



