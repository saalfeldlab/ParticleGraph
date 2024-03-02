import glob
import logging
import time
from shutil import copyfile

import imageio.v2
import matplotlib.pyplot as plt
import networkx as nx
import torch
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
import matplotlib
from matplotlib import rc
# matplotlib.use("Qt5Agg")

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.particle_initialization import init_particles, init_mesh
from ParticleGraph.generators.utils import choose_model, choose_mesh_model
from ParticleGraph.train_utils import choose_training_model, constant_batch_size, increasing_batch_size, \
    set_trainable_parameters, get_embedding, set_trainable_division_parameters

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.data_loaders import *
from ParticleGraph.utils import to_numpy, CustomColorMap, set_device, norm_velocity, norm_acceleration, grads2D
from ParticleGraph.fitting_models import linear_model
from ParticleGraph.embedding_cluster import *
from ParticleGraph.models import division_predictor


def data_generate(config, visualize=True, style='color', erase=False, step=5, alpha=0.2, ratio=1, scenario='none', device=None, bSave=True):
    print('')

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    # create output folder, empty it if bErase=True, copy files into it
    dataset_name = config.dataset
    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/generated_data/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/generated_data/*')
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    # load model parameters and create local varibales
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training
    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_frames = simulation_config.n_frames

    delta_t = simulation_config.delta_t
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_cell_division = simulation_config.has_cell_division
    has_dropout = training_config.dropout > 0

    index_particles = []
    np_i = int(n_particles / n_particle_types)
    for n in range(n_particle_types):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if has_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1-training_config.dropout))
        dropout_mask = draw[0:cut]
        inv_dropout_mask = draw[cut:]
    else:
        dropout_mask = np.arange(n_particles)


    model, bc_pos, bc_dpos = choose_model(config, device=device)

    if has_mesh:
        mesh_model = choose_mesh_model(config, device=device)
    else:
        mesh_model = None
    torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_{dataset_name}/model.pt')

    cycle_length = None

    ################################################################################
    ##### Main loop ################################################################
    ################################################################################

    for run in range(config.training.n_runs):

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        x_mesh_list = []
        y_mesh_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1, cycle_length, cycle_length_distrib = init_particles(config, device=device, cycle_length=cycle_length)

        # create differnet initial conditions
        # X1[:, 0] = X1[:, 0] / n_particle_types - torch.ones_like(X1[:,0])*0.5
        # for n in range(n_particle_types):
        #     X1[index_particles[n], 0] = X1[index_particles[n], 0] + n / n_particle_types

        if has_mesh:
            X1_mesh, V1_mesh, T1_mesh, H1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)
            torch.save(mesh_data, f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt')
            mask_mesh = mesh_data['mask'].squeeze()

            plt.scatter(to_numpy(X1_mesh[:, 0]), to_numpy(X1_mesh[:, 1]), c=to_numpy(T1_mesh[:, 0]))
            plt.show
        if only_mesh | (model_config.particle_model_name == 'PDE_O'):
            X1 = X1_mesh.clone().detach()
            H1 = H1_mesh.clone().detach()
            T1 = T1_mesh.clone().detach()

        index_particles = []
        for n in range(n_particle_types):
            pos = torch.argwhere(T1 == n)
            pos = to_numpy(pos[:, 0].squeeze()).astype(int)
            index_particles.append(pos)

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            # calculate cell division
            if (it >=0) & has_cell_division & (n_particles < 20000):
                pos = torch.argwhere(A1.squeeze() > cycle_length_distrib)
                y_division = (A1.squeeze() > cycle_length_distrib).clone().detach()*1.0
                # cell division
                if len(pos) > 1:
                    n_add_nodes = len(pos)
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)

                    y_division = torch.concatenate((y_division,torch.zeros((n_add_nodes),device=device)),0)

                    n_particles = n_particles + n_add_nodes
                    N1 = torch.arange(n_particles, device=device)
                    N1 = N1[:, None]
                    separation = 1E-3 * torch.randn((n_add_nodes, 2), device=device)
                    X1 = torch.cat((X1, X1[pos, :] + separation), dim=0)
                    X1[pos, :] = X1[pos, :] - separation
                    phi = torch.randn(n_add_nodes, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                    cos_phi = torch.cos(phi)
                    sin_phi = torch.sin(phi)
                    new_x = cos_phi * V1[pos, 0] + sin_phi * V1[pos, 1]
                    new_y = -sin_phi * V1[pos, 0] + cos_phi * V1[pos, 1]
                    V1[pos, 0] = new_x
                    V1[pos, 1] = new_y
                    V1 = torch.cat((V1, -V1[pos, :]), dim=0)
                    T1 = torch.cat((T1, T1[pos, :]), dim=0)
                    H1 = torch.cat((H1, H1[pos, :]), dim=0)
                    A1[pos, :] = 0
                    A1 = torch.cat((A1, A1[pos, :]), dim=0)
                    nd = torch.ones(len(pos), device=device) + 0.05 * torch.randn(len(pos), device=device)
                    cycle_length_distrib = torch.cat((cycle_length_distrib, cycle_length[to_numpy(T1[pos, 0])].squeeze() * nd),dim=0)
                    y_timer = A1.squeeze().clone().detach()

                    index_particles = []
                    for n in range(n_particles):
                        pos = torch.argwhere(T1 == n)
                        pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                        index_particles.append(pos)

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach()), 1)
            if has_mesh:
                x_mesh = torch.concatenate((N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(), 
                                            T1_mesh.clone().detach(), H1_mesh.clone().detach()), 1)
                dataset_mesh = data.Data(x=x_mesh, edge_index=mesh_data['edge_index'], edge_attr=mesh_data['edge_weight'], device=device)
 
            # compute connectivity rule
            distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
            adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

            # model prediction
            with torch.no_grad():
                y = model(dataset)

            # append list
            if (it >= 0) & bSave:

                if has_cell_division:
                    x_list.append(x.clone().detach())
                    y_ = torch.concatenate((y,y_timer[:, None], y_division[:, None]), 1)
                    y_list.append(y_.clone().detach())
                else:
                    if has_dropout:
                        x_ = x[dropout_mask].clone().detach()
                        x_[:,0] = torch.arange(len(x_), device=device)
                        x_list.append(x_)
                        y_list.append(y[dropout_mask].clone().detach())
                    else:
                        x_list.append(x.clone().detach())
                        y_list.append(y.clone().detach())
                        if (run==1) & (it==200):
                            torch.save(x, f'graphs_data/graphs_{dataset_name}/x_200.pt')
                            torch.save(y, f'graphs_data/graphs_{dataset_name}/y_200.pt')


            # Particle update
            if model_config.particle_model_name == 'PDE_O':
                H1[:, 2] = H1[:, 2] + y.squeeze() * delta_t
                X1[:, 0] = H1[:, 0] + (3/8) * mesh_data['size'] * torch.cos(H1[:, 2])
                X1[:, 1] = H1[:, 1] + (3/8) * mesh_data['size'] * torch.sin(H1[:, 2])
                X1 = bc_pos(X1)
            else:
                if model_config.prediction == '2nd_derivative':
                    V1 += y * delta_t
                else:
                    V1 = y
                X1 = bc_pos(X1 + V1 * delta_t)
                if config.graph_model.mesh_model_name=='Chemotaxism_Mesh':
                    grad = grads2D(torch.reshape(H1_mesh[:, 0], (300, 300)))
                    x_ = np.clip(to_numpy(X1[:, 0]) * 300, 0, 299)
                    y_ = np.clip(to_numpy(X1[:, 1]) * 300, 0, 299)
                    X1[:,0] += torch.clamp(grad[1][y_, x_] / 5E4, min=-0.5, max=0.5)
                    X1[:,1] += torch.clamp(grad[0][y_, x_] / 5E4, min=-0.5, max=0.5)

            A1 = A1 + delta_t

            # Mesh update
            if has_mesh:
                x_mesh_list.append(x_mesh.clone().detach())
                match config.graph_model.mesh_model_name:
                    case 'DiffMesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                            H1[mesh_data['mask'].squeeze(), 1:2] = pred[mask]
                        H1_mesh[mask_mesh, 0:1] += pred[:] * delta_t
                        new_pred = torch.zeros_like(pred)
                        new_pred[mask_mesh] = pred[mask_mesh]
                        pred = new_pred
                    case 'WaveMesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                        H1_mesh[mask_mesh, 1:2] += pred[mask_mesh,:] * delta_t
                        H1_mesh[mask_mesh, 0:1] += H1_mesh[mask_mesh, 1:2] * delta_t
                            # x_ = to_numpy(x_mesh)
                            # plt.scatter(x_[:, 1], x_[:, 2], c=to_numpy(H1_mesh[:, 0]))
                    case 'RD_Gray_Scott_Mesh' | 'RD_FitzHugh_Nagumo_Mesh' | 'RD_RPS_Mesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                            H1_mesh[mesh_data['mask'].squeeze(), :] += pred[mesh_data['mask'].squeeze(), :] * delta_t
                            H1 = H1_mesh.clone().detach()
                    case 'Chemotaxism_Mesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                            H1_mesh[mesh_data['mask'].squeeze(), :] += pred[mesh_data['mask'].squeeze(), :] * delta_t
                            distance = torch.sum(bc_dpos(x[:, None, 1:3] - x_mesh[None, :, 1:3]) ** 2, dim=2)
                            distance = distance < 0.00015
                            distance = torch.sum(distance, dim=0)
                            H1_mesh = torch.relu(H1_mesh*1.0012 - 2*distance[:,None])
                            H1_mesh = torch.clamp(H1_mesh, min=0, max=5000)
                    case 'PDE_O_Mesh':
                        pred=[]

                y_mesh_list.append(pred)

            # output plots
            if visualize & (run == 0) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')

                if 'graph' in style:
                    fig = plt.figure(figsize=(10, 10))

                    distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                    adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                    edge_index2 = adj_t2.nonzero().t().contiguous()
                    dataset2 = data.Data(x=x, edge_index=edge_index2)
                    pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                    vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=alpha)

                    if model_config.particle_model_name == 'PDE_G':
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=40, color=cmap.color(n))
                    elif has_mesh:
                        pts = x[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                        if model_config.mesh_model_name == 'WaveMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                          vmax=2500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=2500)
                    elif model_config.particle_model_name == 'PDE_E':
                        for n in range(n_particle_types):
                            g = 40
                            if simulation_config.params[n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n),
                                        alpha=0.5)
                    if has_mesh | (simulation_config.boundary == 'periodic'):
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    else:
                        plt.xlim([-0.5, 0.5])
                        plt.ylim([-0.5, 0.5])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_g_color_{it}.tif", dpi=300)
                    plt.close()

                if 'color' in style:

                    if model_config.particle_model_name == 'PDE_O':
                        fig = plt.figure(figsize=(12, 12))
                        plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=100,
                                    c=np.sin(to_numpy(H1[:, 2])), vmin=-1, vmax=1, cmap='viridis')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Lut_Fig_{it}.jpg", dpi=170.7)
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        # plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=5, c='b')
                        plt.scatter(X1[:, 0].detach().cpu().numpy(), X1[:, 1].detach().cpu().numpy(), s=10, c='lawngreen',
                                    alpha=0.75)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Rot_Fig{it}.jpg", dpi=170.7)
                        plt.close()

                    elif model_config.mesh_model_name == 'Chemotaxism_Mesh':

                        # dx_ = to_numpy(grad[1][y_,x_])/100
                        # dy_ = to_numpy(grad[0][y_,x_])/100
                        # H1_IM = torch.reshape(H1_mesh[:, 0], (300, 300))
                        # plt.imshow(to_numpy(grad[1]+grad[0]), cmap='viridis')
                        # for i in range(1700):
                        #     plt.arrow(x=x_[i],y=y_[i],dx=dx_[i],dy=dy_[i], head_width=2, length_includes_head=True, color='w')

                        fig = plt.figure(figsize=(12,12))
                        H1_IM = torch.reshape(H1_mesh[:, 0], (300, 300))
                        plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=5000, cmap='viridis')
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy()*300,
                                        x[index_particles[n], 2].detach().cpu().numpy()*300, s=1, color='w')
                        plt.xlim([0, 300])
                        plt.ylim([0, 300])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/All_{it}.jpg", dpi=170.7)
                        plt.close()

                        # fig = plt.figure(figsize=(12,12))
                        # H1_IM = torch.reshape(distance, (300, 300))
                        # plt.imshow(H1_IM.detach().cpu().numpy()*30, vmin=0, vmax=500)
                        # for n in range(n_particle_types):
                        #     plt.scatter(x[index_particles[n], 1].detach().cpu().numpy() * 300,
                        #                 x[index_particles[n], 2].detach().cpu().numpy() * 300, s=1, color='w')
                        # plt.xlim([0, 300])
                        # plt.ylim([0, 300])
                        # plt.xticks([])
                        # plt.yticks([])
                        # plt.tight_layout()
                        # plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Boids_{it}.jpg", dpi=170.7)
                        # plt.close()

                    else:

                        fig = plt.figure(figsize=(12, 12))
                        if has_mesh:
                            pts = x_mesh[:, 1:3].detach().cpu().numpy()
                            tri = Delaunay(pts)
                            colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                            match model_config.mesh_model_name:
                                case 'DiffMesh':
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                                case 'WaveMesh':
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                                case 'RD_Gray_Scott_Mesh':
                                    fig = plt.figure(figsize=(12, 6))
                                    ax = fig.add_subplot(1, 2, 1)
                                    colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                                    plt.xticks([])
                                    plt.yticks([])
                                    plt.axis('off')
                                    ax = fig.add_subplot(1, 2, 2)
                                    colors = torch.sum(x[tri.simplices, 7], dim=1) / 3.0
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                                    plt.xticks([])
                                    plt.yticks([])
                                    plt.axis('off')
                                case 'RD_RPS_Mesh':
                                    fig = plt.figure(figsize=(12, 12))
                                    H1_IM = torch.reshape(H1, (100, 100, 3))
                                    plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                                    plt.xticks([])
                                    plt.yticks([])
                                    plt.axis('off')
                        else:
                            s_p = 25
                            if simulation_config.has_cell_division:
                                s_p = 10
                            for n in range(n_particle_types):
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
                            if training_config.dropout>0:
                                plt.scatter(x[inv_dropout_mask, 1].detach().cpu().numpy(), x[inv_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k', alpha=0.75)
                                plt.plot(x[inv_dropout_mask, 1].detach().cpu().numpy(), x[inv_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')


                        if (has_mesh | (simulation_config.boundary == 'periodic')):
                            if (model_config.mesh_model_name != 'RD_RPS_Mesh'):
                                plt.xlim([0, 1])
                                plt.ylim([0, 1])
                        else:
                            plt.xlim([-4, 4])
                            plt.ylim([-4, 4])

                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{it}.jpg", dpi=170.7)
                        plt.close()

                        if not(has_mesh):
                            fig = plt.figure(figsize=(12, 12))
                            s_p = 25
                            if simulation_config.has_cell_division:
                                s_p = 10
                            for n in range(n_particle_types):
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=s_p, color='w')
                            if (simulation_config.boundary == 'periodic'):
                                plt.xlim([0, 1])
                                plt.ylim([0, 1])
                            else:
                                plt.xlim([-4, 4])
                                plt.ylim([-4, 4])
                            plt.xticks([])
                            plt.yticks([])
                            plt.tight_layout()
                            plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_bw_{it}.jpg", dpi=170.7)
                            plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(x_mesh_list, f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt')
            torch.save(y_mesh_list, f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt')

    # x_ = torch.load(f'graphs_data/graphs_{dataset_name}/x_200.pt', map_location=device)
    # y_ = torch.load(f'graphs_data/graphs_{dataset_name}/y_200.pt', map_location=device)

    if bSave:
        torch.save(cycle_length, f'graphs_data/graphs_{dataset_name}/cycle_length.pt')
        torch.save(cycle_length_distrib, f'graphs_data/graphs_{dataset_name}/cycle_length_distrib.pt')



def data_train(config):
    print('')

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    n_epochs = train_config.n_epochs
    radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
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
    # net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_6.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr_embedding = train_config.learning_rate_embedding_start
    lr = train_config.learning_rate_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')

    if  has_cell_division:
        model_division = division_predictor(config, device)
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
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')

    config.simulation.n_particles = n_particles
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    model.train()

    model_forward, bc_pos, bc_dpos = choose_model(config, device=device)

    list_loss = []
    time.sleep(0.5)

    for epoch in range(n_epochs + 1):

        old_batch_size = batch_size
        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        if epoch == 1:
            repeat_factor = batch_size // old_batch_size
            if has_mesh:
                mask_mesh = mask_mesh.repeat(repeat_factor, 1)

        total_loss = 0
        total_loss_division = 0

        Niter = n_frames * data_augmentation_loop // batch_size
        if (has_mesh) & (batch_size == 1):
            Niter = Niter // 4

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            time_batch=[]

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)
                x = x_list[run][k].clone().detach()

                if has_mesh:
                    x_mesh = x_mesh_list[run][k].clone().detach()
                    if train_config.noise_level > 0:
                        x_mesh[:, 6:7] = x_mesh[:, 6:7] + train_config.noise_level * torch.randn_like(x_mesh[:, 6:7])
                    dataset = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
                    dataset_batch.append(dataset)
                    y = y_mesh_list[run][k].clone().detach() / hnorm
                    if batch == 0:
                        y_batch = y
                    else:
                        y_batch = torch.cat((y_batch, y), dim=0)
                else:

                    distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                    adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    t = torch.Tensor([radius ** 2])
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)
                    dataset_batch.append(dataset)
                    y = y_list[run][k].clone().detach()
                    if model_config.prediction == '2nd_derivative':
                        y = y / ynorm
                    else:
                        y = y / vnorm
                    if data_augmentation:
                        new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                        new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                        y[:, 0] = new_x
                        y[:, 1] = new_y
                    if batch == 0:
                        y_batch = y[:, 0:2]
                    else:
                        y_batch = torch.cat((y_batch, y[:,0:2]), dim=0)

                    if has_cell_division:
                        if batch == 0:
                            time_batch = torch.concatenate( (x[:, 0:1], torch.ones_like(y[:, 3:4], device=device) * k), dim = 1)
                            y_batch_division = y[:, 2:3]
                        else:
                            time_batch = torch.concatenate((time_batch, torch.concatenate((x[:, 0:1], torch.ones_like(y[:, 2:3], device=device) * k), dim=1)), dim=0)
                            y_batch_division = torch.concatenate((y_batch_division, y[:, 3:4]), dim=0)


            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                if has_mesh:
                    pred = model(batch, data_id=run - 1)
                else:
                    pred = model(batch, data_id=run - 1, training=True, vnorm=vnorm, phi=phi)
            if has_cell_division:
                optimizer_division.zero_grad()

                pred_division = model_division(time_batch, data_id=run - 1)
                loss_division = (pred_division - y_batch_division).norm(2)
                loss_division.backward()
                optimizer_division.step()
                total_loss_division += loss_division.item()

            if model_config.mesh_model_name == 'RD_RPS_Mesh':
                loss = ((pred - y_batch) * mask_mesh).norm(2)
            else:
                loss = (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            visualize_embedding=False
            if visualize_embedding & ( (epoch == 0) & (N < 100) & (N % 2 == 0)  |  (epoch==0)&(N<10000) & (N%200==0)  |  (epoch==0)&(N%(Niter//100)==0)   | (epoch>0)&(N%(Niter//4)==0)):

                if model_config.mesh_model_name == 'WaveMesh':
                    rr = torch.tensor(np.linspace(-150, 150, 200)).to(device)
                else:
                    rr = torch.tensor(np.linspace(0, radius, 200)).to(device)
                popt_list = []
                for n in range(n_particles):
                    embedding_ = model.a[0, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
                    if (model_config.particle_model_name == 'PDE_A') :
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                    elif (model_config.particle_model_name == 'PDE_B'):
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    elif model_config.particle_model_name == 'PDE_E':
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                    elif model_config.mesh_model_name == 'WaveMesh':
                        in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    else:
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)

                    if model_config.mesh_model_name == 'WaveMesh':
                        h = model.lin_phi(in_features.float())
                        h = h[:, 0]
                        popt, pcov = curve_fit(linear_model, to_numpy(rr.squeeze()), to_numpy(h.squeeze()))
                        popt_list.append(popt)
                        # if n % 100 == 0:
                        #      plt.scatter(to_numpy(rr), to_numpy(h) * to_numpy(hnorm) * 100, c='k', s=0.01)
                    else:
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        if n % 5 == 0:
                            plt.plot(to_numpy(rr),
                                     to_numpy(func) * to_numpy(ynorm),
                                     linewidth=1,
                                     color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)

                # plt.xlim([-150,150])
                # plt.ylim([-150,150])
                #
                # plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{dataset_name}_function_{epoch}_{N}.tif",dpi=300)
                # plt.close()

                if model_config.mesh_model_name == 'WaveMesh':
                    fig = plt.figure(figsize=(8, 8))
                    embedding, embedding_particle = get_embedding(model.a, index_particles, n_particles,
                                                                  n_particle_types)
                    plt.scatter(embedding[:, 0], embedding[:, 1], c=t, s=3, cmap='viridis')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{dataset_name}_embedding_{epoch}_{N}.tif",
                                dpi=300)
                    plt.close()

                    fig = plt.figure(figsize=(8, 8))
                    t = np.array(popt_list)
                    t = t[:, 0]
                    t = np.reshape(t, (100, 100))
                    plt.imshow(t, cmap='viridis')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{dataset_name}_map_{epoch}_{N}.tif",
                                dpi=300)

                    # imageio.imwrite(f"./{log_dir}/tmp_training/embedding/{dataset_name}_map_{epoch}_{N}.tif", t, 'TIFF')

                    fig = plt.figure(figsize=(8, 8))
                    t = np.array(popt_list)
                    t = t[:, 0]
                    pts = x_mesh[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = np.sum(t[tri.simplices],axis=1)
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(), facecolors=colors)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{dataset_name}_delaunay_{epoch}_{N}.tif",
                                dpi=300)
                    plt.close()



        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))
        if has_cell_division:
            print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss_division / (N + 1) / n_particles / batch_size))
            logger.info("Epoch {}. Division Loss: {:.6f}".format(epoch, total_loss_division / (N + 1) / n_particles / batch_size))
            torch.save({'model_state_dict': model_division.state_dict(),
                        'optimizer_state_dict': optimizer_division.state_dict()}, os.path.join(log_dir, 'models', f'best_model_division_with_{NGraphs - 1}_graphs_{epoch}.pt'))


        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)

        fig = plt.figure(figsize=(22, 4))
        # white background
        # plt.style.use('classic')
        ax = fig.add_subplot(1, 6, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding, embedding_particle = get_embedding(model.a, index_particles, n_particles, n_particle_types)
        if (embedding.shape[1] > 2):
            ax = fig.add_subplot(2, 4, 2, projection='3d')
            for n in range(n_particle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],
                           color=cmap.color(n), s=1)
        else:
            if (embedding.shape[1] > 1):
                for m in range(model.a.shape[0]):
                    for n in range(n_particle_types):
                        if simulation_config.has_cell_division:
                            plt.scatter(embedding_particle[n + m * n_particle_types][:, 0],
                                        embedding_particle[n + m * n_particle_types][:, 1], color='k', s=3)
                        else:
                            plt.scatter(embedding_particle[n + m * n_particle_types][:, 0],
                                    embedding_particle[n + m * n_particle_types][:, 1], color=cmap.color(n), s=3)
                plt.xlabel('Embedding 0', fontsize=12)
                plt.ylabel('Embedding 1', fontsize=12)
            else:
                for n in range(n_particle_types):
                    plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5, color=cmap.color(n))

        ax = fig.add_subplot(1, 6, 3)
        if (simulation_config.n_interactions < 100) & (simulation_config.has_cell_division == False) :  # cluster embedding
            if model_config.particle_model_name == 'PDE_E':
                func_list = []
                rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                for n in range(n_particles):
                        embedding_ = model.a[0, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
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
                plt.xlabel('Distance [a.u]', fontsize=12)
                plt.ylabel('MLP [a.u]', fontsize=12)
                coeff_norm = to_numpy(func_list)
                trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
                proj_interaction = trans.transform(coeff_norm)
            elif model_config.particle_model_name == 'PDE_G':
                func_list = []
                rr = torch.tensor(np.linspace(0, radius * 1.3, 1000)).to(device)
                for n in range(n_particles):
                    embedding_ = model.a[0, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
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
                plt.xlabel('Distance [a.u]', fontsize=12)
                plt.ylabel('MLP [a.u]', fontsize=12)
                coeff_norm = to_numpy(func_list)
                trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
                proj_interaction = trans.transform(coeff_norm)
            elif (model_config.particle_model_name == 'PDE_A') | (model_config.particle_model_name == 'PDE_B'):
                func_list = []
                rr = torch.tensor(np.linspace(0, radius, 200)).to(device)
                for n in range(n_particles):
                    embedding_ = model.a[0, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
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
                plt.xlabel('Distance [a.u]', fontsize=12)
                plt.ylabel('MLP [a.u]', fontsize=12)
                func_list = torch.stack(func_list)
                coeff_norm = to_numpy(func_list)
                new_index = np.random.permutation(coeff_norm.shape[0])
                new_index = new_index[0:min(1000, coeff_norm.shape[0])]
                trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm[new_index])
                proj_interaction = trans.transform(coeff_norm)
            elif has_mesh:
                f_list = []
                popt_list = []
                for n in range(n_particles):
                    embedding_ = model.a[0, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                    if model_config.mesh_model_name == 'RD_RPS_Mesh':
                        embedding_ = model.a[0, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
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
                    f_list.append(h)
                    if (n % 24):
                        plt.plot(to_numpy(r),
                                 to_numpy(h) * to_numpy(hnorm), linewidth=1,
                                 color='k', alpha=0.05)
                f_list = torch.stack(f_list)
                coeff_norm = to_numpy(f_list)
                popt_list = np.array(popt_list)

                if model_config.mesh_model_name == 'RD_RPS_Mesh':
                    trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm)
                    proj_interaction = trans.transform(coeff_norm)
                else:
                    proj_interaction = popt_list
                    proj_interaction[:, 1] = proj_interaction[:, 0]

            ax = fig.add_subplot(1, 6, 4)
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

            plt.xlabel('proj 0', fontsize=12)
            plt.ylabel('proj 1', fontsize=12)
            plt.text(0., 1.1, f'Nclusters: {n_clusters}', ha='left', va='top', transform=ax.transAxes)

            ax = fig.add_subplot(1, 6, 5)
            new_labels = labels.copy()
            for n in range(n_particle_types):
                new_labels[labels == label_list[n]] = n
                pos = np.argwhere(labels == label_list[n])
                pos = np.array(pos)
                if pos.size>0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                                color=cmap.color(n), s=0.1)
            Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
            plt.text(0, 1.1, f'Accuracy: {np.round(Accuracy, 3)}', ha='left', va='top', transform=ax.transAxes,
                     fontsize=10)
            print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')
            logger.info(f'Accuracy: {np.round(Accuracy, 3)}    n_clusters: {n_clusters}')

            ax = fig.add_subplot(1, 6, 6)
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
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)
            plt.xticks(fontsize=10.0)
            plt.yticks(fontsize=10.0)

            if (replace_with_cluster) & ((epoch == 1 * n_epochs // 4) | (epoch == 2 * n_epochs // 4) | (epoch == 3 * n_epochs // 4)):
                # Constrain embedding
                with torch.no_grad():
                    for n in range(model.a.shape[0]):
                        model.a[n] = model_a_[0].clone().detach()
                print(f'regul_embedding: replaced')
                logger.info(f'regul_embedding: replaced')
                plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
                if train_config.fix_cluster_embedding:
                    lr_embedding = 1E-5
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
            else:
                if (epoch > n_epochs - 3) & (replace_with_cluster):
                    lr_embedding = 1E-5
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
                elif epoch > 3 * n_epochs // 4 + 1:
                    lr_embedding = train_config.learning_rate_embedding_end
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                    lr = train_config.learning_rate_start
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()



def data_plot_training(config):
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
        model_division = division_predictor(config, device)
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

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    model.train()

    list_loss = [0.002002, 0.000676, 0.000481, 0.000385, 0.000334, 0.000299, 0.000144, 0.000268, 0.000249, 0.000232, 0.000218, 0.000108, 0.000203, 0.000192, 0.000184, 0.000177, 0.000088, 0.000168, 0.000086, 0.000084, 0.000082,0.000082,0.000082]
    time.sleep(0.5)

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
        embedding, embedding_particle = get_embedding(model.a, index_particles, n_particles, n_particle_types)
        if (embedding.shape[1] > 2):
            ax = fig.add_subplot(2, 4, 2, projection='3d')
            for n in range(n_particle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],
                           color=cmap.color(n), s=1)
        else:
            if (embedding.shape[1] > 1):
                for m in range(model.a.shape[0]):
                    for n in range(n_particle_types):
                        if simulation_config.has_cell_division:
                            plt.scatter(embedding_particle[n + m * n_particle_types][:, 0],
                                        embedding_particle[n + m * n_particle_types][:, 1], color='k', s=3)
                        else:
                            plt.scatter(embedding_particle[n + m * n_particle_types][:, 0],
                                    embedding_particle[n + m * n_particle_types][:, 1], color=cmap.color(n), s=3)
                plt.xlabel('dim 0', fontsize=14)
                plt.ylabel('dim 1', fontsize=14)
            else:
                for n in range(n_particle_types):
                    plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5, color=cmap.color(n))
        plt.xlim([-1, 2])
        plt.ylim([-1, 2])

        ax = fig.add_subplot(1, 5, 3)
        if (simulation_config.n_interactions < 100) & (simulation_config.has_cell_division == False) :  # cluster embedding
            if model_config.particle_model_name == 'PDE_E':
                func_list = []
                rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                for n in range(n_particles):
                        embedding_ = model.a[0, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
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
                    embedding_ = model.a[0, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
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
            elif (model_config.particle_model_name == 'PDE_A') | (model_config.particle_model_name == 'PDE_B'):
                func_list = []
                rr = torch.tensor(np.linspace(0, 0.075, 200)).to(device)
                for n in range(n_particles):
                    embedding_ = model.a[0, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
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
                f_list = []
                popt_list = []
                for n in range(n_particles):
                    embedding_ = model.a[0, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                    if model_config.mesh_model_name == 'RD_RPS_Mesh':
                        embedding_ = model.a[0, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
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
                    f_list.append(h)
                    if (n % 24):
                        plt.plot(to_numpy(r),
                                 to_numpy(h) * to_numpy(hnorm), linewidth=1,
                                 color='k', alpha=0.05)
                f_list = torch.stack(f_list)
                coeff_norm = to_numpy(f_list)
                popt_list = np.array(popt_list)

                if model_config.mesh_model_name == 'RD_RPS_Mesh':
                    trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm)
                    proj_interaction = trans.transform(coeff_norm)
                else:
                    proj_interaction = popt_list
                    proj_interaction[:, 1] = proj_interaction[:, 0]
            plt.xlim([0,0.075])
            plt.ylim([-0.04,0.03])

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



def data_test(config, visualize=False, verbose=True, best_model=0, step=5, forced_embedding=[], ratio=1):
    print('')
    print('Plot roll-out inference ... ')

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_division = simulation_config.has_cell_division

    print(f'Test data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    index_particles = []
    np_i = int(n_particles / n_particle_types)
    for n in range(n_particle_types):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))
    if best_model == -1:
        net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    else:
        net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"

    print('Graph files N: ', NGraphs - 1)
    print(f'network: {net}')

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    if has_mesh:
        mesh_model, bc_pos, bc_dpos  = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        mesh_model.load_state_dict(state_dict['model_state_dict'])
        mesh_model.eval()
    else:
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        mesh_model = None

    if has_division:
        model_division = division_predictor(config, device)
        net = f"./log/try_{dataset_name}/models/best_model_division_with_{NGraphs - 1}_graphs_20.pt"
        state_dict = torch.load(net, map_location=device)
        model_division.load_state_dict(state_dict['model_state_dict'])
        model_division.eval()

    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
        print('Use learned labels')
        labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))
    else:
        # labels = T1
        print('Use ground truth labels')

    # nparticles larger than initially
    if ratio > 1:

        prev_index_particles = index_particles

        new_nparticles = n_particles * ratio
        prev_nparticles = n_particles

        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio}')
        print('')

        embedding = model.a[0].data.clone().detach()
        new_embedding = []
        new_labels = []

        for n in range(n_particle_types):
            for m in range(ratio):
                if (n == 0) & (m == 0):
                    new_embedding = embedding[prev_index_particles[n].astype(int), :]
                    new_labels = labels[prev_index_particles[n].astype(int)]
                else:
                    new_embedding = torch.cat((new_embedding, embedding[prev_index_particles[n].astype(int), :]), dim=0)
                    new_labels = torch.cat((new_labels, labels[prev_index_particles[n].astype(int)]), dim=0)

        model.a = nn.Parameter(
            torch.tensor(np.ones((NGraphs - 1, int(prev_nparticles) * ratio, 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        model.a.requires_grad = False
        model.a[0] = new_embedding
        n_particles = new_nparticles

        index_particles = []
        np_i = int(n_particles / n_particle_types)
        for n in range(n_particle_types):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if verbose:
        print(table)
        print(f"Total Trainable Params: {total_params}")

    x_list = []
    y_list = []
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))

    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()

    if has_mesh:
        hnorm = torch.load(f'./log/try_{dataset_name}/hnorm.pt', map_location=device).to(device)

        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_0.pt',map_location=device)
        mask_mesh = mesh_data['mask']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']

        xy = to_numpy(mesh_data['mesh_pos'])
        x_ = xy[:,0]
        y_ = xy[:,1]
        mask = to_numpy(mask_mesh)

        mask_mesh = (x_ > np.min(x_)+0.02) & (x_ < np.max(x_)-0.02) & (y_ > np.min(y_)+0.02) & (y_ < np.max(y_)-0.02)
        mask_mesh = torch.tensor(mask_mesh, dtype=torch.bool, device=device)

        # plt.scatter(x_, y_, s=2, c=to_numpy(mask_mesh))

    if verbose:
        print('')
        print(f'x: {x.shape}')
        print(f'index_particles: {index_particles[0].shape}')
        print('')
    time.sleep(0.5)

    x = x_list[0][n_particles].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    if simulation_config.has_cell_division:
        cycle_length = torch.load(f'./graphs_data/graphs_{dataset_name}/cycle_length.pt', map_location=device).to(device)
        cycle_length_distrib = cycle_length[to_numpy(x[:,5]).astype(int)].squeeze()
        A1 = torch.rand(cycle_length_distrib.shape[0], device=device)
        A1 = A1  * cycle_length_distrib
        A1 = A1[:,None]

    time.sleep(1)
    for it in trange(n_frames - 1):

        x0 = x_list[0][it].clone().detach()

        if has_mesh:
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset_mesh = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)

        if model_config.mesh_model_name == 'DiffMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0, )
            x[:, 6:7] += pred * hnorm * delta_t
        elif model_config.mesh_model_name == 'WaveMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0)
            x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
            x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif model_config.mesh_model_name == 'RD_RPS_Mesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0)
                x[mask_mesh.squeeze(), 6:9] += pred[mask_mesh.squeeze()] * hnorm * delta_t
        else:
            distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

            edge_index = adj_t.nonzero().t().contiguous()

            dataset = data.Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset, data_id=0, training=False, vnorm=vnorm,
                          phi=torch.zeros(1, device=device))  # acceleration estimation

            if model_config.prediction == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm
                x[:, 3:5] = y

            if has_division:
                
                x_time = torch.concatenate((x[:, 0:1], torch.ones_like(x[:, 0:1], device=device) * it), dim=1)
                with torch.no_grad():
                    y_time = model_division(x_time, data_id=0)
                gt_y_time = y_list[0][it].clone().detach()

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t)  # position update

        if simulation_config.has_cell_division:
            A1 = A1 + delta_t

        if (it % step == 0) & (it >= 0) & visualize:

            plt.style.use('dark_background')

            fig = plt.figure(figsize=(12, 12))
            if has_mesh:
                pts = x[:, 1:3].detach().cpu().numpy()
                tri = Delaunay(pts)
                colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                if model_config.mesh_model_name == 'DiffMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                if model_config.mesh_model_name == 'WaveMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                if model_config.mesh_model_name == 'RD_Gray_Scott_Mesh':
                    fig = plt.figure(figsize=(12, 6))
                    ax = fig.add_subplot(1, 2, 1)
                    colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                    ax = fig.add_subplot(1, 2, 2)
                    colors = torch.sum(x[tri.simplices, 7], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                if model_config.mesh_model_name == 'RD_RPS_Mesh':
                    fig = plt.figure(figsize=(12, 12))
                    H1_IM = torch.reshape(x[:, 6:9], (100, 100, 3))
                    plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
            else:
                s_p = 25
                if simulation_config.has_cell_division:
                    s_p = 10
                for n in range(n_particle_types):
                    plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                x[index_particles[n], 2].detach().cpu().numpy(), s=s_p, color=cmap.color(n))

            if (has_mesh | (simulation_config.boundary == 'periodic')) & (model_config.mesh_model_name != 'RD_RPS_Mesh'):
                plt.xlim([0, 1])
                plt.ylim([0, 1])
            if model_config.particle_model_name == 'PDE_G':
                plt.xlim([-4, 4])
                plt.ylim([-4, 4])

            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{dataset_name}_{it}.tif", dpi=170.7)
            plt.close()




if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')

    config_list = ['arbitrary_3_dropout_5']

    for config_file in config_list:

        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        cmap = CustomColorMap(config=config)  # create colormap for given model_config

        data_generate(config, device=device, visualize=True , style='color', alpha=1, erase=False, step=config.simulation.n_frames // 40, bSave=True)
        data_train(config)
        # data_plot_training(config)

        # data_test(config, visualize=True, verbose=True, best_model=20, step=config.simulation.n_frames // 40)


