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
from ParticleGraph.generators.utils import choose_model, choose_mesh_model, generate_from_data
from ParticleGraph.models.utils import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.data_loaders import *
from ParticleGraph.utils import *
from ParticleGraph.fitting_models import linear_model
from ParticleGraph.embedding_cluster import *
from ParticleGraph.models import Division_Predictor


def data_generate(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1, scenario='none', device=None, bSave=True):
    print('')

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

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

    if config.data_folder_name != 'none':
        generate_from_data(config=config, device=device, visualize=visualize, folder=folder, step=step)
        return

    # load model parameters and create local varibales
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training
    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_cell_division = simulation_config.has_cell_division
    n_frames = simulation_config.n_frames
    cycle_length = None
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
        x_removed_list = []
    else:
        dropout_mask = np.arange(n_particles)

    model, bc_pos, bc_dpos = choose_model(config, device=device)

    if has_mesh:
        mesh_model = choose_mesh_model(config, device=device)
    else:
        mesh_model = None

    # fig = plt.figure(figsize=(8, 8))
    # pp=model.p.clone().detach()
    # max_radius = config.simulation.max_radius
    # rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    # for n in range(n_particle_types):
    #     for m in range(n_particle_types):
    #         plt.plot(to_numpy(rr), to_numpy(model.psi(rr, pp[n,m])), color=cmap.color(n), linewidth=1)
    # plt.show()

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
                        x_ = x[inv_dropout_mask].clone().detach()
                        x_[:,0] = torch.arange(len(x_), device=device)
                        x_removed_list.append(x[inv_dropout_mask].clone().detach())
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
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                plt.style.use('dark_background')

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
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Lut_Fig_{run}_{it}.jpg", dpi=170.7)
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
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Rot_{run}_Fig{it}.jpg", dpi=170.7)
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
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/All__{run}_{it}.jpg", dpi=170.7)
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
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
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
            if has_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
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
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    has_large_range = train_config.large_range

    embedding_cluster = EmbeddingCluster(config)

    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)

    l_dir, log_dir,logger = create_log_dir(config, dataset_name)

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs}')
    logger.info(f'Graph files N: {NGraphs}')

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
        time.sleep(0.5)
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
        time.sleep(0.5)
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
    # net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_4.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

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

    # update variable if dropout, cell_division, etc ...
    x = x_list[1][n_frames-1].clone().detach()
    T1 = x[:, 5:6].clone().detach()
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    if has_ghost:
        ghosts_particles = Ghost_Particles(config, n_particles, device)
        if train_config.ghost_method == 'MLP':
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.data], lr=5E-4)
        else:
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
        
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
            if has_ghost:
                mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
                mask_ghost = np.tile(mask_ghost,batch_size)
                mask_ghost = np.argwhere(mask_ghost==1)
                mask_ghost = mask_ghost[:,0].astype(int)

        total_loss = 0
        total_loss_division = 0

        Niter = n_frames * data_augmentation_loop // batch_size
        if (has_mesh) & (batch_size == 1):
            Niter = Niter // 4

        for N in range(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            time_batch=[]

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)
                x = x_list[run][k].clone().detach()

                if has_ghost:
                    if train_config.ghost_method == 'MLP':
                        x_ghost = ghosts_particles.get_pos_t(dataset_id=run, frame=k)
                    else:
                        x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k)
                    x = torch.cat((x, x_ghost), 0)
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
                    if model.edges==[]:
                        distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                        adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
                        t = torch.Tensor([radius ** 2])
                        edges = adj_t.nonzero().t().contiguous()
                        dataset = data.Data(x=x[:, :], edge_index=edges)
                    else:
                        dataset = data.Data(x=x[:, :], edge_index=model.edges)
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
            if has_ghost:
                optimizer_ghost_particles.zero_grad()
            if has_cell_division:
                optimizer_division.zero_grad()

            for batch in batch_loader:
                if has_mesh:
                    pred = model(batch, data_id=run)
                else:
                    pred = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi)

            if has_cell_division:
                pred_division = model_division(time_batch, data_id=run)
                loss_division = (pred_division - y_batch_division).norm(2)
                loss_division.backward()
                optimizer_division.step()
                total_loss_division += loss_division.item()

            if has_mesh:
                loss = ((pred - y_batch) * mask_mesh).norm(2)
            elif has_ghost:
                loss = ((pred[mask_ghost] - y_batch)).norm(2)
            else:
                 if not(has_large_range):
                     loss = (pred - y_batch).norm(2)
                 else:
                    loss = ((pred - y_batch)/(y_batch+1E-10)).norm(2) / 1E8

            loss.backward()
            optimizer.step()
            if has_ghost:
                optimizer_ghost_particles.step()
                if (N>0) & (N % 1000 == 0) & (train_config.ghost_method == 'MLP'):
                    fig = plt.figure(figsize=(8, 8))
                    plt.imshow(to_numpy(ghosts_particles.data[run, :, 120, :].squeeze()))
                    fig.savefig(f"{log_dir}/tmp_training/embedding/ghosts_{N}.jpg", dpi=300)
                    plt.close()

            if N%100==0:
                print(loss.item()/batch_size)

            total_loss += loss.item()

            visualize_embedding=True
            if visualize_embedding & ( (epoch == 0) & (N < 100) & (N % 2 == 0)  |  (epoch==0)&(N<10000) & (N%200==0)  |  (epoch==0)&(N%(Niter//100)==0)   | (epoch>0)&(N%(Niter//4)==0)):
                plot_training(dataset_name=dataset_name, filename='embedding', log_dir=log_dir, epoch=epoch, N=N, x=x, model=model, dataset_num = 1,
                              index_particles=index_particles, n_particles=n_particles, n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, device=device)
                if model_config.particle_model_name == 'PDE_GS':
                    fig = plt.figure(figsize=(8, 8))
                    rr = torch.tensor(np.logspace(7,9,1000)).to(device)
                    for n in range(n_particles):
                        embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None], rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                                 color=cmap.color(to_numpy(x[n, 5]).astype(int)), linewidth=1)
                    plt.xlabel('Distance [a.u]', fontsize=14)
                    plt.ylabel('MLP [a.u]', fontsize=14)
                    plt.xscale('log')
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/func_{dataset_name}_{epoch}_{N}.tif", dpi=300)
                    plt.close()

                if model_config.particle_model_name == 'PDE_B':
                    x = x_list[1][3000].clone().detach()
                    x[:, 2:5] = 0
                    distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                    adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    t = torch.Tensor([radius ** 2])
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)
                    with torch.no_grad():
                        y = model(dataset, data_id=1, training=False, vnorm=vnorm, phi=torch.zeros(1, device=device))  # acceleration estimation
                        lin_edge_out = model.lin_edge_out * ynorm
                        diffx = model.diffx
                        particle_id = to_numpy(model.particle_id)
                    type = to_numpy(T1[particle_id])
                    fig = plt.figure(figsize=(8, 8))
                    for n in range(n_particle_types):
                        pos = np.argwhere(type == n)
                        pos = pos[:, 0].astype(int)
                        plt.scatter(to_numpy(diffx[pos, 0]), to_numpy(lin_edge_out[pos, 0]), color=cmap.color(n), s=1, alpha=0.5)
                    plt.xlim([-0.04, 0.04])
                    plt.ylim([-5E-5, 5E-5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/func_{dataset_name}_{epoch}_{N}.tif",dpi=300)
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
        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()}, os.path.join(log_dir, 'models', f'best_ghost_particles_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(22, 4))
        # white background
        # plt.style.use('classic')

        ax = fig.add_subplot(1, 6, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)


        if (simulation_config.n_interactions < 100) & (simulation_config.has_cell_division == False) :

            ax = fig.add_subplot(1, 6, 3)
            if model_config.particle_model_name == 'PDE_G':
                rr = torch.tensor(np.linspace(0, radius * 1.3, 1000)).to(device)
            elif model_config.particle_model_name == 'PDE_GS':
                rr = torch.tensor(np.logspace(7,9,1000)).to(device)

            else:
                rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
            if has_mesh==False:
                func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=True, config=config,
                                                               model_lin_edge=model.lin_edge, model_a=model.a, dataset_number = 1, n_particles=n_particles, ynorm=ynorm, types=to_numpy(x[:, 5]), cmap=cmap, device=device)
            else:
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
                if pos.size>0:
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), color='k', s=5)
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)
            plt.xticks(fontsize=10.0)
            plt.yticks(fontsize=10.0)

            if (replace_with_cluster) & ((epoch == 1 * n_epochs // 4) | (epoch == 2 * n_epochs // 4) | (epoch == 3 * n_epochs // 4)):
                match train_config.sparsity:
                    case 'replace_embedding':
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
                        if train_config.fix_cluster_embedding:
                            lr_embedding = 1E-8
                            lr = train_config.learning_rate_end
                            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                            logger.info(f'Learning rates: {lr}, {lr_embedding}')
                    case 'replace_embedding_function':
                        logger.info(f'replace_embedding_function')
                        # Constrain function domain
                        y_func_list = func_list * 0
                        for n in range(n_particles):
                            pos = np.argwhere(new_labels == n)
                            pos = pos.squeeze()
                            if pos.size > 0:
                                target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                                y_func_list[pos] = target_func
                        lr_embedding = 1E-8
                        if has_ghost:
                            lr = 1E-8
                        else:
                            lr = train_config.learning_rate_end
                        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                        for sub_epochs in range(20):
                            loss=0
                            rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                            pred=[]
                            optimizer.zero_grad()
                            for n in range(n_particles):
                                embedding_ = model.a[1, n, :].clone().detach() * torch.ones((1000, model_config.embedding_dim), device=device)
                                match model_config.particle_model_name:
                                    case 'PDE_A':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                    case 'PDE_A_bis':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_),
                                            dim=1)
                                    case 'PDE_B':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_G':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_E':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                                pred.append(model.lin_edge(in_features.float()))

                            pred=torch.stack(pred)
                            loss = (pred[:,:,0] - y_func_list.clone().detach()).norm(2)
                            logger.info(f'    loss: {np.round(loss.item()/n_particles, 3)}')
                            loss.backward()
                            optimizer.step()
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
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



def data_test(config, visualize=False, verbose=True, best_model=20, step=5, ratio=1, run=1, test_simulation=False):
    print('')
    print('Plot roll-out inference ... ')

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_division = simulation_config.has_cell_division
    has_ghost = config.training.n_ghosts > 0

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

    # update variable if dropout


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
        model_division = Division_Predictor(config, device)
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

        embedding = model.a[1].data.clone().detach()
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
        model.a[1] = new_embedding
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
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device))

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

    x = x_list[0][0].clone().detach()
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    if has_ghost:
        model_ghost = Ghost_Particles(config, n_particles, device)
        net = f"./log/try_{dataset_name}/models/best_ghost_particles_with_{NGraphs - 1}_graphs_20.pt"
        state_dict = torch.load(net, map_location=device)
        model_ghost.load_state_dict(state_dict['model_state_dict'])
        model_ghost.eval()
        x_removed_list = torch.load(f'graphs_data/graphs_{dataset_name}/x_removed_list_0.pt', map_location=device)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)

    if simulation_config.has_cell_division:
        cycle_length = torch.load(f'./graphs_data/graphs_{dataset_name}/cycle_length.pt', map_location=device).to(device)
        cycle_length_distrib = cycle_length[to_numpy(x[:,5]).astype(int)].squeeze()
        A1 = torch.rand(cycle_length_distrib.shape[0], device=device)
        A1 = A1  * cycle_length_distrib
        A1 = A1[:,None]

    time.sleep(1)
    for it in trange(n_frames - 1):

        x0 = x_list[0][it].clone().detach()
        y0 = y_list[0][it].clone().detach()

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

            x_ = x
            if has_ghost:
                x_ghost = model_ghost.get_pos(dataset_id=run, frame=it)
                x_ = torch.cat((x_, x_ghost), 0)

            distance = torch.sum(bc_dpos(x_[:, None, 1:3] - x_[None, :, 1:3]) ** 2, dim=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = ((distance < radius ** 2) & (distance > min_radius ** 2)).float() * 1

            edge_index = adj_t.nonzero().t().contiguous()

            dataset = data.Data(x=x_, edge_index=edge_index)

            if test_simulation:
                y = y0 / ynorm
            else:
                with torch.no_grad():
                    y = model(dataset, data_id=run, training=False, vnorm=vnorm,
                              phi=torch.zeros(1, device=device))  # acceleration estimation

            if has_ghost:
                y = y[mask_ghost]

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

            # plt.style.use('dark_background')

            # matplotlib.use("Qt5Agg")
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
            if model_config.particle_model_name == 'PDE_GS':
                plt.xlim([-0.5E10, 0.5E10])
                plt.ylim([-0.5E10, 0.5E10])

            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{dataset_name}_{it}.tif", dpi=170.7)
            plt.close()

            if has_ghost:
                fig = plt.figure(figsize=(12, 12))
                x_ghost_pos = bc_pos(x_ghost[:, 1:3])
                plt.scatter(x_ghost_pos[:, 0].detach().cpu().numpy(),
                         x_ghost_pos[:, 1].detach().cpu().numpy(), s=s_p, color='k')
                x_removed = x_removed_list[it]
                plt.scatter(x_removed[:, 1].detach().cpu().numpy(),
                         x_removed[:, 2].detach().cpu().numpy(), s=s_p/2, color='g')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Ghost_{dataset_name}_{it}.tif", dpi=170.7)
                plt.close()



if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')

    config_list = ['gravity_solar_system']   #['arbitrary_3_3', 'arbitrary_3', 'gravity_16']  # ['Coulomb_3', 'boids_16', 'arbitrary_16', 'gravity_100']  # ['arbitrary_3_dropout_40_pos','arbitrary_3_dropout_50_pos']  #    ## ['arbitrary_3_3', 'arbitrary_3', 'gravity_16']

    for config_file in config_list:

        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        cmap = CustomColorMap(config=config)  # create colormap for given model_config

        # data_generate(config, device=device, visualize=False, run_vizualized=1, style='color', alpha=1, erase=True, step=config.simulation.n_frames // 1000, bSave=True)
        data_train(config)
        # data_test(config, visualize=True, verbose=True, best_model=2, run=1, step=config.simulation.n_frames // 100, test_simulation=True)


