import matplotlib.cm as cmplt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as pyg_utils
import os
from ParticleGraph.MLP import MLP
import imageio
from matplotlib import rc
import time
from ParticleGraph.utils import *
from ParticleGraph.fitting_models import *
from ParticleGraph.kan import *
# from pysr import PySRRegressor
# import cv2

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# from data_loaders import *

from GNN_particles_Ntype import *
from ParticleGraph.embedding_cluster import *
from ParticleGraph.utils import to_numpy, CustomColorMap, choose_boundary_values
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

# matplotlib.use("Qt5Agg")

class Interaction_Particles_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, config, device, aggr_type=None, bc_dpos=None):

        super(Interaction_Particles_extract, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.max_radius = simulation_config.max_radius
        self.data_augmentation = train_config.data_augmentation
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.sigma = simulation_config.sigma
        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        if simulation_config.has_cell_division :
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, 20500, 2)), device=self.device,
                             requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

        if self.update_type != 'none':
            self.lin_update = MLP(input_size=self.output_size + self.embedding_dim + 2, output_size=self.output_size,
                                  nlayers=self.n_layers_update, hidden_size=self.hidden_dim_update, device=self.device)

    def forward(self, data, data_id, training, vnorm, phi):

        self.data_id = data_id
        self.vnorm = vnorm
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pos = x[:, 1:3]
        d_pos = x[:, 3:5]
        particle_id = x[:, 0:1]

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, particle_id=particle_id)

        return pred, self.in_features, self.lin_edge_out

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, particle_id_i, particle_id_j):
        # squared distance
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm

        if self.data_augmentation & (self.training == True):
            new_delta_pos_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_delta_pos_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_delta_pos_x
            delta_pos[:, 1] = new_delta_pos_y
            new_dpos_x_i = self.cos_phi * dpos_x_i + self.sin_phi * dpos_y_i
            new_dpos_y_i = -self.sin_phi * dpos_x_i + self.cos_phi * dpos_y_i
            dpos_x_i = new_dpos_x_i
            dpos_y_i = new_dpos_y_i
            new_dpos_x_j = self.cos_phi * dpos_x_j + self.sin_phi * dpos_y_j
            new_dpos_y_j = -self.sin_phi * dpos_x_j + self.cos_phi * dpos_y_j
            dpos_x_j = new_dpos_x_j
            dpos_y_j = new_dpos_y_j

        embedding_i = self.a[self.data_id, to_numpy(particle_id_i), :].squeeze()
        embedding_j = self.a[self.data_id, to_numpy(particle_id_j), :].squeeze()

        match self.model:
            case 'PDE_A':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
            case 'PDE_B' | 'PDE_B_bis':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                                     dpos_y_j[:, None], embedding_i), dim=-1)
            case 'PDE_G':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None],
                                         dpos_x_j[:, None], dpos_y_j[:, None],embedding_j),dim=-1)
            case 'PDE_GS':
                in_features = torch.cat((r[:, None], embedding_j),dim=-1)
            case 'PDE_E':
                in_features = torch.cat(
                    (delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)

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
        else:  # PDE_A
            return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2)) - p[2] * torch.exp(
                -r ** (2 * p[3]) / (2 * self.sigma ** 2)))
class PDE_B_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=None, p=None, bc_dpos=None):
        super(PDE_B_extract, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.bc_dpos = bc_dpos

        self.a1 = 0.5E-5
        self.a2 = 5E-4
        self.a3 = 1E-8

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        sum = self.cohesion + self.alignment + self.separation

        return acc, sum, self.cohesion, self.alignment, self.separation, self.diffx, self.diffv, self.r, self.type

    def message(self, x_i, x_j):

        r = torch.sum(self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, dim=1)  # distance squared

        pp = self.p[to_numpy(x_i[:, 5]), :]

        cohesion = pp[:, 0:1].repeat(1, 2) * self.a1 * self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3])
        alignment = pp[:, 1:2].repeat(1, 2) * self.a2 * self.bc_dpos(x_j[:, 3:5] - x_i[:, 3:5])
        separation = pp[:, 2:3].repeat(1, 2) * self.a3 * self.bc_dpos(x_i[:, 1:3] - x_j[:, 1:3]) / (r[:, None].repeat(1, 2))

        self.cohesion = cohesion
        self.alignment = alignment
        self.separation = separation

        self.r = r
        self.diffx = self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3])
        self.diffv = self.bc_dpos(x_j[:, 3:5] - x_i[:, 3:5])
        self.type = x_i[:, 5]

        return (separation + alignment + cohesion)

    def psi(self, r, p):
        cohesion = p[0] * self.a1 * r
        separation = -p[2] * self.a3 / r
        return (cohesion + separation)
class Mesh_RPS_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None):
        super(Mesh_RPS_extract, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_size = model_config.hidden_dim
        self.nlayers = model_config.n_mp_layers
        self.embedding_dim = model_config.embedding_dim
        self.nparticles = simulation_config.n_particles
        self.ndataset = config.training.n_runs
        self.bc_dpos = bc_dpos

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                           hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.ndataset), int(self.nparticles), self.embedding_dim)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        uvw = data.x[:, 6:9]

        laplacian_uvw = self.propagate(edge_index, uvw=uvw, discrete_laplacian=edge_attr)

        particle_id = to_numpy(x[:, 0])
        embedding = self.a[self.data_id, particle_id, :]

        input_phi = torch.cat((laplacian_uvw, uvw, embedding), dim=-1)

        pred = self.lin_phi(input_phi)

        return pred, input_phi, embedding

    def message(self, uvw_j, discrete_laplacian):
        return discrete_laplacian[:,None] * uvw_j

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r

    def plot_embedding_func_cluster(model, config, config_file, embedding_cluster, cmap, index_particles,
                                    n_particle_types, n_particles, ynorm, epoch, log_dir, device):

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=400)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif",dpi=170.7)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                            model_lin_edge=model.lin_edge, model_a=model.a,
                                                            dataset_number=1,
                                                            n_particles=n_particles, ynorm=ynorm,
                                                            types=to_numpy(x[:, 5]),
                                                            cmap=cmap, device=device)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim([-0.5E6, 0.5E6])
        plt.tight_layout()
        plt.close()

        labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                          train_config.cluster_distance_threshold, index_particles,
                                                          n_particle_types, embedding_cluster)

        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')

        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values

        model_a_first = model.a.clone().detach()

        with torch.no_grad():
            model.a[1] = model_a_.clone().detach()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model_a_first, 1)
        csv_ = embedding
        np.save(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.txt", csv_)
        if n_particle_types > 1000:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                        cmap=cc)
        else:
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n),
                            s=400, alpha=0.1)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

def plot_embedding_func_cluster(model, config, config_file, embedding_cluster, cmap, index_particles, type_list, n_particle_types, n_particles, ynorm, epoch, log_dir, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    embedding = get_embedding(model.a, 1)
    for n in range(n_particle_types):
        plt.scatter(embedding[index_particles[n], 0],
                    embedding[index_particles[n], 1], color=cmap.color(n), s=400)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    # plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif",dpi=170.7)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                        model_lin_edge=model.lin_edge, model_a=model.a,
                                                        dataset_number=1,
                                                        n_particles=n_particles, ynorm=ynorm,
                                                        types=to_numpy(type_list),
                                                        cmap=cmap, device=device)
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
    # xticks with sans serif font
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, max_radius])
    plt.ylim([-0.5E6, 0.5E6])
    plt.tight_layout()
    plt.close()

    labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                      train_config.cluster_distance_threshold, index_particles,
                                                      n_particle_types, embedding_cluster)

    accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
    print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')

    model_a_ = model.a[1].clone().detach()
    for n in range(n_clusters):
        pos = np.argwhere(labels == n).squeeze().astype(int)
        pos = np.array(pos)
        if pos.size > 0:
            median_center = model_a_[pos, :]
            median_center = torch.median(median_center, dim=0).values
            model_a_[pos, :] = median_center

    model_a_first = model.a.clone().detach()

    with torch.no_grad():
        model.a[1] = model_a_.clone().detach()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    embedding = get_embedding(model_a_first, 1)
    csv_ = embedding
    np.save(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.txt", csv_)
    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                    cmap=cc)
    else:
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n),
                        s=400, alpha=0.1)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    return  accuracy, n_clusters, new_labels

def plot_embedding(index, model_a, dataset_number, index_particles, n_particles, n_particle_types, epoch, it, fig, ax, cmap, device):

    embedding = get_embedding(model_a, dataset_number)

    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Particle embedding', fontsize=12)
    for n in range(n_particle_types):
        plt.scatter(embedding[index_particles[n], 0],
                    embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
    plt.text(.05, .94, f'e: {epoch} it: {it}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
    plt.text(.05, .86, f'N: {n_particles}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)

    return embedding

def plot_function(bVisu, index, model_name, model_MLP, model_a, dataset_number, label, pos, max_radius, ynorm, index_particles, n_particles, n_particle_types, epoch, it, fig, ax, cmap, device):

    # print(f'plot functions epoch:{epoch} it: {it}')

    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (model)', fontsize=12)
    func_list = []
    for n in range(n_particles):
        embedding_ = model_a[1, n, :] * torch.ones((1000, 2), device=device)

        match model_name:
            case 'PDE_A':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                     pos[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_A_bis':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                     pos[:, None] / max_radius, embedding_, embedding_), dim=1)
            case 'PDE_B' | 'PDE_B_bis':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                     pos[:, None] / max_radius, 0 * pos[:, None], 0 * pos[:, None],
                                     0 * pos[:, None], 0 * pos[:, None], embedding_), dim=1)
            case 'PDE_G':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                     pos[:, None] / max_radius, 0 * pos[:, None], 0 * pos[:, None],
                                     0 * pos[:, None], 0 * pos[:, None], embedding_), dim=1)
            case 'PDE_GS':
                in_features = torch.cat((pos[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_E':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                     pos[:, None] / max_radius, embedding_, embedding_), dim=-1)
            
        with torch.no_grad():
            func = model_MLP(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        if bVisu:
            plt.plot(to_numpy(pos),
                     to_numpy(func) * to_numpy(ynorm), color=cmap.color(label[n]), linewidth=1)
    func_list = torch.stack(func_list)
    func_list = to_numpy(func_list)
    if bVisu:
        plt.xlabel(r'$d_{ij} [a.u.]$', fontsize=12)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij} [a.u.]$', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)
        # plt.ylim([-0.04, 0.03])
        plt.text(.05, .86, f'N: {n_particles // 50}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
        plt.text(.05, .94, f'e: {epoch} it: {it}', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    return func_list

def plot_umap(index, func_list, log_dir, n_neighbors, index_particles, n_particles, n_particle_types, embedding_cluster, epoch, it, fig, ax, cmap, device):

    # print(f'plot umap epoch:{epoch} it: {it}')
    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    if False: # os.path.exists(os.path.join(log_dir, f'proj_interaction_{epoch}.npy')):
        proj_interaction = np.load(os.path.join(log_dir, f'proj_interaction_{epoch}.npy'))
    else:
        new_index = np.random.permutation(func_list.shape[0])
        new_index = new_index[0:min(1000, func_list.shape[0])]
        trans = umap.UMAP(n_neighbors=n_neighbors, n_components=2, transform_queue_size=0).fit(func_list[new_index])
        proj_interaction = trans.transform(func_list)
    np.save(os.path.join(log_dir, f'proj_interaction_{epoch}.npy'), proj_interaction)
    plt.title(r'UMAP of $f(\ensuremath{\mathbf{a}}_i, d_{ij}$)', fontsize=12)

    labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')

    label_list = []
    for n in range(n_particle_types):
        tmp = labels[index_particles[n]]
        label_list.append(np.round(np.median(tmp)))
    label_list = np.array(label_list)
    new_labels = labels.copy()
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=1)

    plt.xlabel(r'UMAP 0', fontsize=12)
    plt.ylabel(r'UMAP 1', fontsize=12)

    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.text(.05, .86, f'N: {n_particles}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
    plt.text(.05, .94, f'e: {epoch} it: {it}', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    return proj_interaction, new_labels, n_clusters

def plot_confusion_matrix(index, true_labels, new_labels, n_particle_types, epoch, it, fig, ax):

    # print(f'plot confusion matrix epoch:{epoch} it: {it}')
    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    confusion_matrix = metrics.confusion_matrix(true_labels, new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if n_particle_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False, colorbar=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d', colorbar=False)
    Accuracy = metrics.accuracy_score(true_labels, new_labels)
    plt.title(f'Accuracy: {np.round(Accuracy,3)}', fontsize=12)
    # print(f'Accuracy: {np.round(Accuracy,3)}')
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlabel(r'Predicted label', fontsize=12)
    plt.ylabel(r'True label', fontsize=12)

    return Accuracy

def data_plot_attraction_repulsion(config_file, epoch_list, device):
    print('')

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_mesh = (config.graph_model.mesh_model_name != '')
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)

    embedding_cluster = EmbeddingCluster(config)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
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

    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
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

    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    time.sleep(0.5)

    # matplotlib.use("Qt5Agg")
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0
    # style = {
    #     "pgf.rcfonts": False,
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,
    #     "font.family": "sans-serif"
    # }
    # matplotlib.rcParams.update(style)
    # plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "HelveticaNeue", "Helvetica-Neue", "Helvetica", "Arial",
    #                                    "Liberation"]

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        # matplotlib.use("Qt5Agg")
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})

        model_a_first = model.a.clone().detach()
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster, cmap, index_particles, type_list,
                                    n_particle_types, n_particles, ynorm, epoch, log_dir, device)

        # matplotlib.use("Qt5Agg")

        p = config.simulation.params
        if len(p) > 1:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt',map_location=device)
        rmserr_list = []

        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        for n in range(int(n_particles*(1-train_config.particle_dropout))):
            embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(), p[to_numpy(type_list[n]).astype(int)].squeeze())
            rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xlim([0, max_radius])
        plt.ylim([-0.04, 0.03])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_all_{config_file}_{epoch}.tif",dpi=170.7)
        rmserr_list = torch.stack(rmserr_list)
        rmserr_list = to_numpy(rmserr_list)
        print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plots = []
        rr = torch.tensor(np.linspace(0, 1.5*max_radius, 1000)).to(device)
        plots.append(rr)
        for n in range(n_particle_types):
            embedding_ = model.a[1, index_particles[n][0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            match config.graph_model.particle_model_name:
                case 'PDE_A':
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                case 'PDE_A_bis':
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_, embedding_), dim=1)
                case 'PDE_B' | 'PDE_B_bis':
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                case 'PDE_G':
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                case 'PDE_GS':
                    in_features = torch.cat((rr[:, None] / max_radius, embedding_), dim=1)
                case 'PDE_E':
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_, embedding_), dim=-1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            plots.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm), linewidth=8, alpha=1)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xlim([0, 2*max_radius])
        plt.ylim([-0.04, 0.03])
        plt.tight_layout()
        torch.save(plots,f"./{log_dir}/tmp_training/plots_{config_file}_{epoch}.pt")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plots = []
        plots.append(rr)
        for n in range(n_particle_types):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
            plots.append(model.psi(rr, p[n], p[n]).squeeze())
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.1])
        # plt.xlim([0, 0.02])
        # plt.ylim([0, 0.5E6])
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}.tif",dpi=170.7)
        plt.close()

        rr = torch.tensor(np.linspace(-1.5 * max_radius, 1.5 * max_radius, 2000)).to(device)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plots = []
        plots.append(rr)
        for n in range(n_particle_types):
            t = model.psi(rr, p[n], p[n])
            plt.plot(to_numpy(rr), to_numpy(t), color=cmap.color(n), linewidth=8)
            plots.append(model.psi(rr, p[n], p[n]).squeeze())
        # plt.xlabel(r'$d_{ij}$', fontsize=64)
        # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([-max_radius * 1.5, max_radius * 1.5])
        # plt.ylim([-0.15, 0.15])
        # plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.1])
        # plt.xlim([0, 0.02])
        # plt.ylim([0, 0.5E6])
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.tight_layout()
        torch.save(plots, f"./{log_dir}/tmp_training/plots_true_{config_file}_{epoch}.pt")
        plt.close()

def data_plot_attraction_repulsion_asym(config_file, epoch_list, device):
    print('')

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    dataset_name = config.dataset
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    embedding_cluster = EmbeddingCluster(config)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
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

    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)

    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    x=[]
    y=[]

    print('done ...')

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    time.sleep(0.5)

    matplotlib.use("Qt5Agg")
    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0
    # style = {
    #     "pgf.rcfonts": False,
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,
    #     "font.family": "sans-serif"
    # }
    # matplotlib.rcParams.update(style)
    # plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "HelveticaNeue", "Helvetica-Neue", "Helvetica", "Arial",
    #                                    "Liberation"]

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        # matplotlib.use("Qt5Agg"

        model_a_first = model.a.clone().detach()
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster, cmap, index_particles, type_list,
                                    n_particle_types, n_particles, ynorm, epoch, log_dir, device)

        x = x_list[0][100].clone().detach()
        index_particles = get_index_particles(x, n_particle_types, dimension)
        type_list = to_numpy(get_type_list(x, dimension))
        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
        t = torch.Tensor([max_radius ** 2])
        edges = adj_t.nonzero().t().contiguous()
        indexes = np.random.randint(0, edges.shape[1], 5000)
        edges = edges[:, indexes]

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list = []

        for n in trange(edges.shape[1]):
            embedding_1 = model.a[1, edges[0,n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            embedding_2 = model.a[1, edges[1,n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                 device=device)
            type = type_list[to_numpy(edges[0,n])].astype(int)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(type), linewidth=8)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim([-0.03, 0.03])
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

        p = config.simulation.params
        if len(p) > 0:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_{dataset_name}/p.pt')
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        true_func=[]
        for n in range(n_particle_types):
            for m in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[3 * n + m], p[3 * n + m]).squeeze()), color=cmap.color(n), linewidth=8)
                true_func.append(model.psi(rr, p[3 * n + m].squeeze(), p[n * 3 + m].squeeze()))
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim([-0.03, 0.03])
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}.tif", dpi=170.7)
        plt.close()

        true_func_list=[]
        for k in trange(edges.shape[1]):
                n = type_list[to_numpy(edges[0,k])].astype(int)
                m = type_list[to_numpy(edges[1,k])].astype(int)
                true_func_list.append(true_func[3 * n.squeeze() + m.squeeze()])

        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)

        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2,axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')

def data_plot_attraction_repulsion_continuous(config_file, epoch_list, device):
    print('')

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension=simulation_config.dimension
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
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
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

    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
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
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    time.sleep(0.5)


    # matplotlib.use("Qt5Agg")
    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0

    # style = {
    #     "pgf.rcfonts": False,
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,
    #     "font.family": "sans-serif"
    # }
    # matplotlib.rcParams.update(style)
    # plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "HelveticaNeue", "Helvetica-Neue", "Helvetica", "Arial",
    #                                    "Liberation"]

    epoch_list = [20]
    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        n_particle_types = 3
        index_particles = []
        for n in range(n_particle_types):
            index_particles.append(
                np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
        type = torch.zeros(int(n_particles / n_particle_types), device=device)
        for n in range(1, n_particle_types):
            type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
        x[:,5]=type

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model.a, 1)
        csv_ = embedding
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=400, alpha=0.1)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif",dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.txt", csv_)
        plt.close()


        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        func_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particles):
            embedding = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            csv=to_numpy(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n//1600), linewidth=8,alpha=0.1)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        # plt.ylim([-0.15, 0.15])
        plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.06])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.tif",dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.txt", csv_)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        if os.path.exists(f'graphs_data/graphs_{dataset_name}/model_p.pt'):
            p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt')
        else:
            p = config.simulation.params
        true_func_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particles):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n//1600), linewidth=8,alpha=0.1)
            true_func_list.append(model.psi(rr, p[n], p[n]))
            csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        # plt.ylim([-0.15, 0.15])
        plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.06])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}.tif",dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/true_func_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/true_func_{config_file}_{epoch}.txt", csv_)
        plt.close()

        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)

        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2,axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')

def data_plot_gravity(config_file, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    dimension = simulation_config.dimension

    # print(config.pretty())

    cmap = CustomColorMap(config=config)
    aggr_type = config.graph_model.aggr_type

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load data ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_1.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.use("Qt5Agg")

    fig = plt.figure(figsize=(10.5, 9.6))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)
    embedding = embedding[0:int(n_particles*(1-train_config.particle_dropout))]

    ax = fig.add_subplot(3, 3, 2)
    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    func_list = plot_function(False, 'b)', config.graph_model.particle_model_name, model.lin_edge, model.a, 1, to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles, int(n_particles*(1-train_config.particle_dropout)), n_particle_types, 20, '$10^6$', fig, ax, cmap, device)
    proj_interaction, new_labels, n_clusters = plot_umap('b)', func_list, log_dir, 500, index_particles, int(n_particles*(1-train_config.particle_dropout)), n_particle_types, embedding_cluster, 20, '$10^6$', fig, ax, cmap,device)

    match config.training.cluster_method:
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance', thresh=0.5)
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=0.5)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance')

    ax = fig.add_subplot(3, 3, 3)

    label_list = []
    for n in range(n_particle_types):
        tmp = labels[index_particles[n]]
        label_list.append(np.round(np.median(tmp)))
    label_list = np.array(label_list)
    new_labels = labels.copy()
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n
    Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)

    print(f'Accuracy: {Accuracy}  n_clusters: {n_clusters}')

    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for n in range(n_clusters):
        pos = np.argwhere(labels == n)
        pos = np.array(pos)
        if pos.size > 0:
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=200, alpha=0.1)
    label_list = []
    for n in range(n_particle_types):
        tmp = labels[index_particles[n]]
        label_list.append(np.round(np.median(tmp)))
    label_list = np.array(label_list)
    plt.xlabel(r'UMAP 0', fontsize=64)
    plt.ylabel(r'UMAP 1', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/UMAP_{config_file}.tif", dpi=300)
    plt.close()

    model_a_first = model.a.clone().detach()

    model_a_ = model.a[1].clone().detach()
    for k in range(n_clusters):
        pos = np.argwhere(new_labels == k).squeeze().astype(int)
        if len(pos) > 0:
            temp = model_a_[pos, :].clone().detach()
            model_a_[pos, :] = torch.median(temp, dim=0).values.repeat((len(pos), 1))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_

    embedding = get_embedding(model_a_first, 1)

    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    csv_ = []
    for n in range(n_particle_types):
        plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=400,
                    alpha=0.1)
        csv_.append(embedding[index_particles[n], :])
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}.tif", dpi=300)
    # csv_ = np.reshape(csv_, (csv_.shape[0]*csv_.shape[1], 2))
    # np.save(f"./{log_dir}/tmp_training/embedding_{config_file}.npy", csv_)
    # np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}.txt", csv_)
    plt.close()

    ax = fig.add_subplot(3, 3, 4)
    plt.text(-0.25, 1.1, f'd)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Clustered particle embedding', fontsize=12)
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        if len(pos) > 0:
            plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=6)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    plt.text(-0.25, 1.1, f'e)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (model)', fontsize=12)
    func_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        if len(pos) > 0:
            embedding = model.a[0, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    p = config.simulation.params
    if len(p) > 1:
        p = torch.tensor(p, device=device)
    else:
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)

    type_list = x[:, 5:6].clone().detach()
    rmserr_list = []
    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    csv_ = []
    csv_.append(to_numpy(rr))
    for n in range(int(n_particles * (1 - train_config.particle_dropout))):
        embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        csv_.append(to_numpy(func))
        true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                              p[to_numpy(type_list[n]).astype(int)].squeeze())
        rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
        plt.plot(to_numpy(rr),
                 to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=64)
    plt.xlim([0, max_radius])
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/func_all{config_file}.tif", dpi=300)
    csv_ = np.array(csv_)
    np.save(f"./{log_dir}/tmp_training/func_all{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/func_all{config_file}.txt", csv_)
    plt.close()

    rmserr_list = torch.stack(rmserr_list)
    rmserr_list = to_numpy(rmserr_list)
    print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')

    ax = fig.add_subplot(3, 3, 6)
    plt.text(-0.25, 1.1, f'k)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (true)', fontsize=12)
    p = np.linspace(0.5, 5, n_particle_types)
    p = torch.tensor(p, device=device)
    for n in range(n_particle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])


    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    p = np.linspace(0.5, 5, n_particle_types)
    p = torch.tensor(p, device=device)
    csv_ = []
    csv_.append(to_numpy(rr))
    for n in range(n_particle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
        csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/true_func{config_file}.tif", dpi=300)
    csv_ = np.array(csv_)
    np.save(f"./{log_dir}/tmp_training/true_func{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/true_func{config_file}.txt", csv_)
    plt.close()

    plot_list = []
    for n in range(int(n_particles * (1 - train_config.particle_dropout))):
        embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            pred = model.lin_edge(in_features.float())
        pred = pred[:, 0]
        plot_list.append(pred * ynorm)
    p = np.linspace(0.5, 5, n_particle_types)
    popt_list = []
    for n in range(int(n_particles * (1 - train_config.particle_dropout))):
        popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    p_list = p[to_numpy(type_list).astype(int)]

    ax = fig.add_subplot(3, 3, 7)
    plt.text(-0.25, 1.1, f'g)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    x_data = p_list.squeeze()
    y_data = popt_list[:, 0]
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    plt.scatter(p_list, popt_list[:, 0], color='k', s=20)
    plt.title(r'Reconstructed masses', fontsize=12)
    plt.xlabel(r'True mass ', fontsize=12)
    plt.ylabel(r'Predicted mass ', fontsize=12)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])

    threshold = 0.4
    relative_error = np.abs(y_data-x_data)/x_data
    pos = np.argwhere(relative_error<threshold)
    pos_outliers = np.argwhere(relative_error>threshold)

    x_data_ = x_data[pos[:,0]]
    y_data_ = y_data[pos[:,0]]
    lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
    residuals = y_data_ - linear_model(x_data_, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data_)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(0.5, 4.5, f"$R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error>0.2)} / {n_particles}", fontsize=10)
    plt.text(0.5, 5, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    print(f'R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error>threshold)}  ')

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    csv_ = []
    csv_.append(p_list)
    csv_.append(popt_list[:, 0])
    plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
    plt.scatter(p_list, popt_list[:, 0], color='k', s=50, alpha=0.5)
    plt.scatter(p_list[pos_outliers[:,0]], popt_list[pos_outliers[:,0], 0], color='r', s=50)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'True mass ', fontsize=64)
    plt.ylabel(r'Reconstructed mass ', fontsize=64)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/mass{config_file}.tif", dpi=300)
    # csv_ = np.array(csv_)
    # np.save(f"./{log_dir}/tmp_training/mass{config_file}.npy", csv_)
    # np.savetxt(f"./{log_dir}/tmp_training/mass{config_file}.txt", csv_)
    plt.close()

    relative_error = np.abs(popt_list[:, 0] - p_list.squeeze()) / p_list.squeeze() * 100

    print (f'all mass relative error: {np.round(np.mean(relative_error), 3)}+/-{np.round(np.std(relative_error), 3)}')
    print (f'all mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:,0]]), 3)}+/-{np.round(np.std(relative_error[pos[:,0]]), 3)}')

    ax = fig.add_subplot(3, 3, 8)
    plt.text(-0.25, 1.1, f'h)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.scatter(p_list, -popt_list[:, 1], color='k', s=20)
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.title(r'Reconstructed exponent', fontsize=12)
    plt.xlabel(r'True mass ', fontsize=12)
    plt.ylabel(r'Exponent fit ', fontsize=12)
    plt.text(0.5, -0.5, f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)
    print(f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}")

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    csv_ = []
    csv_.append(p_list.squeeze())
    csv_.append(-popt_list[:, 1])
    csv_ = np.array(csv_)
    plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
    plt.scatter(p_list, -popt_list[:, 1], color='k', s=50, alpha=0.5)
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'True mass', fontsize=64)
    plt.ylabel(r'Reconstructed exponent', fontsize=64)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/exponent_{config_file}.tif", dpi=300)
    np.save(f"./{log_dir}/tmp_training/exponent_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/exponent_{config_file}.txt", csv_)
    plt.close()


    # find last image file in logdir
    ax = fig.add_subplot(3, 3, 9)
    files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        last_file = files[-1]
        # load image file with imageio
        image = imageio.imread(last_file)
        print('12')
        plt.text(-0.25, 1.1, f'l)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Rollout inference (frame 1000)', fontsize=12)
        plt.imshow(image)
        # rmove xtick
        plt.xticks([])
        plt.yticks([])

    time.sleep(1)
    plt.tight_layout()
    # plt.savefig('Fig3.pdf', format="pdf", dpi=300)
    plt.savefig(f'./log/try_{config_file}/Fig_3.jpg', dpi=300)
    plt.close()

def data_plot_gravity_continuous(config_file, device):

    # config_file = 'gravity_16'
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    # print(config.pretty())

    cmap = CustomColorMap(config=config)
    aggr_type = config.graph_model.aggr_type

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load data ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.use("Qt5Agg")

    fig = plt.figure(figsize=(10.5, 9.6))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 3, 2)
    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    func_list = plot_function(True, 'b)', config.graph_model.particle_model_name, model.lin_edge, model.a, 1, to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap, device)

    model_a_first = model.a.clone().detach()

    embedding = get_embedding(model_a_first, 1)

    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    csv_ = []
    for n in range(n_particle_types):
        plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n%256), s=400,
                    alpha=0.1)
        csv_.append(embedding[index_particles[n], :])
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    csv_ = np.array(csv_)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}.tif", dpi=300)
    csv_ = np.array(csv_)
    csv_ = np.reshape(csv_, (csv_.shape[0]*csv_.shape[1], 2))
    np.save(f"./{log_dir}/tmp_training/embedding_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}.txt", csv_)
    plt.close()

    p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)

    type_list = x[:, 5:6].clone().detach()
    rmserr_list = []
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    csv_ = []
    csv_.append(to_numpy(rr))
    for n in range(int(n_particles * (1 - train_config.particle_dropout))):
        embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        csv_.append(to_numpy(func))
        true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                              p[to_numpy(type_list[n]).astype(int)].squeeze())
        rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
        plt.plot(to_numpy(rr),
                 to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(n%256), linewidth=8, alpha=0.1)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=64)
    plt.xlim([0, max_radius])
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/func_{config_file}.tif", dpi=300)
    csv_ = np.array(csv_)
    np.save(f"./{log_dir}/tmp_training/func_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/func_{config_file}.txt", csv_)
    plt.close()

    rmserr_list = torch.stack(rmserr_list)
    rmserr_list = to_numpy(rmserr_list)
    print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')

    ax = fig.add_subplot(3, 3, 6)
    print('6')
    plt.text(-0.25, 1.1, f'k)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (true)', fontsize=12)
    p = np.linspace(0.5, 5, n_particle_types)
    p = torch.tensor(p, device=device)
    for n in range(n_particle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    p = np.linspace(0.5, 5, n_particle_types)
    p = torch.tensor(p, device=device)
    csv_ = []
    csv_.append(to_numpy(rr))
    for n in range(n_particle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n%256), linewidth=8)
        csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}.tif", dpi=300)
    csv_ = np.array(csv_)
    np.save(f"./{log_dir}/tmp_training/true_func_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/true_func_{config_file}.txt", csv_)
    plt.close()

    plot_list = []
    for n in range(int(n_particles * (1 - train_config.particle_dropout))):
        embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            pred = model.lin_edge(in_features.float())
        pred = pred[:, 0]
        plot_list.append(pred * ynorm)
    p = np.linspace(0.5, 5, n_particle_types)
    popt_list = []
    for n in range(int(n_particles * (1 - train_config.particle_dropout))):
        popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    p_list = p[to_numpy(type_list).astype(int)]

    ax = fig.add_subplot(3, 3, 7)
    print('7')
    plt.text(-0.25, 1.1, f'g)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    x_data = p_list.squeeze()
    y_data = popt_list[:, 0]
    x_data =np.delete(x_data,726)   # manual removal of one outlier which mess up the linear fit
    y_data =np.delete(y_data,726)

    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    plt.scatter(p_list, popt_list[:, 0], color='k', s=20)
    plt.title(r'Reconstructed masses', fontsize=12)
    plt.xlabel(r'True mass ', fontsize=12)
    plt.ylabel(r'Predicted mass ', fontsize=12)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.text(0.5, 5, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(0.5, 4.5, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    csv_ = []
    csv_.append(x_data)
    csv_.append(y_data)
    plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
    for n in range(n_particles):
        plt.scatter(p_list[n], popt_list[n, 0], color=cmap.color(n%256), s=400, alpha=0.1)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, 6])
    plt.ylim([0, 6])
    plt.xlabel(r'True mass ', fontsize=64)
    plt.ylabel(r'Reconstructed mass ', fontsize=64)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/mass_{config_file}.tif", dpi=300)
    csv_ = np.array(csv_)
    np.save(f"./{log_dir}/tmp_training/mass_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/mass_{config_file}.txt", csv_)
    plt.close()

    relative_error = np.abs(x_data - y_data) / x_data * 100

    print (f'all mass relative error: {np.round(np.mean(relative_error), 4)}+/-{np.round(np.std(relative_error), 4)}')

    ax = fig.add_subplot(3, 3, 8)
    print('8')
    plt.text(-0.25, 1.1, f'h)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.scatter(p_list, -popt_list[:, 1], color='k', s=20)
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.title(r'Reconstructed exponent', fontsize=12)
    plt.xlabel(r'True mass ', fontsize=12)
    plt.ylabel(r'Exponent fit ', fontsize=12)
    plt.text(0.5, -0.5, f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)

    print(f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}")

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    csv_ = []
    csv_.append(p_list.squeeze())
    csv_.append(-popt_list[:, 1])
    csv_ = np.array(csv_)
    plt.scatter(p_list, -popt_list[:, 1], color='k', s=400)
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'True mass', fontsize=64)
    plt.ylabel(r'Reconstructed exponent', fontsize=64)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/exponent_{config_file}.tif", dpi=300)
    np.save(f"./{log_dir}/tmp_training/exponent_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/exponent_{config_file}.txt", csv_)
    plt.close()

    # find last image file in logdir
    ax = fig.add_subplot(3, 3, 9)
    files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        last_file = files[-1]
        # load image file with imageio
        image = imageio.imread(last_file)
        print('12')
        plt.text(-0.25, 1.1, f'l)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Rollout inference (frame 1000)', fontsize=12)
        plt.imshow(image)
        # rmove xtick
        plt.xticks([])
        plt.yticks([])

    time.sleep(1)
    plt.tight_layout()
    # plt.savefig('Fig3.pdf', format="pdf", dpi=300)
    plt.savefig(f'Fig3_{config_file}.jpg', dpi=300)
    plt.close()

def data_plot_gravity_solar_system(config_file, device):

    config_file = 'gravity_solar_system'
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    print(config.pretty())

    cmap = CustomColorMap(config=config)
    aggr_type = config.graph_model.aggr_type

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load data ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_2.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.use("Qt5Agg")

    fig = plt.figure(figsize=(10.5, 9.6))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 3, 2)
    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    func_list = plot_function(True, 'b)', config.graph_model.particle_model_name, model.lin_edge,
                              model.a, 1, to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles,
                              n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 3, 3)

    it = 2000
    x0 = x_list[0][it].clone().detach()
    y0 = y_list[0][it].clone().detach()
    x = x_list[0][it].clone().detach()
    distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
    t = torch.Tensor([max_radius ** 2])  # threshold
    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)) * 1.0
    edge_index = adj_t.nonzero().t().contiguous()
    dataset = data.Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        y = model(dataset, data_id=1, training=False, vnorm=vnorm, phi=torch.zeros(1,device=device))  # acceleration estimation
    y = y * ynorm



    proj_interaction, new_labels, n_clusters = plot_umap('b)', func_list, log_dir, 500, index_particles,
                                                         n_particles, n_particle_types, embedding_cluster, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 3, 3)
    Accuracy = plot_confusion_matrix('c)', to_numpy(x[:,5:6]), new_labels, n_particle_types, 20, '$10^6$', fig, ax)
    plt.tight_layout()

    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    for k in range(n_clusters):
        pos = np.argwhere(new_labels == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        model_a_[pos, :] = torch.median(temp, dim=0).values.repeat((len(pos), 1))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_
    embedding, embedding_particle = get_embedding(model.a, 1)

    ax = fig.add_subplot(3, 3, 4)
    plt.text(-0.25, 1.1, f'd)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Clustered particle embedding', fontsize=12)
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=6)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    print('5')
    plt.text(-0.25, 1.1, f'e)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (model)', fontsize=12)
    func_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        embedding = model.a[0, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                 0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        plt.plot(to_numpy(rr),
                 to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 3, 6)
    print('6')
    plt.text(-0.25, 1.1, f'k)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (true)', fontsize=12)
    p = np.linspace(0.5, 5, n_particle_types)
    p = torch.tensor(p, device=device)
    for n in range(n_particle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])

    plot_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        embedding = model.a[0, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if config.graph_model.prediction == '2nd_derivative':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        else:
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        with torch.no_grad():
            pred = model.lin_edge(in_features.float())
        pred = pred[:, 0]
        plot_list.append(pred * ynorm)
    p = np.linspace(0.5, 5, n_particle_types)
    popt_list = []
    for n in range(n_particle_types):
        popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    ax = fig.add_subplot(3, 3, 7)
    print('7')
    plt.text(-0.25, 1.1, f'g)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    x_data = p
    y_data = popt_list[:, 0]
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    plt.plot(p, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    plt.scatter(p, popt_list[:, 0], color='k', s=20)
    plt.title(r'Reconstructed masses', fontsize=12)
    plt.xlabel(r'True mass ', fontsize=12)
    plt.ylabel(r'Predicted mass ', fontsize=12)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.text(0.5, 5, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(0.5, 4.5, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)

    ax = fig.add_subplot(3, 3, 8)
    print('8')
    plt.text(-0.25, 1.1, f'h)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.scatter(p, -popt_list[:, 1], color='k', s=20)
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.title(r'Reconstructed exponent', fontsize=12)
    plt.xlabel(r'True mass ', fontsize=12)
    plt.ylabel(r'Exponent fit ', fontsize=12)
    plt.text(0.5, -0.5, f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)

    # find last image file in logdir
    ax = fig.add_subplot(3, 3, 9)
    files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        last_file = files[-1]
        # load image file with imageio
        image = imageio.imread(last_file)
        print('12')
        plt.text(-0.25, 1.1, f'l)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Rollout inference (frame 1000)', fontsize=12)
        plt.imshow(image)
        # rmove xtick
        plt.xticks([])
        plt.yticks([])

    time.sleep(1)
    plt.tight_layout()
    # plt.savefig('Fig3.pdf', format="pdf", dpi=300)
    plt.savefig('Fig3.jpg', dpi=300)
    plt.close()

def data_plot_Coulomb(config_file, device):

    # config_file = 'Coulomb_3'
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    # print(config.pretty())

    cmap = CustomColorMap(config=config)
    aggr_type = config.graph_model.aggr_type

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    dimension = simulation_config.dimension

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load data ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    epoch=20
    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.use("Qt5Agg")

    accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster, cmap, index_particles, type_list,
                                n_particle_types, n_particles, ynorm, epoch, log_dir, device)


    x = x_list[0][100].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = to_numpy(get_type_list(x, dimension))
    distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
    t = torch.Tensor([max_radius ** 2])
    edges = adj_t.nonzero().t().contiguous()
    indexes = np.random.randint(0, edges.shape[1], 5000)
    edges = edges[:, indexes]

    p = [2, 1, -1]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    func_list = []
    type_list_ = []
    for n in trange(edges.shape[1]):
        embedding_1 = model.a[1, edges[0, n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        embedding_2 = model.a[1, edges[1, n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                              device=device)
        # type = p[type_list[to_numpy(edges[0, n])].astype(int).squeeze()] * p[type_list[to_numpy(edges[1, n])].astype(int).squeeze()]
        # type_list_.append(type)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
    type_list_ = np.array(type_list_)

    qiqj_list = [-2,-1,1,2,4]

    for n in qiqj_list:
        pos = np.argwhere(type_list_==n)
        if len(pos>0):
            func = func_list[pos[:,0]]
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     linewidth=1, alpha=0.1)


        plt.plot(to_numpy(rr),
                 to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(type), linewidth=8, alpha=0.1)
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, 0.02])
    plt.ylim([-0.5E6, 0.5E6])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    fig_ = plt.figure(figsize=(12,12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    func_list=[]
    for n in range(n_clusters):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        if len(pos)>0:
            embedding0 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            for m in range(n_clusters):
                pos = np.argwhere(new_labels == m).squeeze().astype(int)
                if len(pos)>0:
                    embedding1 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                     device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding0, embedding1), dim=1)
                    func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             linewidth=8, alpha=0.5)
                    func_list.append(func * ynorm)
                    # temp = model.psi(rr, p[n], p[m])
                    # plt.plot(to_numpy(rr), np.array(temp.cpu()), linewidth=1)
                    popt, pocv = curve_fit(power_model, to_numpy(rr), to_numpy(func * ynorm), bounds=([0, 1.5], [5., 2.5]))
    plt.xlim([0, 0.01])
    plt.ylim([-2E6, 2E6])
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij}$', fontsize=64)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.tif", dpi=170.7)
    csv_ = to_numpy(torch.stack(func_list))
    csv_ = np.concatenate((csv_,to_numpy(rr[None,:])))
    np.save(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.txt", csv_)
    plt.close()

    ax = fig.add_subplot(3, 3, 6)
    plt.text(-0.25, 1.1, f'f)', ha='right', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (true)', fontsize=12)
    p = config.simulation.params
    if len(p) > 0:
        p = torch.tensor(p, device=device)
    else:
        p = torch.load(f'graphs_data/graphs_{dataset_name}/p.pt')
    psi_output = []
    for m in range(n_particle_types):
        for n in range(n_particle_types):
            temp = model.psi(rr, p[n], p[m])
            plt.plot(to_numpy(rr), np.array(temp.cpu()), linewidth=1)
    plt.xlim([0, 0.02])
    plt.ylim([-0.5E6, 0.5E6])
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)


    fig_ = plt.figure(figsize=(12,12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    csv_ = []
    csv_.append(to_numpy(rr))
    true_func_list = []
    for n in range(n_particle_types):
        for m in range(n_particle_types):
            temp = model.psi(rr, p[n], p[m])
            true_func_list.append(temp)
            plt.plot(to_numpy(rr), np.array(temp.cpu()), linewidth=8, alpha=0.5)
            csv_.append(to_numpy(temp.squeeze()))
    plt.xlim([0, 0.01])
    plt.ylim([-2E6, 2E6])
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij}$', fontsize=64)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}_{epoch}.tif", dpi=170.7)
    np.save(f"./{log_dir}/tmp_training/true_func_{config_file}_{epoch}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/true_func_{config_file}_{epoch}.txt", csv_)
    plt.close()

    func_list = torch.stack(func_list)
    true_func_list = torch.stack(true_func_list)
    rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
    rmserr_list = to_numpy(rmserr_list)
    print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')


    p = [2, 1, -1]
    popt_list = []
    ptrue_list = []
    nn = 0
    for m in range(n_particle_types):
        for n in range(n_particle_types):
            if func_list[nn][10] < 0:
                popt, pocv = curve_fit(power_model, to_numpy(rr),
                                       -to_numpy(func_list[nn]), bounds=([0, 1.5], [5., 2.5]))
            else:
                popt, pocv = curve_fit(power_model, to_numpy(rr),
                                       to_numpy(func_list[nn]), bounds=([0, 1.5], [5., 2.5]))
                popt[0] = -popt[0]
            nn += 1
            popt_list.append(popt)
            ptrue_list.append(-p[n] * p[m])
    bounds0 = np.mean(popt_list, 0)[1] - 1E-10
    bounds1 = np.mean(popt_list, 0)[1] + 1E-10
    popt_list = np.array(popt_list)
    ptrue_list = -np.array(ptrue_list)
    M_p0 = np.reshape(popt_list[:, 0], (3, 3))
    # print(M_p0)
    # print(f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 5)}+/-{np.round(np.std(popt_list[:, 1]), 5)}")
    popt_list = []
    ptrue_list = []
    nn = 0
    for m in range(n_particle_types):
        for n in range(n_particle_types):
            if func_list[nn][10] < 0:
                popt, pocv = curve_fit(power_model, to_numpy(rr),
                                       -to_numpy(func_list[nn]), bounds=([0, bounds0], [5., bounds1]))
            else:
                popt, pocv = curve_fit(power_model, to_numpy(rr),
                                       to_numpy(func_list[nn]), bounds=([0, bounds0], [5., bounds1]))
                popt[0] = -popt[0]
            nn += 1
            popt_list.append(popt)
            ptrue_list.append(-p[n] * p[m])
    popt_list = np.array(popt_list)
    ptrue_list = -np.array(ptrue_list)
    M_ptrue = np.reshape(ptrue_list, (3, 3))
    M_p1 = np.reshape(popt_list[:, 0], (3, 3))
    # print(M_p1)


    # row_sum = np.sum(M_p,0)
    # A,B,C = row_sum[0], row_sum[1], row_sum[2]
    # M = [[-B, A+C, -B], [-C, -C, A+B], [C+B, -A, -A]]
    # b = np.array([0, 0, 0])
    # b = [2 , 1 , -1]
    # np.matmul(M,b)
    # x = linalg.solve(M, b)

    ax = fig.add_subplot(3, 3, 7)
    plt.title(r'Reconstructed charges', fontsize=12)
    plt.text(-0.25, 1.1, f'g)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    x_data = ptrue_list
    y_data = popt_list[:, 0]
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    plt.plot(ptrue_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    plt.scatter(ptrue_list, popt_list[:, 0], color='k', s=20)
    plt.ylabel(r'Predicted $q_i q_j$', fontsize=12)
    plt.text(-1.8, 4.0, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res= np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(-1.8, 3.4, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)
    plt.xlim([-2,5])
    plt.ylim([-2,5])
    print(f'R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}')

    ax = fig.add_subplot(3, 3, 8)
    plt.title(r'Reconstructed exponent', fontsize=12)
    plt.text(-0.25, 1.1, f'h)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.scatter(ptrue_list, -popt_list[:, 1], color='k', s=20)
    plt.ylim([-4, 0])
    plt.ylabel(r'Exponent fit ', fontsize=12)
    plt.text(-2, -0.5, f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 5)}+/-{np.round(np.std(popt_list[:, 1]), 5)}",
             fontsize=10)
    print(f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 5)}+/-{np.round(np.std(popt_list[:, 1]), 5)}")

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    csv_ = []
    csv_.append(ptrue_list)
    csv_.append(popt_list[:, 0])
    plt.plot(ptrue_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
    plt.scatter(ptrue_list, popt_list[:, 0], color='k', s=400)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'True $q_i q_j$', fontsize=64)
    plt.ylabel(r'Reconstructed $q_i q_j$', fontsize=64)
    plt.xlim([-3, 5])
    plt.ylim([-3, 5])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/qiqj_{config_file}.tif", dpi=300)
    csv_ = np.array(csv_)
    np.save(f"./{log_dir}/tmp_training/qiqj_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/qiqj_{config_file}.txt", csv_)
    plt.close

    print(f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 5)}+/-{np.round(np.std(popt_list[:, 1]), 5)}")
    relative_error = np.abs(popt_list[:, 0] - ptrue_list.squeeze()) / np.abs(ptrue_list.squeeze()) * 100
    print (f'all charge relative error: {np.round(np.mean(relative_error), 3)}+/-{np.round(np.std(relative_error), 3)}')

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    csv_ = []
    csv_.append(ptrue_list.squeeze())
    csv_.append(-popt_list[:, 1])
    csv_ = np.array(csv_)
    plt.scatter(ptrue_list, -popt_list[:, 1], color='k', s=400)
    plt.xlim([-3, 5])
    plt.ylim([-4, 0])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'True $q_i q_j$', fontsize=64)
    plt.ylabel(r'Reconstructed exponent', fontsize=64)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/exponent_{config_file}.tif", dpi=300)
    np.save(f"./{log_dir}/tmp_training/exponent_{config_file}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/exponent_{config_file}.txt", csv_)
    plt.close()

    # find last image file in logdir
    ax = fig.add_subplot(3, 3, 9)
    files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        last_file = files[-1]

        image = imageio.imread(last_file)
        plt.text(-0.25, 1.1, f'l)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Rollout inference (frame 2000)', fontsize=12)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    time.sleep(1)
    plt.tight_layout()
    plt.savefig('Fig4.jpg', dpi=300)
    plt.close()
    print(' ')

def data_plot_boids(config_file, device):
    # config_file = 'boids_16'
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    # print(config.pretty())

    cmap = CustomColorMap(config=config)

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    dimension = simulation_config.dimension

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load data ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    n_particles = int(n_particles * (1 - train_config.particle_dropout))

    net_list=['20'] #'0_0','0_2000','0_5000', '0_9800', '5', '20']
    epoch = 20

    for net_ in net_list:

        print (f'Plot net_{net_} ...')

        model, bc_pos, bc_dpos = choose_training_model(config, device)
        model = Interaction_Particles_extract(config, device, aggr_type=config.graph_model.aggr_type, bc_dpos=bc_dpos)

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{net_}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        matplotlib.use("Qt5Agg")

        model_a_first = model.a.clone().detach()
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster, cmap, index_particles, type_list,
                                    n_particle_types, n_particles, ynorm, epoch, log_dir, device)

        it = 7000
        # compute model output for frame 7000
        x = x_list[0][it].clone().detach()
        distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)  # threshold
        adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)) * 1.0
        edge_index = adj_t.nonzero().t().contiguous()
        dataset = data.Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            y, in_features, lin_edge_out = model(dataset, data_id=1, training=False, vnorm=vnorm, phi=torch.zeros(1, device=device))  # acceleration estimation
        y = y * ynorm
        lin_edge_out = lin_edge_out * ynorm

        # compute ground truth output
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        model_B = PDE_B_extract(aggr_type=config.graph_model.aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        psi_output = []
        for n in range(n_particle_types):
            with torch.no_grad():
                psi_output.append(model.psi(rr, torch.squeeze(p[n])))
                y_B, sum, cohesion, alignment, separation, diffx, diffv, rr, type = model_B(dataset)  # acceleration estimation
        type = to_numpy(type)
        # cohesion_ = p[type, 0:1].repeat(1, 2) * model_B.a1 * diffx
        # alignment_ = p[type, 1:2].repeat(1, 2) * model_B.a2 * diffv
        # separation_ = p[type, 2:3].repeat(1, 2) * model_B.a3 * diffx / (rr[:, None].repeat(1, 2))
        # sum_ = cohesion_ + alignment_ + separation_
        # fig_ = plt.figure(figsize=(12, 12))
        # plt.scatter(to_numpy(y_B), to_numpy(y), color='k', s=50, alpha=0.5)
        # fig_ = plt.figure(figsize=(12, 12))
        # plt.scatter(to_numpy(sum), to_numpy(lin_edge_out), color='k', s=50, alpha=0.5)

        cohesion_fit = np.zeros(n_particle_types)
        alignment_fit = np.zeros(n_particle_types)
        separation_fit = np.zeros(n_particle_types)
        for loop in range (2):
            for n in range(n_particle_types):
                pos = np.argwhere(type == n)
                pos = pos[:, 0].astype(int)
                xdiff = to_numpy(diffx[pos, :])
                vdiff = to_numpy(diffv[pos, :])
                rdiff = to_numpy(rr[pos])
                x_data = np.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                y_data = to_numpy(torch.norm(lin_edge_out[pos, :], dim=1))
                if loop==0:
                    lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox')
                else:
                    lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox', p0=p00)
                cohesion_fit[n] = lin_fit[0]
                alignment_fit[n] = lin_fit[1]
                separation_fit[n] = lin_fit[2]
            p00 = [np.mean(cohesion_fit), np.mean(alignment_fit), np.mean(separation_fit)]

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(p[:,0]*model_B.a1), cohesion_fit, color='k', s=50, alpha=0.5)

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(p[:,1]*model_B.a2), alignment_fit, color='k', s=50, alpha=0.5)

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(p[:,2]*model_B.a3), separation_fit, color='k', s=50, alpha=0.5)






        fig_ = plt.figure(figsize=(12, 12))
        ax = fig_.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        for n in range(n_particle_types):
            pos = np.argwhere(type == n)
            pos = pos[:, 0].astype(int)
            plt.scatter(to_numpy(diffx[pos, 0]), to_numpy(lin_edge_out[pos, 0]), color=cmap.color(n), s=50, alpha=0.5)
        # plt.ylim([-5E-5, 5E-5])
        plt.xlabel(r'$x_j-x_i$', fontsize=64)
        plt.ylabel( r'$f_{ij,x}$',fontsize=64)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_all_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        fig_ = plt.figure(figsize=(12, 12))
        ax = fig_.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        for n in range(n_particle_types):
            pos = np.argwhere(type == n)
            pos = pos[:, 0].astype(int)
            plt.scatter(to_numpy(diffx[pos, 0]), to_numpy(sum[pos, 0]), color=cmap.color(n), s=50, alpha=0.5)
        # plt.ylim([-0.08, 0.08])
        # plt.ylim([-5E-5, 5E-5])
        plt.xlabel(r'$x_j-x_i$', fontsize=64)
        plt.ylabel( r'$f_{ij,x}$',fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
        func_list = []
        true_func_list = []
        for n in range(n_particles):
            embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            type = to_numpy(x[n,5]).astype(int)
            true_func = model_B.psi(rr, p[type])
            true_func_list.append(true_func)
            if (n % 10 == 0) :
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(type), linewidth=4, alpha=0.25)
        func_list = torch.stack(func_list)
        true_func_list = torch.stack(true_func_list)
        plt.ylim([-1E-4, 1E-4])
        plt.xlabel(r'$x_j-x_i$', fontsize=64)
        plt.ylabel(r'$f_{ij}$', fontsize=64)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_dij_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        for n in range(n_particle_types):
            true_func  = model_B.psi(rr, p[n])
            plt.plot(to_numpy(rr), to_numpy(true_func), color=cmap.color(n), linewidth=4)
        plt.ylim([-1E-4, 1E-4])
        plt.xlabel(r'$x_j-x_i$', fontsize=64)
        plt.ylabel(r'$f_{ij}$', fontsize=64)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_dij_{config_file}_{net_}.tif", dpi=300)

        func_list = func_list * ynorm
        func_list_ = torch.clamp(func_list, min=torch.tensor(-1.0E-4,device=device), max=torch.tensor(1.0E-4,device=device))
        true_func_list_ = torch.clamp(true_func_list, min=torch.tensor(-1.0E-4, device=device),
                                 max=torch.tensor(1.0E-4, device=device))
        rmserr_list = torch.sqrt(torch.mean((func_list_ - true_func_list_) ** 2,1))
        rmserr_list = to_numpy(rmserr_list)
        print(f'all function RMS error : {np.round(np.mean(rmserr_list), 8)}+/-{np.round(np.std(rmserr_list), 8)}')

        bFit = True
        if bFit:
            cohesion_fit = np.zeros(n_particle_types)
            alignment_fit = np.zeros(n_particle_types)
            separation_fit = np.zeros(n_particle_types)
            for n in range(n_particle_types):
                pos = np.argwhere(new_labels == n)
                pos = pos[:, 0].astype(int)
                xdiff = to_numpy(diffx[pos, :])
                vdiff = to_numpy(diffv[pos, :])
                rdiff = to_numpy(r[pos])
                x_data = np.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                y_data = to_numpy(torch.norm(lin_edge_out[pos, :], dim=1))
                lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox')
                cohesion_fit[n] = lin_fit[0]
                alignment_fit[n] = lin_fit[1]
                separation_fit[n] = lin_fit[2]
            p00 = [np.mean(cohesion_fit), np.mean(alignment_fit), np.mean(separation_fit)]
            for n in range(n_particle_types):
                pos = np.argwhere(new_labels == n)
                pos = pos[:, 0].astype(int)
                xdiff = to_numpy(diffx[pos, :])
                vdiff = to_numpy(diffv[pos, :])
                rdiff = to_numpy(r[pos])
                x_data = np.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                y_data = to_numpy(torch.norm(lin_edge_out[pos, :], dim=1))
                lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox', p0=p00)
                cohesion_fit[n] = lin_fit[0]
                alignment_fit[n] = lin_fit[1]
                separation_fit[n] = lin_fit[2]

            index_classified = np.unique(new_labels)
            threshold = 0.25
            relative_error = np.abs(y_data-x_data)/x_data

            pos = np.argwhere(relative_error<threshold)
            pos_outliers = np.argwhere(relative_error>threshold)
            x_data_ = x_data[pos[:,0]]
            y_data_ = y_data[pos[:,0]]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
            for id, n in enumerate(index_classified):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=20)
            plt.xlabel(r'True cohesion coeff. ', fontsize=12)
            plt.ylabel(r'Predicted cohesion coeff. ', fontsize=12)
            residuals = y_data_ - linear_model(x_data_, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(4E-5, 4.5E-4, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
            plt.text(4E-5, 4.1E-4, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)
            print(f'cohesion Slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error>threshold)} ')

            ax = fig.add_subplot(3, 3, 8)
            plt.text(-0.25, 1.1, f'h)', ha='right', va='top', transform=ax.transAxes, fontsize=12)
            x_data = np.abs(to_numpy(p[:, 1]) * 5E-4)
            y_data = alignment_fit
            x_data = x_data[index_classified]
            y_data = y_data[index_classified]
            relative_error = np.abs(y_data-x_data)/x_data
            pos = np.argwhere(relative_error<threshold)
            pos_outliers = np.argwhere(relative_error>threshold)
            x_data_ = x_data[pos[:,0]]
            y_data_ = y_data[pos[:,0]]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
            for id, n in enumerate(index_classified):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=20)
            plt.xlabel(r'True alignment coeff. ', fontsize=12)
            plt.ylabel(r'Predicted alignment coeff. ', fontsize=12)
            plt.text(5e-3, 0.046, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
            residuals = y_data_ - linear_model(x_data_, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(5e-3, 0.042, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)
            print(f'alignment Slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}   outliers: {np.sum(relative_error>threshold)} ')

            ax = fig.add_subplot(3, 3, 9)
            plt.text(-0.25, 1.1, f'i)', ha='right', va='top', transform=ax.transAxes, fontsize=12)
            x_data = np.abs(to_numpy(p[:, 2]) * 1E-8)
            y_data = separation_fit
            x_data = x_data[index_classified]
            y_data = y_data[index_classified]
            relative_error = np.abs(y_data-x_data)/x_data
            pos = np.argwhere(relative_error<threshold)
            pos_outliers = np.argwhere(relative_error>threshold)
            x_data_ = x_data[pos[:,0]]
            y_data_ = y_data[pos[:,0]]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
            for id, n in enumerate(index_classified):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=20)
            plt.xlabel(r'True separation coeff. ', fontsize=12)
            plt.ylabel(r'Predicted separation coeff. ', fontsize=12)
            plt.text(5e-8, 4.4E-7, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
            residuals = y_data_ - linear_model(x_data_, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(5e-8, 4E-7, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)
            print(f'separation Slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}   outliers: {np.sum(relative_error>threshold)} ')


            threshold = 0.1
            index_classified = np.unique(new_labels)

            fig_ = plt.figure(figsize=(12, 12))
            ax = fig_.add_subplot(1, 1, 1)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
            fmt = lambda x, pos: '{:.1f}e-4'.format((x) * 1e4, pos)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            x_data = np.abs(to_numpy(p[:, 0]) * 0.5E-5)
            y_data = np.abs(cohesion_fit)
            x_data = x_data[index_classified]
            y_data = y_data[index_classified]
            relative_error = np.abs(y_data-x_data)/x_data
            pos = np.argwhere(relative_error<threshold)
            pos_outliers = np.argwhere(relative_error>threshold)
            x_data_ = x_data[pos[:,0]]
            y_data_ = y_data[pos[:,0]]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            for id, n in enumerate(index_classified):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
            plt.xlabel(r'True cohesion coeff. ', fontsize=56)
            plt.ylabel(r'Reconstructed cohesion coeff. ', fontsize=56)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            csv_=[]
            csv_.append(x_data)
            csv_.append(y_data)
            plt.savefig(f"./{log_dir}/tmp_training/cohesion_{config_file}_{net_}.tif", dpi=300)
            np.save(f"./{log_dir}/tmp_training/cohesion_{config_file}_{net_}.npy", csv_)
            np.savetxt(f"./{log_dir}/tmp_training/cohesion_{config_file}_{net_}.txt", csv_)
            plt.close()

            fig_ = plt.figure(figsize=(12, 12))
            ax = fig_.add_subplot(1, 1, 1)
            fmt = lambda x, pos: '{:.1f}e-2'.format((x) * 1e2, pos)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            x_data = np.abs(to_numpy(p[:, 1]) * 5E-4)
            y_data = alignment_fit
            x_data = x_data[index_classified]
            y_data = y_data[index_classified]
            relative_error = np.abs(y_data-x_data)/x_data
            pos = np.argwhere(relative_error<threshold)
            pos_outliers = np.argwhere(relative_error>threshold)
            x_data_ = x_data[pos[:,0]]
            y_data_ = y_data[pos[:,0]]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            for id, n in enumerate(index_classified):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
            plt.xlabel(r'True alignement coeff. ', fontsize=56)
            plt.ylabel(r'Reconstructed alignement coeff. ', fontsize=56)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            csv_=[]
            csv_.append(x_data)
            csv_.append(y_data)
            plt.savefig(f"./{log_dir}/tmp_training/alignment_{config_file}_{net_}.tif", dpi=300)
            np.save(f"./{log_dir}/tmp_training/alignment_{config_file}_{net_}.npy", csv_)
            np.savetxt(f"./{log_dir}/tmp_training/alignement_{config_file}_{net_}.txt", csv_)
            plt.close()

            fig_ = plt.figure(figsize=(12, 12))
            ax = fig_.add_subplot(1, 1, 1)
            fmt = lambda x, pos: '{:.1f}e-7'.format((x) * 1e7, pos)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            x_data = np.abs(to_numpy(p[:, 2]) * 1E-8)
            y_data = separation_fit
            x_data = x_data[index_classified]
            y_data = y_data[index_classified]
            relative_error = np.abs(y_data-x_data)/x_data
            pos = np.argwhere(relative_error<threshold)
            pos_outliers = np.argwhere(relative_error>threshold)
            x_data_ = x_data[pos[:,0]]
            y_data_ = y_data[pos[:,0]]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            for id, n in enumerate(index_classified):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
            plt.xlabel(r'True separation coeff. ', fontsize=56)
            plt.ylabel(r'Reconstructed separation coeff. ', fontsize=56)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            csv_=[]
            csv_.append(x_data)
            csv_.append(y_data)
            plt.savefig(f"./{log_dir}/tmp_training/separation_{config_file}_{net_}.tif", dpi=300)
            np.save(f"./{log_dir}/tmp_training/separation_{config_file}_{net_}.npy", csv_)
            np.savetxt(f"./{log_dir}/tmp_training/separation_{config_file}_{net_}.txt", csv_)
            plt.close()

def data_plot_wave(config_file, cc='viridis'):

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_node_types = simulation_config.n_node_types
    n_frames = config.simulation.n_frames
    node_value_map = simulation_config.node_value_map
    n_runs = config.training.n_runs
    embedding_cluster = EmbeddingCluster(config)
    cmap = CustomColorMap(config=config)
    node_type_map = simulation_config.node_type_map
    has_pic = 'pics' in simulation_config.node_type_map
    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_mesh_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    hnorm = torch.load(f'./log/try_{config_file}/hnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(NGraphs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()

    print(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    x_mesh = x_mesh_list[0][n_frames - 1].clone().detach()
    type_list = x_mesh[:, 5:6].clone().detach()
    n_nodes = x_mesh.shape[0]
    print(f'N nodes: {n_nodes}')

    index_nodes = []
    x_mesh = x_mesh_list[1][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == n)
        index_nodes.append(index.squeeze())

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.use("Qt5Agg")

    if has_pic:
        i0 = imread(f'graphs_data/{simulation_config.node_type_map}')
        coeff = i0[(to_numpy(x_mesh[:, 1]) * 255).astype(int), (to_numpy(x_mesh[:, 2]) * 255).astype(int)] / 255
        coeff_ = coeff
        coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis))
        coeff = np.flipud(coeff) * simulation_config.beta
    else:
        c = initialize_random_values(n_node_types, device)
        for n in range(n_node_types):
            c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
        c = to_numpy(c)
        i0 = imread(f'graphs_data/{node_type_map}')
        values = i0[(to_numpy(x_mesh[:, 1]) * 255).astype(int), (to_numpy(x_mesh[:, 2]) * 255).astype(int)]
        features_mesh = values
        coeff = c[features_mesh]
        coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis)) * simulation_config.beta
        coeff = np.flipud(coeff)
    vm = np.max(coeff)
    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    axf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    axf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, cmap=cc, vmin=0, vmax=vm)
    plt.xlabel(r'$x$', fontsize=64)
    plt.ylabel(r'$y$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    # cbar = plt.colorbar(shrink=0.5)
    # cbar.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/true_wave_coeff_{config_file}.tif", dpi=300)

    net_list=['0_1000','0_2000','0_5000', '1', '5', '20']

    net_list = glob.glob(f"./log/try_{config_file}/models/*.pt")

    for net in net_list:

        # net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{net_}.pt"
        net_ = net.split('graphs')[1]

        mesh_model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        mesh_model.load_state_dict(state_dict['model_state_dict'])
        mesh_model.eval()

        embedding = get_embedding(mesh_model.a, 1)

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if has_pic:
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        color=cmap.color(np.round(coeff_*256).astype(int)), s=100, alpha=1)
        else:
            for n in range(n_node_types):
                    c_ = np.round(n / (n_node_types - 1) * 256).astype(int)
                    plt.scatter(embedding[index_nodes[n], 0], embedding[index_nodes[n], 1], c=cmap.color(c_), s=200, alpha=1)
        plt.xlabel(r'$a_{i0}$', fontsize=32)
        plt.ylabel(r'$a_{i1}$', fontsize=32)
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        rr = torch.tensor(np.linspace(-150, 150, 200)).to(device)
        popt_list = []
        func_list = []
        for n in range(n_nodes):
            embedding_ = mesh_model.a[1, n, :] * torch.ones((200, 2), device=device)
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
            h = mesh_model.lin_phi(in_features.float())
            h = h[:, 0]
            popt, pcov = curve_fit(linear_model, to_numpy(rr.squeeze()), to_numpy(h.squeeze()))
            popt_list.append(popt)
            func_list.append(h)
        func_list = torch.stack(func_list)
        popt_list = np.array(popt_list)

        t = np.array(popt_list) * to_numpy(hnorm)
        t = t[:, 0]
        t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
        t = np.flipud(t)

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
        axf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        axf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.imshow(t, cmap=cc, vmin=0, vmax=vm)
        # plt.xlabel(r'$x$', fontsize=64)
        # plt.ylabel(r'$y$', fontsize=64)
        plt.xlabel('x', fontsize=32)
        plt.ylabel('y', fontsize=32)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        fmt = lambda x, pos: '{:.3%}'.format(x)
        # cbar = plt.colorbar(format=FuncFormatter(fmt),shrink=0.5)
        # cbar.ax.tick_params(labelsize=32)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/wave_coeff_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        if not(has_pic):
            proj_interaction = popt_list
            proj_interaction[:, 1] = proj_interaction[:, 0]
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

            label_list = []
            for n in range(n_node_types):
                tmp = labels[index_nodes[n]]
                label_list.append(np.round(np.median(tmp)))
            label_list = np.array(label_list)
            new_labels = labels.copy()
            for n in range(n_node_types):
                new_labels[labels == label_list[n]] = n
            Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)

            print(f'Accuracy: {Accuracy}  n_clusters: {n_clusters}')

        if False:

            convert_color = [0,2,3,4,1]
            fig_ = plt.figure(figsize=(12, 12))
            for n in range(n_nodes):
                embedding_ = mesh_model.a[1, n, :] * torch.ones((200, 2), device=device)
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
                h = mesh_model.lin_phi(in_features.float())
                h = h[:, 0]
                if (n % 4):
                    if has_pic:
                        plt.plot(to_numpy(rr), to_numpy(h) * to_numpy(hnorm), linewidth=4,
                                 color=cmap.color(np.round(coeff_[n]*256).astype(int)), alpha=0.05)
                    else:
                        # plt.plot(to_numpy(rr), to_numpy(h) * to_numpy(hnorm), linewidth=4, color=cmap.color(new_labels[n]%256), alpha=0.05)
                        c_ = np.round(convert_color[int(to_numpy(type_list[n]))]*256/(n_node_types-1))
                        plt.plot(to_numpy(rr), to_numpy(h) * to_numpy(hnorm), linewidth=4, color=cmap.color(c_.astype(int)), alpha=0.01)
            plt.xlabel(r'$\Delta u_{i}$', fontsize=64)
            plt.ylabel(r'$\Phi (\ensuremath{\mathbf{a}}_i, \Delta u_i)$', fontsize=64)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/phi_{config_file}_{net_}.tif", dpi=300)
            plt.close()

            fig_ = plt.figure(figsize=(12, 12))
            axf = fig_.add_subplot(1, 1, 1)

            pos=torch.argwhere(mask_mesh==1)
            pos=to_numpy(pos[:,0]).astype(int)
            x_data = np.reshape(coeff, (n_nodes))
            y_data = np.reshape(t, (n_nodes))
            x_data = x_data.squeeze()
            y_data = y_data.squeeze()
            x_data = x_data[pos]
            y_data = y_data[pos]

            axf.xaxis.set_major_locator(plt.MaxNLocator(3))
            axf.yaxis.set_major_locator(plt.MaxNLocator(3))
            axf.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axf.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            plt.scatter(x_data, y_data, c='k', s=100, alpha=0.01)
            plt.ylabel(r'Reconstructed coefficients', fontsize=48)
            plt.xlabel(r'True coefficients', fontsize=48)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.xlim([0, vm * 1.1])
            plt.ylim([0, vm * 1.1])

            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/scatter_{config_file}_{net_}.tif", dpi=300)
            plt.close()

            print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}   ")

def data_plot_particle_field(config_file, cc, device):
    print('')

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_node_types = simulation_config.n_node_types
    node_type_map = simulation_config.node_type_map
    node_value_map = simulation_config.node_value_map
    has_video = 'video' in node_value_map
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_frames = simulation_config.n_frames
    has_siren = 'siren' in model_config.field_type
    has_siren_time = 'siren_with_time' in model_config.field_type
    target_batch_size = train_config.batch_size
    has_ghost = train_config.n_ghosts > 0
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)


    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)

    x_list = []
    y_list = []
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_0.pt', map_location=device)
    x_mesh_list.append(x_mesh)
    y_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_0.pt', map_location=device)
    y_mesh_list.append(y_mesh)
    hnorm = torch.load(f'./log/try_{dataset_name}/hnorm.pt', map_location=device).to(device)

    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_0.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)


    # matplotlib.use("Qt5Agg")
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    x_mesh = x_mesh_list[0][0].clone().detach()
    i0 = imread(f'graphs_data/{node_value_map}')
    if has_video:
        i0 = i0[0]
        target = i0[(to_numpy(x_mesh[:, 2]) * 100).astype(int), (to_numpy(x_mesh[:, 1]) * 100).astype(int)]
        target = np.reshape(target, (n_nodes_per_axis, n_nodes_per_axis))
    else:
        target = i0[(to_numpy(x_mesh[:, 1]) * 255).astype(int), (to_numpy(x_mesh[:, 2]) * 255).astype(int)]
        target = np.reshape(target, (n_nodes_per_axis, n_nodes_per_axis))
        target = np.flipud(target)
    vm = np.max(target)
    if vm == 0:
        vm = 0.01

    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.imshow(target, cmap=cc, vmin=0, vmax=vm)
    plt.xlabel(r'$x$', fontsize=64)
    plt.ylabel(r'$y$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=32)
    # cbar.set_label(r'$Coupling$',fontsize=64)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/target_field.tif", dpi=300)
    plt.close()

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
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
    index_nodes = []
    x_mesh = x_mesh_list[0][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == -n - 1)
        index_nodes.append(index.squeeze())

    if has_siren:

        image_width = int(np.sqrt(n_nodes))
        if has_siren_time:
            model_f = Siren_Network(image_width=image_width, in_features=3, out_features=1, hidden_features=128,
                                        hidden_layers=5, outermost_linear=True, device=device, first_omega_0=80,
                                        hidden_omega_0=80.)
        else:
            model_f = Siren_Network(image_width=image_width, in_features=2, out_features=1, hidden_features=64,
                                        hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80,
                                        hidden_omega_0=80.)
        model_f.to(device=device)
        model_f.eval()


    epoch_list = [20]
    for epoch in epoch_list:
        print(f'epoch: {epoch}')

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        if has_siren:
            net = f'./log/try_{config_file}/models/best_model_f_with_1_graphs_{epoch}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])

        embedding = get_embedding(model.a, 1)

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        csv_ = []
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=400, alpha=0.1)
            csv_.append(embedding[index_particles[n], :])
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        # csv_ = np.array(csv_)
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif", dpi=300)
        # np.save(f"./{log_dir}/tmp_training/embedding_{config_file}.npy", csv_)
        # csv_= np.reshape(csv_,(csv_.shape[0]*csv_.shape[1],2))
        # np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}.txt", csv_)
        plt.close()


        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=False, config=config,
                                                                model_lin_edge=model.lin_edge, model_a=model.a,
                                                                dataset_number=1,
                                                                n_particles=int(n_particles*(1-train_config.particle_dropout)), ynorm=ynorm,
                                                                types=to_numpy(x[:, 5]),
                                                                cmap=cmap, device=device)


        match train_config.cluster_method:
            case 'kmeans':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans')
            case 'kmeans_auto_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
            case 'kmeans_auto_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                proj_interaction = embedding
            case 'distance_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
            case 'distance_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=0.05)
                proj_interaction = embedding
            case 'distance_both':
                new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
                labels, n_clusters = embedding_cluster.get(new_projection, 'distance')

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for n in range(n_clusters):
            pos = np.argwhere(labels == n)
            pos = np.array(pos)
            if pos.size > 0:
                print(f'cluster {n}  {len(pos)}')
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=100,alpha=0.1)
        label_list = []
        for n in range(n_particle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)
        plt.xlabel(r'UMAP-proj 0', fontsize=64)
        plt.ylabel(r'UMAP-proj 1', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/UMAP_{config_file}_{epoch}.tif", dpi=300)
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
        type_list = x[:, 5:6].clone().detach()
        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')
        plt.close

        p = config.simulation.params
        if len(p) > 1:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt',map_location=device)
        model_a_first = model.a.clone().detach()

        if False:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1,1,1)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            csv_ = []
            csv_.append(to_numpy(rr))
            rmserr_list = []
            for n in range(int(n_particles*(1-train_config.particle_dropout))):
                embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                match config.graph_model.particle_model_name:
                    case 'PDE_A' | 'PDE_ParticleField_A':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                    case 'PDE_B' | 'PDE_ParticleField_B':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                csv_.append(to_numpy(func))
                true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(), p[to_numpy(type_list[n]).astype(int)].squeeze())
                rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func) ** 2)))
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
            plt.xticks(fontsize=32)
            plt.yticks(fontsize=32)
            # plt.xlabel(r'$d_{ij}$', fontsize=64)
            # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
            plt.xlim([0, max_radius])
            # plt.ylim([-0.15, 0.15])
            plt.ylim([-0.04, 0.03])
            # plt.ylim([-0.1, 0.1])
            # plt.ylim([-0.03, 0.03])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/func_all_{config_file}_{epoch}.tif",dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')
            plt.close()


        match config.graph_model.field_type:

            case 'siren_with_time' | 'siren':

                s_p = 100

                if has_video:

                    x_mesh = x_mesh_list[0][0].clone().detach()
                    i0 = imread(f'graphs_data/{node_value_map}')

                    os.makedirs(f"./{log_dir}/tmp_training/video", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/video/generated1", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/video/generated2", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/video/target", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/video/field", exist_ok=True)

                    print('Output per frame ...')

                    RMSE_list = []
                    PSNR_list = []
                    SSIM_list = []
                    for frame in trange(0, n_frames):
                        x = x_list[0][frame].clone().detach()
                        fig = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 2]), 1-to_numpy(x[index_particles[n], 1]), s=s_p,
                                        color='k')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/video/generated1/generated_1_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 2]), 1-to_numpy(x[index_particles[n], 1]), s=s_p)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/video/generated2/generated_2_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()

                        i0_ = i0[frame]
                        y = i0_[(to_numpy(x_mesh[:, 2]) * 100).astype(int), (to_numpy(x_mesh[:, 1]) * 100).astype(int)]
                        y = np.reshape(y, (n_nodes_per_axis, n_nodes_per_axis))
                        fig_ = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.imshow(y, cmap=cc, vmin=0, vmax=vm)
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/video/target/target_field_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()


                        pred = model_f(time=frame / n_frames) ** 2
                        pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                        pred = to_numpy(torch.sqrt(pred))
                        pred = np.flipud(pred)
                        fig_ = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        pred=np.rot90(pred)
                        pred =np.fliplr(pred)
                        plt.imshow(pred, cmap=cc, vmin=0, vmax=1)
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/video/field/reconstructed_field_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()

                        RMSE = np.sqrt(np.mean((y - pred) ** 2))
                        RMSE_list = np.concatenate((RMSE_list, [RMSE]))
                        PSNR = calculate_psnr(y, pred, max_value=np.max(y))
                        PSNR_list = np.concatenate((PSNR_list, [PSNR]))
                        SSIM = calculate_ssim(y, pred)
                        SSIM_list = np.concatenate((SSIM_list, [SSIM]))

                else:

                    x_mesh = x_mesh_list[0][0].clone().detach()
                    node_value_map = simulation_config.node_value_map
                    n_nodes_per_axis = int(np.sqrt(n_nodes))
                    i0 = imread(f'graphs_data/{node_value_map}')
                    target = i0[(to_numpy(x_mesh[:, 1]) * 255).astype(int), (to_numpy(x_mesh[:, 2]) * 255).astype(
                        int)] * 5000 / 255
                    target = np.reshape(target, (n_nodes_per_axis, n_nodes_per_axis))
                    target = np.flipud(target)

                    os.makedirs(f"./{log_dir}/tmp_training/rotation", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/rotation/generated1", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/rotation/generated2", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/rotation/target", exist_ok=True)
                    os.makedirs(f"./{log_dir}/tmp_training/rotation/field", exist_ok=True)


                    match model_config.field_type:
                        case 'siren':
                            angle_list = [0]
                        case 'siren_with_time':
                            angle_list = trange(0, n_frames, 5)
                    print('Output per angle ...')

                    RMSE_list = []
                    PSNR_list = []
                    SSIM_list = []
                    for angle in angle_list:

                        x=x_list[0][angle].clone().detach()
                        fig = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]), s=s_p, color='k')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/rotation/generated1/generated_1_{epoch}_{angle}.tif", dpi=150)
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]), s=s_p)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/rotation/generated2/generated_2_{epoch}_{angle}.tif", dpi=150)
                        plt.close()
                        y = ndimage.rotate(target, -angle, reshape=False, cval=np.mean(target) * 1.1)
                        fig_ = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.imshow(y, cmap=cc, vmin=0, vmax=vm)
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/rotation/target/target_field_{epoch}_{angle}.tif", dpi=150)
                        plt.close()

                        match model_config.field_type:
                            case 'siren':
                                pred = model_f() ** 2
                            case 'siren_with_time':
                                pred = model_f(time=angle / n_frames) ** 2
                        pred = torch.reshape(pred,(n_nodes_per_axis,n_nodes_per_axis))
                        pred = to_numpy(torch.sqrt(pred))
                        pred = np.flipud(pred)

                        fig_ = plt.figure(figsize=(12, 12))
                        axf = fig_.add_subplot(1, 1, 1)
                        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.imshow(pred, cmap=cc, vmin=0, vmax=vm)
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_training/rotation/field/reconstructed_field_{epoch}_{angle}.tif", dpi=150)
                        plt.close()

                        RMSE= np.sqrt(np.mean((y - pred) ** 2))
                        RMSE_list = np.concatenate((RMSE_list, [RMSE]))
                        PSNR = calculate_psnr(y, pred, max_value=np.max(y))
                        PSNR_list = np.concatenate((PSNR_list, [PSNR]))
                        SSIM = calculate_ssim(y, pred)
                        SSIM_list = np.concatenate((SSIM_list, [SSIM]))

                fig_ = plt.figure(figsize=(12, 12))
                axf = fig_.add_subplot(1, 1, 1)
                plt.scatter(np.linspace(0, n_frames, len(SSIM_list)),SSIM_list, color='k', linewidth=4)
                plt.xlabel(r'$Frame$', fontsize=64)
                plt.ylabel(r'$SSIM$', fontsize=64)
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                plt.ylim([0,1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/ssim_{epoch}.tif", dpi=150)
                plt.close()

                print(f'SSIM: {np.round(np.mean(SSIM_list), 3)}+/-{np.round(np.std(SSIM_list), 3)}')

                fig_ = plt.figure(figsize=(12, 12))
                axf = fig_.add_subplot(1, 1, 1)
                plt.scatter(np.linspace(0, n_frames, len(SSIM_list)),RMSE_list, color='k', linewidth=4)
                plt.xlabel(r'$Frame$', fontsize=64)
                plt.ylabel(r'RMSE', fontsize=64)
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                plt.ylim([0,1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/rmse_{epoch}.tif", dpi=150)
                plt.close()

                fig_ = plt.figure(figsize=(12, 12))
                axf = fig_.add_subplot(1, 1, 1)
                plt.scatter(np.linspace(0, n_frames, len(SSIM_list)),PSNR_list, color='k', linewidth=4)
                plt.xlabel(r'$Frame$', fontsize=64)
                plt.ylabel(r'PSNR', fontsize=64)
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                plt.ylim([0,50])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/psnr_{epoch}.tif", dpi=150)
                plt.close()

            case 'tensor':

                fig_ = plt.figure(figsize=(12, 12))
                axf = fig_.add_subplot(1, 1, 1)
                axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                pts = to_numpy(torch.reshape(model.field[1],(100,100)))
                pts = np.flipud(pts)
                plt.imshow(pts, cmap=cc,vmin=0,vmax=vm)
                plt.xlabel(r'$x$', fontsize=64)
                plt.ylabel(r'$y$', fontsize=64)
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                cbar = plt.colorbar(shrink=0.5)
                cbar.ax.tick_params(labelsize=32)
                # cbar.set_label(r'$Coupling$',fontsize=64)
                plt.tight_layout()
                imsave(f"./{log_dir}/tmp_training/field_pic_{config_file}_{epoch}.tif", pts)
                plt.savefig(f"./{log_dir}/tmp_training/field_{config_file}_{epoch}.tif", dpi=300)
                # np.save(f"./{log_dir}/tmp_training/embedding_{config_file}.npy", csv_)
                # csv_= np.reshape(csv_,(csv_.shape[0]*csv_.shape[1],2))
                # np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}.txt", csv_)
                plt.close()
                rmse = np.sqrt(np.mean((target-pts)**2))
                print(f'RMSE: {rmse}')

                fig_ = plt.figure(figsize=(12, 12))
                axf = fig_.add_subplot(1, 1, 1)
                axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.scatter(target,pts,c='k',s=100, alpha=0.1)
                plt.ylabel(r'Reconstructed coupling', fontsize=32)
                plt.xlabel(r'True coupling', fontsize=32)
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                plt.xlim([-vm*0.1, vm*1.5])
                plt.ylim([-vm*0.1, vm*1.5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field_scatter_{config_file}_{epoch}.tif", dpi=300)

                x_data = np.reshape(pts,(n_nodes))
                y_data = np.reshape(target,(n_nodes))
                threshold = 0.25
                relative_error = np.abs(y_data - x_data)
                print(f'outliers: {np.sum(relative_error > threshold)} / {n_particles}')
                pos = np.argwhere(relative_error < threshold)
                pos_outliers = np.argwhere(relative_error > threshold)

                x_data_ = x_data[pos].squeeze()
                y_data_ = y_data[pos].squeeze()
                # x_data_ = x_data.squeeze()
                # y_data_ = y_data.squeeze()

                lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
                residuals = y_data_ - linear_model(x_data_, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                print(f'R^2$: {np.round(r_squared, 3)} ')
                print(f"Slope: {np.round(lin_fit[0], 2)}")

                plt.plot(x_data_, linear_model(x_data_, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                plt.xlim([-vm*0.1, vm*1.1])
                plt.ylim([-vm*0.1, vm*1.1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field_scatter_{config_file}_{epoch}.tif", dpi=300)

def data_plot_RD(config_file, cc, device):

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_node_types = simulation_config.n_node_types
    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    node_value_map = simulation_config.node_value_map
    aggr_type = config.graph_model.aggr_type
    delta_t = config.simulation.delta_t
    cmap = CustomColorMap(config=config)
    node_type_map = simulation_config.node_type_map
    has_pic = 'pics' in simulation_config.node_type_map

    embedding_cluster = EmbeddingCluster(config)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    hnorm = torch.load(f'./log/try_{config_file}/hnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(NGraphs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()

    print(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    x_mesh = x_mesh_list[0][n_frames - 1].clone().detach()
    type_list = x_mesh[:, 5:6].clone().detach()
    n_nodes = x_mesh.shape[0]
    print(f'N nodes: {n_nodes}')

    index_nodes = []
    x_mesh = x_mesh_list[1][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == n)
        index_nodes.append(index.squeeze())

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.use("Qt5Agg")

    if has_pic:
        i0 = imread(f'graphs_data/{simulation_config.node_type_map}')
        coeff = i0[(to_numpy(x_mesh[:, 1]) * 255).astype(int), (to_numpy(x_mesh[:, 2]) * 255).astype(int)]
        coeff_ = coeff
        coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis))
        coeff = np.flipud(coeff) * simulation_config.beta
    else:
        c = initialize_random_values(n_node_types, device)
        for n in range(n_node_types):
            c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
        c = to_numpy(c)
        i0 = imread(f'graphs_data/{node_type_map}')
        values = i0[(to_numpy(x_mesh[:, 1]) * 255).astype(int), (to_numpy(x_mesh[:, 2]) * 255).astype(int)]
        features_mesh = values
        coeff = c[features_mesh]
        coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis)) * simulation_config.beta
        coeff = np.flipud(coeff)
        coeff = np.fliplr(coeff)
    vm = np.max(coeff)
    print(f'vm: {vm}')
    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    axf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    axf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, cmap=cc, vmin=0, vmax=vm)
    plt.xlabel(r'$x$', fontsize=64)
    plt.ylabel(r'$y$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/true_coeff_{config_file}.tif", dpi=300)
    plt.close()

    net_list = ['20', '0_1000', '0_2000', '0_5000', '1', '5']

    for net_ in net_list:

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{net_}.pt"
        model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print(f'net: {net}')
        embedding = get_embedding(model.a, 1)
        first_embedding = embedding

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if has_pic:
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        color=cmap.color(np.round(coeff_*256).astype(int)), s=100, alpha=1)
        else:
            for n in range(n_node_types):
                    c_ = np.round(n / (n_node_types - 1) * 256).astype(int)
                    plt.scatter(embedding[index_nodes[n], 0], embedding[index_nodes[n], 1], s=200)  # , color=cmap.color(c_)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        if not(has_pic):

            print('domain clustering...')
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            label_list = []
            for n in range(n_node_types):
                tmp = labels[index_nodes[n]]
                label_list.append(np.round(np.median(tmp)))
            label_list = np.array(label_list)
            new_labels = labels.copy()
            for n in range(n_node_types):
                new_labels[labels == label_list[n]] = n
            Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            print(f'Accuracy: {Accuracy}  n_clusters: {n_clusters}')

            model_a_ = model.a[1].clone().detach()
            for n in range(n_clusters):
                pos = np.argwhere(labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    median_center = model_a_[pos, :]
                    median_center = torch.median(median_center, dim=0).values
                    # plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                    model_a_[pos, :] = median_center
                    # plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
            with torch.no_grad():
                model.a[1] = model_a_.clone().detach()

        print('fitting diffusion coeff with domain clustering...')

        if True:

            k=2400

            # collect data from sing
            x_mesh = x_mesh_list[1][k].clone().detach()
            dataset = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
            with torch.no_grad():
                pred, laplacian_uvw, uvw, embedding, input_phi = model(dataset, data_id=1, return_all=True)
            pred = pred * hnorm
            y = y_mesh_list[1][k].clone().detach()

            # RD_RPS_model :
            c_ = torch.ones(n_node_types, 1, device=device) + torch.rand(n_node_types, 1, device=device)
            for n in range(n_node_types):
                c_[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
            c = c_[to_numpy(dataset.x[:, 5])].squeeze()
            u = uvw[:, 0]
            v = uvw[:, 1]
            w = uvw[:, 2]
            # laplacian = mesh_model.beta * c * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
            laplacian_u = c * laplacian_uvw[:, 0]
            laplacian_v = c * laplacian_uvw[:, 1]
            laplacian_w = c * laplacian_uvw[:, 2]
            a = 0.6
            p = u + v + w
            du = laplacian_u + u * (1 - p - a * v)
            dv = laplacian_v + v * (1 - p - a * w)
            dw = laplacian_w + w * (1 - p - a * u)
            increment = torch.cat((du[:, None], dv[:, None], dw[:, None]), dim=1)
            increment = increment.squeeze()

            lin_fit_true = np.zeros((np.max(new_labels)+1, 3, 10))
            lin_fit_reconstructed = np.zeros((np.max(new_labels)+1, 3, 10))
            eq_list = ['u', 'v', 'w']
            if has_pic:
                n_node_types_list=[0]
            else:
                n_node_types_list = np.arange(n_node_types)
            for n in np.unique(new_labels):
                if has_pic:
                    pos = np.argwhere(to_numpy(mask_mesh.squeeze()) == 1)
                else:
                    pos = np.argwhere((new_labels == n) & (to_numpy(mask_mesh.squeeze()) == 1))
                    pos = pos[:,0].astype(int)

                for it, eq in enumerate(eq_list):
                    fitting_model = reaction_diffusion_model(eq)
                    laplacian_u = to_numpy(laplacian_uvw[pos, 0])
                    laplacian_v = to_numpy(laplacian_uvw[pos, 1])
                    laplacian_w = to_numpy(laplacian_uvw[pos, 2])
                    u = to_numpy(uvw[pos, 0])
                    v = to_numpy(uvw[pos, 1])
                    w = to_numpy(uvw[pos, 2])
                    x_data = np.concatenate((laplacian_u[:, None], laplacian_v[:, None], laplacian_w[:, None], u[:, None], v[:, None], w[:, None]), axis=1)
                    y_data = to_numpy(increment[pos, 0+it:1+it])
                    p0 = np.ones((10,1))
                    lin_fit, lin_fitv = curve_fit(fitting_model, np.squeeze(x_data), np.squeeze(y_data), p0=np.squeeze(p0),method='trf')
                    lin_fit_true[n, it] = lin_fit
                    y_data = to_numpy(pred[pos, it:it+1])
                    lin_fit, lin_fitv = curve_fit(fitting_model, np.squeeze(x_data), np.squeeze(y_data), p0=np.squeeze(p0), method='trf')
                    lin_fit_reconstructed[n, it] = lin_fit



            coeff_reconstructed = np.round(np.median(lin_fit_reconstructed, axis=0),2)
            diffusion_coeff_reconstructed = np.round(np.median(lin_fit_reconstructed, axis=1),2)[:,9]
            coeff_true = np.round(np.median(lin_fit_true, axis=0),2)
            diffusion_coeff_true = np.round(np.median(lin_fit_true, axis=1),2)[:,9]

            print(f'frame {k}')
            print (f'coeff_reconstructed: {coeff_reconstructed}')
            print(f'diffusion_coeff_reconstructed: {diffusion_coeff_reconstructed}')
            print(f'coeff_true: {coeff_true}')
            print(f'diffusion_coeff_true: {diffusion_coeff_true}')



            cp = ['uu','uv','uw','vv','vw','ww','u','v','w']
            results = {
                'True': coeff_true[0,0:9],
                'Reconstructed': coeff_reconstructed[0,0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient',fontsize=48)
            ax.set_xticks(x + width, cp,fontsize=48)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.title('First equation',fontsize=48)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/first_equation_{config_file}_{net_}.tif", dpi=300)
            plt.close()
            cp = ['uu','uv','uw','vv','vw','ww','u','v','w']
            results = {
                'True': coeff_true[1,0:9],
                'Reconstructed': coeff_reconstructed[1,0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient',fontsize=48)
            ax.set_xticks(x + width, cp,fontsize=48)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.title('Second equation',fontsize=48)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/second_equation_{config_file}_{net_}.tif", dpi=300)
            plt.close()
            cp = ['uu','uv','uw','vv','vw','ww','u','v','w']
            results = {
                'True': coeff_true[2,0:9],
                'Reconstructed': coeff_reconstructed[2,0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient',fontsize=48)
            ax.set_xticks(x + width, cp,fontsize=48)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.title('Third equation',fontsize=48)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/third_equation_{config_file}_{net_}.tif", dpi=300)
            plt.close()




            fig_ = plt.figure(figsize=(12, 12))
            t =diffusion_coeff_reconstructed [new_labels]
            t_ = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
            t_ = np.flipud(t_)
            t_ = np.fliplr(t_)
            fig_ = plt.figure(figsize=(12, 12))
            axf = fig_.add_subplot(1, 1, 1)
            axf.xaxis.set_major_locator(plt.MaxNLocator(3))
            axf.yaxis.set_major_locator(plt.MaxNLocator(3))
            fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
            axf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            axf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            plt.imshow(t_, cmap=cc, vmin=0, vmax=vm)
            plt.xlabel(r'$x$', fontsize=64)
            plt.ylabel(r'$y$', fontsize=64)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            fmt = lambda x, pos: '{:.3%}'.format(x)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/diff_coeff_map_{config_file}_{net_}.tif", dpi=300)
            plt.close()

            t_ = np.reshape(t, (n_nodes_per_axis*n_nodes_per_axis))
            fig_ = plt.figure(figsize=(12, 12))
            axf = fig_.add_subplot(1, 1, 1)
            axf.xaxis.set_major_locator(plt.MaxNLocator(3))
            axf.yaxis.set_major_locator(plt.MaxNLocator(3))
            axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            plt.scatter(first_embedding[:, 0], first_embedding[:, 1],
                                s=200, c=t_, cmap='viridis', alpha=0.5,vmin=0,vmax=vm)
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{net_}.tif", dpi=300)
            plt.close()

    bContinuous=False
    if bContinuous:
        laplacian_uvw_list=[]
        uvw_list=[]
        pred_list=[]
        input_phi_list=[]
        for k in trange(n_frames-1):
            x_mesh = x_mesh_list[1][k].clone().detach()
            dataset = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
            with torch.no_grad():
                pred, laplacian_uvw, uvw, embedding, input_phi = model(dataset, data_id=1, return_all=True)
            pred = pred * hnorm
            pred_list.append(pred)
            laplacian_uvw_list.append(laplacian_uvw)
            uvw_list.append(uvw)
            input_phi_list.append(input_phi)

        laplacian_uvw_list= torch.stack(laplacian_uvw_list)
        uvw_list = torch.stack(uvw_list)
        pred_list = torch.stack(pred_list)

        print('Fit node level ...')
        t = np.zeros((n_nodes,1))
        for n in trange(n_nodes):
            for it, eq in enumerate(eq_list[0]):
                fitting_model = reaction_diffusion_model(eq)
                laplacian_u = to_numpy(laplacian_uvw_list[:,n, 0].squeeze())
                laplacian_v = to_numpy(laplacian_uvw_list[:,n, 1].squeeze())
                laplacian_w = to_numpy(laplacian_uvw_list[:,n, 2].squeeze())
                u = to_numpy(uvw_list[:,n, 0].squeeze())
                v = to_numpy(uvw_list[:,n, 1].squeeze())
                w = to_numpy(uvw_list[:,n,  2].squeeze())
                x_data = np.concatenate((laplacian_u[:, None], laplacian_v[:, None], laplacian_w[:, None], u[:, None], v[:, None], w[:, None]),axis=1)
                y_data = to_numpy(pred_list[:,n,it:it+1].squeeze())
                lin_fit, lin_fitv = curve_fit(fitting_model, np.squeeze(x_data), y_data,  method='trf')
                t[n]=lin_fit[-1:]

                if ((n % 1000 == 0) | (n == n_nodes - 1)):
                    t_ = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
                    t_ = np.flipud(t_)
                    t_ = np.fliplr(t_)
                    fig_ = plt.figure(figsize=(12, 12))
                    axf = fig_.add_subplot(1, 1, 1)
                    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    axf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    axf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    plt.imshow(t_ * to_numpy(hnorm), cmap=cc, vmin=0, vmax=1)
                    plt.xlabel(r'$x$', fontsize=64)
                    plt.ylabel(r'$y$', fontsize=64)
                    plt.xticks(fontsize=32.0)
                    plt.yticks(fontsize=32.0)
                    fmt = lambda x, pos: '{:.3%}'.format(x)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/diff_node_coeff_{config_file}_{net_}.tif", dpi=300)
                    plt.close()

        input_phi_list = torch.stack(input_phi_list)
        t = np.zeros((n_nodes, 1))
        for node in trange(n_nodes):
            gg=[]
            for sample in range(100):
                k = 1 + np.random.randint(n_frames - 2)
                input = input_phi_list[k,node,:].clone().detach().squeeze()
                input.requires_grad = True
                L = model.lin_phi(input)[sample%3]
                [g] = torch.autograd.grad(L,[input])
                gg.append(g[sample%3])
            t[node]=to_numpy(torch.median(torch.stack(gg)))
            if ((node%1000==0)|(node==n_nodes-1)):
                t_ = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
                t_ = np.flipud(t_)
                t_ = np.fliplr(t_)
                fig_ = plt.figure(figsize=(12, 12))
                axf = fig_.add_subplot(1, 1, 1)
                axf.xaxis.set_major_locator(plt.MaxNLocator(3))
                axf.yaxis.set_major_locator(plt.MaxNLocator(3))
                fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                axf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                axf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                plt.imshow(t_*to_numpy(hnorm), cmap=cc, vmin=0,vmax=vm)
                plt.xlabel(r'$x$', fontsize=64)
                plt.ylabel(r'$y$', fontsize=64)
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                fmt = lambda x, pos: '{:.3%}'.format(x)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/diff_coeff_{config_file}_{net_}.tif", dpi=300)
                plt.close()

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)

        pos = torch.argwhere(mask_mesh == 1)
        pos = to_numpy(pos[:, 0]).astype(int)
        x_data = np.reshape(coeff, (n_nodes))
        y_data = np.reshape(t_, (n_nodes))
        x_data = x_data.squeeze()
        y_data = y_data.squeeze()
        x_data = x_data[pos]
        y_data = y_data[pos]

        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.scatter(x_data, y_data, c='k', s=100, alpha=0.01)
        plt.ylabel(r'Reconstructed diffusion coeff.', fontsize=48)
        plt.xlabel(r'True diffusion coeff.', fontsize=48)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.xlim([0, vm * 1.1])
        plt.ylim([0, vm * 1.1])

        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/scatter_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}")

def data_plot_signal(config_file, cc, device):

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_particle_types = simulation_config.n_particle_types
    aggr_type = config.graph_model.aggr_type
    delta_t = config.simulation.delta_t
    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension

    embedding_cluster = EmbeddingCluster(config)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    NGraphs = min(2,NGraphs)

    x_list = []
    y_list = []
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x = x_list[1][n_frames - 1].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles

    mat = scipy.io.loadmat(simulation_config.connectivity_file)
    adjacency = torch.tensor(mat['A'], device=device)
    adj_t = adjacency > 0
    edge_index = adj_t.nonzero().t().contiguous()
    gt_weight = to_numpy(adjacency[adj_t])
    norm_gt_weight = max(gt_weight)

    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.imshow(to_numpy(adjacency) / norm_gt_weight, cmap=cc, vmin=0, vmax=0.1)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/True_Aij_{config_file}.tif", dpi=300)
    plt.close()
    fig_ = plt.figure(figsize=(12, 12))
    axf = fig_.add_subplot(1, 1, 1)
    axf.xaxis.set_major_locator(plt.MaxNLocator(3))
    axf.yaxis.set_major_locator(plt.MaxNLocator(3))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.imshow(to_numpy(adjacency) / norm_gt_weight, cmap=cc, vmin=0, vmax=0.1)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/True_Aij_bar_{config_file}.tif", dpi=300)
    plt.close()


    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.use("Qt5Agg")

    GT_model, bc_pos, bc_dpos = choose_model(config, device=device)

    net_list = ['20','25','30','39'] # [,'1','5','10'] # , '0', '1', '5']
    # net_list = glob.glob(f"./log/try_{config_file}/models/*.pt")

    for net_ in net_list:

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{net_}.pt"
        # net_ = net.split('graphs')[1]

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{net_}.pt"
        model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.edges = edge_index
        print(f'net: {net}')
        embedding = get_embedding(model.a, 1)

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for n in range(n_particle_types):
                c_ = np.round(n / (n_particle_types - 1) * 256).astype(int)
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=400, alpha=0.1)  # , color=cmap.color(c_)
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{net_}.tif", dpi=300)
        plt.close()


        k = 500
        x = x_list[1][k].clone().detach()
        dataset = data.Data(x=x[:, :], edge_index=model.edges)
        y = y_list[1][k].clone().detach()
        y = y
        pred = model(dataset, data_id=1)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]
        dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        gt_weight = to_numpy(adjacency[adj_t])
        pred_weight = to_numpy(model.weight_ij[adj_t])
        plt.scatter(gt_weight, pred_weight, s=200, c='k')
        x_data=gt_weight
        y_data=pred_weight.squeeze()
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.ylabel('Reconstructed $A_{ij}$ values', fontsize=64)
        plt.xlabel('True network $A_{ij}$ values', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Matrix_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}   offset: {np.round(lin_fit[1], 2)}  ")

        # fig_ = plt.figure(figsize=(12, 12))
        # axf = fig_.add_subplot(1, 1, 1)
        # axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        # axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        # axf.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # axf.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # plt.imshow(to_numpy(model.A), cmap=cc)
        # plt.xticks(fontsize=32.0)
        # plt.yticks(fontsize=32.0)
        # plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/Reconstructed_Aij_{config_file}_{net_}.tif", dpi=300)
        # plt.close()

        # fig_ = plt.figure(figsize=(12, 12))
        # axf = fig_.add_subplot(1, 1, 1)
        # for n in range(n_particles):
        #     embedding_ = torch.tensor(embedding[n, :],device=device)
        #     input =torch.cat((uu[:,None],torch.ones_like(uu[:,None])*embedding_),dim=1)
        #     with torch.no_grad():
        #         func = model.lin_phi(input.float())
        #     plt.plot(to_numpy(uu), to_numpy(func), linewidth=4, alpha=0.1, color=cmap.color(to_numpy(type_list[n])))
        # # plt.scatter(to_numpy(uu), to_numpy(true_func) * to_numpy(ynorm), linewidth=8)
        # plt.ylabel(r'Update', fontsize=48)
        # plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/Update_{config_file}_{net_}.tif", dpi=300)
        # plt.close()


        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        uu = x[:, 6:7].squeeze()
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        uu = torch.tensor(np.linspace(0, 3, 1000)).to(device)
        print(n_particles)
        func_list, proj_interaction = analyze_edge_function(rr=uu, vizualize=True, config=config,
                                                                model_lin_edge=model.lin_phi, model_a=model.a,
                                                                dataset_number=1,
                                                                n_particles=int(n_particles*(1-train_config.particle_dropout)), ynorm=ynorm,
                                                                types=to_numpy(x[:, 5]),
                                                                cmap=cmap, device=device)
        # plt.xlabel(r'$d_{ij}$', fontsize=64)
        # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlabel(r'$u$', fontsize=64)
        plt.ylabel(r'Reconstructed $\Phi(u)$', fontsize=64)
        plt.ylim([-0.25, 0.25])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/phi_u_{config_file}_{net_}.tif",dpi=170.7)
        plt.close()

        embedding_ = model.a[1, :, :]
        u = torch.tensor(0.5,device=device).float()
        u = u * torch.ones((n_particles, 1),device=device)
        in_features = torch.cat((u, embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        func = func[:,0]
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        for n in range(n_particle_types):
            plt.scatter(0 * to_numpy(func[index_particles[n]]), to_numpy(func[index_particles[n]]), s=200, alpha=0.05)
        proj_interaction = to_numpy(func[:,None])
        labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for n in range(n_clusters):
            pos = np.argwhere(labels == n)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(np.ones_like(pos)*0.5, proj_interaction[pos, 0], color=cmap.color(n), s=400,alpha=0.1)
        label_list = []
        for n in range(n_particle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)
        plt.xlabel(r'$u$', fontsize=64)
        plt.ylabel(r'$\Phi(u)$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.ylim([-0.25, 0.25])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/cluster_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        new_labels = labels.copy()
        for n in range(n_particle_types):
            new_labels[labels == label_list[n]] = n
        type_list = x[:, 5:6].clone().detach()
        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')

        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                model_a_[pos, :] = median_center
        with torch.no_grad():
            model.a[1] = model_a_.clone().detach()

        uu = torch.tensor(np.linspace(0, 3, 1000)).to(device)
        p = config.simulation.params
        if len(p) > 1:
            p = torch.tensor(p, device=device)
        fig_ = plt.figure(figsize=(12, 12))
        for n in range(n_particle_types):
            phi = -p[n,0]*uu + p[n,1]*torch.tanh(uu)
            plt.plot(to_numpy(uu), to_numpy(phi), linewidth=8)
        plt.xlabel(r'$u$', fontsize=64)
        plt.ylabel(r'True $\Phi(u)$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.ylim([-0.25, 0.25])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_phi_u_{config_file}_{net_}.tif",dpi=170.7)
        plt.close()

        uu = torch.tensor(np.linspace(0, 3, 1000)).to(device)
        with torch.no_grad():
            func = model.lin_edge(uu[:,None].float())
        true_func = torch.tanh(uu[:,None].float())

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        plt.xlabel(r'$u$', fontsize=64)
        plt.ylabel(r'Reconstructed $f(u)$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.scatter(to_numpy(uu), to_numpy(func), linewidth=8, c='k', label='Reconstructed')
        plt.ylim([-3,3])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/f_u_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        plt.xlabel(r'$u$', fontsize=64)
        plt.ylabel(r'True $f(u)$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.scatter(to_numpy(uu), to_numpy(true_func), linewidth=8, c='k', label='Reconstructed')
        plt.ylim([-3,3])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_f_u_{config_file}_{net_}.tif", dpi=300)
        plt.close()

        bFit=False

        if bFit:
            uu = torch.tensor(np.linspace(0, 3, 1000)).to(device)
            in_features = uu[:, None]
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
                func = func[:, 0]

            uu = uu.to(dtype=torch.float32)
            func = func.to(dtype=torch.float32)
            dataset = {}
            dataset['train_input'] = uu[:, None]
            dataset['test_input'] = uu[:, None]
            dataset['train_label'] = func[:, None]
            dataset['test_label'] = func[:, None]

            model_pysrr = PySRRegressor(
                niterations=30,  # < Increase me for better results
                binary_operators=["+", "*"],
                unary_operators=[
                    "cos",
                    "exp",
                    "sin",
                    "tanh"
                ],
                random_state=0,
                temp_equation_file=True
            )

            model_pysrr.fit(to_numpy(dataset["train_input"]), to_numpy(dataset["train_label"]))

            print(model_pysrr)
            print(model_pysrr.equations_)

            k = 500
            x = x_list[1][k].clone().detach()
            dataset = data.Data(x=x[:, :], edge_index=model.edges)
            y = y_list[1][k].clone().detach()
            y = y
            pred = model(dataset, data_id=1)
            adj_t = adjacency > 0
            edge_index = adj_t.nonzero().t().contiguous()
            edge_attr_adjacency = adjacency[adj_t]
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
            fig_ = plt.figure(figsize=(12, 12))
            axf = fig_.add_subplot(1, 1, 1)
            axf.xaxis.set_major_locator(plt.MaxNLocator(3))
            axf.yaxis.set_major_locator(plt.MaxNLocator(3))
            axf.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axf.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            gt_weight = to_numpy(adjacency[adj_t])
            pred_weight = to_numpy(model.weight_ij[adj_t]) * -1.878
            plt.scatter(gt_weight, pred_weight, s=200, c='k')
            x_data=gt_weight
            y_data=pred_weight.squeeze()
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.ylabel('Reconstructed $A_{ij}$ values', fontsize=64)
            plt.xlabel('True network $A_{ij}$ values', fontsize=64)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/Matrix_bis_{config_file}_{net_}.tif", dpi=300)
            plt.close()

            print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}   offset: {np.round(lin_fit[1], 2)}  ")


            fig_ = plt.figure(figsize=(12, 12))
            axf = fig_.add_subplot(1, 1, 1)
            plt.xlabel(r'$u$', fontsize=64)
            plt.ylabel(r'Reconstructed $f(u)$', fontsize=64)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.plot(to_numpy(uu), to_numpy(true_func), linewidth=20, c='g', label='True')
            plt.plot(to_numpy(uu), to_numpy(func)/ -1.878, linewidth=8, c='k', label='Reconstructed')
            plt.legend(fontsize=32.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/comparison_f_u_{config_file}_{net_}.tif", dpi=300)
            plt.close()


            uu = torch.tensor(np.linspace(0, 3, 1000)).to(device)
            fig_ = plt.figure(figsize=(12, 12))
            n = 0
            pos = np.argwhere(labels == n).squeeze().astype(int)
            func = torch.mean(func_list[pos,:],dim=0)
            true_func = -to_numpy(uu) * to_numpy(p[n,0]) + to_numpy(p[n,1]) * np.tanh(to_numpy(uu))
            plt.plot(to_numpy(uu), true_func, linewidth=20, label='True', c='orange') #xkcd:sky blue') #'orange') #
            plt.plot(to_numpy(uu), to_numpy(func), linewidth=8, c='k', label='Reconstructed')
            plt.xlabel(r'$u$', fontsize=64)
            plt.ylabel(r'Reconstructed $\Phi_1(u)$', fontsize=64)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.legend(fontsize=32.0)
            plt.ylim([-0.25, 0.25])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/comparison_phi_1_{config_file}_{net_}.tif", dpi=300)
            plt.close()

            uu = uu.to(dtype=torch.float32)
            func = func.to(dtype=torch.float32)
            dataset = {}
            dataset['train_input'] = uu[:, None]
            dataset['test_input'] = uu[:, None]
            dataset['train_label'] = func[:, None]
            dataset['test_label'] = func[:, None]

            model_pysrr = PySRRegressor(
                niterations=300,  # < Increase me for better results
                binary_operators=["+", "*"],
                unary_operators=[
                    "tanh"
                ],
                random_state=0,
                temp_equation_file=True
            )
            model_pysrr.fit(to_numpy(dataset["train_input"]), to_numpy(dataset["train_label"]))

            print(model_pysrr)
            print(model_pysrr.equations_)

            # for col in model_pysrr.equations_.columns:
            #     print(col)

        k = 500
        x = x_list[1][k].clone().detach()
        dataset = data.Data(x=x[:, :], edge_index=model.edges)
        y = y_list[1][k].clone().detach()
        y = y
        pred, msg, phi, input_phi = model(dataset, data_id=1, return_all=True)
        u_j = model.u_j
        activation = model.activation
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]
        dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
        du_gt, msg_gt, phi_gt = GT_model(dataset, return_all=True)
        u_j_gt = GT_model.u_j
        activation_gt = GT_model.activation
        uu = x[:, 6:7].squeeze()

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(uu), to_numpy(msg + phi), s=100)
        plt.scatter(to_numpy(uu), to_numpy(phi), s=20)
        plt.scatter(to_numpy(uu), to_numpy(msg), s=20)
        # plt.scatter(to_numpy(uu), to_numpy(msg_gt+phi_gt), s=40, c='r')
        plt.xlim([0, 3])
        plt.ylim([0, 1])
        plt.savefig(f"./{log_dir}/tmp_training/model_{config_file}_{net_}.tif", dpi=300)

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(uu), to_numpy(msg_gt+phi_gt), s=100)
        plt.scatter(to_numpy(uu), to_numpy(phi_gt), s=20)
        plt.scatter(to_numpy(uu), to_numpy(msg_gt), s=20)
        plt.xlim([0, 3])
        plt.ylim([0, 1])
        plt.savefig(f"./{log_dir}/tmp_training/true_{config_file}_{net_}.tif", dpi=300)

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(uu), to_numpy(msg + phi), s=100)
        plt.scatter(to_numpy(uu), to_numpy(msg_gt+phi_gt), s=20)
        plt.savefig(f"./{log_dir}/tmp_training/comparison_all_{config_file}_{net_}.tif", dpi=300)


        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(msg_gt), to_numpy(msg), s=20, c='k')
        plt.savefig(f"./{log_dir}/tmp_training/comparison_msg_{config_file}_{net_}.tif", dpi=300)

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(u_j_gt), to_numpy(activation_gt), s=20)
        plt.scatter(to_numpy(u_j), to_numpy(activation), s=20)
        plt.scatter(to_numpy(uu), to_numpy(phi_gt), s=20)
        plt.scatter(to_numpy(uu), to_numpy(phi), s=20)
        plt.savefig(f"./{log_dir}/tmp_training/funky_comparison_{config_file}_{net_}.tif", dpi=300)

        fig_ = plt.figure(figsize=(12, 12))
        plt.scatter(to_numpy(uu), to_numpy(phi), s=400, c='g', label='True')
        plt.scatter(to_numpy(uu), to_numpy(phi_gt), s=20, c='k', label='Reconstructed')
        plt.xlabel(r'$u$', fontsize=64)
        plt.ylabel(r'$f(u)$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.legend(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/phi_u_{config_file}_{net_}.tif", dpi=300)
        plt.close()


        # model_kan = KAN(width=[1, 1], grid=5, k=3, seed=0)
        # model_kan.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
        # lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
        # model_kan.auto_symbolic(lib=lib)
        # model_kan.train(dataset, steps=20)
        # formula, variables = model_kan.symbolic_formula()
        # print(formula)
        #
        # model_kan = KAN(width=[1, 5, 1], grid=5, k=3, seed=0)
        # model_kan.train(dataset, opt="LBFGS", steps=50, lamb=0.01, lamb_entropy=10.)
        # model_kan = model_kan.prune()
        # model_kan.train(dataset, opt="LBFGS", steps=50);
        # for k in range(10):
        #     lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
        #     model_kan.auto_symbolic(lib=lib)
        #     model_kan.train(dataset, steps=100)
        #     formula, variables = model_kan.symbolic_formula()
        #     print(formula)

def data_video_validation(config_file, device):
    print('')

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Save movie ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"./graphs_data/graphs_{dataset_name}/generated_data/*")
    N_files = len(graph_files)
    recons_files = glob.glob(f"{log_dir}/tmp_recons/*")

    # import cv2
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(f"video/validation_{dataset_name}.avi", fourcc, 20.0, (1024, 2048))

    os.makedirs(f"video_tmp/{config_file}", exist_ok=True)

    for n in trange(N_files):
        generated = imread(graph_files[n])
        reconstructed = imread(recons_files[n])
        frame = np.concatenate((generated[:,:,0:3], reconstructed[:,:,0:3]), axis=1)
        # out.write(frame)
        imsave(f"video_tmp/{config_file}/{dataset_name}_{10000+n}.tif", frame)

    # Release the video writer
    # out.release()

    # print("Video saved as 'output.avi'")

def data_video_training(config_file, device):
    print('')

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = config.simulation.max_radius
    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name


    print(f'Save movie ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"./graphs_data/graphs_{dataset_name}/generated_data/*")

    embedding = imread(f"{log_dir}/embedding.tif")
    function = imread(f"{log_dir}/function.tif")
    # field = imread(f"{log_dir}/field.tif")

    matplotlib.use("Qt5Agg")

    os.makedirs(f"video_tmp/{config_file}_training", exist_ok=True)

    for n in trange(embedding.shape[0]):
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(embedding[n, :, :, 0:3])
        plt.xlabel(r'$a_{i0}$', fontsize=32)
        plt.ylabel(r'$a_{i1}$', fontsize=32)
        plt.xticks([])
        plt.yticks([])
        match config_file:
            case 'wave_slit':
                if n<50:
                    plt.text(0, 1.1, f'epoch = 0,   it = {n*200}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
                else:
                    plt.text(0, 1.1, f'Epoch={n-49}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
            case 'arbitrary_3':
                if n < 17:
                    plt.text(0, 1.1, f'epoch = 0,   it = {n * 200}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
                else:
                    plt.text(0, 1.1, f'epoch = {n - 16}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
            case 'arbitrary_3_field_video_bison_siren_with_time':
                if n<13*3:
                    plt.text(0, 1.1, f'epoch= {n//13} ,   it = {(n%13)*500}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
                else:
                    plt.text(0, 1.1, f'epoch = {n-13*3+3}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
            case 'arbitrary_64_256':
                if n<51:
                    plt.text(0, 1.1, f'epoch= 0 ,   it = {n*200}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
                else:
                    plt.text(0, 1.1, f'epoch = {n-50}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
            case 'boids_16_256' | 'gravity_16':
                if n<50:
                    plt.text(0, 1.1, f'epoch = 0,   it = {n*200}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
                else:
                    plt.text(0, 1.1, f'Epoch={n-49}', ha='left', va='top', transform=ax.transAxes, fontsize=32)

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(function[n, :, :, 0:3])
        # plt.ylabel(r'$f(a_i,d_{ij})$', fontsize=32)
        # plt.xlabel(r'$d_{ij}$', fontsize=32)
        plt.ylabel('x', fontsize=32)
        plt.xlabel('y', fontsize=32)
        plt.xticks(fontsize=16.0)
        plt.yticks(fontsize=16.0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        # fmt = lambda x, pos: '{:.3f}'.format(x / 1000 * max_radius, pos)
        # ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

        match config_file:
            case 'wave_slit':
                fmt = lambda x, pos: '{:.1f}'.format((x / 1000), pos)
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                fmt = lambda x, pos: '{:.1f}'.format((1 - x / 1000), pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'arbitrary_3_field_video_bison_siren_with_time':
                fmt = lambda x, pos: '{:.2f}'.format(-x / 1000 * 0.7 + 0.3, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'arbitrary_3':
                fmt = lambda x, pos: '{:.2f}'.format(-x / 1000 * 0.7 + 0.3, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'arbitrary_64_256':
                fmt = lambda x, pos: '{:.2f}'.format(-x/1000*0.7+0.3, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'boids_16_256':
                fmt = lambda x, pos: '{:.2f}e-4'.format((-x/1000+0.5)*2, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'boids_16_256' | 'gravity_16':
                fmt = lambda x, pos: '{:.1f}e5'.format((1-x / 1000) * 5, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

        # ax = fig.add_subplot(1, 3, 3)
        # ax.imshow(field[n, :, :, 0:3],cmap='grey')

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f"video_tmp/{config_file}_training/training_{config_file}_{10000+n}.tif", dpi=64)
        plt.close()

    # plt.text(0, 1.05, f'Frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
    # ax.tick_params(axis='both', which='major', pad=15)


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    # config_list = ['boids_16_256_bison_siren_with_time_2']
    config_list = ['boids_16_256']
    # config_list = ['wave_slit_test']
    # config_list = ['Coulomb_3_256']

    for config_file in config_list:

        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

        epoch_list = [20]

        match config.graph_model.particle_model_name:
            case 'PDE_A':
                data_plot_attraction_repulsion(config_file, epoch_list, device)
            case 'PDE_A_bis':
                data_plot_attraction_repulsion_asym(config_file, epoch_list, device)
            case 'PDE_B':
                data_plot_boids(config_file, device)
            case 'PDE_E':
                data_plot_Coulomb(config_file, device)
            case 'PDE_ParticleField_B' | 'PDE_ParticleField_A':
                data_plot_particle_field(config_file, 'grey', device)

        match config.graph_model.mesh_model_name:
            case 'WaveMesh':
                data_plot_wave(config_file,cc='viridis', device=device)




        # data_plot_attraction_repulsion_short(config_file, device=device)
        # data_plot_boids(config_file)
        # data_plot_gravity(config_file)
        # data_plot_RD(config_file,cc='viridis')
        # data_plot_particle_field(config_file, mode='figures', cc='grey', device=device)
        # data_plot_wave(config_file,cc='viridis')
        # data_plot_signal(config_file,cc='viridis')

        # data_video_validation(config_file,device=device)
        # data_video_training(config_file,device=device)
