import numpy as np
import umap
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as pyg_utils
import imageio.v2 as imageio
from matplotlib import rc

from ParticleGraph.utils import set_size
from scipy.ndimage import median_filter

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# from data_loaders import *

from GNN_particles_Ntype import *

from ParticleGraph.fitting_models import *
from ParticleGraph.sparsify import *
from ParticleGraph.models.utils import *
from ParticleGraph.models.MLP import *
from ParticleGraph.utils import to_numpy, CustomColorMap
import matplotlib as mpl
from io import StringIO
import sys
from scipy.stats import pearsonr
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.mixture import GaussianMixture
import warnings
import seaborn as sns

# from pysr import PySRRegressor


class Interaction_Particle_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, config, device, aggr_type=None, bc_dpos=None):

        super(Interaction_Particle_extract, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        config.simulation = config.simulation
        config.graph_model = config.graph_model
        config.training = config.training

        self.device = device
        self.input_size = config.graph_model.input_size
        self.output_size = config.graph_model.output_size
        self.hidden_dim = config.graph_model.hidden_dim
        self.n_layers = config.graph_model.n_mp_layers
        self.n_particles = config.simulation.n_particles
        self.max_radius = config.simulation.max_radius
        self.rotation_augmentation = config.training.rotation_augmentation
        self.noise_level = config.training.noise_level
        self.embedding_dim = config.graph_model.embedding_dim
        self.n_dataset = config.training.n_runs
        self.prediction = config.graph_model.prediction
        self.update_type = config.graph_model.update_type
        self.n_layers_update = config.graph_model.n_layers_update
        self.hidden_dim_update = config.graph_model.hidden_dim_update
        self.sigma = config.simulation.sigma
        self.model = config.graph_model.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(config.training.n_ghosts)
        self.n_particles_max = config.simulation.n_particles_max

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        if config.simulation.has_cell_division:
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, self.n_particles_max, 2)), device=self.device,
                             requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)),
                             device=self.device,
                             requires_grad=True, dtype=torch.float32))

        if self.update_type != 'none':
            self.lin_update = MLP(input_size=self.output_size + self.embedding_dim + 2, output_size=self.output_size,
                                  nlayers=self.n_layers_update, hidden_size=self.hidden_dim_update, device=self.device)

    def forward(self, data=[], data_id=[], training=[], vnorm=[], phi=[], has_field=False):

        self.data_id = data_id
        self.vnorm = vnorm
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        self.has_field = has_field

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pos = x[:, 1:3]
        d_pos = x[:, 3:5]
        particle_id = x[:, 0:1]
        if has_field:
            field = x[:, 6:7]
        else:
            field = torch.ones_like(x[:, 6:7])

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, particle_id=particle_id, field=field)

        return pred, self.in_features, self.lin_edge_out

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, particle_id_i, particle_id_j, field_j):
        # squared distance
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm

        if self.rotation_augmentation & (self.training == True):
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
            case 'PDE_B' | 'PDE_B_bis' | 'PDE_Cell_B':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                                         dpos_y_j[:, None], embedding_i), dim=-1)
            case 'PDE_G':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None],
                                         dpos_x_j[:, None], dpos_y_j[:, None], embedding_j), dim=-1)
            case 'PDE_GS':
                in_features = torch.cat((r[:, None], embedding_j), dim=-1)
            case 'PDE_E':
                in_features = torch.cat(
                    (delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)

        out = self.lin_edge(in_features) * field_j

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


class model_qiqj(nn.Module):

    def __init__(self, size=None, device=None):

        super(model_qiqj, self).__init__()

        self.device = device
        self.size = size

        self.qiqj = nn.Parameter(torch.randn((int(self.size), 1), device=self.device,requires_grad=True, dtype=torch.float32))


    def forward(self):

        x = []
        for l in range(self.size):
            for m in range(l,self.size,1):
                x.append(self.qiqj[l] * self.qiqj[m])

        return torch.stack(x)


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
        acc = self.propagate(edge_index, x=x)

        sum = self.cohesion + self.alignment + self.separation

        return acc, sum, self.cohesion, self.alignment, self.separation, self.diffx, self.diffv, self.r, self.type

    def message(self, x_i, x_j):
        r = torch.sum(self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, dim=1)  # distance squared

        pp = self.p[to_numpy(x_i[:, 5]), :]

        cohesion = pp[:, 0:1].repeat(1, 2) * self.a1 * self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3])
        alignment = pp[:, 1:2].repeat(1, 2) * self.a2 * self.bc_dpos(x_j[:, 3:5] - x_i[:, 3:5])
        separation = pp[:, 2:3].repeat(1, 2) * self.a3 * self.bc_dpos(x_i[:, 1:3] - x_j[:, 1:3]) / (
            r[:, None].repeat(1, 2))

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

        config.simulation = config.simulation
        config.graph_model = config.graph_model

        self.device = device
        self.input_size = config.graph_model.input_size
        self.output_size = config.graph_model.output_size
        self.hidden_size = config.graph_model.hidden_dim
        self.nlayers = config.graph_model.n_mp_layers
        self.embedding_dim = config.graph_model.embedding_dim
        self.nparticles = config.simulation.n_particles
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
        return discrete_laplacian[:, None] * uvw_j

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r


def load_training_data(dataset_name, n_runs, log_dir, device):
    x_list = []
    y_list = []
    print('load data ...')
    time.sleep(0.5)
    for run in trange(n_runs):
        # check if path exists
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
            y = torch.tensor(y, dtype=torch.float32, device=device)

        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device).squeeze()
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device).squeeze()
    print("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = []
    y = []

    return x_list, y_list, vnorm, ynorm


def plot_embedding_func_cluster_tracking(model, config, embedding_cluster, cmap, index_particles, indexes, type_list,
                                n_particle_types, n_particles, ynorm, epoch, log_dir, embedding_type, style, device):

    if embedding_type == 1:
        embedding = to_numpy(model.a.clone().detach())
        embedding = embedding[indexes.astype(int)]
        fig, ax = fig_init()
        for n in range(n_particle_types):
            pos = np.argwhere(type_list == n).squeeze().astype(int)
            plt.scatter(embedding[pos, 0], embedding[pos, 1], s=1, alpha=0.25)
        if 'latex' in style:
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
        else:
            plt.xlabel(r'$a_{i0}$', fontsize=68)
            plt.ylabel(r'$a_{i1}$', fontsize=68)
        plt.xlim(config.plotting.embedding_lim)
        plt.ylim(config.plotting.embedding_lim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/all_embedding_{epoch}.tif", dpi=170.7)
        plt.close()
    else:
        fig, ax = fig_init()
        for k in trange(0, config.simulation.n_frames - 2):
            embedding = to_numpy(model.a[k * n_particles:(k + 1) * n_particles, :].clone().detach())
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=1,
                            color=cmap.color(n), alpha=0.025)
        if 'latex' in style:
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
        else:
            plt.xlabel(r'$a_{i0}$', fontsize=68)
            plt.ylabel(r'$a_{i1}$', fontsize=68)
        plt.xlim(config.plotting.embedding_lim)
        plt.ylim(config.plotting.embedding_lim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/all_embedding_{epoch}.tif", dpi=170.7)
        plt.close()

    func_list, proj_interaction = analyze_edge_function_tracking(rr=[], vizualize=False, config=config,
                                                        model_MLP=model.lin_edge, model_a=model.a,
                                                        n_particles=n_particles, ynorm=ynorm,
                                                        indexes=indexes, type_list = type_list,
                                                        cmap=cmap, embedding_type = embedding_type, device=device)

    fig, ax = fig_init()
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
    if embedding_type == 1:
        for n in range(n_particle_types):
            pos = np.argwhere(type_list == n).squeeze().astype(int)
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=1, alpha=0.25)
    else:
        for n in range(n_particle_types):
            plt.scatter(proj_interaction[index_particles[n], 0],
                        proj_interaction[index_particles[n], 1], color=cmap.color(n), s=1, alpha=0.25)
    plt.xlabel(r'UMAP 0', fontsize=68)
    plt.ylabel(r'UMAP 1', fontsize=68)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/UMAP_{epoch}.tif", dpi=170.7)
    plt.close()

    embedding = to_numpy(model.a.clone().detach())
    if embedding_type == 1:
        embedding = embedding[indexes.astype(int)]
    else:
        embedding = embedding[0:n_particles]


    labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
                                                      config.training.cluster_distance_threshold, type_list,
                                                      n_particle_types, embedding_cluster)

    accuracy = metrics.accuracy_score(type_list, new_labels)

    fig, ax = fig_init()
    for n in np.unique(labels):
        pos = np.argwhere(labels == n).squeeze().astype(int)
        plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=1, alpha=0.25)

    return accuracy, n_clusters, new_labels


def plot_embedding_func_cluster_state(model, config,embedding_cluster, cmap, type_list, type_stack, id_list,
                                n_particle_types, ynorm, epoch, log_dir, style, device):

    fig, ax = fig_init()
    for n in range(n_particle_types):
        pos = torch.argwhere(type_stack == n).squeeze()
        plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=1, color=cmap.color(n), alpha=0.25)
    if 'latex' in style:
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
    else:
        plt.xlabel(r'$a_{i0}$', fontsize=68)
        plt.ylabel(r'$a_{i1}$', fontsize=68)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/embedding_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    func_list, true_type_list, short_model_a_list, proj_interaction = analyze_edge_function_state(rr=[], config=config,
                                                        model=model,
                                                        id_list=id_list, type_list=type_list, ynorm=ynorm,
                                                        cmap=cmap, visualize=True, device=device)
    plt.savefig(f"./{log_dir}/results/function_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    for n in range(n_particle_types):
        pos = np.argwhere(true_type_list == n).squeeze().astype(int)
        if len(pos)>0:
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=100, alpha=0.25, edgecolors='none')
    plt.xlabel(r'UMAP 0', fontsize=68)
    plt.ylabel(r'UMAP 1', fontsize=68)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/UMAP_{epoch}.tif", dpi=170.7)
    plt.close()


    embedding = proj_interaction
    labels, n_clusters, new_labels = sparsify_cluster_state(config.training.cluster_method, proj_interaction, embedding,
                                                      config.training.cluster_distance_threshold, true_type_list,
                                                      n_particle_types, embedding_cluster)

    fig, ax = fig_init()
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        if len(pos)>0:
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=10, alpha=0.25)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()
    plt.close()

    accuracy = metrics.accuracy_score(true_type_list, new_labels)

    # calculate type for all nodes

    fig, ax = fig_init()
    median_center_list = []
    for n in range(n_clusters):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        pos = np.array(pos)
        if pos.size > 0:
            median_center = short_model_a_list[pos, :]
            plt.scatter(to_numpy(short_model_a_list[pos,0]),to_numpy(short_model_a_list[pos,1]))
            median_center = torch.mean(median_center, dim=0)
            plt.scatter(to_numpy(median_center[0]), to_numpy(median_center[1]), s=100, color='black')
            median_center_list.append(median_center)
    median_center_list = torch.stack(median_center_list)
    median_center_list = median_center_list.to(dtype=torch.float32)
    plt.close()
    distance = torch.sum((model.a[:, None, :] - median_center_list[None, :, :]) ** 2, dim=2)
    result = distance.min(dim=1)
    min_index = result.indices
    new_labels = to_numpy(min_index).astype(int)
    accuracy = metrics.accuracy_score(to_numpy(type_stack.squeeze()), new_labels)

    return accuracy, n_clusters, new_labels


def plot_embedding_func_cluster(model, config,embedding_cluster, cmap, index_particles, type_list,
                                n_particle_types, n_particles, ynorm, epoch, log_dir, alpha, style, device):

    fig, ax = fig_init()
    if config.training.do_tracking:
        embedding = to_numpy(model.a[0:n_particles])
    else:
        embedding = get_embedding(model.a, 1)
    if config.training.particle_dropout > 0:
        embedding = embedding[0:n_particles]

    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                    cmap=cc)
    else:
        for n in range(n_particle_types):
            pos = torch.argwhere(type_list == n)
            pos = to_numpy(pos)
            if len(pos) > 0:
                plt.scatter(embedding[pos, 0], embedding[pos, 1], color=cmap.color(n), s=10)
    if 'latex' in style:
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
    else:
        plt.xlabel(r'$a_{i0}$', fontsize=68)
        plt.ylabel(r'$a_{i1}$', fontsize=68)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/embedding_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    if 'PDE_N' in config.graph_model.signal_model_name:
        model_MLP_ = model.lin_phi
    else:
        model_MLP_ = model.lin_edge
    func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config, model_MLP=model_MLP_, model_a=model.a, type_list=to_numpy(type_list), n_particles=n_particles, dataset_number=1, ynorm=ynorm, cmap=cmap, device=device)
    plt.close()

    # trans = umap.UMAP(n_neighbors=100, n_components=2, init='spectral').fit(func_list_)
    # proj_interaction = trans.transform(func_list_)
    # tsne = TSNE(n_components=2, random_state=0)
    # proj_interaction =  tsne.fit_transform(func_list_)

    fig, ax = fig_init()
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (
                np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
    for n in range(n_particle_types):
        pos = torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            plt.scatter(proj_interaction[pos, 0],
                        proj_interaction[pos, 1], color=cmap.color(n), s=200, alpha=0.1)
    plt.xlabel(r'UMAP 0', fontsize=68)
    plt.ylabel(r'UMAP 1', fontsize=68)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/UMAP_functions_{epoch}.tif", dpi=170.7)
    plt.close()

    config.training.cluster_distance_threshold = 0.01
    labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
                                                      config.training.cluster_distance_threshold, type_list,
                                                      n_particle_types, embedding_cluster)
    accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)

    fig, ax = fig_init()
    for n in range(n_clusters):
        pos = np.argwhere(labels == n)
        if pos.size > 0:
            plt.scatter(embedding[pos, 0], embedding[pos, 1], s=10)
    if 'latex' in style:
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
    else:
        plt.xlabel(r'$a_{i0}$', fontsize=68)
        plt.ylabel(r'$a_{i1}$', fontsize=68)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/clustered_embedding_{epoch}.tif", dpi=170.7)
    plt.close()


    # model_a_ = model.a[1].clone().detach()
    # for n in range(n_clusters):
    #     pos = np.argwhere(labels == n).squeeze().astype(int)
    #     pos = np.array(pos)
    #     if pos.size > 0:
    #         median_center = model_a_[pos, :]
    #         median_center = torch.median(median_center, dim=0).values
    #         model_a_[pos, :] = median_center
    #
    # embedding = to_numpy(model_a_)
    # fig, ax = fig_init()
    # for n in range(n_clusters):
    #     pos = np.argwhere(labels == n)
    #     if pos.size > 0:
    #         plt.scatter(embedding[pos, 0], embedding[pos, 1],s=10)
    # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
    # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
    # plt.tight_layout()
    # plt.savefig(f"./{log_dir}/results/UMAP_clustered_embedding_{epoch}.tif", dpi=170.7)
    # plt.close()

    return accuracy, n_clusters, new_labels


def plot_focused_on_cell(config, run, style, step, cell_id, device):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_ghost = config.training.n_ghosts > 0
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    dimension = simulation_config.dimension
    has_siren = 'siren' in model_config.field_type
    has_siren_time = 'siren_with_time' in model_config.field_type
    has_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)

    l_dir = get_log_dir(config)
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file.split('/')[-1]))
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    print('Load data ...')

    x_list = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)


    mass_time_series = get_time_series(x_list, cell_id, feature='mass')
    vx_time_series = get_time_series(x_list, cell_id, feature='velocity_x')
    vy_time_series = get_time_series(x_list, cell_id, feature='velocity_y')
    v_time_series = np.sqrt(vx_time_series ** 2 + vy_time_series ** 2)
    stage_time_series = get_time_series(x_list, cell_id, feature="stage")
    stage_time_series_color = ["blue" if i == 0 else "orange" if i == 1 else "green" if i == 2 else "pink" for i in stage_time_series]


    for it in trange(0,n_frames,step):

        x = x_list[it].clone().detach()

        T1 = x[:, 5:6].clone().detach()
        H1 = x[:, 6:8].clone().detach()
        X1 = x[:, 1:3].clone().detach()

        index_particles = get_index_particles(x, n_particle_types, dimension)

        pos_cell = torch.argwhere(x[:,0] == cell_id)

        if len(pos_cell)>0:

            if 'latex' in style:
                plt.rcParams['text.usetex'] = True
                rc('font', **{'family': 'serif', 'serif': ['Palatino']})

            if 'color' in style:

                # # matplotlib.use("Qt5Agg")
                matplotlib.rcParams['savefig.pad_inches'] = 0
                fig = plt.figure(figsize=(24, 12))
                ax = fig.add_subplot(1, 2, 1)
                ax.xaxis.get_major_formatter()._usetex = False
                ax.yaxis.get_major_formatter()._usetex = False
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                index_particles = []
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    index_particles.append(pos)
                    # plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                    #             s=marker_size, color=cmap.color(n))

                    size = set_size(x, index_particles[n], 10)

                    plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                s=size, color=cmap.color(n))
                    
                dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
                if len(dead_cell) > 0:
                    plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]), to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                                s=2, color=mc, alpha=0.5)
                if 'latex' in style:
                    plt.xlabel(r'$x$', fontsize=68)
                    plt.ylabel(r'$y$', fontsize=68)
                    plt.xticks(fontsize=48.0)
                    plt.yticks(fontsize=48.0)

                elif 'frame' in style:
                    plt.xlabel(r'x_i', fontsize=13)
                    plt.ylabel('y', fontsize=16)
                    plt.xticks(fontsize=16.0)
                    plt.yticks(fontsize=16.0)
                    ax.tick_params(axis='both', which='major', pad=15)
                    plt.text(0, 1.05,
                             f'frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ',
                             ha='left', va='top', transform=ax.transAxes, fontsize=16)

                plt.xticks([])
                plt.yticks([])

                center_x = to_numpy(x[pos_cell, 1])
                center_y = to_numpy(x[pos_cell, 2])
                plt.xlim([center_x - 0.1, center_x + 0.1])
                plt.ylim([center_y - 0.1, center_y + 0.1])

                ax = fig.add_subplot(2, 2, 2)
                plt.plot(mass_time_series, color=mc, ls="--")

                plt.scatter([i for i in range(it)], mass_time_series[0:it], color=stage_time_series_color[0:it], s=15)
                # plt.plot(mass_time_series[0:it], color = color,linewidth=3)
                plt.ylim(0, max(mass_time_series) + 50)

                ax = fig.add_subplot(2, 2, 4)
                plt.plot(v_time_series, color=mc, ls="--")
                plt.plot(v_time_series[0:it], color = 'red',linewidth=4)

                num = f"{it:06}"

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/cell_{cell_id}_frame_{num}.tif", dpi=80)
                plt.close()


def plot_generated(config, run, style, step, device):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_ghost = config.training.n_ghosts > 0
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    dimension = simulation_config.dimension
    has_siren = 'siren' in model_config.field_type
    has_siren_time = 'siren_with_time' in model_config.field_type
    has_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)

    l_dir = get_log_dir(config)
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file.split('/')[-1]))
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    print('Load data ...')

    x_list = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)


    for it in trange(0,n_frames,step):

        x = x_list[it].clone().detach()

        T1 = x[:, 5:6].clone().detach()
        H1 = x[:, 6:8].clone().detach()
        X1 = x[:, 1:3].clone().detach()

        if 'black' in style:
            plt.style.use('dark_background')

        if 'latex' in style:
            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})


        if 'voronoi' in style:
            # matplotlib.use("Qt5Agg")
            matplotlib.rcParams['savefig.pad_inches'] = 0

            vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            plt.xticks([])
            plt.yticks([])
            index_particles = []

            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='white', line_width=4, line_alpha=1, point_size=0)

            if 'color' in style:
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    index_particles.append(pos)

                    size = set_size(x, index_particles[n], 10) / 10

                    patches = []
                    for i in index_particles[n]:
                        cell = vertices_per_cell[i]
                        vertices = to_numpy(vertices_pos[cell, :])
                        patches.append(Polygon(vertices, closed=True))

                    pc = PatchCollection(patches, alpha=0.75, facecolors=cmap.color(n))
                    ax.add_collection(pc)
                    if 'center' in style:
                        plt.scatter(to_numpy(X1[index_particles[n], 0]), to_numpy(X1[index_particles[n], 1]), s=size,
                                    color=cmap.color(n))

            if 'vertices' in style:
                plt.scatter(to_numpy(vertices_pos[:, 0]), to_numpy(vertices_pos[:, 1]), s=5, color='w')

            plt.xlim([0.5, 0.55])
            plt.ylim([0.5, 0.55])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/detail.tif", dpi=85.35)
            plt.close()


            im = imread(f"./{log_dir}/detail.tif")


            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            plt.xticks([])
            plt.yticks([])
            index_particles = []

            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='white', line_width=1, line_alpha=0.5, point_size=0)
            if 'color' in style:
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    index_particles.append(pos)

                    size = set_size(x, index_particles[n], 10) / 10

                    patches = []
                    for i in index_particles[n]:
                        cell = vertices_per_cell[i]
                        vertices = to_numpy(vertices_pos[cell, :])
                        patches.append(Polygon(vertices, closed=True))

                    pc = PatchCollection(patches, alpha=0.75, facecolors=cmap.color(n))
                    ax.add_collection(pc)
                    if 'center' in style:
                        plt.scatter(to_numpy(X1[index_particles[n], 0]), to_numpy(X1[index_particles[n], 1]), s=size,
                                    color=cmap.color(n))

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            ax = fig.add_subplot(3, 3, 1)
            ax.imshow(im)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()


            num = f"{it:06}"
            plt.savefig(f"./{log_dir}/tmp_recons/frame_{num}.tif", dpi=85.35)

            plt.close()

        else:

            if 'color' in style:

                matplotlib.rcParams['savefig.pad_inches'] = 0
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.get_major_formatter()._usetex = False
                ax.yaxis.get_major_formatter()._usetex = False
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                index_particles = []
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    index_particles.append(pos)
                    # plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                    #             s=marker_size, color=cmap.color(n))

                    size = 10 # set_size(x, index_particles[n], 10) / 10

                    plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                s=size, color=cmap.color(n))
                dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
                if len(dead_cell) > 0:
                    plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]), to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                                s=2, color=mc, alpha=0.5)
                if 'latex' in style:
                    plt.xlabel(r'$x$', fontsize=68)
                    plt.ylabel(r'$y$', fontsize=68)
                    plt.xticks(fontsize=48.0)
                    plt.yticks(fontsize=48.0)
                elif 'frame' in style:
                    plt.xlabel(r'x_i', fontsize=13)
                    plt.ylabel('y', fontsize=16)
                    plt.xticks(fontsize=16.0)
                    plt.yticks(fontsize=16.0)
                    ax.tick_params(axis='both', which='major', pad=15)
                    plt.text(0, 1.05,
                             f'frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ',
                             ha='left', va='top', transform=ax.transAxes, fontsize=16)
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.tight_layout()
                num = f"{it:06}"
                plt.savefig(f"./{log_dir}/tmp_recons/frame_{num}.tif", dpi=80)
                plt.close()

            elif 'bw' in style:

                matplotlib.rcParams['savefig.pad_inches'] = 0
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.get_major_formatter()._usetex = False
                ax.yaxis.get_major_formatter()._usetex = False
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                index_particles = []
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    index_particles.append(pos)
                    # plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                    #             s=marker_size, color=cmap.color(n))
                    size = 10
                    plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                s=size, color='w')
                dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
                if len(dead_cell) > 0:
                    plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]), to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                                s=2, color=mc, alpha=0.5)
                if 'latex' in style:
                    plt.xlabel(r'$x$', fontsize=68)
                    plt.ylabel(r'$y$', fontsize=68)
                    plt.xticks(fontsize=48.0)
                    plt.yticks(fontsize=48.0)
                elif 'frame' in style:
                    plt.xlabel(r'x_i', fontsize=13)
                    plt.ylabel('y', fontsize=16)
                    plt.xticks(fontsize=16.0)
                    plt.yticks(fontsize=16.0)
                    ax.tick_params(axis='both', which='major', pad=15)
                    plt.text(0, 1.05,
                             f'frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ',
                             ha='left', va='top', transform=ax.transAxes, fontsize=16)
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.tight_layout()
                num = f"{it:06}"
                plt.savefig(f"./{log_dir}/tmp_recons/frame_{num}.tif", dpi=80)
                plt.close()


def plot_confusion_matrix(index, true_labels, new_labels, n_particle_types, epoch, it, fig, ax, style):
    # print(f'plot confusion matrix epoch:{epoch} it: {it}')
    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    confusion_matrix = metrics.confusion_matrix(true_labels, new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if n_particle_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False, colorbar=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d', colorbar=False)
    accuracy = metrics.accuracy_score(true_labels, new_labels)
    plt.title(f'accuracy: {np.round(accuracy, 2)}', fontsize=12)
    # print(f'accuracy: {np.round(accuracy,3)}')
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlabel(r'Predicted label', fontsize=12)
    plt.ylabel(r'True label', fontsize=12)

    return accuracy


def plot_cell_rates(config, device, log_dir, n_particle_types, type_list, x_list, new_labels, cmap, logger, style):

    n_frames = config.simulation.n_frames
    delta_t = config.simulation.delta_t

    cell_cycle_length = np.array(config.simulation.cell_cycle_length)
    if len(cell_cycle_length) == 1:
        cell_cycle_length = to_numpy(torch.load(f'graphs_data/graphs_{config.dataset}/cycle_length.pt', map_location=device))


    print('plot cell rates ...')
    N_cells_alive = np.zeros((n_frames, n_particle_types))
    N_cells_dead = np.zeros((n_frames, n_particle_types))

    if os.path.exists(f"./{log_dir}/results/x_.npy"):
        x_ = np.load(f"./{log_dir}/results/x_.npy")
        N_cells_alive = np.load(f"./{log_dir}/results/cell_alive.npy")
        N_cells_dead = np.load(f"./{log_dir}/results/cell_dead.npy")
    else:
        for it in trange(n_frames):

            x = x_list[0][it].clone().detach()
            particle_index = to_numpy(x[:, 0:1]).astype(int)
            x[:, 5:6] = torch.tensor(new_labels[particle_index], device=device)
            if it == 0:
                x_=x_list[0][it].clone().detach()
            else:
                x_=torch.concatenate((x_,x),axis=0)

            for k in range(n_particle_types):
                pos = torch.argwhere((x[:, 5:6] == k) & (x[:, 6:7] == 1))
                N_cells_alive[it, k] = pos.shape[0]
                pos = torch.argwhere((x[:, 5:6] == k) & (x[:, 6:7] == 0))
                N_cells_dead[it, k] = pos.shape[0]

        x_list=[]
        x_ = to_numpy(x_)

        print('save data ...')

        np.save(f"./{log_dir}/results/cell_alive.npy", N_cells_alive)
        np.save(f"./{log_dir}/results/cell_dead.npy", N_cells_dead)
        np.save(f"./{log_dir}/results/x_.npy", x_)

    print('plot results ...')

    last_frame_growth = np.argwhere(np.diff(N_cells_alive[:, 0], axis=0))
    last_frame_growth = last_frame_growth[-1] - 1
    N_cells_alive = N_cells_alive[0:int(last_frame_growth), :]
    N_cells_dead = N_cells_dead[0:int(last_frame_growth), :]

    fig, ax = fig_init()
    for k in range(n_particle_types):
        plt.plot(np.arange(last_frame_growth), N_cells_alive[:, k], color=cmap.color(k), linewidth=4,
                 label=f'Cell type {k} alive')
    plt.xlabel(r'Frame', fontsize=64)
    plt.ylabel(r'Number of cells', fontsize=64)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/cell_alive.tif", dpi=300)
    plt.close()

    fig, ax = fig_init()
    for k in range(n_particle_types):
        plt.plot(np.arange(last_frame_growth), N_cells_dead[:, k], color=cmap.color(k), linewidth=4,
                 label=f'Cell type {k} dead')
    plt.xlabel(r'Frame', fontsize=68)
    plt.ylabel(r'Number of dead cells', fontsize=68)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/cell_dead.tif", dpi=300)
    plt.close()

    #         6,7 H1 cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
    #         8 A1 cell age dim=1

    division_list = {}
    for n in np.unique(new_labels):
        division_list[n] = []
    for n in trange(len(type_list)):
        pos = np.argwhere(x_[:, 0:1] == n)
        if len(pos)>0:
            division_list[new_labels[n]].append(len(pos)* delta_t)

    reconstructed_cell_cycle_length = np.zeros(n_particle_types)
    for k in range(n_particle_types):
        print(f'Cell type {k} division rate: {np.mean(division_list[k])}+/-{np.std(division_list[k])}')
        logger.info(f'Cell type {k} division rate: {np.mean(division_list[k])}+/-{np.std(division_list[k])}')
        reconstructed_cell_cycle_length[k] = np.mean(division_list[k])

    x_data = cell_cycle_length
    y_data = reconstructed_cell_cycle_length.squeeze()
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')
    logger.info(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')

    fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
    plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
    plt.scatter(cell_cycle_length,reconstructed_cell_cycle_length, color=cmap.color(np.arange(n_particle_types)), s=200)
    plt.xlabel(r'True cell cycle length', fontsize=54)
    plt.ylabel(r'Learned cell cycle length', fontsize=54)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/cell_cycle_length.tif", dpi=170)
    plt.close()


    division_list = {}
    for n in np.unique(new_labels):
        division_list[n] = []
    for n in trange(n_frames):
        x = x_list[0][n].clone().detach()
        pos = torch.argwhere(x[:, 7:8] == 0)
        if pos.shape[0]>1:
            x = x[pos]
            for x_ in x:
                division_list[x_[5]].append(x_[8])


def plot_attraction_repulsion(config,epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    n_runs = config.training.n_runs
    has_particle_dropout = config.training.particle_dropout > 0
    dataset_name = config.dataset

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id])//1E7 == 2:
                flag = False
            file_id += 1

        file_id_list0 = np.arange(0,1200,20)
        file_id_list1 = np.arange(1200,file_id,(file_id-40)//60)
        file_id_list2 = np.arange(file_id, len(files), (len(files)-file_id) // 100)
        file_id_list = np.concatenate((file_id_list0,file_id_list1, file_id_list2))

        for file_id_ in trange(0,len(file_id_list)):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:
                epoch = files[file_id].split('graphs')[1][1:-3]
                print(epoch)

                net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')

                fig, ax = fig_init(fontsize=24)
                params = {'mathtext.default': 'regular'}
                plt.rcParams.update(params)
                embedding = get_embedding(model.a, 1)
                for n in range(n_particle_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n)
                    pos = to_numpy(pos)
                    if len(pos) > 0:
                        plt.scatter(embedding[pos, 0], embedding[pos, 1], color=cmap.color(n), s=100, alpha=0.1)
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                match config.dataset:
                    case 'arbitrary_3':
                        plt.xlim([0.5, 1.5])
                        plt.ylim([0.5, 1.5])
                    case 'arbitrary_16':
                        plt.xlim([-2.5, 2.5])
                        plt.ylim([-2.5, 2.5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()


                fig, ax = fig_init(fontsize=24)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                    embedding_ = model.a[1,n] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
                plt.xlabel('$d_{ij}$', fontsize=48)
                plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                plt.xlim([0, max_radius])
                plt.ylim(config.plotting.ylim)
                plt.tight_layout()
                match config.dataset:
                    case 'arbitrary_3':
                        plt.ylim([-0.04, 0.03])
                    case 'arbitrary_16':
                        plt.ylim([-0.1, 0.1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
                plt.close()

    else:
        for epoch in epoch_list:

            net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
            print(f'network: {net}')
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            config.training.cluster_method = 'distance_plot'
            config.training.cluster_distance_threshold = 0.01
            alpha=0.1
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            config.training.cluster_method = 'distance_embedding'
            config.training.cluster_distance_threshold = 0.01
            alpha = 0.1
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            fig, ax = fig_init()
            p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
            rmserr_list = []
            for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                      p[to_numpy(type_list[n]).astype(int)].squeeze())
                rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_function_{epoch}.tif", dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            plt.close()

            fig, ax = fig_init()
            plots = []
            plots.append(rr)
            for n in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
                plots.append(model.psi(rr, p[n], p[n]).squeeze())
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
            plt.close()


def plot_falling_particles(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    n_runs = config.training.n_runs
    has_particle_dropout = config.training.particle_dropout > 0
    dataset_name = config.dataset

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id])//1E7 == 2:
                flag = False
            file_id += 1

        file_id_list0 = np.arange(0,1200,20)
        file_id_list1 = np.arange(1200,file_id,(file_id-40)//60)
        file_id_list2 = np.arange(file_id, len(files), (len(files)-file_id) // 100)
        file_id_list = np.concatenate((file_id_list0,file_id_list1, file_id_list2))

        for file_id_ in trange(0,len(file_id_list)):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:
                epoch = files[file_id].split('graphs')[1][1:-3]
                print(epoch)

                net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')

                fig, ax = fig_init(fontsize=24)
                params = {'mathtext.default': 'regular'}
                plt.rcParams.update(params)
                embedding = get_embedding(model.a, 1)
                for n in range(n_particle_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n)
                    pos = to_numpy(pos)
                    if len(pos) > 0:
                        plt.scatter(embedding[pos, 0], embedding[pos, 1], color=cmap.color(n), s=100, alpha=0.1)
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                match config.dataset:
                    case 'arbitrary_3':
                        plt.xlim([0.5, 1.5])
                        plt.ylim([0.5, 1.5])
                    case 'arbitrary_16':
                        plt.xlim([-2.5, 2.5])
                        plt.ylim([-2.5, 2.5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()


                fig, ax = fig_init(fontsize=24)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                    embedding_ = model.a[1,n] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
                plt.xlabel('$d_{ij}$', fontsize=48)
                plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                plt.xlim([0, max_radius])
                plt.ylim(config.plotting.ylim)
                plt.tight_layout()
                match config.dataset:
                    case 'arbitrary_3':
                        plt.ylim([-0.04, 0.03])
                    case 'arbitrary_16':
                        plt.ylim([-0.1, 0.1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
                plt.close()

    else:
        for epoch in epoch_list:

            net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
            print(f'network: {net}')
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            config.training.cluster_method = 'distance_plot'
            config.training.cluster_distance_threshold = 0.01
            alpha=0.1
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')


            config.training.cluster_method = 'kmeans_auto_embedding'
            config.training.cluster_distance_threshold = 0.01
            alpha = 0.1
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            fig, ax = fig_init()
            p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
            rmserr_list = []
            for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                      p[to_numpy(type_list[n]).astype(int)].squeeze())
                rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_function_{epoch}.tif", dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            plt.close()

            fig, ax = fig_init()
            plots = []
            plots.append(rr)
            for n in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
                plots.append(model.psi(rr, p[n], p[n]).squeeze())
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
            plt.close()


def plot_cell_state(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    n_runs = config.training.n_runs
    n_frames = config.simulation.n_frames
    do_tracking = config.training.do_tracking
    has_cell_division = config.simulation.has_cell_division

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))

    type_list = torch.load(f'graphs_data/{dataset_name}/type_list_1.pt', map_location=device).squeeze()

    for k in trange(n_frames + 1):
        type = x_list[1][k][:, 5]
        if k == 0:
            type_stack = type
        else:
            type_stack = torch.cat((type_stack, type), 0)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id])//1E7 == 2:
                flag = False
            file_id += 1

        file_id_list0 = np.arange(0,file_id,2)
        file_id_list1 = np.arange(file_id, len(files), (len(files)-file_id) // 100)
        file_id_list = np.concatenate((file_id_list0,file_id_list1))

        for file_id_ in trange(0,len(file_id_list)):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:
                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')

                fig, ax = fig_init(fontsize=24)
                for n in range(n_particle_types):
                    pos = torch.argwhere(type_list == n)
                    if len(pos) > 0:
                        plt.scatter(to_numpy(model.a[1][pos, 0]), to_numpy(model.a[1][pos, 1]), s=2, color=cmap.color(n),
                                    alpha=0.5, edgecolor='none')
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()



                max_radius = 0.04
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1,1,1)
                rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)

                if len(type_list) > 10000:
                    step = len(type_list) // 100
                else:
                    step = 5

                for n in range(1, len(type_list), step):
                    embedding_ = model.a[1, n, :] * torch.ones((1000, dimension), device=device)
                    match config.graph_model.particle_model_name:
                        case 'PDE_Cell_A':
                            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                     rr[:, None] / max_radius, embedding_), dim=1)
                        case 'PDE_Cell_A_area':
                            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                     rr[:, None] / max_radius, torch.ones_like(rr[:, None]) * 0.1,
                                                     torch.ones_like(rr[:, None]) * 0.4, embedding_, embedding_), dim=1)
                        case 'PDE_Cell_B':
                            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None],
                                                     0 * rr[:, None],
                                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                        case 'PDE_Cell_B_area':
                            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None],
                                                     0 * rr[:, None],
                                                     0 * rr[:, None], 0 * rr[:, None],
                                                     torch.ones_like(rr[:, None]) * 0.001,
                                                     torch.ones_like(rr[:, None]) * 0.001, embedding_, embedding_),
                                                    dim=1)

                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(int(type_list[n])), linewidth=2)
                plt.xlim([-max_radius, max_radius])
                plt.ylim(config.plotting.ylim)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                plt.xlabel('$d_{ij}$', fontsize=48)
                plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/func_{epoch}.tif", dpi=80)
                plt.close()


    else:

        for epoch in epoch_list:

            net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
            print(f'network: {net}')
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            accuracy, n_clusters, new_labels = plot_embedding_func_cluster_state(model, config,embedding_cluster,
                                                                           cmap, type_list, type_stack, id_list,
                                                                           n_particle_types, ynorm, epoch,
                                                                           log_dir, style, device)


            print(f'result accuracy: {accuracy:0.3f}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(f'result accuracy: {accuracy:0.3f}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            fig, ax = fig_init()
            plots = []
            p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
            plots.append(rr)
            for n in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
                plots.append(model.psi(rr, p[n], p[n]).squeeze())
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
            plt.close()

            learned_time_series = np.reshape(new_labels, (n_frames + 1, n_particles))
            GT_time_series = np.reshape(to_numpy(type_stack), (n_frames + 1, n_particles))

            c_map = ['#1f77b4', '#ff7f0e', '#2ca02c']
            cm = matplotlib.colors.ListedColormap(c_map)

            selected = np.concatenate((np.arange(0,80),np.arange(1600,1680),np.arange(3200,3280)),axis=0)

            fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
            plt.imshow(np.rot90(GT_time_series[:,selected.astype(int)]), aspect='auto', cmap=cm,vmin=0, vmax=2)
            plt.xlabel('frame', fontsize=68)
            plt.ylabel('cell_id', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_kinograph.tif", dpi=170.7)
            plt.close()

            fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
            plt.imshow(np.rot90(learned_time_series[:,selected.astype(int)]), aspect='auto', cmap=cm,vmin=0, vmax=2)
            plt.xlabel('frame', fontsize=68)
            plt.ylabel('cell_id', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_kinograph.tif", dpi=170.7)
            plt.close()


def plot_cell_tracking(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    min_radius = config.simulation.min_radius
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    n_runs = config.training.n_runs
    n_frames = config.simulation.n_frames
    delta_t = config.simulation.delta_t

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))

    n_particles_max = 0
    id_list = []
    type_list = []
    for k in range(n_frames):
        type = x_list[0][k][:, 5]
        type_list.append(type)
        if k == 0:
            type_stack = type
        else:
            type_stack = torch.cat((type_stack, type), 0)
        ids = x_list[0][k][:, -1]
        id_list.append(ids)
        n_particles_max += len(type)
    config.simulation.n_particles_max = n_particles_max + 1
    n_particles = n_particles_max / n_frames

    index_particles = get_index_particles(x_list[0][0], n_particle_types, dimension)

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm

    accuracy_list_=[]
    tracking_index_list_=[]
    tracking_errors_list_=[]

    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_0_graphs_*.pt")
        files.sort(key=sort_key)

        file_id_list =  np.arange(0, len(files),len(files)//50)

        for file_id_ in trange(0,len(file_id_list)):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:

                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_0_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')
                indices = np.random.choice(model.a.shape[0], 1000000, replace=False)
                fig, ax = fig_init(fontsize=24)
                plt.scatter(to_numpy(model.a[indices, 0]), to_numpy(model.a[indices, 1]), s=10, color='w')
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()
    else:

        for epoch in epoch_list:

            net = f"{log_dir}/models/best_model_with_0_graphs_{epoch}.pt"
            print(f'network: {net}')
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            fig = plt.figure(figsize=(8, 8))
            tracking_index = 0
            tracking_index_list = []

            for k in trange(n_frames):
                x = x_list[0][k].clone().detach()
                distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edges = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x[:, :], edge_index=edges)

                pred = model(dataset, training=True, phi=torch.zeros(1, device=device))

                x_next = x_list[1][k + 1]
                x_pos_next = x_next[:, 1:3].clone().detach()
                if config.graph_model.prediction == '2nd_derivative':
                    x_pos_pred = (x[:, 1:3] + delta_t * (x[:, 3:5] + delta_t * pred * ynorm))
                else:
                    x_pos_pred = (x[:, 1:3] + delta_t * pred * ynorm)
                distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_distance_value = result.values
                min_index = result.indices

                first_cell_id = to_numpy(x[:,0])
                next_cell_id = to_numpy(x_next[min_index,0])

                for n in range(n_particle_types):
                    plt.scatter(first_cell_id[index_particles[n]], next_cell_id[index_particles[n]], s=10, color=cmap.color(n), alpha=0.05)

                tracking_index += np.sum((first_cell_id==next_cell_id)*1.0) / n_particles * 100
                tracking_index_list.append(np.sum((first_cell_id==next_cell_id)*1.0) / n_particles * 100)
                x_list[1][k + 1][min_index, 0:1] = x_list[1][k][:, 0:1].clone().detach()

                fig = plt.figure(figsize=(8, 8))
                pos = np.argwhere(first_cell_id==next_cell_id)
                plt.scatter(to_numpy(x[pos, 1]),to_numpy(x[pos, 2]),s=10,c=mc)
                plt.scatter(to_numpy(x_pos_next[pos, 0]),to_numpy(x_pos_next[pos, 1]),s=10,c=mc,alpha=0.5)
                plt.scatter(to_numpy(x_pos_pred[pos, 0]),to_numpy(x_pos_pred[pos, 1]),s=10,c='g',alpha=0.5)

                good_tracking_distance = torch.sqrt(min_distance_value[pos.astype(int)])

                fig = plt.figure(figsize=(8, 8))
                pos = np.argwhere(first_cell_id!=next_cell_id)
                plt.scatter(to_numpy(x[pos, 1]),to_numpy(x[pos, 2]),s=10,c=mc)
                plt.scatter(to_numpy(x_pos_next[pos, 0]),to_numpy(x_pos_next[pos, 1]),s=10,c=mc,alpha=0.5)
                plt.scatter(to_numpy(x_pos_pred[pos, 0]),to_numpy(x_pos_pred[pos, 1]),s=10,c='r',alpha=1)

                bad_tracking_distance = torch.sqrt(min_distance_value[pos.astype(int)])

            x_ = torch.stack(x_list[1])
            x_ = torch.reshape(x_, (x_.shape[0] * x_.shape[1], x_.shape[2]))
            x_ = x_[0:(n_frames - 1) * n_particles]
            indexes = np.unique(to_numpy(x_[:, 0]))

            plt.xlabel(r'True particle index', fontsize=32)
            plt.ylabel(r'Particle index in next frame', fontsize=32)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/proxy_tracking_{epoch}.tif", dpi=170.7)
            plt.close()
            print(f'tracking index: {np.round(tracking_index,3)}')
            logger.info(f'tracking index: {np.round(tracking_index,3)}')
            print(f'{len(indexes)} tracks')
            logger.info(f'{len(indexes)} tracks')

            tracking_index_list_.append(tracking_index)

            tracking_index_list = np.array(tracking_index_list)
            tracking_index_list = n_particles - tracking_index_list

            fig,ax = fig_init(formatx='%.0f', formaty='%.0f')
            plt.plot(np.arange(n_frames), tracking_index_list, color=mc, linewidth=2)
            plt.ylabel(r'tracking errors', fontsize=68)
            plt.xlabel(r'frame', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/tracking_error_{epoch}.tif", dpi=170.7)
            plt.close()

            print(f'tracking errors: {np.sum(tracking_index_list)}')
            logger.info(f'tracking errors: {np.sum(tracking_index_list)}')

            tracking_errors_list_.append(np.sum(tracking_index_list))

            if embedding_type==1:
                type_list = to_numpy(x_[indexes,5])
            else:
                type_list = to_numpy(type_list_first)

            config.training.cluster_distance_threshold = 0.1

            alpha = 0.1
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster_tracking(model, config,embedding_cluster, cmap, index_particles, indexes, type_list,
                                    n_particle_types, n_particles, ynorm, epoch, log_dir, embedding_type, alpha, device)
            print(
                f'accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(
                f'accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            accuracy_list_.append(accuracy)

            if embedding_type==1:
                fig, ax = fig_init()
                p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                rmserr_list = []
                for n, k in enumerate(indexes):
                    embedding_ = model.a[int(k), :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    true_func = model.psi(rr, p[int(type_list[int(n)])].squeeze(),
                                          p[int(type_list[int(n)])].squeeze())
                    rmserr_list.append(torch.sqrt(torch.mean((func - true_func.squeeze()) ** 2)))
                    plt.plot(to_numpy(rr),
                             to_numpy(func),
                             color=cmap.color(int(type_list[int(n)])), linewidth=2, alpha=0.1)
                if 'latex' in style:
                    plt.xlabel(r'$d_{ij}$', fontsize=68)
                    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
                else:
                    plt.xlabel(r'$d_{ij}$', fontsize=68)
                    plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
                plt.xlim([0, max_radius])
                plt.ylim(config.plotting.ylim)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/learned_function_{epoch}.tif", dpi=170.7)
                rmserr_list = torch.stack(rmserr_list)
                rmserr_list = to_numpy(rmserr_list)
                print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
                logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
                plt.close()
            else:
                fig, ax = fig_init()
                p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                rmserr_list = []
                for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    true_func = model.psi(rr, p[int(type_list[n,0])],p[int(type_list[n,0])])
                    rmserr_list.append(torch.sqrt(torch.mean((func - true_func.squeeze()) ** 2)))
                    plt.plot(to_numpy(rr), to_numpy(func), color=cmap.color(int(type_list[n,0])), linewidth=2, alpha=0.1)
                if 'latex' in style:
                    plt.xlabel(r'$d_{ij}$', fontsize=68)
                    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
                else:
                    plt.xlabel(r'$d_{ij}$', fontsize=68)
                    plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
                plt.xlim([0, max_radius])
                plt.ylim(config.plotting.ylim)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/learned_function_{epoch}.tif", dpi=170.7)
                rmserr_list = torch.stack(rmserr_list)
                rmserr_list = to_numpy(rmserr_list)
                print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
                logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
                plt.close()

            fig, ax = fig_init()
            for n in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
            plt.close()

            fig, ax = fig_init()
            plt.plot(tracking_index_list_, color=mc, linewidth=2)
            plt.ylabel(r'tracking_index', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/tracking_index_list.tif", dpi=170.7)
            fig, ax = fig_init()
            plt.plot(accuracy_list_, color=mc, linewidth=2)
            plt.ylabel(r'accuracy', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/accuracy_list.tif", dpi=170.7)
            fig, ax = fig_init()
            plt.plot(tracking_errors_list_, color=mc, linewidth=2)
            plt.ylabel(r'tracking_errors', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/tracking_errors_list.tif", dpi=170.7)


def plot_attraction_repulsion_asym(config, epoch_list, log_dir, logger, style, device):


    dataset_name = config.dataset

    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    cmap = CustomColorMap(config=config)
    embedding_cluster = EmbeddingCluster(config)
    n_runs = config.training.n_runs

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    for epoch in epoch_list:

        net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        config.training.cluster_method = 'distance_embedding'
        config.training.cluster_distance_threshold = 0.01
        alpha = 0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(
            f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        x = x_list[0][100].clone().detach()
        index_particles = get_index_particles(x, n_particle_types, dimension)
        type_list = to_numpy(get_type_list(x, dimension))
        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
        edges = adj_t.nonzero().t().contiguous()
        indexes = np.random.randint(0, edges.shape[1], 5000)
        edges = edges[:, indexes]

        fig, ax = fig_init()
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list = []
        for n in trange(edges.shape[1]):
            embedding_1 = model.a[1, edges[0, n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                  device=device)
            embedding_2 = model.a[1, edges[1, n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                  device=device)
            type = type_list[to_numpy(edges[0, n])].astype(int)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(type), linewidth=8)
        if 'latex' in style:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
        else:
            plt.xlabel('$d_{ij}$', fontsize=68)
            plt.ylabel('$f(a_i, d_{ij})$', fontsize=68)
        plt.ylim(config.plotting.ylim)
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
        true_func = []
        for n in range(n_particle_types):
            for m in range(n_particle_types):
                true_func.append(model.psi(rr, p[n, m].squeeze(), p[n, m].squeeze()))
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n,m], p[n,m]).squeeze()), color=cmap.color(n), linewidth=8)
        if 'latex' in style:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
        else:
            plt.xlabel('$d_{ij}$', fontsize=68)
            plt.ylabel('$f(a_i, d_{ij})$', fontsize=68)
        plt.ylim(config.plotting.ylim)
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
        plt.close()

        true_func_list = []
        for k in trange(edges.shape[1]):
            n = type_list[to_numpy(edges[0, k])].astype(int)
            m = type_list[to_numpy(edges[1, k])].astype(int)
            true_func_list.append(true_func[3 * n.squeeze() + m.squeeze()])
        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)
        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))


def plot_attraction_repulsion_continuous(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    dataset_name = config.dataset
    max_radius = config.simulation.max_radius
    n_runs = config.training.n_runs
    cmap = CustomColorMap(config=config)

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[1][0].clone().detach()

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    for epoch in epoch_list:

        net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        n_particle_types = 3
        index_particles = []
        for n in range(n_particle_types):
            index_particles.append(
                np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))

        fig, ax = fig_init()
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=400, alpha=0.1)
        if 'latex' in style:
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
        else:
            plt.xlabel(r'$a_{i0}$', fontsize=68)
            plt.ylabel(r'$a_{i1}$', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/first_embedding_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list = []
        for n in range(n_particles):
            embedding = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n // 1600), linewidth=2, alpha=0.1)
        if 'latex' in style:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
        else:
            plt.xlabel('$d_{ij}$', fontsize=68)
            plt.ylabel('$f(a_i, d_{ij})$', fontsize=68)
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        p = torch.load(f'graphs_data/{dataset_name}/model_p.pt')
        true_func_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particles):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n // 1600), linewidth=2,
                     alpha=0.1)
            true_func_list.append(model.psi(rr, p[n], p[n]))
            csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
        if 'latex' in style:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
        else:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel('$f(a_i, d_{ij})$', fontsize=68)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
        np.save(f"./{log_dir}/results/true_func_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/results/true_func_{epoch}.txt", csv_)
        plt.close()

        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)

        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))


def plot_gravity(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_runs = config.training.n_runs
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    model, bc_pos, bc_dpos = choose_training_model(config, device)


    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0


        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id])//1E7 == 2:
                flag = False
            file_id += 1

        file_id_list0 = np.arange(0,60)
        file_id_list1 = np.arange(40,file_id,(file_id-40)//60)
        file_id_list2 = np.arange(file_id, len(files), (len(files)-file_id) // 100)
        file_id_list = np.concatenate((file_id_list0,file_id_list1, file_id_list2))


        for file_id_ in trange(0,len(file_id_list)):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:
                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')

                fig, ax = fig_init(fontsize=24)
                embedding = get_embedding(model.a, 1)
                for n in range(n_particle_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n)
                    pos = to_numpy(pos)
                    if len(pos) > 0:
                        plt.scatter(embedding[pos, 0], embedding[pos, 1], color=cmap.color(n), s=100, alpha=0.1)
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                match config.dataset:
                    case 'gravity_16':
                        plt.ylim([0, 3])
                        plt.xlim([-1, 2])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()


                fig, ax = fig_init(fontsize=24)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                    embedding_ = model.a[1,n] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
                plt.xlabel('$d_{ij}$', fontsize=48)
                plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                plt.xlim([0, max_radius])
                plt.ylim(config.plotting.ylim)
                match config.dataset:
                    case 'gravity_16':
                        plt.xlim([0, 0.02])
                        plt.ylim([0, 0.5E6])
                        plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
                plt.close()

    else:

        for epoch in epoch_list:

            net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
            print(f'network: {net}')
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            fig,ax = fig_init()
            embedding = get_embedding(model.a, 1)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=100, alpha=0.1)

            config.training.cluster_method = 'distance_embedding'
            config.training.cluster_distance_threshold = 0.01
            alpha=0.1
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            config.training.cluster_method = 'distance_plot'
            config.training.cluster_distance_threshold = 0.01
            alpha = 0.5
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
            params = {'mathtext.default': 'regular'}
            plt.rcParams.update(params)
            p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
            rmserr_list = []
            for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                      p[to_numpy(type_list[n]).astype(int)].squeeze())
                rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.xlim([0, 0.02])
            plt.ylim([0, 0.5E6])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_function_{epoch}.tif", dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            plt.close()

            fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
            plots = []
            plots.append(rr)
            for n in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
                plots.append(model.psi(rr, p[n], p[n]).squeeze())
            plt.xlim([0, 0.02])
            plt.ylim([0, 0.5E6])
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
            plt.close()

            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
            plot_list = []
            for n in range(int(n_particles)):
                embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                with torch.no_grad():
                    pred = model.lin_edge(in_features.float())
                pred = pred[:, 0]
                plot_list.append(pred * ynorm)
            p = np.linspace(0.5, 5, n_particle_types)
            p_list = p[to_numpy(type_list).astype(int)]
            popt_list = []
            for n in range(int(n_particles)):
                popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
                popt_list.append(popt)
            popt_list=np.array(popt_list)

            x_data = p_list.squeeze()
            y_data = popt_list[:, 0]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

            if '20' in epoch:

                threshold = 0.4
                relative_error = np.abs(y_data - x_data) / x_data
                pos = np.argwhere(relative_error < threshold)
                pos_outliers = np.argwhere(relative_error > threshold)

                if len(pos)>0:
                    x_data_ = x_data[pos[:, 0]]
                    y_data_ = y_data[pos[:, 0]]
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
                    residuals = y_data_ - linear_model(x_data_, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data_)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')
                    logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')

                    fig, ax = fig_init()
                    csv_ = []
                    csv_.append(p_list)
                    csv_.append(popt_list[:, 0])
                    plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='w', linewidth=4, alpha=0.5)
                    plt.scatter(p_list, popt_list[:, 0], color='w', s=100, alpha=0.5)
                    plt.scatter(p_list[pos_outliers[:, 0]], popt_list[pos_outliers[:, 0], 0], color='r', s=50)
                    plt.xlabel(r'true mass ', fontsize=64)
                    plt.ylabel(r'learned mass ', fontsize=64)
                    plt.xlim([0, 5.5])
                    plt.ylim([0, 5.5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/mass.tif", dpi=170.7)
                    # csv_ = np.array(csv_)
                    # np.save(f"./{log_dir}/results/mass.npy", csv_)
                    # np.savetxt(f"./{log_dir}/results/mass.txt", csv_)
                    plt.close()

                    relative_error = np.abs(popt_list[:, 0] - p_list.squeeze()) / p_list.squeeze() * 100

                    print(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
                    print(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
                    logger.info(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
                    logger.info(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


                    fig, ax = fig_init()
                    csv_ = []
                    csv_.append(p_list.squeeze())
                    csv_.append(-popt_list[:, 1])
                    csv_ = np.array(csv_)
                    plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                    plt.scatter(p_list, -popt_list[:, 1], color='w', s=50, alpha=0.5)
                    plt.xlim([0, 5.5])
                    plt.ylim([-4, 0])
                    plt.xlabel(r'True mass', fontsize=68)
                    plt.ylabel(r'Learned exponent', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/exponent.tif", dpi=170)
                    np.save(f"./{log_dir}/results/exponent.npy", csv_)
                    np.savetxt(f"./{log_dir}/results/exponent.txt", csv_)
                    plt.close()

                    print(f'exponent: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')
                    logger.info(f'mass relative error: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')

                else:
                    print('no fit')
                    logger.info('no fit')

                if os.path.exists(f"./{log_dir}/results/coeff_pysrr.npy"):
                    popt_list = np.load(f"./{log_dir}/results/coeff_pysrr.npy")

                else:
                    print('curve fitting ...')
                    text_trap = StringIO()
                    sys.stdout = text_trap
                    popt_list = []
                    for n in range(0,int(n_particles)):
                        model_pysrr, max_index, max_value = symbolic_regression(rr, plot_list[n])
                        # print(f'{p_list[n].squeeze()}/x0**2, {model_pysrr.sympy(max_index)}')
                        logger.info(f'{np.round(p_list[n].squeeze(),2)}/x0**2, pysrr found {model_pysrr.sympy(max_index)}')

                        expr = model_pysrr.sympy(max_index).as_terms()[0]
                        popt_list.append(expr[0][1][0][0])

                    np.save(f"./{log_dir}/results/coeff_pysrr.npy", popt_list)

                    # model_pysrr = PySRRegressor(
                    #     niterations=30,  # < Increase me for better results
                    #     random_state=0,
                    #     temp_equation_file=False
                    # )
                    # model_pysrr.fit(to_numpy(rr[:, None]), to_numpy(plot_list[0]))

                    sys.stdout = sys.__stdout__

                    popt_list = np.array(popt_list)

                x_data = p_list.squeeze()
                y_data = popt_list
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

                threshold = 0.4
                relative_error = np.abs(y_data - x_data) / x_data
                pos = np.argwhere(relative_error < threshold)
                x_data_ = x_data[pos[:, 0]]
                y_data_ = y_data[pos[:, 0]]
                lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)

                residuals = y_data_ - linear_model(x_data_, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')
                logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')

                fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
                plt.scatter(x_data_,y_data_, color='w', s=1, alpha=0.5)
                plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                plt.scatter(p_list, popt_list, color='w', s=50, alpha=1)
                plt.xlabel(r'True mass ', fontsize=64)
                plt.ylabel(r'Learned mass ', fontsize=64)
                plt.xlim([0, 5.5])
                plt.ylim([0, 5.5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/pysrr_mass.tif", dpi=300)
                plt.close()

                relative_error = np.abs(popt_list - p_list.squeeze()) / p_list.squeeze() * 100

                print(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
                print(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
                logger.info(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
                logger.info(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


def plot_gravity_continuous(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs
    dimension= config.simulation.dimension
    cmap = CustomColorMap(config=config)

    embedding_cluster = EmbeddingCluster(config)

    time.sleep(0.5)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    for epoch in epoch_list:

        net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        embedding = get_embedding(model.a, 1)
        fig, ax = fig_init()
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n % 256),
                        s=400,
                        alpha=0.5)
        if 'latex' in style:
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
        else:
            plt.xlabel(r'$a_{i0}$', fontsize=68)
            plt.ylabel(r'$a_{i1}$', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/embedding.tif", dpi=170)
        plt.close()

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
        rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        rmserr_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
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
                     color=cmap.color(n % 256), linewidth=8, alpha=0.1)
        if 'latex' in style:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
        else:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
        plt.xlim([0, max_radius])
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func.tif", dpi=170)
        csv_ = np.array(csv_)
        np.save(f"./{log_dir}/results/func.npy", csv_)
        np.savetxt(f"./{log_dir}/results/func.txt", csv_)
        plt.close()

        rmserr_list = torch.stack(rmserr_list)
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        p = np.linspace(0.5, 5, n_particle_types)
        p = torch.tensor(p, device=device)
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particle_types - 1, -1, -1):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n % 256), linewidth=8)
            csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        if 'latex' in style:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
        else:
            plt.xlabel(r'$d_{ij}$', fontsize=68)
            plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=300)
        csv_ = np.array(csv_)
        np.save(f"./{log_dir}/results/true_func.npy", csv_)
        np.savetxt(f"./{log_dir}/results/true_func.txt", csv_)
        plt.close()

        rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        plot_list = []
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                pred = model.lin_edge(in_features.float())
            pred = pred[:, 0]
            plot_list.append(pred * ynorm)
        p = np.linspace(0.5, 5, n_particle_types)
        p_list = p[to_numpy(type_list).astype(int)]
        popt_list = []
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
            popt_list.append(popt)
        popt_list=np.array(popt_list)

        x_data = p_list.squeeze()
        y_data = popt_list[:, 0]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

        threshold = 0.4
        relative_error = np.abs(y_data - x_data) / x_data
        pos = np.argwhere(relative_error < threshold)
        pos_outliers = np.argwhere(relative_error > threshold)
        x_data_ = x_data[pos[:, 0]]
        y_data_ = y_data[pos[:, 0]]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
        residuals = y_data_ - linear_model(x_data_, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')
        logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')

        fig, ax = fig_init()
        plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(p_list, popt_list[:, 0], color=mc, s=50, alpha=0.5)
        plt.scatter(p_list[pos_outliers[:, 0]], popt_list[pos_outliers[:, 0], 0], color='r', s=50)
        plt.xlabel(r'True mass ', fontsize=64)
        plt.ylabel(r'Learned mass ', fontsize=64)
        plt.xlim([0, 5.5])
        plt.ylim([0, 5.5])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/mass.tif", dpi=300)
        plt.close()

        relative_error = np.abs(popt_list[:, 0] - p_list.squeeze()) / p_list.squeeze() * 100

        print(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        print(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
        logger.info(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        logger.info(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


        fig, ax = fig_init()
        csv_ = []
        csv_.append(p_list.squeeze())
        csv_.append(-popt_list[:, 1])
        csv_ = np.array(csv_)
        plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(p_list, -popt_list[:, 1], color=mc, s=50, alpha=0.5)
        plt.xlim([0, 5.5])
        plt.ylim([-4, 0])
        plt.xlabel(r'True mass', fontsize=68)
        plt.ylabel(r'Learned exponent', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/exponent.tif", dpi=300)
        np.save(f"./{log_dir}/results/exponent.npy", csv_)
        np.savetxt(f"./{log_dir}/results/exponent.txt", csv_)
        plt.close()

        print(f'exponent: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')
        logger.info(f'mass relative error: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')

        if os.path.exists(f"./{log_dir}/results/coeff_pysrr.npy"):
            popt_list = np.load(f"./{log_dir}/results/coeff_pysrr.npy")

        else:
            text_trap = StringIO()
            sys.stdout = text_trap
            popt_list = []
            for n in range(0,int(n_particles * (1 - config.training.particle_dropout))):
                print(n)
                model_pysrr, max_index, max_value = symbolic_regression(rr, plot_list[n])
                # print(f'{p_list[n].squeeze()}/x0**2, {model_pysrr.sympy(max_index)}')
                logger.info(f'{np.round(p_list[n].squeeze(),2)}/x0**2, pysrr found {model_pysrr.sympy(max_index)}')

                expr = model_pysrr.sympy(max_index).as_terms()[0]
                popt_list.append(expr[0][1][0][0])

            np.save(f"./{log_dir}/results/coeff_pysrr.npy", popt_list)

            sys.stdout = sys.__stdout__

            popt_list = np.array(popt_list)

        x_data = p_list.squeeze()
        y_data = popt_list
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

        threshold = 0.4
        relative_error = np.abs(y_data - x_data) / x_data
        pos = np.argwhere(relative_error < threshold)
        pos_outliers = np.argwhere(relative_error > threshold)
        x_data_ = x_data[pos[:, 0]]
        y_data_ = y_data[pos[:, 0]]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
        residuals = y_data_ - linear_model(x_data_, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data_)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')
        logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        csv_ = []
        csv_.append(p_list)
        csv_.append(popt_list)
        plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(p_list, popt_list, color=mc, s=50, alpha=0.5)
        plt.xlabel(r'True mass ', fontsize=64)
        plt.ylabel(r'Learned mass ', fontsize=64)
        plt.xlim([0, 5.5])
        plt.ylim([0, 5.5])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/pysrr_mass.tif", dpi=300)
        # csv_ = np.array(csv_)
        # np.save(f"./{log_dir}/results/mass.npy", csv_)
        # np.savetxt(f"./{log_dir}/results/mass.txt", csv_)
        plt.close()

        relative_error = np.abs(popt_list - p_list.squeeze()) / p_list.squeeze() * 100

        print(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        print(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
        logger.info(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        logger.info(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


def plot_Coulomb(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        p = [2, 1, -1]
        x = x_list[0][100].clone().detach()
        index_particles = get_index_particles(x, n_particle_types, dimension)
        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
        edges = adj_t.nonzero().t().contiguous()
        indexes = np.random.randint(0, edges.shape[1], 5000)
        edges = edges[:, indexes]

        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id])//1E7 == 2:
                flag = False
            file_id += 1

        file_id_list0 = np.arange(0,60)
        file_id_list1 = np.arange(40,file_id,(file_id-40)//60)
        file_id_list2 = np.arange(file_id, len(files), (len(files)-file_id) // 100)
        file_id_list = np.concatenate((file_id_list0,file_id_list1, file_id_list2))

        for file_id_ in trange(0,len(file_id_list)-1):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:
                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')
                amax = torch.max(model.a[1], dim=0).values
                amin = torch.min(model.a[1], dim=0).values
                model_a = to_numpy((model.a[1] - amin) / (amax - amin))

                fig, ax = fig_init(fontsize=24)
                for n in range(n_particle_types - 1, -1, -1):
                    pos = torch.argwhere(type_list.squeeze() == n)
                    pos = to_numpy(pos)
                    plt.scatter(model_a[pos, 0], model_a[pos, 1], s=100, color=cmap.color(n), alpha=0.5)
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()

                fig, ax = fig_init(fontsize=24)
                func_list = []
                rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
                table_qiqj = np.zeros((10, 1))
                tmp = np.array([-2, -1, 1, 2, 4])
                table_qiqj[tmp.astype(int) + 2] = np.arange(5)[:, None]
                qiqj_list = []
                for n in range(edges.shape[1]):
                    embedding_1 = model.a[1, edges[0, n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                          device=device)
                    embedding_2 = model.a[1, edges[1, n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                          device=device)
                    qiqj = p[to_numpy(type_list[to_numpy(edges[0, n])]).astype(int).squeeze()] * p[
                        to_numpy(type_list[to_numpy(edges[1, n])]).astype(int).squeeze()]
                    qiqj_list.append(qiqj)
                    type = table_qiqj[qiqj + 2].astype(int).squeeze()
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    func_list.append(func * ynorm)
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(type), linewidth=8, alpha=0.1)
                plt.xlabel('$d_{ij}$', fontsize=48)
                plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                plt.xlim([0, 0.02])
                plt.ylim([-0.5E6, 0.5E6])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
                plt.close()

    else:
        for epoch in epoch_list:

            net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()


            config.training.cluster_method = 'distance_plot'
            config.training.cluster_distance_threshold = 0.1
            alpha=0.5
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(
                f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            config.training.cluster_method = 'distance_embedding'
            config.training.cluster_distance_threshold = 0.01
            alpha = 0.5
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            x = x_list[0][100].clone().detach()
            index_particles = get_index_particles(x, n_particle_types, dimension)
            type_list = to_numpy(get_type_list(x, dimension))
            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edges = adj_t.nonzero().t().contiguous()
            indexes = np.random.randint(0, edges.shape[1], 5000)
            edges = edges[:, indexes]

            p = [2, 1, -1]

            fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
            func_list = []
            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
            table_qiqj = np.zeros((10,1))
            tmp = np.array([-2, -1, 1, 2, 4])
            table_qiqj[tmp.astype(int)+2]=np.arange(5)[:,None]
            qiqj_list=[]
            for n in trange(edges.shape[1]):
                embedding_1 = model.a[1, edges[0, n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                embedding_2 = model.a[1, edges[1, n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                qiqj = p[type_list[to_numpy(edges[0, n])].astype(int).squeeze()] * p[type_list[to_numpy(edges[1, n])].astype(int).squeeze()]
                qiqj_list.append(qiqj)
                type = table_qiqj[qiqj+2].astype(int).squeeze()
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                func_list.append(func * ynorm)
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(type), linewidth=8, alpha=0.1)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, a_j, d_{ij})$', fontsize=68)
            plt.xlim([0, 0.02])
            plt.ylim([-0.5E6, 0.5E6])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_func_{epoch}.tif", dpi=170.7)
            plt.close()

            fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
            csv_ = []
            csv_.append(to_numpy(rr))
            true_func_list = []
            for n in trange(edges.shape[1]):
                temp = model.psi(rr, p[type_list[to_numpy(edges[0, n])].astype(int).squeeze()], p[type_list[to_numpy(edges[1, n])].astype(int).squeeze()] )
                true_func_list.append(temp)
                type = p[type_list[to_numpy(edges[0, n])].astype(int).squeeze()] * p[type_list[to_numpy(edges[1, n])].astype(int).squeeze()]
                type = table_qiqj[type+2].astype(int).squeeze()
                plt.plot(to_numpy(rr), np.array(temp.cpu()), linewidth=8, color=cmap.color(type))
                csv_.append(to_numpy(temp.squeeze()))
            plt.xlim([0, 0.02])
            plt.ylim([-0.5E6, 0.5E6])
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, a_j, d_{ij})$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_func_{epoch}.tif", dpi=170.7)
            np.save(f"./{log_dir}/results/true_func_{epoch}.npy", csv_)
            np.savetxt(f"./{log_dir}/results/true_func_{epoch}.txt", csv_)
            plt.close()

            func_list = torch.stack(func_list)
            true_func_list = torch.stack(true_func_list)
            rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))

            if os.path.exists(f"./{log_dir}/results/coeff_pysrr.npy"):
                popt_list = np.load(f"./{log_dir}/results/coeff_pysrr.npy")

            else:
                print('curve fitting ...')
                text_trap = StringIO()
                sys.stdout = text_trap
                popt_list = []
                qiqj_list = np.array(qiqj_list)
                for n in range(0,edges.shape[1],5):
                    model_pysrr, max_index, max_value = symbolic_regression(rr, func_list[n])
                    print(f'{-qiqj_list[n]}/x0**2, {model_pysrr.sympy(max_index)}')
                    logger.info(f'{-qiqj_list[n]}/x0**2, pysrr found {model_pysrr.sympy(max_index)}')

                    expr = model_pysrr.sympy(max_index).as_terms()[0]
                    popt_list.append(-expr[0][1][0][0])

                np.save(f"./{log_dir}/results/coeff_pysrr.npy", popt_list)
                np.save(f"./{log_dir}/results/qiqj.npy", qiqj_list)

            qiqj_list = np.load(f"./{log_dir}/results/qiqj.npy")
            qiqj = []
            for n in range(0, len(qiqj_list), 5):
                qiqj.append(qiqj_list[n])
            qiqj_list = np.array(qiqj)

            threshold = 1

            fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
            x_data = qiqj_list.squeeze()
            y_data = popt_list.squeeze()
            lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.scatter(qiqj_list, popt_list, color='w', s=200, alpha=0.1)
            plt.xlim([-2.5, 5])
            plt.ylim([-2.5, 5])
            plt.ylabel(r'Learned $q_i q_j$', fontsize=64)
            plt.xlabel(r'True $q_i q_j$', fontsize=64)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/qiqj_{epoch}.tif", dpi=170)
            plt.close()

            print(f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold: {threshold} ')
            logger.info(f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold: {threshold} ')

            print(f'pysrr_qiqj relative error: {100*np.round(np.mean(relative_error), 2)}+/-{100*np.round(np.std(relative_error), 2)}')
            print(f'pysrr_qiqj relative error wo outliers: {100*np.round(np.mean(relative_error[not_outliers[:, 0]]), 2)}+/-{100*np.round(np.std(relative_error[not_outliers[:, 0]]), 2)}')
            logger.info(f'pysrr_qiqj relative error: {100*np.round(np.mean(relative_error), 2)}+/-{100*np.round(np.std(relative_error), 2)}')
            logger.info(f'pysrr_qiqj relative error wo outliers: {100*np.round(np.mean(relative_error[not_outliers[:, 0]]), 2)}+/-{100*np.round(np.std(relative_error[not_outliers[:, 0]]), 2)}')

            # qi qj retrieval
            qiqj = torch.tensor(popt_list, device=device)[:, None]
            qiqj = qiqj[not_outliers[:, 0]]

            model_qs = model_qiqj(3, device)
            optimizer = torch.optim.Adam(model_qs.parameters(), lr=1E-2)
            qiqj_list = []
            loss_list = []
            for it in trange(20000):

                sample = np.random.randint(0, qiqj.shape[0] - 10)
                qiqj_ = qiqj[sample:sample + 10]

                optimizer.zero_grad()
                qs = model_qs()
                distance = torch.sum((qiqj_[:, None] - qs[None, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_value = result.values
                min_index = result.indices
                loss = torch.mean(min_value) + torch.max(min_value)
                loss.backward()
                optimizer.step()
                if it % 100 == 0:
                    qiqj_list.append(to_numpy(model_qs.qiqj))
                    loss_list.append(to_numpy(loss))
            qiqj_list = np.array(qiqj_list).squeeze()

            print('qi')
            print(np.round(to_numpy(model_qs.qiqj[2].squeeze()),3), np.round(to_numpy(model_qs.qiqj[1].squeeze()),3),np.round(to_numpy(model_qs.qiqj[0].squeeze()),3) )
            logger.info('qi')
            logger.info(f'{np.round(to_numpy(model_qs.qiqj[2].squeeze()),3)}, {np.round(to_numpy(model_qs.qiqj[1].squeeze()),3)}, {np.round(to_numpy(model_qs.qiqj[0].squeeze()),3)}' )

            fig, ax = fig_init()
            plt.plot(qiqj_list[:, 0], linewidth=4)
            plt.plot(qiqj_list[:, 1], linewidth=4)
            plt.plot(qiqj_list[:, 2], linewidth=4)
            plt.xlabel('iteration',fontsize=68)
            plt.ylabel(r'$q_i$',fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/qi_{epoch}.tif", dpi=170)


def plot_boids(config, epoch_list, log_dir, logger, style, device):

    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_runs = config.training.n_runs
    has_cell_division = config.simulation.has_cell_division
    cmap = CustomColorMap(config=config)
    n_frames = config.simulation.n_frames
    dimension = config.simulation.dimension

    embedding_cluster = EmbeddingCluster(config)

    print('load data ...')
    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, 1, log_dir, device)
    x = x_list[0][-1].clone().detach()

    print('done ...')

    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]
    if has_cell_division:
        T1_list = []
        T1_list.append(torch.load(f'graphs_data/{dataset_name}/T1_list_1.pt', map_location=device))
        n_particles_max = np.load(os.path.join(log_dir, 'n_particles_max.npy'))
        config.simulation.n_particles_max = n_particles_max
        type_list = T1_list[0]
        n_particles = len(type_list)

    if epoch_list[0] == 'all':

        model, bc_pos, bc_dpos = choose_training_model(config, device)
        model.ynorm = ynorm
        model.vnorm = vnorm

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id]) // 1E7 == 2:
                flag = False
            file_id += 1

        file_id_list0 = np.arange(0,60)
        file_id_list1 = np.arange(60,file_id,(file_id-60)//60)
        file_id_list2 = np.arange(file_id, len(files), (len(files)-file_id) // 100)
        file_id_list = np.concatenate((file_id_list0,file_id_list1, file_id_list2))

        for file_id_ in trange(0, len(file_id_list)):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:
                epoch = files[file_id].split('graphs')[1][1:-3]
                print(epoch)
                net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')

                fig, ax = fig_init(fontsize=24)
                embedding = get_embedding(model.a, 1)
                # embedding = (embedding-np.min(embedding))/(np.max(embedding)-np.min(embedding))
                for n in range(n_particle_types - 1, -1, -1):
                    pos = torch.argwhere(type_list == n)
                    pos = to_numpy(pos)
                    if len(pos) > 0:
                        plt.scatter(embedding[pos, 0], embedding[pos, 1], c=cmap.color(n), s=100, alpha=0.1)
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                match config.dataset:
                    case 'boids_16_256':
                        plt.xlim([-1.5, 2])
                        plt.ylim([-1.5, 3])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()

                fig, ax = fig_init(fontsize=24)
                rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
                x = x_list[0][-1].clone().detach()
                for n in np.arange(len(x)):
                    embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    type = to_numpy(x[n, 5]).astype(int)
                    if type < n_particle_types:
                        if (n % 10 == 0):
                            plt.plot(to_numpy(rr),
                                     to_numpy(func) * to_numpy(ynorm) * 1E4,
                                     color=cmap.color(type), linewidth=4, alpha=0.25)
                plt.ylim([-1, 1])
                plt.xlabel('$d_{ij}$', fontsize=48)
                plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
                plt.close()

    else:
        for epoch in epoch_list:

            model, bc_pos, bc_dpos = choose_training_model(config, device)
            model = Interaction_Particle_extract(config, device, aggr_type=config.graph_model.aggr_type, bc_dpos=bc_dpos)
            model.ynorm = ynorm
            model.vnorm = vnorm

            net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            alpha = 0.5
            print('clustering ...')

            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(
                f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(
                f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

            if has_cell_division:
                plot_cell_rates(config, device, log_dir, n_particle_types, type_list, x_list, new_labels, cmap, logger)

            print('compare reconstructed interaction with ground truth...')

            p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
            model_B = PDE_B_extract(aggr_type=config.graph_model.aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)

            fig, ax = fig_init()
            rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
            func_list = []
            true_func_list = []
            x = x_list[0][-1].clone().detach()
            for n in np.arange(len(x)):
                embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                type = to_numpy(x[n, 5]).astype(int)
                if type < n_particle_types:
                    func_list.append(func)
                    true_func = model_B.psi(rr, p[type])
                    true_func_list.append(true_func)
                    if (n % 10 == 0):
                        plt.plot(to_numpy(rr),
                                 to_numpy(func) * to_numpy(ynorm),
                                 color=cmap.color(type), linewidth=4, alpha=0.25)
            func_list = torch.stack(func_list)
            true_func_list = torch.stack(true_func_list)
            plt.ylim([-1E-4, 1E-4])
            plt.xlabel(r'$x_j-x_i$', fontsize=68)
            plt.ylabel(r'$f_{ij}$', fontsize=68)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_func_{epoch}.tif", dpi=300)
            plt.close()

            fig, ax = fig_init()
            for n in range(n_particle_types):
                true_func = model_B.psi(rr, p[n])
                plt.plot(to_numpy(rr), to_numpy(true_func), color=cmap.color(n), linewidth=4)
            plt.ylim([-1E-4, 1E-4])
            plt.xlabel(r'$x_j-x_i$', fontsize=68)
            plt.ylabel(r'$f_{ij}$', fontsize=68)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_func_{epoch}.tif", dpi=300)
            func_list = func_list * ynorm
            func_list_ = torch.clamp(func_list, min=torch.tensor(-1.0E-4, device=device),
                                     max=torch.tensor(1.0E-4, device=device))
            true_func_list_ = torch.clamp(true_func_list, min=torch.tensor(-1.0E-4, device=device),
                                          max=torch.tensor(1.0E-4, device=device))
            rmserr_list = torch.sqrt(torch.mean((func_list_ - true_func_list_) ** 2, 1))
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))

            if '20' in epoch:

                lin_edge_out_list = []
                type_list = []
                diffx_list = []
                diffv_list = []
                cohesion_list=[]
                alignment_list=[]
                separation_list=[]
                r_list = []
                for it in range(0,n_frames//2,n_frames//80):
                    x = x_list[0][it].clone().detach()
                    particle_index = to_numpy(x[:, 0:1]).astype(int)
                    x[:, 5:6] = torch.tensor(new_labels[particle_index],
                                             device=device)  # set label found by clustering and mapperd to ground truth
                    pos = torch.argwhere(x[:, 5:6] < n_particle_types).squeeze()
                    pos = to_numpy(pos[:, 0]).astype(int)  # filter out cluster not associated with ground truth
                    x = x[pos, :]
                    distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)  # threshold
                    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)) * 1.0
                    edge_index = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x, edge_index=edge_index)
                    with torch.no_grad():
                        y, in_features, lin_edge_out = model(dataset, data_id=1, training=False, phi=torch.zeros(1, device=device))  # acceleration estimation
                    y = y * ynorm
                    lin_edge_out = lin_edge_out * ynorm

                    # compute ground truth output
                    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                    psi_output = []
                    for n in range(n_particle_types):
                        with torch.no_grad():
                            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
                            y_B, sum, cohesion, alignment, separation, diffx, diffv, r, type = model_B(dataset)  # acceleration estimation

                    if it==0:
                        lin_edge_out_list=lin_edge_out
                        diffx_list=diffx
                        diffv_list=diffv
                        r_list=r
                        type_list=type
                        cohesion_list = cohesion
                        alignment_list = alignment
                        separation_list = separation
                    else:
                        lin_edge_out_list=torch.cat((lin_edge_out_list,lin_edge_out),dim=0)
                        diffx_list=torch.cat((diffx_list,diffx),dim=0)
                        diffv_list=torch.cat((diffv_list,diffv),dim=0)
                        r_list=torch.cat((r_list,r),dim=0)
                        type_list=torch.cat((type_list,type),dim=0)
                        cohesion_list=torch.cat((cohesion_list,cohesion),dim=0)
                        alignment_list=torch.cat((alignment_list,alignment),dim=0)
                        separation_list=torch.cat((separation_list,separation),dim=0)

                type_list = to_numpy(type_list)

                print(f'fitting with known functions {len(type_list)} points ...')
                cohesion_fit = np.zeros(n_particle_types)
                alignment_fit = np.zeros(n_particle_types)
                separation_fit = np.zeros(n_particle_types)
                indexes = np.unique(type_list)
                indexes = indexes.astype(int)

                if False:
                    for n in indexes:
                        pos = np.argwhere(type_list == n)
                        pos = pos[:, 0].astype(int)
                        xdiff = diffx_list[pos, 0:1]
                        vdiff = diffv_list[pos, 0:1]
                        rdiff = r_list[pos]
                        x_data = torch.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                        y_data = lin_edge_out_list[pos, 0:1]
                        xdiff = diffx_list[pos, 1:2]
                        vdiff = diffv_list[pos, 1:2]
                        rdiff = r_list[pos]
                        tmp = torch.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                        x_data = torch.cat((x_data, tmp), dim=0)
                        tmp = lin_edge_out_list[pos, 1:2]
                        y_data = torch.cat((y_data, tmp), dim=0)
                        model_pysrr, max_index, max_value = symbolic_regression_multi(x_data, y_data)

                for loop in range(2):
                    for n in indexes:
                        pos = np.argwhere(type_list == n)
                        pos = pos[:, 0].astype(int)
                        xdiff = to_numpy(diffx_list[pos, :])
                        vdiff = to_numpy(diffv_list[pos, :])
                        rdiff = to_numpy(r_list[pos])
                        x_data = np.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                        y_data = to_numpy(torch.norm(lin_edge_out_list[pos, :], dim=1))
                        if loop == 0:
                            lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox')
                        else:
                            lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox', p0=p00)
                        cohesion_fit[int(n)] = lin_fit[0]
                        alignment_fit[int(n)] = lin_fit[1]
                        separation_fit[int(n)] = lin_fit[2]
                    p00 = [np.mean(cohesion_fit[indexes]), np.mean(alignment_fit[indexes]), np.mean(separation_fit[indexes])]

                threshold = 0.25

                x_data = np.abs(to_numpy(p[:, 0]) * 0.5E-5)
                y_data = np.abs(cohesion_fit)
                x_data = x_data[indexes]
                y_data = y_data[indexes]
                lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)

                fig, ax = fig_init()
                fmt = lambda x, pos: '{:.1f}e-4'.format((x) * 1e4, pos)
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                for id, n in enumerate(indexes):
                    plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
                plt.xlabel(r'True cohesion coeff. ', fontsize=56)
                plt.ylabel(r'Fitted cohesion coeff. ', fontsize=56)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/cohesion_{epoch}.tif", dpi=300)
                plt.close()
                print()
                print(f'cohesion slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold {threshold} ')
                logger.info(f'cohesion slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold {threshold} ')

                x_data = np.abs(to_numpy(p[:, 1]) * 5E-4)
                y_data = alignment_fit
                x_data = x_data[indexes]
                y_data = y_data[indexes]
                lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)

                fig, ax = fig_init()
                fmt = lambda x, pos: '{:.1f}e-2'.format((x) * 1e2, pos)
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                for id, n in enumerate(indexes):
                    plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
                plt.xlabel(r'True alignement coeff. ', fontsize=56)
                plt.ylabel(r'Fitted alignement coeff. ', fontsize=56)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/alignment_{epoch}.tif", dpi=300)
                plt.close()
                print(f'alignment   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')
                logger.info(f'alignment   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')

                x_data = np.abs(to_numpy(p[:, 2]) * 1E-8)
                y_data = separation_fit
                x_data = x_data[indexes]
                y_data = y_data[indexes]
                lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)

                fig, ax = fig_init()
                fmt = lambda x, pos: '{:.1f}e-7'.format((x) * 1e7, pos)
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                for id, n in enumerate(indexes):
                    plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
                plt.xlabel(r'True separation coeff. ', fontsize=56)
                plt.ylabel(r'Fitted separation coeff. ', fontsize=56)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/separation_{epoch}.tif", dpi=300)
                plt.close()
                print(f'separation   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')
                logger.info(f'separation   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')


def plot_wave(config, epoch_list, log_dir, logger, cc, style, device):

    dataset_name = config.dataset

    n_nodes = config.simulation.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs

    hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(n_runs):
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()

    print(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']

    x_mesh = x_mesh_list[0][n_frames - 1].clone().detach()
    n_nodes = x_mesh.shape[0]
    print(f'N nodes: {n_nodes}')
    x_mesh = x_mesh_list[1][0].clone().detach()

    i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
    coeff = i0[(to_numpy(x_mesh[:, 2]) * 255).astype(int), (to_numpy(x_mesh[:, 1]) * 255).astype(int)] / 255
    coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis))
    vm = np.max(coeff)
    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, cmap='grey', vmin=0, vmax=vm)
    plt.xlabel(r'$x$', fontsize=68)
    plt.ylabel(r'$y$', fontsize=68)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_wave_coeff.tif", dpi=300)
    plt.close
    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, cmap='grey', vmin=0, vmax=vm)
    plt.xlabel(r'$x$', fontsize=68)
    plt.ylabel(r'$y$', fontsize=68)
    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_wave_coeff_cbar.tif", dpi=300)
    plt.close

    for epoch in epoch_list:

        net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')

        mesh_model_gene = choose_mesh_model(config=config, X1_mesh=x_mesh[:,1:3], device=device)

        mesh_model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        mesh_model.load_state_dict(state_dict['model_state_dict'])
        mesh_model.eval()

        x_mesh = x_mesh_list[1][7000].clone().detach()
        mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_1.pt', map_location=device)
        dataset_mesh = data.Data(x=x_mesh, edge_index=mesh_data['edge_index'],
                                 edge_attr=mesh_data['edge_weight'], device=device)
        with torch.no_grad():
            pred_gene = mesh_model_gene(dataset_mesh)
            pred = mesh_model(dataset_mesh, data_id=1)

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.scatter(to_numpy(pred_gene),to_numpy(pred)*to_numpy(hnorm),s=1,c=mc,alpha=0.1)
        plt.close()
        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.scatter(to_numpy(mesh_model_gene.laplacian_u),to_numpy(mesh_model.laplacian_u),s=1,alpha=0.1)
        plt.close()

        fig, ax = fig_init(formatx='%.0f', formaty='%.1f')
        rr = torch.tensor(np.linspace(-1800, 1800, 200)).to(device)
        coeff = np.reshape(to_numpy(mesh_model_gene.coeff)/vm/255, (n_nodes_per_axis * n_nodes_per_axis))
        coeff = np.clip(coeff, a_min=0, a_max=1)
        popt_list = []
        func_list = []
        for n in trange(n_nodes):
            embedding_ = mesh_model.a[1, n, :] * torch.ones((200, 2), device=device)
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                h = mesh_model.lin_phi(in_features.float()) * hnorm
            h = h[:, 0]
            popt, pcov = curve_fit(linear_model, to_numpy(rr.squeeze()), to_numpy(h.squeeze()))
            popt_list.append(popt)
            func_list.append(h)
            # plt.scatter(to_numpy(rr), to_numpy(h), c=f'{coeff[n]}', edgecolors='none',alpha=0.1)
            plt.scatter(to_numpy(rr), to_numpy(h), c=mc,alpha=0.1)
        if 'latex' in style:
            plt.xlabel(r'$\nabla^2 u_i$', fontsize=68)
            plt.ylabel(r'$\Phi(\ensuremath{\mathbf{a}}_{i},\nabla^2 u_i)$', fontsize=68)
        else:
            plt.xlabel(r'$\nabla^2 u_i$', fontsize=68)
            plt.ylabel(r'$\Phi(a_{i},\nabla^2 u_i)$', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/functions_{epoch}.tif", dpi=300)
        plt.close()

        func_list = torch.stack(func_list)
        popt_list = np.array(popt_list)

        threshold=-1
        x_data = np.reshape(to_numpy(mesh_model_gene.coeff)/100, (n_nodes_per_axis*n_nodes_per_axis))
        y_data = popt_list[:, 0]
        # discard borders
        pos = np.argwhere(to_numpy(mask_mesh) == 1)
        x_data = x_data[pos[:, 0]]
        y_data = y_data[pos[:, 0]]

        lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)
        print(
            f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  N points {len(x_data)} ')
        logger.info(
            f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}   N points {len(x_data)} ')

        fig, ax = fig_init(formatx='%.5f', formaty='%.5f')
        plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(x_data, y_data, s=200, c=mc, alpha=0.1)
        plt.xlabel('True wave coeff.', fontsize=68)
        plt.ylabel('Learned wave coeff.', fontsize=68)
        fmt = lambda x, pos: '{:.1f}e-3'.format((x) * 1e3, pos)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/scatter_coeff_{epoch}.tif", dpi=300)
        plt.close()

        t = np.array(popt_list)
        t = t[:, 0]
        t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
        t = np.flipud(t)
        fig, ax = fig_init()
        fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.imshow(t, cmap='grey')
        plt.xlabel(r'$x$', fontsize=68)
        plt.ylabel(r'$y$', fontsize=68)
        fmt = lambda x, pos: '{:.3%}'.format(x)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/wave_coeff_{epoch}.tif", dpi=300)
        plt.close()

        embedding = get_embedding(mesh_model.a, 1)
        fig, ax = fig_init()
        # plt.scatter(embedding[pos[:,0], 0], embedding[pos[:,0], 1], c=x_data, s=100, alpha=1, cmap='grey')
        plt.scatter(embedding[pos[:,0], 0], embedding[pos[:,0], 1], c=mc, s=100, alpha=0.1)
        if 'latex' in style:
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
        else:
            plt.xlabel(r'$a_{i0}$', fontsize=68)
            plt.ylabel(r'$a_{i1}$', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/embedding_{epoch}.tif", dpi=300)
        plt.close()


def plot_particle_field(config, epoch_list, log_dir, logger, cc, style, device):

    dataset_name = config.dataset

    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_nodes = config.simulation.n_nodes
    n_node_types = config.simulation.n_node_types
    node_value_map = config.simulation.node_value_map
    has_video = 'video' in node_value_map
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_frames = config.simulation.n_frames
    has_siren = 'siren' in config.graph_model.field_type
    has_siren_time = 'siren_with_time' in config.graph_model.field_type
    target_batch_size = config.training.batch_size
    has_ghost = config.training.n_ghosts > 0
    if config.training.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given config.graph_model
    embedding_cluster = EmbeddingCluster(config)
    n_runs = config.training.n_runs

    x_list = []
    y_list = []
    x_list.append(torch.load(f'graphs_data/{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/{dataset_name}/y_list_1.pt', map_location=device))
    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_0.pt', map_location=device)
    x_mesh_list.append(x_mesh)
    y_mesh = torch.load(f'graphs_data/{dataset_name}/y_mesh_list_0.pt', map_location=device)
    y_mesh_list.append(y_mesh)
    hnorm = torch.load(f'./log/try_{dataset_name}/hnorm.pt', map_location=device).to(device)

    mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_0.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)

    # # matplotlib.use("Qt5Agg")
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    x_mesh = x_mesh_list[0][0].clone().detach()
    i0 = imread(f'graphs_data/{node_value_map}')
    if has_video:
        i0 = i0[0]
        target = i0[(to_numpy(x_mesh[:, 2]) * 100).astype(int), (to_numpy(x_mesh[:, 1]) * 100).astype(int)]
        target = np.reshape(target, (n_nodes_per_axis, n_nodes_per_axis))
    else:
        target = i0[(to_numpy(x_mesh[:, 2]) * 255).astype(int), (to_numpy(x_mesh[:, 1]) * 255).astype(int)] * 5000/255
        target = np.reshape(target, (n_nodes_per_axis, n_nodes_per_axis))
        # target = np.flipud(target)
    vm = np.max(target)
    if vm == 0:
        vm = 0.01

    fig, ax = fig_init()
    plt.imshow(target, cmap=cc, vmin=0, vmax=vm)
    fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    fmty = lambda x, pos: '{:.1f}'.format((100 - x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
    plt.xlabel(r'$x$', fontsize=68)
    plt.ylabel(r'$y$', fontsize=68)
    # cbar = plt.colorbar(shrink=0.5)
    # cbar.ax.tick_params(labelsize=32)
    # cbar.set_label(r'$Coupling$',fontsize=68)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/target_field.tif", dpi=300)
    plt.close()

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    if has_ghost:
        ghosts_particles = Ghost_Particles(config, n_particles, device)
        if config.training.ghost_method == 'MLP':
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


    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_1_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id])//1E7 == 2:
                flag = False
            file_id += 1

        file_id_list0 = np.arange(0,50)
        file_id_list1 = np.arange(50,file_id,(file_id-50)//50)
        file_id_list2 = np.arange(file_id, len(files), (len(files)-file_id) // 50)
        file_id_list = np.concatenate((file_id_list0,file_id_list1, file_id_list2))

        frame=64
        x = x_list[0][frame].clone().detach()

        for file_id_ in trange(0,len(file_id_list)):
            file_id = file_id_list[file_id_]
            if sort_key(files[file_id]) % 1E7 != 0:
                epoch = files[file_id].split('graphs')[1][1:-3]
                print(epoch)

                net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                if has_siren:
                    net = f'{log_dir}/models/best_model_f_with_1_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                plt.style.use('dark_background')

                fig, ax = fig_init(fontsize=24)
                params = {'mathtext.default': 'regular'}
                plt.rcParams.update(params)

                fig, ax = fig_init()
                pred = model_f(time=file_id_ / n_frames) ** 2
                pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                pred = to_numpy(torch.sqrt(pred))
                pred = np.flipud(pred)
                pred = np.rot90(pred, 1)
                pred = np.fliplr(pred)
                grey_values = np.reshape(pred, (n_nodes_per_axis * n_nodes_per_axis))
                plt.scatter(to_numpy(x_mesh[:, 1]), 1 - to_numpy(x_mesh[:, 2]),
                            c=grey_values, s=20, cmap='gray')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/field_{epoch}.tif", dpi=80)
                plt.close()


                fig, ax = fig_init(fontsize=24)
                params = {'mathtext.default': 'regular'}
                plt.rcParams.update(params)
                embedding = get_embedding(model.a, 1)
                for n in range(n_particle_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n)
                    pos = to_numpy(pos)
                    if len(pos) > 0:
                        plt.scatter(embedding[pos, 0], embedding[pos, 1], color=cmap.color(n), s=100, alpha=0.1)
                plt.xlabel(r'$a_{i0}$', fontsize=48)
                plt.ylabel(r'$a_{i1}$', fontsize=48)
                plt.xlim([0.2, 1.6])
                plt.ylim([0.2, 1.6])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()


                fig, ax = fig_init(fontsize=24)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                    embedding_ = model.a[1,n] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
                plt.xlabel('$d_{ij}$', fontsize=48)
                plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                plt.xlim([0, max_radius])
                plt.ylim(config.plotting.ylim)
                plt.tight_layout()
                match config.dataset:
                    case 'arbitrary_3':
                        plt.ylim([-0.04, 0.03])
                    case 'arbitrary_16':
                        plt.ylim([-0.1, 0.1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
                plt.close()
    else:
        for epoch in epoch_list:
            print(f'epoch: {epoch}')

            net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])

            if has_siren:
                net = f'{log_dir}/models/best_model_f_with_1_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])

            config.training.cluster_method = 'distance_plot'
            config.training.cluster_distance_threshold = 0.01
            alpha = 0.1
            accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config,embedding_cluster,
                                                                           cmap, index_particles, type_list,
                                                                           n_particle_types, n_particles, ynorm, epoch,
                                                                           log_dir, alpha, style,device)
            print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
            logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')


            fig, ax = fig_init()
            p = torch.load(f'graphs_data/{dataset_name}/model_p.pt', map_location=device)
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
            rmserr_list = []
            for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                      p[to_numpy(type_list[n]).astype(int)].squeeze())
                rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.1)
            if 'latex' in style:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
            else:
                plt.xlabel(r'$d_{ij}$', fontsize=68)
                plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_function_{epoch}.tif", dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            plt.close()


            match config.graph_model.field_type:

                case 'siren_with_time' | 'siren':

                    os.makedirs(f"./{log_dir}/results/video", exist_ok=True)
                    os.makedirs(f"./{log_dir}/results/video/generated1", exist_ok=True)
                    os.makedirs(f"./{log_dir}/results/video/generated2", exist_ok=True)
                    os.makedirs(f"./{log_dir}/results/video/target", exist_ok=True)
                    os.makedirs(f"./{log_dir}/results/video/field", exist_ok=True)
                    s_p = 100

                    x_mesh = x_mesh_list[0][0].clone().detach()
                    i0 = imread(f'graphs_data/{node_value_map}')

                    print('Output per frame ...')

                    plt.rcParams['text.usetex'] = False
                    plt.rc('font', family='sans-serif')
                    plt.rc('text', usetex=False)
                    matplotlib.rcParams['savefig.pad_inches'] = 0

                    plt.style.use('dark_background')


                    RMSE_list = []
                    PSNR_list = []
                    SSIM_list = []
                    for frame in trange(0, n_frames):
                        x = x_list[0][frame].clone().detach()
                        fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                        # plt.xlabel(r'$x$', fontsize=48)
                        # plt.ylabel(r'$y$', fontsize=48)
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                        s=20,
                                        color='w')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/video/generated1/generated_1_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()

                        fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                        # plt.xlabel(r'$x$', fontsize=48)
                        # plt.ylabel(r'$y$', fontsize=48)
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                        color=cmap.color(n), s=20)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/video/generated2/generated_2_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()

                        i0_ = i0[frame]
                        y = i0_[(to_numpy(x_mesh[:, 2]) * 100).astype(int), (to_numpy(x_mesh[:, 1]) * 100).astype(int)]
                        y = np.reshape(y, (n_nodes_per_axis, n_nodes_per_axis))
                        # fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
                        # plt.imshow(y, cmap=cc, vmin=0, vmax=vm)
                        # # plt.xlabel(r'$x$', fontsize=48)
                        # # plt.ylabel(r'$y$', fontsize=48)
                        # fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                        # fmty = lambda x, pos: '{:.1f}'.format((100-x) / 100, pos)
                        # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
                        # ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
                        # # plt.xlabel(r'$x$', fontsize=48)
                        # # plt.ylabel(r'$y$', fontsize=48)
                        # plt.xticks([])
                        # plt.yticks([])
                        # plt.tight_layout()
                        # plt.savefig(f"./{log_dir}/results/video/target/target_field_{epoch}_{frame}.tif",
                        #             dpi=150)
                        # plt.close()

                        fig, ax = fig_init()
                        grey_values = np.reshape(i0_, (n_nodes_per_axis * n_nodes_per_axis))
                        plt.scatter(to_numpy(x_mesh[:, 1]), 1-to_numpy(x_mesh[:, 2]),
                                    c=grey_values, s=20, cmap='gray')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/video/target/target_field_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()

                        # pred = model_f(time=frame / n_frames) ** 2
                        # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                        # pred = to_numpy(torch.sqrt(pred))
                        # pred = np.flipud(pred)
                        # fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
                        # pred = np.rot90(pred,1)
                        # pred = np.fliplr(pred)
                        # # pred = np.flipud(pred)
                        # plt.imshow(pred, cmap=cc, vmin=0, vmax=vm)
                        # # plt.xlabel(r'$x$', fontsize=68)
                        # # plt.ylabel(r'$y$', fontsize=68)
                        # fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                        # fmty = lambda x, pos: '{:.1f}'.format((100-x) / 100, pos)
                        # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
                        # ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
                        # # plt.xlabel(r'$x$', fontsize=48)
                        # # plt.ylabel(r'$y$', fontsize=48)
                        # plt.xticks([])
                        # plt.yticks([])
                        # plt.tight_layout()
                        # plt.savefig(f"./{log_dir}/results/video/field/reconstructed_field_{epoch}_{frame}.tif",
                        #             dpi=150)
                        # plt.close()

                        fig, ax = fig_init()
                        pred = model_f(time=frame / n_frames) ** 2
                        pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                        pred = to_numpy(torch.sqrt(pred))
                        pred = np.flipud(pred)
                        pred = np.rot90(pred,1)
                        pred = np.fliplr(pred)
                        grey_values = np.reshape(pred, (n_nodes_per_axis * n_nodes_per_axis))
                        plt.scatter(to_numpy(x_mesh[:, 1]), 1-to_numpy(x_mesh[:, 2]),
                                    c=grey_values, s=20, cmap='gray')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/video/field/reconstructed_field_{epoch}_{frame}.tif",
                                    dpi=150)
                        plt.close()

                        RMSE = np.sqrt(np.mean((y - pred) ** 2))
                        RMSE_list = np.concatenate((RMSE_list, [RMSE]))
                        PSNR = calculate_psnr(y, pred, max_value=np.max(y))
                        PSNR_list = np.concatenate((PSNR_list, [PSNR]))
                        SSIM = calculate_ssim(y, pred)
                        SSIM_list = np.concatenate((SSIM_list, [SSIM]))
                        if frame==0:
                            y_list = [y]
                            pred_list = [pred]
                        else:
                            y_list = np.concatenate((y_list, [y]))
                            pred_list = np.concatenate((pred_list, [pred]))

                    fig, ax = fig_init(formatx='%.2f', formaty='%.2f')
                    plt.scatter(y_list, pred_list, color='w', s=0.1, alpha=0.01)
                    plt.xlabel(r'True $b_i(t)$', fontsize=48)
                    plt.ylabel(r'Recons. $b_i(t)$', fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/cues_scatter_{epoch}.tif", dpi=170)
                    plt.close()

                    r, p_value = pearsonr(y_list.flatten(), pred_list.flatten())
                    print(f"Pearson's r: {r:.4f}, p-value: {p_value:.6f}")
                    logger.info(f"Pearson's r: {r:.4f}, p-value: {p_value:.6f}")


                    fig, ax = fig_init()
                    plt.scatter(np.linspace(0, n_frames, len(SSIM_list)), SSIM_list, color=mc, linewidth=4)
                    plt.xlabel(r'$Frame$', fontsize=68)
                    plt.ylabel(r'$SSIM$', fontsize=68)
                    plt.ylim([0, 1])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/ssim_{epoch}.tif", dpi=150)
                    plt.close()

                    print(f'SSIM: {np.round(np.mean(SSIM_list), 3)}+/-{np.round(np.std(SSIM_list), 3)}')

                    fig, ax = fig_init()
                    plt.scatter(np.linspace(0, n_frames, len(SSIM_list)), RMSE_list, color=mc, linewidth=4)
                    plt.xlabel(r'$Frame$', fontsize=68)
                    plt.ylabel(r'RMSE', fontsize=68)
                    plt.ylim([0, 1])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/rmse_{epoch}.tif", dpi=150)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(np.linspace(0, n_frames, len(SSIM_list)), PSNR_list, color=mc, linewidth=4)
                    plt.xlabel(r'$Frame$', fontsize=68)
                    plt.ylabel(r'PSNR', fontsize=68)
                    plt.ylim([0, 50])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/psnr_{epoch}.tif", dpi=150)
                    plt.close()

                case 'tensor':

                    fig, ax = fig_init()
                    pts = to_numpy(torch.reshape(model.field[1], (100, 100)))
                    pts = np.flipud(pts)
                    plt.imshow(pts, cmap=cc)
                    fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    fmty = lambda x, pos: '{:.1f}'.format((100 - x) / 100, pos)
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
                    plt.xlabel(r'$x$', fontsize=68)
                    plt.ylabel(r'$y$', fontsize=68)
                    # cbar = plt.colorbar(shrink=0.5)
                    # cbar.ax.tick_params(labelsize=32)
                    # cbar.set_label(r'$Coupling$',fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field_{epoch}.tif", dpi=300)
                    # np.save(f"./{log_dir}/results/embedding.npy", csv_)
                    # csv_= np.reshape(csv_,(csv_.shape[0]*csv_.shape[1],2))
                    # np.savetxt(f"./{log_dir}/results/embedding.txt", csv_)
                    plt.close()
                    rmse = np.sqrt(np.mean((target - pts) ** 2))
                    print(f'RMSE: {rmse}')
                    logger.info(f'RMSE: {rmse}')

                    fig, ax = fig_init()
                    plt.scatter(target, pts, c=mc, s=10, alpha=0.1)
                    plt.xlabel(r'True $b_i$', fontsize=68)
                    plt.ylabel(r'Recons. $b_i$', fontsize=68)

                    x_data = np.reshape(pts, (n_nodes))
                    y_data = np.reshape(target, (n_nodes))
                    threshold = 0.25
                    relative_error = np.abs(y_data - x_data)
                    print(f'outliers: {np.sum(relative_error > threshold)} / {n_particles}')
                    pos = np.argwhere(relative_error < threshold)

                    x_data_ = x_data[pos].squeeze()
                    y_data_ = y_data[pos].squeeze()

                    lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
                    residuals = y_data_ - linear_model(x_data_, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)

                    plt.plot(x_data_, linear_model(x_data_, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                    plt.xlim([0, 2])
                    plt.ylim([-0, 2])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field_scatter_{epoch}.tif", dpi=300)

                    print(f'R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}')
                    logger.info(f'R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}')


def plot_RD_RPS(config, epoch_list, log_dir, logger, cc, style, device):

    dataset_name = config.dataset

    n_nodes = config.simulation.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_node_types = config.simulation.n_node_types
    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    delta_t = config.simulation.delta_t
    cmap = CustomColorMap(config=config)

    embedding_cluster = EmbeddingCluster(config)

    hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(n_runs):
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()

    print(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    x_mesh = x_mesh_list[1][0].clone().detach()

    i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
    coeff = i0[(to_numpy(x_mesh[:, 2]) * 255).astype(int), (to_numpy(x_mesh[:, 1]) * 255).astype(int)]
    coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis))
    vm = np.max(coeff)
    print(f'vm: {vm}')

    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(np.flipud(coeff), vmin=0, vmax=vm, cmap='grey')
    plt.xlabel(r'$x$', fontsize=68)
    plt.ylabel(r'$y$', fontsize=68)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_coeff.tif", dpi=300)
    plt.close()
    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, vmin=0, vmax=vm, cmap='grey')
    plt.xlabel(r'$x$', fontsize=68)
    plt.ylabel(r'$y$', fontsize=68)
    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_coeff_cbar.tif", dpi=300)
    plt.close()

    for epoch in epoch_list:

        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"

        model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print(f'net: {net}')
        embedding = get_embedding(model.a, 1)

        cluster_method = 'distance_embedding'
        cluster_distance_threshold = 0.01
        labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
        labels_map = np.reshape(labels, (n_nodes_per_axis, n_nodes_per_axis))
        fig, ax = fig_init()
        plt.imshow(labels_map, cmap='tab20', vmin=0, vmax=10)
        fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.xlabel(r'$x$', fontsize=68)
        plt.ylabel(r'$y$', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/labels_map_cbar.tif", dpi=300)
        plt.close

        fig, ax = fig_init()
        for nodes_type in np.unique(labels[labels <5]):
            pos = np.argwhere(labels == nodes_type)
            plt.scatter(embedding[pos, 0], embedding[pos, 1], s=400, cmap=cmap.color(nodes_type*2))
        if 'latex' in style:
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
        else:
            plt.xlabel(r'$a_{i0}$', fontsize=68)
            plt.ylabel(r'$a_{i1}$', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/embedding_{epoch}.tif", dpi=300)
        plt.close()

        if True:

            k = 2400

            # collect data
            x_mesh = x_mesh_list[1][k].clone().detach()
            dataset = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
            with torch.no_grad():
                pred, laplacian_uvw, uvw, embedding, input_phi = model(dataset, data_id=1, return_all=True)
            pred = pred * hnorm
            y = y_mesh_list[1][k].clone().detach()

            # RD_RPS_model :
            c_ = torch.zeros(n_node_types, 1, device=device)
            for n in range(n_node_types):
                c_[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
            c = c_[to_numpy(dataset.x[:, 5])].squeeze()
            c = torch.tensor(np.reshape(coeff,(n_nodes_per_axis*n_nodes_per_axis)),device=device)
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

            lin_fit_true = np.zeros((len(np.unique(labels))-1, 3, 10))
            lin_fit_reconstructed = np.zeros((len(np.unique(labels))-1, 3, 10))
            eq_list = ['u', 'v', 'w']
            # class 0 is discarded (borders)
            for n in np.unique(labels)[1:]-1:
                print(n)
                pos = np.argwhere((labels == n+1) & (to_numpy(mask_mesh.squeeze()) == 1))
                pos = pos[:, 0].astype(int)
                for it, eq in enumerate(eq_list):
                    fitting_model = reaction_diffusion_model(eq)
                    laplacian_u = to_numpy(laplacian_uvw[pos, 0])
                    laplacian_v = to_numpy(laplacian_uvw[pos, 1])
                    laplacian_w = to_numpy(laplacian_uvw[pos, 2])
                    u = to_numpy(uvw[pos, 0])
                    v = to_numpy(uvw[pos, 1])
                    w = to_numpy(uvw[pos, 2])
                    x_data = np.concatenate((laplacian_u[:, None], laplacian_v[:, None], laplacian_w[:, None],
                                             u[:, None], v[:, None], w[:, None]), axis=1)
                    y_data = to_numpy(increment[pos, 0 + it:1 + it])
                    p0 = np.ones((10, 1))
                    lin_fit, lin_fitv = curve_fit(fitting_model, np.squeeze(x_data), np.squeeze(y_data),
                                                  p0=np.squeeze(p0), method='trf')
                    lin_fit_true[n, it] = lin_fit
                    y_data = to_numpy(pred[pos, it:it + 1])
                    lin_fit, lin_fitv = curve_fit(fitting_model, np.squeeze(x_data), np.squeeze(y_data),
                                                  p0=np.squeeze(p0), method='trf')
                    lin_fit_reconstructed[n, it] = lin_fit

            coeff_reconstructed = np.round(np.median(lin_fit_reconstructed, axis=0), 2)
            diffusion_coeff_reconstructed = np.round(np.median(lin_fit_reconstructed, axis=1), 2)[:, 9]
            coeff_true = np.round(np.median(lin_fit_true, axis=0), 2)
            diffusion_coeff_true = np.round(np.median(lin_fit_true, axis=1), 2)[:, 9]

            print(f'frame {k}')
            print(f'coeff_reconstructed: {coeff_reconstructed}')
            print(f'diffusion_coeff_reconstructed: {diffusion_coeff_reconstructed}')
            print(f'coeff_true: {coeff_true}')
            print(f'diffusion_coeff_true: {diffusion_coeff_true}')

            cp = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'u', 'v', 'w']
            results = {
                'True': coeff_true[0, 0:9],
                'Learned': coeff_reconstructed[0, 0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig, ax = fig_init()
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient', fontsize=68)
            ax.set_xticks(x + width, cp, fontsize=36)
            plt.title('First equation', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/first_equation_{epoch}.tif", dpi=300)
            plt.close()
            cp = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'u', 'v', 'w']
            results = {
                'True': coeff_true[1, 0:9],
                'Learned': coeff_reconstructed[1, 0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig, ax = fig_init()
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient', fontsize=68)
            ax.set_xticks(x + width, cp, fontsize=36)
            plt.title('Second equation', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/second_equation_{epoch}.tif", dpi=300)
            plt.close()
            cp = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'u', 'v', 'w']
            results = {
                'True': coeff_true[2, 0:9],
                'Learned': coeff_reconstructed[2, 0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig, ax = fig_init()
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient', fontsize=68)
            ax.set_xticks(x + width, cp, fontsize=36)
            plt.title('Third equation', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/third_equation_{epoch}.tif", dpi=300)
            plt.close()

            true_diffusion_coeff = [0.01, 0.02, 0.03, 0.04]

            fig, ax = fig_init(formatx='%.3f', formaty='%.3f')
            x_data = np.array(true_diffusion_coeff)
            y_data = diffusion_coeff_reconstructed
            plt.scatter(x_data, y_data, c=mc, s=400)
            plt.ylabel(r'Learned diffusion coeff.', fontsize=64)
            plt.xlabel(r'True diffusion coeff.', fontsize=64)
            plt.xlim([0, vm * 1.1])
            plt.ylim([0, vm * 1.1])
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/scatter_{epoch}.tif", dpi=300)
            plt.close()

            print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}")

            fig, ax = fig_init(formatx='%.3f', formaty='%.3f')
            x_data = coeff_true.flatten()
            y_data = coeff_reconstructed.flatten()
            plt.scatter(x_data, y_data, c=mc, s=400)
            plt.ylabel(r'Learned coeff.', fontsize=64)
            plt.xlabel(r'True  coeff.', fontsize=64)
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.tight_layout()
            print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}")


def plot_synaptic2(config, epoch_list, log_dir, logger, cc, style, device):

    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_particle_types = config.simulation.n_particle_types
    delta_t = config.simulation.delta_t
    p = config.simulation.params
    omega = model_config.omega
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    embedding_cluster = EmbeddingCluster(config)
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = True
    else:
        has_field = False

    x_list = []
    y_list = []
    for run in trange(1):
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
            x = to_numpy(torch.stack(x))
            y = to_numpy(torch.stack(y))
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    print('update variables ...')
    x = x_list[0][n_frames - 1]
    n_particles = x.shape[0]
    print(f'N neurons: {n_particles}')
    logger.info(f'N neurons: {n_particles}')
    config.simulation.n_particles = n_particles
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)

    # activity = torch.tensor(x_list[0],device=device)
    # activity = activity[:, :, 6:7].squeeze()
    # distrib = to_numpy(activity.flatten())
    # activity = activity.t()

    activity = torch.tensor(x_list[0][:, :, 6:7],device=device)
    activity = activity.squeeze()
    distrib = to_numpy(activity.flatten())
    activity = activity.t()

    # plt.figure(figsize=(15, 10))
    # window_size = 25
    # window_end = 50000
    # ts = to_numpy(activity[600, :])
    # ts_avg = np.convolve(ts, np.ones(window_size) / window_size, mode='valid')
    # plt.plot(ts[window_size // 2:window_end + window_size // 2], linewidth=1)
    # plt.plot(ts_avg, linewidth=2)
    # plt.plot(ts[window_size // 2:window_end + window_size // 2] - ts_avg[0:window_end])
    # plt.xlim([window_end - 5000, window_end])
    # plt.close
    # signal_power = np.mean(ts_avg[window_size // 2:window_end + window_size // 2] ** 2)
    # # Compute the noise power
    # noise_power = np.mean((ts[window_size // 2:window_end + window_size // 2] - ts_avg[0:window_end]) ** 2)
    # # Calculate the signal-to-noise ratio (SNR)
    # snr = signal_power / noise_power
    # print(f"Signal-to-Noise Ratio (SNR): {snr:0.2f} 10log10 {10 * np.log10(snr):0.2f}")

    # # Parameters
    # fs = 1000  # Sampling frequency
    # t = np.arange(0, 1, 1 / fs)  # Time vector
    # frequency = 5  # Frequency of the sine wave
    # desired_snr_db = snr  # Desired SNR in dB
    # # Generate a clean signal (sine wave)
    # clean_signal = np.sin(2 * np.pi * frequency * t)
    # # Calculate the power of the clean signal
    # signal_power = np.mean(clean_signal ** 2)
    # # Calculate the noise power required to achieve the desired SNR
    # desired_snr_linear = 10 ** (desired_snr_db / 10)
    # noise_power = signal_power / desired_snr_linear
    # # Generate noise with the calculated power
    # noise = np.sqrt(noise_power) * np.random.randn(len(t))
    # # Create a noisy signal by adding noise to the clean signal
    # noisy_signal = clean_signal + noise
    # # Plot the clean signal and the noisy signal
    # plt.figure(figsize=(15, 10))
    # plt.subplot(2, 1, 1)
    # plt.plot(t, clean_signal)
    # plt.title('Clean Signal')
    # plt.subplot(2, 1, 2)
    # plt.plot(t, noisy_signal)
    # plt.plot(t, noise)
    # plt.title(f'Noisy Signal with SNR = {desired_snr_db} dB')
    # plt.tight_layout()
    # plt.show()


    # if os.path.exists(f'./graphs_data/{dataset_name}/X1.pt') > 0:
    #     X1_first = torch.load(f'./graphs_data/{dataset_name}/X1.pt', map_location=device)
    #     X_msg = torch.load(f'./graphs_data/{dataset_name}/X_msg.pt', map_location=device)
    # else:
    xc, yc = get_equidistant_points(n_points=n_particles)
    X1_first = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(X1_first.size(0))
    X1_first = X1_first[perm]
    torch.save(X1_first, f'./graphs_data/{dataset_name}/X1.pt')
    xc, yc = get_equidistant_points(n_points=n_particles ** 2)
    X_msg = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(X_msg.size(0))
    X_msg = X_msg[perm]
    torch.save(X_msg, f'./graphs_data/{dataset_name}/X_msg.pt')

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    if has_field:
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                        hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device, first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        model_f.train()

        modulation = torch.tensor(x_list[0], device=device)
        modulation = modulation[:, :, 8:9].squeeze()
        modulation = modulation.t()
        modulation = modulation.clone().detach()
        d_modulation = (modulation[:, 1:] - modulation[:, :-1]) / delta_t

    if epoch_list[0] == 'all':

        files = glob.glob(f"{log_dir}/models/*.pt")
        files.sort(key=os.path.getmtime)

        model, bc_pos, bc_dpos = choose_training_model(config, device)

        # plt.rcParams['text.usetex'] = False
        # plt.rc('font', family='sans-serif')
        # plt.rc('text', usetex=False)
        # matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id]) >0:
                flag = False
                file_id = file_id - 1
            file_id += 1

        files = files[file_id:]

        # file_id_list0 = np.arange(0, file_id, file_id // 90)
        # file_id_list1 = np.arange(file_id, len(files), (len(files) - file_id) // 40)
        # file_id_list = np.concatenate((file_id_list0, file_id_list1))

        file_id_list = np.arange(0, len(files), (len(files)/100)).astype(int)
        r_squared_list = []
        slope_list = []


        with torch.no_grad():
            for file_id_ in trange(95, 100):
                file_id = file_id_list[file_id_]

                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                if simulation_config.connectivity_mask:
                    inv_mask = torch.load(f'./graphs_data/{dataset_name}/inv_mask.pt', map_location=device)
                    with torch.no_grad():
                        model.W.copy_(model.W * inv_mask)
                model.eval()

                if has_field:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                amax = torch.max(model.a, dim=0).values
                amin = torch.min(model.a, dim=0).values
                model_a = (model.a - amin) / (amax - amin)

                fig, ax = fig_init()
                for n in range(n_particle_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n).squeeze()
                    plt.scatter(to_numpy(model_a[pos, 0]), to_numpy(model_a[pos, 1]), s=100, color=cmap.color(n), alpha=0.5)
                plt.xlabel(r'$a_{i0}$', fontsize=68)
                plt.ylabel(r'$a_{i1}$', fontsize=68)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                plt.close()

                correction = torch.load(f'{log_dir}/correction.pt',map_location=device)
                second_correction = np.load(f'{log_dir}/second_correction.npy')

                i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                # fig, ax = fig_init()
                # ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1,vmax=0.1)
                # cbar = ax.collections[0].colorbar
                # cbar.ax.tick_params(labelsize=48)
                # plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=24)
                # plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=24)
                # plt.subplot(2, 2, 1)
                # ax = sns.heatmap(to_numpy(A[0:20, 0:20])/second_correction, cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
                # plt.xticks([])
                # plt.yticks([])
                # plt.tight_layout()
                # plt.savefig(f"./{log_dir}/results/all/W_{epoch}.tif", dpi=80)
                # plt.close()


                rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
                if model_config.signal_model_name == 'PDE_N5':
                    fig, ax = fig_init()
                    plt.axis('off')
                    for k in range(n_particle_types):
                        ax = fig.add_subplot(2, 2, k + 1)
                        if k==0:
                            plt.ylabel(r'learned $MLP_1(x_i, a_i, a_j)$', fontsize=32)
                        for n in range(n_particle_types):
                            for m in range(250):
                                pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                                pos1 = to_numpy(torch.argwhere(type_list == n).squeeze())
                                n0 = np.random.randint(len(pos0))
                                n0 = pos0[n0, 0]
                                n1 = np.random.randint(len(pos1))
                                n1 = pos1[n1, 0]
                                embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)
                                embedding1 = model.a[n1, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)
                                in_features = torch.cat((rr[:, None], embedding0, embedding1), dim=1)
                                func = model.lin_edge(in_features.float()) * correction
                                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(k), linewidth=2, alpha=0.25)
                        plt.ylim([-1.6, 1.6])
                        plt.xlim([-5, 5])
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.tif", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N8'):
                    fig, ax = fig_init()
                    for k in range(n_particle_types):
                        for m in range(250):
                            pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                            n0 = np.random.randint(len(pos0))
                            n0 = pos0[n0, 0]
                            embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                     device=device)
                            in_features = torch.cat((rr[:, None], embedding0), dim=1)
                            func = model.lin_edge(in_features.float()) * correction
                            plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(k), linewidth=8, alpha=0.25)
                    plt.xlabel(r'$x_i$', fontsize=68)
                    if model_config.signal_model_name == 'PDE_N8':
                        plt.ylabel(r'learned $MLP_1(x_i, a_j)$', fontsize=68)
                    else:
                        plt.ylabel(r'learned $MLP_1(x_i, a_i)$', fontsize=68)
                    plt.ylim([-1.6, 1.6])
                    plt.xlim([-5,5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.tif", dpi=80)
                    plt.close()
                else:
                    fig, ax = fig_init()
                    in_features = rr[:, None]
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float()) * correction
                    plt.plot(to_numpy(rr), to_numpy(func), color=mc, linewidth=8, label=r'learned')
                    plt.xlabel(r'$x_i$', fontsize=68)
                    # plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
                    plt.ylabel(r'learned $MLP_1(a_i, x_i)$', fontsize=68)
                    plt.ylim([-1.5, 1.5])
                    plt.xlim([-5,5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.tif", dpi=80)
                    plt.close()

                fig, ax = fig_init()
                func_list = []
                config_model = config.graph_model.signal_model_name
                for n in range(n_particles):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8 // ( 1 + (n_particle_types>16)*1.0), alpha=0.25)
                plt.xlabel(r'$x_i$', fontsize=68)
                # plt.ylabel(r'learned $\phi^*(a_i, x_i)$', fontsize=68)
                plt.ylabel(r'learned $MLP_0(a_i, x_i)$', fontsize=68)
                plt.xlim(config.plotting.xlim)
                plt.ylim(config.plotting.ylim)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP0_{epoch}.tif", dpi=80)
                plt.close()

                adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
                adjacency_ = adjacency.t().clone().detach()
                adj_t = torch.abs(adjacency_) > 0
                edge_index = adj_t.nonzero().t().contiguous()


                i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                fig, ax = fig_init()
                gt_weight = to_numpy(adjacency)
                pred_weight = to_numpy(A) / second_correction
                plt.scatter(gt_weight, pred_weight, s=0.1, c=mc, alpha=0.1)
                plt.xlabel(r'true $W_{ij}$', fontsize=68)
                plt.ylabel(r'learned $W_{ij}$', fontsize=68)
                if n_particles == 8000:
                    plt.xlim([-0.05, 0.05])
                    plt.ylim([-0.05, 0.05])
                else:
                    # plt.xlim([-0.2, 0.2])
                    # plt.ylim([-0.2, 0.2])
                    plt.xlim([-0.15, 0.15])
                    plt.ylim([-0.15, 0.15])

                x_data = np.reshape(gt_weight, (n_particles * n_particles))
                y_data = np.reshape(pred_weight, (n_particles * n_particles))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                r_squared_list.append(r_squared)
                slope_list.append(lin_fit[0])

                if n_particles == 8000:
                    plt.text(-0.042, 0.042, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.042, 0.036, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                else:
                    # plt.text(-0.17, 0.15, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    # plt.text(-0.17, 0.12, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                    plt.text(-0.13, 0.13, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.13, 0.11, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/comparison_{epoch}.tif", dpi=80)
                plt.close()

                if has_field:

                    if 'Siren_short_term_plasticity' in field_type:
                        fig, ax = fig_init()
                        t = torch.zeros((1, 100000, 1), dtype=torch.float32, device=device)
                        t[0] = torch.linspace(0, 1, 100000, dtype=torch.float32, device=device)[:, None]
                        prediction = model_f(t) ** 2
                        prediction = prediction.squeeze()
                        prediction = prediction.t()
                        plt.imshow(to_numpy(prediction), aspect='auto')
                        plt.title(r'learned $MLP_2(i,t)$', fontsize=68)
                        plt.xlabel(r'$t$', fontsize=68)
                        plt.ylabel(r'$i$', fontsize=68)
                        plt.xticks([10000,100000], [10000, 100000], fontsize=48)
                        plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/yi_{epoch}.tif", dpi=80)
                        plt.close()

                        prediction = prediction * torch.tensor(second_correction,device=device)

                        fig, ax = fig_init()
                        ids = np.arange(0,100000,100).astype(int)
                        plt.scatter(to_numpy(modulation[:,ids]), to_numpy(prediction[:,ids]), s=1, color=mc, alpha=0.1)
                        plt.xlim([0,0.5])
                        plt.ylim([0,2])
                        plt.xticks([0,0.5], [0,0.5], fontsize=48)
                        plt.yticks([0,1,2], [0,1,2], fontsize=48)
                        x_data = to_numpy(modulation[:,ids]).flatten()
                        y_data = to_numpy(prediction[:,ids]).flatten()
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        plt.xlabel(r'true $y_i(t)$', fontsize=68)
                        plt.ylabel(r'learned $y_i(t)$', fontsize=68)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/comparison_yi_{epoch}.tif", dpi=80)
                        plt.close()

                else:

                    fig, ax = fig_init()
                    pred = model_f(time=file_id_ / len(file_id_list), enlarge=True) ** 2
                    # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = torch.reshape(pred, (640, 640))
                    pred = to_numpy(torch.sqrt(pred))
                    pred = np.flipud(pred)
                    pred = np.rot90(pred, 1)
                    pred = np.fliplr(pred)
                    plt.imshow(pred, cmap='grey')
                    plt.ylabel(r'learned $MLP_2(x_i, t)$', fontsize=68)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/field_{epoch}.tif", dpi=80)
                    plt.close()

                if 'derivative' in field_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 400)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid) / 20
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig = plt.figure(figsize=(12, 12))
                    # plt.subplot(1, 2, 1)
                    # plt.title(r'true $\dot{y_i}$', fontsize=48)
                    # # plt.title(r'$\dot{y_i}=(1-y)/100 - 0.02 x_iy_i$', fontsize=48)
                    # plt.imshow(to_numpy(true_derivative))
                    # plt.xticks([0, 50, 100, 150, 200], [-8, -4, 0, 4, 8], fontsize=24)
                    # plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=24)
                    # plt.xlabel(r'$x_i$', fontsize=48)
                    # plt.ylabel(r'$y_i$', fontsize=48)
                    # # plt.colorbar()
                    # plt.subplot(1, 2, 2)
                    plt.title(r'learned $MLP_3(x_i, y_i)$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation), vmin=-0.1, vmax=0)
                    # plt.xticks([0, 50, 100, 150, 200], [-8, -4, 0, 4, 8], fontsize=24)
                    # plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=24)
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel(r'$x_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    # plt.colorbar()
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/all/derivative_yi_{epoch}.tif", dpi=80)
                    plt.close()

                    fig = plt.figure(figsize=(12, 12))
                    plt.scatter(to_numpy(true_derivative.flatten()), to_numpy(pred_modulation.flatten()), s=5, color=mc, alpha=0.1)
                    x_data = to_numpy(true_derivative.flatten())
                    y_data = to_numpy(pred_modulation.flatten())
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    plt.text(0.05, 0.75, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.text(0.05, 0.7, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.xlabel(r'true $\dot{y_i}(t)$', fontsize=68)
                    plt.ylabel(r'learned $\dot{y_i}(t)$', fontsize=68)

                    plt.xticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    plt.yticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    plt.xlim([-0.2,0.025])
                    plt.ylim([-0.2,0.025])

                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/comparison_derivative_yi_{epoch}.tif", dpi=80)
                    plt.close()





        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(r_squared_list, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('$R^2$', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/R2.png', dpi=300)
        plt.close()
        np.save(f'./{log_dir}/results/R2.npy', r_squared_list)

        slope_list = np.array(slope_list) / p[0][0]
        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(slope_list*10, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('slope', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/slope.png', dpi=300)
        plt.close()

    else:

        fig_init(formatx='%.0f', formaty='%.0f')
        plt.hist(distrib, bins=100, color=mc, alpha=0.5)
        plt.ylabel('counts', fontsize=64)
        plt.xlabel('$x_{ij}$', fontsize=64)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/signal_distribution.png', dpi=300)
        plt.close()
        print(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')
        logger.info(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')

        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(to_numpy(activity), center=0, cmap='viridis', cbar_kws={'fraction': 0.046})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        ax.invert_yaxis()
        plt.ylabel('neurons', fontsize=64)
        plt.xlabel('time', fontsize=64)
        plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.yticks([0, 999], [1, 1000], fontsize=48)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/kinograph.png', dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        n = np.random.permutation(n_particles)
        for i in range(25):
            plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        plt.xlabel('time', fontsize=64)
        plt.ylabel('$x_{i}$', fontsize=64)
        plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.yticks(fontsize=48)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/firing rate.png', dpi=300)
        plt.close()
        #
        # # if os.path.exists(f"./{log_dir}/neuron_gt_list.pt"):
        # #
        # #     os.makedirs(f"./{log_dir}/results/activity", exist_ok=True)
        # #
        # #     neuron_gt_list = torch.load(f"./{log_dir}/neuron_gt_list.pt", map_location=device)
        # #     neuron_pred_list = torch.load(f"./{log_dir}/neuron_pred_list.pt", map_location=device)
        # #
        # #     neuron_gt_list = torch.cat(neuron_gt_list, 0)
        # #     neuron_pred_list = torch.cat(neuron_pred_list, 0)
        # #     neuron_gt_list = torch.reshape(neuron_gt_list, (1000, n_particles))
        # #     neuron_pred_list = torch.reshape(neuron_pred_list, (1000, n_particles))
        # #
        # #     n = [20, 30, 100, 150, 260, 270, 520, 620, 720, 820]
        # #
        # #     r_squared_list = []
        # #     slope_list = []
        # #     for i in trange(0,750,5):
        # #         plt.figure(figsize=(20, 10))
        # #         ax = plt.subplot(121)
        # #         plt.plot(neuron_gt_list[:, n[0]].detach().cpu().numpy(), c='w', linewidth=8, label='true', alpha=0.25)
        # #         plt.plot(neuron_pred_list[0:i, n[0]].detach().cpu().numpy(), linewidth=4, c='w', label='learned')
        # #         plt.legend(fontsize=24)
        # #         plt.plot(neuron_gt_list[:, n[1:10]].detach().cpu().numpy(), c='w', linewidth=8, alpha=0.25)
        # #         plt.plot(neuron_pred_list[0:i, n[1:10]].detach().cpu().numpy(), linewidth=4)
        # #         plt.xlim([0, 750])
        # #         plt.xlabel('time index', fontsize=48)
        # #         plt.ylabel(r'$x_i$', fontsize=48)
        # #         plt.xticks(fontsize=24)
        # #         plt.yticks(fontsize=24)
        # #         plt.ylim([-30,30])
        # #         plt.text(40, 26, f'time: {i}', fontsize=34)
        # #         ax = plt.subplot(122)
        # #         plt.scatter(to_numpy(neuron_gt_list[i, :]), to_numpy(neuron_pred_list[i, :]), s=10, c=mc)
        # #         plt.xlim([-30,30])
        # #         plt.ylim([-30,30])
        # #         plt.xticks(fontsize=24)
        # #         plt.yticks(fontsize=24)
        # #         x_data = to_numpy(neuron_gt_list[i, :])
        # #         y_data = to_numpy(neuron_pred_list[i, :])
        # #         lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        # #         residuals = y_data - linear_model(x_data, *lin_fit)
        # #         ss_res = np.sum(residuals ** 2)
        # #         ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        # #         r_squared = 1 - (ss_res / ss_tot)
        # #         r_squared_list.append(r_squared)
        # #         slope_list.append(lin_fit[0])
        # #         plt.xlabel(r'true $x_i$', fontsize=48)
        # #         plt.ylabel(r'learned $x_i$', fontsize=48)
        # #         plt.text(-28, 25.6, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
        # #         plt.text(-28, 22, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
        # #         plt.tight_layout()
        # #         plt.savefig(f'./{log_dir}/results/activity/comparison_{i}.png', dpi=80)
        # #         plt.close()
        # #
        # #     plt.figure(figsize=(10, 10))
        # #     plt.plot(r_squared_list, linewidth=4, label='$R^2$')
        # #     plt.plot(slope_list, linewidth=4, label='slope')
        # #     plt.xticks([0,75,150],[0,375,750],fontsize=24)
        # #     plt.yticks(fontsize=24)
        # #     plt.ylim([0,1.4])
        # #     plt.xlim([0,150])
        # #     plt.xlabel(r'time', fontsize=48)
        # #     plt.title(r'true vs learned $x_i$', fontsize=48)
        # #     plt.legend(fontsize=24)
        # #     plt.tight_layout()
        # #     plt.savefig(f'./{log_dir}/results/activity_comparison.png', dpi=80)
        # #     plt.close()


        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        adjacency_ = adjacency.t().clone().detach()
        adj_t = torch.abs(adjacency_) > 0
        edge_index = adj_t.nonzero().t().contiguous()
        weights = to_numpy(adjacency.flatten())
        pos = np.argwhere(weights != 0)
        weights = weights[pos]

        fig_init()
        plt.hist(weights, bins=1000, color=mc, alpha=0.5)
        plt.ylabel(r'counts', fontsize=64)
        plt.xlabel(r'$W$', fontsize=64)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.xlim([-0.1, 0.1])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/weights_distribution.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt'
            model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edge_index
            print(f'net: {net}')

            fig, ax = fig_init()
            for n in range(n_particle_types,-1,-1):
                pos = torch.argwhere(type_list == n).squeeze()
                plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=200, color=cmap.color(n), alpha=0.1)
            if 'latex' in style:
                plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
                plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
            else:
                plt.xlabel(r'$a_{i0}$', fontsize=68)
                plt.ylabel(r'$a_{i1}$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/all_embedding_{epoch}.tif", dpi=170.7)
            plt.close()

            fig, ax = fig_init()
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            for n in trange(0,n_particles,n_particles//100):
                if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5') | (model_config.signal_model_name == 'PDE_N8'):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                else:
                    in_features = rr[:, None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func_list.append(func)
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=8 // ( 1 + (n_particle_types>16)*1.0), alpha=0.25)
            func_list = torch.stack(func_list).squeeze()
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'Learned $\psi^*(a_i, x_i)$', fontsize=68)
            # if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
            #     plt.ylim([-0.5,0.5])
            plt.xlim([-5,5])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/raw_psi.tif", dpi=170.7)
            plt.close()

            upper = func_list[:,950:1000].flatten()
            upper = torch.sort(upper, descending=True).values
            correction = 1 / torch.mean(upper[:upper.shape[0]//10])
            # correction = 1 / torch.mean(torch.mean(func_list[:,900:1000], dim=0))
            print(f'correction: {to_numpy(correction):0.2f}')
            torch.save(correction, f'{log_dir}/correction.pt')

            print('update functions ...')
            if model_config.signal_model_name == 'PDE_N5':
                psi_list = []
                fig, ax = fig_init()
                rr = torch.tensor(np.linspace(-7.5, 7.5, 1500)).to(device)

                ax.set_frame_on(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                for k in range(n_particle_types):
                    ax = fig.add_subplot(2, 2, k + 1)
                    for m in range(n_particle_types):
                        true_func = true_model.func(rr, k, m, 'phi')
                        plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=8, label='original', alpha=0.21)
                    for n in range(n_particle_types):
                        for m in range(250):
                            pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                            pos1 = to_numpy(torch.argwhere(type_list == n).squeeze())
                            n0 = np.random.randint(len(pos0))
                            n0 = pos0[n0,0]
                            n1 = np.random.randint(len(pos1))
                            n1 = pos1[n1,0]
                            embedding0 = model.a[n0, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                            embedding1 = model.a[n1, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                            in_features = torch.cat((rr[:,None],embedding0, embedding1), dim=1)
                            func = model.lin_edge(in_features.float()) * correction
                            psi_list.append(func)
                            plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(k),linewidth=1, alpha=0.25)
                    plt.ylim([-1.1, 1.1])
                    plt.xlim([-5, 5])
                    plt.xticks(fontsize=18)
                    plt.yticks(fontsize=18)
                    # plt.ylabel(r'learned $\psi^*(a_i, a_j, x_i)$', fontsize=24)
                    # plt.xlabel(r'$x_i$', fontsize=24)
                    # plt.ylim([-1.5, 1.5])
                    # plt.xlim([-5, 5])

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/learned_psi.tif", dpi=170.7)
                plt.close()
                psi_list = torch.stack(psi_list)
                psi_list = psi_list.squeeze()
            else:
                psi_list = []
                fig, ax = fig_init()
                rr = torch.tensor(np.linspace(-7.5, 7.5, 1500)).to(device)
                if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N8'):
                    for n in range(n_particle_types):
                        true_func = true_model.func(rr, n, 'phi')
                        plt.plot(to_numpy(rr), to_numpy(true_func), c = 'k', linewidth = 16, label = 'original', alpha = 0.21)
                else:
                    true_func = true_model.func(rr, 0, 'phi')
                    plt.plot(to_numpy(rr), to_numpy(true_func), c = 'k', linewidth = 16, label = 'original', alpha = 0.21)

                for n in trange(0,n_particles):
                    if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5') | (model_config.signal_model_name == 'PDE_N8'):
                        embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                        in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                    else:
                        in_features = rr[:, None]
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float()) * correction
                        psi_list.append(func)
                    if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5') | (model_config.signal_model_name == 'PDE_N8'):
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)), linewidth=2, alpha=0.25)
                    else:
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)
                plt.xlabel(r'$x_i$', fontsize=68)
                if model_config.signal_model_name == 'PDE_N4':
                    plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
                elif model_config.signal_model_name == 'PDE_N8':
                    plt.ylabel(r'learned $\psi^*(a_j, x_i)$', fontsize=68)
                elif model_config.signal_model_name == 'PDE_N5':
                    plt.ylabel(r'learned $\psi^*(a_i, a_j, x_i)$', fontsize=68)
                else:
                    plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
                plt.ylim([-1.1, 1.1])
                plt.xlim(config.plotting.xlim)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/learned_psi.tif", dpi=170.7)
                plt.close()
                psi_list = torch.stack(psi_list)
                psi_list = psi_list.squeeze()

            print('interaction functions ...')

            fig, ax = fig_init()
            for n in trange(n_particle_types):
                if model_config.signal_model_name == 'PDE_N5':
                    true_func = true_model.func(rr, n, n, 'update')
                else:
                    true_func = true_model.func(rr, n, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=16, label='original', alpha=0.21)
            phi_list = []
            for n in trange(n_particles):
                embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                func = func[:, 0]
                phi_list.append(func)
                plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25)
            phi_list = torch.stack(phi_list)
            func_list_ = to_numpy(phi_list)
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'learned $\phi^*(a_i, x_i)$', fontsize=68)
            plt.tight_layout()
            plt.xlim(config.plotting.xlim)
            plt.ylim(config.plotting.ylim)
            plt.savefig(f'./{log_dir}/results/learned phi.png', dpi=300)
            plt.close()

            print('UMAP reduction ...')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0,
                                  random_state=config.training.seed).fit(func_list_)
                proj_interaction = trans.transform(func_list_)

            proj_interaction = (proj_interaction - np.min(proj_interaction)) / (
                    np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
            fig, ax = fig_init()
            for n in trange(n_particle_types):
                pos = torch.argwhere(type_list == n)
                pos = to_numpy(pos)
                if len(pos) > 0:
                    plt.scatter(proj_interaction[pos, 0],
                                proj_interaction[pos, 1], s=200, alpha=0.1)
            plt.xlabel(r'UMAP 0', fontsize=68)
            plt.ylabel(r'UMAP 1', fontsize=68)
            plt.xlim([-0.2, 1.2])
            plt.ylim([-0.2, 1.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/UMAP_{epoch}.tif", dpi=170.7)
            plt.close()

            config.training.cluster_distance_threshold = 0.1
            config.training.cluster_method = 'distance_embedding'
            embedding = to_numpy(model.a.squeeze())
            labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
                                                              config.training.cluster_distance_threshold, type_list,
                                                              n_particle_types, embedding_cluster)
            accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            print(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}  ')
            logger.info(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method} ')

            # config.training.cluster_method = 'kmeans_auto_embedding'
            # labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
            #                                                   config.training.cluster_distance_threshold, type_list,
            #                                                   n_particle_types, embedding_cluster)
            # accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            # print(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}  ')
            # logger.info(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method} ')

            plt.figure(figsize=(10, 10))
            plt.scatter(to_numpy(X1_first[:, 0]), to_numpy(X1_first[:, 1]), s=150, color=cmap.color(to_numpy(type_list).astype(int)))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_types_{epoch}.tif", dpi=170.7)
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.scatter(to_numpy(X1_first[:, 0]), to_numpy(X1_first[:, 1]), s=150, color=cmap.color(new_labels.astype(int)))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_types_{epoch}.tif", dpi=170.7)
            plt.close()

            i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
            A = model.W.clone().detach() / correction
            A[i, i] = 0

            fig, ax = fig_init()
            gt_weight = to_numpy(adjacency)
            pred_weight = to_numpy(A)
            plt.scatter(gt_weight, pred_weight / 10 , s=0.1, c=mc, alpha=0.1)
            plt.xlabel(r'true $W_{ij}$', fontsize=68)
            plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            if n_particles == 8000:
                plt.xlim([-0.05,0.05])
                plt.ylim([-0.05,0.05])
            else:
                plt.xlim([-0.2,0.2])
                plt.ylim([-0.2,0.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/first_comparison_{epoch}.tif", dpi=87)
            plt.close()

            x_data = np.reshape(gt_weight, (n_particles * n_particles))
            y_data =  np.reshape(pred_weight, (n_particles * n_particles))
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
            logger.info(f'R^2$: {np.round(r_squared, 4)}  slope: {np.round(lin_fit[0], 4)}')

            second_correction = lin_fit[0]
            print(f'second_correction: {second_correction:0.2f}')
            np.save(f'{log_dir}/second_correction.npy', second_correction)

            fig, ax = fig_init()
            gt_weight = to_numpy(adjacency)
            pred_weight = to_numpy(A)
            plt.scatter(gt_weight, pred_weight / second_correction, s=0.1, c=mc, alpha=0.1)
            plt.xlabel(r'true $W_{ij}$', fontsize=68)
            plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            if n_particles == 8000:
                plt.xlim([-0.05,0.05])
                plt.ylim([-0.05,0.05])
            else:
                plt.xlim([-0.2,0.2])
                plt.ylim([-0.2,0.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/second_comparison_{epoch}.tif", dpi=87)
            plt.close()

            plt.figure(figsize=(10, 10))
            # plt.title(r'learned $W_{ij}$', fontsize=68)
            ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1,vmax=0.1)
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=32)
            plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
            plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
            plt.xticks(rotation=0)
            plt.subplot(2, 2, 1)
            ax = sns.heatmap(to_numpy(A[0:20, 0:20]/second_correction), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
            plt.xticks(rotation=0)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/learned connectivity.png', dpi=300)
            plt.close()

            if has_field:

                print('plot field ...')

                os.makedirs(f"./{log_dir}/results/field", exist_ok=True)

                if 'derivative' in field_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 200)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid)
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig = plt.figure(figsize=(16, 12))
                    plt.subplot(1, 2, 1)
                    plt.title(r'true $\dot{y_i}$', fontsize=48)
                    # plt.title(r'$\dot{y_i}=(1-y)/100 - 0.02 x_iy_i$', fontsize=48)
                    plt.imshow(to_numpy(true_derivative))
                    plt.xticks([0, 50, 100, 150, 200], [-6, -3, 0, 3, 6], fontsize=24)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=24)
                    plt.xlabel(r'$x_i$', fontsize=48)
                    plt.ylabel(r'$y_i$', fontsize=48)
                    # plt.colorbar()
                    plt.subplot(1, 2, 2)
                    plt.title(r'learned $\dot{y_i}$', fontsize=48)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([0, 50, 100, 150, 200], [-6, -3, 0, 3, 6], fontsize=24)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=24)
                    plt.xlabel(r'$x_i$', fontsize=48)
                    plt.ylabel(r'$y_i$', fontsize=48)
                    # plt.colorbar()
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/field_derivative.tif", dpi=80)
                    plt.close()

                    # fig = plt.figure(figsize=(12, 12))
                    # ind_list = [320]
                    # ids = np.arange(0, 100000, 100)
                    # ax = fig.add_subplot(2, 1, 1)
                    # for ind in ind_list:
                    #     plt.plot(to_numpy(modulation[ind, ids]))
                    #     plt.plot(to_numpy(model.b[ind, 0:1000]**2))

                else:

                    net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                    im = imread(f"graphs_data/{simulation_config.node_value_map}")

                    x = x_list[0][0]

                    slope_list = list([])
                    im_list = list([])
                    pred_list = list([])

                    for frame in trange(0, n_frames, n_frames // 100):

                        fig, ax = fig_init()
                        im_ = np.zeros((44, 44))
                        if (frame >= 0) & (frame < n_frames):
                            im_ = im[int(frame / n_frames * 256)].squeeze()
                        plt.imshow(im_, cmap='gray', vmin=0, vmax=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/true_field{epoch}_{frame}.tif", dpi=80)
                        plt.close()

                        pred = model_f(time=frame / n_frames, enlarge=True) ** 2 * second_correction / 10
                        pred = torch.reshape(pred, (640, 640))
                        pred = to_numpy(pred)
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray', vmin=0, vmax=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_HR {epoch}_{frame}.tif", dpi=80)
                        plt.close()

                        pred = model_f(time=frame / n_frames, enlarge=False) ** 2 * second_correction / 10
                        pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                        pred = to_numpy(pred)
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray', vmin=0, vmax=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_LR {epoch}_{frame}.tif", dpi=80)
                        plt.close()

                        x_data = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                        y_data = np.reshape(pred, (n_nodes_per_axis * n_nodes_per_axis))
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        # print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
                        slope_list.append(lin_fit[0])

                        fig, ax = fig_init()
                        plt.scatter(im_, pred, s=10, c=mc)
                        plt.xlim([0.3, 1.6])
                        plt.ylim([0.3, 1.6])
                        plt.xlabel(r'true neuromodulation', fontsize=48)
                        plt.ylabel(r'learned neuromodulation', fontsize=48)
                        plt.text(0.35, 1.5, f'$R^2$: {r_squared:0.2f}  slope: {np.round(lin_fit[0], 2)}', fontsize=42)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/comparison {epoch}_{frame}.tif", dpi=80)
                        plt.close()
                        im_list.append(im_)
                        pred_list.append(pred)

                    im_list = np.array(np.array(im_list))
                    pred_list = np.array(np.array(pred_list))

                    im_list_ = np.reshape(im_list,(100,1024))
                    pred_list_ = np.reshape(pred_list,(100,1024))
                    im_list_ = np.rot90(im_list_)
                    pred_list_ = np.rot90(pred_list_)
                    im_list_ = scipy.ndimage.zoom(im_list_, (1024 / im_list_.shape[0], 1024 / im_list_.shape[1]))
                    pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))

                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)
                    plt.title('true field')
                    plt.imshow(im_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(1, 2, 2)
                    plt.title('reconstructed field')
                    plt.imshow(pred_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.tif", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(im_list, pred_list, s=1, c=mc, alpha=0.1)
                    plt.xlim([0.3, 1.6])
                    plt.ylim([0.3, 1.6])
                    plt.xlabel(r'true $\Omega_i$', fontsize=68)
                    plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all_comparison {epoch}.tif", dpi=80)
                    plt.close()

                    x_data = np.reshape(im_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                    y_data = np.reshape(pred_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    print(f'field R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')

            elif 'PDE_N6' in model_config.signal_model_name:

                modulation = torch.tensor(x_list[0], device=device)
                modulation = modulation[:, :, 8:9].squeeze()
                modulation = modulation.t()
                modulation = modulation.clone().detach()
                modulation = to_numpy(modulation)

                modulation = scipy.ndimage.zoom(modulation, (1024 / modulation.shape[0], 1024 / modulation.shape[1]))
                pred_list_ = to_numpy(model.b**2)
                pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))

                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.title('true field')
                plt.imshow(modulation, cmap='grey')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(1, 2, 2)
                plt.title('reconstructed field')
                plt.imshow(pred_list_, cmap='grey')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.tif", dpi=80)
                plt.close()

                for frame in trange(0, modulation.shape[1], modulation.shape[1] // 257):
                    im = modulation[:, frame]
                    im = np.reshape(im, (32, 32))
                    plt.figure(figsize=(8, 8))
                    plt.axis('off')
                    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/true_field_{frame}.tif", dpi=80)
                    plt.close()



            if False:
                print ('symbolic regression ...')

                def get_pyssr_function(model_pysrr, rr, func):

                    text_trap = StringIO()
                    sys.stdout = text_trap

                    model_pysrr.fit(to_numpy(rr[:, None]), to_numpy(func[:, None]))

                    sys.stdout = sys.__stdout__

                    return model_pysrr.sympy

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
                    temp_equation_file=False
                )

                match model_config.signal_model_name:

                    case 'PDE_N2':

                        func = torch.mean(psi_list, dim=0).squeeze()

                        symbolic = get_pyssr_function(model_pysrr, rr, func)

                        for n in range(0,7):
                            print(symbolic(n))
                            logger.info(symbolic(n))

                    case 'PDE_N4':

                        for k in range(n_particle_types):
                            print('  ')
                            print('  ')
                            print('  ')
                            print(f'psi{k} ................')
                            logger.info(f'psi{k} ................')

                            pos = np.argwhere(labels == k)
                            pos = pos.squeeze()

                            func = psi_list[pos]
                            func = torch.mean(psi_list[pos], dim=0)

                            symbolic = get_pyssr_function(model_pysrr, rr, func)

                            # for n in range(0, 5):
                            #     print(symbolic(n))
                            #     logger.info(symbolic(n))

                    case 'PDE_N5':

                        for k in range(4**2):

                            print('  ')
                            print('  ')
                            print('  ')
                            print(f'psi {k//4} {k%4}................')
                            logger.info(f'psi {k//4} {k%4} ................')

                            pos =np.arange(k*250,(k+1)*250)
                            func = psi_list[pos]
                            func = torch.mean(psi_list[pos], dim=0)

                            symbolic = get_pyssr_function(model_pysrr, rr, func)

                            # for n in range(0, 7):
                            #     print(symbolic(n))
                            #     logger.info(symbolic(n))

                for k in range(n_particle_types):
                    print('  ')
                    print('  ')
                    print('  ')
                    print(f'phi{k} ................')
                    logger.info(f'phi{k} ................')

                    pos = np.argwhere(labels == k)
                    pos = pos.squeeze()

                    func = phi_list[pos]
                    func = torch.mean(phi_list[pos], dim=0)

                    symbolic = get_pyssr_function(model_pysrr, rr, func)

                    # for n in range(4, 7):
                    #     print(symbolic(n))
                    #     logger.info(symbolic(n))


def plot_synaptic3(config, epoch_list, log_dir, logger, cc, style, device):

    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_particle_types = config.simulation.n_particle_types
    delta_t = config.simulation.delta_t
    p = config.simulation.params
    omega = model_config.omega
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    embedding_cluster = EmbeddingCluster(config)
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = True
    else:
        has_field = False

    x_list = []
    y_list = []
    for run in trange(1):
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
            x = to_numpy(torch.stack(x))
            y = to_numpy(torch.stack(y))
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    print('update variables ...')
    x = x_list[0][n_frames - 1]
    n_particles = x.shape[0]
    print(f'N neurons: {n_particles}')
    logger.info(f'N neurons: {n_particles}')
    config.simulation.n_particles = n_particles
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)

    activity = torch.tensor(x_list[0],device=device)
    activity = activity[:, :, 6:7].squeeze()
    distrib = to_numpy(activity.flatten())
    activity = activity.t()

    type = x_list[0][0][:, 5]
    type_stack = torch.tensor(type, dtype=torch.float32, device=device)
    type_stack = type_stack[:,None].repeat(n_frames,1)


    if has_field:
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                        hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device, first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        model_f.train()

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    if epoch_list[0] == 'all':

        files = glob.glob(f"{log_dir}/models/*.pt")
        files.sort(key=os.path.getmtime)

        model, bc_pos, bc_dpos = choose_training_model(config, device)

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=[], device=device)

        # plt.rcParams['text.usetex'] = False
        # plt.rc('font', family='sans-serif')
        # plt.rc('text', usetex=False)
        # matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id]) >0:
                flag = False
                file_id = file_id - 1
            file_id += 1

        files = files[file_id:]

        # file_id_list0 = np.arange(0, file_id, file_id // 90)
        # file_id_list1 = np.arange(file_id, len(files), (len(files) - file_id) // 40)
        # file_id_list = np.concatenate((file_id_list0, file_id_list1))

        file_id_list = np.arange(0, len(files), (len(files)/100)).astype(int)
        r_squared_list = []
        slope_list = []

        with torch.no_grad():
            for file_id_ in trange(0, 100):
                file_id = file_id_list[file_id_]

                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                if has_field:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                amax = torch.max(model.a, dim=0).values
                amin = torch.min(model.a, dim=0).values
                model_a = (model.a - amin) / (amax - amin)

                # fig, ax = fig_init()
                # for n in range(n_particle_types):
                #     c1 = cmap.color(n)
                #     c2 = cmap.color((n+1)%4)
                #     c_list = np.linspace(c1, c2, 100)
                #     for k in range(250*n,250*(n+1)):
                #         plt.scatter(to_numpy(model.a[k*100:(k+1)*100, 0:1]), to_numpy(model.a[k*100:(k+1)*100, 1:2]), s=10, color=c_list, alpha=0.1, edgecolors='none')
                # if 'latex' in style:
                #     plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                #     plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
                # else:
                #     plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                #     plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
                # plt.xlim([0.94, 1.08])
                # plt.ylim([0.9, 1.10])
                # # plt.xlim([0.7, 1.2])
                # # plt.ylim([0.7, 1.2])
                # plt.tight_layout()
                # plt.savefig(f"./{log_dir}/results/all/all_embedding_0_{epoch}.tif", dpi=80)
                # plt.close()

                fig, ax = fig_init()
                for k in range(n_particle_types):
                    # plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), s=1, color=mc, alpha=0.5, edgecolors='none')
                    plt.scatter(to_numpy(model.a[k*25000:(k+1)*25000, 0]), to_numpy(model.a[k*25000:(k+1)*25000, 1]), s=1, color=cmap.color(k),alpha=0.5, edgecolors='none')
                    # plt.scatter(to_numpy(model.a[k * 25000: k * 25000 + 100, 0]),
                    #             to_numpy(model.a[k * 25000: k * 25000 + 100, 1]), s=10, color=c_list, alpha=1)
                if 'latex' in style:
                    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
                else:
                    plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
                plt.xlim([0.94, 1.08])
                plt.ylim([0.9, 1.10])
                # plt.xlim([0.7, 1.2])
                # plt.ylim([0.7, 1.2])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/all_embedding_1_{epoch}.tif", dpi=80)
                plt.close()

                correction = torch.load(f'{log_dir}/correction.pt',map_location=device)
                second_correction = np.load(f'{log_dir}/second_correction.npy')

                i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                fig, ax = fig_init()
                ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1,vmax=0.1)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=48)
                plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
                plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
                plt.subplot(2, 2, 1)
                ax = sns.heatmap(to_numpy(A[0:20, 0:20])/second_correction, cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/W_{epoch}.tif", dpi=80)
                plt.close()

                rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
                func_list = []
                k_list = [0, 250, 500, 750]
                fig, ax = fig_init()
                plt.axis('off')
                for it, k in enumerate(k_list):
                    ax = plt.subplot(2, 2, it + 1)
                    c1 = cmap.color(it)
                    c2 = cmap.color((it + 1) % 4)
                    c_list = np.linspace(c1, c2, 100)
                    for n in range(k * 100, (k + 1) * 100):
                        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                        in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                        with torch.no_grad():
                            func = model.lin_phi(in_features.float())
                        func_list.append(func)
                        # plt.plot(to_numpy(rr), to_numpy(func), 2, color=c_list[n%100], alpha=0.25)
                        # linewidth=4, alpha=0.15-0.15*(n%100)/100)
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(it), alpha=0.25)
                    # true_func = true_model.func(rr, it, 'update')
                    # plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                    # true_func = true_model.func(rr, it + 1, 'update')
                    # plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                    # plt.xlabel(r'$x_i$', fontsize=24)
                    # plt.ylabel(r'Learned $\phi^*(a_i(t), x_i)$', fontsize=68)
                    if k==0:
                        plt.ylabel(r'Learned $MLP_0(a_i(t), x_i)$', fontsize=32)
                    plt.ylim([-8, 8])
                    plt.xlim([-5, 5])
                    plt.xticks([])
                    plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP0_{epoch}.tif", dpi=80)
                plt.close()

                fig, ax = fig_init()
                for n in range(0, n_particles):
                    in_features = rr[:, None]
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float()) * correction
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)
                plt.xlabel(r'$x_i$', fontsize=68)
                # plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
                plt.ylabel(r'learned $MLP_1(x_i)$', fontsize=68)
                plt.ylim([-1.1, 1.1])
                plt.xlim(config.plotting.xlim)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.tif", dpi=80)
                plt.close()

                adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
                adjacency_ = adjacency.t().clone().detach()
                adj_t = torch.abs(adjacency_) > 0
                edge_index = adj_t.nonzero().t().contiguous()

                i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                fig, ax = fig_init()
                gt_weight = to_numpy(adjacency)
                pred_weight = to_numpy(A)
                plt.scatter(gt_weight, pred_weight / 10 , s=0.1, c=mc, alpha=0.1)
                plt.xlabel(r'true $W_{ij}$', fontsize=68)
                plt.ylabel(r'learned $W_{ij}$', fontsize=68)
                if n_particles == 8000:
                    plt.xlim([-0.05, 0.05])
                    plt.ylim([-0.05, 0.05])
                else:
                    plt.xlim([-0.2, 0.2])
                    plt.ylim([-0.2, 0.2])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/comparison_{epoch}.tif", dpi=80)
                plt.close()

                x_data = np.reshape(gt_weight, (n_particles * n_particles))
                y_data = np.reshape(pred_weight, (n_particles * n_particles))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                r_squared_list.append(r_squared)
                slope_list.append(lin_fit[0])

                if has_field:

                    fig, ax = fig_init()
                    pred = model_f(time=file_id_ / len(file_id_list), enlarge=True) ** 2
                    # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = torch.reshape(pred, (640, 640))
                    pred = to_numpy(torch.sqrt(pred))
                    pred = np.flipud(pred)
                    pred = np.rot90(pred, 1)
                    pred = np.fliplr(pred)
                    plt.imshow(pred, cmap='grey')
                    plt.ylabel(r'learned $MLP_2(x_i, t)$', fontsize=68)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/field_{epoch}.tif", dpi=80)
                    plt.close()

        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(r_squared_list, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('$R^2$', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/R2.png', dpi=300)
        plt.close()
        np.save(f'./{log_dir}/results/R2.npy', r_squared_list)

        slope_list = np.array(slope_list) / p[0][0]
        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(slope_list, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('slope', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/slope.png', dpi=300)
        plt.close()

    elif epoch_list[0] == 'time':

        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch = filename
        net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt'
        model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print(f'net: {net}')

        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for n in trange(100):

            indices = np.arange(n_particles)*100+n

            fig, ax = fig_init()
            plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), s=1, color=mc, alpha=0.01)
            for k in range(n_particle_types):
                plt.scatter(to_numpy(model.a[indices[k * 250:(k + 1) * 250], 0]),
                            to_numpy(model.a[indices[k * 250:(k + 1) * 250], 1]), s=100, color=cmap.color(k), alpha=0.5,
                            edgecolors='none')
            if 'latex' in style:
                plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
            else:
                plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
            plt.xlim([0.92, 1.08])
            plt.ylim([0.9, 1.10])
            plt.text(0.93, 1.08, f'time: {n}', fontsize=48)

            # plt.xlim([0.7, 1.2])
            # plt.ylim([0.7, 1.2])
            # plt.text(0.72, 1.16, f'time: {n}', fontsize=48)

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/all2/all_embedding_1_{n}.tif", dpi=80)
            plt.close()


            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            fig, ax = fig_init()
            plt.axis('off')
            ax = plt.subplot(2, 2, 1)
            plt.ylabel(r'learned $MLP_0(a_i(t), x_i)$', fontsize=38)
            for it, k in enumerate(indices):
                if (it%250 == 0) and (it>0):
                    ax = plt.subplot(2, 2, it//250+1)
                plt.xticks([])
                plt.yticks([])
                embedding_ = model.a[k, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                    plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(it//250), alpha=0.5)
                plt.ylim([-8,8])
                plt.xlim([-5,5])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/all2/phi_{n}.tif", dpi=80)
            plt.close()



            fig, ax = fig_init()


            # rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            # k_list = [0, 250, 500, 750]
            # fig, ax = fig_init()
            # plt.axis('off')
            # for it, k in enumerate(k_list):
            #     ax = plt.subplot(2, 2, it + 1)
            #     for n in range(k * 100, (k + 25) * 100):
            #         embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            #         in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
            #         with torch.no_grad():
            #             func = model.lin_phi(in_features.float())
            #         plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, alpha=0.025)
            #     plt.xlabel(r'$x_i$', fontsize=16)
            #     # plt.ylabel(r'Learned $\phi^*(a_i(t), x_i)$', fontsize=68)
            #     plt.ylabel(r'Learned $MLP_0(a_i(t), x_i)$', fontsize=16)
            #
            #
            # for k in range(n_particle_types):
            #     ax = plt.subplot(2, 2, k + 1)
            #     n = indices[k * 250:(k + 1) * 250]
            #     plt.scatter(to_numpy(model.a[indices[k * 250:(k + 1) * 250], 0]),
            #                 to_numpy(model.a[indices[k * 250:(k + 1) * 250], 1]), s=10, color=cmap.color(k),
            #                 alpha=0.5,
            #                 edgecolors='none')
            #
            #     plt.ylim([-8, 8])
            #     plt.xlim([-5, 5])
            #     plt.tight_layout()
            #
            # plt.savefig(f"./{log_dir}/results/all/MLP0_{epoch}.tif", dpi=80)
            # plt.close()

    else:

        fig_init(formatx='%.0f', formaty='%.0f')
        plt.hist(distrib, bins=100, color=mc, alpha=0.5)
        plt.ylabel('counts', fontsize=64)
        plt.xlabel('$x_{ij}$', fontsize=64)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/signal_distribution.png', dpi=300)
        plt.close()
        print(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')
        logger.info(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')

        # plt.figure(figsize=(15, 10))
        # ax = sns.heatmap(to_numpy(activity), center=0, cmap='viridis', cbar_kws={'fraction': 0.046})
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=32)
        # ax.invert_yaxis()
        # plt.ylabel('neurons', fontsize=64)
        # plt.xlabel('time', fontsize=64)
        # plt.xticks([1000, 9900], [1000, 10000], fontsize=48)
        # plt.yticks([0, 999], [1, 1000], fontsize=48)
        # plt.xticks(rotation=0)
        # plt.tight_layout()
        # plt.savefig(f'./{log_dir}/results/kinograph.png', dpi=300)
        # plt.close()
        #
        # plt.figure(figsize=(15, 10))
        # n = np.random.permutation(n_particles)
        # for i in range(25):
        #     plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        # plt.xlabel('time', fontsize=64)
        # plt.ylabel('$x_{i}$', fontsize=64)
        # plt.xticks([0, 10000], fontsize=48)
        # plt.yticks(fontsize=48)
        # plt.tight_layout()
        # plt.savefig(f'./{log_dir}/results/firing rate.png', dpi=300)
        # plt.close()

        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        adjacency_ = adjacency.t().clone().detach()
        adj_t = torch.abs(adjacency_) > 0
        edge_index = adj_t.nonzero().t().contiguous()
        weights = to_numpy(adjacency.flatten())
        pos = np.argwhere(weights != 0)
        weights = weights[pos]

        fig_init()
        plt.hist(weights, bins=1000, color=mc, alpha=0.5)
        plt.ylabel(r'counts', fontsize=64)
        plt.xlabel(r'$W$', fontsize=64)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.xlim([-0.1, 0.1])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/weights_distribution.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt'
            model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edge_index
            print(f'net: {net}')

            if has_field:

                im = imread(f"graphs_data/{simulation_config.node_value_map}")

                net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])

                os.makedirs(f"./{log_dir}/results/field", exist_ok=True)
                x = x_list[0][0]

                slope_list = list([])
                im_list=list([])
                pred_list=list([])

                for frame in trange(0, n_frames, n_frames//100):

                    fig, ax = fig_init()
                    im_ = np.zeros((44,44))
                    if (frame>=0) & (frame<n_frames):
                        im_ =  im[int(frame / n_frames * 256)].squeeze()
                    plt.imshow(im_,cmap='gray',vmin=0,vmax=2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/true_field{epoch}_{frame}.tif", dpi=80)
                    plt.close()


                    pred = model_f(time=frame / n_frames, enlarge=True) ** 2
                    # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = torch.reshape(pred, (640, 640))
                    pred = to_numpy(pred)
                    pred = np.flipud(pred)
                    pred = np.rot90(pred, 1)
                    pred = np.fliplr(pred)
                    fig, ax = fig_init()
                    plt.imshow(pred,cmap='gray',vmin=0,vmax=2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/reconstructed_field_HR {epoch}_{frame}.tif", dpi=80)
                    plt.close()

                    pred = model_f(time=frame / n_frames, enlarge=False) ** 2
                    pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = to_numpy(pred)
                    pred = np.flipud(pred)
                    pred = np.rot90(pred, 1)
                    pred = np.fliplr(pred)
                    fig, ax = fig_init()
                    plt.imshow(pred,cmap='gray',vmin=0,vmax=2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/reconstructed_field_LR {epoch}_{frame}.tif", dpi=80)
                    plt.close()

                    x_data = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    y_data = np.reshape(pred, (n_nodes_per_axis * n_nodes_per_axis))
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
                    slope_list.append(lin_fit[0])

                    fig, ax = fig_init()
                    plt.scatter(im_,pred, s=10, c=mc)
                    plt.xlim([0.3,1.6])
                    plt.ylim([0.3,1.6])
                    plt.xlabel(r'true $\Omega_i$', fontsize=68)
                    plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                    plt.text(0.5, 1.4, f'$R^2$: {r_squared:0.2f}  slope: {np.round(lin_fit[0], 2)}', fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/comparison {epoch}_{frame}.tif", dpi=80)
                    plt.close()
                    im_list.append(im_)
                    pred_list.append(pred)

                im_list = np.array(np.array(im_list))
                pred_list = np.array(np.array(pred_list))

                fig, ax = fig_init()
                plt.scatter(im_list, pred_list, s=1, c=mc, alpha=0.1)
                plt.xlim([0.3, 1.6])
                plt.ylim([0.3, 1.6])
                plt.xlabel(r'true $\Omega_i$', fontsize=68)
                plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/field/all_comparison {epoch}.tif", dpi=80)
                plt.close()

                x_data = np.reshape(im_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                y_data = np.reshape(pred_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')

            if model_config.embedding_dim == 4:
                for k in range(n_particle_types):
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), to_numpy(model.a[:, 2]), s=1, color=mc, alpha=0.01, edgecolors='none')
                    ax.scatter(to_numpy(model.a[k * 25000:(k + 1) * 25000, 0]),
                            to_numpy(model.a[k * 25000:(k + 1) * 25000, 1]), to_numpy(model.a[k * 25000:(k + 1) * 25000, 2]), s=0.1, color=cmap.color(k), alpha=0.5)
                    # ax.scatter(to_numpy(model.a[k*25000:k*25000+100, 0]), to_numpy(model.a[k*25000:k*25000+100, 1]), to_numpy(model.a[k*25000:k*25000+100, 1]), color=mc)
                    plt.ylim([0, 2])
                    plt.xlim([0, 2])
                    ax.set_zlim([-2, 3.5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/embedding_{k}_{epoch}.tif", dpi=80)
                    plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                for k in range(n_particle_types):
                    # ax.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), to_numpy(model.a[:, 2]), s=1, color=mc, alpha=0.01, edgecolors='none')
                    ax.scatter(to_numpy(model.a[k * 25000:(k + 1) * 25000, 0]),
                            to_numpy(model.a[k * 25000:(k + 1) * 25000, 1]), to_numpy(model.a[k * 25000:(k + 1) * 25000, 2]), s=10, color=cmap.color(k), alpha=0.1, edgecolors='none')
                    # ax.scatter(to_numpy(model.a[k*25000:k*25000+100, 0]), to_numpy(model.a[k*25000:k*25000+100, 1]), to_numpy(model.a[k*25000:k*25000+100, 1]), color=mc)
                plt.ylim([0.5, 2])
                plt.xlim([0, 1.5])
                ax.set_zlim([-2, 2.5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all_embedding_{epoch}.tif", dpi=80)
                plt.close()

            else:

                fig, ax = fig_init()
                for n in range(n_particle_types):
                    c1 = cmap.color(n)
                    c2 = cmap.color((n+1)%4)
                    c_list = np.linspace(c1, c2, 100)
                    for k in range(250*n,250*(n+1)):
                        plt.scatter(to_numpy(model.a[k*100:(k+1)*100, 0:1]), to_numpy(model.a[k*100:(k+1)*100, 1:2]), s=10, color=c_list, alpha=0.1, edgecolors='none')
                if 'latex' in style:
                    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
                else:
                    plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all_embedding_0_{epoch}.tif", dpi=80)
                plt.close()

                fig, ax = fig_init()
                for k in range(n_particle_types):
                    # plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), s=1, color=mc, alpha=0.5, edgecolors='none')
                    plt.scatter(to_numpy(model.a[k*25000:(k+1)*25000, 0]), to_numpy(model.a[k*25000:(k+1)*25000, 1]), s=1, color=cmap.color(k),alpha=0.5, edgecolors='none')
                    # plt.scatter(to_numpy(model.a[k * 25000: k * 25000 + 100, 0]),
                    #             to_numpy(model.a[k * 25000: k * 25000 + 100, 1]), s=10, color=c_list, alpha=1)
                if 'latex' in style:
                    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
                else:
                    plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all_embedding_1_{epoch}.tif", dpi=80)
                plt.close()

                for k in range(n_particle_types):
                    fig, ax = fig_init()
                    # plt.scatter(to_numpy(model.a[0:100000, 0]), to_numpy(model.a[0:100000, 1]), s=1, color=mc, alpha=0.25, edgecolors='none')
                    plt.scatter(to_numpy(model.a[k*25000:(k+1)*25000, 0]), to_numpy(model.a[k*25000:(k+1)*25000, 1]), s=1, color=cmap.color(k),alpha=0.5, edgecolors='none')
                    if 'latex' in style:
                        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
                        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
                    else:
                        plt.xlabel(r'$a_{i0}$', fontsize=68)
                        plt.ylabel(r'$a_{i1}$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/embedding_{k}_{epoch}.tif", dpi=80)
                    plt.close()


            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            k_list=[0,250,500,750]
            for it, k in enumerate(k_list):
                c1 = cmap.color(it)
                c2 = cmap.color((it + 1) % 4)
                c_list = np.linspace(c1, c2, 100)
                fig, ax = fig_init()
                for n in trange(k*100,(k+1)*100):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func_list.append(func)
                    # plt.plot(to_numpy(rr), to_numpy(func), 2, color=c_list[n%100], alpha=0.25)
                             # linewidth=4, alpha=0.15-0.15*(n%100)/100)
                    plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(it), alpha=0.25)
                true_func = true_model.func(rr, it, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                true_func = true_model.func(rr, it+1, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                plt.xlabel(r'$x_i$', fontsize=68)
                plt.ylabel(r'Learned $\phi^*(a_i(t), x_i)$', fontsize=68)
                plt.ylim([-8,8])
                plt.xlim([-5,5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/phi_{k}.tif", dpi=170.7)
                plt.close()

            fig, ax = fig_init()
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            for n in trange(0,n_particles,n_particles):
                if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                else:
                    in_features = rr[:, None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
                    if n<250:
                        func_list.append(func)
                else:
                    func_list.append(func)
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=8 // ( 1 + (n_particle_types>16)*1.0), alpha=0.25)
            func_list = torch.stack(func_list)
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'Learned $\psi^*(a_i, x_i)$', fontsize=68)
            plt.xlim([-5,5])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/raw_psi.tif", dpi=170.7)
            plt.close()

            correction = 1 / torch.mean(torch.mean(func_list[:,900:1000], dim=0))
            print(f'correction: {correction:0.2f}')
            torch.save(correction, f'{log_dir}/correction.pt')

            psi_list = []
            fig, ax = fig_init()
            rr = torch.tensor(np.linspace(-7.5, 7.5, 1500)).to(device)
            if model_config.signal_model_name == 'PDE_N4':
                for n in range(n_particle_types):
                    true_func = true_model.func(rr, n, 'phi')
                    plt.plot(to_numpy(rr), to_numpy(true_func), c = 'k', linewidth = 16, label = 'original', alpha = 0.21)
            else:
                true_func = true_model.func(rr, 0, 'phi')
                plt.plot(to_numpy(rr), to_numpy(true_func), c = 'k', linewidth = 16, label = 'original', alpha = 0.21)

            for n in trange(0,n_particles):
                in_features = rr[:, None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float()) * correction
                    psi_list.append(func)
                    plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
            plt.ylim([-1.1, 1.1])
            plt.xlim(config.plotting.xlim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_psi.tif", dpi=170.7)
            plt.close()


            i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
            A = model.W.clone().detach() / correction
            A[i, i] = 0

            fig, ax = fig_init()
            gt_weight = to_numpy(adjacency)
            pred_weight = to_numpy(A)
            plt.scatter(gt_weight, pred_weight / 10 , s=0.1, c=mc, alpha=0.1)
            plt.xlabel(r'true $W_{ij}$', fontsize=68)
            plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            if n_particles == 8000:
                plt.xlim([-0.05,0.05])
                plt.ylim([-0.05,0.05])
            else:
                plt.xlim([-0.2,0.2])
                plt.ylim([-0.2,0.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/comparison_{epoch}.tif", dpi=87)
            plt.close()

            x_data = np.reshape(gt_weight, (n_particles * n_particles))
            y_data =  np.reshape(pred_weight, (n_particles * n_particles))
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
            logger.info(f'R^2$: {np.round(r_squared, 4)}  slope: {np.round(lin_fit[0], 4)}')

            second_correction = lin_fit[0]
            print(f'second_correction: {second_correction:0.2f}')
            np.save(f'{log_dir}/second_correction.npy', second_correction)

            plt.figure(figsize=(10, 10))
            # plt.title(r'learned $W_{ij}$', fontsize=68)
            ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1,vmax=0.1)
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=32)
            plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
            plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
            plt.xticks(rotation=0)
            plt.subplot(2, 2, 1)
            ax = sns.heatmap(to_numpy(A[0:20, 0:20]/second_correction), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
            plt.xticks(rotation=0)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/learned connectivity.png', dpi=300)
            plt.close()


def plot_agents(config, epoch_list, log_dir, logger, style, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames

    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    has_state = (config.simulation.state_type != 'discrete')

    l_dir = get_log_dir(config)
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file.split('/')[-1]))

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"{log_dir}/models/best_model_with_1_graphs_3.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    print('Load data ...')
    time_series, signal = load_agent_data(dataset_name, device=device)

    velocities = [t.velocity for t in time_series]
    velocities.pop(0)  # the first element is always NaN
    velocities = torch.stack(velocities)
    if torch.any(torch.isnan(velocities)):
        raise ValueError('Discovered NaN in velocities. Aborting.')
    velocities = bc_dpos(velocities)

    positions = torch.stack([t.pos for t in time_series])
    min = torch.min(positions[:, :, 0])
    max = torch.max(positions[:, :, 0])
    mean = torch.mean(positions[:, :, 0])
    std = torch.std(positions[:, :, 0])
    print(f"min: {min}, max: {max}, mean: {mean}, std: {std}")

    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))

    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    n_particles = config.simulation.n_particles
    print(f'N particles: {n_particles}')
    logger.info(f'N particles:  {n_particles}')

    if os.path.exists(f'{log_dir}/edge_p_p_list.npz'):
        print('Load list of edges index ...')
        edge_p_p_list = np.load(f'{log_dir}/edge_p_p_list.npz')
    else:
        print('Create list of edges index ...')
        edge_p_p_list = []
        for k in trange(n_frames):
            time_point = time_series[k]
            x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
            x = torch.column_stack((torch.arange(0, n_particles, device=device), x))

            nbrs = NearestNeighbors(n_neighbors=simulation_config.n_neighbors, algorithm='auto').fit(to_numpy(x[:, 1:dimension + 1]))
            distances, indices = nbrs.kneighbors(to_numpy(x[:, 1:dimension + 1]))
            edge_index = []
            for i in range(indices.shape[0]):
                for j in range(1, indices.shape[1]):  # Start from 1 to avoid self-loop
                    edge_index.append((i, indices[i, j]))
            edge_index = np.array(edge_index)
            edge_index = torch.tensor(edge_index, device=device).t().contiguous()
            edge_p_p_list.append(to_numpy(edge_index))
        np.savez(f'{log_dir}/edge_p_p_list', *edge_p_p_list)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    for epoch in epoch_list:

        net = f"{log_dir}/models/best_model_with_0_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()


        for k in trange(2, n_frames-2,10):

            time_point = time_series[k]
            x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
            x = torch.column_stack((torch.arange(0, n_particles, device=device), x))
            x[:, 1:5] = x[:, 1:5] / 1000

            edges = edge_p_p_list[f'arr_{k}']
            edges = torch.tensor(edges, dtype=torch.int64, device=device)
            dataset = data.Data(x=x[:, :], edge_index=edges)

            if model_config.prediction == 'first_derivative':
                time_point = time_series[k + 1]
                y = bc_dpos(time_point.velocity.clone().detach() / 1000)
            else:
                time_point = time_series[k + 1]
                v_prev = bc_dpos(time_point.velocity.clone().detach() / 1000)
                time_point = time_series[k - 1]
                v_next = bc_dpos(time_point.velocity.clone().detach() / 1000)
                y = (v_next - v_prev)

            embedding = to_numpy(model.a[0][k].squeeze())

            ax, fig = fig_init()
            plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c = embedding,  s=0.1, alpha=1)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{k}.tif", dpi=87)
            plt.close()

            # in_features = torch.cat((torch.zeros((300000,7),device=device), embedding), dim=-1)
            # out = model.lin_edge(in_features.float())

        for x_ in [0,1]:
            for y_ in [0,1]:
                fig = plt.figure(figsize=(5, 5))
                plt.scatter(to_numpy(y[:, x_]), embedding[:, y_], s=0.1, c=mc, alpha=0.01)
                plt.xlabel(f'$x_{x_}$', fontsize=48)
                plt.ylabel(f'$y_{y_}$', fontsize=48)
                plt.tight_layout()

        x = model.a[0][k].squeeze()


        model_kan = KAN(width=[2, 5, 2])
        dataset={}

        dataset['train_input'] = x[0:10000]
        dataset['test_input'] = x[12000:13000]
        dataset['train_label'] = y[0:10000]
        dataset['test_label'] = y[12000:13000]

        model_kan.train(dataset, opt="LBFGS", steps=200, lamb=0.01, lamb_entropy=10.)

        model_kan.plot()

        lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
        model_kan.auto_symbolic(lib=lib)
        model_kan.train(dataset, steps=20)


        fig = plt.figure(figsize=(5, 5))
        plt.scatter(to_numpy(y[:, 1]), to_numpy(embedding[:, 0]), s=0.1, c=mc, alpha=0.01)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), to_numpy(y[:, 0]), s=0.1, c=mc, alpha=0.1)






        # if has_state:
        #     ax, fig = fig_init()
        #     embedding = torch.reshape(model.a[0], (n_particles * n_frames, 2))
        #     plt.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), s=0.1, alpha=0.01, c=mc)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.tight_layout()
        #     plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{epoch}_{N}.tif", dpi=87)
        # else:
        #     ax, fig = fig_init()
        #     embedding = model.a[0]
        #     plt.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), s=1, alpha=0.1, c=mc)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.tight_layout()
        #     plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{epoch}_{N}.tif", dpi=87)
        #

        # # plt.scatter(to_numpy(y[:, 1]), to_numpy(pred[:, 1]), s=0.1, alpha=0.1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/particle/Fig_{epoch}_{N}.tif", dpi=87)


def plot_mouse(config, epoch_list, log_dir, logger, style, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    time_step = simulation_config.time_step
    delta_t = simulation_config.delta_t * time_step
    data_folder_name = config.data_folder_name
    dataset_name = config.dataset
    time_step = simulation_config.time_step
    entropy_loss = KoLeoLoss()
    xlim = config.plotting.xlim
    ylim = config.plotting.ylim
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_runs = train_config.n_runs
    cmap = CustomColorMap(config=config)
    pic_folder = config.plotting.pic_folder
    pic_format = config.plotting.pic_format
    pic_size = config.plotting.pic_size

    if os.path.exists(pic_folder):
        files = glob.glob(f'{pic_folder}/*.jpg')
        sorted_files = sorted(files, key=extract_number)

    x_list = []
    x = torch.load(f'graphs_data/{dataset_name}/x_list_0.pt', map_location=device)
    x_list.append(x)
    edge_p_p_list=[]
    edge_p_p = np.load(f'graphs_data/{dataset_name}/edge_p_p_list_0.npz')
    edge_p_p_list.append(edge_p_p)
    n_frames = len(x)
    config.simulation.n_frames = n_frames

    vnorm = torch.ones(1, dtype=torch.float32, device=device)
    ynorm = torch.ones(1, dtype=torch.float32, device=device)
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    n_particles_max = 0
    id_list = []
    type_list = []
    for k in range(n_frames):
        type = x_list[0][k][:, 5]
        type_list.append(type)
        if k==0:
            type_stack = type
        else:
            type_stack = torch.cat((type_stack, type), 0)
        ids = x_list[0][k][:, -1]
        id_list.append(ids)
        n_particles_max += len(type)
    config.simulation.n_particles_max = n_particles_max

    print('load models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)


    if epoch_list[0] == 'all':

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_0_graphs_*.pt")
        files.sort(key=sort_key)
        entropy_list=[]

        for file_id in trange(0,len(files),2):
            # print(files[file_id], sort_key(files[file_id]), (sort_key(files[file_id]) % 1E7 != 0))

            if (sort_key(files[file_id]) % 1E7 != 0):
                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_0_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                plt.style.use('dark_background')

                if True:

                    # get permutation random of model.a

                    idx = torch.randperm(len(model.a))
                    model_a = model.a[idx[0:10000]].clone().detach()
                    entropy = entropy_loss(model_a)
                    entropy_list.append(entropy)

                    n_pts = min(40000, len(model.a))

                    fig, ax = fig_init(fontsize=24)
                    params = {'mathtext.default': 'regular'}
                    plt.rcParams.update(params)
                    embedding = to_numpy(model.a[idx[0:n_pts]])
                    plt.scatter(embedding[:,0], embedding[:, 1], c='w', s=0.5, alpha=1, edgecolor='None')
                    plt.xlabel(r'$a_{i0}$', fontsize=48)
                    plt.ylabel(r'$a_{i1}$', fontsize=48)
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
                    plt.close()

                    # student_output = F.normalize(model_a, eps=1E-8, p=2, dim=0)
                    # plt.scatter(to_numpy(student_output[:, 0]), to_numpy(student_output[:, 1]), c='w', s=1, alpha=1, edgecolor='None')
                    # I = entropy_loss.pairwise_NNs_inner(student_output)  # noqa: E741
                    # distances = entropy_loss.pdist(student_output, student_output[I])  # BxD, BxD -> B
                    # loss = -torch.log(distances + 1E-8).mean()

                    fig, ax = fig_init(fontsize=24)
                    rr = torch.tensor(np.linspace(0, 0.75, 1000)).to(device)
                    for n in range(0,len(model.a),len(model.a)//2000):
                        embedding_ = model.a[n] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                        in_features = torch.cat((rr[:, None] , 0 * rr[:, None],
                                                 rr[:, None] , embedding_), dim=1)
                        with torch.no_grad():
                            func = model.lin_edge(in_features.float())
                            func = func[:, 0]
                        plt.plot(to_numpy(rr),
                                 to_numpy(func) * to_numpy(ynorm),
                                 c='w', linewidth=1, alpha=0.15)
                    plt.xlabel('$d_{ij}$', fontsize=48)
                    plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                    if 'rat_city' in dataset_name:
                        plt.ylim([-0.2, 0.2])
                    else:
                        plt.ylim([-0.05, 0.05])
                    plt.tight_layout()

                    plt.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
                    plt.close()

                else:

                    embedding = to_numpy(model.a.clone().detach())

                    fig, ax = fig_init(fontsize=24)
                    for k in np.unique(labels):
                        pos = np.argwhere(labels == k)
                        plt.scatter(embedding[pos, 0], embedding[pos, 1], s=1, c=cmap.color(k), alpha=1, edgecolors='None')
                    plt.xlabel(r'$a_{i0}$', fontsize=48)
                    plt.ylabel(r'$a_{i1}$', fontsize=48)
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/clustered_embedding_{epoch}.tif", dpi=80)
                    plt.close()

                    fig, ax = fig_init(fontsize=24)
                    rr = torch.tensor(np.linspace(0, 0.75, 1000)).to(device)
                    for n in range(0, len(model.a), len(model.a) // 2000):
                        embedding_ = model.a[n] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                        in_features = torch.cat((rr[:, None], 0 * rr[:, None],
                                                 rr[:, None], embedding_), dim=1)
                        with torch.no_grad():
                            func = model.lin_edge(in_features.float())
                            func = func[:, 0]
                        plt.plot(to_numpy(rr),
                                 to_numpy(func) * to_numpy(ynorm),
                                 color=cmap.color(labels[n].astype(int)), linewidth=2, alpha=0.15)
                    plt.xlabel('$d_{ij}$', fontsize=48)
                    plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
                    plt.ylim([-0.025, 0.025])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/clustered_functions_{epoch}.tif", dpi=80)
                    plt.close()

        entropy_list = torch.stack(entropy_list)
        entropy_list = to_numpy(entropy_list)
        fig = plt.figure(figsize=(10, 10))
        plt.plot(entropy_list, linewidth=1, color='w')
        plt.xlabel('iteration', fontsize=48)
        plt.ylabel('entropy', fontsize=48)
        plt.ylim([0,7])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/entropy.tif", dpi=80)


    else:

        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch_list[0]}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        time.sleep(1)

        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)
        matplotlib.rcParams['savefig.pad_inches'] = 0
        plt.style.use('dark_background')

        # print('clustering ...')
        embedding = to_numpy(model.a.clone().detach())
        map_behavior = np.zeros((100,6000))

        # Define the colors: black for zero, and two other colors
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Black, Red, Blue

        # Create the colormap
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'custom_cmap'
        cbm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        next_id = max(x_list[0][0][:, -1]) + 1
        for k in trange(1000, 6000): #n_frames-1):
            x = x_list[0][k]
            edges = edge_p_p_list[0][f'arr_{k}']
            edges = torch.tensor(edges, dtype=torch.int64, device=device)
            dataset = data.Data(x=x[:, :], edge_index=edges)

            pred = model(dataset, data_id=0, training=True, phi=torch.zeros(1, device=device), has_field=False)

            x_next = x_list[0][k + time_step]
            x_pos_next = x_next[:,1:3].clone().detach()
            if model_config.prediction == '2nd_derivative':
                x_pos_pred = (x[:, 1:3] + delta_t * (x[:, 3:5] + delta_t * pred * ynorm))
            else:
                x_pos_pred = (x[:,1:3] + delta_t * pred * ynorm)

            V = (x_pos_pred - x[:, 1:3])

            distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
            result = distance.min(dim=1)
            min_value = result.values
            indices = result.indices
            loss = torch.sum(min_value)*1E5

            if 'rat_city' in dataset_name:

                if k==0:
                    x_prev = x
                    x[:,-3] = x[:,-1]
                else:
                    x_prev = x_list[0][k-1].clone().detach()
                    for m in range(len(x)):
                        pos = torch.argwhere(x_prev[:,-1]==x[m,-2])
                        if pos.numel():
                            x[m,-3] = x_prev[pos[0],-3]
                        else:
                            x[m, -3] = next_id
                            next_id += 1
                    t = to_numpy(x_list[0][k])
                    t_prev = to_numpy(x_list[0][k-1])
                    t_prev_prev = to_numpy(x_list[0][k - 2])

                # matplotlib.use("Qt5Agg")
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(2, 2, 1)
                if os.path.exists(pic_folder):
                    im = imageio.imread(sorted_files[k])
                    im = np.flipud(im)
                    plt.imshow(im)
                    plt.axis('off')
                pos = x[:, 1:3].clone().detach()
                pos[:, 0] = 1110 * pos[:, 0]
                pos[:, 1] = 1000 * pos[:, 1]
                # ax.axvline(x=1.05, ymin=0, ymax=0.7, color='r', linestyle='--', linewidth=2)
                plt.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), s=100, c='b')
                plt.scatter(to_numpy(x_next[:, 1])*1110, to_numpy(x_next[:, 2])*1000, s=100, c='g', alpha=0.5)
                for n in range(len(x)):
                    plt.arrow(x=to_numpy(pos[n, 0]), y=to_numpy(pos[n, 1]), dx=to_numpy(V[n, 0])*1110, dy=to_numpy(V[n, 1])*1000,
                              head_width=0.01, length_includes_head=True)
                plt.xlim([0, 2300])
                plt.ylim([0, 1000])
                # plt.title('GNN tracking')
                plt.xticks([])
                plt.yticks([])

                ax = fig.add_subplot(2, 2, 3)
                if os.path.exists(pic_folder):
                    im = imageio.imread(sorted_files[k])
                    im = np.flipud(im)
                    plt.imshow(im)
                    plt.axis('off')
                # ax.axvline(x=1.05, ymin=0, ymax=0.7, color='r', linestyle='--', linewidth=2)
                particle_id = to_numpy(x[:, -3])
                dataset = data.Data(x=x, pos=pos, edge_index=edges)
                vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                nx.draw_networkx(vis, pos=to_numpy(pos), node_size=0, linewidths=0, with_labels=False, ax=ax, edge_color='g', width=1, alpha=0.4)
                for n in range(edges.shape[1]):
                    plt.arrow(x=to_numpy(model.pos[n, 0])*1110, y=to_numpy(model.pos[n, 1])*1000, dx=to_numpy(model.msg[n, 0])*1110, dy=to_numpy(model.msg[n, 1])*1000, head_width=0.01, length_includes_head=True, alpha=0.2)
                for n in range(len(x)):
                    plt.arrow(x=to_numpy(pos[n, 0]), y=to_numpy(pos[n, 1]), dx=to_numpy(V[n, 0])*1110, dy=to_numpy(V[n, 1])*1000, head_width=0.01, length_includes_head=True)
                for n, id in enumerate(particle_id.astype(int)):
                    plt.text(to_numpy(x[n, 1])*1110+25, to_numpy(x[n, 2])*1000-10, f'{id}', fontsize=8, color='w')
                plt.xlim([0, 2300])
                plt.ylim([0, 1000])
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()

                detection_id = to_numpy(x[:,-1]).astype(int)
                ax = fig.add_subplot(2, 4, 3)
                # plt.axis('off')
                # plt.title('embedding : rat state')
                plt.scatter(embedding[:, 0], embedding[:, 1], s=1, c='g', alpha=1, edgecolors='None')
                for n, id in enumerate(detection_id):
                    # plt.scatter(embedding[id, 0], embedding[id, 1], s=30, alpha=1, color='w', edgecolors='None')
                    plt.text(embedding[id, 0], embedding[id, 1], f'{particle_id[n].astype(int)}', fontsize=8, color='w')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0,2])
                plt.ylim([0,2])

                ax = fig.add_subplot(2, 4, 7)
                # plt.axis('off')
                # plt.title('MLP plot : rat small action')
                rr = torch.tensor(np.linspace(0, 0.6, 1000)).to(device)
                for n, id in enumerate(detection_id):
                    embedding_ = model.a[id] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None], 0 * rr[:, None],
                                             rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                    map_behavior[particle_id[n].astype(int)-1,k] = to_numpy(func[500])
                    plt.plot(to_numpy(rr),to_numpy(func) * to_numpy(ynorm), linewidth=1, alpha=1, color='w')
                    plt.text(to_numpy(rr[200 + 50*n]), to_numpy(func[200 + 50*n]) * to_numpy(ynorm) + 0.0035, f'{particle_id[n].astype(int)}', fontsize=8, color='w')
                if 'rat_city' in dataset_name:
                    plt.ylim([-0.2, 0.2])
                plt.xticks([])
                plt.yticks([])

                ax = fig.add_subplot(2, 4, 8)
                # plt.axis('off')
                plt.imshow(map_behavior[0:to_numpy(next_id-1).astype(int), :], aspect='auto', cmap=cbm, vmin=-0.1, vmax=0.1)
                plt.xticks(fontsize=8)
                plt.yticks(np.arange(0, to_numpy(next_id-1).astype(int)), np.arange(1, to_numpy(next_id).astype(int)), fontsize=8)

                # if (N+1)%40 == 0:
                #     ax = fig.add_subplot(2, 3, 4)
                #     xp = x[0:4, :]
                #     xp[:, 1:3] = torch.randn_like(xp[:, 1:3]) * 0.01
                #     xp[0, 1:3] = 0 * xp[0, 1:3]
                #     pos_i = xp[0, 1:3]
                #     pos_j = xp[1:, 1:3]
                #     r = torch.sqrt(torch.sum(bc_dpos(pos_j - pos_i) ** 2, dim=1)) / max_radius
                #     delta_pos = bc_dpos(pos_j - pos_i) / max_radius
                #     embedding_size = embedding.shape[0]
                #     point_list = []
                #     for n in range(200000):
                #         particle_id = torch.randint(0, embedding_size, (1,),device=device).repeat(3,1)
                #         embedding_i = model.a[to_numpy(particle_id), :].squeeze()
                #         in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
                #         out = torch.mean(model.lin_edge(in_features.float()), dim=0)
                #         point_list.append(out)
                #     point_list = torch.stack(point_list)
                #     plt.scatter(to_numpy(point_list[:, 0]), to_numpy(point_list[:, 1]), s=10, c='w', alpha=0.1, edgecolors='None')
                #     plt.scatter(to_numpy(xp[:, 1]), to_numpy(xp[:, 2]), s=100, c='g', alpha=1)
                #     plt.ylim([-0.02, 0.022])
                #     plt.xlim([-0.02, 0.022])
                # plt.show()

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{k}.tif", dpi=120)
                plt.close()

            else:

                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 2, 1)
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=100, c='b')
                for n in range(len(x)):
                    plt.arrow(x=to_numpy(x[n, 1]), y=to_numpy(x[n, 2]), dx=to_numpy(V[n, 0]), dy=to_numpy(V[n, 1]), head_width=0.01, length_includes_head=True)
                plt.scatter(to_numpy(x_next[:, 1]), to_numpy(x_next[:, 2]), s=100, c='g', alpha=0.5)

                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.title('GNN tracking')
                plt.text(0.05, 0.95, f'frame = {N*time_step}', fontsize=12)
                plt.xticks([])
                plt.yticks([])
                # ax = fig.add_subplot(2, 2, 3)
                # plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=100, color=cmap.color(to_numpy(type_list[k]).astype(int)))
                # plt.xlim([0,1])
                # plt.ylim([0,1])
                # plt.title('Yolo tracking')
                # plt.xticks([])
                # plt.yticks([])
                ax = fig.add_subplot(1, 2, 2)
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=100, color=cmap.color(to_numpy(x[:, 6]).astype(int)))
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.title('GNN cluster')
                plt.xticks([])
                plt.yticks([])

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{N}.tif", dpi=100)
                plt.close()

        if 'rat_city' in dataset_name:
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(map_behavior[0:to_numpy(next_id).astype(int), 0:1000],  aspect='auto', cmap='bwr', vmin=-0.2, vmax=0.2)
            plt.yticks([0,to_numpy(next_id).astype(int)],[0,to_numpy(next_id).astype(int)])
            plt.savefig(f"./{log_dir}/behavior.tif", dpi=100)
            plt.close
            np.save(f"./{log_dir}/behavior.npy",map_behavior)


def data_plot(config, epoch_list, style, device):

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    if 'black' in style:
        plt.style.use('dark_background')
        mc ='w'
    else:
        plt.style.use('default')
        mc = 'k'

    if 'latex' in style:
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    # plt.rc('font', family='sans-serif')
    # plt.rc('font', family='serif')
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rc('text', usetex=False)
    matplotlib.rcParams['savefig.pad_inches'] = 0

    l_dir = get_log_dir(config)

    log_dir, logger = create_log_dir(config=config, erase=False)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results/all'), exist_ok=True)
    # files = glob.glob(f"{log_dir}/results/all/*")
    # for f in files:
    #     os.remove(f)
    os.makedirs(f"./{log_dir}/results/field", exist_ok=True)
    files = glob.glob(f"{log_dir}/results/field/*")
    for f in files:
        os.remove(f)

    if epoch_list==['best']:
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch_list=[filename]
        print(f'best model: {epoch_list}')
        logger.info(f'best model: {epoch_list}')


    if config.training.sparsity != 'none':
        print(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), with cluster method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), with cluster method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
    else:
        print(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), no clustering')
        logger.info(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), no clustering')

    if os.path.exists(f'{log_dir}/loss.pt'):
        loss = torch.load(f'{log_dir}/loss.pt')
        fig, ax = fig_init(formatx='%.0f', formaty='%.5f')
        plt.plot(loss, color=mc, linewidth=4)
        plt.xlim([0, 20])
        plt.ylabel('loss', fontsize=68)
        plt.xlabel('epochs', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/loss.tif", dpi=170.7)
        plt.close()
        # print('final loss {:.3e}'.format(loss[-1]))
        # logger.info('final loss {:.3e}'.format(loss[-1]))

    match config.graph_model.particle_model_name:
        case 'PDE_Agents_A' | 'PDE_Agents_B':
            plot_agents(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_Cell_A' | 'PDE_Cell_B' | 'PDE_Cell' | 'PDE_Cell_area':
            if config.training.do_tracking:
                plot_cell_tracking(config, epoch_list, log_dir, logger, style, device)
            else:
                plot_cell_state(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_A':
            if config.simulation.non_discrete_level>0:
                plot_attraction_repulsion_continuous(config, epoch_list, log_dir, logger, style, device)
            elif config.training.do_tracking:
                plot_attraction_repulsion_tracking(config, epoch_list, log_dir, logger, style, device)
            else:
                plot_attraction_repulsion(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_A_bis':
            plot_attraction_repulsion_asym(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_B' | 'PDE_Cell_B':
            plot_boids(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_ParticleField_B' | 'PDE_ParticleField_A':
            plot_particle_field(config, epoch_list, log_dir, logger, 'grey', style, device)
        case 'PDE_E':
            plot_Coulomb(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_F':
            plot_falling_particles(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_G':
            if config_file == 'gravity_continuous':
                plot_gravity_continuous(config, epoch_list, log_dir, logger, style, device)
            else:
                plot_gravity(config, epoch_list, log_dir, logger, style, device)
        case 'PDE_M':
                plot_mouse(config, epoch_list, log_dir, logger, style, device)

    match config.graph_model.mesh_model_name:
        case 'WaveMesh':
            plot_wave(config=config, epoch_list=epoch_list, log_dir=log_dir, logger=logger, cc='viridis', style=style, device=device)
        case 'RD_RPS_Mesh':
            plot_RD_RPS(config=config, epoch_list=epoch_list, log_dir=log_dir, logger=logger, cc='viridis',  style=style, device=device)

    if ('PDE_N' in config.graph_model.signal_model_name):
        if ('PDE_N3' in config.graph_model.signal_model_name):
            plot_synaptic3(config, epoch_list, log_dir, logger, 'viridis', style, device)
        else:
            plot_synaptic2(config, epoch_list, log_dir, logger, 'viridis', style, device)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_figures(index):

    epoch_list = ['20']
    match index:
        case '3':
            config_list = ['arbitrary_3', 'arbitrary_3_continuous', 'arbitrary_3_3', 'arbitrary_16', 'arbitrary_32','arbitrary_64']
        case '4':
            config_list = ['arbitrary_3_field_video_bison_quad']
        case '4_bis':
            config_list = ['arbitrary_3_field_video_bison']
        case 'supp1':
            config_list = ['arbitrary_3']
            epoch_list= ['0_0', '0_200', '0_1000', '20']
        case 'supp4':
            config_list = ['arbitrary_16', 'arbitrary_16_noise_0_3', 'arbitrary_16_noise_0_4', 'arbitrary_16_noise_0_5']
        case 'supp5':
            config_list = ['arbitrary_3_dropout_30', 'arbitrary_3_dropout_10', 'arbitrary_3_dropout_10_no_ghost']
        case 'supp6':
            config_list = ['arbitrary_3_field_boats']
        case 'supp7':
            config_list = ['gravity_16']
            epoch_list= ['20', '0_0', '0_5000', '1_0']
        case 'supp8':
            config_list = ['gravity_16', 'gravity_continuous', 'Coulomb_3_256']
        case 'supp9':
            config_list = ['gravity_16_noise_0_4', 'Coulomb_3_noise_0_4', 'Coulomb_3_noise_0_3', 'gravity_16_noise_0_3']
        case 'supp10':
            config_list = ['gravity_16_dropout_10', 'gravity_16_dropout_30', 'Coulomb_3_dropout_10_no_ghost', 'Coulomb_3_dropout_10']
        case 'supp11':
            config_list = ['boids_16_256']
            epoch_list = ['0_0', '0_2000', '0_10000', '20']
        case 'supp12':
            config_list = ['boids_16_256', 'boids_32_256', 'boids_64_256']
        case 'supp14':
            config_list = ['boids_16_noise_0_3', 'boids_16_noise_0_4', 'boids_16_dropout_10', 'boids_16_dropout_10_no_ghost']
        case 'supp15':
            config_list = ['wave_slit_ter']
            epoch_list = ['20', '0_1600', '1', '5']
        case 'supp16':
            config_list = ['wave_boat_ter']
            epoch_list = ['20', '0_1600', '1', '5']
        case 'supp17':
            config_list = ['RD_RPS']
        case 'supp18':
            config_list = ['signal_N_100_2_a']

        case 'synaptic_2_fig2':
            config_list = ['signal_N2_a10']
            epoch_list = ['best']
        case 'synaptic_supp1':
            config_list = [f'signal_N2_a{i}' for i in range(0, 11)]
            epoch_list = ['all']
        case 'synaptic_supp6':
            config_list = ['signal_N2_a_SNR1', 'signal_N2_a_SNR3', 'signal_N2_a_SNR7']
            epoch_list = ['best']

        case _:
            config_list = []


    match index:
        case 'synaptic_2':
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device, style=True)
                data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                                  best_model='best', run=0, step=100, test_simulation=False,
                                  sample_embedding=False, device=device)
                print(' ')
                print(' ')
        case 'synaptic_supp1' | 'synaptic_supp6':
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=['best'], device=device, style=True)
                data_plot(config=config, config_file=config_file, epoch_list=['all'], device=device, style=True)
        case 'synaptic_supp2':
            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            matplotlib.rcParams['savefig.pad_inches'] = 0

            config_list = ['signal_N2_a1', 'signal_N2_a2', 'signal_N2_a3', 'signal_N2_a4', 'signal_N2_a5', 'signal_N2_a10']
            it_list = [1, 2, 3, 4, 5, 10]
            fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
            plt.xlim([0, 100])
            plt.ylim([0, 1.1])
            plt.yticks(fontsize=24)
            plt.xticks([0, 100], [0, 20], fontsize=24)
            plt.ylabel(r'$R^2$', fontsize=48)
            plt.xlabel(r'epoch', fontsize=48)
            for it, config_file_ in enumerate(config_list):

                config_file, pre_folder = add_pre_folder(config_file_)
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                l_dir = get_log_dir(config)
                log_dir = os.path.join(l_dir, config_file.split('/')[-1])
                r_squared_list = np.load(f'./{log_dir}/results/R2.npy')
                plt.plot(r_squared_list, linewidth = 4, label=f'{(it_list[it])*10000}')
                plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/R2_all.png', dpi=300)
            plt.close()

        case 'synaptic_supp5':

            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            matplotlib.rcParams['savefig.pad_inches'] = 0

            config_list = ['signal_N2_a10','signal_N2_e1','signal_N2_e2','signal_N2_e3','signal_N2_e']
            labels = [r'100\%', r'50\%', r'20\%', r'10\%', r'5\%']
            fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
            plt.xlim([0, 100])
            plt.ylim([0, 1.1])
            plt.yticks(fontsize=24)
            plt.xticks([0, 100], [0, 20], fontsize=24)
            plt.ylabel(r'$R^2$', fontsize=48)
            plt.xlabel(r'epoch', fontsize=48)
            for it, config_file_ in enumerate(config_list):

                config_file, pre_folder = add_pre_folder(config_file_)
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                l_dir = get_log_dir(config)
                log_dir = os.path.join(l_dir, config_file.split('/')[-1])
                r_squared_list = np.load(f'./{log_dir}/results/R2.npy')
                plt.plot(r_squared_list, linewidth = 4, label=labels[it])
                plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/R2_all.png', dpi=300)
            plt.close()





        case '3' | '4' |'4_bis' | 'supp4' | 'supp5' | 'supp6' | 'supp7' | 'supp8' | 'supp9' | 'supp10' | 'supp11' | 'supp12' | 'supp15' |'supp16' |'supp18':
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
                data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                                  best_model=20, run=0, step=64, test_simulation=False,
                                  sample_embedding=False, device=device)  # config.simulation.n_frames // 7
                print(' ')
                print(' ')

        case 'supp1':
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=0, style='latex bw', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                              best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                              sample_embedding=False, device=device)

        case 'supp2':
            config_file = 'arbitrary_3_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='stripes', ratio = 1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)
            config_file = 'arbitrary_3_ter'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='pattern', ratio = 1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, device=device)
            config_file = 'arbitrary_3_quad'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_quad.yaml')
            # data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
            #               scenario='pattern', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, ratio = 3, device=device)

        case 'supp3':
            config_file = 'arbitrary_3_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='uniform 0', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, device=device)
            config_file = 'arbitrary_3_ter'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='uniform 1', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, device=device)
            config_file = 'arbitrary_3_quad'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_quad.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='uniform 2', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, ratio = 3, device=device)

        case 'supp6':
            config_file = 'arbitrary_3_field_boats'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_field_boats.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp7':
            config = ParticleGraphConfig.from_yaml(f'./config/gravity_16_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex bw', alpha=1, erase=True,
                          scenario='stripes', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'gravity_16'
            config = ParticleGraphConfig.from_yaml(f'./config/gravity_16.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='stripes', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp11':
            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex bw', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'boids_16_256'
            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp13':

            r=[]
            for n in range(16):
                result = np.load(f'./log/try_boids_16_256_{n}/rmserr_geomloss_boids_16_256_{n}.npy')
                print (n,result)
                r.append(result)
            print('mean',np.mean(r,axis=0))

            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1,
                          erase=True,
                          scenario='stripes', ratio=4, bSave=True, step=config.simulation.n_frames // 7)
            config_file = f'boids_16_256_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_bis.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 7, test_simulation=False,
                      sample_embedding=True, device=device)

            for n in range(16):
                copyfile(f'./config/boids_16_256.yaml', f'./config/boids_16_256_{n}.yaml')
                config_file = f'boids_16_256_{n}'
                config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_{n}.yaml')
                data_generate(config, device=device, visualize=True, run_vizualized=1, style='no_ticks color', alpha=1, erase=True,
                              scenario=f'uniform {n}', ratio=4, bSave=True, step=config.simulation.n_frames // 3)
                data_test(config=config, config_file=config_file, visualize=True, style='no_ticks color', verbose=False,
                          best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                          sample_embedding=True, device=device)

        case 'supp15':
            config = ParticleGraphConfig.from_yaml(f'./config/wave_slit_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'wave_slit_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_slit_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1,
                          erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 100)
            config_file = 'wave_slit_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_slit_bis.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 100, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp16':
            config = ParticleGraphConfig.from_yaml(f'./config/wave_boat_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'wave_boat_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_boat_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1,
                          erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'wave_boat_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_boat_bis.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 100, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp17':
            config = ParticleGraphConfig.from_yaml(f'./config/RD_RPS.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            config = ParticleGraphConfig.from_yaml(f'./config/RD_RPS_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style='latex color', verbose=False,
                              best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                              sample_embedding=False, device=device)

        case 'supp18':
            config = ParticleGraphConfig.from_yaml(f'./config/signal_N_100_2.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            config = ParticleGraphConfig.from_yaml(f'./config/signal_N_100_2_a.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style=' color', verbose=False,
                              best_model=20, run=0, step=config.simulation.n_frames // 100, test_simulation=False,
                              sample_embedding=False, device=device)

    print(' ')
    print(' ')

    return config_list,epoch_list


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(' ')
    print(f'device {device}')

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass


    # config_list = ['signal_N2_a1', 'signal_N2_a2', 'signal_N2_a3', 'signal_N2_a4', 'signal_N2_a5', 'signal_N2_a10']
    # config_list = ['signal_N6_a28']
    config_list = ['signal_N6_a28_11_3']

    for config_file_ in config_list:

        print(' ')

        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        print(f'config_file  {config.config_file}')

        # data_plot(config=config, epoch_list=['best'], style='black color', device=device)
        data_plot(config=config, epoch_list=['all'], style='black color', device=device)
        # data_plot(config=config, epoch_list=['time'], style='black color', device=device)

        # plot_generated(config=config, run=0, style='black voronoi color', step = 10, style=False, device=device)
        # plot_focused_on_cell(config=config, run=0, style='color', cell_id=175, step = 5, device=device)

    # f_list = ['synaptic_supp5']
    # for f in f_list:
    #     config_list,epoch_list = get_figures(f)




