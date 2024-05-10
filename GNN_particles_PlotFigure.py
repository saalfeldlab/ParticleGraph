import matplotlib.cm as cmplt
from matplotlib.ticker import FormatStrFormatter
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as pyg_utils
import os
from ParticleGraph.MLP import MLP
import imageio
from matplotlib import rc


from ParticleGraph.generators import RD_RPS
from ParticleGraph.models import Interaction_Particles, Mesh_Laplacian
from ParticleGraph.fitting_models import power_model, boids_model, reaction_diffusion_model

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# from data_loaders import *
from GNN_particles_Ntype import *
from ParticleGraph.embedding_cluster import *
from ParticleGraph.utils import to_numpy, CustomColorMap, choose_boundary_values
import matplotlib as mpl

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


class RD_RPS_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=None, c=None, beta=None, bc_dpos=None):
        super(RD_RPS_extract, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.bc_dpos = bc_dpos

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # dx = 2./size
        # dt = 0.9 * dx**2/2
        # params = {"Du":5e-3, "Dv":2.8e-4, "tau":0.1, "k":-0.005,
        # su = (Du*Lu + v - u)/tau
        # sv = Dv*Lv + v - v*v*v - u + k

        c = self.c[to_numpy(x[:, 5])]
        c = c[:, None]

        u = x[:, 6]
        v = x[:, 7]
        w = x[:, 8]

        laplacian = self.beta * c * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        laplacian_u = laplacian[:, 0]
        laplacian_v = laplacian[:, 1]
        laplacian_w = laplacian[:, 2]

        # Du = 5E-3
        # Dv = 2.8E-4
        # k = torch.tensor(-0.005,device=device)
        # tau = torch.tensor(0.1,device=device)
        #
        # dU = (Du * laplacian[:,0] + v - u) / tau
        # dV = Dv * laplacian[:,1] + v - v**3 - u + k

        D = 0.05
        a = 0.6
        p = u + v + w

        du = D * laplacian_u + u * (1 - p - a * v)
        dv = D * laplacian_v + v * (1 - p - a * w)
        dw = D * laplacian_w + w * (1 - p - a * u)

        # U = U + 0.125 * dU
        # V = V + 0.125 * dV

        increment = torch.cat((du[:, None], dv[:, None], dw[:, None]), dim=1)

        return increment

    def message(self, x_i, x_j, edge_attr):
        # U column 6, V column 7

        # L = edge_attr * (x_j[:, 6]-x_i[:, 6])

        Lu = edge_attr * x_j[:, 6]
        Lv = edge_attr * x_j[:, 7]
        Lw = edge_attr * x_j[:, 8]

        Laplace = torch.cat((Lu[:, None], Lv[:, None], Lw[:, None]), dim=1)

        return Laplace

    def psi(self, I, p):
        return I


class Mesh_RPS_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None):
        super(Mesh_RPS_extract, self).__init__(aggr=aggr_type)

        self.device = device
        self.input_size = config.graph_model.input_size
        self.output_size = config.graph_model.output_size
        self.hidden_dim = config.graph_model.hidden_dim
        self.n_layers = config.graph_model.n_mp_layers
        self.n_particles = config.simulation.n_particles
        self.max_radius = config.simulation.max_radius
        self.data_augmentation = config.training.data_augmentation
        self.embedding = config.graph_model.embedding_dim
        self.n_datasets = config.training.n_runs - 1
        self.bc_dpos = bc_dpos

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                           hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.n_datasets), int(self.n_particles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        laplacian = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        u = x[:, 6]
        v = x[:, 7]
        w = x[:, 8]

        laplacian = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        laplacian_u = laplacian[:, 0]
        laplacian_v = laplacian[:, 1]
        laplacian_w = laplacian[:, 2]

        embedding = self.a[self.data_id, to_numpy(x[:, 0]), :]

        input_phi = torch.cat((laplacian_u[:, None], laplacian_v[:, None], laplacian_w[:, None], u[:, None], v[:, None],
                               w[:, None], embedding), dim=-1)

        pred = self.lin_phi(input_phi)

        return pred, input_phi, embedding

    def message(self, x_i, x_j, edge_attr):
        # U column 6, V column 7

        # L = edge_attr * (x_j[:, 6]-x_i[:, 6])

        Lu = edge_attr * x_j[:, 6]
        Lv = edge_attr * x_j[:, 7]
        Lw = edge_attr * x_j[:, 8]

        Laplace = torch.cat((Lu[:, None], Lv[:, None], Lw[:, None]), dim=1)

        return Laplace

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r


class Mesh_RPS_learn(torch.nn.Module):

    def __init__(self):
        super(Mesh_RPS_learn, self).__init__()

        self.cc = nn.Parameter(torch.zeros(5, requires_grad=True))

        self.A = nn.Parameter(torch.zeros((9, 1), requires_grad=True))
        self.B = nn.Parameter(torch.zeros((9, 1), requires_grad=True))
        self.C = nn.Parameter(torch.zeros((9, 1), requires_grad=True))

    def forward(self, in_features, type, list_index):
        u = in_features[:, 3:4]
        v = in_features[:, 4:5]
        w = in_features[:, 5:6]

        l = list_index.shape[0]
        type = to_numpy(type)

        laplacian_u = self.cc[type, None] * in_features[:, 0:1]
        laplacian_v = self.cc[type, None] * in_features[:, 1:2]
        laplacian_w = self.cc[type, None] * in_features[:, 2:3]

        uu = u * u
        uv = u * v
        uw = u * w
        vv = v * v
        vw = v * w
        ww = w * w

        uu = uu[list_index]
        uv = uv[list_index]
        uw = uw[list_index]
        vv = vv[list_index]
        vw = vw[list_index]
        ww = ww[list_index]

        u = u[list_index]
        v = v[list_index]
        w = w[list_index]

        D = 0.05

        du = D * laplacian_u[list_index] + self.A[0] * uu + self.A[1] * uv + self.A[2] * uw + self.A[3] * vv + self.A[
            4] * vw + self.A[5] * ww + self.A[6] * u + self.A[7] * v + self.A[8] * w
        dv = D * laplacian_v[list_index] + self.B[0] * uu + self.B[1] * uv + self.B[2] * uw + self.B[3] * vv + self.B[
            4] * vw + self.B[5] * ww + self.B[6] * u + self.B[7] * v + self.B[8] * w
        dw = D * laplacian_w[list_index] + self.C[0] * uu + self.C[1] * uv + self.C[2] * uw + self.C[3] * vv + self.C[
            4] * vw + self.C[5] * ww + self.C[6] * u + self.C[7] * v + self.C[8] * w

        increment = torch.cat((du[:, None], dv[:, None], dw[:, None]), dim=1)

        return increment.squeeze()

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

def plot_umap(index, func_list, log_dir, n_neighbors, index_particles, n_particles, n_particle_types, embedding_cluster, epoch, it, fig, ax, cmap,device):

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

def data_plot_attraction_repulsion(config_file):

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    # print(config.pretty())

    cmap = CustomColorMap(config=config)
    aggr_type = config.graph_model.aggr_type

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    nrun = config.training.n_runs

    index_particles = []
    np_i = int(n_particles / n_particle_types)
    for n in range(n_particle_types):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    n_graphs = len(graph_files)
    print('Graph files N: ', n_graphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load normalizations ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_1.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    epoch = 20

    net = f"./log/try_{config_file}/models/best_model_with_{nrun - 1}_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.use("Qt5Agg")

    fig = plt.figure(figsize=(12.5, 9.6))
    plt.ion()
    ax = fig.add_subplot(3, 4, 1)
    embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 1, '$5.10^4$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 4, 2)
    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    func_list = plot_function(True,'b)', config.graph_model.particle_model_name, model.lin_edge, model.a, 1, to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles, n_particles, n_particle_types, 1, '$5.10^4$', fig, ax, cmap, device)

    ax = fig.add_subplot(3, 4, 3)
    proj_interaction, new_labels, n_clusters = plot_umap('c)', func_list, log_dir, 500, index_particles, n_particles, n_particle_types, embedding_cluster, 1, '$5.10^4$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 4, 4)
    Accuracy = plot_confusion_matrix('d)', to_numpy(x[:,5:6]), new_labels, n_particle_types, 1, '$5.10^4$', fig, ax)

    net = f"./log/try_{config_file}/models/best_model_with_{nrun - 1}_graphs_18.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    ax = fig.add_subplot(3, 4, 5)
    embedding = plot_embedding('e)', model.a, 1, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 4, 6)
    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    func_list = plot_function(True,'f)', config.graph_model.particle_model_name, model.lin_edge, model.a, 1, to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 4, 7)
    proj_interaction, new_labels, n_clusters = plot_umap('g)', func_list, log_dir, 500, index_particles, n_particles, n_particle_types, embedding_cluster, 20, '$10^6$', fig, ax, cmap,device)

    match config.training.cluster_method:
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


    ax = fig.add_subplot(3, 4, 8)
    Accuracy = plot_confusion_matrix('h)', to_numpy(x[:,5:6]), new_labels, n_particle_types, 1, '$5.10^$4', fig, ax)
    plt.tight_layout()

    model_a_ = model.a[1].clone().detach()
    for k in range(n_clusters):
        pos = np.argwhere(new_labels == k).squeeze().astype(int)
        if pos.size > 0:
            temp = model_a_[pos, :].clone().detach()
            model_a_[pos, :] = torch.median(temp, dim=0).values.repeat((len(pos), 1))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_
    embedding = get_embedding(model.a, 1)

    ax = fig.add_subplot(3, 4, 9)
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

    ax = fig.add_subplot(3, 4, 10)
    print('10')
    plt.text(-0.25, 1.1, f'j)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (model)', fontsize=12)
    func_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        embedding = model.a[0, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, embedding), dim=1)
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
    plt.ylim([-0.04, 0.03])
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 4, 11)
    print('11')
    plt.text(-0.25, 1.1, f'k)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (true)', fontsize=12)
    p = config.simulation.params
    if len(p) > 0:
        p = torch.tensor(p, device=device)
    else:
        p = torch.load(f'graphs_data/graphs_{dataset_name}/p.pt')
    for n in range(n_particle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.ylim([-0.04, 0.03])

    # find last image file in logdir
    ax = fig.add_subplot(3, 4, 12)
    files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        last_file = files[-1]
        # load image file with imageio
        image = imageio.imread(last_file)
        print('12')
        plt.text(-0.25, 1.1, f'l)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Rollout inference (frame 250)', fontsize=12)
        plt.imshow(image)
        # rmove xtick
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    # plt.savefig('Fig2.pdf', format="pdf", dpi=300)
    plt.savefig('Fig2.jpg', dpi=300)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    embedding = get_embedding(model.a, 1)
    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                    cmap='viridis')
    else:
        for n in range(n_particle_types):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            if pos.size > 0:
                plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=200)

    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
    # xticks with 1 digit

    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    # ax.xaxis.get_major_formatter()._usetex = False
    # ax.yaxis.get_major_formatter()._usetex = False
    func_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        if pos.size > 0:
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
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
    # xticks with sans serif font
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, max_radius])

    plt.ylim([-0.04, 0.03])
    # plt.ylim([-0.04, 0.03])
    # plt.ylim([-0.08, 0.08])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    # ax.xaxis.get_major_formatter()._usetex = False
    # ax.yaxis.get_major_formatter()._usetex = False
    p = config.simulation.params
    true_func_list = []
    for n in range(n_particle_types):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=4,)
        true_func_list.append(model.psi(rr, p[n], p[n]))
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, max_radius])
    # plt.ylim([-0.15, 0.15])
    plt.ylim([-0.04, 0.03])
    # plt.ylim([-0.08, 0.08])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}.tif", dpi=170.7)
    plt.close()

    func_list = torch.stack(func_list)*ynorm
    true_func_list = torch.stack(true_func_list)

    rmserr = torch.sqrt(torch.mean((func_list - true_func_list) ** 2))
    print(f'RMS error: {np.round(rmserr.item(),7)}')
    rmserr = rmserr / ( torch.max(true_func_list) - torch.min(true_func_list))
    print(f'RMS error / (max-min): {np.round(rmserr.item(), 7)}')

def data_plot_attraction_repulsion_asym(config_file):

    config_file = 'arbitrary_3_3'
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    # print(config.pretty())

    cmap = CustomColorMap(config=config)
    aggr_type = config.graph_model.aggr_type

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    nrun = config.training.n_runs

    index_particles = []
    np_i = int(n_particles / n_particle_types)
    for n in range(n_particle_types):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    n_graphs = len(graph_files)
    print('Graph files N: ', n_graphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load normalizations ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_1.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{dataset_name}/models/best_model_with_{nrun - 1}_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

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


    fig = plt.figure(figsize=(12.5, 9.6))
    plt.ion()
    ax = fig.add_subplot(3, 4, 1)
    embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 1, '$5.10^4$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 4, 2)
    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    func_list = plot_function(True,'f)', config.graph_model.particle_model_name, model.lin_edge, model.a, 1, to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 4, 3)
    proj_interaction, new_labels, n_clusters = plot_umap('g)', func_list, log_dir, 500, index_particles, n_particles, n_particle_types, embedding_cluster, 20, '$10^6$', fig, ax, cmap,device)

    ax = fig.add_subplot(3, 4, 4)
    Accuracy = plot_confusion_matrix('h)', to_numpy(x[:,5:6]), new_labels, n_particle_types, 1, '$5.10^$4', fig, ax)
    plt.tight_layout()

    model_a_ = model.a[1].clone().detach()
    for k in range(n_clusters):
        pos = np.argwhere(new_labels == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        model_a_[pos, :] = torch.median(temp, dim=0).values.repeat((len(pos), 1))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_
    embedding = get_embedding(model.a, 1)

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
        embedding_1 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        for m in range(n_particle_types):
            pos = np.argwhere(new_labels == m).squeeze().astype(int)
            embedding_2 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)

            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
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
    plt.ylim([-0.15, 0.15])
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 4, 7)
    print('11')
    plt.text(-0.25, 1.1, f'k)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (true)', fontsize=12)
    p = config.simulation.params
    p = torch.ones(n_particle_types, n_particle_types, 4, device=device)
    params=config.simulation.params                                                                             
    if params[0] != [-1]:
        for n in range(n_particle_types):
            for m in range(n_particle_types):
                p[n, m] = torch.tensor(params[n * 3 + m])
    for n in range(n_particle_types):
        for m in range(n_particle_types):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n,m], p[n,m])), color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.ylim([-0.15, 0.15])

    epoch=20


    # find last image file in logdir
    ax = fig.add_subplot(3, 4, 12)
    files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        last_file = files[-1]
        # load image file with imageio
        image = imageio.imread(last_file)
        print('12')
        plt.text(-0.25, 1.1, f'l)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Rollout inference (frame 250)', fontsize=12)
        plt.imshow(image)
        # rmove xtick
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    # plt.savefig('Fig8.pdf', format="pdf", dpi=300)
    plt.savefig('Fig8.jpg', dpi=300)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    embedding = get_embedding(model.a, 1)
    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                    cmap='viridis')
    else:
        for n in range(n_particle_types):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            if pos.size > 0:
                plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=200)

    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
    # xticks with 1 digit

    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    func_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        embedding_1 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        for m in range(n_particle_types):
            pos = np.argwhere(new_labels == m).squeeze().astype(int)
            embedding_2 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)

            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n), linewidth=4)
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
    # xticks with sans serif font
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.ylim([-0.15, 0.15])
    plt.xlim([0, max_radius])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/func_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    for n in range(n_particle_types):
        for m in range(n_particle_types):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n, m], p[n, m])), color=cmap.color(n), linewidth=4)
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
    # xticks with sans serif font
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.ylim([-0.15, 0.15])
    plt.xlim([0, max_radius])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}.tif", dpi=170.7)
    plt.close()

def data_plot_gravity(config_file):

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
    nrun = config.training.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    n_graphs = len(graph_files)
    print('Graph files N: ', n_graphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load normalizations ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_1.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    type_list = x[:, 5:6].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{config_file}/models/best_model_with_{nrun - 1}_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.use("Qt5Agg")

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
    Accuracy = plot_confusion_matrix('c)', to_numpy(x[:,5:6]), new_labels, n_particle_types, 20, '$10^6$', fig, ax)
    plt.tight_layout()

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
    plt.xlabel(r'UMAP-proj 0', fontsize=64)
    plt.ylabel(r'UMAP-proj 1', fontsize=64)
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

def data_plot_gravity_continuous(config_file):

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
    nrun = config.training.n_runs

    index_particles = []
    np_i = int(n_particles / n_particle_types)
    for n in range(n_particle_types):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    n_graphs = len(graph_files)
    print('Graph files N: ', n_graphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load normalizations ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{config_file}/models/best_model_with_{nrun - 1}_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.use("Qt5Agg")

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

def data_plot_gravity_solar_system(config_file):

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
    nrun = config.training.n_runs
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    index_particles = []
    np_i = int(n_particles / n_particle_types)
    for n in range(n_particle_types):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    n_graphs = len(graph_files)
    print('Graph files N: ', n_graphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load normalizations ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{config_file}/models/best_model_with_{nrun - 1}_graphs_2.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.use("Qt5Agg")

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

def data_plot_boids(config_file):
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
    nrun = config.training.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    n_graphs = len(graph_files)
    print('Graph files N: ', n_graphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load normalizations ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()

    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    n_particles = int(n_particles * (1 - train_config.particle_dropout))

    epoch_list = [10]

    for epoch in epoch_list:

        model, bc_pos, bc_dpos = choose_training_model(config, device)
        model = Interaction_Particles_extract(config, device, aggr_type=config.graph_model.aggr_type, bc_dpos=bc_dpos)

        net = f"./log/try_{config_file}/models/best_model_with_{nrun - 1}_graphs_{epoch}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        matplotlib.use("Qt5Agg")

        fig = plt.figure(figsize=(10.5, 9.6))
        plt.ion()
        ax = fig.add_subplot(3, 3, 1)
        embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap, device)

        fig_ = plt.figure(figsize=(12, 12))
        ax = fig_.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model.a, 1)
        embedding = embedding[0:n_particles]
        csv_ = embedding
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=200)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        csv_ = np.array(csv_)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.tif", dpi=300)
        np.save(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.txt", csv_)
        plt.close()

        ax = fig.add_subplot(3, 3, 2)

        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=False, config=config,
                                                            model_lin_edge=model.lin_edge, model_a=model.a,
                                                            dataset_number=1,
                                                            n_particles=n_particles,
                                                            ynorm=ynorm,
                                                            types=to_numpy(x[:, 5]),
                                                            cmap=cmap, device=device)
        # train_config.cluster_method = 'distance_embedding'
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
                labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=0.025)
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

        new_labels = labels.copy()
        for n in range(n_particle_types):
            new_labels[labels == label_list[n]] = n

        if dimension == 2:
            type_list = x[:, 5:6].clone().detach()
        elif dimension == 3:
            type_list = x[:, 7:8].clone().detach()

        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')

        it = 7000

        x = x_list[0][it].clone().detach()

        distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
        t = torch.Tensor([max_radius ** 2])  # threshold
        adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)) * 1.0
        edge_index = adj_t.nonzero().t().contiguous()
        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y, in_features, lin_edge_out = model(dataset, data_id=1, training=False, vnorm=vnorm,
                                                 phi=torch.zeros(1, device=device))  # acceleration estimation
        y = y * ynorm
        lin_edge_out = lin_edge_out * ynorm
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        model_B = PDE_B_extract(aggr_type=config.graph_model.aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)
        psi_output = []
        for n in range(n_particle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
        with torch.no_grad():
            y_B, sum, cohesion, alignment, separation, diffx, diffv, r, type = model_B(dataset)  # acceleration estimation
        type = to_numpy(type)

        ax = fig.add_subplot(3, 3, 4)
        plt.text(-0.25, 1.1, 'e)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Clustered particle embedding', fontsize=12)
        for n in range(n_particle_types):
            pos = np.argwhere(type == n)
            pos = pos[:, 0].astype(int)
            plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=6)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)
        plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

        ax = fig.add_subplot(3, 3, 5)
        plt.text(-0.25, 1.1, f'e)', ha='right', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Interaction functions (model)', fontsize=12)
        for n in range(n_particle_types):
            pos = np.argwhere(type == n)
            pos = pos[:, 0].astype(int)
            plt.scatter(to_numpy(diffx[pos, 0]), to_numpy(lin_edge_out[pos, 0]), color=cmap.color(n), s=50, alpha=0.5)
        plt.ylim([-0.08, 0.08])
        plt.ylim([-5E-5, 5E-5])
        plt.xlabel(r'$d_{ij}$', fontsize=12)
        plt.ylabel(r'$\left| \left| f(\ensuremath{\mathbf{a}}_i, x_j-x_i, \dot{x}_i, \dot{x}_j, d_{ij} \right| \right|[a.u.]$',fontsize=12)

        fig_ = plt.figure(figsize=(12, 12))
        ax = fig_.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        for n in range(n_particle_types):
            pos = np.argwhere(type == n)
            pos = pos[:, 0].astype(int)
            plt.scatter(to_numpy(diffx[pos, 0]), to_numpy(lin_edge_out[pos, 0]), color=cmap.color(n), s=50, alpha=0.5)
        plt.ylim([-5E-5, 5E-5])
        plt.xlabel(r'$x_j-x_i$', fontsize=64)
        plt.ylabel( r'$f_{ij,x}$',fontsize=64)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_all_{config_file}_{epoch}.tif", dpi=300)
        np.save(f"./{log_dir}/tmp_training/func_all_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/func_all_{config_file}_{epoch}.txt", csv_)
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
        plt.ylim([-5E-5, 5E-5])
        plt.xlabel(r'$x_j-x_i$', fontsize=64)
        plt.ylabel( r'$f_{ij,x}$',fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{config_file}_{epoch}.tif", dpi=300)
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
        plt.savefig(f"./{log_dir}/tmp_training/func_dij_{config_file}_{epoch}.tif", dpi=300)
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
        plt.savefig(f"./{log_dir}/tmp_training/true_func_dij_{config_file}_{epoch}.tif", dpi=300)

        func_list = func_list * ynorm
        func_list_ = torch.clamp(func_list, min=torch.tensor(-1.0E-4,device=device), max=torch.tensor(1.0E-4,device=device))
        true_func_list_ = torch.clamp(true_func_list, min=torch.tensor(-1.0E-4, device=device),
                                 max=torch.tensor(1.0E-4, device=device))
        rmserr_list = torch.sqrt(torch.mean((func_list_ - true_func_list_) ** 2,1))
        rmserr_list = to_numpy(rmserr_list)
        print(f'all function RMS error : {np.round(np.mean(rmserr_list), 8)}+/-{np.round(np.std(rmserr_list), 8)}')

        xs = torch.linspace(0, 1, 400)
        ys = torch.linspace(-1, 1, 400)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        # find last image file in logdir
        ax = fig.add_subplot(3, 3, 6)
        files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
        files.sort(key=os.path.getmtime)
        if len(files) > 0:
            last_file = files[-1]
            image = imageio.imread(last_file)
            plt.text(-0.25, 1.1, f'f)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
            plt.title(r'Rollout inference (frame 8000)', fontsize=12)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

        cohesion_fit = np.zeros(n_particle_types)
        alignment_fit = np.zeros(n_particle_types)
        separation_fit = np.zeros(n_particle_types)

        for n in range(n_particle_types):
            pos = np.argwhere(type == n)
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
            pos = np.argwhere(type == n)
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

        ax = fig.add_subplot(3, 3, 7)
        plt.text(-0.25, 1.1, f'g)', ha='right', va='top', transform=ax.transAxes, fontsize=12)
        x_data = np.abs(to_numpy(p[:, 0]) * 0.5E-5)
        y_data = np.abs(cohesion_fit)
        x_data = x_data[index_classified]
        y_data = y_data[index_classified]

        threshold = 0.1

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

        time.sleep(1)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/Fig5.jpg", dpi=300)
        plt.close()

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
        plt.savefig(f"./{log_dir}/tmp_training/cohesion_{config_file}_{epoch}.tif", dpi=300)
        np.save(f"./{log_dir}/tmp_training/cohesion_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/cohesion_{config_file}_{epoch}.txt", csv_)
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
        plt.savefig(f"./{log_dir}/tmp_training/alignment_{config_file}_{epoch}.tif", dpi=300)
        np.save(f"./{log_dir}/tmp_training/alignment_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/alignement_{config_file}_{epoch}.txt", csv_)
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
        plt.savefig(f"./{log_dir}/tmp_training/separation_{config_file}_{epoch}.tif", dpi=300)
        np.save(f"./{log_dir}/tmp_training/separation_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/separation_{config_file}_{epoch}.txt", csv_)
        plt.close()







def data_plot_Coulomb(config_file):

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
    nrun = config.training.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    n_graphs = len(graph_files)
    print('Graph files N: ', n_graphs - 1)
    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load normalizations ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()
    if dimension == 2:
        type_list = x[:, 5:6].clone().detach()
    elif dimension == 3:
        type_list = x[:, 7:8].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    epoch=20
    net = f"./log/try_{config_file}/models/best_model_with_{nrun - 1}_graphs_{epoch}.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.use("Qt5Agg")

    fig = plt.figure(figsize=(10.5, 9.6))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap,device)
    embedding = embedding[0:int(n_particles * (1 - train_config.particle_dropout))]

    ax = fig.add_subplot(3, 3, 2)

    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    if dimension == 2:
        column_dimension = 5
    if dimension == 3:
        column_dimension = 7

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=False, config=config,
                                                        model_lin_edge=model.lin_edge, model_a=model.a,
                                                        dataset_number=1,
                                                        n_particles=int(
                                                            n_particles * (1 - train_config.particle_dropout)),
                                                        ynorm=ynorm,
                                                        types=to_numpy(x[:, 5]),
                                                        cmap=cmap, device=device)
    plt.xlim([0, 0.02])
    plt.ylim([-0.5E6, 0.5E6])
    plt.xlabel(r'$d_{ij}$', fontsize=64)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_i, d_{ij}$)', fontsize=64)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function_ai_ai_{config_file}_{epoch}.tif", dpi=170.7)
    csv_ = to_numpy(func_list)
    np.save(f"./{log_dir}/tmp_training/function_ai_ai_{config_file}_{epoch}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/function_ai_ai_{config_file}_{epoch}.txt", csv_)
    plt.close()
    #
    # train_config.cluster_method = 'kmeans_auto_embedding'
    # train_config.cluster_method = 'kmeans'
    match train_config.cluster_method:
        case 'kmeans':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans')
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=0.25)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance')
    ax = fig.add_subplot(3, 3, 2)

    label_list = []
    for n in range(n_particle_types):
        tmp = labels[index_particles[n]]
        label_list.append(np.round(np.median(tmp)))
    label_list = np.array(label_list)
    new_labels = labels.copy()
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n

    plt.xlabel('proj 0', fontsize=12)
    plt.ylabel('proj 1', fontsize=12)
    plt.text(0., 1.1, f'Nclusters: {n_clusters}', ha='left', va='top', transform=ax.transAxes)

    new_labels = labels.copy()
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n

    Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
    print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    embedding = get_embedding(model.a, 1)
    csv_ = embedding
    np.save(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training/embedding_{config_file}_{epoch}.txt", csv_)
    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                    cmap='viridis')
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

    fig_ = plt.figure(figsize=(12, 12))
    ax = fig_.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for n in range(n_particle_types):
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1], color=cmap.color(n), s=400, alpha=0.1)
    plt.xlabel(r'$\ensuremath{\mathbf{UMAP}_{0}$', fontsize=64)
    plt.ylabel(r'$\ensuremath{\mathbf{UMAP}}_{1}$', fontsize=64)
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    csv_ = proj_interaction
    np.save(f"./{log_dir}/tmp_training//umap_{config_file}_{epoch}.npy", csv_)
    np.savetxt(f"./{log_dir}/tmp_training//umap_{config_file}_{epoch}.txt", csv_)
    plt.savefig(f"./{log_dir}/tmp_training/umap_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    ax = fig.add_subplot(3, 3, 3)
    Accuracy = plot_confusion_matrix('c)', to_numpy(x[:,5:6]), new_labels, n_particle_types, 20, '$10^6$', fig, ax)
    plt.tight_layout()

    model_a_ = model.a[1].clone().detach()
    for n in range(n_clusters):
        pos = np.argwhere(labels == n).squeeze().astype(int)
        pos = np.array(pos)
        if pos.size > 0:
            median_center = model_a_[pos, :]
            median_center = torch.median(median_center, dim=0).values
            model_a_[pos, :] = median_center
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_
    embedding = get_embedding(model.a, 1)

    ax = fig.add_subplot(3, 3, 4)
    plt.text(-0.25, 1.1, f'd)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Clustered particle embedding', fontsize=12)
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        if len(pos)>0:
            plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=6)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    plt.text(-0.25, 1.1, f'e)', ha='right', va='top', transform=ax.transAxes, fontsize=12)
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)
    plt.title(r'Interaction functions (model)', fontsize=12)
    t = to_numpy(model.a)
    tmean = np.ones((n_particle_types, config.graph_model.embedding_dim))
    for n in range(n_particle_types):
        tmean[n] = np.mean(t[:, index_particles[n], :], axis=(0, 1))
    for m in range(n_particle_types):
        for n in range(n_particle_types):
            embedding0 = torch.tensor(tmean[m], device=device) * torch.ones((1000, config.graph_model.embedding_dim),
                                                                            device=device)
            embedding1 = torch.tensor(tmean[n], device=device) * torch.ones((1000, config.graph_model.embedding_dim),
                                                                            device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding0, embedding1), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            plt.plot(to_numpy(rr),
                     to_numpy(acc) * to_numpy(ynorm),
                     linewidth=1)
    plt.xlim([0, 0.02])
    plt.ylim([-0.5E6, 0.5E6])

    p = [2, 1, -1]
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


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')
    #
    # epoch=20
    # config_file = 'arbitrary_32'
    # data_plot_FIG2()

    # config_list = ['gravity_16','gravity_16_noise_1E-5','gravity_16_noise_1E-4','gravity_16_noise_1E-3','gravity_16_noise_1E-2','gravity_16_noise_1E-1']
    # config_list = ['gravity_16_dropout_10_no_ghost', 'gravity_16_dropout_10', 'gravity_16_dropout_20', 'gravity_16_dropout_30', 'gravity_16_dropout_40', 'gravity_16_dropout_50']
    # config_list = ['gravity_16_dropout_20'] # ['gravity_16_dropout_10_no_ghost','gravity_16_dropout_10','gravity_16_dropout_20','gravity_16_dropout_30']
    # config_list = ['gravity_16_noise_0_2']
    # config_list = ['gravity_16_noise_0_5'] #'gravity_16_noise_0_4']# , 'gravity_16_noise_0_3', 'gravity_16_noise_0_5']
    # config_list = ['boids_64_256'] #['boids_32_256','boids_64_256'] # 'boids_16_256_1_epoch', 'boids_32_256_1_epoch', 'boids_64_256_1_epoch'] #,
    config_list = ['boids_16_noise_0_3','boids_16_noise_0_4'] # ['boids_16_noise_1E-1'] #,'boids_16_noise_0_2','boids_16_noise_0_3','boids_16_noise_0_4','boids_16_noise_0_5']
    config_list = ['boids_64_256_20_epoch']


    # config_list = ['Coulomb_3_noise_0_2','Coulomb_3_noise_0_3','Coulomb_3_noise_0_4']
    # config_list = ['Coulomb_3_dropout_10'] #,'Coulomb_3_dropout_10', 'Coulomb_3_dropout_10_no_ghost'] # ,'Coulomb_3_dropout_20', 'Coulomb_3_dropout_30', 'Coulomb_3_dropout_40'


    for config_file in config_list:

        data_plot_boids(config_file)
        # data_plot_gravity(config_file)

