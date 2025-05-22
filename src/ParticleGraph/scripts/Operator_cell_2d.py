
import umap
import torch
from ParticleGraph.models.MLP import MLP
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *

# from ParticleGraph.models.utils import reparameterize
# from ParticleGraph.models.Siren_Network import Siren
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



def density_laplace(y, x):
    grad = density_gradient(y, x)
    return density_divergence(grad, x)


def density_divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def density_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class Operator_smooth(pyg.nn.MessagePassing):

    """
    Model learning kernel operators.
    The methods follows the particle smoothing techniques proposed in the paper:
    'Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids'

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the kernel operators and their convolution with the data
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2, model_density=[]):

        super(Operator_smooth, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers
        self.n_particles = simulation_config.n_particles
        self.n_particles_max = simulation_config.n_particles_max
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.time_window_noise = train_config.time_window_noise
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update
        self.model_type = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.time_window = train_config.time_window
        self.model_density = model_density
        self.sub_sampling = simulation_config.sub_sampling
        self.prediction = model_config.prediction
        self.kernel_var = 2 * self.max_radius ** 2

        self.kernel_norm = np.pi * self.kernel_var * (1 - np.exp(-self.max_radius ** 2/ self.kernel_var))
        self.field_type = model_config.field_type

        if self.update_type == 'pre_mlp':
            self.pre_lin_edge = MLP(input_size=self.pre_input_size, output_size=self.pre_output_size, nlayers=self.pre_n_layers,
                                hidden_size=self.pre_hidden_dim, device=self.device)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if 'mlp' in self.update_type:
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles_max) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

        self.siren = Siren_Network(image_width=100, in_features=model_config.input_size_nnr,
                                out_features=model_config.output_size_nnr,
                                hidden_features=model_config.hidden_dim_nnr,
                                hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80,
                                hidden_omega_0=model_config.omega )

    def forward(self, data=[], data_id=[], training=[], phi=[], continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_id = x[:, 0:1].long()
        embedding = self.a[data_id, particle_id, :].squeeze()
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        density_null = torch.zeros((pos.shape[0], 2), device=self.device)
        if continuous_field:
            self.mode = 'pre_mlp'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            density = torch.zeros((pos.shape[0], 1), device=self.device)
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density)
        else:
            self.mode = 'pre_mlp'
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=self.density)

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, embedding_i, embedding_j, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'pre_mlp':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            density_kernel = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:,None]

            # self.modulation = self.siren(coords=mgrid) * max_radius **2
            # kernel_modified = torch.exp(-2*(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20*self.kernel_var))[:, None] * self.modulation
            kernel_modified = torch.exp(-2 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20 * self.kernel_var))[:, None]

            grad_autograd = -density_gradient(density_kernel, mgrid)
            laplace_autograd = density_laplace(density_kernel, mgrid)

            self.kernel_operators = torch.cat((density_kernel, grad_autograd, laplace_autograd), dim=-1)

            return density_kernel

            kernel_modified = torch.exp(-2 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (self.kernel_var))[:, None]
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=10, c=to_numpy(kernel_modified), vmin=0, vmax=1)
            plt.colorbar()
            plt.show()

        else:
            # out = self.lin_edge(field_j) * self.kernel_operators[:,1:2] / density_j
            # out = self.lin_edge(field_j) * self.kernel_operators[:,3:4] / density_j
            # out = field_j * self.kernel_operators[:, 1:2] / density_j

            grad_density = self.kernel_operators[:, 1:3]  # d_rho_x d_rho_y

            # velocity = self.kernel_operators[:, 0:1] * torch.sum(d_pos_j**2, dim=1)[:,None] / density_j
            # grad_velocity = self.kernel_operators[:, 1:3] * torch.sum(d_pos_j**2, dim=1)[:,None].repeat(1,2) / density_j.repeat(1,2)
            # out = torch.cat((grad_density, velocity, grad_velocity), dim = 1) # d_rho_x d_rho_y, velocity
            # out = field_j * self.kernel_operators[:, 1:2] / density_j  # grad_x

            if 'laplacian' in self.field_type:
                out = field_j * self.kernel_operators[:, 3:4] / density_j  # laplacian
            elif 'grad_density' in self.field_type:
                out = grad_density
            else:
                out = grad_density

            return out


        fig = plt.figure(figsize=(6, 6))
        plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=100, c=to_numpy(self.kernel_operators[:,3:4]))

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=100, c=to_numpy(self.pre_lin_edge(mgrid)))

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)



if __name__ == '__main__':


    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import torch_geometric.data as data
    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import os
    import shutil
    from torch_geometric.loader import DataLoader

    mode = 'cell_MDCK'

    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/cell/cell_MDCK_3.yaml')

    device = 'cuda:0'
    dimension = 2
    bc_pos, bc_dpos = choose_boundary_values('periodic')
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    lr = config.training.learning_rate_start
    batch_size = config.training.batch_size

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')

    model = Operator_smooth(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos, dimension=dimension)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    phi = torch.zeros(1, device=device)
    threshold = 0.05


    x_list = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/cell/cell_MDCK_3/full_vertice_list0.pt', map_location=device, weights_only=True)
    x_list = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/cell/cell_MDCK_3/x_list_0.pt', map_location=device, weights_only=True)

    for frame in trange(0,len(x_list)):

        x = x_list[frame]
        x[:,1:3] = x[:,1:3] / 1024

        tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dimension)
        mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid, torch.zeros((mgrid.shape[0], 2))), 1)
        mgrid = mgrid.to(device)

        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        data_id = torch.zeros((x.shape[0], 1), dtype=torch.int)
        dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

        pred = model(dataset, data_id=data_id, training=False, phi=phi)
        density = model.density.clone().detach()

        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - mgrid[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance > 0)).float() * 1
        edge_index_mgrid = adj_t.nonzero().t().contiguous()
        xp = torch.cat((mgrid, x[:, 0:2 * dimension + 1]), 0)
        edge_index_mgrid[0, :] = edge_index_mgrid[0, :] + mgrid.shape[0]
        edge_index_mgrid, _ = pyg_utils.remove_self_loops(edge_index_mgrid)

        dataset = data.Data(x=xp, pos=xp[:, 1:dimension + 1], edge_index=edge_index_mgrid)
        data_id = torch.zeros((xp.shape[0], 1), dtype=torch.int)
        pred_field = model(dataset, data_id=data_id, training=False, phi=phi, continuous_field=True, continuous_field_size=mgrid.shape)[0: mgrid.shape[0]]
        density_field = model.density[0: mgrid.shape[0]]

        # matplotlib.use("Qt5Agg")
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111)
        # plt.scatter(to_numpy(xp[0: mgrid.shape[0], 2:3]), to_numpy(xp[0: mgrid.shape[0], 1:2]), s=10, c=to_numpy(density_field))
        # Q = ax.quiver(to_numpy(x[:, 2]), to_numpy(x[:, 1]), -10*to_numpy(pred[:,1]), -10*to_numpy(pred[:,0]), color='w')
        # plt.show()

        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(2,4,1)
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c='w')
        pixel = 7020
        plt.scatter(mgrid[pixel, 2].detach().cpu().numpy(),
                    mgrid[pixel, 1].detach().cpu().numpy(), s=2, c='r')
        pos = torch.argwhere(edge_index_mgrid[1, :] == pixel).squeeze()
        if pos.numel()>0:
            plt.scatter(xp[edge_index_mgrid[0, pos], 2].detach().cpu().numpy(), xp[edge_index_mgrid[0, pos], 1].detach().cpu().numpy(), s=1,c='b')
        plt.xticks([])
        plt.yticks([])
        plt.title('pos', fontsize=8)
        ax = fig.add_subplot(2,4,5)
        plt.scatter(to_numpy(xp[0: mgrid.shape[0], 2:3]), to_numpy(xp[0: mgrid.shape[0], 1:2]), s=10, c=to_numpy(density_field))
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c='w')
        plt.xticks([])
        plt.yticks([])
        plt.title('density_field', fontsize=8)
        # ax = fig.add_subplot(2,4,6)
        # plt.scatter(to_numpy(xp[0: mgrid.shape[0], 2:3]), to_numpy(xp[0: mgrid.shape[0], 1:2]), s=10, c=to_numpy(pred_field[:,0]))
        # plt.xticks([])
        # plt.yticks([])
        # plt.title('density_field_x', fontsize=8)
        # ax = fig.add_subplot(2,4,7)
        # plt.scatter(to_numpy(xp[0: mgrid.shape[0], 2:3]), to_numpy(xp[0: mgrid.shape[0], 1:2]), s=10, c=to_numpy(pred_field[:,1]))
        # plt.xticks([])
        # plt.yticks([])
        # plt.title('density_field_y', fontsize=8)
        ax = fig.add_subplot(2,4,2)
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c=to_numpy(density))
        plt.xticks([])
        plt.yticks([])
        plt.title('density', fontsize=8)
        ax = fig.add_subplot(2,4,3)
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c=to_numpy(pred[:,0]))
        plt.xticks([])
        plt.yticks([])
        plt.title('density_y', fontsize=8)
        ax = fig.add_subplot(2,4,4)
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c=to_numpy(pred[:,1]))
        plt.xticks([])
        plt.yticks([])
        plt.title('density_x', fontsize=8)
        plt.show()
        plt.savefig(f'tmp/kernels_{frame}.tif')
        plt.close()



























