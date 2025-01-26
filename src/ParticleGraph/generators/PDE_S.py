import umap
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import *


def arbitrary_gaussian_grad_laplace (mgrid, n_gaussian, device):

    mgrid.requires_grad = True
    x = mgrid[:, 0]
    y = mgrid[:, 1]
    size = np.sqrt(mgrid.shape[0]).astype(int)

    u = torch.zeros(mgrid.shape[0], device=device)

    for k in range(n_gaussian):
        x0 = np.random.uniform(0, 1)
        y0 = np.random.uniform(0, 1)
        a = np.random.uniform(0, 1)
        sigma = np.random.uniform(0.05, 0.1)
        u = u + a * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    grad_autograd = density_gradient(u, mgrid)
    laplacian_autograd = density_laplace(u, mgrid)

    return u, grad_autograd, laplacian_autograd

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131)
    plt.imshow(to_numpy(u).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('u(x,y)')
    ax = fig.add_subplot(132)
    plt.imshow(to_numpy(grad_autograd[:,0]).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('Grad_x(u(x,y)) autograd')
    ax = fig.add_subplot(133)
    plt.imshow(to_numpy(laplacian_autograd).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('Laplacian(u(x,y)) autograd')
    plt.show()


class PDE_S(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of fluidic particles.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2, model_density=[]):
        super(PDE_S, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_mp_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
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
        self.kernel_var = self.max_radius ** 2
        self.kernel_norm = np.pi * self.kernel_var * (1 - np.exp(-self.max_radius ** 2/ self.kernel_var))

        if self.update_type == 'pre_mlp':
            self.pre_lin_edge = MLP(input_size=self.pre_input_size, output_size=self.pre_output_size, nlayers=self.pre_n_layers,
                                hidden_size=self.pre_hidden_dim, device=self.device)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if 'mlp' in self.update_type:
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))


    def forward(self, data, continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_type = to_numpy(x[:, 1 + 2*self.dimension])
        pos = x[:, 1:self.dimension+1]
        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        if continuous_field:
            self.mode = 'kernel'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=torch.zeros_like(x[:, 0:1]))
            density = torch.zeros_like(x[:, 0:1])
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=density)
        else:
            self.mode = 'kernel'
            self.density = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=torch.zeros_like(x[:, 0:1]))
            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=self.density)

        # out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
        # out = torch.where(torch.isnan(out), torch.zeros_like(out), out)

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, field_i, field_j, particle_type_i, density_i, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'pre_mlp':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            density_kernel = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:,None] / self.kernel_norm
            first_kernel = torch.exp(-4*(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:, None] / self.kernel_norm
            kernel_modified = first_kernel * self.pre_lin_edge(mgrid)

            grad_autograd = -density_gradient(kernel_modified, mgrid)
            laplace_autograd = density_laplace(kernel_modified, mgrid)

            self.kernel_operators = torch.cat((kernel_modified, grad_autograd, laplace_autograd), dim=-1)

            return density_kernel

        elif self.mode == 'message_passing':

            if 'laplacian' in self.field_type:
                laplacian = field_j * self.kernel_operators[:, 3:4] / density_j
                out = laplacian


            return out


if __name__ == '__main__':


    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import torch_geometric.data as data
    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import matplotlib.pyplot as plt
    from utils import density_gradient, density_laplace
    from ParticleGraph.models.MLP import MLP

    device = 'cuda:0'
    dimension = 2
    bc_pos, bc_dpos = choose_boundary_values('no')
    max_radius = 0.05
    min_radius = 0
    lr = 1E-4

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')


    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/wave/wave_smooth_particle.yaml')

    params = config.simulation.params
    dimension = config.simulation.dimension
    delta_t = config.simulation.delta_t
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    aggr_type = config.graph_model.aggr_type


    tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
    x = torch.stack(torch.meshgrid(*tensors), dim=-1)
    x = x.reshape(-1, dimension)
    x = torch.cat((torch.arange(x.shape[0])[:,None], x, torch.zeros((x.shape[0], 9))), 1)
    x = x.to(device)
    x.requires_grad = False
    size = np.sqrt(x.shape[0]).astype(int)
    x0 = x

    model = PDE_S(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos, dimension=dimension)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()


    for epoch in trange(0, 5000):

        optimizer.zero_grad()

        x = x0.clone().detach() + 0.05 * torch.randn_like(x0)
        # x = x[torch.randperm(x.size(0))[:int(0.5 * x.size(0))]] # removal of 10%

        u, grad_u, laplace_u = arbitrary_gaussian_grad_laplace(mgrid = x[:,1:3], n_gaussian = 5, device=device)
        # L_u = grad_u.clone().detach()
        L_u = laplace_u.clone().detach()
        x[:, 6:7] = u[:, None].clone().detach()

        discrete_pos = torch.argwhere((u >= threshold) | (u <= -threshold))
        x = x[discrete_pos].squeeze()
        L_u = L_u[discrete_pos].squeeze()

        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        data_id = torch.ones((x.shape[0], 1), dtype=torch.int)
        dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

        pred = model(dataset, data_id=data_id, training=False, phi=phi)
        loss = (pred-L_u).norm(2)

        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:

            u = u[discrete_pos]
            grad_u = grad_u[discrete_pos]
            laplace_u = laplace_u[discrete_pos]

            print(epoch, loss)

            # matplotlib.use("Qt5Agg")
            fig = plt.figure(figsize=(12,3))
            ax = fig.add_subplot(141)
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c='w')
            ax.invert_yaxis()
            plt.title('density')
            ax = fig.add_subplot(142)
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(u))
            ax.invert_yaxis()
            plt.title('u')
            ax = fig.add_subplot(143)
            # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(L_u[:,0]))
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(L_u))
            ax.invert_yaxis()
            plt.title('true L_u')
            ax = fig.add_subplot(144)
            # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(pred))
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(pred))
            ax.invert_yaxis()
            plt.title('pred L_u')
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'tmp/learning_{epoch}.tif')
            plt.close()

            matplotlib.use("Qt5Agg")
            fig = plt.figure(figsize=(12,3))
            ax = fig.add_subplot(141)
            plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1, c=to_numpy(model.kernel_operators[:, 0:1]))
            plt.title('kernel')
            ax = fig.add_subplot(142)
            plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1, c=to_numpy(model.kernel_operators[:, 1:2]))
            plt.title('grad_x')
            ax = fig.add_subplot(143)
            plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1, c=to_numpy(model.kernel_operators[:, 2:3]))
            plt.title('grad_y')
            ax = fig.add_subplot(144)
            plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1, c=to_numpy(model.kernel_operators[:, 3:4]))
            plt.title('laplace')
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'tmp/kernels_{epoch}.tif')
            plt.close()







