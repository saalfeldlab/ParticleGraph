import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
# from ParticleGraph.models.utils import reparameterize
# from ParticleGraph.models.Siren_Network import Siren
import torch.nn as nn
import numpy as np



def kernel_laplace(y, x):
    grad = kernel_gradient(y, x)
    return kernel_divergence(grad, x)


def kernel_divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def kernel_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class Interaction_Falling_Water_Smooth(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the acceleration of particles as a function of their relative distance and relative velocities.
    The interaction function is defined by a MLP self.lin_edge
    The particle embedding is defined by a table self.a

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=[]):

        super(Interaction_Falling_Water_Smooth, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
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
        self.smooth_radius = simulation_config.smooth_radius


        # self.kernel = MLP(input_size=1, output_size=1, nlayers=5, hidden_size=128, device=self.device)
        # self.kernel = Siren(in_features=1, out_features=1, hidden_features=256,
        #                   hidden_layers=3, outermost_linear=True, first_omega_0=30, hidden_omega_0=30.)
        # self.kernel.to(self.device)


        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)


        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)


        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))
        self.embedding_size = self.a.shape[1]

    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False, tasks=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.time_window == 0:
            if x.shape[0] <=self.embedding_size:
                particle_id = x[:, 0:1]
                embedding = self.a[data_id, to_numpy(particle_id), :].squeeze()
            else:
                embedding = torch.zeros((x.shape[0],2), device=device)

            pos = x[:, 1:self.dimension+1]
            d_pos = x[:, self.dimension+1:2*self.dimension+1]

        else:
            if x[0].shape[0] <= self.embedding_size:
                particle_id = x[0][:, 0:1]
                embedding = self.a[data_id, to_numpy(particle_id), :].squeeze()
            else:
                embedding = torch.zeros_like(x[0:self.dimension])
            x = torch.stack(x)
            pos = x[:, :, 1:self.dimension + 1]
            pos = pos.transpose(0, 1)
            pos = torch.reshape(pos, (pos.shape[0], pos.shape[1] * pos.shape[2]))

            d_pos = x[:, :, self.dimension + 1:2 * self.dimension + 1]
            d_pos = d_pos.transpose(0, 1)
            d_pos = torch.reshape(d_pos, (d_pos.shape[0], d_pos.shape[1] * d_pos.shape[2]))

        if training & (self.time_window_noise > 0):
            noise = torch.randn_like(pos) * self.time_window_noise
            pos = pos + noise

        pred = None
        for task in tasks:
            self.task = task
            match task:
                case 'density':
                    density_null = torch.zeros_like(pos[:, 0:1])
                    self.density = self.propagate(edge_index, pos=pos, d_pos=d_pos, embedding=embedding, density=density_null)
                    out = self.density
                case 'grad_velocity':
                    out = self.propagate(edge_index, pos=pos, d_pos=d_pos, embedding=embedding, density=self.density)
                case 'pred_smooth_particle':
                    pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, embedding=embedding, density=self.density)

        if pred != None:
            out = pred
            if self.update_type == 'mlp':
                if self.time_window == 0:
                    pos_p = pos
                else:
                    pos_p = (pos - pos[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
                out = self.lin_phi(torch.cat((pred, embedding, pos_p), dim=-1))

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, density_i, density_j):


        if self.time_window == 0:
            delta_pos = self.bc_dpos(pos_j - pos_i)
            match self.task:
                case 'density':
                    self.W_j, self.dW_j, self.d2W_j = self.W_d_W_d2W(delta_pos)
                    return torch.cat((self.W_j, self.dW_j, self.d2W_j), dim=1)
                case 'grad_velocity':
                    vx_x = self.dW_j[:, 0:1] * d_pos_j[:,0:1] / self.delta_t / density_j[:,0:1]
                    vy_y = self.dW_j[:, 1:2] * d_pos_j[:,1:2] / self.delta_t / density_j[:,0:1]
                    return torch.cat((vx_x, vy_y), dim=1)
                case 'pred_smooth_particle':
                    in_features = torch.cat((delta_pos, density_i, density_j, embedding_i, embedding_j), dim=-1)
                    out = self.lin_edge(in_features)
                    return out

        else:
            pos_i_p = (pos_i - pos_i[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
            pos_j_p = (pos_j - pos_i[:, 0:self.dimension].repeat(1, self.time_window))
            match self.task:
                case 'density':
                    W_j = self.W(pos_j_p[:, 0:2])
                    self.W_j = W_j
                    return W_j
                case 'smooth_particle':
                    in_features = torch.cat((pos_i_p, pos_j_p, density_i, density_j, embedding_i, embedding_j), dim=-1)
                    out = self.lin_edge(in_features)
                    return out
                case 'smooth_particle_dW':
                    dW_j = self.d_W(pos_j_p[:, 0:2])
                    in_features = torch.cat((pos_i_p, pos_j_p, density_i, density_j, embedding_i, embedding_j, dW_j), dim=-1)
                    out = self.lin_edge(in_features)
                    return out

    def W_d_W_d2W(self, x):

        d = torch.norm(x, dim=-1)
        d = d[:, None]
        W = 1 / (np.pi * self.smooth_radius ** 8) * (self.smooth_radius ** 2 - d ** 2) ** 3 / 6E3

        dW = [-3 * x[:,0] * (self.smooth_radius ** 2 - torch.sqrt(x[:,0] ** 2 + x[:,1] ** 2)) ** 2 / torch.sqrt(x[:,0] ** 2 + x[:,1] ** 2),
         -3 * x[:,1] * (self.smooth_radius ** 2 - torch.sqrt(x[:,0] ** 2 + x[:,1] ** 2)) ** 2 / torch.sqrt(x[:,0] ** 2 + x[:,1] ** 2)]
        dW = torch.stack(dW, dim=-1)
        dW = 1 / (np.pi * self.smooth_radius ** 8) * dW /6E3
        dW = torch.where(torch.isnan(dW), torch.zeros_like(dW), dW)
        d2W = torch.abs(dW) / d.repeat(1, 2)
        d2W = torch.where(torch.isnan(d2W), torch.zeros_like(d2W), d2W)

        return W, dW, d2W


    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)





if __name__ == '__main__':

    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import torch_geometric.data as data


    device = 'cuda:0'

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')

    bc_pos, bc_dpos = choose_boundary_values('no')
    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/test_smooth_particle.yaml')
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    smooth_radius = config.simulation.smooth_radius
    time_window = config.training.time_window

    x_list = np.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_falling_water_ramp_wall/x_list_2.npy')
    x_list = torch.tensor(x_list, dtype=torch.float32, device=device)

    tensors = tuple(dimension * [torch.linspace(0, 1, steps=512)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dimension)
    mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid, torch.zeros((mgrid.shape[0], 2))), 1)
    mgrid = mgrid.to(device)


    model = Interaction_Falling_Water_Smooth(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos, dimension=dimension)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    model.train()

    # x = mgrid

    it = 80
    x = x_list[it].squeeze()
    x = torch.cat((x, torch.zeros((x.shape[0], 5), device=device)), 1)
    data_id = torch.ones((x.shape[0], 1), dtype=torch.int)

    distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
    adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
    edge_index = adj_t.nonzero().t().contiguous()

    dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

    with torch.no_grad():
        pred = model(dataset, data_id=data_id, training=False, phi=torch.zeros(1, device=device), tasks=['density'])
    x[:, 6:9] = pred[:, 0:3]

    matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 2].detach().cpu().numpy(),
                x[:, 1].detach().cpu().numpy(), s=10, c=x[:, 6].detach().cpu().numpy(), vmin=0, vmax=0.5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.tight_layout()
    plt.show()

    with torch.no_grad():
        grad_v = model(dataset, data_id=data_id, training=False, phi=torch.zeros(1, device=device),
                               tasks=['grad_velocity'])
    x[:, 9:11] = grad_v

    matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 2].detach().cpu().numpy(),
                x[:, 1].detach().cpu().numpy(), s=10, c=x[:, 9].detach().cpu().numpy())
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.tight_layout()
    plt.show()

    matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 2].detach().cpu().numpy(),
                x[:, 1].detach().cpu().numpy(), s=10, c=x[:, 10].detach().cpu().numpy())
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.tight_layout()
    plt.show()

    matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 2].detach().cpu().numpy(),
                x[:, 1].detach().cpu().numpy(), s=10, c=(x[:, 9] + x[:, 10]).detach().cpu().numpy())
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.tight_layout()
    plt.show()




    for k in trange(0,len(x_list),4):

        x = x_list[k].squeeze()

        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - mgrid[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance <  smooth_radius ** 2) & (distance > 0)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        xp = torch.cat((mgrid, x[:, 0:2*dimension + 1]), 0)
        edge_index[0,:] = edge_index[0,:] + mgrid.shape[0]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        edges = edge_index

        matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x[:, 2].detach().cpu().numpy(),
                    x[:, 1].detach().cpu().numpy(), s=10, c='w')
        plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                    mgrid[:, 1].detach().cpu().numpy(), s=0.1, c='r')
        pixel = 5020
        plt.scatter(mgrid[pixel, 2].detach().cpu().numpy(),
                    mgrid[pixel, 1].detach().cpu().numpy(), s=40, c='g')
        pos = torch.argwhere(edges[1, :] == pixel).squeeze()
        plt.scatter(xp[edges[0, pos], 2].detach().cpu().numpy(), xp[edges[0, pos], 1].detach().cpu().numpy(), s=10,
                    c='b')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()

        dataset = data.Data(x=xp, pos=xp[:, 1:dimension + 1], edge_index=edge_index)
        with torch.no_grad():
            density = self.density(dataset, data_id=data_id, training=False, phi=torch.zeros(1, device=device), tasks=['density'])

        density = density[0:mgrid.shape[0]]
        density = torch.reshape(density, (512, 512))

        matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plt.imshow(to_numpy(density), cmap='viridis', vmin=0, vmax=1, extent=[0, 1, 0, 1])
        ax.invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.show()







        density[0:mgrid.shape[0]]
        density = torch.reshape(density, (512, 512))

        matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plt.imshow(to_numpy(density), cmap='viridis', vmin=0, vmax=1, extent=[0, 1, 0, 1])
        ax.invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.show()


        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

        pred = model(dataset, data_id=data_id, training=False, phi=torch.zeros(1, device=device))

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x[:, 2].detach().cpu().numpy(),
                    x[:, 1].detach().cpu().numpy(), s=10, c=model.density.detach().cpu().numpy(), vmin=0, vmax=0.75)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"tmp/particle_density_{k}.png")
        plt.close()

        



