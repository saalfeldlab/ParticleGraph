import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize


def kernel_laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


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

    def __init__(self, config, device, aggr_type=None, bc_dpos=None):

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


        self.kernel = MLP(input_size=1, output_size=1, nlayers=5, hidden_size=128, device=self.device)


        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)


        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)


        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))




    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False):

        self.data_id = data_id

        x, edge_index = data.x, data.edge_index

        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.time_window == 0:
            particle_id = x[:, 0:1]
            embedding = self.a[self.data_id, to_numpy(particle_id), :].squeeze()
            pos = x[:, 1:self.dimension+1]
            d_pos = x[:, self.dimension+1:1+2*self.dimension]

            pred = self.propagate(edge_index,pos=pos, embedding=embedding)

        else:
            particle_id = x[0][:, 0:1]
            embedding = self.a[self.data_id, to_numpy(particle_id), :].squeeze()
            x = torch.stack(x)
            pos = x[:, :, 1:self.dimension + 1]
            pos = pos.transpose(0, 1)
            pos = torch.reshape(pos, (pos.shape[0], pos.shape[1] * pos.shape[2]))

            if training & (self.time_window_noise > 0):
                noise = torch.randn_like(pos) * self.time_window_noise
                pos = pos + noise

            density_null = torch.zeros_like(pos[:, 0:1])
            self.mode = 'density'
            density = self.propagate(edge_index, pos=pos, embedding=embedding, density=density_null)
            self.mode = 'smooth_particle'
            pred = self.propagate(edge_index, pos=pos, embedding=embedding, density=density)

            if self.update_type == 'mlp':
                pos_p = (pos - pos[:, 0:self.dimension].repeat(1, 4))[:, 2:]
                out = self.lin_phi(torch.cat((pred, embedding, pos_p), dim=-1))
            else:
                out = pred

            return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, embedding_i, embedding_j, density_i, density_j):
        # distance normalized by the max radius

        # delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        # r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        # k_ij = self.kernel(torch.cat((r[:, None], delta_pos[:,0:2]), dim=-1))

        pos_i_p = (pos_i - pos_i[:, 0:2].repeat(1, 4))[:, 2:]
        pos_j_p = (pos_j - pos_i[:, 0:2].repeat(1, 4))



        match self.mode:
            case 'density':
                W_j = self.W(pos_j_p[:, 0:2])
                return W_j
            case 'smooth_particle':
                W_j, dW_j = self.W_d_W(pos_j_p[:, 0:2])
                in_features = torch.cat((pos_i_p, pos_j_p, density_i, density_j,  W_j, dW_j, embedding_i, embedding_j), dim=-1)
                out = self.lin_edge(in_features)
                return out

    def W(self, x):

        d = torch.norm(x, dim=-1) / self.smooth_radius
        d = d[:, None]
        W = self.kernel(d)

        return W

    def W_d_W(self, x):

        coords = x.clone().detach().requires_grad_(True)
        d = torch.norm(coords, dim=-1) / self.smooth_radius
        d = d[:, None]
        W = self.kernel(d)
        dW = kernel_gradient(W, coords)

        return W, dW


    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)





if __name__ == '__main__':

    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import torch.nn as nn
    import torch.optim as optim


    device = 'cuda:0'
    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    x_list = np.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_falling_water_ramp/x_list_2.npy')
    x_list = torch.tensor(x_list, dtype=torch.float32, device=device)

    plt.style.use('dark_background')

    bc_pos, bc_dpos = choose_boundary_values('no')
    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/test_smooth_particle.yaml')
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    smooth_radius = config.simulation.smooth_radius

    model_density = Interaction_Falling_Water_Smooth(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos)
    model_density.train()

    tensors = tuple(dimension * [torch.linspace(-1, 1, steps=200)*smooth_radius])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dimension)
    mgrid = mgrid.to(device)

    y = torch.exp(-(mgrid[:,0] ** 2 + mgrid[:,1] ** 2) / (0.25 * smooth_radius ** 2))
    y = y.clone().detach()
    y = y[:,None]

    # Initialize the model, loss function, and optimizer
    model = model_density
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10000
    for epoch in trange(num_epochs):
        model.train()
        optimizer.zero_grad()
        W = model.W(mgrid)
        loss = criterion(W, y)
        loss.backward()
        optimizer.step()

    W,dW = model.W_d_W(mgrid)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(y[20000:20200].cpu().numpy())
    plt.plot(W[20000:20200].detach().cpu().numpy())
    plt.plot(dW[20000:20200,1].detach().cpu().numpy()/40)


    fig = plt.figure(figsize=(8, 8))
    plt.scatter(mgrid[:, 0].detach().cpu().numpy(), mgrid[:, 1].detach().cpu().numpy(), c=y.cpu().numpy(), label='True Function')
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(mgrid[:, 0].detach().cpu().numpy(), mgrid[:, 1].detach().cpu().numpy(), c=W.detach().cpu().numpy(), label='True Function')


    fig = plt.figure(figsize=(8, 8))
    plt.scatter(mgrid[:, 0].detach().cpu().numpy(), mgrid[:, 1].detach().cpu().numpy(), c=dW[0].detach().cpu().numpy(), label='True Function')
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(mgrid[:, 0].detach().cpu().numpy(), mgrid[:, 1].detach().cpu().numpy(), c=dW[1].detach().cpu().numpy(), label='True Function')


