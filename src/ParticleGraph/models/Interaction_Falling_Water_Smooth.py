import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize
from ParticleGraph.models.Siren_Network import Siren



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


        # self.kernel = MLP(input_size=1, output_size=1, nlayers=5, hidden_size=128, device=self.device)

        self.kernel = Siren(in_features=1, out_features=1, hidden_features=256,
                          hidden_layers=3, outermost_linear=True, first_omega_0=30, hidden_omega_0=30.)
        self.kernel.to(self.device)


        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)


        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)


        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))



    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False):

        x, edge_index = data.x, data.edge_index

        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.time_window == 0:
            particle_id = x[:, 0:1]
            embedding = self.a[data_id, to_numpy(particle_id), :].squeeze()
            pos = x[:, 1:self.dimension+1]

        else:
            particle_id = x[0][:, 0:1]
            embedding = self.a[data_id, to_numpy(particle_id), :].squeeze()
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
            pos_p = (pos - pos[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
            out = self.lin_phi(torch.cat((pred, embedding, pos_p), dim=-1))
        else:
            out = pred

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, embedding_i, embedding_j, density_i, density_j):

        pos_i_p = (pos_i - pos_i[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
        pos_j_p = (pos_j - pos_i[:, 0:self.dimension].repeat(1, self.time_window))

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

        coords = x.clone().detach().requires_grad_(True)
        d = torch.norm(x, dim=-1) / self.smooth_radius
        d = d[None, :, None]

        W = 1 / (np.pi * s ** 8) * (self.smooth_radius ** 2 - d ** 2) ** 3 / 6E3
        return W


        # W = self.kernel.net(d)
        # return W[0, :, :]

    def W_d_W(self, x):

        coords = x.clone().detach().requires_grad_(True)
        d = torch.norm(coords, dim=-1) / self.smooth_radius
        d = d[None, :, None]

        W = self.kernel.net(d)
        dW = kernel_gradient(W, coords)

        return W[0, :, :], dW

    def W_d_W_d2W(self, x):

        coords = x.clone().detach().requires_grad_(True)
        d = torch.norm(coords, dim=-1) / self.smooth_radius
        d = d[None, :, None]
        W = self.kernel.net(d)
        dW = kernel_gradient(W, coords)

        d2W = []
        for i in range(dW.shape[-1]):
            d2W.append(torch.autograd.grad(dW[..., i], coords, torch.ones_like(dW[..., i]), create_graph=True)[0][..., i:i + 1])

        return W[0, :, :], dW, d2W


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

    tensors = tuple(dimension * [torch.linspace(-1, 1, steps=200)*smooth_radius])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dimension)
    mgrid = mgrid.to(device)



    model = Interaction_Falling_Water_Smooth(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    model.train()



    x = mgrid

    for smooth_radius in [0.1, 0.2, 0.3, 0.4, 0.5]:

        config.simulation.smooth_radius = smooth_radius
        model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)

        density = model_density(x=x, has_field=False)

        print(smooth_radius, density[4550])

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x[:, 2].detach().cpu().numpy(),
                    x[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()

        


    for k in trange(0,len(x_list),10):

        x = x_list[k].squeeze()
        density = model_density(x=x, has_field=False)

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x[:, 2].detach().cpu().numpy(),
                    x[:, 1].detach().cpu().numpy(), s=10, c=density[:,0].detach().cpu().numpy(), vmin=0, vmax=1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"tmp/particle_density_{k}.png")
        plt.close()






    num_epochs = 5000
    loss_list=[]
    for epoch in trange(num_epochs):

        input = mgrid + torch.randn_like(mgrid) * smooth_radius
        y = torch.exp(-(input[:, 0] ** 2 + input[:, 1] ** 2) / (0.25 * smooth_radius ** 2))
        y = y.clone().detach()
        y = y[:, None]

        optimizer.zero_grad()
        W = model.W(input)
        loss = (W-y).norm(2)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.item())

    fig = plt.figure(figsize=(8, 8))
    plt.plot(np.log(loss_list))

    W, dW, d2W = model.W_d_W_d2W(mgrid)
    y = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (0.25 * smooth_radius ** 2))
    fig = plt.figure(figsize=(8, 8))
    plt.plot(y[20000:20200].cpu().numpy())
    plt.plot(W[20000:20200].detach().cpu().numpy())
    plt.plot(dW[20000:20200,1].detach().cpu().numpy()/40)
    d2W_im = torch.reshape(d2W[0], (200,200))
    plt.plot(d2W_im[100,:].detach().cpu().numpy()/16000)

    dW_im = torch.reshape(dW[:,0], (200,200))
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(dW_im.detach().cpu().numpy())
    d2W_im = torch.reshape(d2W[0], (200,200))
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(d2W_im.detach().cpu().numpy())




















    bc_pos, bc_dpos = choose_boundary_values('no')
    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/test_smooth_particle.yaml')
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    smooth_radius = config.simulation.smooth_radius

    model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)
    model_density.train()

    tensors = tuple(dimension * [torch.linspace(-1, 1, steps=200)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dimension)
    mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid), 1)
    mgrid = mgrid.to(device)

    coords = mgrid[:, 1:3].clone().detach().requires_grad_(True)

    y = torch.exp(-(coords[:,0] ** 2 + coords[:,1] ** 2) / 0.25)
    y = y.clone().detach()
    y = y[:,None]

    # Initialize the model, loss function, and optimizer
    model = model_density.lin_rho
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 1000
    for epoch in trange(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(coords)
        # model_output = gradient(output, coords)[:,0]
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0].detach().cpu().numpy(), coords[:, 1].detach().cpu().numpy(), c=y.cpu().numpy(), label='True Function')

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0].detach().cpu().numpy(), coords[:, 1].detach().cpu().numpy(), c=output.detach().cpu().numpy(), label='True Function')


    # Create the dataset

    s = torch.tensor([smooth_radius], device=device)
    x = torch.linspace(-1, 1, 1000).view(-1, 1).to(device)
    y = torch.exp(-x ** 2 / 0.25)

    coords = x.clone().detach().requires_grad_(True)

    # Initialize the model, loss function, and optimizer
    model = model_density.lin_rho
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 10000
    for epoch in trange(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(coords)
        model_output = gradient(output, coords)
        loss = criterion(model_output, y)
        loss.backward()
        optimizer.step()

    pred = model(coords)
    model_output = gradient(pred, coords)
    plt.figure(figsize=(8, 8))
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), label='True Function')
    plt.scatter(x.cpu().numpy(), pred.detach().cpu().numpy(), label='MLP Approximation')
    plt.scatter(x.cpu().numpy(), model_output.detach().cpu().numpy(), label='grad MLP Approximation')
    plt.legend()
    plt.show()











    s = torch.tensor([smooth_radius], device=device)

    optim = torch.optim.Adam(lr=1e-1, params=model_density.parameters())

    sample = torch.rand(100, 1, device=device)
    pred = model_density.lin_rho(sample.clone().detach())
    y = (s ** 2 * torch.ones_like(sample)[:, None] - sample[:, None] ** 2) ** 2
    y = y.clone().detach()
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(sample.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.scatter(sample.detach().cpu().numpy(), pred.detach().cpu().numpy())

    for step in trange(10000):

        optim.zero_grad()
        sample = torch.rand(5, 1, device=device)
        pred = model_density.lin_rho(sample.clone().detach())
        y = (s**2*torch.ones_like(sample)[:,None] - sample[:,None]**2)**2
        y = y.clone().detach()
        loss = ((pred - y) ** 2).norm(2)
        loss.backward()
        optim.step()

    print(step, loss)

    sample = torch.rand(100, 1, device=device)
    pred = model_density.lin_rho(sample.clone().detach())
    y = (s ** 2 * torch.ones_like(sample)[:, None] - sample[:, None] ** 2) ** 2
    y = y.clone().detach()
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(sample.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.scatter(sample.detach().cpu().numpy(), pred.detach().cpu().numpy())






    x = mgrid

    for smooth_radius in [0.1, 0.2, 0.3, 0.4, 0.5]:

        config.simulation.smooth_radius = smooth_radius
        model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)

        density = model_density(x=x, has_field=False)

        print(smooth_radius, density[4550])

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x[:, 2].detach().cpu().numpy(),
                    x[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()



