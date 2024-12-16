# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file

import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import matplotlib
from ParticleGraph.models.MLP import MLP




class Smooth_Particle(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the speed of the particles (dimension 2)
    """

    def __init__(self, config=[], aggr_type='add', bc_dpos=[], dimension=[], device='cuda:0'):
        super(Smooth_Particle, self).__init__(aggr='add')  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.smooth_radius = config.simulation.smooth_radius
        self.smooth_function = config.simulation.smooth_function
        self.device = device

        self.K = nn.Parameter(torch.tensor([1.0], device=self.device, requires_grad=True, dtype=torch.float32))
        self.rho_0 = nn.Parameter(torch.tensor([0.5], device=self.device, requires_grad=True, dtype=torch.float32))
        self.lin_rho = MLP(input_size=1, output_size=1, nlayers=5, hidden_size=128, device=self.device)


    def forward(self, x=[], has_field=False):

        distance = torch.sum(self.bc_dpos(x[:, None, 1:self.dimension + 1] - x[None, :, 1:self.dimension + 1]) ** 2, dim=2)
        adj_t = ((distance <  self.smooth_radius ** 2) & (distance >= 0)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        self.edge_index = edge_index

        rho = self.propagate(edge_index, pos=x[:, 1:self.dimension+1])
        p = self.K * (rho - self.rho_0)

        # Compute the pressure at each particle's position (κ is the isentropic exponent, ρ₀ is a base density)
        # pᵢ = κ * (ρ − ρ₀)
        # https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/smoothed_particle_hydrodynamics_simple_python.py


        out = torch.cat((rho, p), 1)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)
        w = self.W(distance_squared, self.smooth_radius, self.smooth_function)

        return w[:,None]


    def W(self, distance_squared, s, function):

        match function:
            case 'Gaussian':
                w_density = 1 / (np.pi*s**8)*(s**2-distance_squared)**3 / 6E3
            case 'Triangular':
                w_density = 1 / (np.pi*s**8)*(s**2-distance_squared)**3 / 6E3
            case 'Nine':
                w_density = 315 / (64 * np.pi * s ** 9) * (s ** 2 - distance_squared) ** 3
            case 'mlp':
                w_density = self.lin_rho(distance_squared)
                w_density = w_density.squeeze()
            # ρᵢ = (315 M) / (64 π L⁹) ∑ⱼ(L² − dᵢⱼ²)³
            # https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/smoothed_particle_hydrodynamics_simple_python.py

        return w_density






    # r = torch.linspace(0, self.smooth_radius, 100, device=self.device)
    # fig = plt.figure(figsize=(8, 8))
    # k = self.W(r**2, self.smooth_radius, self.smooth_function)
    # plt.scatter(r.detach().cpu().numpy(), k.detach().cpu().numpy(), c='w')
    #
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(x[:, 2].detach().cpu().numpy(),
    #             x[:, 1].detach().cpu().numpy(), s=10, c='w')
    # pos = torch.argwhere(edge_index[0,:]==50)
    # if pos.numel()>0:
    #     print(pos.numel())
    #     plt.scatter(x[edge_index[1,pos], 2].detach().cpu().numpy(),
    #                 x[edge_index[1,pos], 1].detach().cpu().numpy(), s=10, c='r')
    #     t = torch.sum(self.W(distance[edge_index[1,pos],edge_index[0,pos]], self.smooth_radius, self.smooth_function))
    #     print(t)
    #
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(x[:, 2].detach().cpu().numpy(),
    #             x[:, 1].detach().cpu().numpy(), s=10, c='w')
    # pos = torch.argwhere(edge_index[0,:]==479)
    # if pos.numel()>0:
    #     print(pos.numel())
    #     plt.scatter(x[edge_index[1,pos], 2].detach().cpu().numpy(),
    #                 x[edge_index[1,pos], 1].detach().cpu().numpy(), s=10, c='r')
    #     t = torch.sum(self.W(distance[edge_index[0,pos],edge_index[1,pos]], self.smooth_radius, self.smooth_function))
    #     print(t)




if __name__ == '__main__':

    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import torch.nn as nn
    import torch.optim as optim


    def laplace(y, x):
        grad = gradient(y, x)
        return divergence(grad, x)


    def divergence(y, x):
        div = 0.
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
        return div


    def gradient(y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad



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

    model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)
    model_density.train()

    tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dimension)
    mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid), 1)
    mgrid = mgrid.to(device)

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









