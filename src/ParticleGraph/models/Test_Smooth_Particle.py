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
        self.smooth_function = config.training.smooth_function
        self.device = device

        # self.smooth_radius = config.training.smooth_radius
        self.smooth_radius = nn.Parameter(torch.tensor(config.training.smooth_radius, device=self.device,requires_grad=True, dtype=torch.float32))

        tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dimension)
        mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid), 1)
        self.mgrid = mgrid.to(device)


    def forward(self, x=[], has_field=False):


        distance = torch.sum(self.bc_dpos(x[:, None, 0:self.dimension] - x[None, :, 0:self.dimension]) ** 2, dim=2)
        adj_t = ((distance <  self.smooth_radius ** 2) & (distance >= 0)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        self.edge_index = edge_index

        out = self.propagate(edge_index, pos=x[:, 0:self.dimension])

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)
        w = self.W(distance_squared, self.smooth_radius, self.smooth_function)

        return w[:,None]


    def W(self, distance_squared, s, function):

        match function:
            case 'gaussian':
                w_density = 1/(np.pi*s**8)*(s**2-distance_squared)**3 / 6E3
            case triangular:
                w_density = 1/(np.pi*s**8)*(s**2-distance_squared)**3 / 6E3


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



def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad




if __name__ == '__main__':

    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import torch.nn as nn

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
    smooth_radius  = config.training.smooth_radius

    config.training.smooth_radius = smooth_radius
    model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)
    mgrid = model_density.mgrid.clone().detach()


    for k in trange(0,len(x_list),10):

        x = x_list[k].squeeze()

        coords = x[:,1:3].clone().detach().requires_grad_(True)

        density = model_density(x=coords, has_field=False)

        density_grad = gradient(density.squeeze(), coords)

        fig = plt.figure(figsize=(8, 8))

        plt.scatter(x[:, 2].detach().cpu().numpy(),
                    x[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)



        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"tmp/particle_density_{k}.png")
        plt.close()

    x = mgrid

    for smooth_radius in [0.1, 0.2, 0.3, 0.4]:

        config.training.smooth_radius = smooth_radius
        model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)

        density = model_density(x=x, has_field=False)

        print(smooth_radius, density[4550])

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x[:, 2].detach().cpu().numpy(),
                    x[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
        plt.scatter(x[4550, 2].detach().cpu().numpy(),
                    x[4550, 1].detach().cpu().numpy(), s=10, c='r')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()

        plt.savefig(f"tmp/particle_density_{smooth_radius}.png")










