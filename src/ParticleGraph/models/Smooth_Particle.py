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

# from ParticleGraph.utils import choose_boundary_values
# from ParticleGraph.config import ParticleGraphConfig


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

    def __init__(self, config=[], aggr_type='mean', bc_dpos=[], dimension=[], device='cuda:0'):
        super(Smooth_Particle, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.smooth_radius = config.training.smooth_radius
        self.smooth_function = config.training.smooth_function
        self.device = device

        tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dimension)
        mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid), 1)
        self.mgrid = mgrid.to(device)


    def forward(self, x=[], has_field=False):

        distance = torch.sum(self.bc_dpos(x[:, None, 1:self.dimension + 1] - x[None, :, 1:self.dimension + 1]) ** 2, dim=2)
        adj_t = ((distance <  self.smooth_radius ** 2) & (distance > 0)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        xp = x
        self.edge_index = edge_index


        if has_field:
            field = xp[:,6:7]
        else:
            field = torch.ones_like(xp[:,0:1])

        out = self.propagate(edge_index, pos=xp[:, 1:self.dimension+1], field=field)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, field_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)
        distance = torch.sqrt(distance_squared)

        d = self.W(distance, self.smooth_radius, self.smooth_function)

        return d[:,None]


    def W(self, d, s, function):

        match function:
            case 'gaussian':
                w_density = 1/(np.pi*s**2)*torch.exp(-d**2/s**2) / 1E3
            case triangular:
                w_density = 4/(np.pi*s**8)*(s**2-d**2)**3 / 1E3


        return w_density


class Smooth_Particle_Field(pyg.nn.MessagePassing):
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

    def __init__(self, config=[], aggr_type='mean', bc_dpos=[], dimension=[]):
        super(Smooth_Particle_Field, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.smooth_radius = config.training.smooth_radius
        self.smooth_function = config.training.smooth_function

        tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dimension)
        mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid), 1)
        self.mgrid = mgrid.to(device)


    def forward(self, x=[], has_field=False):


        distance = torch.sum(bc_dpos(x[:, None, 1:self.dimension + 1] - mgrid[None, :, 1:self.dimension + 1]) ** 2, dim=2)
        adj_t = ((distance <  self.smooth_radius ** 2) & (distance > 0)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        xp = torch.cat((mgrid, x[:, 0:self.dimension + 1]), 0)
        edge_index[0,:] = edge_index[0,:] + mgrid.shape[0]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        self.edge_index = edge_index

        if has_field:
            field = xp[:,6:7]
        else:
            field = torch.ones_like(xp[:,0:1])

        out = self.propagate(edge_index, pos=xp[:, 1:self.dimension+1], field=field)

        density = out[0:self.mgrid.shape[0],0]
        grad = out[0:self.mgrid.shape[0],1:]

        return density, grad


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, field_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)
        distance = torch.sqrt(distance_squared)

        d, d_d = self.W(distance, self.smooth_radius, self.smooth_function)
        pos = torch.argwhere(distance==0)
        if pos.numel() > 0:
            d[pos] = 0
            d_d[pos] = 0

        d_d = d_d/distance
        d_d = d_d[:,None].repeat(1,2) * self.bc_dpos(pos_j - pos_i)

        t = torch.cat((d[:,None], d_d), 1)

        return t

    def W(self, d, s, function):

        match function:
            case 'gaussian':
                w_density = 1/(np.pi*s**2)*torch.exp(-d**2/s**2) / 1E3
                wp_density = 2/(np.pi*s**4)*d*torch.exp(-d**2/s**2) / 1E6
            case triangular:
                w_density = 4/(np.pi*s**8)*(s**2-d**2)**3 / 1E3
                wp_density = 24/(np.pi*s**8)*d*(s**2-d**2)**2 / 1E6


        return w_density, wp_density




if __name__ == '__main__':

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

    model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)
    # model_density = Smooth_Particle_Field(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)
    mgrid = model_density.mgrid.clone().detach()


    for k in trange(0,len(x_list),10):

        x = x_list[k].squeeze()
        density = model_density(x=x, has_field=False)
        # density, grad = model_density(x=x, has_field=False)


        bViz = 2
        if (bViz == 1):
            xp = torch.cat((mgrid, x[:, 0:dimension + 1]), 0)
            edges = model_density.edge_index

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(x[:, 2].detach().cpu().numpy(),
                        x[:, 1].detach().cpu().numpy(), s=10, c='w')
            plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                        mgrid[:, 1].detach().cpu().numpy(), s=1, c='r')
            pixel = 5020
            plt.scatter(mgrid[pixel, 2].detach().cpu().numpy(),
                        mgrid[pixel, 1].detach().cpu().numpy(), s=40, c='g')
            pos = torch.argwhere(edges[1,:] == pixel).squeeze()
            plt.scatter(xp[edges[0,pos], 2].detach().cpu().numpy(), xp[edges[0,pos], 1].detach().cpu().numpy(), s=10, c='b')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/field_grid_{k}.png")
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                        mgrid[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/field_density_{k}.png")
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                        mgrid[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
            idx = torch.arange(0, mgrid.shape[0], 1)
            idx = torch.randperm(idx.shape[0])
            idx = idx[0:2000]
            for n in idx:
                plt.arrow(mgrid[n, 2].detach().cpu().numpy(), mgrid[n, 1].detach().cpu().numpy(),
                          grad[n, 1].detach().cpu().numpy(), grad[n, 0].detach().cpu().numpy(), head_width=0.005, color='w', alpha=0.5)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/field_grad_{k}.png")
            plt.close()

        if (bViz == 2):

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(x[:, 2].detach().cpu().numpy(),
                        x[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/particle_density_{k}.png")
            plt.close()




