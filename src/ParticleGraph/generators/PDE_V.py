import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import matplotlib.pyplot as plt
import matplotlib
import torch
from ParticleGraph.utils import *

class PDE_V(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute particle velocity as a function of relative position and attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    d_pos : float
        the velocity of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], sigma=[], bc_dpos=[], dimension=2):
        super(PDE_V, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.sigma = sigma
        self.bc_dpos = bc_dpos
        self.dimension = dimension

    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,0:1])

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = x[:, 1 + 2*self.dimension][:,None]
        parameters = self.p.repeat(x.shape[0], 1)
        d_pos = self.propagate(edge_index, pos=x[:, 1:self.dimension+1], parameters=parameters, particle_type=particle_type)

        return d_pos


    def message(self, pos_i, pos_j, parameters_i, particle_type_i, particle_type_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # squared distance
        f = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2))
               - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
        d_pos = f[:, None] * self.bc_dpos(pos_j - pos_i)
        pos = torch.argwhere(particle_type_i == particle_type_j)
        if pos.numel()>0:
            d_pos[pos] = -d_pos[pos]*0.5

        return d_pos / 2

        matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(to_numpy(torch.sqrt(distance_squared)), to_numpy(d_pos[:,0]))
        plt.scatter(to_numpy(torch.sqrt(distance_squared)), to_numpy(d_pos[:,1]))
        plt.show()

    def psi(self, r, p):
        return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2))
                    - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * self.sigma ** 2)))
