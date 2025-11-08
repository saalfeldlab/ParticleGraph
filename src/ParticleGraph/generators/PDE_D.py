
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import torch
from ParticleGraph.utils import *

class PDE_D(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=[], p=[], bc_dpos=[], dimension=2):
        super(PDE_D, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.bc_dpos = bc_dpos
        self.dimension = dimension

    def forward(self, data=[], has_field=False, k=0):
        x, edge_index = data.x, data.edge_index

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,0:1])

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = x[:, 1 + 2*self.dimension].long()
        parameters = self.p[to_numpy(particle_type),:]
        d_pos = self.propagate(edge_index, pos=x[:, 1:self.dimension+1], particle_type=particle_type[:,None], parameters=parameters.squeeze(), field=field, )

        return d_pos

    def message(self, pos_i, pos_j, particle_type_i, parameters_i, field_j):


        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)
        distance = torch.sqrt(distance_squared)

        d_pos = torch.zeros_like(pos_i)
        d_pos = field_i

        return d_pos

    def psi(self, r, p, func='arbitrary'):

        return r


