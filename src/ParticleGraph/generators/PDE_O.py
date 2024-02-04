import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_O(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the speed of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], bc_diff=[], p=[], beta=[]):
        super(PDE_O, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_diff = bc_diff
        self.p = p
        self.beta = beta

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_type = to_numpy(x[:, 5])
        p = self.p[particle_type]
        p = p[:, None]

        pos = x[:, 1:3]
        pos_0 = x[:,6:8]
        w = self.beta * p

        d_pos = torch.zeros_like(pos)
        d_pos[:,0:1] = -w.repeat(1,1)*self.bc_diff(pos[:,1:2] - pos_0[:,1:2])
        d_pos[:,1:2] = w.repeat(1,1)*self.bc_diff(pos[:,0:1] - pos_0[:,0:1])

        # d_pos_p = self.propagate(edge_index, d_pos=d_pos)
        # d_pos = d_pos + d_pos_p

        return d_pos

    def message(self, d_pos_j):

        return d_pos_j

    def psi(self, r, p):
        return r * p
