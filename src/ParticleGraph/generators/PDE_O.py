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

    def __init__(self, aggr_type=[], bc_diff=[], p=[], beta=[], rr=[]):
        super(PDE_O, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_diff = bc_diff
        self.p = p
        self.beta = beta
        self.rr = rr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # particle_type = to_numpy(x[:, 5])
        # p = self.p[particle_type]
        # p = p[:, None]

        degree = pyg_utils.degree(edge_index[0], x.size(0), dtype=x.dtype)

        theta = data.x[:,8:9]
        d_theta0 = data.x[:,10:11]

        d_theta = self.propagate(edge_index, theta=theta)

        d_theta = d_theta0 +  5e-4 * d_theta
        return d_theta

    def message(self, theta_i, theta_j):


        return torch.sin(theta_j - theta_i)

    def psi(self, r, p):
        return r * p
