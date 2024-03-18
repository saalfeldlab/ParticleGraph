import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_N(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        
    """

    def __init__(self, aggr_type=[], p=[], bc_dpos=[]):
        super(PDE_N, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.bc_dpos = bc_dpos

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type, :]
        b = parameters[:, 0]
        c = parameters[:, 1]

        u = x[:, 6:7]

        msg = self.propagate(edge_index, u=u, edge_attr=edge_attr)

        du = -b*u + c*torch.tanh(u) + msg

        return du

    def message(self, u_j, edge_attr):
        return edge_attr[:,None] * torch.tanh(u_j)


    def psi(self, r, p):
        return r * p
