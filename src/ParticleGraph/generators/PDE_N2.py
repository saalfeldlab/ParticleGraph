import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_N2(pyg.nn.MessagePassing):
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
        super(PDE_N2, self).__init__(aggr=aggr_type)

        self.p = p
        self.bc_dpos = bc_dpos

    def forward(self, data=[], return_all=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type]
        s = parameters[:, 0:1]
        g = parameters[:, 1:2]

        u = x[:, 6:7]

        msg = self.propagate(edge_index, u=u, edge_attr=edge_attr)

        du = -u + s * torch.tanh(u) + g * torch.relu(msg)

        if return_all:
            return du, s * torch.tanh(u), g * torch.relu(msg)
        else:
            return du

    def message(self, u_j, edge_attr):

        self.activation = torch.tanh(u_j)
        self.u_j = u_j

        return edge_attr[:,None] * torch.tanh(u_j)




    def psi(self, r, p):
        return r * p
