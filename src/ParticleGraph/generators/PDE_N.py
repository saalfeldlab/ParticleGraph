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
        parameters = self.p[particle_type]
        b = parameters[:, 0]
        c = parameters[:, 1]

        u = x[:, 6:7]

        indices = torch.arange(0, x.size(0),device=x.device)

        msg = self.propagate(edge_index, u=u, edge_attr=edge_attr, indices=indices[:,None])

        i = torch.argwhere(edge_index[1] == 0)
        i = to_numpy(i)
        i = i[:,0].astype(int)

        j = to_numpy(edge_index[0,i])
        j = j.astype(int)

        edge_attr[i,None]
        torch.tanh(u[j])
        edge_attr[i, None] * torch.tanh(u[j])



        du = -b*u + c*torch.tanh(u) + msg

        return du

    def message(self, u_j, edge_attr, indices_i, indices_j):

        j = [  1,   2,   3,   4,  10,  11,  19,  20,  39,  40,  43,  46,  47, 493, 496]
        i = [   15,    45,    65,    87,   226,   250,   437,   462,  1089,
        1107,  1250,  1343,  1356, 17889, 17987]

        return edge_attr[:,None] * torch.tanh(u_j)

        torch.sum(edge_attr[i, None] * torch.tanh(u_j[i]))
        edge_attr[i, None]
        torch.tanh(u_j[i])
        edge_attr[i, None] * torch.tanh(u_j[i])



    def psi(self, r, p):
        return r * p
