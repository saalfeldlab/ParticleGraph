import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy
from scipy import sparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

class PDE_N2(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Comput
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        
    """

    def __init__(self, aggr_type=[], p=[], W=[], phi=[]):
        super(PDE_N2, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi

    def forward(self, data=[], return_all=False, has_field=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]
        c = parameters[:, 2:3]

        u = x[:, 6:7]

        # msg = self.propagate(edge_index, u=u, edge_attr=edge_attr)
        msg = torch.matmul(self.W, self.phi(u))

        du = -c * u + s * self.phi(u) + g * msg

        if return_all:
            return du, s * self.phi(u), g * msg
        else:
            return du

    def message(self, u_j, edge_attr):

        self.activation = self.phi(u_j)
        self.u_j = u_j

        return edge_attr[:,None] * self.phi(u_j)




    def psi(self, r, p):
        return r * p
