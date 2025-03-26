
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import torch
from ParticleGraph.utils import *


class PDE_N8(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-dependent
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    du : float
    the update rate of the signals (dim 1)
        
    """

    def __init__(self, aggr_type=[], p=[], W=[], phi=[]):
        super(PDE_N8, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi

    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index
        particle_type = x[:, 5].long()
        parameters = self.p[particle_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]
        c = parameters[:, 2:3]
        t = parameters[:, 3:4]

        u = x[:, 6:7]
        if has_field:
            field = x[:, 8:9]
        else:
            field = torch.ones_like(x[:, 6:7])

        msg = self.propagate(edge_index, u=u, t=t, field=field)

        du = -c * u + s * torch.tanh(u) + g * msg

        return du

    def message(self, edge_index_i, edge_index_j, u_j, t_j, field_i):
        T = self.W
        return T[edge_index_i, edge_index_j][:, None] * self.phi(u_j / t_j) * field_i

    def func(self, u, type, function):

        if function=='phi':

            t = self.p[type, 3:4]
            return self.phi(u/t)

        elif function=='update':
            g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * torch.tanh(u)
