
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy
import torch
from ParticleGraph.utils import *

class PDE_N6(pyg.nn.MessagePassing):
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
        super(PDE_N6, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi

    def forward(self, data=[], has_field=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2] * 0
        c = parameters[:, 2:3]
        tau = parameters[:, 3:4]
        alpha = parameters[:, 4:5]

        u = x[:, 6:7]
        p = x[:, 8:9]

        self.msg = self.W*self.phi(u)
        msg = torch.matmul(self.W, self.phi(u))

        du = -c * u + s * self.phi(u) + g * p * msg
        dp = (1-p)/tau - alpha * p * torch.abs(u)

        return du, dp, g * p * msg



    def message(self, u_j, edge_attr):

        self.activation = self.phi(u_j)
        self.u_j = u_j

        return edge_attr[:,None] * self.phi(u_j)


    def func(self, u, type, function):

        if function=='phi':
            return self.phi(u)

        elif function=='update':
            g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * self.phi(u)

