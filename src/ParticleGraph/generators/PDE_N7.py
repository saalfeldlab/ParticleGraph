
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy
import torch
from ParticleGraph.utils import *

class PDE_N7(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=[], p=[], W=[], phi=[], short_term_plasticity_mode=''):
        super(PDE_N7, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi
        self.short_term_plasticity_mode = short_term_plasticity_mode

    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]
        c = parameters[:, 2:3]
        t = parameters[:, 3:4]
        tau = parameters[:, 4:5]
        alpha = parameters[:, 5:6]

        u = x[:, 6:7]
        p = x[:, 8:9]

        # self.msg = self.W*self.phi(u)
        # msg = torch.matmul(self.W, self.phi(u))

        msg = self.propagate(edge_index, u=u, t=t)

        du = -c * u + s * self.phi(u) + g * p * msg
        dp = (1-p)/tau - alpha * p * torch.abs(u)

        return du, dp

    def message(self, edge_index_i, edge_index_j, u_j, t_i):

        T = self.W
        return T[edge_index_i, edge_index_j][:, None] * self.phi(u_j / t_i)


    def func(self, u, type, function):

        if function=='phi':
            return self.phi(u)

        elif function=='update':
            g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * self.phi(u)

