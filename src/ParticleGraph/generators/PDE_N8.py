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
    Compute network signaling, the transfer functions are neuron-neuron-dependent
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    dv : float
    the update rate of the voltages (dim 1)
        
    """

    def __init__(self, aggr_type="add", p=[], f=torch.nn.functional.relu, device=None):
        super(PDE_N8, self).__init__(aggr=aggr_type)

        self.p = p
        self.f = f
        self.device = device


    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index
        v = x[:, 3:4]
        v_rest = self.p["V_i_rest"][:, None]
        e = x[:, 4:5]
        msg = self.propagate(edge_index, v=v)
        tau = self.p["tau_i"][:, None]
        dv = (-v + msg + e + v_rest) / tau
        return dv

    def message(self, v_j):
        return self.p["w"][:, None] * self.f(v_j)
