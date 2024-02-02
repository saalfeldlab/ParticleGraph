import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy


class RD_FitzHugh_Nagumo(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the reaction diffusion according to FitzHugh_Nagumo model.

    Inputs
    ----------
    data : a torch_geometric.data object
    Note the Laplacian coeeficients are in data.edge_attr

    Returns
    -------
    increment : float
        the first derivative of two scalar fields u and v
    """

    def __init__(self, aggr_type=[], c=[], beta=[], bc_diff=[]):
        super(RD_FitzHugh_Nagumo, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.bc_diff = bc_diff

        self.a1 = 5E-3
        self.a2 = -2.8E-3
        self.a3 = 5E-3

    def forward(self, data, device):
        c = self.c[to_numpy(x[:, 5])]
        c = c[:, None]

        u = data.x[:, 6]
        v = data.x[:, 7]
        laplace_u = c * self.beta * self.propagate(data.edge_index, u=u, edge_attr=data.edge_attr)

        d_u = self.a3 * laplace_u + 0.02 * (v - v ** 3 - u * v + torch.randn(4225, device=device))
        d_v = (self.a1 * u + self.a2 * v)

        d_uv = 0.125 * torch.cat((d_u[:, None], d_v[:, None]), axis=1)
        return d_uv

    def message(self, u_i, u_j, edge_attr):
        return edge_attr * u_j

    def psi(self, I, p):
        return I
