import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy


class RD_Gray_Scott(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the reaction diffusion according to Gray_Scott model.

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
        super(RD_Gray_Scott, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.bc_diff = bc_diff

        self.Du = 5E-2
        self.Dv = 1E-2

    def forward(self, data, device):
        F = torch.tensor(0.0283, device=device)
        k = torch.tensor(0.0475, device=device)

        c = self.c[to_numpy(x[:, 5])]
        c = c[:, None]

        uv = data.x[:, 6:8]
        laplace_uv = c * self.beta * self.propagate(data.edge_index, uv=uv, edge_attr=data.edge_attr)

        uxv2 = torch.prod(uv, axis=1) ** 2
        d_u = Du * laplace_uv[:, 0] - uxv2 + F * (1 - uv[:, 0])
        d_v = Dv * laplace_uv[:, 1] + uxv2 - (F + k) * uv[:, 1]

        d_uv = self.beta * torch.cat((d_u[:, None], d_v[:, None]), axis=1)
        return d_uv

    def message(self, uv_i, uv_j, edge_attr):
        return edge_attr * uv_j

    def psi(self, I, p):
        return I
