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

    def forward(self, data, device):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        c = self.c[to_numpy(x[:, 5])]
        c = c[:, None]

        laplacian = c * self.beta * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        Du = 5E-2
        Dv = 1E-2
        F = torch.tensor(0.0283, device=device)
        k = torch.tensor(0.0475, device=device)

        dU = Du * laplacian[:, 0] - x[:, 6] * x[:, 7] ** 2 + F * (1 - x[:, 6])
        dV = Dv * laplacian[:, 1] + x[:, 6] * x[:, 7] ** 2 - (F + k) * x[:, 7]

        pred = self.beta * torch.cat((dU[:, None], dV[:, None]), axis=1)

        return pred

    def message(self, x_i, x_j, edge_attr):
        # U column 6, V column 7

        # L = edge_attr * (x_j[:, 6]-x_i[:, 6])

        L1 = edge_attr * x_j[:, 6]
        L2 = edge_attr * x_j[:, 7]

        L = torch.cat((L1[:, None], L2[:, None]), axis=1)

        return L

    def psi(self, I, p):
        return I
