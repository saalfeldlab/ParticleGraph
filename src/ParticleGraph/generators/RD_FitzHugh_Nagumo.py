import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy


class RD_FitzHugh_Nagumo(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], c=[], beta=[], bc_diff=[]):
        super(RD_FitzHugh_Nagumo, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.bc_diff = bc_diff

    def forward(self, data, device):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # dx = 2./size
        # dt = 0.9 * dx**2/2
        # params = {"Du":5e-3, "Dv":2.8e-4, "tau":0.1, "k":-0.005,
        # su = (Du*Lu + v - u)/tau
        # sv = Dv*Lv + v - v*v*v - u + k

        c = self.c[to_numpy(x[:, 5])]
        c = c[:, None]

        u = x[:, 6]
        v = x[:, 7]

        laplacian = c * self.beta * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        laplacian_U = laplacian[:, 0]
        laplacian_V = laplacian[:, 1]

        # Du = 5E-3
        # Dv = 2.8E-4
        # k = torch.tensor(-0.005,device=device)
        # tau = torch.tensor(0.1,device=device)
        #
        # dU = (Du * laplacian[:,0] + v - u) / tau
        # dV = Dv * laplacian[:,1] + v - v**3 - u + k

        a1 = 5E-3
        a2 = -2.8E-3
        a3 = 5E-3

        dU = a3 * laplacian_U + 0.02 * (v - v ** 3 - u * v + torch.randn(4225, device=device))
        dV = (a1 * u + a2 * v)

        # U = U + 0.125 * dU
        # V = V + 0.125 * dV

        increment = 0.125 * torch.cat((dU[:, None], dV[:, None]), axis=1)

        return increment

    def message(self, x_i, x_j, edge_attr):
        # U column 6, V column 7

        # L = edge_attr * (x_j[:, 6]-x_i[:, 6])

        Lu = edge_attr * x_j[:, 6]
        Lv = edge_attr * x_j[:, 7]

        L = torch.cat((Lu[:, None], Lv[:, None]), axis=1)

        return L

    def psi(self, I, p):
        return I
