import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], delta_t=[], sigma=[], bc_diff=[]):
        super(PDE_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.sigma = sigma
        self.delta_t = delta_t
        self.bc_diff = bc_diff

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        pred = self.propagate(edge_index, x=(x, x))

        return pred

    def message(self, x_i, x_j):
        r = torch.sum(self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)  # squared distance
        pp = self.p[to_numpy(x_i[:, 5]), :]
        psi = pp[:, 0] * torch.exp(-r ** pp[:, 1] / (2 * self.sigma ** 2)) - pp[:, 2] * torch.exp(
            -r ** pp[:, 3] / (2 * self.sigma ** 2))
        return psi[:, None] * self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3])

    def psi(self, r, p):
        return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2)) - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * self.sigma ** 2)))