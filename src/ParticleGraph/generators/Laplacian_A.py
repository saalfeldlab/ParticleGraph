import torch_geometric as pyg
from ParticleGraph.utils import to_numpy

class Laplacian_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], c=[], beta=[], bc_diff=[]):
        super(Laplacian_A, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.bc_diff = bc_diff

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        c = self.c[to_numpy(x[:, 5])]
        c = c[:, None]

        laplacian = self.beta * c * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        # laplacian = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        # laplacian[2178]
        # g = edge_index.detach().cpu().numpy()
        # xx = x.detach().cpu().numpy()
        # ll = edge_attr.detach().cpu().numpy()
        # pos = np.argwhere(g[0, :] == 2178)
        # pos = np.squeeze(pos)
        # np.sum(ll[pos] * xx[g[1, pos], 6])

        return laplacian

    def message(self, x_i, x_j, edge_attr):

        L = edge_attr * x_j[:, 6]

        return L[:, None]

    def psi(self, I, p):

        return I