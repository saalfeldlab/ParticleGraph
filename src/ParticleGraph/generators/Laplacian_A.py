import torch_geometric as pyg
from ParticleGraph.utils import to_numpy


class Laplacian_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the Laplacian of a scalar field.

    Inputs
    ----------
    data : a torch_geometric.data object
    note the Laplacian coeeficients are in data.edge_attr

    Returns
    -------
    laplacian : float
        the Laplacian
    """

    def __init__(self, aggr_type=[], c=[], beta=[], bc_diff=[]):
        super(Laplacian_A, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.bc_diff = bc_diff

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        particle_type = to_numpy(x[:, 5])
        c = self.c[particle_type]
        c = c[:, None]

        u = x[:, 6:7]

        laplacian_u = self.propagate(edge_index, u=x, edge_attr=edge_attr)
        dd_u = self.beta * c * laplacian_u

        return dd_u

    def message(self, u_j, edge_attr):
        L = edge_attr * u_j

        return L[:, None]

    def psi(self, I, p):
        return I
