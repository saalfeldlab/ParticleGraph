import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_B(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of Boids as a function of their relative positions and relative positions.
    The interaction function is defined by three parameters p = (p1, p2, p3)

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the Boids (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], delta_t=[], bc_diff=[]):
        super(PDE_B, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.delta_t = delta_t
        self.bc_diff = bc_diff

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        return acc

    def message(self, x_i, x_j):
        r = torch.sum(self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, axis=1)  # distance squared

        pp = self.p[to_numpy(x_i[:, 5]), :]

        cohesion = pp[:, 0:1].repeat(1, 2) * 0.5E-5 * self.bc_diff(x_j[:, 1:3] - x_i[:, 1:3])

        alignment = pp[:, 1:2].repeat(1, 2) * 5E-4 * self.bc_diff(x_j[:, 3:5] - x_i[:, 3:5])

        separation = (pp[:, 2:3].repeat(1, 2) * 1E-8 * self.bc_diff(x_i[:, 1:3] - x_j[:, 1:3])
                      / r[:, None].repeat(1, 2))

        return (separation + alignment + cohesion)

    def psi(self, r, p):
        cohesion = p[0] * 0.5E-5 * r
        separation = -p[2] * 1E-8 / r
        return (cohesion + separation)  # 5E-4 alignement
