import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_F(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of particles falling with gravity and bouncing on walls.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], dimension=2, delta_t=0.1):
        super(PDE_F, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.dimension = dimension
        self.p = None
        self.delta_t = delta_t


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particles = torch.argwhere(x[:, 1 + 2*self.dimension] == 1).squeeze()
        dd_pos = torch.zeros_like(x[:, 1:3])
        dd_pos[particles, 0] = -9.81

        X1 = x[particles, 1:3]
        V1 = x[particles, 3:5]
        V1 += dd_pos[particles] * self.delta_t
        X1 = X1 + V1 * self.delta_t

        # Bounce on walls
        bouncing_pos = torch.argwhere((X1[:, 0] <= 0) | (X1[:, 0] >= 1)).squeeze()
        if bouncing_pos.numel() > 0:
            dd_pos[particles[bouncing_pos], 0] = -1.6 * V1[bouncing_pos, 0] / self.delta_t
            dd_pos[particles[bouncing_pos], 1] = - 0.4 * V1[bouncing_pos, 1] / self.delta_t
        bouncing_pos = torch.argwhere((X1[:, 1] <= 0) | (X1[:, 1] >= 1)).squeeze()
        if bouncing_pos.numel() > 0:
        #     dd_pos[particles[bouncing_pos], 0] = - 0.4 * V1[bouncing_pos, 0] / self.delta_t
            dd_pos[particles[bouncing_pos], 1] = -1.6 * V1[bouncing_pos, 1] / self.delta_t

        return dd_pos

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j):

        dd_pos = pos_i * 0

        return dd_pos

