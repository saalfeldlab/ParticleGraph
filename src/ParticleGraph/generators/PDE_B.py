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
        particle_type = x[:, 5]
        parameters = self.p[to_numpy(particle_type), :]
        velocity = x[:, 3:5]
        acc = self.propagate(edge_index, pos=x[:,1:3], x=x, parameters=parameters, velocity=velocity)

        return acc

    def message(self, pos_i, pos_j, parameters_i, velocity_i, velocity_j):
        distance_squared = torch.sum(self.bc_diff(pos_j - pos_i) ** 2, axis=1)  # distance squared

        cohesion = parameters_i[:,0:1].repeat(1, 2) * 0.5E-5 * self.bc_diff(pos_j - pos_i)

        alignment = parameters_i[:, 1:2].repeat(1, 2) * 5E-4 * self.bc_diff(velocity_j - velocity_i)

        separation = (parameters_i[:, 2:3].repeat(1, 2) * 1E-8 * self.bc_diff(pos_i - pos_j)
                      / distance_squared[:, None].repeat(1, 2))
        
        return (separation + alignment + cohesion)

    def psi(self, r, p):
        cohesion = p[0] * 0.5E-5 * r
        separation = -p[2] * 1E-8 / r
        return (cohesion + separation)  # 5E-4 alignement
