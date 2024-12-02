import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_D(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the speed of the particles (dimension 2)
    """

    def __init__(self, config=[], aggr_type='mean', bc_dpos=[], dimension=[]):
        super(PDE_D, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.smooth_radius = config.training.smooth_radius


    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,0:1])

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        density, d_density = self.propagate(edge_index, pos=x[:, 1:self.dimension+1], field=field)
        return density


    def message(self, pos_i, pos_j, field_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)
        distance = torch.sqrt(distance_squared)

        d, d_d = self.W(distance, self.smooth_radius)
        pos = torch.argwhere(distance_squared > self.smooth_radius^2)
        if pos.numel() > 0:
            d[pos] = 0
            d_d[pos] = 0

        d_d = d_d * self.bc_dpos(pos_j - pos_i)/distance

        return d, d_d

    def w(self, d, s):

        w_density = 4/(np.pi*s^8)(s^2-s^2)^3

        wp_density = -24/(np.pi*s^8)*d*(s^2-d^2)^2


        return w_density, wp_density
