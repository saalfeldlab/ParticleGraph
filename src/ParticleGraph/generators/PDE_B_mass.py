import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_B_mass(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=[], p=[], bc_dpos=[]):
        super(PDE_B_mass, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.bc_dpos = bc_dpos

        self.a1 = 0.5E-5
        self.a2 = 5E-4
        self.a3 = 1E-8
        self.a4 = 0.5E-5
        self.a5 = 1E-8

    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,0:1])

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type, :]
        d_pos = x[:, 3:5].clone().detach()
        mass = x[:, 10:11].clone_detach()
        mass_coeff = self.set_mass_coeff(0.1, 1, mass, x.device)

        dd_pos = self.propagate(edge_index, pos=x[:,1:3], parameters=parameters, d_pos=d_pos, field=field)

        return dd_pos

    def message(self, pos_i, pos_j, parameters_i, d_pos_i, d_pos_j, field_j):
        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # distance squared

        cohesion = parameters_i[:,0,None] * self.a1 * self.bc_dpos(pos_j - pos_i)
        alignment = parameters_i[:,1,None] * self.a2 * self.bc_dpos(d_pos_j - d_pos_i)
        separation = - parameters_i[:,2,None] * self.a3 * self.bc_dpos(pos_j - pos_i) / distance_squared[:, None]

        return (separation + alignment + cohesion) * field_j

    def set_mass_coeff(mass_coeff_range, final_mass, current_mass, device):
        power = -1 * (current_mass - (3 / 4) * final_mass) / mass_coeff_range

        mass_coeff = 0.3 / (1 + np.exp(to_numpy(power))) + 0.75

        # return torch.Tensor(mass_coeff, device=device)[:, None]
        return mass_coeff[:, None]

    def psi(self, r, p):
        cohesion = p[0] * self.a4 * r
        separation = -p[2] * self.a5 / r
        return (cohesion + separation)  # 5E-4 alignement
