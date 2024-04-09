import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_P(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=[],  pos_rate=[], neg_rate=[], beta=[], p=[], bc_dpos=[]):
        super(PDE_P, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.bc_dpos = bc_dpos

        self.a1 = 0.5E-5
        self.a2 = 5E-4
        self.a3 = 1E-8
        self.a4 = 0.5E-5
        self.a5 = 0.5E-5
        self.a6 = 1E-8

        self.pos_rate = pos_rate
        self.neg_rate = neg_rate
        self.beta = beta

    def forward(self, data):

        x, edge_index, edge_mesh, edge_attr = data.x, data.edge_index, data.edge_mesh, data.edge_attr
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        particle_type = x[:, 5]
        parameters = self.p[to_numpy(particle_type), :]
        d_pos = x[:, 3:5].clone().detach()

        dd_pos = self.propagate(edge_index, u=x[:, 6:7], pos=x[:,1:3], parameters=parameters, d_pos=d_pos, particle_type=particle_type, edge_attr=edge_attr, mode='particle')

        laplacian_u = self.propagate(edge_mesh, u=x[:, 6:7], pos=x[:, 1:3], parameters=parameters, d_pos=d_pos,
                                particle_type=particle_type, edge_attr=edge_attr, mode='laplacian')

        d_u_neg = self.propagate(edge_index, u=x[:, 6:7], pos=x[:, 1:3], parameters=parameters, d_pos=d_pos,
                                particle_type=particle_type, edge_attr=edge_attr, mode='mesh_neg')

        d_u = self.beta * laplacian_u + self.pos_rate * u - self.neg_rate * d_u_neg

        return dd_pos, d_u

    def message(self, u, pos_i, pos_j, parameters_i, d_pos_i, d_pos_j, particle_type_j, edge_attr):

        if mode ='laplacian':

            L = edge_attr[:, None] * u_j

            return L

        elif mode = 'mesh_neg':

            distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1) # distance squared

            degradation = torch.abs(self.bc_dpos(pos_j - pos_i)) / distance_squared[:, None] * ((particle_type_j > -1) & (particle_type_i == -1)).float()/ distance_squared[:, None]

            return degradation

        elif mode = 'particle':

            distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1) # distance squared

            cohesion_particle = parameters_i[:,0,None] * self.a1 * self.bc_dpos(pos_j - pos_i) * (particle_type_j > -1).float()
            cohesion_field = parameters_i[:, 3, None] * self.a4 * u_j * self.bc_dpos(pos_j - pos_i) * (particle_type_j == -1).float()  # towards barycentre of local mesh density

            alignment = parameters_i[:,1,None] * self.a2 * self.bc_dpos(d_pos_j - d_pos_i)
            separation = - parameters_i[:,2,None] * self.a3 * self.bc_dpos(pos_j - pos_i) / distance_squared[:, None]

            return (separation + alignment + cohesion + cohesion_field) * (particle_type_j > -1).float()



    def psi(self, r, p):
        cohesion = p[0] * self.a5 * r
        separation = -p[2] * self.a6 / r
        return (cohesion + separation)  # 5E-4 alignement
