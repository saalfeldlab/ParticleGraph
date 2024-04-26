import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy
import numpy as np


class PDE_ParticleField_A(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=[], p=[], sigma=[], bc_dpos=[], dimension=2, n_particles=0, n_nodes=0):

        super(PDE_ParticleField_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.n_particles = n_particles
        self.n_nodes = n_nodes
        self.l_nodes = int(np.sqrt(n_nodes))

        self.p = p
        self.bc_dpos = bc_dpos

        self.sigma = sigma
        self.dimension = dimension


    def forward(self, data):

        x, edge, field = data.x, data.edge, data.field

        edge, _ = pyg_utils.remove_self_loops(edge)

        field = torch.reshape(u, (self.l_nodes, self.l_nodes))

        particle_id = to_numpy(x[0:self.n_nodes, 0])
        particle_id = particle_id.astype(int)
        particle_type = x[:, 5:6]
        parameters = self.p[t, :]

        pos = x[:, 1:3]
        d_pos = x[:, 3:5]

        dd_pos = self.propagate(edge_index=edge_all, u=field, discrete_laplacian=edge_attr, pos=pos, d_pos=d_pos, parameters=parameters.squeeze())
        deg_particle[deg_particle==0] = 1
        dd_pos = dd_pos[self.n_nodes:] / deg_particle[:, None].repeat(1,2)

        # chemotaxism
        dd_pos_field_to_particle = self.propagate(edge_index=edge_all, u=field, discrete_laplacian=edge_attr, mode ='field_to_particle', pos=pos, d_pos=d_pos, parameters=parameters.squeeze())
        node_neighbour = dd_pos_field_to_particle[self.n_nodes:,2:4]
        node_neighbour[node_neighbour==0]=1
        dd_pos_field_to_particle = dd_pos_field_to_particle[self.n_nodes:,0:2]
        dd_pos_field_to_particle_dd_pos = dd_pos_field_to_particle/node_neighbour

        return dd_pos, dd_pos_field_to_particle_dd_pos



    def message(self, u_j, discrete_laplacian, mode, pos_i, pos_j, d_pos_i, d_pos_j, particle_type_i, particle_type_j, parameters_i, parameters_j):

        if mode == 'particle-particle':

            distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # squared distance
            psi = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2))
                   - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
            d_pos = psi[:, None] * self.bc_dpos(pos_j - pos_i) * ((particle_type_i > -1)&(particle_type_j > -1)).float()

            return d_pos

        elif mode == 'field_to_particle':

            distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # squared distance
            psi = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2))
                   - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
            d_pos = u_j * psi[:, None] * self.bc_dpos(pos_j - pos_i) * (
                        (particle_type_i > -1) & (particle_type_j < 0)).float()
            node_neighbour = (particle_type_j < 0).float()

            return torch.cat((d_pos,node_neighbour.repeat(1,2)),1)


    def psi(self, I, p):
        return I
