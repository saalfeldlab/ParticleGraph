import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_ParticleField(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=[],  pos_rate=[], neg_rate=[], coeff_diff=[], delta_t=[], p=[], bc_dpos=[], n_particles=0, n_nodes=0):

        super(PDE_ParticleField, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.n_particles = n_particles
        self.n_nodes = n_nodes

        self.p = p
        self.bc_dpos = bc_dpos

        self.a1 = 0.5E-5
        self.a2 = 5E-4
        self.a3 = 1E-8
        self.a4 = 5E-7

        self.coeff_diff = coeff_diff
        self.pos_rate = pos_rate
        self.neg_rate = neg_rate

        self.delta_t = delta_t

    def forward(self, data):

        x, edge_all, edge_particle, edge_mesh, edge_attr = data.x, data.edge_index, data.edge_particle, data.edge_mesh, data.edge_attr

        edge_all, _ = pyg_utils.remove_self_loops(edge_all)

        deg_particle = pyg_utils.degree(edge_particle[0, :].squeeze(), self.n_particles)

        u = x[:, 6:7]
        particle_id = to_numpy(x[0:self.n_nodes, 0])
        particle_id = particle_id.astype(int)
        particle_type = x[:, 5:6]
        parameters = self.p[to_numpy(particle_type), :]

        pos = x[:, 1:3]
        d_pos = x[:, 3:5]

        # edge_attr = torch.clamp(edge_attr, -1, 1)
        laplacian_u = self.propagate(edge_index=edge_mesh, u=u, discrete_laplacian=edge_attr, mode ='field_to_field_laplacian', pos=pos, d_pos=d_pos, particle_type=particle_type, parameters=parameters.squeeze()) # , parameters=parameters, particle_type=particle_type)
        laplacian_u = laplacian_u[0:self.n_nodes]
        d_u_particle_to_field = self.propagate(edge_index=edge_all, u=u, discrete_laplacian=edge_attr, mode ='particle_to_field', pos=pos, d_pos=d_pos, particle_type=particle_type, parameters=parameters.squeeze())
        d_u_particle_to_field = d_u_particle_to_field[0:self.n_nodes]

        coeff_diff = self.coeff_diff[particle_id]
        pos_rate = self.pos_rate[particle_id]
        neg_rate = self.neg_rate[particle_id]

        dd_u = torch.clamp(coeff_diff * laplacian_u, -10, 10) + pos_rate * u[0:self.n_nodes] + neg_rate * d_u_particle_to_field


        dd_pos = self.propagate(edge_index=edge_all, u=u, discrete_laplacian=edge_attr, mode ='particle-particle', pos=pos, d_pos=d_pos, particle_type=particle_type, parameters=parameters.squeeze())
        deg_particle[deg_particle==0] = 1
        dd_pos = dd_pos[self.n_nodes:] / deg_particle[:, None].repeat(1,2)

        # chemotaxism
        dd_pos_field_to_particle = self.propagate(edge_index=edge_all, u=u, discrete_laplacian=edge_attr, mode ='field_to_particle', pos=pos, d_pos=d_pos, particle_type=particle_type, parameters=parameters.squeeze())
        node_neighbour = dd_pos_field_to_particle[self.n_nodes:,2:4]
        node_neighbour[node_neighbour==0]=1
        dd_pos_field_to_particle = dd_pos_field_to_particle[self.n_nodes:,0:2]
        dd_pos_field_to_particle_dd_pos = dd_pos_field_to_particle/node_neighbour

        return dd_pos, dd_pos_field_to_particle_dd_pos, dd_u

        fig = plt.figure(figsize=(10, 10))
        plt.hist(to_numpy(self.beta * laplacian_u), 100)


    def message(self, u_j, discrete_laplacian, mode, pos_i, pos_j, d_pos_i, d_pos_j, particle_type_i, particle_type_j, parameters_i, parameters_j):

        if mode == 'field_to_field_laplacian':

            Laplacian_component = discrete_laplacian[:,None] * u_j * ((particle_type_j < 0) & (particle_type_i < 0)).float()

            return Laplacian_component

        elif mode == 'particle_to_field':

            distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # distance squared
            msg = 1 / distance_squared[:, None] * ((particle_type_j > -1) & (particle_type_i < 0)).float()

            return msg

        elif mode == 'particle-particle':

            distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1) # distance squared

            cohesion_particle = parameters_i[:,0,None] * self.a1 * self.bc_dpos(pos_j - pos_i) * (particle_type_j > -1).float()
            alignment = parameters_i[:,1,None] * self.a2 * self.bc_dpos(d_pos_j - d_pos_i) * (particle_type_j > -1).float()
            separation = - parameters_i[:,2,None] * self.a3 * self.bc_dpos(pos_j - pos_i) / distance_squared[:, None] * (particle_type_j > -1).float()

            return (separation + alignment + cohesion_particle)

        elif mode == 'field_to_particle':

            distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1) # distance squared

            msg = parameters_i[:, 3, None] * self.a4 * torch.clamp(u_j,0,10000) * self.bc_dpos(pos_j - pos_i) * (particle_type_j < 0).float()
            node_neighbour = (particle_type_j < 0).float()

            return torch.cat((msg,node_neighbour.repeat(1,2)),1)


    def psi(self, I, p):
        return I
