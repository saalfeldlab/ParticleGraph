import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize


class Interaction_Particle(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the acceleration of particles as a function of their relative distance and relative velocities.
    The interaction function is defined by a MLP self.lin_edge
    The particle embedding is defined by a table self.a

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_Particle, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.sigma = simulation_config.sigma
        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.delta_t = simulation_config.delta_t
        self.prediction = model_config.prediction
        self.time_window = train_config.time_window
        self.integration = model_config.integration

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))
        # self.a = nn.Parameter(
        #         torch.tensor(np.random.randn(self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim), device=self.device,
        #                      requires_grad=True, dtype=torch.float32))

        if self.model =='PDE_K1':
            self.vals = nn.Parameter(
                torch.ones((self.n_dataset, int(self.n_particles * (self.n_particles + 1) / 2)), device=self.device,
                            requires_grad=True, dtype=torch.float32))


    def forward(self, data=[], data_id=[], training=[], vnorm=[], phi=[], has_field=False, frame=[]):

        self.data_id = data_id
        self.vnorm = vnorm
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        self.has_field = has_field

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_id = to_numpy(x[:, 0])

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,6:7])

        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        particle_id = x[:, 0:1]
        embedding = self.a[self.data_id.clone().detach(), to_numpy(particle_id), :].squeeze()

        pred = self.propagate(edge_index, particle_id=particle_id, pos=pos, d_pos=d_pos, embedding=embedding, field=field)

        if self.integration:

            if self.prediction == '2nd_derivative':
                d_pos = d_pos + self.delta_t/2 * pred * self.ynorm
            else:
                d_pos = pred * self.vnorm
            pos = pos + self.delta_t/2 * d_pos
            pred = (pred + self.propagate(edge_index, particle_id=particle_id, pos=pos, d_pos=d_pos, embedding=embedding, field=field)) / 2

        return pred

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, field_j):

        if torch.isnan(pos_i).any():
            print('nan')
        if torch.isnan(pos_i).any():
            print('nan')

        # distance normalized by the max radius
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm

        if self.rotation_augmentation & (self.training == True):
            new_delta_pos_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_delta_pos_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_delta_pos_x
            delta_pos[:, 1] = new_delta_pos_y
            new_dpos_x_i = self.cos_phi * dpos_x_i + self.sin_phi * dpos_y_i
            new_dpos_y_i = -self.sin_phi * dpos_x_i + self.cos_phi * dpos_y_i
            dpos_x_i = new_dpos_x_i
            dpos_y_i = new_dpos_y_i
            new_dpos_x_j = self.cos_phi * dpos_x_j + self.sin_phi * dpos_y_j
            new_dpos_y_j = -self.sin_phi * dpos_x_j + self.cos_phi * dpos_y_j
            dpos_x_j = new_dpos_x_j
            dpos_y_j = new_dpos_y_j

        match self.model:
            case 'PDE_A'|'PDE_ParticleField_A':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
            case 'PDE_A_bis':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)
            case 'PDE_B' | 'PDE_B_bis' | 'PDE_B_mass':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                                         dpos_y_j[:, None], embedding_i), dim=-1)
            case 'PDE_G':
                in_features = torch.cat(
                    (delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None], dpos_y_j[:, None],
                     embedding_j),
                    dim=-1)
            case 'PDE_GS':
                in_features = torch.cat((delta_pos, r[:, None], 10**embedding_j),dim=-1)
            case 'PDE_E':
                in_features = torch.cat(
                    (delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)
            case 'PDE_K':
                in_features = torch.cat((delta_pos, embedding_i, embedding_j), dim=-1)
            case 'PDE_K1':
                in_features = delta_pos

        if self.model == 'PDE_K1':
            A = torch.zeros(self.n_particles, self.n_particles, device=self.device, requires_grad=False, dtype=torch.float32)
            i, j = torch.triu_indices(self.n_particles, self.n_particles, requires_grad=False, device=self.device)
            A[i, j] = self.vals[self.data_id]**2
            A.T[i, j] = self.vals[self.data_id]**2
            A[i,i] = 0
            out = A[edge_index_i, edge_index_j].repeat(2, 1).t() * self.lin_edge(in_features)

        else:
            out = self.lin_edge(in_features) * field_j

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p1, p2):

        if (self.model == 'PDE_A') | (self.model =='PDE_A_bis') | (self.model=='PDE_ParticleField_A'):
            return r * (p1[0] * torch.exp(-torch.abs(r) ** (2 * p1[1]) / (2 * self.sigma ** 2)) - p1[2] * torch.exp(-torch.abs(r) ** (2 * p1[3]) / (2 * self.sigma ** 2)))
        if self.model == 'PDE_B':
            cohesion = p1[0] * 0.5E-5 * r
            separation = -p1[2] * 1E-8 / r
            return (cohesion + separation) * p1[1] / 500
        if self.model == 'PDE_G':
            psi = p1 / r ** 2
            return psi[:, None]
        if self.model == 'PDE_E':
            acc = p1 * p2 / r ** 2
            return -acc  # Elec particles

        # fig = plt.figure()
        # plt.scatter(-delta_pos.detach().cpu().numpy(), self.lin_edge(in_features).detach().cpu().numpy(), s=20, c='k')
