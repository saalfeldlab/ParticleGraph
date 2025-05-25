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
        self.n_layers = model_config.n_layers

        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update

        self.model = model_config.particle_model_name
        self.n_dataset = train_config.n_runs
        self.dimension = dimension
        self.delta_t = simulation_config.delta_t
        self.n_particles = simulation_config.n_particles
        self.embedding_dim = model_config.embedding_dim
        self.embedding_trial = config.training.embedding_trial

        self.n_frames = simulation_config.n_frames
        self.prediction = model_config.prediction
        self.bc_dpos = bc_dpos
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.reflection_augmentation = train_config.reflection_augmentation
        self.time_window = train_config.time_window
        self.recursive_loop = train_config.recursive_loop
        self.state = simulation_config.state_type
        self.remove_self = train_config.remove_self

        self.sigma = simulation_config.sigma
        self.n_ghosts = int(train_config.n_ghosts)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update,
                               nlayers=self.n_layers_update,
                               hidden_size=self.hidden_dim_update, device=self.device)

        if self.state == 'sequence':
            self.a = nn.Parameter(torch.ones((self.n_dataset, int(self.n_particles*100 + 100 ), self.embedding_dim), device=self.device, requires_grad=True,dtype=torch.float32))
            self.embedding_step = self.n_frames // 100
        else:
            self.a = nn.Parameter(
                    torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                                 requires_grad=True, dtype=torch.float32))

        if self.model =='PDE_K':
            self.vals = nn.Parameter(
                torch.ones((self.n_dataset, int(self.n_particles * (self.n_particles + 1) / 2)), device=self.device,
                            requires_grad=True, dtype=torch.float32))

    def get_interp_a(self, k, particle_id, data_id):
        id = particle_id * 100 + k // self.embedding_step
        alpha = (k % self.embedding_step) / self.embedding_step
        return alpha * self.a[data_id.clone().detach(), id+1, :].squeeze() + (1 - alpha) * self.a[data_id.clone().detach(), id, :].squeeze()


    def forward(self, data=[], data_id=[], training=[], has_field=False, k=[]):

        self.data_id = data_id
        self.training = training
        self.has_field = has_field

        x, edge_index = data.x, data.edge_index
        if self.remove_self:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,6:7])

        if self.model == 'PDE_T':
            derivatives = x[:, 6:]
        else:
            derivatives = torch.zeros_like(x[:, 6:7])

        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension] / self.vnorm
        if self.rotation_augmentation & self.training:
            self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
            self.rotation_matrix = torch.stack([
                torch.stack([torch.cos(self.phi), torch.sin(self.phi)]),
                torch.stack([-torch.sin(self.phi), torch.cos(self.phi)])
            ])
            self.rotation_matrix = self.rotation_matrix.permute(*torch.arange(self.rotation_matrix.ndim - 1, -1, -1)).squeeze()

            d_pos[:, :2] = d_pos[:, :2] @ self.rotation_matrix

            for n in range(derivatives.shape[1]//2):
                derivatives[:, n*2:n*2+2] = derivatives[:, n*2:n*2+2] @ self.rotation_matrix

        # if self.reflection_augmentation & self.training == True:
        #     group = np.random.randint(0, 3)
        #     if group in [0, 1]:
        #         pos[:, group] = 1 - pos[:, group]
        #         d_pos[:, group] = -d_pos[:, group]
        #     else:
        #         pos = 1 - pos
        #         d_pos[:, 1] = -d_pos[:, 1]

        if self.state == 'sequence':
            particle_id = x[:, 0:1].long()
            embedding = self.get_interp_a(k, particle_id, self.data_id)
        else:
            particle_id = x[:, 0:1].long()
            embedding = self.a[self.data_id.clone().detach(), particle_id, :].squeeze()

        out = self.propagate(edge_index, particle_id=particle_id, pos=pos, d_pos=d_pos, embedding=embedding, field=field, derivatives=derivatives)

        if self.update_type == 'mlp':
            if has_field:
                out = self.lin_phi(torch.cat((out, embedding, d_pos, field), dim=-1))
            else:
                out = self.lin_phi(torch.cat((out, embedding, d_pos), dim=-1))
        if self.rotation_augmentation & self.training:
            self.rotation_inv_matrix = torch.stack([torch.stack([torch.cos(self.phi), -torch.sin(self.phi)]),torch.stack([torch.sin(self.phi), torch.cos(self.phi)])])
            self.rotation_inv_matrix = self.rotation_inv_matrix.permute(*torch.arange(self.rotation_inv_matrix.ndim - 1, -1, -1)).squeeze()
            out[:, :2] = out[:, :2] @ self.rotation_inv_matrix
        if self.reflection_augmentation & self.training:
            if group in [0, 1]:
                out[:, group] = -out[:, group]
            else:
                out = -out

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, field_j, derivatives_j):

        # distance normalized by the max radius
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        if self.rotation_augmentation & self.training:
            delta_pos[:, :2] = delta_pos[:, :2] @ self.rotation_matrix

        match self.model:
            case 'PDE_A':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
            case 'PDE_A_bis':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)
            case 'PDE_B' | 'PDE_F':
                in_features = torch.cat((delta_pos, r[:, None], d_pos_i, d_pos_j, embedding_i), dim=-1)
            case 'PDE_G':
                in_features = torch.cat((delta_pos, r[:, None], d_pos_i, d_pos_j, embedding_j), dim=-1)
            case 'PDE_GS':
                in_features = torch.cat((delta_pos, r[:, None], 10**embedding_j),dim=-1)
            case 'PDE_E':
                in_features = torch.cat(
                    (delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)
            case 'PDE_K':
                in_features = delta_pos
            case 'PDE_T':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i, derivatives_j), dim=-1)

        if self.model == 'PDE_K':
            A = torch.zeros(self.n_particles, self.n_particles, device=self.device, requires_grad=False, dtype=torch.float32)
            i, j = torch.triu_indices(self.n_particles, self.n_particles, requires_grad=False, device=self.device)
            A[i, j] = self.vals[self.data_id[0]]**2
            A.T[i, j] = self.vals[self.data_id[0]]**2
            A[i,i] = 0
            out = A[edge_index_i, edge_index_j].repeat(2, 1).t() * self.lin_edge(in_features)


        out = self.lin_edge(in_features)


        if self.training==False:
            pos = torch.argwhere(edge_index_i == self.particle_of_interest)
            if pos.numel()>0:
                self.msg = out[pos[:,0]]
            else:
                self.msg = out[0]

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
