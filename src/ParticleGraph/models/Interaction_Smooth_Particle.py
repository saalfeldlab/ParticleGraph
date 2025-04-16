import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
from ParticleGraph.utils import *
# from ParticleGraph.models.utils import reparameterize


class Interaction_Smooth_Particle(pyg.nn.MessagePassing):
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

        super(Interaction_Smooth_Particle, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers

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

        self.n_frames = simulation_config.n_frames
        self.prediction = model_config.prediction
        self.bc_dpos = bc_dpos
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.time_window = train_config.time_window
        self.time_window_noise = train_config.time_window_noise
        self.sub_sampling = simulation_config.sub_sampling
        self.state = simulation_config.state_type

        self.sigma = simulation_config.sigma
        self.n_ghosts = int(train_config.n_ghosts)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update,
                               nlayers=self.n_layers_update,
                               hidden_size=self.hidden_dim_update, device=self.device)

        if (self.model =='PDE_F_C') | (self.model == 'PDE_F_E'):
            self.lin_mass = MLP(input_size=self.embedding_dim, output_size=1, nlayers=2,
                                    hidden_size=4, device=self.device)

        if self.state == 'sequence':
            self.a = nn.Parameter(torch.ones((self.n_dataset, int(self.n_particles*100 + 100 ), self.embedding_dim), device=self.device, requires_grad=True,dtype=torch.float32))
            self.embedding_step =  self.n_frames // 100
        else:
            self.a = nn.Parameter(
                    torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                                 requires_grad=True, dtype=torch.float32))

        self.kernel_var = self.max_radius ** 2

    def get_interp_a(self, k, particle_id, data_id):
        id = particle_id * 100 + k // self.embedding_step
        alpha = (k % self.embedding_step) / self.embedding_step
        return alpha * self.a[data_id.clone().detach(), id+1, :].squeeze() + (1 - alpha) * self.a[data_id.clone().detach(), id, :].squeeze()


    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False, k=[]):

        self.data_id = data_id
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        self.has_field = has_field
        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,6:7])

        pos = x[:, 1:self.dimension+1]
        if training & (self.time_window_noise > 0):
            noise = torch.randn_like(pos) * self.time_window_noise
            pos = pos + noise

        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        if self.state == 'sequence':
            particle_id = x[:, 0:1].long()
            embedding = self.get_interp_a(k, particle_id, self.data_id)
        else:
            particle_id = x[:, 0:1].long()
            embedding = self.a[self.data_id.clone().detach(), particle_id, :].squeeze()

        if (self.model == 'PDE_F_C') | (self.model == 'PDE_F_E'):
            self.mass = self.lin_mass(embedding)
        else:
            self.mass = torch.ones_like(embedding[:, 0:1])

        self.mode = 'kernel'
        self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=torch.zeros_like(x[:, 0:1]), mass=self.mass)
        self.mode = 'message_passing'
        out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=self.density, mass=self.mass)

        if self.update_type == 'mlp':
            out = self.lin_phi(torch.cat((out, embedding, d_pos), dim=-1))

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, embedding_i, embedding_j, field_j, density_i, density_j, mass_i, mass_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'kernel':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            Gaussian_kernel = torch.exp(-4 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:, None] / 2

            dist = torch.sqrt(torch.sum(mgrid ** 2, dim=1))
            triangle_kernel = ((self.max_radius - dist) ** 2 / self.kernel_var)[:, None] / 1.309

            grad_triangle_kernel = density_gradient(triangle_kernel, mgrid)
            grad_triangle_kernel = torch.where(torch.isnan(grad_triangle_kernel),
                                               torch.zeros_like(grad_triangle_kernel), grad_triangle_kernel)

            self.kernel_operators = dict()
            self.kernel_operators['Gaussian'] = Gaussian_kernel.clone().detach()
            self.kernel_operators['grad_triangle'] = grad_triangle_kernel.clone().detach()

            if self.model == 'PDE_F_B':
                grad_Gaussian_kernel = density_gradient(Gaussian_kernel, mgrid)
                laplacian_kernel = density_laplace(Gaussian_kernel, mgrid)
                self.kernel_operators['grad_Gaussian'] = grad_Gaussian_kernel.clone().detach()
                self.kernel_operators['laplacian'] = laplacian_kernel.clone().detach()

            density = Gaussian_kernel * mass_j

            return density.clone().detach()

        elif self.mode == 'message_passing':

            if self.model == 'PDE_F_A':
                in_features = torch.cat((d_pos_i - d_pos_j, embedding_i, embedding_j, density_i, density_j, self.kernel_operators['grad_triangle'], self.kernel_operators['Gaussian'] ), dim=-1)
            elif self.model == 'PDE_F_B':
                in_features = torch.cat((d_pos_i - d_pos_j, embedding_i, embedding_j, density_i, density_j, self.kernel_operators['Gaussian'],self.kernel_operators['grad_Gaussian'],self.kernel_operators['laplacian']), dim=-1)
            elif self.model == 'PDE_F_C':
                in_features = torch.cat((mass_i, mass_j, d_pos_i - d_pos_j, embedding_i, embedding_j, density_i, density_j, self.kernel_operators['Gaussian'],self.kernel_operators['grad_triangle']), dim=-1)
            elif self.model == 'PDE_F_D':
                r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
                delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
                in_features = torch.cat((delta_pos, r[:, None], d_pos_i - d_pos_j, embedding_i, embedding_j, density_i, density_j, self.kernel_operators['Gaussian'],self.kernel_operators['grad_triangle']), dim=-1)
            elif self.model == 'PDE_F_E':
                r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
                delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
                in_features = torch.cat((delta_pos, r[:, None], mass_i, mass_j, d_pos_i - d_pos_j, embedding_i, embedding_j, density_i, density_j, self.kernel_operators['Gaussian'],self.kernel_operators['grad_triangle']), dim=-1)

            out = self.lin_edge(in_features) / density_j.repeat(1,2)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


