import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
from ParticleGraph.utils import *
# from ParticleGraph.models.utils import reparameterize


class Interaction_PDE_Particle(pyg.nn.MessagePassing):
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

        super(Interaction_PDE_Particle, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

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

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers

        mlp_params = model_config.multi_mlp_params

        self.MLP = nn.ModuleList([
            MLP(input_size=params[0], output_size=params[3], nlayers=params[2], hidden_size=params[1],
                device=self.device)
            for params in mlp_params
        ])

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_particles) , self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))

        # self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
        #                         hidden_size=self.hidden_dim, device=self.device)

        self.kernel_var = self.max_radius ** 2


    def forward(self, data=[], data_id=[], training=[], has_field=False, k=[]):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        self.training = training

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,6:7])

        pos = x[:, 1:self.dimension+1]
        if training & (self.time_window_noise > 0):
            noise = torch.randn_like(pos) * self.time_window_noise
            pos = pos + noise

        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        if training & self.rotation_augmentation:
            self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
            self.rotation_matrix = torch.stack([torch.stack([torch.cos(self.phi), torch.sin(self.phi)]), torch.stack([-torch.sin(self.phi), torch.cos(self.phi)])])
            d_pos[:, :2] = d_pos[:, :2] @ self.rotation_matrix.T

        # if translation_augmentation:
        #     displacement = torch.randn(1, dimension, dtype=torch.float32, device=device) * 5
        #     displacement = displacement.repeat(pos.shape[0], 1)
        #     pos = pos + displacement
        # if velocity_augmentation:
        #     d_pos = d_pos + torch.randn((1, 2), device=device).repeat(d_pos.shape[0], 1) * vnorm

        particle_id = x[:, 0:1].long()
        embedding = self.a[data_id.long(), particle_id, :].squeeze()

        if self.model == 'PDE_MLPs_A':
            for self.mode in ['kernel_new_features', 'message_passing_kernel', 'update']:
                if self.mode == 'kernel_new_features':
                    new_features = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=torch.zeros_like(embedding))
                elif self.mode == 'message_passing_kernel':
                    out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=new_features)
                    if self.rotation_augmentation & (self.training == True):
                        self.rotation_inv_matrix = torch.stack([torch.stack([torch.cos(self.phi), -torch.sin(self.phi)]), torch.stack([torch.sin(self.phi), torch.cos(self.phi)])])
                        out[:, :2] = out[:, :2] @ self.rotation_inv_matrix.T
                        d_pos[:, :2] = d_pos[:, :2] @ self.rotation_inv_matrix.T
                elif self.mode == 'update':
                    in_features = torch.cat((embedding, d_pos, out), dim=-1)
                    out = self.MLP[3](in_features)

        if self.model == 'PDE_MLPs_B':
            for self.mode in ['encode_features', 'update_features', 'update_features', 'update_features', 'decode_features', 'update']:
                if self.mode == 'encode_features':
                    new_features = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=torch.zeros_like(embedding))
                elif self.mode == 'update_features':
                    new_features = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=new_features)
                elif self.mode == 'decode_features':
                    new_features = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=new_features)
                elif self.mode == 'update':
                    if self.rotation_augmentation & (self.training == True):
                        self.rotation_inv_matrix = torch.stack([torch.stack([torch.cos(self.phi), -torch.sin(self.phi)]), torch.stack([torch.sin(self.phi), torch.cos(self.phi)])])
                        new_features[:, :2] = new_features[:, :2] @ self.rotation_inv_matrix.T
                        d_pos[:, :2] = d_pos[:, :2] @ self.rotation_inv_matrix.T
                    in_features = torch.cat((embedding, d_pos, new_features), dim=-1)
                    out = self.MLP[3](in_features)

        if (self.model == 'PDE_MLPs_C') | (self.model == 'PDE_MLPs_D'):
            for self.mode in ['defined_kernel_features', 'message_passing_defined_kernel']:
                if self.mode == 'defined_kernel_features':
                    new_features = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=torch.zeros_like(embedding))
                elif self.mode == 'message_passing_defined_kernel':
                    out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=new_features)
                    if self.rotation_augmentation & (self.training == True):
                        self.rotation_inv_matrix = torch.stack([torch.stack([torch.cos(self.phi), -torch.sin(self.phi)]), torch.stack([torch.sin(self.phi), torch.cos(self.phi)])])
                        out[:, :2] = out[:, :2] @ self.rotation_inv_matrix.T
            if (self.model == 'PDE_MLPs_D'):
                in_features = torch.cat((embedding, new_features), dim=-1)
                out = self.MLP[1](in_features)

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, embedding_i, embedding_j, new_features_i, new_features_j ):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        if self.training & self.rotation_augmentation:
            delta_pos[:, :2] = delta_pos[:, :2] @ self.rotation_matrix.T
        if self.mode == 'encode_features':
            in_features = torch.cat((embedding_i, embedding_j, delta_pos, d_pos_i, d_pos_j), dim=-1)
            new_features = self.MLP[0](in_features)
            return new_features
        if self.mode == 'update_features':
            in_features = torch.cat((new_features_i, new_features_j), dim=-1)
            new_features = self.MLP[1](in_features)
            return new_features
        if self.mode == 'decode_features':
            in_features = torch.cat((new_features_i, new_features_j), dim=-1)
            new_features = self.MLP[2](in_features)
            return new_features
        if self.mode == 'defined_kernel_features':
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
            density = Gaussian_kernel
            return density.clone().detach()
        if self.mode == 'message_passing_defined_kernel':
            in_features = torch.cat((d_pos_i - d_pos_j, embedding_i, embedding_j, new_features_i, new_features_j, self.kernel_operators['grad_triangle'], self.kernel_operators['Gaussian']), dim=-1)
            out = self.MLP[0](in_features) / new_features_j.repeat(1, 2)
        if self.mode == 'kernel_new_features':
            self.kernels = self.MLP[0](delta_pos)
            in_features = torch.cat((embedding_i, self.kernels), dim=-1)
            new_features = self.MLP[1](in_features)
            return new_features
        elif self.mode == 'message_passing_kernel':
            in_features = torch.cat((embedding_i, embedding_j, delta_pos, d_pos_i, d_pos_j, new_features_i, new_features_j, self.kernels), dim=-1)
            out = self.MLP[2](in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

        mgrid = np.mgrid[-self.max_radius:self.max_radius:100j, -self.max_radius:self.max_radius:100j]
        mgrid = torch.tensor(mgrid, device=self.device).permute(2, 1, 0).reshape(-1, 2).float()
        kernels = self.MLP[0](mgrid)

        matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(20, 5))
        for k in range(self.kernels.shape[1]):
            ax = fig.add_subplot(1, self.kernels.shape[1], k + 1)
            plt.scatter(to_numpy(delta_pos[:, 0]), to_numpy(delta_pos[:, 1]), c=to_numpy(self.kernels[:, k]), s=5,cmap='viridis')
            ax.set_title(f'kernel {k}')
            ax.set_xlim(-self.max_radius, self.max_radius)
            ax.set_ylim(-self.max_radius, self.max_radius)
            ax.set_aspect('equal')
            plt.colorbar()
        plt.tight_layout()
        fig = plt.figure(figsize=(20, 5))
        for k in range(self.kernels.shape[1]):
            ax = fig.add_subplot(1, self.kernels.shape[1], k + 1)
            plt.scatter(to_numpy(mgrid[:, 0]), to_numpy(mgrid[:, 1]), c=to_numpy(kernels[:, k]), s=5,
                        cmap='viridis')
            ax.set_title(f'kernel {k}')
            ax.set_xlim(-self.max_radius, self.max_radius)
            ax.set_ylim(-self.max_radius, self.max_radius)
            ax.set_aspect('equal')
            plt.colorbar()
        plt.tight_layout()
        plt.show()


