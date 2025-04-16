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

        mlp_params = model_config.multi_mlp_params

        self.MLP=[]
        for params in mlp_params:
            self.MLP.append(MLP(input_size=params[0], output_size=params[3], nlayers=params[2], hidden_size=params[1], device=self.device))

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_particles) , self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))


    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False, k=[]):

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

        particle_id = x[:, 0:1].long()
        embedding = self.a[data_id.long(), particle_id, :].squeeze()

        for self.mode in ['kernel_new_features', 'message_passing', 'update']:
            if self.mode == 'kernel_new_features':
                new_features = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, new_features=torch.zeros_like(embedding))
            elif self.mode == 'message_passing':
                out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding,new_features=new_features)
            elif self.mode == 'update':
                in_features = torch.cat((embedding, d_pos, out), dim=-1)
                pred = self.MLP[3](in_features)

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, embedding_i, embedding_j, new_features_i, new_features_j ):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'kernel_new_features':

            self.kernels = self.MLP[0](d_pos_j - d_pos_i)
            in_features = torch.cat((embedding_i, self.kernels), dim=-1)
            new_features = self.MLP[1](in_features)

            return new_features

        elif self.mode == 'message_passing':

            in_features = torch.cat((embedding_i, embedding_j, pos_j-pos_i, d_pos_i, d_pos_j, new_features_i, new_features_j, self.kernels), dim=-1)
            out = self.MLP[2](in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


