import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
from ParticleGraph.models import Siren_Network


class Interaction_Mouse(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the mouse movements as a function of their relative distance.
    The interaction function is defined by a MLP self.lin_edge
    The particle (mouse) embedding is defined by a table self.a

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the velocity of the particles (mouses) (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_Mouse, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.n_frames = simulation_config.n_frames
        self.n_nodes = simulation_config.n_nodes
        self.n_nodes_per_axis = int(np.sqrt(self.n_nodes))
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.sigma = simulation_config.sigma
        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.n_particles_max = simulation_config.n_particles_max
        self.ctrl_tracking = train_config.ctrl_tracking

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        # self.a = nn.Parameter(torch.tensor(np.ones((self.n_particles_max, self.embedding_dim)), device=self.device,requires_grad=True, dtype=torch.float32))

        self.a = nn.Parameter(torch.ones((int(self.n_particles), 1001, self.embedding_dim), device=self.device, requires_grad=True,
                                         dtype=torch.float32) * 0.44)

        self.embedding_step = self.n_frames // 1000

        if self.update_type != 'none':
            self.lin_update = MLP(input_size=self.output_size + self.embedding_dim + 2, output_size=self.output_size,
                                  nlayers=self.n_layers_update, hidden_size=self.hidden_dim_update, device=self.device)

    def get_interp_a(self, k, particle_id):

        alpha = (k % self.embedding_step) / self.embedding_step

        return alpha * self.a[particle_id, k // self.embedding_step + 1,:] + (1 - alpha) * self.a[particle_id,k // self.embedding_step,:]

    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False, k=0):

        self.data_id = data_id
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.has_field = has_field
        self.training = training

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,6:7])

        particle_id = x[:, -1][:, None].long()
        embedding = self.get_interp_a(k, particle_id).squeeze()

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, embedding=embedding, field=field)

        return pred

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, field_j):
        # squared distance
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius

        if self.rotation_augmentation & (self.training == True):
            new_delta_pos_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_delta_pos_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_delta_pos_x
            delta_pos[:, 1] = new_delta_pos_y

        in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)

        out = self.lin_edge(in_features) * field_j

        self.pos = pos_i
        self.msg = out

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


