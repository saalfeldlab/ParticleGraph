import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize


class Interaction_Particle3(pyg.nn.MessagePassing):
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

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2, model_density=[]):

        super(Interaction_Particle3, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.model = model_config.particle_model_name

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_mp_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers

        self.input_size_decoder = model_config.input_size_decoder
        self.output_size_decoder = model_config.output_size_decoder
        self.hidden_dim_decoder = model_config.hidden_dim_decoder
        self.n_layers_decoder = model_config.n_mp_layers_decoder
        
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update

        self.n_dataset = train_config.n_runs
        self.n_particles = simulation_config.n_particles
        self.embedding_dim = model_config.embedding_dim
        self.dimension = dimension
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.bc_dpos = bc_dpos
        self.rotation_augmentation = train_config.rotation_augmentation

        self.time_window = train_config.time_window
        self.time_window_noise = train_config.time_window_noise
        self.sub_sampling = simulation_config.sub_sampling
        self.prediction = model_config.prediction

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        self.lin_edge2 = MLP(input_size=self.output_size + 4, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        self.lin_decoder = MLP(input_size=self.input_size_decoder, output_size=self.output_size_decoder, nlayers=self.n_layers_decoder,
                            hidden_size=self.hidden_dim_decoder, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles), self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[], has_field=False, k =[]):

        self.data_id = data_id
        self.training = training
        self.has_field = has_field

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.rotation_augmentation & self.training == True:
            self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
            self.rotation_matrix = torch.stack([
                torch.stack([torch.cos(self.phi), torch.sin(self.phi)]),
                torch.stack([-torch.sin(self.phi), torch.cos(self.phi)])
            ])
            self.rotation_matrix = self.rotation_matrix.permute(*torch.arange(self.rotation_matrix.ndim - 1, -1, -1)).squeeze()

        if self.time_window == 0:
            particle_id = x[:, 0:1].long()
            embedding = self.a[self.data_id.clone().detach(), particle_id, :].squeeze()
            pos = x[:, 1:self.dimension+1]
        else:
            particle_id = x[0][:, 0:1].long()
            embedding = self.a[self.data_id.clone().detach(), particle_id, :].squeeze()
            x = torch.stack(x)
            pos = x[:, :, 1:self.dimension + 1]
            pos = pos.transpose(0, 1)
            pos = torch.reshape(pos, (pos.shape[0], pos.shape[1] * pos.shape[2]))

        if training & (self.time_window_noise > 0):
            noise = torch.randn_like(pos) * self.time_window_noise
            pos = pos + noise

        self.step = 0
        pred = self.propagate(edge_index=edge_index, pos=pos, embedding=embedding)
        self.step = 1
        pred = self.propagate(edge_index=edge_index, pos=pred, embedding=embedding)


        pred = self.lin_decoder(pred)
        if self.rotation_augmentation & self.training:
            self.rotation_inv_matrix = torch.stack([torch.stack([torch.cos(self.phi), -torch.sin(self.phi)]),torch.stack([torch.sin(self.phi), torch.cos(self.phi)])])
            self.rotation_inv_matrix = self.rotation_inv_matrix.permute(*torch.arange(self.rotation_inv_matrix.ndim - 1, -1, -1)).squeeze()
            pred = pred @ self.rotation_inv_matrix

        if has_field:
            out = self.lin_phi(torch.cat((pred, embedding, field), dim=-1))
        else:
            out = self.lin_phi(torch.cat((pred, embedding), dim=-1))
        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, embedding_i, embedding_j):

        if self.step == 0:
            if self.time_window == 0:
                delta_pos = self.bc_dpos(pos_j - pos_i)
                if self.rotation_augmentation & (self.training == True):
                    delta_pos = delta_pos @ self.rotation_matrix
                in_features = torch.cat((delta_pos, embedding_i, embedding_j), dim=-1)
            else:
                pos_i_p = (pos_i - pos_i[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
                pos_j_p = (pos_j - pos_i[:, 0:self.dimension].repeat(1, self.time_window))
                if self.rotation_augmentation & (self.training == True):
                    for k in range(pos_i_p.shape[1]//2):
                        pos_i_p[:, k*2:(k+1)*2] = pos_i_p[:, k*2:(k+1)*2] @ self.rotation_matrix
                    for k in range(pos_j_p.shape[1] // 2):
                        pos_j_p[:, k * 2:(k + 1) * 2] = pos_j_p[:, k * 2:(k + 1) * 2] @ self.rotation_matrix
                in_features = torch.cat((pos_i_p, pos_j_p, embedding_i, embedding_j), dim=-1)
            out = self.lin_edge(in_features)
            return out

        elif self.step == 1:
            delta_pos = pos_j - pos_i
            in_features = torch.cat((delta_pos, embedding_i, embedding_j), dim=-1)
            out = self.lin_edge2(in_features)
            return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


