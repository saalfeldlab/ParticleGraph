import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize


class Interaction_Falling_Water_Wall(pyg.nn.MessagePassing):
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

        super(Interaction_Falling_Water_Wall, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_mp_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.time_window_noise = train_config.time_window_noise
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update
        self.model_type = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.time_window = train_config.time_window
        self.model_density = model_density
        self.sub_sampling = simulation_config.sub_sampling
        self.prediction = model_config.prediction

        if self.update_type == 'pre_mlp':
            self.pre_lin_edge = MLP(input_size=self.pre_input_size, output_size=self.pre_output_size, nlayers=self.pre_n_layers,
                                hidden_size=self.pre_hidden_dim, device=self.device)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if 'mlp' in self.update_type:
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False):

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.time_window == 0:
            particle_id = x[:, 0:1]
            embedding = self.a[data_id, to_numpy(particle_id), :].squeeze()
            pos = x[:, 1:self.dimension+1]

        else:
            particle_id = x[0][:, 0:1]
            embedding = self.a[data_id, to_numpy(particle_id), :].squeeze()
            x = torch.stack(x)
            pos = x[:, :, 1:self.dimension + 1]
            pos = pos.transpose(0, 1)
            pos = torch.reshape(pos, (pos.shape[0], pos.shape[1] * pos.shape[2]))

        if training & (self.time_window_noise > 0):
            noise = torch.randn_like(pos) * self.time_window_noise
            pos = pos + noise

        if self.update_type == 'pre_mlp':
            self.mode = 'pre_mlp'
            kernel_null = torch.zeros((pos.shape[0], 2), device=self.device)
            kernel = self.propagate(edge_index=edge_index, pos=pos, embedding=embedding, kernel=kernel_null)
            self.mode = 'mlp'
            pred = self.propagate(edge_index=edge_index, pos=pos, embedding=embedding, kernel=kernel)
            pos_p = (pos - pos[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
            out = self.lin_phi(torch.cat((pred, embedding, pos_p), dim=-1))
        elif self.update_type == 'mlp':
            kernel_null = torch.zeros((pos.shape[0], 2), device=self.device)
            pred = self.propagate(edge_index=edge_index, pos=pos, embedding=embedding, kernel=kernel_null)
            pos_p = (pos - pos[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
            out = self.lin_phi(torch.cat((pred, embedding, pos_p), dim=-1))
        else:
            out = pred

        if self.sub_sampling>1:

            pred = out
            d_pos = x[:, :, self.dimension + 1:1 + 2 * self.dimension]
            d_pos = d_pos.transpose(0, 1)
            d_pos = torch.reshape(d_pos, (d_pos.shape[0], d_pos.shape[1] * d_pos.shape[2]))

            for k in range(self.sub_sampling):
                if self.prediction == '2nd_derivative':
                    y = pred * self.ynorm * self.delta_t / self.sub_sampling
                    d_pos = d_pos + y  # speed update
                else:
                    y = pred * self.vnorm
                    d_pos = y
                pos = pos + d_pos * self.delta_t / self.sub_sampling

                out = pos

                if self.update_type == 'mlp':
                    pos_p = (pos - pos[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
                    pred = self.lin_phi(torch.cat((self.propagate(edge_index, pos=pos, embedding=embedding), embedding, pos_p), dim=-1))
                else:
                    pred = self.propagate(edge_index, pos=pos, embedding=embedding)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, embedding_i, embedding_j, kernel_j):

        if self.update_type == 'pre_mlp':
            if self.mode == 'pre_mlp':
                delta_pos = self.bc_dpos(pos_j - pos_i)
                delta_pos = delta_pos[:, 0:self.dimension]
                self.W_ijs = self.pre_lin_edge(torch.cat((delta_pos, embedding_i, embedding_j), dim=-1))
                return self.W_ijs[:,0:3]
            else:
                if self.time_window == 0:
                    delta_pos = self.bc_dpos(pos_j - pos_i)
                    in_features = torch.cat((delta_pos, embedding_i, embedding_j, kernel_j, self.W_ijs[:,1:]), dim=-1)
                else:
                    pos_i_p = (pos_i - pos_i[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
                    pos_j_p = (pos_j - pos_i[:, 0:self.dimension].repeat(1, self.time_window))
                    in_features = torch.cat((pos_i_p, pos_j_p, embedding_i, embedding_j, kernel_j, self.W_ijs[:,1:]), dim=-1)
                out = self.lin_edge(in_features)
                return out
        else:
            if self.time_window == 0:
                delta_pos = self.bc_dpos(pos_j - pos_i)
                in_features = torch.cat((delta_pos, embedding_i, embedding_j), dim=-1)
            else:
                pos_i_p = (pos_i - pos_i[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
                pos_j_p = (pos_j - pos_i[:, 0:self.dimension].repeat(1, self.time_window))
                in_features = torch.cat((pos_i_p, pos_j_p, embedding_i, embedding_j), dim=-1)
            out = self.lin_edge(in_features)
            return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


