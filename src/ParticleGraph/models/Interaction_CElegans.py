import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.MLP import MLP
from ParticleGraph.utils import to_numpy


class Interaction_CElegans(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, config, device, bc_dpos=None):

        super(Interaction_CElegans, self).__init__(aggr='mean')

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = simulation_config.input_size
        self.output_size = simulation_config.output_size
        self.hidden_dim = simulation_config.hidden_dim
        self.nlayers = simulation_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.max_radius = simulation_config.max_radius
        self.data_augmentation = train_config.data_augmentation
        self.noise_level = simulation_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.ndataset = train_config.n_runs - 1
        self.update_type = model_config.update_type
        self.prediction = model_config.prediction
        self.update_type = model_config.update_type
        self.nlayers_update = model_config.n_layers_update
        self.hidden_size_update = model_config.hidden_dim_update
        self.bc_dpos = bc_dpos

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.n_particles + 1), self.embedding_dim)), device=self.device,
                         requires_grad=True, dtype=torch.float64))

        if self.update_type == 'linear':
            self.lin_update = MLP(input_size=self.output_size + self.embedding_dim + 2, output_size=self.output_size,
                                  nlayers=self.nlayers_update, hidden_size=self.hidden_size_update, device=self.device)

        self.to(device=self.device)
        self.to(torch.float64)

    def forward(self, data, data_id, time):

        self.data_id = data_id

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pred = self.propagate(edge_index, x=(x, x), time=time)

        if self.update_type == 'linear':
            embedding = self.a[self.data_id, to_numpy(x[:, 0]), :]
            pred = self.lin_update(torch.cat((pred, x[:, 3:5], embedding), dim=-1))

        return pred

    def message(self, x_i, x_j, time):

        r = torch.sqrt(torch.sum(self.bc_dpos(x_i[:, 1:4] - x_j[:, 1:4]) ** 2, axis=1))  # squared distance
        r = r[:, None]

        delta_pos = self.bc_dpos(x_i[:, 1:4] - x_j[:, 1:4])
        embedding = self.a[self.data_id, to_numpy(x_i[:, 0]).astype(int), :]
        in_features = torch.cat((delta_pos, r, x_i[:, 4:7], x_j[:, 4:7], embedding, time[:, None]), dim=-1)

        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)
