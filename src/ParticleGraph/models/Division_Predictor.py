import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ParticleGraph.MLP import MLP
from ParticleGraph.utils import to_numpy

class Division_Predictor(nn.Module):

    def __init__(self, config, device):
        super(Division_Predictor, self).__init__()
        self.mlp = MLP(input_size=config.graph_model.division_predictor_input_size,
                       output_size=config.graph_model.division_predictor_output_size,
                       nlayers=config.graph_model.division_predictor_n_layers,
                       hidden_size=config.graph_model.division_predictor_hidden_dim,
                       device=device)

        self.sigmoid = nn.Sigmoid()

        self.n_dataset = config.training.n_runs

        self.t = nn.Parameter(torch.tensor(np.ones((self.n_dataset, 20500, 2)), device=device, requires_grad=True, dtype=torch.float32))

    def forward(self, x, data_id):

        self.data_id = data_id
        particle_id = x[:, 0:1]
        time_embedding = self.t[self.data_id, to_numpy(particle_id), :].squeeze()

        x = torch.concatenate((x[:,1:2], time_embedding), dim=1)
        x = self.mlp(x)
        # x = self.sigmoid(x)

        return x