from prettytable import PrettyTable
import torch.nn as nn
import torch.nn.functional as F

from ParticleGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import os
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

class MLP(nn.Module):

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

        if activation=='tanh':
            self.activation = F.tanh
        else:
            self.activation = F.relu

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

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

    def __init__(self, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_Particle, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = 'cuda:0'
        self.input_size = 9
        self.output_size = 2
        self.hidden_dim = 256
        self.n_layers = 5
        self.n_particles = 1792
        self.n_particle_types = 16
        self.max_radius = 0.04
        self.data_augmentation = True
        self.noise_level = 0.0
        self.embedding_dim = 2
        self.n_dataset = 2
        self.prediction = '2nd_derivative'
        self.n_particles_max = 20000
        self.update_type = 'none'
        self.n_layers_update = 3
        self.hidden_dim_update = 64
        self.sigma = 0.005
        self.model = 'PDE_B'
        self.n_ghosts = 0
        self.dimension = 2
        self.has_state = False
        self.n_frames = 8000
        self.state_hot_encoding = False

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)),
                         device=self.device,
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

        if has_field:
            field = x[:, 6:7]
        else:
            field = torch.ones_like(x[:, 6:7])

        pos = x[:, 1:self.dimension + 1]
        d_pos = x[:, self.dimension + 1:1 + 2 * self.dimension]
        particle_id = x[:, 0:1]

        if not (self.state_hot_encoding):
            embedding = self.a[self.data_id, frame, to_numpy(particle_id), :].squeeze()
        else:
            # model_a = gumbel_softmax(self.a[self.data_id, frame, to_numpy(particle_id), :].squeeze(), self.temperature, hard=True, device=self.device)
            # embedding = torch.matmul(model_a, self.b)

            model_a = torch.softmax(self.a[self.data_id, frame, to_numpy(particle_id), :].squeeze(), dim=1)
            model_a = gumbel_softmax(model_a, self.temperature, hard=True, device=self.device)
            mu = torch.matmul(model_a, self.mu)
            logvar = torch.matmul(model_a, self.logvar.repeat(self.n_particle_types))
            logvar = logvar[:, None].repeat(1, 2)
            embedding = reparameterize(mu, logvar)

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, embedding=embedding, field=field)

        if self.update_type == 'linear':
            embedding = self.a[self.data_id, to_numpy(particle_id), :].squeeze()
            pred = self.lin_update(torch.cat((pred, x[:, 3:5], embedding), dim=-1))

        if self.update_type == 'embedding_Siren':
            embedding = self.b[self.data_id, to_numpy(particle_id), :].squeeze()
            in_features = torch.cat((x[:, 8:9] / 250, embedding), dim=-1)
            self.phi_ = self.phi(in_features).repeat(1, 2)
            pred = pred * self.phi_

        return pred

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, field_j):
        # distance normalized by the max radius
        r = torch.sqrt(torch.sum((pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm

        if self.data_augmentation & (self.training == True):
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
            case 'PDE_A' | 'PDE_ParticleField_A':
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
                in_features = torch.cat((delta_pos, r[:, None], 10 ** embedding_j), dim=-1)
            case 'PDE_E':
                in_features = torch.cat(
                    (delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)

        out = self.lin_edge(in_features) * field_j

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p1, p2):

        if (self.model == 'PDE_A') | (self.model == 'PDE_A_bis') | (self.model == 'PDE_ParticleField_A'):
            return r * (p1[0] * torch.exp(-torch.abs(r) ** (2 * p1[1]) / (2 * self.sigma ** 2)) - p1[2] * torch.exp(
                -torch.abs(r) ** (2 * p1[3]) / (2 * self.sigma ** 2)))
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


if __name__ == '__main__':

    device = 'cuda:0'
    print(f'device {device}')

    model = Interaction_Particle(device=device, aggr_type='add', bc_dpos=None, dimension=2)

    net = "model_20_epoch.pt"
    print(f'Loading existing model {net}...')
    state_dict = torch.load(net,map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    table = PrettyTable(["Modules", "Parameters"])
    n_total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        n_total_params += param
    print(table)
    print(f"Total Trainable Params: {n_total_params}")

    # matplotlib.use("Qt5Agg")

    max_radius = 0.04
    rr = torch.tensor(np.linspace(0, max_radius, 1000),device=device)

    n_particles = 1792

    print('interaction functions ...')
    fig = plt.figure(figsize=(8, 8))
    for n in range(n_particles):
        embedding_ = model.a[1, n, :] * torch.ones((1000, 2), device=device)

        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        if (n % 5 == 0) :
            plt.plot(rr.detach().cpu(), func.detach().cpu(), 2, color='k', linewidth=2, alpha=0.25)
    plt.show()




