import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.MLP import MLP
from ParticleGraph.utils import to_numpy


class GravityParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the acceleration of particles as a function of their relative distance and masses.
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

    def __init__(self, aggr_type=[], model_config=[], device=[], bc_diff=[]):

        super(GravityParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.upgrade_type = model_config['upgrade_type']
        self.ndataset = model_config['nrun'] - 1
        self.clamp = model_config['clamp']
        self.pred_limit = model_config['pred_limit']
        self.bc_diff = bc_diff

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id, training, vnorm, phi):

        self.data_id = data_id
        self.vnorm = vnorm
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pos = x[:, 1:3]
        d_pos = x[:, 3:5]
        particle_id = x[:, 0:1]

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, particle_id=particle_id)

        return pred

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, particle_id_i, particle_id_j):
        # squared distance
        r = torch.sqrt(torch.sum(self.bc_diff(pos_j - pos_i) ** 2, axis=1)) / self.radius
        delta_pos = self.bc_diff(pos_j - pos_i) / self.radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm

        if (self.data_augmentation) & (self.training == True):
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

        embedding = self.a[self.data_id, to_numpy(particle_id_j), :].squeeze()
        in_features = torch.cat(
            (delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None], dpos_y_j[:, None], embedding),
            dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        psi = p / r ** 2
        return psi[:, None]
