import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy


class Mesh_RPS(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the first derivative of a scalar field on a mesh.
    The node embedding is defined by a table self.a
    Note the Laplacian coeeficients are in data.edge_attr

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the first derivative of a scalar field on a mesh (dimension 3).
    """

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None):
        super(Mesh_RPS, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.mesh_model_name

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_size = model_config.hidden_dim
        self.nlayers = model_config.n_mp_layers

        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update
        self.n_layers_update = model_config.n_layers_update

        self.embedding_dim = model_config.embedding_dim
        self.nparticles = simulation_config.n_particles
        self.ndataset = config.training.n_runs
        self.bc_dpos = bc_dpos
        self.time_window_noise = train_config.time_window_noise
        self.rotation_augmentation = train_config.rotation_augmentation


        if self.model == 'RD_RPS_Mesh2':
            self.lin_edge = MLP(input_size=8, output_size=self.output_size, nlayers=self.nlayers,
                                hidden_size=self.hidden_size, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.nlayers,
                       hidden_size=self.hidden_dim_update, device=self.device)


        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.ndataset), int(self.nparticles), self.embedding_dim)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[], has_field=False, return_all=False):
        self.data_id = data_id
        self.has_field = has_field
        self.training = training
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        uvw = data.x[:, 6:9]
        pos = x[:, 1:3]
        particle_id = x[:, 0:1].long()
        embedding = self.a[self.data_id, particle_id, :].squeeze()

        if self.has_field:
            field = x[:, 9:10]
        else:
            field = torch.zeros_like(x[:, 9:10])

        if self.rotation_augmentation:
            self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
            self.rotation_matrix = torch.stack([
                torch.stack([torch.cos(self.phi), torch.sin(self.phi)]),
                torch.stack([-torch.sin(self.phi), torch.cos(self.phi)])
            ])
            self.rotation_matrix = self.rotation_matrix.permute(*torch.arange(self.rotation_matrix.ndim - 1, -1, -1))

        self.step = 0
        laplacian_uvw = self.propagate(edge_index, uvw=uvw, pos=pos, embedding=embedding, discrete_laplacian=edge_attr)
        self.laplacian_uvw = laplacian_uvw

        if self.model == 'RD_RPS_Mesh2':
            self.step = 1
            uvw_msg = self.propagate(edge_index, uvw=uvw, pos=pos, embedding=embedding, discrete_laplacian=edge_attr)
            input_phi = torch.cat((laplacian_uvw, uvw, uvw_msg, embedding), dim=-1)
            if self.time_window_noise > 0:
                noise = torch.randn_like(input_phi[:,0:9]) * self.time_window_noise
                input_phi[:,0:9] = input_phi[:,0:9] + noise
        else:
            input_phi = torch.cat((laplacian_uvw, uvw, embedding), dim=-1)
            if self.time_window_noise > 0:
                noise = torch.randn_like(input_phi[:,0:6]) * self.time_window_noise
                input_phi[:,0:6] = input_phi[:,0:6] + noise

        if self.has_field:
            input_phi = torch.cat((input_phi, field), dim=-1)

        pred = self.lin_phi(input_phi)

        if return_all:
            return pred, laplacian_uvw, uvw, embedding, input_phi
        else:
            return pred

    def message(self, uvw_j, pos_i, pos_j, embedding_i, discrete_laplacian):

        if self.step == 0:
            return discrete_laplacian[:,None] * uvw_j
        elif self.step == 1:
            r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1))
            delta_pos = self.bc_dpos(pos_j - pos_i)
            if self.rotation_augmentation & (self.training == True):
                delta_pos[:, :2] = delta_pos[:, :2] @ self.rotation_matrix
            in_features = torch.cat((uvw_j, delta_pos, r[:,None], embedding_i), dim=-1)
            return self.lin_edge(in_features)

    def update(self, aggr_out):
        return aggr_out

    def psi(self, r, p):
        return p * r
