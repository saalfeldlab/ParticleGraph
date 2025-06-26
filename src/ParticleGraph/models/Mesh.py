import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *


class Mesh(pyg.nn.MessagePassing):
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
        super(Mesh, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.mesh_model_name
        self.max_radius = simulation_config.max_radius

        self.embedding_dim = model_config.embedding_dim
        self.nparticles = simulation_config.n_particles
        self.ndataset = config.training.n_runs
        self.bc_dpos = bc_dpos
        self.noise_model_level = train_config.noise_model_level
        self.rotation_augmentation = train_config.rotation_augmentation
        self.field_type = model_config.field_type


        if (self.model == 'RD_Mesh2') | (self.model == 'RD_Mesh3') | (self.model == 'RD_Mesh4'):
            self.lin_edge = MLP(input_size=model_config.input_size, output_size=model_config.output_size, nlayers=model_config.n_layers,
                                hidden_size=model_config.hidden_dim, device=self.device)

        if (self.model == 'RD_Mesh3'):
            self.siren = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                      hidden_layers=model_config.n_layers_nnr, outermost_linear=True,first_omega_0=model_config.omega,hidden_omega_0=model_config.omega)
            self.siren = self.siren.to(self.device)

        self.lin_phi = MLP(input_size=model_config.input_size_update, output_size=model_config.output_size_update, nlayers=model_config.n_layers,
                       hidden_size=model_config.hidden_dim_update, device=self.device)


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
            if (self.has_field) & ('replace_blue' in self.field_type):
                x[:, 8:9] = field
        else:
            field = torch.zeros_like(x[:, 9:10])

        if self.rotation_augmentation:
            self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
            self.rotation_matrix = torch.stack([
                torch.stack([torch.cos(self.phi), torch.sin(self.phi)]),
                torch.stack([-torch.sin(self.phi), torch.cos(self.phi)])
            ])
            self.rotation_matrix = self.rotation_matrix.permute(*torch.arange(self.rotation_matrix.ndim - 1, -1, -1))

        match self.model:
            case 'RD_Mesh':
                self.step = 0
                laplacian_uvw = self.propagate(edge_index, uvw=uvw, pos=pos, embedding=embedding, discrete_laplacian=edge_attr)
                self.laplacian_uvw = laplacian_uvw
                input_phi = torch.cat((laplacian_uvw, uvw, embedding), dim=-1)
                if self.noise_model_level > 0:
                    noise = torch.randn_like(input_phi[:, 0:6]) * self.noise_model_level
                    input_phi[:, 0:6] = input_phi[:, 0:6] + noise
            case 'RD_Mesh2':
                self.step = 0
                laplacian_uvw = self.propagate(edge_index, uvw=uvw, pos=pos, embedding=embedding, discrete_laplacian=edge_attr)
                self.laplacian_uvw = laplacian_uvw
                self.step = 1
                uvw_msg = self.propagate(edge_index, uvw=uvw, pos=pos, embedding=embedding, discrete_laplacian=edge_attr)
                input_phi = torch.cat((laplacian_uvw, uvw, uvw_msg, embedding), dim=-1)
                if self.noise_model_level > 0:
                    noise = torch.randn_like(input_phi[:,0:9]) * self.noise_model_level
                    input_phi[:,0:9] = input_phi[:,0:9] + noise
            case 'RD_Mesh3':
                self.step = 2
                uvw_msg = self.propagate(edge_index, uvw=uvw, pos=pos, embedding=embedding, discrete_laplacian=edge_attr)
                input_phi = torch.cat((uvw, uvw_msg, embedding), dim=-1)
                if self.noise_model_level > 0:
                    noise = torch.randn_like(input_phi[:, 0:9]) * self.noise_model_level
                    input_phi[:, 0:9] = input_phi[:, 0:9] + noise
            case 'RD_Mesh4':
                self.step = 3
                uvw_msg = self.propagate(edge_index, uvw=uvw, pos=pos, embedding=embedding, discrete_laplacian=edge_attr)
                input_phi = torch.cat((uvw, uvw_msg, embedding), dim=-1)
                if self.noise_model_level > 0:
                    noise = torch.randn_like(input_phi[:, 0:9]) * self.noise_model_level
                    input_phi[:, 0:9] = input_phi[:, 0:9] + noise

        if (self.has_field) & ('additive' in self.field_type):
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
        elif self.step == 2:
            delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
            if self.rotation_augmentation & (self.training == True):
                delta_pos[:, :2] = delta_pos[:, :2] @ self.rotation_matrix
            self.kernel = self.siren(delta_pos)
            in_features = torch.cat((uvw_j, embedding_i), dim=-1)
            out = torch.cat((self.lin_edge(in_features), self.kernel*uvw_j), dim=-1)
            return out
        elif self.step == 3:
            delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
            if self.rotation_augmentation & (self.training == True):
                delta_pos[:, :2] = delta_pos[:, :2] @ self.rotation_matrix
            in_features = torch.cat((uvw_j, embedding_i), dim=-1)
            return self.lin_edge(in_features)


            fig = plt.figure(figsize=(10, 10))
            plt.scatter(to_numpy(delta_pos[:, 0]), to_numpy(delta_pos[:, 1]), s=1000, c=to_numpy(discrete_laplacian[:,None]), alpha=0.5)

            fig = plt.figure(figsize=(10, 10))
            plt.scatter(to_numpy(delta_pos[:, 0]), to_numpy(delta_pos[:, 1]), s=1000, c=to_numpy(self.kernel[:,0]), alpha=0.5)

            # fig = plt.figure(figsize=(10, 10))
            # plt.scatter(to_numpy(delta_pos[:, 0]), to_numpy(delta_pos[:, 1]), s=1, c='k', alpha=0.5)

    def update(self, aggr_out):
        return aggr_out

    def psi(self, r, p):
        return p * r
