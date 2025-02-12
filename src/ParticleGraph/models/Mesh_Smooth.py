import numpy as np
import torch
import torch.nn as nn
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *

import torch_geometric as pyg


def density_laplace(y, x):
    grad = density_gradient(y, x)
    return density_divergence(grad, x)
def density_divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div
def density_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class Mesh_Smooth(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the second derivative of a scalar field on a mesh.
    The node embedding is defined by a table self.a
    Note the Laplacian coeeficients are in data.edge_attr

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the second derivative of a scalar field on a mesh (dimension 1).
    """

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None):
        super(Mesh_Smooth, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.embedding = model_config.embedding_dim
        self.n_particles = simulation_config.n_particles
        self.n_datasets = config.training.n_runs
        self.bc_dpos = bc_dpos
        self.input_size_nnr= model_config.input_size_nnr
        self.output_size_nnr= model_config.output_size_nnr
        self.hidden_dim_nnr = model_config.hidden_dim_nnr
        self.omega = model_config.omega

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                           hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.n_datasets), int(self.n_particles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

        self.siren = Siren_Network(image_width=100, in_features=self.input_size_nnr,
                                out_features=self.output_size_nnr,
                                hidden_features=self.hidden_dim_nnr,
                                hidden_layers=3, outermost_linear=True, device=self.device, first_omega_0=self.omega,
                                hidden_omega_0=self.omega )



    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        particle_id = x[:, 0:1].long()
        embedding = self.a[data_id, particle_id, :].squeeze()
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        field = x[:, 6:7]

        density_null = torch.zeros((pos.shape[0], 2), device=self.device)
        self.mode = 'pre_mlp'
        self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
        self.mode = 'mlp'
        laplacian_u = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=self.density)
        pred = self.lin_phi(torch.cat((laplacian_u, embedding), dim=-1))

        self.laplacian_u = laplacian_u

        return pred


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, embedding_i, embedding_j, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'pre_mlp':

            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True
            density_kernel = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:,None]

            self.modulation = self.siren(coords=mgrid) * max_radius **2
            kernel_modified = torch.exp(-2*(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20*self.kernel_var))[:, None] * self.modulation

            grad_autograd = -density_gradient(kernel_modified, mgrid)
            laplace_autograd = density_laplace(kernel_modified, mgrid)

            self.kernel_operators = torch.cat((kernel_modified, grad_autograd, laplace_autograd), dim=-1)

            return density_kernel

            # kernel_modified = torch.exp(-2 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20*self.kernel_var))[:, None]
            # fig = plt.figure(figsize=(6, 6))
            # plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=10, c=to_numpy(kernel_modified))
            # plt.show()

        else:

            out = field_j * self.kernel_operators[:, 3:4] / density_j

            return out

    def message(self, u_j, discrete_laplacian):
        L = discrete_laplacian[:,None] * u_j

        return L

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r
