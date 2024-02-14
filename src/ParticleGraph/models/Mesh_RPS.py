import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from ParticleGraph.MLP import MLP
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

    def __init__(self, aggr_type=[], model_config=[], device=[], bc_dpos=[]):
        super(Mesh_RPS, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.embedding = model_config['embedding']
        self.nparticles = model_config['nparticles']
        self.ndataset = model_config['nrun'] - 1
        self.bc_dpos = bc_dpos

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                           hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.ndataset), int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        uvw = data.x[:, 6:9]

        laplacian_uvw = self.propagate(edge_index, uvw=uvw, discrete_laplacian=edge_attr)

        particle_id = to_numpy(x[:, 0])
        embedding = self.a[self.data_id, particle_id, :]

        input_phi = torch.cat((laplacian_uvw, uvw, embedding), dim=-1)

        pred = self.lin_phi(input_phi)

        return pred

    def message(self, uvw_j, discrete_laplacian):
        return discrete_laplacian[:,None] * uvw_j

        return Laplace

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r
