import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from ParticleGraph.MLP import MLP
from ParticleGraph.utils import to_numpy


class Mesh_Laplacian(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=[], model_config=[], device=[], bc_diff=[]):
        super(Mesh_Laplacian, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.embedding = model_config['embedding']
        self.nparticles = model_config['nparticles']
        self.ndataset = model_config['nrun'] - 1
        self.bc_diff = bc_diff

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                           hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.ndataset), int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        u = x[:, 6:7]

        laplacian = self.propagate(edge_index, u=u, discrete_laplacian=edge_attr)

        particle_id = to_numpy(x[:, 0])
        embedding = self.a[self.data_id, particle_id, :]

        pred = self.lin_phi(torch.cat((laplacian, embedding), dim=-1))

        return pred

    def message(self, u_j, discrete_laplacian):
        L = discrete_laplacian[:,None] * u_j

        return L

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r
