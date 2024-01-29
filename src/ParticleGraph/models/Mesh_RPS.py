import torch
import torch.nn as nn
import numpy as np
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.MLP import MLP
from ParticleGraph.utils import to_numpy

class Mesh_RPS(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], model_config=[], device=[], bc_diff=[]):
        super(Mesh_RPS, self).__init__(aggr=aggr_type)  # "Add" aggregation.

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
        self.ndataset = model_config['nrun']-1
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

        laplacian = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        u = x[:,6]
        v = x[:,7]
        w = x[:,8]

        laplacian = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        laplacian_u = laplacian[:, 0]
        laplacian_v = laplacian[:, 1]
        laplacian_w = laplacian[:, 2]

        embedding = self.a[self.data_id, to_numpy(x[:, 0]), :]

        input_phi = torch.cat((laplacian_u[:,None], laplacian_v[:,None], laplacian_w[:,None], u[:,None], v[:,None], w[:,None], embedding), dim=-1)

        pred = self.lin_phi(input_phi)

        return pred

    def message(self, x_i, x_j, edge_attr):

        # U column 6, V column 7

        # L = edge_attr * (x_j[:, 6]-x_i[:, 6])

        Lu = edge_attr * x_j[:, 6]
        Lv = edge_attr * x_j[:, 7]
        Lw = edge_attr * x_j[:, 8]

        Laplace = torch.cat((Lu[:, None], Lv[:, None], Lw[:, None]), axis=1)

        return Laplace

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r