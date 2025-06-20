import torch
import torch.nn as nn
import torch_geometric as pyg
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
import numpy as np


class Signal_Propagation_FlyVis(pyg.nn.MessagePassing):
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

    def __init__(
        self, aggr_type=None, config=None, device=None, bc_dpos=None, projections=None
    ):
        super(Signal_Propagation_FlyVis, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device

        self.lin_edge = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            nlayers=self.n_layers,
            hidden_size=self.hidden_dim,
            device=self.device,
        )

        self.lin_phi = MLP(
            input_size=self.input_size_update,
            output_size=self.output_size,
            nlayers=self.n_layers_update,
            hidden_size=self.hidden_dim_update,
            device=self.device,
        )

        # embedding
        self.a = nn.Parameter(
            torch.tensor(
                projections,
                device=self.device,
                requires_grad=True,
                dtype=torch.float32,
            )
        )

        self.W = nn.Parameter(
            torch.randn(
                simulation_config.data_ids
                self.edge_index.shape[1],
                device=self.device,
                requires_grad=True,
                dtype=torch.float32,
            )[:, None]
        )

    def forward(self, data=[], data_id=[], k=[], return_all=False):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        self.data_id = data_id.squeeze().long().clone().detach()

        v = data.x[:, 3:4]
        excitation = data.x[:, 4:5]

        particle_id = x[:, 0].long()
        embedding = self.a[particle_id, :]

        msg = self.propagate(
            edge_index, v=v, embedding=embedding, data_id=self.data_id[:, None]
        )

        in_features = torch.cat([v, embedding, msg, excitation], dim=1)
        pred = self.lin_phi(in_features)
        
        if return_all:
            return pred, in_features
        else:
            return pred

    def message(
        self, v_i, v_j, embedding_i, embedding_j, data_id_i
    ):

        in_features = torch.cat([v_i, v_j, embedding_i, embedding_j], dim=1)

        lin_edge = self.lin_edge(in_features)

        return self.W[data_id_i.squeeze(), :, :] * lin_edge

    def update(self, aggr_out):
        return aggr_out
