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
        self, aggr_type='add', config=None, device=None ):
        super(Signal_Propagation_FlyVis, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.model = model_config.signal_model_name
        self.embedding_dim = model_config.embedding_dim
        self.n_neurons = simulation_config.n_neurons
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.field_type = model_config.field_type
        self.embedding_trial = config.training.embedding_trial
        self.multi_connectivity = config.training.multi_connectivity

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.lin_edge_positive = model_config.lin_edge_positive

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

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

        if config.training.init_update_gradient:
            for i, layer in enumerate(self.lin_phi.layers[:-1]):  # All except final layer
                nn.init.normal_(layer.weight, mean=0, std=0.01)  # Very small weights
                nn.init.zeros_(layer.bias)

            # Initialize final layer to achieve desired gradients:
            # v: -1, embedding: 0, msg: 1, excitation: 1
            final_layer = self.lin_phi.layers[-1]
            with torch.no_grad():
                # Create a weight vector that approximates desired gradients
                # This is approximate since we have hidden layers, but should be close
                desired_gradients = torch.tensor([-1.0, 0.0, 0.0, 1.0, 1.0], device=self.device)

                # Initialize final layer weights as a scaled version of desired gradients
                # Scale by 1/hidden_dim to account for the fact that input gets spread across hidden units
                scale_factor = 1.0 / self.hidden_dim_update
                final_layer.weight.data = desired_gradients.unsqueeze(0) * scale_factor

                # Initialize final bias to zero
                final_layer.bias.data.zero_()

        # embedding
        self.a = nn.Parameter(
            torch.tensor(
                np.ones((int(self.n_neurons), self.embedding_dim)),
                         device=self.device,
                         requires_grad=False, dtype=torch.float32))

        self.W = nn.Parameter(
            torch.zeros(
                self.n_edges + self.n_extra_null_edges,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )[:, None]
        )

    def forward(self, data=[], data_id=[], mask=[], return_all=False):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        self.data_id = data_id.squeeze().long().clone().detach()
        self.mask = mask.squeeze().long().clone().detach()

        v = data.x[:, 3:4]
        excitation = data.x[:, 4:5]

        particle_id = x[:, 0].long()
        embedding = self.a[particle_id].squeeze()

        msg = self.propagate(
            edge_index, v=v, embedding=embedding, data_id=self.data_id[:, None]
        )

        if 'PDE_N9_C' in self.model:
            in_features = torch.cat([v, embedding, excitation], dim=1)
            pred = self.lin_phi(in_features) + msg
        else:
            in_features = torch.cat([v, embedding, msg, excitation], dim=1)
            pred = self.lin_phi(in_features)
        
        if return_all:
            return pred, in_features
        else:
            return pred

    def message(self, edge_index_i, edge_index_j, v_i, v_j, embedding_i, embedding_j, data_id_i):

        if (self.model=='PDE_N9_A') | (self.model=='PDE_N9_C'):
            in_features = torch.cat([v_j, embedding_j], dim=1)
        elif self.model=='PDE_N9_D':
            in_features = torch.cat([v_j, embedding_i], dim=1)
        elif (self.model=='PDE_N9_B'):
            in_features = torch.cat([v_i, v_j, embedding_i, embedding_j], dim=1)

        lin_edge = self.lin_edge(in_features)
        if self.lin_edge_positive:
            lin_edge = lin_edge**2

        return self.W[self.mask % (self.n_edges+ self.n_extra_null_edges)] * lin_edge

    def update(self, aggr_out):
        return aggr_out
