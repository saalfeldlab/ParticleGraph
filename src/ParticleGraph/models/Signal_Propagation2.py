import torch
import torch.nn as nn
import torch_geometric as pyg
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
import numpy as np

class Signal_Propagation2(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None, projections=None):
        super(Signal_Propagation2, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.model = model_config.signal_model_name
        self.embedding_dim = model_config.embedding_dim
        self.n_particles = simulation_config.n_particles
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.field_type = model_config.field_type
        self.embedding_trial = config.training.embedding_trial
        self.multi_connectivity = config.training.multi_connectivity

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.lin_edge_positive = model_config.lin_edge_positive

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.input_size_modulation = model_config.input_size_modulation
        self.output_size_modulation = model_config.output_size_modulation
        self.hidden_dim_modulation = model_config.hidden_dim_modulation
        self.n_layers_modulation = model_config.n_layers_modulation

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        self.bc_dpos = bc_dpos
        self.adjacency_matrix = simulation_config.adjacency_matrix
        self.n_virtual_neurons = int(config.training.n_virtual_neurons)
        self.excitation_dim = model_config.excitation_dim

        self.n_layers_excitation = model_config.n_layers_excitation
        self.hidden_dim_excitation = model_config.hidden_dim_excitation
        self.input_size_excitation = model_config.input_size_excitation


        if self.model == 'PDE_N3':
            self.embedding_evolves = True
        else:
            self.embedding_evolves = False

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size, nlayers=self.n_layers_update,
                            hidden_size=self.hidden_dim_update, device=self.device)

        if 'excitation' in self.update_type:
            self.lin_exc = MLP(input_size=self.input_size_excitation, output_size= 1, nlayers=self.n_layers_excitation, hidden_size=self.hidden_dim_excitation, device=self.device)

        if self.embedding_trial:
            self.b = nn.Parameter(
                torch.ones((int(self.n_dataset), self.embedding_dim), device=self.device, requires_grad=True, dtype=torch.float32))

        if self.model == 'PDE_N3':
            self.a = nn.Parameter(torch.ones((int(self.n_particles*100 + 1000), self.embedding_dim), device=self.device, requires_grad=True,dtype=torch.float32))
            self.embedding_step =  self.n_frames // 100
        elif model_config.embedding_init =='':
            self.a = nn.Parameter(torch.ones((int(self.n_particles + self.n_virtual_neurons), self.embedding_dim), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(projections, device=self.device, requires_grad=True, dtype=torch.float32))

        if (self.model == 'PDE_N6') | (self.model == 'PDE_N7'):
            self.b = nn.Parameter(torch.ones((int(self.n_particles), 1000 + 10), device=self.device, requires_grad=True,dtype=torch.float32)*0.44)
            self.embedding_step = self.n_frames // 1000
            self.lin_modulation = MLP(input_size=self.input_size_modulation, output_size=self.output_size_modulation, nlayers=self.n_layers_modulation,
                                hidden_size=self.hidden_dim_modulation, device=self.device)

        if self.multi_connectivity:
            self.W = nn.Parameter(torch.randn((int(self.n_dataset),int(self.n_particles + self.n_virtual_neurons),int(self.n_particles + self.n_virtual_neurons)), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.W = nn.Parameter(torch.randn((int(self.n_particles + self.n_virtual_neurons),int(self.n_particles + self.n_virtual_neurons)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.mask = torch.ones((int(self.n_particles + self.n_virtual_neurons),int(self.n_particles + self.n_virtual_neurons)), device=self.device, requires_grad=False, dtype=torch.float32)
        self.mask.fill_diagonal_(0)

    def get_interp_a(self, k, particle_id):

        id = particle_id * 100 + k // self.embedding_step
        alpha = (k % self.embedding_step) / self.embedding_step

        return alpha * self.a[id.squeeze()+1, :] + (1 - alpha) * self.a[id.squeeze(), :]


    def forward(self, data=[], data_id=[], k = [], return_all=False):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        self.data_id = data_id.squeeze().long().clone().detach()

        u = data.x[:, 6:7]

        if self.model == 'PDE_N3':
            particle_id = x[:, 0:1].long()
            embedding = self.get_interp_a(k, particle_id)
        else:
            particle_id = x[:, 0].long()
            embedding = self.a[particle_id, :]
            if self.embedding_trial:
                embedding = torch.cat((self.b[self.data_id, :], embedding), dim=1)

        msg = self.propagate(edge_index, u=u, embedding=embedding)

        if 'generic' in self.update_type:        # MLP1(u, embedding, \sum MLP0(u, embedding), field )
            field = x[:, 8:9]
            if 'excitation' in self.update_type:
                excitation = x[:, 10: 10 + self.excitation_dim]
                in_features = torch.cat([u, embedding, msg, field, excitation], dim=1)
            else:
                in_features = torch.cat([u, embedding, msg, field], dim=1)
            pred = self.lin_phi(in_features)
        else:
            field = x[:, 8:9]
            if 'excitation' in self.update_type:
                excitation = x[:, 10: 10 + self.excitation_dim]
                in_features = torch.cat([u, embedding, msg], dim=1)
                pred = self.lin_phi(in_features) + msg * field + excitation
            else:
                in_features = torch.cat([u, embedding], dim=1)
                pred = self.lin_phi(in_features) + msg * field  # MLP1(u, embedding) + field * \sum MLP0(u, embedding)

        if return_all:
            return pred, in_features
        else:
            return pred

    def message(self, edge_index_i, edge_index_j, u_i, u_j, embedding_i, embedding_j):

        if (self.model=='PDE_N4') | (self.model=='PDE_N7'):
            in_features = torch.cat([u_j, embedding_j], dim=1)
        elif (self.model=='PDE_N5'):
            in_features = torch.cat([u_j, embedding_i, embedding_j], dim=1)
        elif (self.model=='PDE_N8'):
            in_features = torch.cat([u_i, u_j, embedding_i, embedding_j], dim=1)
        else:
            in_features = u_j


        line_edge = self.lin_edge(in_features)
        if self.lin_edge_positive:
            line_edge = line_edge**2

        if self.multi_connectivity:
            assert self.batch_size==1, 'Warning: multiple_connectivity is not implemented for batch_size > 1'
            T = self.W[self.data_id[0], :, :] * self.mask
            return T[edge_index_i, edge_index_j][:, None] * line_edge
        else:
            T = self.W * self.mask
            if (self.batch_size==1):
                return T[edge_index_i, edge_index_j][:, None] * line_edge
            else:
                return T[edge_index_i%(self.W.shape[0]), edge_index_j%(self.W.shape[0])][:,None] * line_edge


    def update(self, aggr_out):
        return aggr_out

    def psi(self, r, p):
        return p * r




# if (self.model=='PDE_N4') | (self.model=='PDE_N5'):
#     msg = self.propagate(edge_index, u=u, embedding=embedding, field=field)
# elif self.model=='PDE_N6':
#     msg = torch.matmul(self.W * self.mask, self.lin_edge(u)) * field
# else:
#     msg = torch.matmul(self.W * self.mask, self.lin_edge(u))
#     if self.return_all:
#         self.msg = torch.matmul(self.W * self.mask, self.lin_edge(u))
# if (self.model=='PDE_N2') & (self.batch_size==1):
#     msg = torch.matmul(self.W * self.mask, self.lin_edge(u))

# self.n_layers_update2 = model_config.n_layers_update2
# self.hidden_dim_update2 = model_config.hidden_dim_update2
# self.input_size_update2 = model_config.input_size_update2

# if self.update_type=='2steps':
#     self.lin_phi2 = MLP(input_size=self.input_size_update2, output_size=self.output_size,
#                        nlayers=self.n_layers_update2,
#                        hidden_size=self.hidden_dim_update2, device=self.device)

# if self.update_type == '2steps':                  # MLP2( MLP1(u, embedding), \sum MLP0(u, embedding), field )
#     in_features1 = torch.cat([u, embedding], dim=1)
#     pred1 = self.lin_phi(in_features1)
#     field = x[:, 8:9]
#     in_features2 = torch.cat([pred1, msg, field], dim=1)
# pred = self.lin_phi2(in_features2)