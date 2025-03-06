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

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.input_size_modulation = model_config.input_size_modulation
        self.output_size_modulation = model_config.output_size_modulation
        self.hidden_dim_modulation = model_config.hidden_dim_modulation
        self.n_layers_modulation = model_config.n_layers_modulation
        self.batch_size = config.training.batch_size

        self.bc_dpos = bc_dpos
        self.adjacency_matrix = simulation_config.adjacency_matrix


        if self.model == 'PDE_N3':
            self.embedding_evolves = True
        else:
            self.embedding_evolves = False

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size, nlayers=self.n_layers_update,
                            hidden_size=self.hidden_dim_update, device=self.device)

        if self.model == 'PDE_N3':
            self.a = nn.Parameter(
                torch.ones((int(self.n_particles*100 + 1000), self.embedding_dim), device=self.device, requires_grad=True,dtype=torch.float32))
            self.embedding_step =  self.n_frames // 100
        elif model_config.embedding_init =='':
            self.a = nn.Parameter(torch.ones((int(self.n_particles), self.embedding_dim), device=self.device, requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(projections, device=self.device, requires_grad=True, dtype=torch.float32))

        if (self.model == 'PDE_N6') | (self.model == 'PDE_N7'):
            self.b = nn.Parameter(torch.ones((int(self.n_particles), 1000 + 10), device=self.device, requires_grad=True,dtype=torch.float32)*0.44)
            self.embedding_step = self.n_frames // 1000
            self.lin_modulation = MLP(input_size=self.input_size_modulation, output_size=self.output_size_modulation, nlayers=self.n_layers_modulation,
                                hidden_size=self.hidden_dim_modulation, device=self.device)

        self.W = nn.Parameter(torch.randn((int(self.n_particles),int(self.n_particles)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.mask = torch.ones((int(self.n_particles),int(self.n_particles)), device=self.device, requires_grad=False, dtype=torch.float32)
        self.mask.fill_diagonal_(0)

    def get_interp_a(self, k, particle_id):

        id = particle_id * 100 + k // self.embedding_step
        alpha = (k % self.embedding_step) / self.embedding_step

        return alpha * self.a[id.squeeze()+1, :] + (1 - alpha) * self.a[id.squeeze(), :]


    def forward(self, data=[], return_all=False, data_id=0, k = 0):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        u = data.x[:, 6:7]

        if self.model == 'PDE_N3':
            particle_id = x[:, 0:1].long()
            embedding = self.get_interp_a(k, particle_id)
        else:
            particle_id = x[:, 0].long()
            embedding = self.a[particle_id, :]

        field = torch.ones((self.n_particles,1), requires_grad=False, dtype=torch.float32, device=self.device)
        if (self.model == 'PDE_N4') | (self.model == 'PDE_N5') | (self.model == 'PDE_N6') | (self.model == 'PDE_N7') :
            field = x[:, 8:9]

        in_features = torch.cat([u, embedding], dim=1)

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
        # else:

        msg = self.propagate(edge_index, u=u, embedding=embedding, field=field)

        pred = self.lin_phi(in_features) + msg

        return pred

    def message(self, edge_index_i, edge_index_j, u_j, embedding_i, embedding_j, field_i):

        if (self.model=='PDE_N4') | (self.model=='PDE_N7'):
            in_features = torch.cat([u_j, embedding_i], dim=1)
        elif (self.model=='PDE_N5'):
            in_features = torch.cat([u_j, embedding_i, embedding_j], dim=1)
        else:
            in_features = u_j

        T = self.W * self.mask

        if self.return_all:
            self.msg = T[edge_index_i%self.n_particles, edge_index_j%self.n_particles][:,None] * self.lin_edge(in_features) * field_i
            return self.msg
        else:
            if (self.batch_size==1):
                return T[edge_index_i, edge_index_j][:, None] * self.lin_edge(in_features) * field_i
            else:
                return T[edge_index_i%self.n_particles, edge_index_j%self.n_particles][:,None] * self.lin_edge(in_features) * field_i


    def update(self, aggr_out):
        return aggr_out

    def psi(self, r, p):
        return p * r
