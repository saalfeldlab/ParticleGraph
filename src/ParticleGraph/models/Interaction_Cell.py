import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *


class Interaction_Cell(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the acceleration of particles as a function of their relative distance and relative velocities.
    The interaction function is defined by a MLP self.lin_edge
    The particle embedding is defined by a table self.a

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_Cell, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device



        self.n_particles = simulation_config.n_particles
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.n_particles_max = simulation_config.n_particles_max
        self.len_directed_edges = simulation_config.len_directed_edges

        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.n_frames = simulation_config.n_frames
    

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.n_layers_update = model_config.n_layers_update
        self.output_size_update = model_config.output_size_update



        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device, initialisation='std')
        
        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                           hidden_size=self.hidden_dim_update, device=self.device, initialisation='std')

        self.a = nn.Parameter(torch.tensor(np.ones((self.n_particles_max, 2)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.edges_embedding = nn.Parameter(torch.tensor(np.ones((self.len_directed_edges, 2)), device=self.device, requires_grad=True, dtype=torch.float32))



    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False, edge_pointers=None):

        self.data_id = data_id
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        self.has_field = has_field

        if 'edges_embedding' in self.model:
            self.edge_pointers = edge_pointers

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        field = x[:,6:7]    # e.g. GCamp signal

        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        particle_id = x[:, 0].long()

        embedding = self.a[particle_id, :].squeeze()

        msg = self.propagate(edge_index, pos=pos, d_pos=d_pos, embedding=embedding, field=field)


        in_features = torch.cat([field, embedding, msg], dim=1)

        pred = self.lin_phi(in_features)

        return pred

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, field_i, field_j):
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm

        if self.dimension == 3:
            dpos_z_i = d_pos_i[:, 2] / self.vnorm
            dpos_z_j = d_pos_j[:, 2] / self.vnorm

        if self.rotation_augmentation & (self.training == True):
            new_delta_pos_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_delta_pos_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_delta_pos_x
            delta_pos[:, 1] = new_delta_pos_y
            new_dpos_x_i = self.cos_phi * dpos_x_i + self.sin_phi * dpos_y_i
            new_dpos_y_i = -self.sin_phi * dpos_x_i + self.cos_phi * dpos_y_i
            dpos_x_i = new_dpos_x_i
            dpos_y_i = new_dpos_y_i
            new_dpos_x_j = self.cos_phi * dpos_x_j + self.sin_phi * dpos_y_j
            new_dpos_y_j = -self.sin_phi * dpos_x_j + self.cos_phi * dpos_y_j
            dpos_x_j = new_dpos_x_j
            dpos_y_j = new_dpos_y_j

        match self.model:
            case 'PDE_Cell_A':
                in_features = torch.cat((delta_pos, embedding_i), dim=-1)
            case 'PDE_Cell':
                in_features = torch.cat((delta_pos, embedding_i), dim=-1)
            case 'PDE_Cell_area':
                in_features = torch.cat((delta_pos, features_i[:,0:1] /100, features_j[:,0:1]  /100, embedding_i), dim=-1)
            case 'PDE_Cell_Gcamp':
                in_features = torch.cat((delta_pos, embedding_j, field_j), dim=-1)  
            case 'PDE_Cell_Gcamp_edges_embedding':
                in_features = torch.cat((delta_pos, self.edges_embedding[self.edge_pointers], field_i, field_j), dim=-1)

        out = self.lin_edge(in_features)

        

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)





    # 0 N1 cell index dim=1
    # 1,2 X1 positions dim=2
    # 3,4 V1 velocities dim=2
    # 5 T1 cell type dim=1
    # 6,7 H1 cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
    # 8 A1 cell age dim=1
    # 9 S1 cell stage dim=1  0 = G1 , 1 = S, 2 = G2, 3 = M
    # 10 M1 cell_mass dim=1 (per node)
    # 11 R1 cell growth rate dim=1
    # 12 CL1 cell cycle length dim=1
    # 13 DR1 cell death rate dim=1
    # 14 AR1 area of the cell
    # 15 P1 cell perimeter
    # 16 ASR1 aspect ratio
    # 17 OR1 orientation
