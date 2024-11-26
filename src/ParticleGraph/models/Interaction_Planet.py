import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize


class Interaction_Planet(pyg.nn.MessagePassing):
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

    def __init__(self, config, device, aggr_type=None, bc_dpos=None):

        super(Interaction_Planet, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.n_particle_types = simulation_config.n_particle_types
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.n_particles_max = simulation_config.n_particles_max
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.sigma = simulation_config.sigma
        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = simulation_config.dimension
        self.has_state = config.simulation.state_type != 'discrete'
        self.n_frames = simulation_config.n_frames
        self.state_hot_encoding = train_config.state_hot_encoding
        self.do_tracking = train_config.do_tracking
        
        temperature = train_config.state_temperature
        self.temperature = torch.tensor(temperature, device=self.device)


        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles), self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

        self.mass = torch.tensor([1.989e30,3.30e23, 4.87e24,5.97e24,6.42e23,1.90e27,5.68e26,8.68e25,1.02e26,1.31e22,8.93e22,4.80e22, 1.48e23,1.08e23,3.75e19,1.08e20,
                6.18e20,1.10e21,2.31e21,1.35e23,5.62e18,7.35e22,1.07e16,1.48e15, 1.52e21], dtype=torch.float32, device=self.device)

        self.log10_mass = torch.log10(self.mass) / 30


    def forward(self, data=[], data_id=[]):

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pos = x[:, 1:self.dimension+1]
        particle_id = x[:, 0:1]
        mass = self.mass[to_numpy(particle_id)] / 1E26

        pred = self.propagate(edge_index, pos=pos, mass=mass)

        return pred

    def message(self, pos_i, pos_j, mass_j):

        r = torch.sqrt(torch.sum((pos_j - pos_i) ** 2, dim=1)) / 1E6
        delta_pos = (pos_j - pos_i) /1E6

        in_features = torch.cat((delta_pos, r[:, None], mass_j),dim=-1)

        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)
