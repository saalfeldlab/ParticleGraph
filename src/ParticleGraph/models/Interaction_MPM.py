import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax

class Interaction_MPM(nn.Module):

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_MPM, self).__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.model = model_config.particle_model_name
        self.n_dataset = train_config.n_runs
        self.dimension = dimension
        self.n_particles = simulation_config.n_particles
        self.embedding_dim = model_config.embedding_dim

        self.siren = Siren(in_features=3, out_features=4, hidden_features=128, hidden_layers=5, outermost_linear=True).to(device)
        # self.mlp0 = MLP(input_size=3, output_size=1, nlayers=5, hidden_size=128, device=device)
        # self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_particles) , self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[]):

        x = data.x
        return self.siren(x)