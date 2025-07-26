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

        self.input_size_nnr = model_config.input_size_nnr
        self.n_layers_nnr = model_config.n_layers_nnr
        self.hidden_dim_nnr = model_config.hidden_dim_nnr
        self.output_size_nnr = model_config.output_size_nnr
        self.outermost_linear_nnr = model_config.outermost_linear_nnr
        self.omega= model_config.omega

        self.siren = Siren(in_features=self.input_size_nnr, out_features=self.output_size_nnr, hidden_features=self.hidden_dim_nnr,
                           hidden_layers=self.n_layers_nnr, first_omega_0=self.omega, hidden_omega_0=self.omega, outermost_linear=True).to(device)

        # self.mlp0 = MLP(input_size=3, output_size=1, nlayers=5, hidden_size=128, device=device)
        # self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_particles) , self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[]):

        x = data.x
        return self.siren(x)