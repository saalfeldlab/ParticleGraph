import matplotlib.pyplot as plt
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize


class Interaction_Falling_Box(pyg.nn.MessagePassing):
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

        super(Interaction_Falling_Box, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update
        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.time_window = train_config.time_window
        self.recursive_loop = train_config.recursive_loop
        self.recursive_param = train_config.recursive_param


        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[], vnorm=[], phi=[], has_field=False, frame=[]):

        self.data_id = data_id
        self.vnorm = vnorm

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.time_window == 0:
            pos = x[:, 1:self.dimension+1]
            d_pos = x[:, self.dimension+1:1+2*self.dimension]
            particle_id = x[:, 0:1]
            embedding = self.a[self.data_id, to_numpy(particle_id), :].squeeze()
            pred = self.propagate(edge_index, particle_id=particle_id, pos=pos, d_pos=d_pos, embedding=embedding)

            # pred = self.lin_phi(torch.cat((d_pos, pred, embedding), dim=-1))

        else:
            x = torch.stack(x)
            pos = x[:, :, 1:self.dimension + 1]
            pos = pos.transpose(0, 1)
            pos = torch.reshape(pos, (pos.shape[0], pos.shape[1] * pos.shape[2]))
            d_pos = x[:, :, self.dimension + 1:1 + 2 * self.dimension]
            d_pos = d_pos.transpose(0, 1)
            d_pos = torch.reshape(d_pos, (d_pos.shape[0], d_pos.shape[1] * d_pos.shape[2]))
            particle_id = x[0, :, 0:1].squeeze()
            embedding = self.a[self.data_id, to_numpy(particle_id), :].squeeze()
            pred = self.propagate(edge_index, particle_id=particle_id, pos=pos, d_pos=d_pos, embedding=embedding)

            # pred = self.lin_phi(torch.cat((d_pos, pred, embedding), dim=-1))

        if self.recursive_loop>0:

            pred_= pred
            for k in range(self.recursive_loop):
                if self.prediction == '2nd_derivative':
                    new_d_pos = d_pos[:, 0:self.dimension:] + self.delta_t * pred_[:,-self.dimension:] * self.ynorm
                else:
                    new_d_pos = pred * self.vnorm
                new_pos = pos[:, 0:self.dimension:] + self.delta_t * new_d_pos
                if self.time_window == 0:
                    d_pos = new_d_pos
                    pos = new_pos
                else:
                    d_pos = torch.cat((new_d_pos, d_pos[:,0:-2]), dim=1)
                    pos = torch.cat((new_pos, pos[:,0:-2]), dim=1)
                pred_ = torch.cat((pred_, self.recursive_param[k] * self.propagate(edge_index, particle_id=particle_id, pos=pos, d_pos=d_pos, embedding=embedding, field=field)), dim=1)
            return pred_

        else:

            return pred


        fig = plt.figure(figsize=(10, 10))
        plt.scatter(to_numpy(pos[:, 1]), to_numpy(pos[:, 0]), s=2)

        # for k in range(4):
        #     plt.scatter(to_numpy(pos[:, 1+k*2]), to_numpy(pos[:, 0+k*2]),s=10)
        # print('')
        # print(pos[1200])
        # print((pos[1200,0:2]-pos[1200,2:4])/0.0025)
        # print(d_pos[1200,0:2])
        # print((pos[1200,4:6]-pos[1200,6:8])/0.0025)
        # print(d_pos[1200,4:6])


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j):
        # distance normalized by the max radius

        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius

        if self.time_window == 0:
            in_features = torch.cat((r[:, None], delta_pos, d_pos_i, embedding_i, embedding_j), dim=-1)
        else:
            in_features = torch.cat((delta_pos, embedding_i, embedding_j), dim=-1)

        # if self.time_window == 0:
        #     in_features = torch.cat((d_pos_i, d_pos_j, delta_pos, embedding_i, embedding_j), dim=-1)
        # else:
        #     in_features = torch.cat((pos_i-pos_j, d_pos_i-d_pos_j, embedding_i, embedding_j), dim=-1)

        out = self.lin_edge(in_features)


        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


