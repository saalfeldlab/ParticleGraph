
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import torch
from ParticleGraph.utils import *

class PDE_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute particle velocity as a function of relative position and attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    d_pos : float
        the velocity of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], func_p = None, sigma=[], bc_dpos=[], dimension=2, embedding_step=0):
        super(PDE_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.func_p = func_p
        self.sigma = sigma
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.embedding_step = embedding_step

        self.evolving_function = False

        for n in range(self.p.shape[0]):
            if self.func_p == None:
                self.func_p[n] = ['arbitrary', n ,n]
            else:
                if self.func_p[n][1] != self.func_p[n][2]:
                    self_evolving_function == True

    def get_interp_a(self, k, particle_id):

        id = particle_id * 100 + k // self.embedding_step
        alpha = (k % self.embedding_step) / self.embedding_step

        return alpha * self.a[id+1, :] + (1 - alpha) * self.a[id, :]

    def forward(self, data=[], has_field=False, k=0):
        x, edge_index = data.x, data.edge_index

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,0:1])

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = x[:, 1 + 2*self.dimension].long()
        parameters = self.p[to_numpy(particle_type),:]
        d_pos = self.propagate(edge_index, pos=x[:, 1:self.dimension+1], particle_type=particle_type[:,None], parameters=parameters.squeeze(), field=field, )

        return d_pos

    def message(self, pos_i, pos_j, particle_type_i, parameters_i, field_j):


        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)
        distance = torch.sqrt(distance_squared)
        d_pos = torch.zeros_like(pos_i)

        f1 = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2)) - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
        f1 = f1[:, None] * self.bc_dpos(pos_j - pos_i) * field_j
        f2 = parameters_i[:, 0] * torch.tanh((distance-parameters_i[:, 1])*parameters_i[:, 2]) / distance
        # f2[torch.isinf(f2)] = -parameters_i[torch.isinf(f2), 0]
        # if torch.isinf(f2).sum() > 0:
        #     print('isinf')

        f2 = f2[:, None] * self.bc_dpos(pos_j - pos_i) * field_j

        for n in range(self.p.shape[0]):
            pos = torch.argwhere(particle_type_i == n)
            if pos.numel() > 0:
                pos = pos[:, 0]
                if self.func_p[n][0] == 'arbitrary':
                    d_pos[pos] = f1[pos]
                if self.func_p[n][0] == 'tanh':
                    d_pos[pos] = f2[pos]

        return d_pos

        # f = parameters_i[:, 0] * torch.tanh((distance - parameters_i[:, 1]) * parameters_i[:, 2])
        # fig = plt.figure()
        # plt.scatter(to_numpy(distance[pos]), to_numpy(f[pos]),s=0.1)
        # fig = plt.figure()
        # plt.hist(to_numpy(distance[pos]), bins=100)

    def psi(self, r, p, func='arbitrary'):

        if func == 'arbitrary':
            out = r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2)) - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * self.sigma ** 2)))
        elif func == 'linear':
            out = p[0] * r + p[1]


