
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

class PDE_O(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the state synchronization

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    d_theta : float
    angle change rate
    """

    def __init__(self, aggr_type=[], bc_dpos=[], p=[], beta=[], rr=[]):
        super(PDE_O, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.p = p
        self.beta = beta
        self.rr = rr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_type = to_numpy(x[:, 5])
        p = self.p[particle_type]
        p = p[:, None]

        theta = data.x[:,8:9]
        d_theta0 = data.x[:,10:11]

        d_theta = p * self.propagate(edge_index, theta=theta)

        d_theta = d_theta0 +  1e-3 * d_theta
        return d_theta

    def message(self, theta_i, theta_j):


        return torch.sin(theta_j - theta_i)

    def psi(self, r, p):
        return r * p
