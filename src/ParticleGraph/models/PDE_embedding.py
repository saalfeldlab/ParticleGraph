import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils


class PDE_embedding(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the displacement of particles according to an attraction kernel
    The interaction function is defined by a MLP self.lin_edge
    The parameters of the kernel are defined by p =[0, 1.65, 0, 1.35] and sigma=0.7.
    

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the displacements of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], delta_t=[], prediction=[], sigma=[], bc_diff=[], device=[]):
        super(PDE_embedding, self).__init__(aggr='mean')  # "mean" aggregation.

        self.p = p
        self.delta_t = delta_t
        self.prediction = prediction
        self.sigma = torch.tensor([sigma], device=device)
        self.bc_diff = bc_diff

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.delta_t * self.propagate(edge_index, x=(x, x))

        return newv

    def message(self, x_i, x_j):
        r = torch.sum((x_i[:, :] - x_j[:, :]) ** 2, axis=1)  # squared distance
        pp = self.p[0].repeat(x_i.shape[0], 1)
        ssigma = self.sigma[0].repeat(x_i.shape[0], 1)
        psi = - pp[:, 2] * torch.exp(-r ** pp[:, 0] / (2 * ssigma[:, 0] ** 2)) + pp[:, 3] * torch.exp(
            -r ** pp[:, 1] / (2 * ssigma[:, 0] ** 2))
        return psi[:, None] * (x_i - x_j)

    def psi(self, r, p):
        return r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * self.sigma ** 2)) + p[3] * torch.exp(
            -r ** (2 * p[1]) / (2 * self.sigma ** 2)))
