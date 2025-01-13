
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

class PDE_K(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of particles according to spring law as a function of their relative position.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    dd_pos : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], connection_matrix=[], bc_dpos=[]):
        super(PDE_K, self).__init__(aggr='add')  # "mean" aggregation.

        self.connection_matrix = connection_matrix
        self.p = connection_matrix
        self.bc_dpos = bc_dpos

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        dd_pos = self.propagate(edge_index, pos=x[:,1:3])
        return dd_pos

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j):

        dd_pos = self.connection_matrix[edge_index_i, edge_index_j].repeat(2,1).t() * self.bc_dpos(pos_j - pos_i)

        return dd_pos

