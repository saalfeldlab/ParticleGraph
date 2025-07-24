import torch
import torch_geometric as pyg


class MPM_P2G(pyg.nn.MessagePassing):
    def __init__(self, aggr_type=[], device=[]):
        super(MPM_P2G, self).__init__(aggr=aggr_type)

        self.subgrid = 3
        self.offset_idx = torch.arange(self.subgrid, device=device) % 9
        self.i_idx = self.offset_idx // 3  # [0,0,0,1,1,1,2,2,2, 0,0,0,1,1,1,2,2,2, ...]
        self.j_idx = self.offset_idx % 3
        self.n_edges = torch.arange(self.subgrid, device=device).long()

    def forward(self, data):
        x, edge_index, w = data.x, data.edge_index, data.w

        return self.propagate(edge_index, x=x)

    def message(self, edge_index_i, edge_index_j, x_j):

        # B-spline weight = w[particle, i, 0] * w[particle, j, 1]
        # weights = w_i[self.n_edges, i_idx, 0] * w_i[self.n_edges, j_idx, 1]
        # return x_j.squeeze(-1) * weights

        return x_j