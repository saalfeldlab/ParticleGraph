import torch
import torch_geometric as pyg


class MPM_P2G(pyg.nn.MessagePassing):
    def __init__(self, aggr_type='add', device='cpu'):
        super(MPM_P2G, self).__init__(aggr=aggr_type)
        self.device = device

    def forward(self, data):
        x, edge_index, fx_per_edge = data.x, data.edge_index, data.fx_per_edge
        return self.propagate(edge_index, x=x, fx=fx_per_edge)

    def message(self, x_i, fx):
        # Each edge corresponds to one 3x3 offset
        n_edges = x_i.size(0)
        offset_idx = torch.arange(n_edges, device=self.device) % 9
        # Convert flat index to 2D: 0->(0,0), 1->(0,1), 2->(0,2), 3->(1,0), etc.
        i_idx = offset_idx // 3  # [0,0,0,1,1,1,2,2,2, ...]
        j_idx = offset_idx % 3  # [0,1,2,0,1,2,0,1,2, ...]


        # Quadratic B-spline weights
        w_0 = 0.5 * (1.5 - fx) ** 2
        w_1 = 0.75 - (fx - 1) ** 2
        w_2 = 0.5 * (fx - 0.5) ** 2
        w = torch.stack([w_0, w_1, w_2], dim=1)  # [n_edges, 3, 2]

        # Select weights for each edge based on its offset
        edge_indices = torch.arange(n_edges, device=self.device)
        weights = w[edge_indices, i_idx, 0] * w[edge_indices, j_idx, 1]

        return x_i.squeeze(-1) * weights
