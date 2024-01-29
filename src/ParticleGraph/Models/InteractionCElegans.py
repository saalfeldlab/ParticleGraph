class InteractionCElegans(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device, aggr_type=[], bc_diff=[]):

        super(InteractionCElegans, self).__init__(aggr='mean')  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.ndataset = model_config['nrun'] - 1
        self.upgrade_type = model_config['upgrade_type']
        self.prediction = model_config['prediction']
        self.upgrade_type = model_config['upgrade_type']
        self.nlayers_update = model_config['nlayers_update']
        self.hidden_size_update = model_config['hidden_size_update']
        self.bc_diff = bc_diff

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles + 1), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float64))

        if self.upgrade_type == 'linear':
            self.lin_update = MLP(input_size=self.output_size + self.embedding + 2, output_size=self.output_size,
                                  nlayers=self.nlayers_update, hidden_size=self.hidden_size_update, device=self.device)

        self.to(device=self.device)
        self.to(torch.float64)

    def forward(self, data, data_id, time):

        self.data_id = data_id

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pred = self.propagate(edge_index, x=(x, x), time=time)

        if self.upgrade_type == 'linear':
            embedding = self.a[self.data_id, to_numpy(x[:, 0]), :]
            pred = self.lin_update(torch.cat((pred, x[:, 3:5], embedding), dim=-1))

        return pred

    def message(self, x_i, x_j, time):

        r = torch.sqrt(torch.sum(self.bc_diff(x_i[:, 1:4] - x_j[:, 1:4]) ** 2, axis=1))  # squared distance
        r = r[:, None]

        delta_pos = self.bc_diff(x_i[:, 1:4] - x_j[:, 1:4])
        embedding = self.a[self.data_id, to_numpy(x_i[:, 0]).astype(int), :]
        in_features = torch.cat((delta_pos, r, x_i[:, 4:7], x_j[:, 4:7], embedding, time[:, None]), dim=-1)

        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)