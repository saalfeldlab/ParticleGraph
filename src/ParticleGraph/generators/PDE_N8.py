import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import torch
from ParticleGraph.utils import *


class PDE_N8(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-neuron-dependent
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    dv : float
    the update rate of the voltages (dim 1)
        
    """

    def __init__(self, aggr_type="add", p=[], f=torch.nn.functional.relu):
        super(PDE_N8, self).__init__(aggr=aggr_type)

        self.p = p
        self.f = f

    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index
        v = x[:, 3:4]
        v_rest = self.p["V_i_rest"][:, None]
        e = x[:, 4:5]
        msg = self.propagate(edge_index, v=v)
        tau = self.p["tau_i"][:, None]
        dv = (-v + msg + e + v_rest) / tau
        return dv

    def message(self, v_j):
        return self.p["w_ij"][:, None] * self.f(v_j)


if __name__ == "__main__":
    from datamate import Namespace
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis import NetworkView, Network
    from flyvis.utils.config_utils import get_default_config, CONFIG_PATH
    from flyvis.utils.hex_utils import get_num_hexals
    from tqdm import tqdm

    extent = 8
    dt = 1 / 50

    # Initialize input stimulus data

    config = Namespace(
        n_frames=19,
        flip_axes=[0, 1],
        n_rotations=[0, 1, 2, 3, 4, 5],
        temporal_split=True,
        dt=dt,
        interpolate=True,
        boxfilter=dict(extent=8, kernel_size=13),
        vertical_splits=3,
        center_crop_fraction=0.7,
    )

    stimulus_dataset = AugmentedSintel(**config)

    # Initialize a model with a connectome/eye of less extent to save memory
    # Fine with this connectome version, because inputs don't span more than 8 hexals
    config = get_default_config(
        overrides=[], path=f"{CONFIG_PATH}/network/network.yaml"
    )
    config.connectome.extent = extent
    net = Network(**config)

    # Now load pretrained weights
    nnv = NetworkView("flow/0000/000")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())

    torch.set_grad_enabled(False)

    params = net._param_api()
    p = {
        "tau_i": params.nodes.time_const,
        "V_i_rest": params.nodes.bias,
        "w_ij": params.edges.syn_strength * params.edges.syn_count * params.edges.sign,
    }

    edge_index = torch.stack(
        [
            torch.tensor(net.connectome.edges.source_index[:]),
            torch.tensor(net.connectome.edges.target_index[:]),
        ],
        dim=0,
    )

    # How to generate dummy x?
    state = net.steady_state(t_pre=2.0, dt=dt, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)
    n_edges = len(edge_index[0])
    x = torch.zeros(n_neurons, 5)
    x[:, 0] = torch.arange(n_neurons, dtype=torch.float32)
    x[:, 3] = initial_state
    # frame = torch.randn(1, 1, 1, get_num_hexals(config.connectome.extent))
    # print(frame.shape)

    # (n_frames, 1, n_receptors)
    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)
    x[:, 4] = net.stimulus().squeeze()

    dataset = pyg.data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

    pde = PDE_N8(p=p, f=torch.nn.functional.relu)
    y = pde(dataset, has_field=False)
    print(y)
    print(len(stimulus_dataset))

    y_list = []
    x_list = []
    for data in tqdm(stimulus_dataset):
        x[:, 3] = initial_state
        sequences = data["lum"]
        for frame_id in range(sequences.shape[0]):
            frame = sequences[frame_id][None, None]
            net.stimulus.add_input(frame)
            x[:, 4] = net.stimulus().squeeze()
            dataset = pyg.data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            y = pde(dataset, has_field=False)
            y_list.append(y)
            x_list.append(x)
            x[:, 3:4] = x[:, 3:4] + dt * y

    np.save(f"graphs_data/flyvis/y_list.npy", y_list)
    np.save(f"graphs_data/flyvis/x_list.npy", x_list)
