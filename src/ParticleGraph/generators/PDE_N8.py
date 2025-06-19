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

    def message(self, edge_index_i, edge_index_j, v_j):
        return self.p["w_ij"][:, None] * self.f(v_j)


if __name__ == "__main__":
    from flyvis import NetworkView, Network
    from flyvis.utils.config_utils import get_default_config, CONFIG_PATH
    from flyvis.utils.hex_utils import get_num_hexals

    # Initialize a model with a connectome/eye of less extent to save memory
    # Fine with this connectome version, because inputs don't span more than 8 hexals
    config = get_default_config(
        overrides=[], path=f"{CONFIG_PATH}/network/network.yaml"
    )
    config.connectome.extent = 8
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
    state = net.steady_state(t_pre=1.0, dt=1 / 100, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)
    x = torch.zeros(n_neurons, 5)
    x[:, 0] = torch.arange(n_neurons, dtype=torch.float32)
    x[:, 3] = initial_state
    frame = torch.randn(1, 1, 1, get_num_hexals(config.connectome.extent))
    net.stimulus.add_input(frame)
    x[:, 4] = net.stimulus().squeeze()

    dataset = pyg.data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

    pde = PDE_N8(p=p, f=torch.nn.functional.relu)
    y = pde(dataset, has_field=False)
    print(y.shape)
