import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import torch
from ParticleGraph.utils import *


class PDE_N9(pyg.nn.MessagePassing):
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

    def __init__(self, aggr_type="add", p=[], params=[], f=torch.nn.functional.relu, model_type=None, n_neuron_types=None, device=None):
        super(PDE_N9, self).__init__(aggr=aggr_type)

        self.p = p
        self.f = f
        self.model_type = model_type
        self.device = device

        if 'multiple_ReLU' in self.model_type:

            if n_neuron_types is None:
                raise ValueError("n_neuron_types must be provided for multiple_ReLU model type")
            if params[0][0]>0:
                self.params = torch.tensor(params[0], dtype=torch.float32, device=device).expand((n_neuron_types, 1))
            else:
                self.params = torch.abs(1 + 0.5 * torch.randn((n_neuron_types, 1), dtype=torch.float32, device=device))
        else:
            self.params = torch.tensor(params, dtype=torch.float32, device=device).squeeze()


    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index
        v = x[:, 3:4]
        v_rest = self.p["V_i_rest"][:, None]
        e = x[:, 4:5]
        particle_type = x[:, 6: 7].long()


        msg = self.propagate(edge_index, v=v, particle_type=particle_type)
        tau = self.p["tau_i"][:, None]

        if 'tanh'in self.model_type:
            s = self.params
            dv = (-v + msg + e + v_rest + s * torch.tanh(v) ) / tau
        else:
            dv = (-v + msg + e + v_rest) / tau

        return dv

    def message(self, v_j, particle_type_j):
        if 'multiple_ReLU' in self.model_type:
            return self.p["w"][:, None] * self.f(v_j) * self.params[particle_type_j.squeeze()]
        else:
            return self.p["w"][:, None] * self.f(v_j)

def group_by_direction_and_function(neuron_type):
    if neuron_type in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
        return 0  # Outer photoreceptors
    elif neuron_type in ['R7', 'R8']:
        return 1  # Inner photoreceptors
    elif neuron_type in ['L1', 'L2', 'L3', 'L4', 'L5']:
        return 2  # Lamina monopolar
    elif neuron_type in ['Am', 'C2', 'C3']:
        return 3  # Lamina interneurons
    elif neuron_type in ['Mi1', 'Mi2', 'Mi3', 'Mi4']:
        return 4  # Early Mi neurons
    elif neuron_type in ['Mi9', 'Mi10', 'Mi11', 'Mi12']:
        return 5  # Mid Mi neurons
    elif neuron_type in ['Mi13', 'Mi14', 'Mi15']:
        return 6  # Late Mi neurons
    elif neuron_type in ['Tm1', 'Tm2', 'Tm3', 'Tm4']:
        return 7  # Early Tm neurons
    elif neuron_type in ['Tm5a', 'Tm5b', 'Tm5c', 'Tm5Y']:
        return 8  # Tm5 family
    elif neuron_type in ['Tm9', 'Tm16', 'Tm20']:
        return 9  # Mid Tm neurons
    elif neuron_type in ['Tm28', 'Tm30']:
        return 10  # Late Tm neurons
    elif neuron_type.startswith('TmY'):
        return 11  # TmY neurons
    elif neuron_type == 'T4a':
        return 12  # T4a (upward motion)
    elif neuron_type == 'T4b':
        return 13  # T4b (rightward motion)
    elif neuron_type == 'T4c':
        return 14  # T4c (downward motion)
    elif neuron_type == 'T4d':
        return 15  # T4d (leftward motion)
    elif neuron_type in ['T5a', 'T5b', 'T5c', 'T5d']:
        return 16  # T5 OFF motion detectors
    elif neuron_type in ['T1', 'T2', 'T2a', 'T3']:
        return 17  # Tangential neurons
    elif neuron_type.startswith('Lawf'):
        return 18  # Wide-field neurons
    else:
        return 19  # Other/CT1


def get_photoreceptor_positions_from_net(net):
    """
    Extract photoreceptor positions from flyvis network.
    Returns x, y coordinates for all input neurons (R1-R8).
    """
    # Get all nodes from connectome
    nodes = net.connectome.nodes

    print(f"Total nodes: {len(nodes['u'])}")

    # Get coordinates and types
    u_coords = np.array(nodes['u'])  # hex u coordinate
    v_coords = np.array(nodes['v'])  # hex v coordinate
    node_types = np.array(nodes['type'])  # node types
    node_roles = np.array(nodes['role'])  # node roles

    # Convert bytes to strings for comparison
    node_types_str = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in node_types]
    node_roles_str = [r.decode('utf-8') if isinstance(r, bytes) else str(r) for r in node_roles]

    print(f"available node types: {set(node_types_str)}")
    print(f"available node roles: {set(node_roles_str)}")

    # Find all photoreceptors - R1 through R8
    photoreceptor_types = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']

    # Method 1: Find by photoreceptor types (should get all 1736)
    photoreceptor_mask = np.array([t in photoreceptor_types for t in node_types_str])

    # Method 2: Alternative - find by input role (should also get 1736)
    input_mask = np.array([r == 'input' for r in node_roles_str])

    print(f"photoreceptor type mask (R1-R8): {np.sum(photoreceptor_mask)} neurons")
    print(f"input role mask: {np.sum(input_mask)} neurons")

    # Use photoreceptor types (should now get all 1736)
    mask = photoreceptor_mask
    print("using photoreceptor type mask (R1-R8)")

    # Extract coordinates for photoreceptors
    u_photo = u_coords[mask]
    v_photo = v_coords[mask]

    # Convert hex coordinates (u,v) to Cartesian coordinates (x,y)
    # Standard hex-to-cartesian conversion
    x_coords = u_photo + 0.5 * v_photo
    y_coords = v_photo * np.sqrt(3) / 2

    # print(f"Found {len(x_coords)} photoreceptor positions")
    # print(f"U range: {u_photo.min()} to {u_photo.max()}")
    # print(f"V range: {v_photo.min()} to {v_photo.max()}")
    # print(f"X range: {x_coords.min():.3f} to {x_coords.max():.3f}")
    # print(f"Y range: {y_coords.min():.3f} to {y_coords.max():.3f}")

    return x_coords, y_coords, u_photo, v_photo


