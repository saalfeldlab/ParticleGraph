
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import torch
from ParticleGraph.utils import *

def constructRandomMatrices(n_neurons=1000, density=1.0, connectivity_mask=[], device=[]):
    """
    n_neurons = Number
    density = density of connections
    """
    if connectivity_mask=='./graphs_data/':
        K = n_neurons * density
        W = np.multiply(np.random.normal(loc=0, scale=1, size=(n_neurons, n_neurons)),
                        np.random.rand(n_neurons, n_neurons) < density)
        W = W / np.sqrt(K)
    elif 'conn' in connectivity_mask:
        W = imread(connectivity_mask)
        n_neurons = W.shape[0]
        polarity = (np.random.rand(W.shape[0],W.shape[1])>0.5)*2-1
        W = W  / np.max(W)
        W = W * polarity

        weights = W.flatten()
        pos = np.argwhere(weights != 0)
        weights = weights[pos]
        # plt.figure(figsize=(10, 10))
        # plt.hist(weights, bins=1000, color='k', alpha=0.5)
        # plt.ylabel(r'counts', fontsize=64)
        # plt.xlabel(r'$W$', fontsize=64)
        # plt.yticks(fontsize=24)
        # plt.xticks(fontsize=24)
        # plt.xlim([0, 100])
        # plt.tight_layout()


    else:
        mask = (imread(connectivity_mask)>0.1)*1.0
        plt.imshow(mask,vmin=0,vmax=1)
        density = np.sum(mask) / (n_neurons**2)
        K = n_neurons * density
        W = np.multiply(np.random.normal(loc=0, scale=1, size=(n_neurons, n_neurons)),mask)
        W = W / np.sqrt(K)

    np.fill_diagonal(W, 0)

    W = torch.tensor(W, dtype=torch.float32, device=device)

    return W


def runNetworkSimulation(W, n_neurons, density, I,
                         g=2.0, s=1.0,
                         Tmax=100, dt=0.01, tau=1.0, phi=np.tanh, showplots=True, device=[]):
    """
    Wee = random connectivity matrix
    n_neurons = number of units
    density = desnity of connectivity
    g = Overall global coupling parameter
    s = self coupling
    Tmax = Number of total time
    dt = time steps
    phi = transfer function (default: np.phi)
    """

    T = torch.arange(0, Tmax, dt)

    # Initial conditions and empty arrays
    X = torch.zeros((n_neurons, len(T)), device=device)
    Xinit = torch.rand(n_neurons, )  # Initial conditions
    X[:, 0] = Xinit

    for t in range(len(T) - 1):
        # Solve using Euler Method
        k1 = -X[:, t]
        k1 += s * phi(X[:, t])
        k1 += g * torch.matmul(W, torch.tanh(X[:, t])) + I[:, t]
        k1 = k1 / tau
        #
        X[:, t + 1] = X[:, t] + k1 * dt

    # W_ = W.detach().cpu().numpy()
    # X_ = X.detach().cpu().numpy()
    # tmp_numpy = np.dot(W_, np.tanh(X_[:, t]))
    # tmp_torch = torch.matmul(W, torch.tanh(X[:, t]))


    return X


class PDE_N4(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-dependent
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    du : float
    the update rate of the signals (dim 1)
        
    """

    def __init__(self, aggr_type=[], p=[], W=[], phi=[]):
        super(PDE_N4, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi

    def forward(self, data=[], has_field=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]
        c = parameters[:, 2:3]
        t = parameters[:, 3:4]

        u = x[:, 6:7]
        if has_field:
            field = x[:, 8:9]
        else:
            field = torch.ones_like(x[:, 6:7])

        msg = self.propagate(edge_index, u=u, t=t, field=field)

        # msg_ = torch.matmul(self.W, self.phi(u))

        du = -c * u + s * self.phi(u) + g * msg

        return du, g * msg


    def message(self, edge_index_i, edge_index_j, u_j, t_i, field_i):

        T = self.W
        return T[to_numpy(edge_index_i), to_numpy(edge_index_j)][:, None]  * self.phi(u_j/t_i) * field_i


    def func(self, u, type, function):

        if function=='phi':

            t = self.p[type, 3:4]
            return self.phi(u/t)

        elif function=='update':
            g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * self.phi(u)
