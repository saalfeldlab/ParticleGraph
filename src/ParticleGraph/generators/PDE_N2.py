import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy
from scipy import sparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread


def constructRandomMatrices(n_neurons=1000, density=1.0, showplots=True, connectivity_mask=[], device=[]):
    """
    n_neurons = Number
    density = density of connections
    """
    if connectivity_mask=='./graphs_data/':
        K = n_neurons * density
        W = np.multiply(np.random.normal(loc=0, scale=1, size=(n_neurons, n_neurons)),
                        np.random.rand(n_neurons, n_neurons) < density)
        W = W / np.sqrt(K)
    else:
        mask = (imread(connectivity_mask)>0.1)*1.0
        density = np.sum(mask) / (n_neurons**2)
        K = n_neurons * density
        W = np.multiply(np.random.normal(loc=0, scale=1, size=(n_neurons, n_neurons)),mask)
        W = W / np.sqrt(K)

    np.fill_diagonal(W, 0)

    if showplots:
        plt.figure(figsize=(3, 3))
        ax = sns.heatmap(W, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        ax.invert_yaxis()
        plt.title('Random connectivity matrix', fontsize=12);
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=10)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=10)

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


class PDE_N2(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        
    """

    def __init__(self, aggr_type=[], p=[], W=[], phi=[]):
        super(PDE_N2, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi

    def forward(self, data=[], return_all=False, excitation=[]):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]

        u = x[:, 6:7]

        msg_ = self.propagate(edge_index, u=u, edge_attr=edge_attr)
        msg = torch.matmul(self.W, self.phi(u))

        du = -u + s * self.phi(u) + g * msg + excitation[:,None]

        if return_all:
            return du, s * self.phi(u), g * msg
        else:
            return du

    def message(self, u_j, edge_attr):

        self.activation = self.phi(u_j)
        self.u_j = u_j

        return edge_attr[:,None] * self.phi(u_j)




    def psi(self, r, p):
        return r * p
