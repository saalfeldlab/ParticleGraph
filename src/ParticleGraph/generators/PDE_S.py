import umap
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import *


def arbitrary_gaussian_grad_laplace (mgrid, n_gaussian, device):

    mgrid.requires_grad = True
    x = mgrid[:, 0]
    y = mgrid[:, 1]
    size = np.sqrt(mgrid.shape[0]).astype(int)

    u = torch.zeros(mgrid.shape[0], device=device)

    for k in range(n_gaussian):
        x0 = np.random.uniform(0, 1)
        y0 = np.random.uniform(0, 1)
        a = np.random.uniform(0, 1)
        sigma = np.random.uniform(0.05, 0.1)
        u = u + a * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    grad_autograd = density_gradient(u, mgrid)
    laplacian_autograd = density_laplace(u, mgrid)

    return u, grad_autograd, laplacian_autograd

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131)
    plt.imshow(to_numpy(u).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('u(x,y)')
    ax = fig.add_subplot(132)
    plt.imshow(to_numpy(grad_autograd[:,0]).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('Grad_x(u(x,y)) autograd')
    ax = fig.add_subplot(133)
    plt.imshow(to_numpy(laplacian_autograd).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('Laplacian(u(x,y)) autograd')
    plt.show()


class PDE_S(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of fluidic particles.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=None, bc_dpos=None, dimension=2, delta_t=0.1, max_radius=0.05, field_type=None):
        super(PDE_S, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.field_type = field_type

        self.dimension = dimension
        self.delta_t = delta_t
        self.max_radius = max_radius
        self.bc_dpos = bc_dpos
        self.p = p

        self.kernel_var = self.max_radius ** 2
        # self.kernel_norm = np.pi * self.kernel_var * (1 - np.exp(-self.max_radius ** 2/ self.kernel_var))
        # self.kernel_norm = 2


    def forward(self, data, continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_type = to_numpy(x[:, 1 + 2*self.dimension])
        pos = x[:, 1:self.dimension+1]
        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        if continuous_field:
            self.mode = 'kernel'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=torch.zeros_like(x[:, 0:1]))
            density = torch.zeros_like(x[:, 0:1])
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=density)
        else:
            self.mode = 'kernel'
            self.density = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=torch.zeros_like(x[:, 0:1]))
            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, field=field, particle_type=particle_type, density=self.density)

        # out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
        # out = torch.where(torch.isnan(out), torch.zeros_like(out), out)

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, field_i, field_j, particle_type_i, density_i, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'kernel':
            self.mgrid = delta_pos.clone().detach()
            self.mgrid.requires_grad = True

            density_kernel = torch.exp(-0.5 * (self.mgrid[:, 0] ** 2 + self.mgrid[:, 1] ** 2) / self.kernel_var)[:, None] / 2

            grad_density_kernel = density_gradient(density_kernel, self.mgrid)
            laplace_autograd = density_laplace(density_kernel, self.mgrid)

            self.kernel_operators = torch.cat((density_kernel, grad_density_kernel, laplace_autograd), dim=-1)

            return density_kernel

        elif self.mode == 'message_passing':

            if 'laplacian' in self.field_type:
                laplacian = self.p[0] * field_j * self.kernel_operators[:, 3:4] # / density_j
                out = laplacian


            return out


if __name__ == '__main__':


    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import torch_geometric.data as data
    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import matplotlib.pyplot as plt
    from utils import density_gradient, density_laplace

    device = 'cuda:0'
    dimension = 2
    bc_pos, bc_dpos = choose_boundary_values('no')
    max_radius = 0.05
    min_radius = 0
    lr = 1E-4

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')


    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/wave/wave_smooth_particle.yaml')

    params = config.simulation.params
    dimension = config.simulation.dimension
    delta_t = config.simulation.delta_t
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    aggr_type = config.graph_model.aggr_type

    model = PDE_S(aggr_type=aggr_type, bc_dpos=bc_dpos, p=torch.tensor(params, dtype=torch.float32, device=device),
                       dimension=dimension, delta_t=delta_t, max_radius=max_radius,
                       field_type=config.graph_model.field_type)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # model.train()

    phi = torch.zeros(1, device=device)
    threshold = 0.05

    tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
    x = torch.stack(torch.meshgrid(*tensors), dim=-1)
    x = x.reshape(-1, dimension)
    x = torch.cat((torch.arange(x.shape[0])[:,None], x, torch.zeros((x.shape[0], 9))), 1)
    x = x.to(device)
    x.requires_grad = False
    size = np.sqrt(x.shape[0]).astype(int)
    x0 = x

    x = x0.clone().detach() + 0.05 * torch.randn_like(x0)
    u, grad_u, laplace_u = arbitrary_gaussian_grad_laplace(mgrid=x[:, 1:3], n_gaussian=5, device=device)
    L_u = laplace_u.clone().detach()
    x[:, 6:7] = u[:, None].clone().detach()

    # matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(18, 4))
    ax = fig.add_subplot(141)
    plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=1, c='w')
    ax.invert_yaxis()
    plt.title('distribution')
    ax = fig.add_subplot(142)
    plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=1, c=to_numpy(u))
    ax.invert_yaxis()
    plt.title('u')
    ax = fig.add_subplot(143)
    # plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=4, c=to_numpy(L_u[:, 0]))
    plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=1, c=to_numpy(L_u))
    ax.invert_yaxis()
    plt.title('true L_u')
    ax = fig.add_subplot(144)
    plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=1, c=to_numpy(pred))
    # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(pred))
    ax.invert_yaxis()
    plt.title('pred L_u')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'tmp/learning_{epoch}.tif')
    plt.close()

    matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(141)
    plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
                c=to_numpy(model.kernel_operators[:, 0:1]))
    plt.title('kernel')
    ax = fig.add_subplot(142)
    plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
                c=to_numpy(model.kernel_operators[:, 1:2]))
    plt.title('grad_x')
    ax = fig.add_subplot(143)
    plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
                c=to_numpy(model.kernel_operators[:, 2:3]))
    plt.title('grad_y')
    ax = fig.add_subplot(144)
    plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
                c=to_numpy(model.kernel_operators[:, 3:4]))
    plt.title('laplace')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'tmp/kernels_{epoch}.tif')
    plt.close()








