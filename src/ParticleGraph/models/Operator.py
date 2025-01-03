import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
# from ParticleGraph.models.utils import reparameterize
# from ParticleGraph.models.Siren_Network import Siren
import torch.nn as nn
import numpy as np



def density_laplace(y, x):
    grad = density_gradient(y, x)
    return density_divergence(grad, x)


def density_divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def density_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad




class Operator_smooth(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning kernel operators.
    The methods follows the particle smoothing techniques proposed in the paper:
    'Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids'

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the kernel operators and their convolution with the data
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2, model_density=[]):

        super(Operator_smooth, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_mp_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.time_window_noise = train_config.time_window_noise
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update
        self.model_type = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.time_window = train_config.time_window
        self.model_density = model_density
        self.sub_sampling = simulation_config.sub_sampling
        self.prediction = model_config.prediction
        self.var = self.max_radius ** 2 / 8

        if self.update_type == 'pre_mlp':
            self.pre_lin_edge = MLP(input_size=self.pre_input_size, output_size=self.pre_output_size, nlayers=self.pre_n_layers,
                                hidden_size=self.pre_hidden_dim, device=self.device)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if 'mlp' in self.update_type:
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_id = x[:, 0:1]
        embedding = self.a[data_id, to_numpy(particle_id), :].squeeze()
        pos = x[:, 1:self.dimension+1]
        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        self.mode = 'pre_mlp'
        density_null = torch.zeros((pos.shape[0], 2), device=self.device)
        self.density = self.propagate(edge_index=edge_index, pos=pos, field=field, embedding=embedding, density=density_null)
        self.mode = 'mlp'
        out = self.propagate(edge_index=edge_index, pos=pos,  field=field, embedding=embedding, density=self.density)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, field_i, field_j, embedding_i, embedding_j, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'pre_mlp':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            d_squared = torch.sum(mgrid ** 2, dim=-1)[:,None]
            first_kernel = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (self.max_radius ** 2))[:,None]

            # kernel_modified = self.pre_lin_edge(torch.cat((torch.sqrt(first_kernel), d_squared), dim=-1)) ** 2
            kernel_modified = self.pre_lin_edge(first_kernel)

            grad_autograd = -density_gradient(kernel_modified, mgrid) / self.max_radius
            laplace_autograd = density_laplace(kernel_modified, mgrid) / self.max_radius

            self.kernel_operators = torch.cat((first_kernel.clone().detach(), grad_autograd, laplace_autograd), dim=-1)

            return first_kernel

        else:
            # out = self.lin_edge(field_j) * self.kernel_operators[:,1:2] / density_j
            # out = self.lin_edge(field_j) * self.kernel_operators[:,3:4] / density_j
            out = field_j * self.kernel_operators[:, 1:2] / density_j
            # out = field_j * self.kernel_operators[:, 3:4] / density_j


            return out

        fig = plt.figure(figsize=(6, 10))
        ax = fig.add_subplot(321)
        plt.scatter(to_numpy(delta_pos[:,0]), to_numpy(delta_pos[:,1]), s=1, c=to_numpy(first_kernel[:,None]))
        ax = fig.add_subplot(322)
        plt.scatter(to_numpy(delta_pos[:,0]), to_numpy(delta_pos[:,1]), s=1, c=to_numpy(kernel_modified[:,None]))
        ax = fig.add_subplot(323)
        plt.scatter(to_numpy(delta_pos[:,0]), to_numpy(delta_pos[:,1]), s=1, c=to_numpy(grad_autograd[:,0:1]))
        ax = fig.add_subplot(324)
        plt.scatter(to_numpy(delta_pos[:,0]), to_numpy(delta_pos[:,1]), s=1, c=to_numpy(self.kernel_operators[:,1:2]))
        ax = fig.add_subplot(325)
        plt.scatter(to_numpy(delta_pos[:,0]), to_numpy(delta_pos[:,1]), s=1, c=to_numpy(self.kernel_operators[:,2:3]))
        ax = fig.add_subplot(326)
        plt.scatter(to_numpy(delta_pos[:,0]), to_numpy(delta_pos[:,1]), s=1, c=to_numpy(self.kernel_operators[:,3:4]))
        plt.show()

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)



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

def arbitrary_test_grad_laplace (mgrid):

    mgrid.requires_grad = True
    x = mgrid[:, 0]
    y = mgrid[:, 1]
    size = np.sqrt(mgrid.shape[0]).astype(int)

    a =  x ** 2 * torch.sin(5*y) + y ** 3 * torch.cos(5*x)

    da_x = 2 * x * torch.sin(5*y)  - y ** 3 * 5 * torch.sin(5*x)
    da_y = x**2 * 5 * torch.cos(5*y) + 3 * y **2 * torch.cos(5*x)
    d2a_x = 2 * torch.sin(5*y) - 25 * y ** 3 * torch.cos(5*x)
    d2a_y = - 25 * x ** 2 * torch.sin(5*y) + 6 * y * torch.cos(5*x)
    laplacian = d2a_x + d2a_y

    grad_autograd = density_gradient(a, mgrid)
    laplacian_autograd  = density_laplace(a, mgrid)

    return a, laplacian



if __name__ == '__main__':


    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import torch_geometric.data as data
    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig

    config = ParticleGraphConfig.from_yaml('test_smooth_particle.yaml')

    device = 'cuda:0'
    dimension = 2
    bc_pos, bc_dpos = choose_boundary_values('no')
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    lr = config.training.learning_rate_start

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')

    tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
    x = torch.stack(torch.meshgrid(*tensors), dim=-1)
    x = x.reshape(-1, dimension)
    x = torch.cat((torch.arange(x.shape[0])[:,None], x, torch.zeros((x.shape[0], 9))), 1)
    x = x.to(device)
    x.requires_grad = False
    size = np.sqrt(x.shape[0]).astype(int)
    x0 = x

    model = Operator_smooth(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos, dimension=dimension)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    phi = torch.zeros(1, device=device)
    threshold = 0.05

    for epoch in trange(0, 10000):

        optimizer.zero_grad()

        x = x0.clone().detach() + 0.05 * torch.randn_like(x0)
        u, grad_u, laplace_u = arbitrary_gaussian_grad_laplace(mgrid = x[:,1:3], n_gaussian = 5, device=device)
        L_u = grad_u.clone().detach()
        # L_u = laplace_u.clone().detach()
        x[:, 6:7] = u[:, None].clone().detach()

        discrete_pos = torch.argwhere((u >= threshold) | (u <= -threshold))
        x = x[discrete_pos].squeeze()
        L_u = L_u[discrete_pos].squeeze()

        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)
        data_id = torch.ones((x.shape[0], 1), dtype=torch.int)
        dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

        pred = model(dataset, data_id=data_id, training=False, phi=phi)
        loss = (pred-L_u).norm(2)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 500 == 0:

            u = u[discrete_pos]
            grad_u = grad_u[discrete_pos]
            laplace_u = laplace_u[discrete_pos]

            print(epoch, loss)

            fig = plt.figure(figsize=(18, 4.75))
            ax = fig.add_subplot(141)
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(model.density))
            ax.invert_yaxis()
            plt.title('density')
            ax = fig.add_subplot(142)
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(u))
            ax.invert_yaxis()
            plt.title('u')
            ax = fig.add_subplot(143)
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(L_u[:,0]))
            ax.invert_yaxis()
            plt.title('true L_u')
            ax = fig.add_subplot(144)
            plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=4, c=to_numpy(pred[:,0]))
            ax.invert_yaxis()
            plt.title('pred L_u')
            plt.tight_layout()

            fig = plt.figure(figsize=(10, 3.5))
            ax = fig.add_subplot(131)
            plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1, c=to_numpy(model.kernel_operators[:, 1:2]))
            plt.title('grad_x')
            ax = fig.add_subplot(132)
            plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1, c=to_numpy(model.kernel_operators[:, 2:3]))
            plt.title('grad_y')
            ax = fig.add_subplot(133)
            plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1, c=to_numpy(model.kernel_operators[:, 3:4]))
            plt.title('laplace')
            plt.tight_layout()
            plt.show()



    for k in range(100000):

        x = x_list[it].squeeze()
        x = torch.cat((x, torch.zeros((x.shape[0], 7), device=device)), 1)
        data_id = torch.ones((x.shape[0], 1), dtype=torch.int)

        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

        optimizer.zero_grad()

        density_s = model(dataset, data_id=data_id, training=False, phi=torch.zeros(1, device=device), tasks=['density_tensor'])
        x[:, 6:9] = density_s[:, 0:3]
        v_s = model(dataset, data_id=data_id, training=False, phi=torch.zeros(1, device=device), tasks=['velocity_tensor'])
        x[:, 9:13] = v_s
        pred = model(dataset, data_id=data_id, training=False, phi=torch.zeros(1, device=device), tasks=['pred_smooth_particle'])
        loss = torch.std(pred)/torch.mean(pred)
        loss.backward()
        optimizer.step()

        if k % 1000 == 0:
            print(k,loss)


