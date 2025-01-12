import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy, density_laplace, density_divergence, density_gradient
import numpy as np

class PDE_F(pyg.nn.MessagePassing):
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
        super(PDE_F, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.field_type = field_type
        self.p = p[0]
        self.dimension = dimension
        self.delta_t = delta_t
        self.max_radius = max_radius
        self.bc_dpos = bc_dpos

        self.kernel_var = self.max_radius ** 2
        # self.kernel_norm = np.pi * self.kernel_var * (1 - np.exp(-self.max_radius ** 2/ self.kernel_var))
        # self.kernel_norm = 2


    def forward(self, data, continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_type = to_numpy(x[:, 1 + 2*self.dimension])
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        density_null = torch.zeros((pos.shape[0], 2))
        if continuous_field:
            self.mode = 'pre_mlp'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=density_null)
            density = torch.zeros((pos.shape[0], 1))
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=density)
        else:
            self.mode = 'pre_mlp'
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=density_null)
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=self.density)

        out = out / (self.density.repeat(1,out.shape[1]) + 1E-7)

        out[:, 1] = out[:, 1] + torch.ones_like(out[:, 1]) * self.p[0] * 9.8

        # # Bounce on walls
        # bouncing_pos = torch.argwhere((X1[:, 0] <= 0) | (X1[:, 0] >= 1)).squeeze()
        # if bouncing_pos.numel() > 0:
        #     dd_pos[particles[bouncing_pos], 0] = -1.6 * V1[bouncing_pos, 0] / self.delta_t
        #     dd_pos[particles[bouncing_pos], 1] = - 0.4 * V1[bouncing_pos, 1] / self.delta_t
        # bouncing_pos = torch.argwhere((X1[:, 1] <= 0) | (X1[:, 1] >= 1)).squeeze()
        # if bouncing_pos.numel() > 0:
        #     dd_pos[particles[bouncing_pos], 1] = -1.6 * V1[bouncing_pos, 1] / self.delta_t

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, particle_type_i, particle_type, density_i, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'pre_mlp':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            if 'gaussian' in self.field_type:
                density_kernel = torch.exp(-4*(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:, None] / 2
            elif 'triangle' in self.field_type:
                dist = torch.sqrt(torch.sum(mgrid ** 2, dim=1))
                density_kernel = ((self.max_radius - dist)**2 / self.kernel_var)[:, None] / 1.309

            grad_autograd = density_gradient(density_kernel, mgrid)
            laplace_autograd = density_laplace(density_kernel, mgrid)

            self.kernel_operators = torch.cat((density_kernel, grad_autograd, laplace_autograd), dim=-1)

            return density_kernel

        else:
            # out = self.lin_edge(field_j) * self.kernel_operators[:,1:2] / density_j
            # out = self.lin_edge(field_j) * self.kernel_operators[:,3:4] / density_j
            # out = field_j * self.kernel_operators[:, 1:2] / density_j

            grad_density = ((density_i+density_j)/2 - self.p[2])  * self.kernel_operators[:, 1:3] * self.p[1] / density_j  # / 1E7  # d_rho_x d_rho_y
            velocity = self.kernel_operators[:, 0:1] * torch.sum(d_pos_j**2, dim=1)[:,None] / density_j
            grad_velocity = self.kernel_operators[:, 1:3] * torch.sum(d_pos_j**2, dim=1)[:,None].repeat(1,2) / density_j.repeat(1,2)

            out = torch.cat((grad_density, velocity, grad_velocity), dim = 1) # d_rho_x d_rho_y, velocity

            # out = field_j * self.kernel_operators[:, 1:2] / density_j

            return out

