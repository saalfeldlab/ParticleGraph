
import umap
import torch
from ParticleGraph.models.MLP import MLP
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import tifffile
from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *

# from ParticleGraph.models.utils import reparameterize
# from ParticleGraph.models.Siren_Network import Siren
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ParticleGraph.models.Siren_Network import *


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
        self.n_particles_max = simulation_config.n_particles_max
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

        self.kernel_var = config.image_data.cellpose_diameter**2 * 10

        self.kernel_norm = np.pi * self.kernel_var * (1 - np.exp(-self.max_radius ** 2/ self.kernel_var))
        self.field_type = model_config.field_type

        if self.update_type == 'pre_mlp':
            self.pre_lin_edge = MLP(input_size=self.pre_input_size, output_size=self.pre_output_size, nlayers=self.pre_n_layers,
                                hidden_size=self.pre_hidden_dim, device=self.device)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if 'mlp' in self.update_type:
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles_max) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

        self.siren = Siren_Network(image_width=100, in_features=model_config.input_size_nnr,
                                out_features=model_config.output_size_nnr,
                                hidden_features=model_config.hidden_dim_nnr,
                                hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80,
                                hidden_omega_0=model_config.omega )

    def forward(self, data=[], data_id=[], training=[], phi=[], continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_id = x[:, 0:1].long()
        embedding = self.a[data_id, particle_id, :].squeeze()
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        density_null = torch.zeros((pos.shape[0], 2), device=self.device)
        if continuous_field:
            self.mode = 'density_only'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            density = torch.zeros((pos.shape[0], 1), device=self.device)
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'smooth_interpolation'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density)
        else:
            self.mode = 'density_only'
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            self.density = self.density / torch.sum(self.density, dim=0) * 5E3

            self.mode = 'smooth_interpolation'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=self.density)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, embedding_i, embedding_j, density_j):

        delta_pos = pos_j - pos_i
        self.delta_pos = delta_pos

        if self.mode == 'density_only':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            density = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:,None]
            grad_autograd = -density_gradient(density, mgrid)
            laplace_autograd = density_laplace(density, mgrid)

            # kernel_modified = torch.exp(-2 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:, None]
            # fig = plt.figure(figsize=(8, 6))
            # plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=10, c=to_numpy(kernel_modified), vmin=0, vmax=1)
            # plt.colorbar()
            # plt.show()

            self.modulation = self.siren(coords = mgrid) * max_radius**2
            kernel_modified_1 = density * self.modulation

            self.kernel_operators = dict()
            self.kernel_operators['density'] = density
            self.kernel_operators['grad'] = grad_autograd
            self.kernel_operators['laplace'] = laplace_autograd
            self.kernel_operators['modified_1'] = kernel_modified_1

            return density



        else:
            # out = self.lin_edge(field_j) * self.kernel_operators[:,1:2] / density_j
            # out = self.lin_edge(field_j) * self.kernel_operators[:,3:4] / density_j
            # out = field_j * self.kernel_operators[:, 1:2] / density_j
            # out = torch.cat((grad_density, velocity, grad_velocity), dim = 1) # d_rho_x d_rho_y, velocity
            # out = field_j * self.kernel_operators[:, 1:2] / density_j  # grad_x
            # if 'laplacian' in self.field_type:
            #     out = field_j * self.kernel_operators[:, 3:4] / density_j  # laplacian
            # elif 'grad_density' in self.field_type:
            #     out = grad_density
            # else:
            #     out = grad_density

            density = self.kernel_operators['density']
            grad_density = self.kernel_operators['grad']
            velocity = self.kernel_operators['density'] * torch.sum(d_pos_j**2, dim=1)[:,None] / density_j
            grad_velocity = self.kernel_operators['grad'] * torch.sum(d_pos_j**2, dim=1)[:,None].repeat(1,2) / density_j.repeat(1,2)
            velocity_x = self.kernel_operators['density'] * d_pos_j[:,0:1] / density_j
            grad_velocity_x = self.kernel_operators['grad'][1:2] * d_pos_j[:,0:1].repeat(1,2) / density_j.repeat(1,2)
            velocity_y = self.kernel_operators['density'] * d_pos_j[:,0:1] / density_j
            grad_velocity_y = self.kernel_operators['grad'][1:2] * d_pos_j[:,1:2].repeat(1,2) / density_j.repeat(1,2)

            return torch.cat((density, grad_density, velocity, grad_velocity, velocity_x, grad_velocity_x, velocity_y, grad_velocity_y), dim = 1)

            # 0: rho
            # 1: d_rho_x
            # 2 d_rho_y
            # 3 v
            # 4 d_v_x
            # 5 d_v_y,
            # 6 vx
            # 7 d_vx_x
            # 8 d_vx_y
            # 9 vy
            # 10 d_vy_x
            # 11 d_vy_y


        fig = plt.figure(figsize=(6, 6))
        plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=100, c=to_numpy(self.kernel_operators[:,3:4]))

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=100, c=to_numpy(self.pre_lin_edge(mgrid)))


    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)








if __name__ == '__main__':


    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import torch_geometric.data as data
    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import os
    import shutil
    from torch_geometric.loader import DataLoader

    mode = 'cell_MDCK'


    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/cell/cell_MDCK_3.yaml')

    model_config = config.graph_model

    device = 'cuda:1'
    dimension = 2
    bc_pos, bc_dpos = choose_boundary_values('periodic')
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    lr = config.training.learning_rate_start
    batch_size = config.training.batch_size
    n_frames = config.simulation.n_frames
    dataset_name = config.dataset
    data_folder_name = config.data_folder_name

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')


    files_fluo = os.listdir(data_folder_name)
    files_fluo = [f for f in files_fluo if f.endswith('.tif')]
    files_fluo.sort()
    im_fluo = np.array(tifffile.imread(data_folder_name + files_fluo[0]))
    im_size = im_fluo.shape[0:2]
    im_width = min(im_size)

    # tensors_0 = torch.linspace(0, im_size[0], steps=im_size[0])
    # tensors_1 = torch.linspace(0, im_size[1], steps=im_size[1])
    tensors_0 = torch.linspace(0, im_size[0], steps=100)
    tensors_1 = torch.linspace(0, im_size[1], steps=100)
    # tensors_0 = torch.linspace(0, 1, steps=100)
    # tensors_1 = torch.linspace(0, 1, steps=100)
    mgrid = torch.stack(torch.meshgrid(tensors_0, tensors_1), dim=-1)
    mgrid = mgrid.reshape(-1, 2)
    mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid, torch.zeros((mgrid.shape[0], 2))), 1)
    mgrid = mgrid.to(device)

    model = Operator_smooth(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos, dimension=dimension)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    phi = torch.zeros(1, device=device)
    threshold = 0.05

    x_list = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/cell/cell_MDCK_3/x_list_0.pt', map_location=device, weights_only=True)
    vertices_list = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/cell/cell_MDCK_3/full_vertice_list0.pt', map_location=device, weights_only=True)

    # x_list = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/cell/cell_MDCK_3/track_list_0.pt',map_location=device, weights_only=True)

    # x_list = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/cell/cell_MDCK_12/x_list_0.pt', map_location=device, weights_only=True)

    for frame in trange(10,n_frames):

        num = f"{frame:04}"
        # im_ = np.array(tifffile.imread(f"graphs_data/cell/{dataset_name}/Fig/RGB{num}.tif"))
        im_fluo = np.array(tifffile.imread(data_folder_name + files_fluo[frame]))

        x = x_list[frame].clone().detach()
        vertices = vertices_list[frame].clone().detach()

        distance = torch.sum((x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        d = to_numpy(distance)
        adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        data_id = torch.zeros((x.shape[0], 1), dtype=torch.int)
        dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

        pred = model(dataset, data_id=data_id, training=False, phi=phi)
        density = model.density.clone().detach()

        distance = torch.sum((x[:, None, 1:dimension + 1] - mgrid[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance > 0)).float() * 1
        edge_index_mgrid = adj_t.nonzero().t().contiguous()
        xp = torch.cat((mgrid, x[:, 0:2 * dimension + 1]), 0)
        edge_index_mgrid[0, :] = edge_index_mgrid[0, :] + mgrid.shape[0]
        edge_index_mgrid, _ = pyg_utils.remove_self_loops(edge_index_mgrid)

        dataset = data.Data(x=xp, pos=xp[:, 1:dimension + 1], edge_index=edge_index_mgrid)
        data_id = torch.zeros((xp.shape[0], 1), dtype=torch.int)
        pred_field = model(dataset, data_id=data_id, training=False, phi=phi, continuous_field=True, continuous_field_size=mgrid.shape)[0: mgrid.shape[0]]
        density_field = model.density[0: mgrid.shape[0]]

        sp = 100

        fig = plt.figure(figsize=(30, 12))

        ax = fig.add_subplot(2, 7, 1)
        plt.axis('off')
        plt.imshow(im_fluo)
        plt.xticks([])
        plt.yticks([])

        ax = fig.add_subplot(2, 7, 8)
        plt.axis('off')
        plt.imshow(im_fluo)
        plt.xticks([])
        plt.yticks([])
        for ids in torch.unique(vertices[:,5]):
            pos = torch.argwhere(vertices[:, 5:6] == ids).squeeze()
            plt.plot(to_numpy(vertices[pos, 2]), to_numpy(vertices[pos, 1]), c='w', linewidth=1)

        ax = fig.add_subplot(2,7,2)
        plt.axis('off')
        plt.imshow(im_fluo*0)
        plt.xticks([])
        plt.yticks([])
        plt.title('pos', fontsize=18)
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=20, c=to_numpy(pred[:,0]))

        ax = fig.add_subplot(2,7,3)
        plt.title('density_field', fontsize=18)
        plt.imshow((pred_field[:,0].cpu().view(100, 100).detach().numpy()))
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0,100])
        plt.ylim([0,100])

        ax = fig.add_subplot(2,7,4)
        plt.axis('off')
        plt.imshow(im_fluo*0)
        plt.xticks([])
        plt.yticks([])
        pixel = 7021
        plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                    mgrid[:, 1].detach().cpu().numpy(), s=1, c=to_numpy(pred_field[:,0]))
        plt.scatter(mgrid[pixel, 2].detach().cpu().numpy(),
                    mgrid[pixel, 1].detach().cpu().numpy(), s=10, c='r')
        pos = torch.argwhere(edge_index_mgrid[1, :] == pixel).squeeze()
        if pos.numel()>0:
            plt.scatter(xp[edge_index_mgrid[0, pos], 2].detach().cpu().numpy(), xp[edge_index_mgrid[0, pos], 1].detach().cpu().numpy(), s=10,c='b')
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.savefig(f'tmp/fig_{frame}.tif')
        plt.close()





        # plt.xlim([0,1])
        # plt.ylim([0,1])
        #
        # ax = fig.add_subplot(2,7,3)
        # plt.title('density', fontsize=18)
        # plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=sp, c=to_numpy(density), vmin=0, vmax=100)
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlim([0,1])
        # plt.ylim([0,1])

        # ax = fig.add_subplot(2,7,2)
        # plt.title('density_field', fontsize=18)
        # plt.imshow((pred_field[:,0].cpu().view(100, 100).detach().numpy()),vmin=0, vmax=10)
        # plt.scatter(to_numpy(x[:, 2])/im_width, to_numpy(x[:, 1])/im_width, s=1, c='w')
        # plt.xticks([])
        # plt.yticks([])
        # # plt.xlim([0,100])
        # # plt.ylim([0,100])

        ax = fig.add_subplot(2,7,3)
        plt.title('density_field_x', fontsize=18)
        plt.imshow((pred_field[:,2].cpu().view(100, 100).detach().numpy()),vmin=-100, vmax=100, cmap='bwr')
        plt.scatter(to_numpy(x[:, 2])*100, to_numpy(x[:, 1])*100, s=1, c='w')
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0,100])
        plt.ylim([0,100])

        ax = fig.add_subplot(2,7,4)
        plt.title('density_field_y', fontsize=18)
        plt.imshow((pred_field[:,1].cpu().view(100, 100).detach().numpy()),vmin=-100, vmax=100, cmap='bwr')
        plt.scatter(to_numpy(x[:, 2])*100, to_numpy(x[:, 1])*100, s=1, c='w')
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0,100])
        plt.ylim([0,100])

        # ax = fig.add_subplot(2,7,10)
        # plt.title('velocity', fontsize=18)
        # plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=sp, c=to_numpy(pred[:,3]), vmin=0, vmax=20)
        # plt.quiver(to_numpy(x[:, 2]), to_numpy(x[:, 1]), to_numpy(x[:, 4]), to_numpy(x[:, 3]), color='w')
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlim([0,1])
        # plt.ylim([0,1])

        # ax = fig.add_subplot(2,7,9)
        # plt.title('velocity_field', fontsize=18)
        # plt.imshow((pred_field[:,3].cpu().view(100, 100).detach().numpy()),vmin=0, vmax=50)
        # plt.scatter(to_numpy(x[:, 2])*100, to_numpy(x[:, 1])*100, s=1, c='w')
        # plt.quiver(to_numpy(x[:, 2])*100, to_numpy(x[:, 1])*100, to_numpy(x[:, 4])*100, to_numpy(x[:, 3])*100, color='w')
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlim([0,100])
        # plt.ylim([0,100])
        #
        # ax = fig.add_subplot(2,7,10)
        # plt.title('velocity_field_x', fontsize=18)
        # plt.imshow((pred_field[:,5].cpu().view(100, 100).detach().numpy()),vmin=-500, vmax=500, cmap='bwr')
        # plt.scatter(to_numpy(x[:, 2])*100, to_numpy(x[:, 1])*100, s=1, c='w')
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlim([0,100])
        # plt.ylim([0,100])
        #
        # ax = fig.add_subplot(2,7,11)
        # plt.title('velocity_field_y', fontsize=18)
        # plt.imshow((pred_field[:,4].cpu().view(100, 100).detach().numpy()),vmin=-500, vmax=500, cmap='bwr')
        # plt.scatter(to_numpy(x[:, 2])*100, to_numpy(x[:, 1])*100, s=1, c='w')
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlim([0,100])
        # plt.ylim([0,100])
        #
        # plt.tight_layout()
        # plt.savefig(f'tmp/fig_{frame}.tif')
        # plt.close()




        # plt.show()


        train_NNR = False
        if train_NNR:
            model_f = Siren_Network(image_width=100, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                    first_omega_0=30.0, hidden_omega_0=30.0)

            optimizer_f = torch.optim.Adam(lr=1E-4, params=model_f.parameters())
            model_f.train()

            if frame==0:
                total_steps = 2000  # Since the whole image is our dataset, this just means 500 gradient descent steps.
            else:
                total_steps = 2000
            steps_til_summary = 500

            for step in range(total_steps):

                optimizer_f.zero_grad()

                model_output = model_f(time=frame/n_frames) ** 2
                # model_output = laplace(model_output, coords)
                loss = (model_output - density_field.clone().detach()).norm(2)

                loss.backward()
                optimizer_f.step()

                if not step % steps_til_summary:
                    print("Step %d, Total loss %0.6f" % (step, loss))

            fig = plt.figure(figsize=(18, 6))
            ax = fig.add_subplot(1,3,1)
            plt.imshow(density_field.cpu().view(100, 100).detach().numpy(),vmin=0, vmax=300)
            plt.scatter(to_numpy(100*x[:, 2]), to_numpy(100*x[:, 1]), s=1, c='w')
            plt.xlim([0,100])
            plt.ylim([0,100])
            ax = fig.add_subplot(1,3,2)
            plt.imshow(np.flipud(density_field.cpu().view(100, 100).detach().numpy()),vmin=0, vmax=300)
            ax = fig.add_subplot(1,3,3)
            plt.imshow(np.flipud(model_output.cpu().view(100, 100).detach().numpy()),vmin=0, vmax=300)
            # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy())
            # axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy())
            plt.savefig(f'tmp/kernels_{frame}.tif')
            # plt.show()
            plt.close()




















