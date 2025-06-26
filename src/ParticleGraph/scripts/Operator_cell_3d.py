
import umap
import torch
from ParticleGraph.models.MLP import MLP
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.utils import *

# from ParticleGraph.models.utils import reparameterize
# from ParticleGraph.models.Siren_Network import Siren
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.io as pio
import napari
from tifffile import imwrite
import scipy.ndimage

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

class Operator_smooth3D(pyg.nn.MessagePassing):

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

        super(Operator_smooth3D, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers
        self.n_particles = simulation_config.n_particles
        self.n_particles_max = simulation_config.n_particles_max
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.noise_model_level = train_config.noise_model_level
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
        self.kernel_var = self.max_radius ** 2

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
            self.mode = 'pre_mlp'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            density = torch.zeros((pos.shape[0], 1), device=self.device)
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density)
        else:
            self.mode = 'pre_mlp'
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=self.density)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, embedding_i, embedding_j, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'pre_mlp':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            density_kernel = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2 + mgrid[:, 2] ** 2) / self.kernel_var)[:,None]

            # self.modulation = self.siren(coords=mgrid) * max_radius **2
            # kernel_modified = torch.exp(-2*(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20*self.kernel_var))[:, None] * self.modulation

            kernel_modified = torch.exp(-2 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2 + mgrid[:, 2] ** 2) / (self.kernel_var))[:, None]

            grad_autograd = -density_gradient(kernel_modified, mgrid)
            # laplace_autograd = density_laplace(kernel_modified, mgrid)

            self.kernel_operators = torch.cat((kernel_modified, grad_autograd), dim=-1)

            return density_kernel

            # mg = mgrid.detach().cpu().numpy().astype(np.float64)
            # dg = kernel_modified.detach().cpu().numpy().astype(np.float64)
            # indices = np.random.choice(mg.shape[0], 100000, replace=False)
            # X=mg[indices, 0].flatten()
            # Y=mg[indices, 1].flatten()
            # Z=mg[indices, 2].flatten()
            # values = np.sin(X * Y * Z) / (X * Y * Z)
            #
            # fig = go.Figure(data=go.Volume(
            #     x=mg[indices, 0].flatten(),
            #     y=mg[indices, 1].flatten(),
            #     z=mg[indices, 2].flatten(),
            #     value=values.flatten(),
            #     isomin=0.8,
            #     isomax=1.2,
            #     opacity=0.5,  # needs to be small to see through all surfaces
            #     surface_count=15,  # needs to be a large number for good volume rendering
            # ))
            # fig.show()
            #
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # sc = ax.scatter(mg[indices, 0], mg[indices, 1], mg[indices, 2], c=dg[indices], cmap='viridis', edgecolors='None')
            # plt.colorbar(sc)
            # plt.title('3D Density Field from kernel_modified')
            # plt.show()
            #
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # sc = ax.scatter(mg[indices, 0], mg[indices, 1], mg[indices, 2], c=dg[indices], cmap='viridis', alpha=0.01, edgecolors='None')
            # plt.colorbar(sc)
            # plt.title('3D Density Field from kernel_modified')
            # plt.show()

            # kernel_modified = torch.exp(-2 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20*self.kernel_var))[:, None]
            # fig = plt.figure(figsize=(6, 6))
            # plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=10, c=to_numpy(kernel_modified))
            # plt.show()

        else:
            # out = self.lin_edge(field_j) * self.kernel_operators[:,1:2] / density_j
            # out = self.lin_edge(field_j) * self.kernel_operators[:,3:4] / density_j
            # out = field_j * self.kernel_operators[:, 1:2] / density_j

            grad_density = self.kernel_operators[:, 1:4]  # d_rho_x d_rho_y
            velocity = self.kernel_operators[:, 0:1] * torch.sum(d_pos_j**2, dim=1)[:,None] / density_j
            out = torch.cat((velocity, grad_density), dim=-1)

            return out



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
        sigma = 2 * np.random.uniform(0.05, 0.1)
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

def remove_files_from_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it is a file and remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it is a directory and remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist.')




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

    remove_files_from_folder('tmp')

    mode = 'cell_gland_SMG2_smooth10_1'


    if mode == 'cell_gland_SMG2_smooth10_1':
        config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/cell/cell_gland_SMG2_smooth10_1.yaml')

    device = 'cuda:0'
    dimension = 3
    bc_pos, bc_dpos = choose_boundary_values('no')
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    lr = config.training.learning_rate_start
    batch_size = config.training.batch_size

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')

    model = Operator_smooth3D(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos, dimension=dimension)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    phi = torch.zeros(1, device=device)
    threshold = 0.05

    config_list = ['cell_gland_SMG2_smooth10_8', 'cell_gland_SMG2_smooth10_9', 'cell_gland_SMG2_smooth10_10',
                   'cell_gland_SMG2_smooth10_5', 'cell_gland_SMG2_smooth10_6', 'cell_gland_SMG2_smooth10_7']

    for config_file in config_list:

        x_list = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/x_inference_list_0.pt', map_location=device, weights_only=True)
        posnorm = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/posnorm.pt', map_location=device, weights_only=True)
        bounding_box = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/bounding_box.pt', map_location=device, weights_only=True)

        xz_ratio = bounding_box[2] / bounding_box[0]
        grid_size = 80
        gx = torch.linspace(0, bounding_box[0] * posnorm, steps=grid_size)
        gy = torch.linspace(0, bounding_box[0] * posnorm, steps=grid_size)
        gz = torch.linspace(0, bounding_box[2] * posnorm, steps=int(grid_size*xz_ratio))
        gx, gy, gz = torch.meshgrid(gx, gy, gz)
        mgrid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid, torch.zeros((mgrid.shape[0], 3))), 1)
        mgrid = mgrid.to(device)


        for frame in trange(0,len(x_list)):

            x = x_list[frame]

            check_and_clear_memory(device=device, iteration_number=frame, every_n_iterations=len(x_list) // 10,
                                   memory_percentage_threshold=0.6)

            # print(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # print(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            optimizer.zero_grad()
            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            data_id = torch.zeros((x.shape[0], 1), dtype=torch.int)
            dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

            pred = model(dataset, data_id=data_id, training=False, phi=phi)
            density = model.density.clone().detach()

            optimizer.zero_grad()
            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - mgrid[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > 0)).float() * 1
            edge_index_mgrid = adj_t.nonzero().t().contiguous()
            xp = torch.cat((mgrid, x[:, 0:2 * dimension + 1]), 0)
            edge_index_mgrid[0, :] = edge_index_mgrid[0, :] + mgrid.shape[0]
            edge_index_mgrid, _ = pyg_utils.remove_self_loops(edge_index_mgrid)

            dataset = data.Data(x=xp, pos=xp[:, 1:dimension + 1], edge_index=edge_index_mgrid)
            data_id = torch.zeros((xp.shape[0], 1), dtype=torch.int)
            pred_field = model(dataset, data_id=data_id, training=False, phi=phi, continuous_field=True, continuous_field_size=mgrid.shape)[0: mgrid.shape[0]]
            density_field = model.density[0: mgrid.shape[0]]

            density_field = density_field.detach().cpu().numpy()
            velocity_field = pred_field[:,0:1].detach().cpu().numpy()
            grid_shape = (grid_size, grid_size, int(grid_size * (bounding_box[2] / bounding_box[0])))
            density_field = density_field.reshape(grid_shape)
            velocity_field = velocity_field.reshape(grid_shape)

            np.save(f"/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/tmp/velocity_field_{frame}.npy",velocity_field)
            np.save(f"/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/tmp/density_field_{frame}.npy",density_field)


            fig = plt.figure(figsize=(24, 8.5))
            ax = fig.add_subplot(1,3,1)
            plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c='w')
            pixel = 7020
            plt.scatter(mgrid[pixel, 2].detach().cpu().numpy(),
                        mgrid[pixel, 1].detach().cpu().numpy(), s=2, c='r')
            pos = torch.argwhere(edge_index_mgrid[1, :] == pixel).squeeze()
            if pos.numel()>0:
                plt.scatter(xp[edge_index_mgrid[0, pos], 2].detach().cpu().numpy(), xp[edge_index_mgrid[0, pos], 1].detach().cpu().numpy(), s=1,c='b')
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0,800])
            plt.ylim([0,800])
            plt.title('nucleus positions (2D projection)', fontsize=12)

            ax = fig.add_subplot(1,3,2)
            plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c=to_numpy(density), vmin=0, vmax=30)
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0,800])
            plt.ylim([0,800])
            plt.title('density (2D projection)', fontsize=12)

            ax = fig.add_subplot(1,3,3)
            plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c=to_numpy(pred[:,0]),  vmin=0, vmax=18)
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0,800])
            plt.ylim([0,800])
            plt.title('velocity (2D projection)', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/tmp_recons2D/frame_{frame}.png")
            plt.close()

















            # velocity_field = np.load(f"/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/tmp/velocity_field_{frame}.npy")
            # density_field = np.load(f"/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/{config_file}/tmp/density_field_{frame}.npy")
            #
            # # upscale_factor = 4  # Adjust the factor as needed
            # # velocity_field = scipy.ndimage.zoom(velocity_field, upscale_factor, order=1)
            # # density_field = scipy.ndimage.zoom(density_field, upscale_factor, order=1)
            #
            # viewer = napari.Viewer()
            # viewer.add_image(velocity_field, name='Density Field', colormap='viridis', contrast_limits=[0, 1])
            # viewer.dims.ndisplay = 3
            # viewer.camera.zoom = 12
            # viewer.camera.angles = (113, 62, 20)
            # viewer.camera.center = (51.6, 38, 6.84)
            # screenshot = viewer.screenshot()
            # screenshot = screenshot[:, :, 0:3]
            # viewer.close()
            # imwrite(f"/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/cell_gland_SMG2_smooth10_1/tmp_recons3D/velocity_field_{frame}.tiff", screenshot, photometric='rgb')
            #
            # viewer = napari.Viewer()
            # viewer.add_image(density_field, name='Density Field', colormap='viridis', contrast_limits=[0, 75])
            # viewer.dims.ndisplay = 3
            # viewer.camera.zoom = 12
            # viewer.camera.angles = (113, 62, 20)
            # viewer.camera.center = (51.6, 38, 6.84)
            # screenshot = viewer.screenshot()
            # screenshot = screenshot[:, :, 0:3]
            # viewer.close()
            # imwrite(f"/groups/saalfeld/home/allierc/Py/ParticleGraph/log/cell/cell_gland_SMG2_smooth10_1/tmp_recons3D/density_field_{frame}.tiff", screenshot, photometric='rgb')
            #
            # napari.run()
            #
            #
            #
            #
            #
            #
            #
            #
            #








        # fig = go.Figure(data=go.Volume(
        #     x=mgrid[:, 0].flatten(),
        #     y=mgrid[:, 1].flatten(),
        #     z=mgrid[:, 2].flatten(),
        #     value=values.flatten(),
        #     isomin=0.0,
        #     isomax=10,
        #     opacity=0.1,  # needs to be small to see through all surfaces
        #     surface_count=32,  # needs to be a large number for good volume rendering
        # ))
        # fig.update_layout(
        #     title='3D Density Field',
        #     scene=dict(
        #         xaxis=dict(title='X', range=[mgrid[:, 0].min(), mgrid[:, 0].max()]),
        #         yaxis=dict(title='Y', range=[mgrid[:, 0].min(), mgrid[:, 0].max()]),
        #         zaxis=dict(title='Z', range=[mgrid[:, 0].min(), mgrid[:, 0].max()]),
        #         aspectmode='manual',
        #     )
        # )
        # fig.show()
        # pio.write_image(fig, '3d_density_field.png')





















