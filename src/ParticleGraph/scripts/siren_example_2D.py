# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file
import os

import numpy as np
import torch
import torch.nn as nn

# from ParticleGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt
from tifffile import imread, imsave
from tqdm import trange
from ParticleGraph.utils import *
from ParticleGraph.config import ParticleGraphConfig
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from GNN_particles_Ntype import *
from ParticleGraph.models.utils import *
from ParticleGraph.utils import *
from ParticleGraph.models.Siren_Network import *

if __name__ == '__main__':

    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import matplotlib.pyplot as plt
    import torch



    matplotlib.use("Qt5Agg")


    config_list = ['signal_CElegans_c2']


    for config_file_ in config_list:

        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f'../../../config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        device = set_device(config.training.device)
        print(f'device  {device}')
        print(f'folder  {config.dataset}')

        dataset_name = config.dataset

        simulation_config = config.simulation
        train_config = config.training
        model_config = config.graph_model

        print(f'training with data {model_config.particle_model_name} {model_config.mesh_model_name}')

        dimension = simulation_config.dimension
        n_epochs = train_config.n_epochs
        n_particles = simulation_config.n_particles
        n_particle_types = simulation_config.n_particle_types
        dataset_name = config.dataset
        n_frames = simulation_config.n_frames
        data_augmentation_loop = train_config.data_augmentation_loop
        recursive_loop = train_config.recursive_loop
        delta_t = simulation_config.delta_t
        particle_batch_ratio = train_config.particle_batch_ratio
        embedding_cluster = EmbeddingCluster(config)
        n_runs = train_config.n_runs
        field_type = model_config.field_type
        coeff_lin_modulation = train_config.coeff_lin_modulation
        coeff_model_b = train_config.coeff_model_b
        coeff_sign = train_config.coeff_sign
        time_step = train_config.time_step
        has_missing_activity = train_config.has_missing_activity
        multi_connectivity = config.training.multi_connectivity

        x_list = []
        y_list = []
        for run in trange(n_runs):
            if os.path.exists(f'../../../graphs_data/{dataset_name}/x_list_{run}.pt'):
                x = torch.load(f'../../../graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
                y = torch.load(f'../../../graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
                x = to_numpy(torch.stack(x))
                y = to_numpy(torch.stack(y))
            else:
                x = np.load(f'../../../graphs_data/{dataset_name}/x_list_{run}.npy')
                y = np.load(f'../../../graphs_data/{dataset_name}/y_list_{run}.npy')
            x_list.append(x)
            y_list.append(y)

        activity = torch.tensor(x_list[0], device=device)
        activity = activity[:, :, 8:9].squeeze()
        activity = activity.t()

        model = nn.ModuleList([
            Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                  hidden_features=model_config.hidden_dim_nnr,
                  hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                  hidden_omega_0=model_config.omega,
                  outermost_linear=model_config.outermost_linear_nnr)
            for n in range(n_runs)
        ])
        model.to(device=device)
        optimizer = torch.optim.Adam(lr=train_config.learning_rate_missing_activity,params=model.parameters())
        model.train()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'number of learnable parameters: {count_parameters(model) //100}')

        list_loss_regul = []
        time.sleep(0.2)

        batch_size = 1

        for epoch in range(20):

            batch_size = 20
            Niter = 10000
            plot_frequency = 1000

            total_loss = 0
            k = 0

            for N in range(Niter):

                optimizer.zero_grad()

                loss = 0
                run = np.random.randint(n_runs)

                for batch in range(batch_size):

                    k = np.random.randint(n_frames - 1 - time_step)
                    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                    if not (torch.isnan(x).any()):
                        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                        missing_activity = model[run](t).squeeze()
                        loss = loss + (missing_activity[:] - x[:, 6].clone().detach()).norm(2)

                if loss !=0:
                    loss.backward()
                    optimizer.step()
                    print(N, loss.item()/batch_size)

                if (N % plot_frequency == 0):
                    with torch.no_grad():

                        n_frames = n_frames - 10
                        t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
                        prediction = model[0](t) ** 2
                        prediction = prediction.t()

                        fig = plt.figure(figsize=(16, 16))
                        ax = fig.add_subplot(2, 2, 1)
                        plt.title('neural field')
                        plt.imshow(to_numpy(prediction), aspect='auto', cmap='viridis')
                        ax = fig.add_subplot(2, 2, 2)
                        plt.title('true activity')
                        activity = torch.tensor(x_list[0][:, :, 6:7], device=device)
                        activity = activity.squeeze()
                        activity = activity.t()
                        plt.imshow(to_numpy(activity), aspect='auto', cmap='viridis')
                        plt.tight_layout()
                        plt.show()






























