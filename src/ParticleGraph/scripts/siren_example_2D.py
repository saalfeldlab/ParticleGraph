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
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats


if __name__ == '__main__':

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

        model = Siren(in_features=1, hidden_features=256, hidden_layers=3, out_features=300, outermost_linear=True,
                      first_omega_0=30., hidden_omega_0=30.)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

        batch_size = 100
        Niter = 10000
        plot_frequency = 100000 // batch_size
        print(f'plot frequency {plot_frequency}')

        for epoch in range(20):

            total_loss = 0
            k = 0

            for N in trange(Niter):

                optimizer.zero_grad()

                loss = 0
                run = 0  #np.random.randint(n_runs)

                for batch in range(batch_size):

                    k = np.random.randint(n_frames - 1 - time_step)
                    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                    if not (torch.isnan(x).any()):
                        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                        missing_activity = model(t).squeeze()
                        loss = loss + (missing_activity - x[:, 6].clone().detach()).norm(2)

                if loss !=0:
                    loss.backward()
                    optimizer.step()
                    # print(N, loss.item()/batch_size)

                if (N % plot_frequency == 0):
                    with torch.no_grad():
                        t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
                        prediction = model(t) ** 2
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
                        activity = torch.nan_to_num(activity, nan=0.0)
                        plt.imshow(to_numpy(activity), aspect='auto', cmap='viridis')
                        ax = fig.add_subplot(2, 2, 3)
                        plt.scatter(to_numpy(activity.flatten()), to_numpy(prediction.flatten()), s=0.1, alpha=0.5,c='k')

                        activity_np = to_numpy(activity.flatten())
                        prediction_np = to_numpy(prediction.flatten())

                        # Remove points where activity = 0
                        non_zero_mask = activity_np != 0
                        activity_filtered = activity_np[non_zero_mask]
                        prediction_filtered = prediction_np[non_zero_mask]

                        # Plot all points (including zeros) in light color
                        plt.scatter(activity_np, prediction_np, s=0.1, alpha=0.3, c='lightgray', label='All data')

                        # Plot non-zero points in dark color
                        plt.scatter(activity_filtered, prediction_filtered, s=0.1, alpha=0.5, c='k',
                                    label='Non-zero activity')

                        # Calculate statistics on filtered data
                        r2 = r2_score(activity_filtered, prediction_filtered)
                        slope, intercept, r_value, p_value, std_err = stats.linregress(activity_filtered,
                                                                                       prediction_filtered)

                        # Add regression line for filtered data
                        x_line = np.linspace(activity_filtered.min(), activity_filtered.max(), 100)
                        y_line = slope * x_line + intercept
                        plt.plot(x_line, y_line, 'r-', linewidth=2)

                        # Statistics text
                        plt.text(0.05, 0.95,
                                 f'R² = {r2:.4f}\nSlope = {slope:.4f}\nIntercept = {intercept:.4f}\nN = {len(activity_filtered)}',
                                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                        plt.xlabel('Actual Activity')
                        plt.ylabel('Predicted Activity')
                        plt.title('Prediction vs Actual (Excluding Zero Activity)')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()






























