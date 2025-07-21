import time
from shutil import copyfile
import argparse
import networkx as nx
import scipy.io
import umap
import torch
import torch.nn as nn
import torch_geometric.data as data
from sklearn import metrics
from tifffile import imread
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torchvision.transforms import GaussianBlur
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from prettytable import PrettyTable

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.graph_data_generator import *
from ParticleGraph.models.graph_trainer import *
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles
from ParticleGraph.models.utils import *

import warnings

if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)
    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass

    parser = argparse.ArgumentParser(description="ParticleGraph")
    parser.add_argument('-o', '--option', nargs='+', help='Option that takes multiple values')

    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option!=None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:

        task = ''  # 'generate', 'train', 'test'
        best_model = ''
        # config_list = ['multimaterial_4_1']
        # config_list = ['fluids_m19']
        # config_list = ['falling_water_ramp_x6_11_1']
        # config_list = ['arbitrary_3']
        # config_list = ['cell_cardio_2_4']
        # config_list = ['RD_RPS_5']
        # config_list = ['cell_U2OS_9_2']
        # config_list = ['springs_matrix_N5_3']
        # config_list = ['cell_MDCK_16']
        # config_list = ['signal_CElegans_d2', 'signal_CElegans_d2a', 'signal_CElegans_d3', 'signal_CElegans_d3a', 'signal_CElegans_d3b']
        # config_list = ['signal_CElegans_c14_4']
        # config_list = ['signal_N5_v11_bis']
        config_list = ['fly_N9_19']
        # config_list = ['signal_N5_l4','signal_N5_l5']

    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f'config_file  {config.config_file}')
        print(f'\033[92mdevice  {device}\033[0m')
        print(f'folder  {config.dataset}')

        if 'generate' in task:
            data_generate(config, device=device, visualize=True, run_vizualized=0, style='black color', alpha=1, erase=False, bSave=True, step=20)  #config.simulation.n_frames // 100)
        if 'train' in task:
            data_train(config=config, erase=False, best_model=best_model, device=device)
        if 'test' in task:
            # for run_ in range(2,10):
                # data_test(config=config, visualize=True, style='black color name', verbose=False, best_model='best',
                #           run=run_, test_mode='fixed_bounce_all', sample_embedding=False, step=4,
                #           device=device)  # particle_of_interest=100, 'fixed_bounce_all'
            data_test(config=config, visualize=True, style='black color name', verbose=False, best_model='best', run=0,
                      test_mode='', sample_embedding=False, step=2, device=device, particle_of_interest=0)  # particle_of_interest=100,  'fixed_bounce_all'


    import os
    import numpy as np
    import torch
    import glob
    from tqdm import tqdm


    def add_individual_types_to_datasets():
        base_path = "/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/fly"

        # First, we need to get the flyvis connectome to extract individual cell types
        try:
            from flyvis import Network
            from flyvis.utils.config_utils import get_default_config, CONFIG_PATH

            # Load flyvis network to get node types
            print("Loading flyvis network to extract cell type mapping...")
            config = get_default_config(
                overrides=[], path=f"{CONFIG_PATH}/network/network.yaml"
            )
            config.connectome.extent = 8  # Use reasonable extent
            net = Network(**config)

            # Extract node types and create integer mapping (0-63)
            node_types = np.array(net.connectome.nodes['type'])
            unique_types, node_types_int = np.unique(node_types, return_inverse=True)

            print(f"Found {len(unique_types)} unique cell types")
            print(f"Cell type range: {node_types_int.min()} to {node_types_int.max()}")

            # Print mapping for verification
            print("\nCell type mapping:")
            for i, cell_type in enumerate(unique_types):
                cell_name = cell_type.decode('utf-8') if isinstance(cell_type, bytes) else str(cell_type)
                print(f"{i}: {cell_name}")

        except ImportError:
            print("Error: Cannot import flyvis. Make sure flyvis is installed.")
            return
        except Exception as e:
            print(f"Error loading flyvis network: {e}")
            return

        # Find all subdirectories in the fly folder
        fly_folders = []
        if os.path.exists(base_path):
            fly_folders = [d for d in os.listdir(base_path)
                           if os.path.isdir(os.path.join(base_path, d))]
        else:
            print(f"Base path {base_path} does not exist!")
            return

        print(f"\nFound {len(fly_folders)} fly dataset folders:")
        for folder in fly_folders:
            print(f"  - {folder}")

        # Process each folder
        for folder in tqdm(fly_folders, desc="Processing folders"):
            folder_path = os.path.join(base_path, folder)
            x_list_path = os.path.join(folder_path, "x_list_0.npy")

            if not os.path.exists(x_list_path):
                print(f"  WARNING: {x_list_path} does not exist, skipping...")
                continue

            try:
                # Load existing x_list
                print(f"\nProcessing {folder}...")
                x_list = np.load(x_list_path)
                print(f"  Original x_list shape: {x_list.shape}")

                # Check if we need to add the column
                if x_list.shape[-1] >= 7:
                    print(f"  Already has {x_list.shape[-1]} columns, checking if column 6 exists...")
                    # Check if column 6 has the right data
                    sample_frame = x_list[0]  # First frame
                    if len(np.unique(sample_frame[:, 6])) > 30:  # Should have ~64 unique types
                        print(
                            f"  Column 6 already exists with {len(np.unique(sample_frame[:, 6]))} unique values, skipping...")
                        continue

                # Create new array with additional column (preserve original dtype!)
                n_frames, n_neurons, n_features = x_list.shape
                print(f"  Original dtype: {x_list.dtype}")
                new_x_list = np.zeros((n_frames, n_neurons, max(7, n_features + 1)), dtype=x_list.dtype)

                # Copy existing data
                new_x_list[:, :, :n_features] = x_list

                # Add individual cell types to column 6
                # node_types_int should match the number of neurons
                if len(node_types_int) != n_neurons:
                    print(f"  ERROR: Mismatch in neuron count. Expected {n_neurons}, got {len(node_types_int)}")
                    continue

                # Add the individual types to all frames
                for frame_idx in range(n_frames):
                    new_x_list[frame_idx, :, 6] = node_types_int

                print(f"  New x_list shape: {new_x_list.shape}")
                print(f"  Individual types range: {node_types_int.min()} to {node_types_int.max()}")
                print(f"  Number of unique individual types: {len(np.unique(node_types_int))}")

                # Save backup of original file
                backup_path = os.path.join(folder_path, "x_list_0_backup.npy")
                if not os.path.exists(backup_path):
                    np.save(backup_path, x_list)
                    print(f"  Saved backup to {backup_path}")

                # Save updated file
                np.save(x_list_path, new_x_list)
                print(f"  Updated {x_list_path}")

            except Exception as e:
                print(f"  ERROR processing {folder}: {e}")
                continue

        print("\nDone processing all folders!")

        # Verification step
        print("\nVerification - checking a few files:")
        for folder in fly_folders[:3]:  # Check first 3 folders
            x_list_path = os.path.join(base_path, folder, "x_list_0.npy")
            if os.path.exists(x_list_path):
                try:
                    x_list = np.load(x_list_path)
                    sample_frame = x_list[0]
                    grouped_types = sample_frame[:, 5]
                    individual_types = sample_frame[:, 6]
                    print(f"\n{folder}:")
                    print(f"  Shape: {x_list.shape}")
                    print(
                        f"  Grouped types (col 5): {len(np.unique(grouped_types))} unique values, range {grouped_types.min():.0f}-{grouped_types.max():.0f}")
                    print(
                        f"  Individual types (col 6): {len(np.unique(individual_types))} unique values, range {individual_types.min():.0f}-{individual_types.max():.0f}")
                except Exception as e:
                    print(f"  ERROR verifying {folder}: {e}")



    add_individual_types_to_datasets()


# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"

