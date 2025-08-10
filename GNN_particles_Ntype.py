import time
from shutil import copyfile
import argparse
import networkx as nx
import os
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
matplotlib.use("Agg")  # Non-interactive backend
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
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass

    parser = argparse.ArgumentParser(description="ParticleGraph")
    parser.add_argument(
        "-o", "--option", nargs="+", help="Option that takes multiple values"
    )

    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option != None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:
        task = 'generate'  # 'generate', 'train', 'test'
        # config_list = ['multimaterial_1_2', 'multimaterial_1_3', 'multimaterial_1_4', 'multimaterial_1_5', 'multimaterial_1_6', 'multimaterial_1_7', 'multimaterial_1_8']
        # config_list = ['multimaterial_1_C']
        # config_list = ['multimaterial_1_12']
        config_list = ['multimaterial_2_3']
        # config_list = ['fluids_m19']
        # config_list = ['falling_water_ramp_x6_11_1']
        # config_list = ['arbitrary_3']
        # config_list = ['cell_cardio_2_4']
        # config_list = ['RD_RPS_5']
        # config_list = ['cell_U2OS_9_2']a
        # config_list = ['springs_matrix_N5_3']
        # config_list = ['cell_MDCK_16']
        # config_list = ['signal_CElegans_d2', 'signal_CElegans_d2a', 'signal_CElegans_d3', 'signal_CElegans_d3a', 'signal_CElegans_d3b']
        # config_list = ['signal_CElegans_c14_4']
        # config_list = ['signal_N5_v11_bis']
        # config_list = ['signal_fig_supp6_4']
        # config_list = ['fly_N9_33_5', 'fly_N9_33_5_1'] #, 'fly_N9_33_2','fly_N9_33_3', 'fly_N9_33_4','fly_N9_32_0', 'fly_N9_32_2']
        config_list = ['fly_N9_18_4_0'] # , 'fly_N9_30_1', 'fly_N9_30_2', 'fly_N9_30_3', 'fly_N9_30_4', 'fly_N9_30_5', 'fly_N9_30_6']
        # config_list = ['signal_N5_l4','signal_N5_l5']

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f"config_file  {config.config_file}")
        print(f"\033[92mdevice  {device}\033[0m")

        if "generate" in task:
            data_generate(
                config,
                device=device,
                visualize=True,
                run_vizualized=0,
                style="black color",
                alpha=1,
                erase=False,
                bSave=False,
                step=2
            )  # config.simulation.n_frames // 100)
            
        if "train" in task:
            data_train(config=config, erase=False, best_model=None, device=device)
            
        if "test" in task:
            # for run_ in range(2,10):
            # data_test(config=config, visualize=True, style='black color name', verbose=False, best_model='best',
            #           run=run_, test_mode='fixed_bounce_all', sample_embedding=False, step=4,
            #           device=device)  # particle_of_interest=100, 'fixed_bounce_all'
            data_test(
                config=config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=20,
                device=device,
                particle_of_interest=0,
            )  # particle_of_interest=100,  'fixed_bounce_all'


# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"
