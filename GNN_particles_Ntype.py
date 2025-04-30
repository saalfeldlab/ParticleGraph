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
from matplotlib import pyplot as plt

from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from prettytable import PrettyTable

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.utils import *
from ParticleGraph.generators.graph_data_generator import *
from ParticleGraph.models.graph_trainer import *
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles
from ParticleGraph.models.utils import *
from ParticleGraph.utils import *
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

        task = 'train'
        best_model = ''

        # config_list = ['multimaterial_12_1','multimaterial_12_2','multimaterial_12_3','multimaterial_12_4','multimaterial_12_5',
        #                'multimaterial_13_1', 'multimaterial_13_2', 'multimaterial_13_3',
        #                'multimaterial_14_1', 'multimaterial_14_2', 'falling_water_ramp_x6_13']
        # config_list = ['fluids_m17_7']
        # config_list = ['signal_N5_v11_bis']
        # config_list = ['falling_water_ramp_x6_13']
        # config_list = ['cell_MDCK_14']
        # config_list = ['signal_N2_a37']
        # config_list = ['signal_N4_CElegans_a6', 'signal_N4_CElegans_a7', 'signal_N4_CElegans_a7_1',
        #                'signal_N4_CElegans_a7_2', 'signal_N4_CElegans_a8',
        #                'signal_N4_CElegans_a8_1', 'signal_N4_CElegans_a8_2', 'signal_N4_CElegans_a8_3',
        #                'signal_N4_CElegans_a9', 'signal_N4_CElegans_a9_1', 'signal_N4_CElegans_a9_2',
        #                'signal_N4_CElegans_a9_3', 'signal_N4_CElegans_a9_4', 'signal_N4_CElegans_a9_5']
        config_list = ['signal_N4_CElegans_a9_6']

    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f'config_file  {config.config_file}')
        print(f'device  {device}')
        print(f'folder  {config.dataset}')

        if 'generate' in task:
            data_generate(config, device=device, visualize=True, run_vizualized=0, style='black field', alpha=1, erase=False, bSave=True, step=100)  #config.simulation.n_frames // 100)
        if 'train' in task:
            data_train(config=config, erase=False, best_model=best_model, device=device)
        if 'test' in task:
            # for run_ in range(0, config.simulation.n_frames, 50):
            #     data_test(config=config, visualize=True, style='black color name', verbose=False, best_model='best',
            #               run=run_, test_mode='fixed_bounce_all', sample_embedding=False, step=4,
            #               device=device)  # particle_of_interest=100,
            data_test(config=config, visualize=True, style='black color name', verbose=False, best_model='best', run=0, test_mode='', sample_embedding=False, step=4, device=device)  # particle_of_interest=100,

    if 'try_func' in task:
            try_func(max_radius=config.simulation.max_radius, device=device)




# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"

