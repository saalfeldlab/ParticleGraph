import time
from shutil import copyfile
import argparse
import networkx as nx
import scipy.io
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

if __name__ == '__main__':

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    parser = argparse.ArgumentParser(description="ParticleGraph")
    parser.add_argument('-o', '--option', nargs='+', help='Option that takes multiple values')

    args = parser.parse_args()

    # Use the argument
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
        task = 'generate'
        best_model = None

        # config_list = ['rat_city_c2']
        # config_list = ['cell_PSC_5']
        config_list = ['falling_water_ramp_x4'] #, 'falling_water_ramp_x2','falling_water_ramp_x3']
        # config_list = ['gravity_16_RK']


    for config_file in config_list:

        print('')
        print(f'config_file {config_file}')
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        device = set_device(config.training.device)
        print(f'device {device}')
        if 'generate' in task:
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='black color', alpha=1, erase=False, bSave=True, step=10)  #config.simulation.n_frames // 100)
        if 'train' in task:
            data_train(config=config, config_file=config_file, erase=True, best_model=best_model, device=device)
        if 'test' in task:
            data_test(config=config, config_file=config_file, visualize=True, style='black color arrow speed acc_learned boundary', verbose=False, best_model='best', run=2, plot_data=True,
                      test_simulation=False, sample_embedding=False, device=device, fixed=True, bounce=True, step=4) # config.simulation.n_frames // 200, )  arrow speed acc_learned   arrow speed acc_true


# bsub -n 4 -gpu "num=1" -q gpu_h100 "python GNN_particles_Ntype.py -o train falling_water_ramp_x1"
