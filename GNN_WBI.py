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

    parser = argparse.ArgumentParser(description="ParticleGraph")

    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='arbitrary_3',
        help='the name of config file'
    )

    args = parser.parse_args()

    config_file = args.config

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    config_list = [config_file]
    config_list = ['signal_N2_WBI']

    for config_file in config_list:

        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        device = set_device(config.training.device)
        print(f'device {device}')
        data_generate(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=False, bSave=True, step=10) #config.simulation.n_frames // 100)
        # data_train(config, config_file, True, device)
        # data_test(config=config, config_file=config_file, visualize=True, style='color', verbose=False, best_model='1', run=0, step=25, test_simulation=True, sample_embedding=False, device=device)    # config.simulation.n_frames // 7
