import time
from shutil import copyfile

import networkx as nx
import scipy.io
import torch
# import networkx as nx
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

    matplotlib.use("Qt5Agg")

    # config_list = ["boids_16_256_division_model_2_small"]
    # config_list = ["boids_16_division_model_2_Voronoi_3D"]
    # config_list = ["boids_division_model_j"]
    # config_list = ["signal_N_100_2_asym_a"]
    config_list = ["arbitrary_3_sequence_d"]
    # config_list = ["boids_voronoi_division_a"]
    # config_list = ["agents_e"]

    for config_file in config_list:
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        device = set_device(config.training.device)
        print(f'device {device}')
        # data_generate(config, device=device, visualize=True, run_vizualized=0, style='voronoi', alpha=1, erase=True, bSave=True, step=1) # config.simulation.n_frames // config.simulation.n_frames)
        data_train(config, config_file, device)
        # data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False, best_model=20, run=1, step=1) #config.simulation.n_frames // 3, test_simulation=False, sample_embedding=False, device=device)    # config.simulation.n_frames // 7

