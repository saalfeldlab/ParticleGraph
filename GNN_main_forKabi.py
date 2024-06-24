import glob
import logging
import time
from shutil import copyfile

import matplotlib.pyplot as plt
# import networkx as nx
import torch.nn as nn
import torch_geometric.data as data
import umap
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from tqdm import trange
import os
import scipy.io
from sklearn import metrics
from matplotlib import rc
import matplotlib
import networkx as nx

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.graph_data_generator import *

from ParticleGraph.models.utils import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles

from ParticleGraph.data_loaders import *
from ParticleGraph.utils import *
from ParticleGraph.fitting_models import linear_model
from ParticleGraph.embedding_cluster import *
from ParticleGraph.models import Division_Predictor
from GNN_particles_Ntype import *


if __name__ == '__main__':


    config_list = ['arbitrary_3_field_video_bison']

    for config_file in config_list:
        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        # print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        data_generate(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 25)
        #data_train(config, device=device)
        #data_test(config, visualize=True, verbose=False, best_model=8, run=0, step=config.simulation.n_frames // 25, test_simulation=False, device=device)



