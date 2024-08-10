import time
from shutil import copyfile

# import networkx as nx
import scipy.io
import torch
# import networkx as nx
import torch.nn as nn
import torch_geometric.data as data
from sklearn import metrics
from tifffile import imread
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
# matplotlib.use("Qt5Agg")
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torchvision.transforms import GaussianBlur
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from prettytable import PrettyTable

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.data_loaders import *
from ParticleGraph.sparsify import *

from ParticleGraph.generators.utils import *
from ParticleGraph.generators.graph_data_generator import *
from ParticleGraph.models.graph_trainer import *

from ParticleGraph.models import Siren_Network
from ParticleGraph.models.Ghost_Particles import Ghost_Particles
from ParticleGraph.models.utils import *
from ParticleGraph.utils import *


from GNN_particles_Ntype import *


# matplotlib.use("Qt5Agg")


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(' ')
    print(f'device {device}')
    print(' ')

    # matplotlib.use("Qt5Agg")

    #config_list = ['arbitrary_3', 'arbitrary_3_3', 'arbitrary_16', 'arbitrary_32', 'arbitrary_64', 'gravity_16', 'boids_16_256', 'Coulomb_3', 'arbitrary_3_sequence', 'arbitrary_3_field_video_bison', 'boids_16_256_division', 'wave_slit', 'RD_RPS_1']
    # config_list = ['boids_16_256_divisionX']

    # config_list = ["boids_16_256_divisionN", "boids_16_256_divisionO", "boids_16_256_divisionP", "boids_16_256_divisionQ", "boids_16_256_divisionR", 
    #                "boids_16_256_divisionS", "boids_16_256_divisionT", "boids_16_256_divisionU", "boids_16_256_divisionV", "boids_16_256_divisionW", ]

    config_list = ["boids_16_256_divisionX_deltat2"]

    for config_file in config_list:
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

        data_generate(config, device=device, visualize=True, run_vizualized=0, style='frame color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 400)


    logging.shutdown()