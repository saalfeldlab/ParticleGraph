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

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

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
        task = 'test'
        best_model = 'best'
        # config_list = ['fluids/fluids_m']
        # config_list = ['falling_water_ramp_x6_11']
        # config_list =['arbitrary/arbitrary_3_11']
        # config_list =['wave/wave_2']
        config_list = ['arbitrary_3']

    for config_file_ in config_list:

        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        device = set_device(config.training.device)
        print(f'device  {device}')
        print(f'folder  {config.dataset}')

        if 'generate' in task:
            data_generate(config, device=device, visualize=True, run_vizualized=0, style='black color', alpha=1, erase=False, bSave=True, step=4)  #config.simulation.n_frames // 100)
        if 'train' in task:
            data_train(config=config, erase=False, best_model=best_model, device=device)
        if 'test' in task:
            data_test(config=config, visualize=True, style='black zoom color', verbose=False, best_model='best', run=1, plot_data=False,
                      test_simulation=False, sample_embedding=False, fixed=False, bounce=False, step=10, particle_of_interest=4400, device=device)
        if 'try_func' in task:
            try_func(max_radius=config.simulation.max_radius, device=device)

            # data_test(config=config, config_file=config_file, visualize=True, style='black color', verbose=False, best_model='best', run=1, plot_data=True,
            #           test_simulation=False, sample_embedding=False, fixed=True, bounce=True, step=4, device=device)
            # data_test(config=config, config_file=config_file, visualize=True, style='black color', verbose=False, best_model='best', run=2, plot_data=False,
            #           test_simulation=False, sample_embedding=False, fixed=True, bounce=False, step=4, device=device)
            # data_test(config=config, config_file=config_file, visualize=True, style='black color', verbose=False, best_model='best', run=15, plot_data=False,
            #           test_simulation=False, sample_embedding=False, fixed=True, bounce=True, step=4, device=device)

# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"


    # for f in ['agents','arbitrary','boids','celegans', 'cell','Coulomb', 'diffusion','fluids','gravity','falling_water_ramp', 'mouse_city', 'rat_city', 'springs', 'wave' ]:
    #     # Define the directory where the folders are located
    #     directory = f'/groups/saalfeld/home/allierc/Py/ParticleGraph/log/{f}/'
    #     # Loop through all items in the directory
    #     for item in os.listdir(directory):
    #         # Check if the item is a directory and starts with 'arbitrary_'
    #         if os.path.isdir(os.path.join(directory, item)) and item.startswith('try_'):
    #             # Extract the new name by removing 'arbitrary_'
    #             new_name = item[len('try_'):]
    #             # Rename the directory
    #             print(item, new_name)
    #             os.rename(os.path.join(directory, item), os.path.join(directory, new_name))