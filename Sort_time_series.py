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
from ParticleGraph.generators.graph_data_generator import *
from ParticleGraph.models.graph_trainer import *
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles
from ParticleGraph.models.utils import *

import warnings


if __name__ == '__main__':



    config_list = ['cell_MDCK_14']


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

        dataset_name = config.dataset
        data_folder_name = config.data_folder_name

        files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
        for f in files:
            os.remove(f)

        files = os.listdir(data_folder_name)
        files = [f for f in files if f.endswith('.tif')]
        files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

        im = tifffile.imread(data_folder_name + files[0])
        print(f'image size {im.shape}, frames {len(files)}')


        # Load the data
        x_list = np.load(f'graphs_data/{dataset_name}/x_list_0.npz', allow_pickle=True)
        time_series = np.load(f'graphs_data/{dataset_name}/time_series_list_0.npz')

        time_series_list = []
        for k in trange(0, len(time_series)):
            if len(time_series[f'arr_{k}'])>0:
                tmp= time_series[f'arr_{k}'][:,1:2]
                first_t = time_series[f'arr_{k}'][0,0]
                if first_t>0:
                    tmp = np.concatenate((np.zeros((int(first_t),1)),tmp))
                if tmp.shape[0] < 4310:
                    tmp = np.concatenate((tmp, np.zeros((4310-tmp.shape[0],1))))
                time_series_list.append(tmp)
            else:
                tmp = np.zeros((4310,1))
                time_series_list.append(tmp)

        time_series_list_map = np.array(time_series_list).squeeze()

        # dark style
        plt.style.use('dark_background')

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        ax.set_title('kinograph')
        ax.set_xlabel('time')
        ax.set_ylabel('cell')
        plt.imshow(time_series_list_map[0:700], cmap='viridis', aspect='auto', interpolation='nearest')
        plt.savefig(f'graphs_data/{dataset_name}/kinograph.png', dpi=300)
        plt.close()


        cell_oi = 494
        time_series_oi = time_series_list_map[cell_oi, 0:4300]

        # Sampling interval and frequency
        dt = 10.0  # seconds between samples
        fs = 1.0 / dt  # sampling frequency in Hz
        N = len(time_series_oi)
        t = np.arange(N) * dt  # time axis

        # Compute FFT
        fft_vals = np.fft.fft(time_series_oi)
        freqs = np.fft.fftfreq(N, dt)

        # Keep only the positive frequencies
        positive_freqs = freqs[:N // 2]
        positive_fft = np.abs(fft_vals[:N // 2])

        plt.figure(figsize=(16, 8))
        ax = plt.subplot(211)
        plt.plot(time_series_list_map[cell_oi, 0:4300], color='white', alpha=1)
        ax = plt.subplot(212)
        # Optionally skip first few low frequencies (e.g., first 2 bins)
        skip = 2
        plt.plot(positive_freqs[skip:], positive_fft[skip:])
        plt.yscale('log')  # Log scale helps reveal smaller peaks
        plt.title("Fourier Spectrum (Log Amplitude)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Log Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()


        time_oi=[0,285]

        for it in trange(time_oi[0], time_oi[1], 1):
            im = tifffile.imread(data_folder_name + files[it+1])
            im = np.array(im).astype('float32')

            x = x_list[f'arr_{it}']
            pos = np.argwhere(x[:,0]==cell_oi)

            plt.figure(figsize=(14, 12))
            ax = plt.subplot(221)
            plt.imshow(im[:,:,0],vmin=0,vmax=255)
            plt.scatter(x[pos, 2], im.shape[0]-x[pos, 1]-75, c='red', s=4)
            ax = plt.subplot(222)
            plt.plot(time_series_list_map[cell_oi, 0:4300], linewidth=0.5, color='white', alpha=1)
            plt.plot(time_series_list_map[cell_oi, 0:it], linewidth=1, color='red', alpha=1)
            plt.scatter(it, time_series_list_map[cell_oi, it], c='red', s=4)
            plt.text(0.01, 2900, f"frame: {it}\ncell: {cell_oi}", verticalalignment='top', horizontalalignment='left', fontsize=10)
            plt.ylim([0,3000])
            ax = plt.subplot(223)
            plt.imshow(np.fliplr(np.flipud(im[:,:,0])),vmin=0,vmax=255)
            plt.xlim([im.shape[1]-x[pos, 2]+200, im.shape[1]-x[pos, 2]-200])
            plt.ylim([x[pos, 1]-200, x[pos, 1]+200])
            ax = plt.subplot(224)
            plt.plot(time_series_list_map[cell_oi, time_oi[0]:time_oi[1]], linewidth=1, color='white', alpha=1)
            plt.plot(time_series_list_map[cell_oi, time_oi[0]:it], linewidth=1, color='red', alpha=1)
            plt.scatter(it, time_series_list_map[cell_oi, it], c='red', s=4)

            plt.tight_layout()
            plt.savefig(f'graphs_data/{dataset_name}/Fig/frame_{it:06}.tif', dpi=100)
            plt.close()














        # ax = plt.subplot(313)
        # # Find the top 5 frequencies (excluding DC component at freq = 0)
        # nonzero_indices = np.where(positive_freqs > 0)
        # top_indices = positive_fft[nonzero_indices].argsort()[-5:][::-1]
        # top_frequencies = positive_freqs[nonzero_indices][top_indices]
        # print("Top 5 frequencies (Hz):", top_frequencies)
        # for i, f in enumerate(top_frequencies):
        #     sine_wave = np.sin(2 * np.pi * f * t)
        #     plt.plot(t, sine_wave, label=f"{f:.4f} Hz")
        # plt.title("Top 5 Frequency Sinusoids")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.grid(True)













