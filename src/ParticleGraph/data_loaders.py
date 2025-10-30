# from vedo.examples.basic.scalarbars import cmaps

from ParticleGraph.generators.utils import *
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Literal
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.units import Unit
from scipy.interpolate import CubicSpline, interp1d, make_interp_spline
from tqdm import trange

from ParticleGraph.TimeSeries import TimeSeries

import json
from tqdm import trange
import matplotlib
from skimage.measure import label, regionprops
import tifffile
import torch_geometric.data as data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from cellpose import models, denoise
from ParticleGraph.generators.cell_utils import *

import scipy.io as sio
import seaborn as sns
from torch_geometric.utils import dense_to_sparse
import pickle
import json
import scipy.io
# import h5py
import re
from skimage.draw import disk
from skimage.transform import resize
from skimage import filters, feature
import pandas as pd
import scipy.io
from matplotlib.colors import LinearSegmentedColormap
from ParticleGraph.models.Siren_Network import *
import pywt
import torch.nn.functional as F
from scipy.optimize import curve_fit


def linear_model(x, a, b):
    return a * x + b


def extract_object_properties(segmentation_image, fluorescence_image=[], radius=40, offset_channel=[0.0, 0.0]):
    # Label the objects in the segmentation image
    labeled_image = label(segmentation_image)
    fluorescence_image = np.flipud(fluorescence_image)
    # fig = plt.figure(figsize=(13, 10.5))
    # plt.imshow(fluorescence_image)
    # plt.show()

    # Extract properties of the labeled objects
    object_properties = []
    for id, region in enumerate(regionprops(labeled_image, intensity_image=fluorescence_image)):
        # Get the cell ID
        cell_id = id

        pos_x = region.centroid[0]
        pos_y = region.centroid[1]

        # Calculate the area of the object
        area = region.area

        if area>8:

            # Calculate the perimeter of the object
            perimeter = region.perimeter

            # Calculate the aspect ratio of the bounding box
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-6)

            # Calculate the orientation of the object
            orientation = region.orientation

            rr, cc = disk((pos_x+offset_channel[0], pos_y+offset_channel[1]), radius, shape=fluorescence_image.shape)

            # Ensure the coordinates are within bounds
            valid_coords = (rr >= 0) & (rr < fluorescence_image.shape[0]) & (cc >= 0) & (
                        cc < fluorescence_image.shape[1])

            rr_valid = rr[valid_coords]
            cc_valid = cc[valid_coords]

            # Extract the fluorescence values inside the circular mask
            fluo_sum_radius = np.sum(fluorescence_image[rr_valid, cc_valid])
            fluo_sum_segmentation = region.mean_intensity * area


            object_properties.append((id, pos_x, pos_y, area, perimeter, aspect_ratio, orientation, fluo_sum_segmentation, fluo_sum_radius))

    # tmp = fluorescence_image
    # tmp[rr_valid_104, cc_valid_104] = tmp[rr_valid_104, cc_valid_104] + 0.25
    # fig = plt.figure(figsize=(13, 10.5))
    # plt.imshow(tmp)
    #
    #
    # fig = plt.figure(figsize=(13, 10.5))
    # plt.imshow(fluorescence_image)
    # for i in range(len(object_properties)):
    #     pos_x = object_properties[i][1]
    #     pos_y = object_properties[i][2]
    #     plt.scatter(pos_y, pos_x, s=100, c=object_properties[i][7], cmap='viridis', vmin=0, vmax=4000, alpha=0.75)
    #     plt.text(pos_y, pos_x, f'{i}', fontsize=10, color='w')
    # plt.show()


    return object_properties


def find_closest_neighbors(track, x):
    closest_neighbors = []
    for row in track:
        distances = torch.sqrt(torch.sum((x[:, 1:3] - row[1:3]) ** 2, dim=1))
        closest_index = torch.argmin(distances)
        closest_neighbors.append(closest_index.item())
    return closest_neighbors


def get_index_particles(x, n_particle_types, dimension):
    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    return index_particles


def skip_to(file, start_line):
    with open(file) as f:
        pos = 0
        cur_line = f.readline()
        while cur_line != start_line:
            pos += 1
            cur_line = f.readline()

        return pos + 1


def load_solar_system(config, device=None, visualize=False, step=1000):
    # create output folder, empty it if bErase=True, copy files into it
    dataset_name = config.data_folder_name
    simulation_config = config.simulation
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_step = simulation_config.n_frames + 3
    n_frames = simulation_config.n_frames
    # Start = 1980 - 03 - 06
    # Stop = 2013 - 03 - 06
    # Step = 4(hours)

    object_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 'io',
                   'europa', 'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea', 'titan', 'hyperion', 'moon',
                   'phobos', 'deimos', 'charon']

    # matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(12, 12))

    all_data = []

    for id, object in enumerate(object_list):

        print(f'object: {object}')
        filename = os.path.join(dataset_name, f'{object}.txt')

        df = skip_to(filename, "$$SOE\n")
        data = pd.read_csv(filename, header=None, skiprows=df, nrows=n_step)
        x = data.iloc[:, 2:3].values
        y = data.iloc[:, 3:4].values
        z = data.iloc[:, 4:5].values

        # convert string to float
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        z = torch.tensor(z, dtype=torch.float32, device=device)
        vx = torch.zeros_like(x)
        vy = torch.zeros_like(y)
        vz = torch.zeros_like(z)
        vx[1:] = (x[1:] - x[:-1]) / simulation_config.delta_t
        vy[1:] = (y[1:] - y[:-1]) / simulation_config.delta_t
        vz[1:] = (z[1:] - z[:-1]) / simulation_config.delta_t
        ax = torch.zeros_like(x)
        ay = torch.zeros_like(y)
        az = torch.zeros_like(z)
        ax[2:] = (vx[2:] - vx[1:-1]) / simulation_config.delta_t
        ay[2:] = (vy[2:] - vy[1:-1]) / simulation_config.delta_t
        az[2:] = (vz[2:] - vz[1:-1]) / simulation_config.delta_t

        object_data = torch.cat((torch.ones_like(x[:, None]) * id, x[:, None], y[:, None], z[:, None], vx[:, None],
                                 vy[:, None], vz[:, None], ax[:, None],
                                 ay[:, None], az[:, None],
                                 torch.zeros_like(x[:, None])), 1)
        object_data = object_data.squeeze()
        object_data = object_data.to(device=device)

        all_data.append(object_data)

    # convert_data

    x_list = []
    y_list = []

    for it in trange(5, n_frames - 5):
        for n in range(25):
            x = all_data[n][it, 1]
            y = all_data[n][it, 2]
            z = all_data[n][it, 3]
            vx = all_data[n][it, 4]
            vy = all_data[n][it, 5]
            vz = all_data[n][it, 6]

            tmp = torch.stack(
                [torch.tensor(n,device=device), x, y, z, vx, vy, vz, torch.tensor(n,device=device), torch.tensor(0,device=device), torch.tensor(0,device=device), torch.tensor(0,device=device)])
            if n == 0:
                object_data = tmp[None, :]
            else:
                object_data = torch.cat((object_data, tmp[None, :]), 0)

            ax = all_data[n][it+1, 7]
            ay = all_data[n][it+1, 8]
            az = all_data[n][it+1, 9]
            tmp = torch.stack([ax, ay, az])
            if n == 0:
                acc_data = tmp[None, :]
            else:
                acc_data = torch.cat((acc_data, tmp[None, :]), 0)

        x_list.append(object_data.to(device))
        y_list.append(acc_data.to(device))

    for run in range(2):
        torch.save(x_list, f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_gravity_solar_system/x_list_{run}.pt')
        torch.save(y_list, f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_gravity_solar_system/y_list_{run}.pt')


def load_LG_ODE(config, device=None, visualize=False, step=1000):
    # create output folder, empty it if bErase=True, copy files into it
    data_folder_name = config.data_folder_name
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    n_particles = simulation_config.n_particles
    n_runs = train_config.n_runs

    # Loading Data

    files = os.listdir(data_folder_name)
    file = files[1][8:-4]

    loc = np.load(data_folder_name + 'loc_train' + file + '.npy', allow_pickle=True)
    vel = np.load(data_folder_name + 'vel_train' + file + '.npy', allow_pickle=True)
    acc = np.load(data_folder_name + 'acc_train' + file + '.npy', allow_pickle=True)
    edges = np.load(data_folder_name + 'edges_train' + file + '.npy', allow_pickle=True) # [500,5,5]
    times = np.load(data_folder_name + 'times_train' + file + '.npy', allow_pickle=True) # 【500，5]

    num_graph = loc.shape[0]
    num_atoms = loc.shape[1]
    feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]

    connection_matrix_list = []

    for run in trange(n_runs):

        connection_matrix = torch.tensor(edges[run], dtype=torch.float32, device=device)
        connection_matrix_list.append(connection_matrix)

        n_frames = loc[run][0].shape[0]

        x_list = []
        y_list = []

        for frame in range(1, n_frames-1):
            x = []
            y = []
            test = times[run][0][frame-1:frame+2]

            if test[2]-test[0] == 2:
                time_= torch.tensor(times[run][0][frame], dtype=torch.float32, device=device).repeat(num_atoms)

                for i in range(n_particles):
                    loc_ = torch.tensor(loc[run][i][frame], dtype=torch.float32, device=device)
                    vel_ = torch.tensor(vel[run][i][frame], dtype=torch.float32, device=device)
                    x_ = torch.cat((loc_, vel_), 0)
                    x.append(x_)
                    acc_ = torch.tensor(acc[run][i][frame], dtype=torch.float32, device=device)
                    y.append(acc_)

                x = torch.stack(x)
                x = torch.cat((torch.arange(n_particles, dtype=torch.float32, device=device).t()[:,None], x, time_.t()[:,None]), 1)
                x_list.append(x)

                y = torch.stack(y)
                y_list.append(y)

                if run == 0:
                    fig = plt.figure(figsize=(12, 12))
                    s_p = 100
                    plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=s_p, c='k')
                    plt.scatter(to_numpy(x[:, 2]+x[:, 4]*0.1), to_numpy(x[:, 1]+x[:, 3]*0.1), s=1, c='r')
                    plt.xlim([-3, 3])
                    plt.ylim([-3, 3])
                    plt.tight_layout()
                    num = f"{to_numpy(time_[0]):06}"
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=80)  # 170.7)
                    plt.close()

        torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')

    torch.save(connection_matrix_list, f'graphs_data/graphs_{dataset_name}/connection_matrix_list.pt')


def load_2Dfluo_data_with_Cellpose(config, device, visualize):

    """
    Pipeline for extracting calcium traces from 2D fluorescence microscopy of MDCK cells.

    Three-step process:
    1. Cellpose Segmentation (auto):
    - Segments cells using Cellpose (GPU-accelerated)
    - Creates: SEG/*.tif (masks), DN/*.tif (denoised), TRK/*.tif (downsampled blobs)
    - Stops when: TRK/_spots.csv exists

    2. TrackMate Tracking (manual):
    - User runs TrackMate in Fiji/ImageJ on TRK/*.tif
    - Settings: diameter=5, gap closing=6-6-3, min track length=20
    - Export: TRK/_spots.csv (track IDs, positions, frames)

    3. Fluorescence Extraction (auto):
    - Matches TrackMate tracks to segmented objects (<20px threshold)
    - Extracts fluorescence from specified channel
    - Creates: graphs_data/{dataset_name}/x_list_0.npz (trajectories + fluorescence)

    Output data structure (x_list):
    Each frame: [n_cells, 13] array
    - [0]: Track ID
    - [1:3]: y, x positions  
    - [3:5]: velocity (placeholder)
    - [5]: frame number
    - [6:9]: fluorescence [R, G, B]
    """

    plt.style.use('dark_background')

    data_folder_name = config.data_folder_name
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    image_data = config.image_data

    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    dimension = simulation_config.dimension
    n_frames = simulation_config.n_frames
    offset_channel = image_data.offset_channel
    delta_t = simulation_config.delta_t
    crop_region = image_data.crop_region
    run = 0

    bc_pos, bc_dpos = choose_boundary_values('no')

    # Loading Data

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)

    # files = glob.glob(f"{folder}/*")
    # for f in files:
    #     if (f[-3:] != 'Fig') & (f[-2:] != 'GT') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (f != 'generation_code.py'):
    #         os.remove(f)
    # files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
    # for f in files:
    #     os.remove(f)

    files = os.listdir(data_folder_name)
    files = [f for f in files if f.endswith('.tif')]
    files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

    im = tifffile.imread(data_folder_name + files[0])
    # crop image according to crop_region: [origin_y, origin_x, size_y, size_x]
    if crop_region != [0,0,0,0]:
        im = im[crop_region[1]:crop_region[1]+crop_region[3], crop_region[0]:crop_region[0]+crop_region[2], :]
    
        # fig = plt.figure(figsize=(13, 13))
        # plt.imshow(im/5000)
        # plt.savefig('cropped_image.png')
        # plt.close()

    print(f'image size {im.shape}, frames {len(files)}')

    os.makedirs(f"{data_folder_name}/SEG", exist_ok=True)
    os.makedirs(f"{data_folder_name}/DN", exist_ok=True)
    os.makedirs(f"{data_folder_name}/TRK", exist_ok=True)
    os.makedirs(f"{data_folder_name}/TRK_RESULT", exist_ok=True)

    cellpose_model_path = image_data.cellpose_model
    cellpose_denoise_model = image_data.cellpose_denoise_model
    cellpose_diameter = image_data.cellpose_diameter
    cellpose_channels = np.array(image_data.cellpose_channel)
    trackmate_size_ratio = image_data.trackmate_size_ratio
    trackmate_frame_step = image_data.trackmate_frame_step
    measure_diameter = image_data.measure_diameter

    # For v4.0.1+, specify pretrained_model directly
    model_path = Path.home() / ".cellpose" / "models" / cellpose_model_path
    model_cellpose = models.CellposeModel(gpu=True, pretrained_model=str(model_path))




    # model_dir = os.path.expanduser("~/.cellpose/models")
    # print("Models in directory:", os.listdir(model_dir))
    # # Check what model names are recognized
    # print("Built-in model names:", models.MODEL_NAMES if hasattr(models, 'MODEL_NAMES') else "Not accessible")
    # # Try downloading specific models manually
    # from cellpose import utils
    # # Download cyto3 if it exists
    # try:
    #     model_url = "https://www.cellpose.org/models/cyto3"
    #     model_path = os.path.join(model_dir, "cyto2_cp3")
    #     utils.download_url_to_file(model_url, model_path)
    #     print(f"Downloaded cyto2_cp3 to {cyto2_cp3}")
    # except:
    #     print("Could not download cyto3")

    import warnings


    warnings.filterwarnings("ignore", message="Resizing is deprecated")

    # step 1
    if not os.path.exists(f"{data_folder_name}/TRK/_spots.csv"):

        print('generate segmentation masks with Cellpose ...')
        for it in trange(0, len(files), ncols=80):
            im = tifffile.imread(data_folder_name + files[it])
            if crop_region != [0,0,0,0]:
                im = im[crop_region[1]:crop_region[1]+crop_region[3], crop_region[0]:crop_region[0]+crop_region[2], :]
    
            im = np.array(im).astype('float32')

            if cellpose_denoise_model != '':
                for i in cellpose_channels:
                    masks, flows, styles, imgs_dn = model_denoise.eval(im[:,:,i-1], diameter=cellpose_diameter, channels=[0,0])
                    im[:,:,i-1:i] = imgs_dn.copy()
                tifffile.imwrite(data_folder_name + 'DN/' + files[it], im[:,:,0])

            masks, flows, styles = model_cellpose.eval(im[:,:,:], 
                                           diameter=cellpose_diameter, 
                                           flow_threshold=0.0, 
                                           invert=False, 
                                           normalize=True)

            tifffile.imwrite(data_folder_name + 'SEG/' + files[it], masks)

            object_properties = extract_object_properties(masks, im[:, :, cellpose_channels[0]], radius=cellpose_diameter)
            image = im[:,:,0] * 0
            for i in range(len(object_properties)):
                cell_id = object_properties[i][0]
                pos_x = object_properties[i][1]
                pos_y = object_properties[i][2]
                rr, cc = disk((pos_x, pos_y), 8, shape=image.shape)
                image[rr, cc] = 255  # White blob
            image = resize(image,(image.shape[0] // trackmate_size_ratio, image.shape[1] // trackmate_size_ratio), anti_aliasing=True)
            tifffile.imwrite(f'{data_folder_name}/TRK/{it:06}.tif', image.astype('uint8'))

            # setp 2 trackmate
            # trackmate settings
            # diameter 5
            # distance closing gap 6 6 3
            # min track length 20

    elif not os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.npz'):

        #step 3
        df = pd.read_csv(f"{data_folder_name}/TRK/_spots.csv")

        trackmate = dict()
        trackmate['x'] = np.array(df['POSITION_X'][3:]).astype(float)
        trackmate['y'] = np.array(df['POSITION_Y'][3:]).astype(float)
        trackmate['frame'] = np.array(df['FRAME'][3:]).astype(int)
        trackmate['track_ID'] = np.array(df['TRACK_ID'][3:])
        trackmate['track_ID'] = pd.Series(trackmate['track_ID']).fillna(-1).astype(int).to_numpy()

        im = tifffile.imread(data_folder_name + files[0])
        im_dim = im.shape
        trackmate['x'] = trackmate['x'] * trackmate_size_ratio
        trackmate['y'] = trackmate['y'] * trackmate_size_ratio
        trackmate['y'] = im_dim[0] - trackmate['y']

        n_cells = np.max(trackmate['track_ID'])+100

        # Create mapping between track IDs and list indices
        unique_track_ids = np.unique(trackmate['track_ID'])
        unique_track_ids = unique_track_ids[unique_track_ids >= 0]  # Remove -1 (untracked)
        track_id_to_index = {track_id: idx for idx, track_id in enumerate(unique_track_ids)}
        n_unique_tracks = len(unique_track_ids)

        run = 0
        x_list = []
        y_list = []

        channels = ['R','G','B']
        print (f'use channel {channels[cellpose_channels[0]-1]} for trace measurements')
        channel_q = cellpose_channels[0]-1

        for it in trange(0, len(files)-2):

            im_fluo = tifffile.imread(data_folder_name + files[it])
            im_fluo = np.array(im_fluo).astype('float32') / 256
            im_seg = np.flipud(np.array(tifffile.imread(data_folder_name + 'SEG/' + files[it])))
            im_seg = np.array(im_seg)
            object_properties = extract_object_properties(im_seg, im_fluo[:,:, channel_q], radius=measure_diameter, offset_channel = offset_channel)
            object_properties = np.array(object_properties, dtype=float)

            X = object_properties[:, 1:3]
            F_fluo = np.zeros((X.shape[0], 3))
            F_fluo[:, 0:1] = object_properties[:,7:8] / 50000

            # Get TrackMate spots for current frame
            pos = np.argwhere(trackmate['frame'] == it // trackmate_frame_step).flatten()

            if len(pos) == 0:
                continue  # Skip frames with no TrackMate data

            # Extract TrackMate coordinates and IDs (keep consistent x,y order)
            X_trackmate = np.column_stack((trackmate['y'][pos], trackmate['x'][pos]))
            trackmate_IDs = trackmate['track_ID'][pos]

            # Calculate distances: each object to each TrackMate spot
            distances = np.linalg.norm(X[:, None, :] - X_trackmate[None, :, :], axis=2)

            # fig = plt.figure(figsize=(12, 12))
            # plt.scatter(X[:, 1], X[:, 0], s=20, c=F_fluo[:, 0], cmap='viridis', vmin=0, vmax=1, alpha=0.75)
            # plt.scatter(X_trackmate[:, 1], X_trackmate[:, 0], s=10, c='r', marker='x')
            # plt.savefig(f"segmentation.tif", dpi=80)
            # plt.close()

            # For each TrackMate spot, find closest object
            closest_object_indices = np.argmin(distances, axis=0)
            min_distances = np.min(distances, axis=0)

            # Apply distance threshold and bounds checking
            distance_threshold = 20  # pixels
            valid_matches = (min_distances < distance_threshold) & (closest_object_indices < F_fluo.shape[0])

            # Assign fluorescence from matched objects to TrackMate spots
            F_assigned = np.full((len(trackmate_IDs), 3), np.nan)
            if np.any(valid_matches):
                valid_object_idx = closest_object_indices[valid_matches]
                F_assigned[valid_matches, 0] = F_fluo[valid_object_idx, 0]

            # Build final array: TrackMate spots with assigned measurements
            x = np.column_stack((
                trackmate_IDs.reshape(-1, 1),  # Track IDs
                X_trackmate,  # TrackMate positions
                np.zeros((len(trackmate_IDs), 2)),  # Velocity (placeholder)
                np.full((len(trackmate_IDs), 1), it),  # Time/frame
                F_assigned  # Fluorescence measurements
            ))

            # Build final array: TrackMate spots with assigned measurements
            x_all = np.column_stack((
                trackmate_IDs.reshape(-1, 1),  # Track IDs
                X_trackmate,  # TrackMate positions
                np.zeros((len(trackmate_IDs), 2)),  # Velocity (placeholder)
                np.full((len(trackmate_IDs), 1), it),  # Time/frame
                F_assigned  # Fluorescence measurements
            ))

            # Keep only rows with valid fluorescence assignments
            valid_fluo_mask = ~np.isnan(F_assigned[:, 0])
            x = x_all[valid_fluo_mask]

            # pa = np.argwhere(X_track_ID==489)[0,0]
            # pb = np.argwhere(X_track_ID==494)[0,0]
            # print(f'cell 494 is {pb}    cell 489 is {pa}')
            # print(x[pb, 6:7], x[pa, 6:7])


            if it%4==0: #True:

                black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])
                im3 = im_fluo[:,:,channel_q]
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(121)
                plt.imshow(im3, vmin=0, vmax=10, cmap=black_to_green)
                plt.axis('off')
                ax = fig.add_subplot(122)
                plt.axis('off')
                plt.imshow(im3*0, vmin=0, vmax=10, cmap=black_to_green)
                if False: #it%100000 == 0:
                    for i in range(X_trackmate.shape[0]):
                        plt.text(X_trackmate[i,1], X_trackmate[i,0], f'{int(X_track_ID[i])}', fontsize=8, color='w')
                else:
                    plt.scatter(x[:,2], x[:,1], s=25, c=x[:,6], cmap=black_to_green, vmin=0, vmax=1)
                plt.xlim([0, im3.shape[0]])
                plt.ylim([0, im3.shape[1]])
                plt.savefig(f"{data_folder_name}/TRK_RESULT/{it:06}.tif", dpi=80)
                plt.close()

            if False:
                fig = plt.figure(figsize=(12, 12))
                plt.axis('off')
                # tmp=np.flipud(im_fluo/np.median(F[closest_indices,0:1]))
                # plt.imshow((tmp[:,:,image_data.membrane_channel]), cmap='gray', vmin=0, vmax=0.001)
                plt.imshow(im_fluo*0)
                plt.scatter(x[:,2], x[:,1], s=100, c=x[:,7], cmap='viridis', alpha=0.75)
                # if it%100==0 :
                #     for i in range(x.shape[0]):
                #         plt.text(x[i,2], x[i,1], f'{int(x[i,0])}', fontsize=8, color='w')
                plt.xlim([0 , im_dim[1]])
                plt.ylim([0 , im_dim[0]])
                plt.tight_layout()
                plt.savefig(f"{data_folder_name}/TRK_RESULT/{it:06}.tif", dpi=80)
                plt.close()

            if False: #(it>0) & (image_data.tracking_file != ''):

                positions_prev = x_list[-1][:, 1:3]
                positions_curr = x[:, 1:3]
                fluo_prev = x_list[-1][:, 7:8]
                fluo_curr = x[:, 7:8]

                track_ids_prev = x_list[-1][:, 12]
                track_ids_curr = x[:, 12]

                V = np.zeros_like(positions_curr)
                # Compute the time step (assuming uniform time step)
                for i, track_id in enumerate(track_ids_curr):
                    # Find the corresponding index in the previous positions
                    prev_index = np.where((track_ids_prev == track_id)&(track_id>-1))
                    prev_index = prev_index[0]
                    try:
                        if prev_index.size > 0:
                            V[i] = (positions_curr[i] - positions_prev[prev_index]) / delta_t
                            F[i,0] = (fluo_curr[i] - fluo_prev[prev_index]) / fluo_curr[i]
                    except:
                        print(f'Error: {prev_index}')


                x = np.concatenate((N.astype(int), X, V, T, F, AREA, PERIMETER, ASPECT, ORIENTATION, X_track_ID, ID.astype(int) - 1), axis=1)

            x_list.append(x)

            y = np.zeros((x.shape[0], 2))
            y_list.append(y)

            if False:

                vertices_list = []
                for n in trange(1, len(x)):
                    mask = (im_seg == n)
                    if np.sum(mask)>0:
                        vertices = mask_to_vertices(mask=mask, num_vertices=20)
                        uniform_points = get_uniform_points(vertices, num_points=20)
                        N = (n-1)*20 + np.arange(20, dtype=np.float32)[:, None]
                        X = uniform_points
                        empty_columns = np.zeros((X.shape[0], 2))
                        T = n_cells + (n-1) * np.ones((X.shape[0], 1))
                        vertices = np.concatenate((N.astype(int), X, empty_columns, T, N.astype(int)), axis=1)
                        vertices_list.append(vertices)
                # vertices_list = torch.stack(vertices_list)
                # vertices_list = torch.reshape(vertices_list, (-1, vertices_list.shape[2]))
                vertices = np.array(vertices_list)
                full_vertice_list.append(vertices)

                # params = torch.tensor([[1.6233, 1.0413, 1.6012, 1.5615]], dtype=torch.float32, device=device)
                # model_vertices = PDE_V(aggr_type='mean', p=torch.squeeze(params), sigma=30, bc_dpos=bc_dpos, dimension=2)
                # max_radius=50
                # min_radius=0
                # for epoch in trange(4):
                #     distance = torch.sum(bc_dpos(vertices[:, None, 1:dimension + 1] - vertices[None, :, 1:dimension + 1]) ** 2, dim=2)
                #     adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
                #     edge_index = adj_t.nonzero().t().contiguous()
                #     dataset = data.Data(x=vertices, pos=vertices[:, 1:3], edge_index=edge_index, field=[])
                #     with torch.no_grad():
                #         y = model_vertices(dataset)
                #     vertices[:,1:3] = vertices[:,1:3] + y
                #     vertices[:, 1:2] = torch.clip(vertices[:, 1:2], 0, im_dim[0])
                #     vertices[:, 2:3] = torch.clip(vertices[:, 2:3], 0, im_dim[1])

                print (f'{files[it]}')
                fig = plt.subplots(figsize=(35, 20))
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')

                ax = plt.subplot(161)
                plt.axis('off')
                plt.imshow(im_fluo)
                for n in range(vertices.shape[0]):
                    plt.plot(vertices[n,:,2], vertices[n,:,1], c='w', linewidth=1)
                    # plt.text(x[n, 2], x[n, 1], f'{x[n,0]:0.0f}', fontsize=12, color='w')
                # plt.scatter(x[:, 2], x[:, 1], s=10, c='w', alpha=0.75)
                plt.xlim([0 , im_dim[1]])
                plt.ylim([0 , im_dim[0]])
                plt.xticks([])
                plt.yticks([])

                ax = plt.subplot(162)
                plt.axis('off')
                plt.imshow(im_fluo*0)
                for n in range(vertices.shape[0]):
                    plt.scatter(vertices[n,:,2], vertices[n,:,1], c='w', s=8, alpha=0.75, edgecolors='none')
                plt.scatter(x[:, 2], x[:, 1], s=10, c='w', alpha=0.75)
                plt.xlim([0 , im_dim[1]])
                plt.ylim([0 , im_dim[0]])
                plt.xticks([])
                plt.yticks([])

                ax = plt.subplot(163)
                plt.imshow(im_fluo*0)
                plt.scatter(x[:, 2], x[:, 1], s=10, c='w', alpha=1)
                # for n in range(len(x)):
                #     plt.text(x[n, 2], x[n, 1], f'{x[n, -2]:0.0f}', fontsize=8, color='w')
                plt.xlim([0 , im_dim[1]])
                plt.ylim([0 , im_dim[0]])
                plt.xticks([])
                plt.yticks([])

                ax = plt.subplot(164)
                plt.title('velocity', fontsize=48)
                plt.scatter(x[:, 2], x[:, 1], s=20, alpha=1, c='w')
                plt.quiver(x[:, 2], x[:, 1], x[:, 4], x[:, 3], color='w', scale = 250)
                # plt.colorbar()
                plt.xlim([0 , im_dim[1]])
                plt.ylim([0 , im_dim[0]])
                plt.xticks([])
                plt.yticks([])

                ax = plt.subplot(165)
                plt.title('F', fontsize=48)
                plt.scatter(x[:, 2], x[:, 1], s=100, c=x[:, 7], alpha=1, cmap='viridis', vmin=0, vmax=0.5E6)
                # plt.colorbar()
                plt.xlim([0 , im_dim[1]])
                plt.ylim([0 , im_dim[0]])
                plt.xticks([])
                plt.yticks([])

                ax = plt.subplot(166)
                plt.title('DF/F', fontsize=48)
                plt.scatter(x[:, 2], x[:, 1], s=100, c=x[:, 6], alpha=1, cmap='viridis', vmin=-0.5, vmax=0.5)
                # plt.colorbar()
                plt.xlim([0 , im_dim[1]])
                plt.ylim([0 , im_dim[0]])
                plt.xticks([])
                plt.yticks([])

                plt.tight_layout()
                plt.xticks([])
                plt.yticks([])
                # plt.show()
                plt.savefig(f"graphs_data/{dataset_name}/Fig/{files[it]}", dpi=100)
                plt.close()

                n_cells = ID[-1] + 1

        np.savez(f'graphs_data/{dataset_name}/x_list_{run}', *x_list)
        np.savez(f'graphs_data/{dataset_name}/y_list_{run}', *y_list)

        print(f'n_cells: {n_cells}')

    elif not os.path.exists(f'graphs_data/{dataset_name}/significant_pairs_1000.npy'):
        frame = 1000
        plt.style.use('dark_background')

        x_list = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npz')
        x_list = [x_list[f'arr_{i}'] for i in range(len(x_list.files))]
        x = x_list[frame]

        channel_q = cellpose_channels[0]-1
        black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])
        im_fluo = tifffile.imread(data_folder_name + files[frame])
        im_fluo = np.array(im_fluo).astype('float32') / 256
        im = im_fluo[:, :, channel_q]

        fig = plt.figure(figsize=(20, 10))
        plt.imshow(np.flipud(im), vmin=0, vmax=5, cmap=black_to_green)
        for i in range(x.shape[0]):
            plt.text(x[i, 2], x[i, 1], f'{int(x[i, 0])}', fontsize=3, color='w')
        plt.scatter(x[:, 2], x[:, 1], s=0.2, c='w')
        plt.xlim([0, im.shape[0]])
        plt.ylim([0, im.shape[1]])
        plt.savefig(f"graphs_data/{dataset_name}/track_ID_{frame:06}.tif", dpi=500)  # 170.7)
        plt.close()


        time_series_dict, track_info_dict = reconstruct_time_series_from_xlist(x_list)
        filtered_time_series = filter_tracks_by_length(time_series_dict, min_length=250, required_frame=frame)
        neighbor_pairs, track_positions = find_average_spatial_neighbors(filtered_time_series, track_info_dict, max_radius=75, save_path= f'graphs_data/{dataset_name}/pairs_{frame}.png')
        granger_results = analyze_neighbor_pairs(neighbor_pairs, filtered_time_series, max_order=20)
        significant_pairs = statistical_testing(granger_results, filtered_time_series, n_surrogates=100)
        np.save(f'graphs_data/{dataset_name}/significant_pairs_{frame}.npy', significant_pairs)
        G = build_causality_network(significant_pairs, track_positions)
        network_scores = compute_network_scores(G)
        visualize_network_leader_follower(G, network_scores, track_positions, save_path= f'graphs_data/{dataset_name}/network_{frame}.png')

    else:
        frame = 1000
        print(f'loading data for frame {frame}')
        x_list = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npz')
        x_list = [x_list[f'arr_{i}'] for i in range(len(x_list.files))]
        time_series_dict, track_info_dict = reconstruct_time_series_from_xlist(x_list)
        filtered_time_series = filter_tracks_by_length(time_series_dict, min_length=250, required_frame=frame)
        neighbor_pairs, track_positions = find_average_spatial_neighbors(filtered_time_series, track_info_dict,max_radius=75,save_path=f'graphs_data/{dataset_name}/pairs_{frame}.png')

        fig = plt.figure(figsize=(20, 10))
        plt.plot(time_series_dict[2430][:,1])
        plt.savefig(f'graphs_data/{dataset_name}/track_2430.png', dpi=200)
        plt.close()


        significant_pairs = np.load(f'graphs_data/{dataset_name}/significant_pairs_{frame}.npy', allow_pickle=True).item()
        G = build_causality_network(significant_pairs, track_positions)
        network_scores = compute_network_scores(G)
        # visualize_network_leader_follower(G, network_scores, track_positions, save_path= f'graphs_data/{dataset_name}/network_{frame}.png')

        granger_diffs = [result['granger_diff'] for result in significant_pairs.values()]
        p_values = [result['p_value'] for result in significant_pairs.values()]

        print(f"granger diff range: {np.min(granger_diffs):.3f} - {np.max(granger_diffs):.3f}")
        print(f"p-value range: {np.min(p_values):.4f} - {np.max(p_values):.4f}")

        interesting_pairs = plot_interesting_causality_pairs(
            significant_pairs=significant_pairs,
            filtered_time_series=filtered_time_series,
            track_positions=track_positions,
            network_scores=network_scores,
            dataset_name=dataset_name,
            n_pairs=20
        )

        l = 2048
        f = 2064
        plot_combined_causality_analysis(
            leader_track_id=l,
            follower_track_id=f,
            filtered_time_series=filtered_time_series,
            track_positions=track_positions,
            significant_pairs=significant_pairs,
            save_path=f'graphs_data/{dataset_name}/causality_pair_{l}_{f}.png'
        )


def load_3Dfluo_data_with_Cellpose(config, device, visualize):


    data_folder_name = config.data_folder_name
    dataset_name = config.dataset
    data_folder_mesh_name = config.data_folder_mesh_name

    simulation_config = config.simulation
    train_config = config.training
    image_data = config.image_data

    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    dimension = simulation_config.dimension

    delta_t = simulation_config.delta_t

    bc_pos, bc_dpos = choose_boundary_values('no')

    files = os.listdir(data_folder_name)
    files = [f for f in files if f.endswith('.csv')]

    mesh_files = os.listdir(data_folder_mesh_name)
    mesh_files = [f for f in mesh_files if f.endswith('.csv')]

    n_cells = 1
    n_cells_max = 0
    run = 0
    x_list = []
    y_list = []

    for it in trange(len(files)):
        object_properties = np.array(pd.read_csv(data_folder_name + files[it], header=0))

        faces = np.array(pd.read_csv(data_folder_mesh_name + mesh_files[3*it+0], header=0))
        cells = np.array(pd.read_csv(data_folder_mesh_name + mesh_files[3*it+1], header=0))
        mesh_pos = np.array(pd.read_csv(data_folder_mesh_name + mesh_files[3*it+2], header=0))

        # 0 label
        # 1 volume
        # 2 surface area
        # 3 x
        # 4 y
        # 5 z
        # 6 elongation
        # 7 eigenvector x
        # 8 eigenvector y
        # 9 eigenvector z
        # 10 sphericity
        # 11 mean_intensity
        # 12 std_intensity
        # 13 snr

        N = np.arange(object_properties.shape[0], dtype=np.float32)[:, None]
        X = object_properties[:,3:6]
        empty_columns = np.zeros((X.shape[0], 6))
        Volume = object_properties[:,1:2]
        Surface = object_properties[:,2:3]
        Sphericity = object_properties[:,10:11]
        Fluo = object_properties[:,11:12]
        Fluo_std = object_properties[:,12:13]
        ID = n_cells + np.arange(object_properties.shape[0])[:, None]

        x = np.concatenate((N.astype(int), X, empty_columns, Volume, Surface, Sphericity, Fluo, Fluo_std, ID.astype(int) -1), axis=1)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        x_list.append(x)

        y = torch.zeros((x.shape[0], 2), dtype=torch.float32, device=device)
        y_list.append(y)

        if len(x)> n_cells_max:
            n_cells_max = len(x)

    print(f'n_cells_max: {n_cells_max}')

    torch.save(x_list, f'graphs_data/{dataset_name}/x_list_{run}.pt')
    torch.save(y_list, f'graphs_data/{dataset_name}/y_list_{run}.pt')

    # mesh_file = '/groups/wang/wanglab/GNN/240408-LVpD80-E10-IAI/SMG2-processed/masks_smooth2_mesh_vtp/240408-E14-SMG-LVpD80-E10-IAI-SMG2-combined-rcan-t049_cp_masks.vtp'
    # visualize_mesh(mesh_file)


def load_2Dgrid_data(config, device, visualize, step):


    n_particles = config.simulation.n_particles
    n_frames = config.simulation.n_frames
    dataset_name = config.dataset
    delta_t = config.simulation.delta_t
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    os.makedirs( f"./graphs_data/{dataset_name}/Fig/Dots", exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/Derivatives", exist_ok=True)
    os.makedirs( f"./graphs_data/{dataset_name}/Fig/Target", exist_ok=True)


    run = 0
    x_list = []
    edge_p_p_list = []

    data = np.load(config.data_folder_name, allow_pickle=True)
    image_width = np.max(data[0][:, 0]) - np.min(data[0][:, 0])
    image_height = np.max(data[0][:, 1]) - np.min(data[0][:, 1])

    N = np.arange(n_particles, dtype=np.float32)[:, None]
    X = np.zeros((n_particles, 2))
    V = np.zeros((n_particles, 2))
    T = np.zeros((n_particles, 1))
    H = np.zeros((n_particles, data.shape[3]-2))


    plt.style.use('dark_background')

    for it in trange(0, n_frames - 1):

        # normalization of the position
        X = data[it,:,:,0:2].copy() / image_width
        H = data[it,:,:,2:]
        X = np.reshape(X, (X.shape[0] *  X.shape[1], X.shape[2]))
        H = np.reshape(H, (H.shape[0] *  H.shape[1], H.shape[2]))

        uv_mapping = [0,2,1,3,4,7,5,8,6,9]
        H = H[:,uv_mapping]

        if it>0:
            X_prev = data[it-1,:,:,0:2].copy() / image_width
            X_prev = np.reshape(X_prev, (X_prev.shape[0] * X_prev.shape[1], X_prev.shape[2]))
            V = (X - X_prev) / delta_t

        x = torch.tensor(np.concatenate((N.astype(int), X, V, T, H), axis=1), dtype=torch.float32, device=device)

        x_list.append(x.clone().detach())

        fig = plt.subplots(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=1, c='w', alpha=0.75)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        num = f"{it:04}"
        plt.savefig(f"./graphs_data/{dataset_name}/Fig/Dots/Fig_{num}", dpi=70)
        plt.close()

        metric_list=['du(t, x, y) / dx', 'du(t, x, y) / dy', 'dv(t, x, y) / dx', 'dv(t, x, y) / dy', 'd2u(t, x, y) / dx2', 'd2u(t, x, y) / dy2', 'd2u(t, x, y) / dxdy', 'd2v(t, x, y) / dx2', 'd2v(t, x, y) / dy2', 'd2v(t, x, y) / dxdy']

        fig = plt.subplots(figsize=(25, 10))
        plt.axis('off')
        for k in range(10):
            plt.subplot(2, 5, k+1)
            plt.title(metric_list[uv_mapping[k]], fontsize=14)
            if k<4:
                plt.imshow(H[:,k].reshape((int(np.sqrt(n_particles)), int(np.sqrt(n_particles)))), cmap='viridis', vmin = -0.2, vmax=0.2)
            else:
                plt.imshow(H[:, k].reshape((int(np.sqrt(n_particles)), int(np.sqrt(n_particles)))), cmap='viridis', vmin=-0.02, vmax=0.02)
            plt.axis('off')
        plt.tight_layout()
        num = f"{it:04}"
        plt.savefig(f"./graphs_data/{dataset_name}/Fig/Derivatives/Derivative_{num}", dpi=70)
        plt.close()

    if config.graph_model.prediction == '2nd_derivative':
        y_list = []
        y_list.append(torch.zeros((n_particles,2), dtype=torch.float32, device=device))
        for it in trange(1, n_frames - 1):

            X_prev = data[it-1, :, :, 0:2].copy() / image_width
            X = data[it, :, :, 0:2].copy() / image_width
            X_next = data[it+1, :, :, 0:2].copy() / image_width

            Y = (X_next - 2 * X + X_prev) / delta_t ** 2

            Y = np.reshape(Y, (Y.shape[0] * Y.shape[1], Y.shape[2]))
            y = torch.tensor(Y, dtype=torch.float32, device=device)
            y_list.append(y.clone().detach())


            X_flat = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
            Y_flat = Y
            indices = np.arange(0, X_flat.shape[0], 10)
            X_sampled = X_flat[indices]
            Y_sampled = Y_flat[indices]

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 10))  # You can adjust the figure size
            ax.quiver(X_sampled[:, 0], X_sampled[:, 1], Y_sampled[:, 0]/5, Y_sampled[:, 1]/5,
                      angles='xy', scale_units='xy', scale=1, color='blue')

            ax.set_aspect('equal')
            ax.set_title("Acceleration Vector Field Plot")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.grid(True)

            num = f"{it:04}"
            plt.savefig(f"./graphs_data/{dataset_name}/Fig/Target/2nd_derivative_{num}", dpi=70)
            plt.close()
        y_list[0] = y_list[1]   # better than zeros

    elif config.graph_model.prediction == 'first_derivative':

        y_list = []
        for it in trange(0, n_frames - 1):  # Notice: loop until n_frames - 1

            X = data[it, :, :, 0:2].copy() / image_width
            X_next = data[it+1, :, :, 0:2].copy() / image_width

            # First derivative using forward difference
            Y = (X_next - X) / delta_t

            # fig = plt.figure(figsize=(10, 10))
            # plt.axis('off')
            # X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
            # plt.scatter(X[:, 0], X[:, 1], s=1, c='w', alpha=0.75)
            # X_next = np.reshape(X_next, (X_next.shape[0] * X_next.shape[1], X_next.shape[2]))
            # plt.scatter(X_next[:, 0], X_next[:, 1], s=1, c='r', alpha=0.75)

            Y = np.reshape(Y, (Y.shape[0] * Y.shape[1], Y.shape[2]))
            y = torch.tensor(Y, dtype=torch.float32, device=device)
            y_list.append(y.clone().detach())

            # For plotting
            X_flat = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
            Y_flat = Y
            indices = np.arange(0, X_flat.shape[0], 10)
            X_sampled = X_flat[indices]
            Y_sampled = Y_flat[indices]

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.quiver(X_sampled[:, 0], X_sampled[:, 1], Y_sampled[:, 0]*5, Y_sampled[:, 1]*5,
                      angles='xy', scale_units='xy', scale=1, color='green')

            ax.set_aspect('equal')
            ax.set_title("Velocity Vector Field Plot")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.grid(True)
            num = f"{it:04}"
            plt.savefig(f"./graphs_data/{dataset_name}/Fig/Target/first_derivative_{num}", dpi=70)
            plt.close()

    x_list = np.array(to_numpy(torch.stack(x_list)))
    y_list = np.array(to_numpy(torch.stack(y_list)))

    np.save(f'graphs_data/{dataset_name}/x_list_{run}.npy', x_list)
    np.save(f'graphs_data/{dataset_name}/y_list_{run}.npy', y_list)
    np.save(f'graphs_data/{dataset_name}/x_list_{run+1}.npy', x_list)
    np.save(f'graphs_data/{dataset_name}/y_list_{run+1}.npy', y_list)

    # torch.save(edge_p_p_list, f'graphs_data/{dataset_name}/edge_p_p_list{run}.pt')
    # torch.save(edge_p_p_list, f'graphs_data/{dataset_name}/edge_p_p_list{run+1}.pt')


def load_2Dfluo_data_on_mesh(config, device, visualize, step):


    n_particles = config.simulation.n_particles
    n_frames = config.simulation.n_frames
    dataset_name = config.dataset
    delta_t = config.simulation.delta_t
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_nodes = config.simulation.n_nodes

    output_dir = f"./graphs_data/{dataset_name}/Fig/"
    os.makedirs(output_dir, exist_ok=True)

    run = 0

    x_mesh_list = []
    y_mesh_list = []

    X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)
    # save mesh grid, edges indices and Laplacian discrete values,
    torch.save(mesh_data, f'graphs_data/{dataset_name}/mesh_data_{run}.pt')
    torch.save(mesh_data, f'graphs_data/{dataset_name}/mesh_data_{run+1}.pt')

    plt.style.use('dark_background')

    file_path = os.path.expanduser(config.data_folder_name)
    im0 = tifffile.imread(file_path)
    im0 = np.array(im0).astype('float32')

    top_freqs, top_amps = get_top_fft_modes_per_pixel(im0, dt=1.0, top_n=1)

    # Example: get top frequency at pixel (100, 150) in channel 0
    # print("Top frequencies:", top_freqs[:, 64, 64, 1])
    # print("Amplitudes:", top_amps[:, 64, 64, 0])

    top_freqs = top_freqs.squeeze()
    top_amps = top_amps.squeeze()

    top_freqs = top_freqs * (top_amps>100)

    x_mesh = torch.concatenate(
        (N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
         T1_mesh.clone().detach(), H1_mesh.clone().detach(), H1_mesh.clone().detach(), A1_mesh.clone().detach()), 1)
    x_mesh[:, 2] = 1 - x_mesh[:, 2]

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(141)
    plt.imshow(top_freqs[:,:,0],vmin=0,vmax=0.05)
    plt.title('top frequency in channel 0')
    ax = fig.add_subplot(142)
    plt.imshow(top_freqs[:,:,1],vmin=0,vmax=0.05)
    plt.title('top frequency in channel 1')
    ax = fig.add_subplot(143)
    plt.imshow(top_amps[:,:,0],vmin=0,vmax=5000)
    plt.title('amplitudes in channel 0')
    ax = fig.add_subplot(144)
    plt.imshow(top_amps[:,:,1],vmin=0,vmax=5000)
    plt.title('amplitudes in channel 1')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/../top_freqs.png", dpi=100)
    plt.close()

    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(121)
    # time_series = im0[:,128,128,0:2]
    # plt.plot(time_series[:, 0], c='r')
    # plt.plot(time_series[:, 1], c='g')
    # # plt.grid(alpha=0.25)
    # plt.title('pixel_128_128')
    # ax = fig.add_subplot(122)
    # time_series = im0[:,72,69,0:2]
    # plt.plot(time_series[:, 0], c='r')
    # plt.plot(time_series[:, 1], c='g')
    # # plt.grid(alpha=0.25)
    # plt.title('pixel_72_69')
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/../pixels.png", dpi=100)
    # plt.close()


    for it in trange(0, n_frames - 1):

        x_mesh[:,6:9] = torch.tensor(im0[it], dtype=torch.float32, device=device).reshape(-1, 3) / 256
        if it>0:
            x_mesh[:, 9:12] = torch.tensor(im0[it+1]-im0[it], dtype=torch.float32, device=device).reshape(-1, 3) / 256 / delta_t
        else:
            x_mesh[:, 9:12] = torch.zeros((n_nodes, 3), dtype=torch.float32, device=device)

        if config.graph_model.prediction == 'first_derivative':
            y_mesh = torch.tensor(im0[it+1]-im0[it], dtype=torch.float32, device=device).reshape(-1, 3) / 256 / delta_t
        elif (config.graph_model.prediction == '2nd_derivative') & (it>0):
            y_mesh = torch.tensor(im0[it+1]-2*im0[it]+im0[it-1], dtype=torch.float32, device=device).reshape(-1, 3) / 256 / delta_t**2
        else:
            y_mesh = torch.zeros((n_nodes, 3), dtype=torch.float32, device=device)

        x_mesh_list.append(x_mesh.clone().detach())
        y_mesh_list.append(y_mesh.clone().detach())

        im = to_numpy(x_mesh[:, 6:9])  # (n_nodes, 3)
        im = im.reshape((int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes)), 3))

        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow((im*255).astype('uint8'))
        num = f"{it:04}"
        plt.savefig(f"./graphs_data/{dataset_name}/Fig/Fig_{num}", dpi=100)
        plt.close()


    x_mesh_list = torch.stack(x_mesh_list)
    y_mesh_list = torch.stack(y_mesh_list)
    torch.save(x_mesh_list, f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt')
    torch.save(y_mesh_list, f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt')
    torch.save(x_mesh_list, f'graphs_data/{dataset_name}/x_mesh_list_{run+1}.pt')
    torch.save(y_mesh_list, f'graphs_data/{dataset_name}/y_mesh_list_{run+1}.pt')


def load_RGB_grid_data(config, device, visualize, step):

    n_nodes = config.simulation.n_nodes
    n_frames = config.simulation.n_frames
    dataset_name = config.dataset
    delta_t = config.simulation.delta_t
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    os.makedirs( f"./graphs_data/{dataset_name}/Fig/Dots", exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/Derivatives", exist_ok=True)
    os.makedirs( f"./graphs_data/{dataset_name}/Fig/Target", exist_ok=True)

    run = 0
    x_list = []

    im0 = tifffile.imread(config.data_folder_name)
    im0 = np.array(im0).astype('float32')

    image_width = im0.shape[1]
    image_height = im0.shape[2]

    N = np.arange(n_nodes, dtype=np.float32)[:, None]
    X = np.zeros((n_nodes, 2))
    V = np.zeros((n_nodes, 2))
    T = np.zeros((n_nodes, 1))
    H = np.zeros((n_nodes, 6))

    xs = torch.linspace(0, 1, steps=image_width)
    ys = torch.linspace(0, image_height/image_width, steps=image_height)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_nodes, 1))
    y_mesh = torch.reshape(y_mesh, (n_nodes, 1))
    pos_mesh = torch.zeros((n_nodes, 2), device=device)
    pos_mesh[0:n_nodes, 0:1] = x_mesh[0:n_nodes]
    pos_mesh[0:n_nodes, 1:2] = y_mesh[0:n_nodes]
    X = to_numpy(pos_mesh)


    plt.style.use('dark_background')

    for it in trange(0, n_frames - 1):

        H = im0[it] / 255
        H = np.reshape(H, (H.shape[0] *  H.shape[1], H.shape[2]))

        if it>0:
            H_prev = im0[it-1] /255
            H_prev = np.reshape(H_prev, (H_prev.shape[0] * H_prev.shape[1], H_prev.shape[2]))
            dH = (H - H_prev) / delta_t
            H = np.concatenate((H, dH), axis=1)
        else:
            H = np.concatenate((H, np.zeros_like(H)), axis=1)

        x = torch.tensor(np.concatenate((N.astype(int), X, V, T, H), axis=1), dtype=torch.float32, device=device)

        x_list.append(x.clone().detach())

        fig = plt.subplots(figsize=(12, 8))
        plt.axis('off')
        for k in range(3):
            plt.subplot(2, 3, k+1)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.scatter(to_numpy(x[:, 1]), 1-to_numpy(x[:, 2]), s=10, c=to_numpy(x[:, 6+k]), vmin=0, vmax=1.1)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
        if it>0:
            for k in range(3):
                plt.subplot(2, 3, k + 4)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.scatter(to_numpy(x[:, 1]), 1 - to_numpy(x[:, 2]), s=10, c=to_numpy(x[:, 9 + k]), vmin=-1, vmax=1)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
        plt.tight_layout()
        num = f"{it:04}"
        plt.savefig(f"./graphs_data/{dataset_name}/Fig/Dots/Fig_{num}", dpi=70)
        plt.close()


    if config.graph_model.prediction == '2nd_derivative':
        y_list = []
        y_list.append(torch.zeros((n_nodes,2), dtype=torch.float32, device=device))
        for it in trange(1, n_frames - 1):

            H_prev = im0[it-1] /255
            H = im0[it] /255
            H_next = im0[it+1] /255

            Y = (H_next - 2 * H + H_prev) / delta_t ** 2

            Y = np.reshape(Y, (Y.shape[0] * Y.shape[1], Y.shape[2]))
            y = torch.tensor(Y, dtype=torch.float32, device=device)
            y_list.append(y.clone().detach())

        y_list[0] = y_list[1]   # better than zeros

    elif config.graph_model.prediction == 'first_derivative':

        y_list = []
        for it in trange(0, n_frames - 1):  # Notice: loop until n_frames - 1

            H = im0[it] /255
            H_next = im0[it+1] /255

            Y = (H_next - H) / delta_t

            Y = np.reshape(Y, (Y.shape[0] * Y.shape[1], Y.shape[2]))
            y = torch.tensor(Y, dtype=torch.float32, device=device)
            y_list.append(y.clone().detach())

    x_list = np.array(to_numpy(torch.stack(x_list)))
    y_list = np.array(to_numpy(torch.stack(y_list)))

    np.save(f'graphs_data/{dataset_name}/x_list_{run}.npy', x_list)
    np.save(f'graphs_data/{dataset_name}/y_list_{run}.npy', y_list)
    np.save(f'graphs_data/{dataset_name}/x_list_{run+1}.npy', x_list)
    np.save(f'graphs_data/{dataset_name}/y_list_{run+1}.npy', y_list)

    # torch.save(edge_p_p_list, f'graphs_data/{dataset_name}/edge_p_p_list{run}.pt')
    # torch.save(edge_p_p_list, f'graphs_data/{dataset_name}/edge_p_p_list{run+1}.pt')


    n_particles = config.simulation.n_particles
    n_frames = config.simulation.n_frames
    dataset_name = config.dataset
    delta_t = config.simulation.delta_t
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_nodes = config.simulation.n_nodes

    output_dir = f"./graphs_data/{dataset_name}/Fig/"
    os.makedirs(output_dir, exist_ok=True)

    run = 0

    x_mesh_list = []
    y_mesh_list = []

    X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)
    # save mesh grid, edges indices and Laplacian discrete values,
    torch.save(mesh_data, f'graphs_data/{dataset_name}/mesh_data_{run}.pt')
    torch.save(mesh_data, f'graphs_data/{dataset_name}/mesh_data_{run+1}.pt')

    plt.style.use('dark_background')

    file_path = os.path.expanduser(config.data_folder_name)
    im0 = tifffile.imread(file_path)
    im0 = np.array(im0).astype('float32')


    top_freqs, top_amps = get_top_fft_modes_per_pixel(im0, dt=1.0, top_n=1)

    # Example: get top frequency at pixel (100, 150) in channel 0
    # print("Top frequencies:", top_freqs[:, 64, 64, 1])
    # print("Amplitudes:", top_amps[:, 64, 64, 0])

    top_freqs = top_freqs.squeeze()
    top_amps = top_amps.squeeze()

    x_mesh = torch.concatenate(
        (N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
         T1_mesh.clone().detach(), H1_mesh.clone().detach(), H1_mesh.clone().detach(), A1_mesh.clone().detach()), 1)
    x_mesh[:, 2] = 1 - x_mesh[:, 2]

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(141)
    plt.imshow(top_freqs[:,:,0],vmin=0,vmax=0.05)
    plt.title('top frequency in channel 0')
    ax = fig.add_subplot(142)
    plt.imshow(top_freqs[:,:,1],vmin=0,vmax=0.05)
    plt.title('top frequency in channel 1')
    ax = fig.add_subplot(143)
    plt.imshow(top_amps[:,:,0],vmin=0,vmax=5000)
    plt.title('amplitudes in channel 0')
    ax = fig.add_subplot(144)
    plt.imshow(top_amps[:,:,1],vmin=0,vmax=5000)
    plt.title('amplitudes in channel 1')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/../top_freqs.png", dpi=100)
    plt.close()

    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(121)
    # time_series = im0[:,128,128,0:2]
    # plt.plot(time_series[:, 0], c='r')
    # plt.plot(time_series[:, 1], c='g')
    # # plt.grid(alpha=0.25)
    # plt.title('pixel_128_128')
    # ax = fig.add_subplot(122)
    # time_series = im0[:,72,69,0:2]
    # plt.plot(time_series[:, 0], c='r')
    # plt.plot(time_series[:, 1], c='g')
    # # plt.grid(alpha=0.25)
    # plt.title('pixel_72_69')
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/../pixels.png", dpi=100)
    # plt.close()


    for it in trange(0, n_frames - 1):

        x_mesh[:,6:9] = torch.tensor(im0[it], dtype=torch.float32, device=device).reshape(-1, 3) / 256
        if it>0:
            x_mesh[:, 9:12] = torch.tensor(im0[it+1]-im0[it], dtype=torch.float32, device=device).reshape(-1, 3) / 256 / delta_t
        else:
            x_mesh[:, 9:12] = torch.zeros((n_nodes, 3), dtype=torch.float32, device=device)

        if config.graph_model.prediction == 'first_derivative':
            y_mesh = torch.tensor(im0[it+1]-im0[it], dtype=torch.float32, device=device).reshape(-1, 3) / 256 / delta_t
        elif (config.graph_model.prediction == '2nd_derivative') & (it>0):
            y_mesh = torch.tensor(im0[it+1]-2*im0[it]+im0[it-1], dtype=torch.float32, device=device).reshape(-1, 3) / 256 / delta_t**2
        else:
            y_mesh = torch.zeros((n_nodes, 3), dtype=torch.float32, device=device)

        x_mesh_list.append(x_mesh.clone().detach())
        y_mesh_list.append(y_mesh.clone().detach())

        im = to_numpy(x_mesh[:, 6:9])  # (n_nodes, 3)
        im = im.reshape((int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes)), 3))

        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow((im*255).astype('uint8'))
        num = f"{it:04}"
        plt.savefig(f"./graphs_data/{dataset_name}/Fig/Fig_{num}", dpi=100)
        plt.close()


    x_mesh_list = torch.stack(x_mesh_list)
    y_mesh_list = torch.stack(y_mesh_list)
    torch.save(x_mesh_list, f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt')
    torch.save(y_mesh_list, f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt')
    torch.save(x_mesh_list, f'graphs_data/{dataset_name}/x_mesh_list_{run+1}.pt')
    torch.save(y_mesh_list, f'graphs_data/{dataset_name}/y_mesh_list_{run+1}.pt')


def load_Goole_data(config, device=None, visualize=None, step=None, cmap=None):

    data_folder_name = config.data_folder_name
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training
    n_frames = simulation_config.n_frames
    dimension = 2

    n_particle_types = simulation_config.n_particle_types
    n_runs = train_config.n_runs
    n_particles = simulation_config.n_particles

    delta_t = simulation_config.delta_t
    bc_pos, bc_dpos = choose_boundary_values('no')

    cmap = CustomColorMap(config=config)


    # Loading Data

    with open(os.path.join(data_folder_name, "metadata.json")) as f:
        metadata = json.load(f)

    n_wall_particles = 400
    n_max_particles = 0

    for run in range(0, n_runs):
        x_list = []
        y_list = []

        gap = 0.008

        wall_pos = torch.linspace(0.1-gap, 0.9+gap, n_wall_particles//4, device=device)
        wall0 = torch.zeros(n_wall_particles//4, 2, device=device)
        wall0[:, 0] = wall_pos
        wall0[:, 1] = 0.1-gap
        wall1 = torch.zeros(n_wall_particles//4, 2, device=device)
        wall1[:, 0] = wall_pos
        wall1[:, 1] = 0.9+gap
        wall2 = torch.zeros(n_wall_particles//4, 2, device=device)
        wall2[:, 1] = wall_pos
        wall2[:, 0] = 0.1-gap
        wall3 = torch.zeros(n_wall_particles//4, 2, device=device)
        wall3[:, 1] = wall_pos
        wall3[:, 0] = 0.9+gap
        # noise_wall = torch.randn((n_wall_particles//4, dimension), device=device) * 0.001
        # wall0 = wall0 + noise_wall
        # wall1 = wall1 + noise_wall
        # wall2 = wall2 + noise_wall
        # wall3 = wall3 + noise_wall

        position = np.load(data_folder_name + 'position.' + str(run) + '.npy', allow_pickle=True)
        # Swap the columns
        position[:, :, [0, 1]] = position[:, :, [1, 0]]
        position = torch.tensor(position, dtype=torch.float32, device=device)
        type = np.load(data_folder_name + 'particle_type.' + str(run) + '.npy', allow_pickle=True)
        print(f'types: {np.unique(type)}')
        type = torch.tensor(type, dtype=torch.float32, device=device)
        if 'multimaterial' in config.dataset:
            type = type - 4     # type = 5,6,7
        elif 'falling_water_ramp_wall' in config.dataset:
            type = (type-3)/2   # type = 3,5
        type = torch.cat((torch.zeros(n_wall_particles, device=device), type), 0)
        type = type[:, None]

        for frame in trange(1,position.shape[0]-2):

            pos_prev = position[frame-1].squeeze()
            pos_next = position[frame+1].squeeze()
            pos = position[frame].squeeze()

            real_n_particles = pos.shape[0]
            if real_n_particles > n_max_particles:
                n_max_particles = real_n_particles
            n_particles = n_wall_particles + pos.shape[0]

            y = torch.zeros((n_particles, dimension), device=device)
            dpos = torch.zeros((n_particles, dimension), device=device)
            dpos[n_wall_particles:] = (pos - pos_prev) / delta_t
            dpos_next = (pos_next - pos) / delta_t

            pos = torch.cat((wall0, wall1, wall2, wall3, pos), dim=0)

            particle_id = torch.arange(n_particles, device=device)
            particle_id = particle_id[:, None]

            x = torch.concatenate((particle_id.clone().detach(), pos.clone().detach(), dpos.clone().detach(), type.clone().detach()), 1)
            x_list.append(x)

            if config.graph_model.prediction == '2nd_derivative':
                y[n_wall_particles:] = (dpos_next - dpos[n_wall_particles:]) / delta_t
            else:
                y[n_wall_particles:] = dpos_next

            y_list.append(y)

            # fig = plt.figure(figsize=(12, 12))
            # plt.scatter(to_numpy(pos_prev[:, 0]), to_numpy(pos_prev[:, 1]), s=100, c='b')
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), s=100, c='g')
            # plt.scatter(to_numpy(pos_next[:, 0]), to_numpy(pos_next[:, 1]), s=100, c='r')

            if (run <21) & (frame%20==0):
                plt.style.use('dark_background')
                fig = plt.figure(figsize=(19, 10))
                ax = fig.add_subplot(121)
                index_particles = get_index_particles(x, n_particle_types, dimension)
                for n in range(n_particle_types):
                    plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]), s=10, color=cmap.color(n))
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xticks([])
                plt.yticks([])
                ax = fig.add_subplot(122)
                plt.scatter(x[:, 2].detach().cpu().numpy(),
                            x[:, 1].detach().cpu().numpy(), s=1, c='w', vmin=0, vmax=1)
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                num = f"{frame-1:06}"
                plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=80)  # 170.7)
                plt.close()

        # torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
        # torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')

        x_list = np.array(to_numpy(torch.stack(x_list)))
        y_list = np.array(to_numpy(torch.stack(y_list)))
        np.save(f'graphs_data/{dataset_name}/x_list_{run}.npy', x_list)
        np.save(f'graphs_data/{dataset_name}/y_list_{run}.npy', y_list)

    print (f'n_max_particles: {n_max_particles}')

    # load corresponding data for this time slice
    # for idx in trange(4000):
    #     window = windows[idx]
    #     size = window["size"]
    #     particle_type = particle_type[window["type"]: window["type"] + size]
    #     # particle_type = torch.from_numpy(particle_type)
    #     position_seq = position[window["pos"]: window["pos"] + window_length * size * dim]
    #     position_seq.resize(window_length, size, dim)
    #     position_seq = position_seq.transpose(1, 0, 2)
    #     target_position = position_seq[:, -1]
    #     position_seq = position_seq[:, :-1]
    #     # target_position = torch.from_numpy(target_position)
    #     position_seq = torch.from_numpy(position_seq)


def process_trace(trace):
    '''
    Returns activity traces with normalization based on mean and standard devation.
    '''
    worm_trace = (trace - np.nanmean(trace))/np.nanstd(trace)
    return worm_trace


def process_activity(activity_worms):
    '''
    Returns a list of matrices corresponding to the data missing in the activity columns of the activity_worms dataframes and
    a matrix of the activity with NaNs replaced by 0's
    '''
    missing_data, activity_data = [],[]
    for id in range(len(activity_worms)):
        worm = (activity_worms[id] - activity_worms[id].mean())/activity_worms[id].std()
        act_matrix = worm
        missing_act = np.zeros(act_matrix.shape)
        missing_act[np.isnan(act_matrix)] = 1
        act_matrix[np.isnan(act_matrix)] = 0
        missing_data.append(missing_act)
        activity_data.append(act_matrix)
    return activity_data, missing_data


def load_agent_data(
        data_directory,
        *,
        device='cuda:0'
):
    """
    Load simulated agent data and convert it to a time series.

    :param data_directory: The directory containing the agent data.
    :param device: The PyTorch device to allocate the tensors on.
    :return: A tuple consisting of:
     * A :py:class:`TimeSeries` object containing the loaded data for each time point.
     * A 2D grid of the signal that the agents are responding to.
    """

    # Check how many files (each a timestep) there are
    print(f"Loading data from '{data_directory}'...")
    files = os.listdir(data_directory)
    file_name_pattern = re.compile(r'particles\d+.txt')
    n_time_points = sum(1 for f in files if file_name_pattern.match(f))

    # Load the data from text (csv) files and convert everything to to Data objects (all fields are float32)
    dtype = {
        "x": np.float32,
        "y": np.float32,
        "internal": np.float32,
        "orientation": np.float32,
        "reversal_timer": np.int64,
        "state": np.int64
    }

    data = []
    time = torch.arange(1, n_time_points + 1, device=device)
    for i in trange(n_time_points):
        file_path = os.path.join(data_directory, f"particles{i + 1}.txt")
        time_point = pd.read_csv(file_path, sep=",", names=list(dtype.keys()), dtype=dtype)
        position = torch.stack([torch.tensor(time_point["x"].to_numpy(), device=device),
                                torch.tensor(time_point["y"].to_numpy(), device=device)], dim=1)
        data.append(Data(
            time=time[i],
            pos=position,
            internal=torch.tensor(time_point["internal"].to_numpy(), device=device),
            orientation=torch.tensor(time_point["orientation"].to_numpy(), device=device),
            reversal_timer=torch.tensor(time_point["reversal_timer"].to_numpy(), dtype=torch.float32, device=device),
            state=torch.tensor(time_point["state"].to_numpy(), dtype=torch.float32, device=device),
        ))

    # Compute the velocity as the derivative of the position and add it to the time series
    time_series = TimeSeries(time, data)
    velocity = time_series.compute_derivative('pos')
    for i, data in enumerate(time_series):
        data.velocity = velocity[i]

    # Load the signal
    signal = np.loadtxt(os.path.join(data_directory, "signal.txt"))
    signal = torch.tensor(signal, device=device)

    return time_series, signal


def ensure_local_path_exists(path):
    """
    Ensure that the local path exists. If it doesn't, create the directory structure.

    :param path: The path to be checked and created if necessary.
    :return: The absolute path of the created directory.
    """

    os.makedirs(path, exist_ok=True)
    return os.path.join(os.getcwd(), path)

@dataclass
class CsvDescriptor:
    """A class to describe the location of data in a dataset as a column of a CSV file."""
    filename: str
    column_name: str
    type: np.dtype
    unit: Unit


def load_csv_from_descriptors(
        column_descriptors: Dict[str, CsvDescriptor],
        **kwargs
) -> pd.DataFrame:
    """
    Load data from a CSV file based on a set of column descriptors.

    :param column_descriptors: A dictionary mapping field names to CsvDescriptors.
    :param kwargs: Additional keyword arguments to pass to pd.read_csv.
    :return: A pandas DataFrame containing the loaded data.
    """
    different_files = set(descriptor.filename for descriptor in column_descriptors.values())
    columns = []

    for file in different_files:
        dtypes = {descriptor.column_name: descriptor.type for descriptor in column_descriptors.values()
                  if descriptor.filename == file}
        print(f"Loading data from '{file}':")
        for column_name, dtype in dtypes.items():
            print(f"  - column {column_name} as {dtype}")
        columns.append(pd.read_csv(file, dtype=dtypes, usecols=list(dtypes.keys()), **kwargs))

    data = pd.concat(columns, axis='columns')
    data.rename(columns={descriptor.column_name: name for name, descriptor in column_descriptors.items()}, inplace=True)

    return data


def load_wanglab_salivary_gland(
        file_path: str,
        *,
        device: str = 'cuda:0'
) -> Tuple[TimeSeries, torch.Tensor]:
    """
    Load the Wanglab salivary gland data from a CSV file and convert it to a pytorch_geometric Data object.

    :param file_path: The path to the CSV file.
    :param device: The PyTorch device to allocate the tensors on.
    :return: A :py:class:`TimeSeries` object containing the loaded data for each time point.
    """

    # Load the data of interest from the CSV file
    column_descriptors = {
        'x': CsvDescriptor(filename=file_path, column_name="Position X", type=np.float32, unit=u.micrometer),
        'y': CsvDescriptor(filename=file_path, column_name="Position Y", type=np.float32, unit=u.micrometer),
        'z': CsvDescriptor(filename=file_path, column_name="Position Z", type=np.float32, unit=u.micrometer),
        't': CsvDescriptor(filename=file_path, column_name="Time", type=np.float32, unit=u.day),
        'track_id': CsvDescriptor(filename=file_path, column_name="TrackID", type=np.int64,
                                  unit=u.dimensionless_unscaled),
    }
    raw_data = load_csv_from_descriptors(column_descriptors, skiprows=3)
    raw_tensors = {name: torch.tensor(raw_data[name].to_numpy(), device=device) for name in column_descriptors.keys()}

    # Split into individual data objects for each time point
    t = raw_tensors['t']
    time_jumps = torch.where(torch.diff(t).ne(0))[0] + 1
    time = torch.unique_consecutive(t)
    x = torch.tensor_split(raw_tensors['x'], time_jumps.tolist())
    y = torch.tensor_split(raw_tensors['y'], time_jumps.tolist())
    z = torch.tensor_split(raw_tensors['z'], time_jumps.tolist())
    global_ids, id_indices = torch.unique(raw_tensors['track_id'], return_inverse=True)
    id = torch.tensor_split(id_indices, time_jumps.tolist())

    # Combine the data into a TimeSeries object
    n_time_steps = len(time)
    data = []
    for i in range(n_time_steps):
        data.append(Data(
            time=time[i],
            pos=torch.stack([x[i], y[i], z[i]], dim=1),
            track_id=id[i],
        ))

    time_series = TimeSeries(time, data)

    # Compute the velocity as the derivative of the position and add it to the time series
    velocity, _ = time_series.compute_derivative('pos', id_name='track_id')
    for i in range(n_time_steps):
        data[i].velocity = velocity[i]

    return time_series, global_ids
