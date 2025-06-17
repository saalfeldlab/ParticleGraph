from ParticleGraph.generators.utils import *
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import astropy.units as u
import h5py
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
# from cellpose import models
from ParticleGraph.generators.cell_utils import *
from cellpose import models, core, utils, io, models, metrics, denoise
import scipy.io as sio
import seaborn as sns
from torch_geometric.utils import dense_to_sparse
import pickle
import json
import scipy.io
import h5py
import re
from skimage.draw import disk
from skimage.transform import resize
from skimage import filters, feature
import pandas as pd
import scipy.io
from matplotlib.colors import LinearSegmentedColormap

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

            if id == 339:
                rr_valid_339 = rr_valid
                cc_valid_339 = cc_valid
                pos_x_339 = pos_x
                pos_y_339 = pos_y
                fluo_sum_radius_339 = np.sum(fluorescence_image[rr_valid_339, cc_valid_339])
                # print(len(object_properties), fluo_sum_radius_339)

            if id == 104:
                rr_valid_104 = rr_valid
                cc_valid_104 = cc_valid
                pos_x_104 = pos_x
                pos_y_104 = pos_y
                fluo_sum_radius_334 = np.sum(fluorescence_image[rr_valid_104, cc_valid_104])
                # print(len(object_properties), fluo_sum_radius_334)

            object_properties.append((id, pos_x, pos_y, area, perimeter, aspect_ratio, orientation, fluo_sum_radius, fluo_sum_segmentation))

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

    if 'models' in cellpose_model_path:
        model_cellpose = models.CellposeModel(gpu=True, pretrained_model=cellpose_model_path)
    elif 'cyto3' in cellpose_model_path:
        model_cellpose = models.CellposeModel(gpu=True, model_type='cyto3', nchan=2)
    elif 'cyto2' in cellpose_model_path:
        model_cellpose = models.CellposeModel(gpu=True, model_type='cyto2', nchan=2)
    elif 'cyto2_cp3' in cellpose_model_path:
        model_cellpose = models.CellposeModel(gpu=True, model_type='cyto2_cp3', nchan=2)

    if cellpose_denoise_model == 'cyto3':
        model_denoise = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")


    # step 1
    if not os.path.exists(f"{data_folder_name}/TRK/_spots.csv"):

        print('generate segmentation masks with Cellpose ...')
        for it in trange(0, len(files)):
            im = tifffile.imread(data_folder_name + files[it])
            im = np.array(im).astype('float32')

            if cellpose_denoise_model != '':
                for i in cellpose_channels:
                    masks, flows, styles, imgs_dn = model_denoise.eval(im[:,:,i-1], diameter=cellpose_diameter, channels=[0,0])
                    im[:,:,i-1:i] = imgs_dn.copy()
                tifffile.imsave(data_folder_name + 'DN/' + files[it], im[:,:,0])

            masks, flows, styles = model_cellpose.eval(im[:,:,:], diameter=cellpose_diameter, flow_threshold=0.0, invert=False, normalize=True, channels=cellpose_channels+1)
            # fig = plt.figure(figsize=(12, 12))
            # plt.imshow(masks)

            tifffile.imsave(data_folder_name + 'SEG/' + files[it], masks)

            object_properties = extract_object_properties(masks, im[:, :, cellpose_channels[0]], radius=cellpose_diameter)
            image = im[:,:,0] * 0
            for i in range(len(object_properties)):
                cell_id = object_properties[i][0]
                pos_x = object_properties[i][1]
                pos_y = object_properties[i][2]
                rr, cc = disk((pos_x, pos_y), 8, shape=image.shape)
                image[rr, cc] = 255  # White blob
            image = resize(image,(image.shape[0] // trackmate_size_ratio, image.shape[1] // trackmate_size_ratio), anti_aliasing=True)
            tifffile.imsave(f'{data_folder_name}/TRK/{it:06}.tif', image.astype('uint8'))

            # setp 2 trackmate
            # trackmate settings
            # diameter 5
            # distance closing gap 6 6 3
            # min track length 20

    else:

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

        time_series_list = []
        for i in range(n_cells-1):
            time_series_list.append(list([]))

        run = 0
        x_list = []
        y_list = []

        for it in trange(0, len(files)-2):

            im_fluo = tifffile.imread(data_folder_name + files[it])
            im_fluo = np.array(im_fluo).astype('float32') / 256
            im_seg = np.flipud(np.array(tifffile.imread(data_folder_name + 'SEG/' + files[it])))
            im_seg = np.array(im_seg)
            object_properties = extract_object_properties(im_seg, im_fluo[:,:,cellpose_channels[1]], radius=measure_diameter, offset_channel = offset_channel)
            object_properties = np.array(object_properties, dtype=float)

            N = np.arange(object_properties.shape[0], dtype=np.float32)[:, None]
            X = object_properties[:,1:3]
            V = np.zeros((X.shape[0], 2))
            T = np.zeros((X.shape[0], 1))
            F = np.zeros((X.shape[0], 3))
            F [:, 0:1] = object_properties[:,7:8]
            AREA = object_properties[:,3:4]
            PERIMETER = object_properties[:,4:5]
            ASPECT = object_properties[:,5:6]
            ORIENTATION = object_properties[:,6:7]
            ID = n_cells + np.arange(object_properties.shape[0])[:, None]

            pos = np.argwhere(trackmate['frame'] == it // trackmate_frame_step)

            X_trackmate = np.concatenate((trackmate['y'][pos], trackmate['x'][pos]), axis=1)
            trackID = trackmate['track_ID'][pos]

            # Calculate distances between each point in X and each point in X_trackmate
            distances = np.linalg.norm(X[:, None, :] - X_trackmate[None, :, :], axis=2)
            # Find the index of the closest point in X_trackmate for each point in X
            closest_indices = np.argmin(distances, axis=0)
            # Map the track_ID from trackmate to X using the closest indices

            X_track_ID = trackmate['track_ID'][pos]

            F[:, 1:2] = F[:, 0:1] / np.median(F[closest_indices,0:1])

            x = np.concatenate((X_track_ID, X[closest_indices], V[closest_indices], T[closest_indices], F[closest_indices]), axis=1)
            x [:,]

            # pa = np.argwhere(X_track_ID==489)[0,0]
            # pb = np.argwhere(X_track_ID==494)[0,0]
            # print(f'cell 494 is {pb}    cell 489 is {pa}')
            # print(x[pb, 6:7], x[pa, 6:7])


            for i in range(x.shape[0]):
                time_series_list[int(x[i, 0])].append([it,x[i, 6],x[i, 7]])

            if True:

                shifted_im2 = np.roll(im_fluo[:, :, cellpose_channels[0]], shift= -offset_channel[0], axis=0)
                im3 = im_fluo[:,:,cellpose_channels[1]]/5 + feature.canny(shifted_im2*100,sigma=10)
                fig = plt.figure(figsize=(12, 12))
                plt.imshow(np.flipud(im3), vmin=0, vmax=0.25)
                for i in range(X_trackmate.shape[0]):
                    plt.text(X_trackmate[i,1], X_trackmate[i,0], f'{int(X_track_ID[i])}', fontsize=8, color='w')
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
        np.savez(f'graphs_data/{dataset_name}/time_series_list_{run}', *time_series_list)

        # torch.save(x_list, f'graphs_data/{dataset_name}/x_list_{run}.pt')
        # torch.save(y_list, f'graphs_data/{dataset_name}/y_list_{run}.pt')
        # torch.save(full_vertice_list, f'graphs_data/{dataset_name}/full_vertice_list{run}.pt')

        print(f'n_cells: {n_cells}')


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


def load_worm_Kato_data(config, device=None, visualize=None, step=None, cmap=None):

    # data from https://osf.io/2395t/    Global Brain Dynamics Embed the Motor Command Sequence of Caenorhabditis elegans


    data_folder_name = config.data_folder_name
    dataset_name = config.dataset
    connectome_folder_name = config.connectome_folder_name

    simulation_config = config.simulation
    train_config = config.training
    n_frames = simulation_config.n_frames

    n_runs = train_config.n_runs
    n_particles = simulation_config.n_particles

    delta_t = simulation_config.delta_t
    bc_pos, bc_dpos = choose_boundary_values('no')
    cmap = CustomColorMap(config=config)

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)

    with h5py.File(data_folder_name, 'r') as f:
        # View top-level structure
        print(list(f.keys()))
        wt_data = f['WT_NoStim']
        print(list(wt_data.keys()))

    with h5py.File(data_folder_name, 'r') as f:
        # Access the deltaFOverF dataset
        delta_f_over_f = f['WT_NoStim']['deltaFOverF']

        # Iterate through the object references
        for i, ref in enumerate(delta_f_over_f):
            # Dereference the object reference
            dereferenced_data = f[ref[0]]
            # Check if the dereferenced object is a dataset
            if isinstance(dereferenced_data, h5py.Dataset):
                print(f"Dereferenced Dataset {i}: {dereferenced_data[()].shape}")
            elif isinstance(dereferenced_data, h5py.Group):
                print(f"Dereferenced Group {i}: Contains keys {list(dereferenced_data.keys())}")

        wt_data = f['WT_NoStim']['NeuronNames']
        first_ref_data = f[wt_data[1][0]]  # This should point to another object that stores data
        neuron_references = first_ref_data[:]  # Read all the references from this dataset
        for i, neuron_ref in enumerate(neuron_references):
            # Dereference each object reference to get the actual neuron name
            dereferenced_neuron = f[neuron_ref[0]]
            neuron_name = dereferenced_neuron[()]  # Get the actual neuron name

            # Convert the neuron name from numbers (if they are ASCII values) to characters
            decoded_name = ''.join(chr(int(num[0])) for num in neuron_name)

            print(f"Neuron {i + 1} name:", decoded_name)


def plot_worm_adjacency_matrix(weights, all_neuron_list, title, output_path):
    """
    Plots the adjacency matrix and weights for the given chemical and electrical weights.

    Parameters:
        chem_weights (torch.Tensor): Chemical weights matrix.
        eassym_weights (torch.Tensor): Electrical weights matrix.
        all_neuron_list (list): List of neuron names.
        output_path (str): Path to save the output plot.
    """


    fig = plt.figure(figsize=(30, 15))

    # Plot adjacency matrix
    ax = fig.add_subplot(121)
    sns.heatmap(weights > 0, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(all_neuron_list)))
    ax.set_xticklabels(all_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(all_neuron_list)))
    ax.set_yticklabels(all_neuron_list, fontsize=6)
    plt.title(title, fontsize=18)
    plt.xlabel('presynaptic', fontsize=18)
    plt.ylabel('postsynaptic', fontsize=18)

    # Plot weights
    ax = fig.add_subplot(122)
    sns.heatmap(weights, center=0, square=True, cmap='bwr', vmin=0, vmax=30, cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(all_neuron_list)))
    ax.set_xticklabels(all_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(all_neuron_list)))
    ax.set_yticklabels(all_neuron_list, fontsize=6)
    plt.title('weights', fontsize=18)
    plt.xlabel('presynaptic', fontsize=18)
    plt.ylabel('postsynaptic', fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()


def load_wormvae_data(config, device=None, visualize=None, step=None, cmap=None):

    data_folder_name = config.data_folder_name
    dataset_name = config.dataset
    connectome_folder_name = config.connectome_folder_name

    simulation_config = config.simulation
    train_config = config.training
    n_frames = simulation_config.n_frames

    n_runs = train_config.n_runs
    n_particles = simulation_config.n_particles
    baseline = simulation_config.baseline_value

    delta_t = simulation_config.delta_t
    bc_pos, bc_dpos = choose_boundary_values('no')
    cmap = CustomColorMap(config=config)

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/AVFL/', exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/I1L/', exist_ok=True)

    # Loading Data from class Worm_Data_Loader(Dataset) in https://github.com/TuragaLab/wormvae

    with open(connectome_folder_name+"all_neuron_names.json", "r") as f:
        all_neuron_list = json.load(f)
    all_neuron_list = [str(neuron) for neuron in all_neuron_list]
    with open(f"graphs_data/{dataset_name}/all_neuron_list.json", "w") as f:
        json.dump(all_neuron_list, f)
    with open(connectome_folder_name+"activity_neuron_list.pkl", "rb") as f:
        activity_neuron_list = pickle.load(f)
    activity_neuron_list = [str(neuron) for neuron in activity_neuron_list]
    with open(f"graphs_data/{dataset_name}/activity_neuron_list.json", "w") as f:
        json.dump(activity_neuron_list, f)

    # Find neurons in all_neuron_list but not in activity_neuron_list
    not_recorded_neurons = list(set(all_neuron_list) - set(activity_neuron_list))

    print(f"neurons with activity data: {len(activity_neuron_list)}")
    print(f"Neurons without activity data: {len(not_recorded_neurons)}")
    print (f"total {len(all_neuron_list)} {len(not_recorded_neurons) + len(activity_neuron_list)}")
    # all_neuron_list = [*activity_neuron_list, *not_recorded_neurons]

    print ('load data from Worm_Data_Loader ...')
    odor_channels = 3
    step = 0.25
    n_runs = 21
    n_neurons = 189
    T = 960
    N_length = 109
    T_start = 0
    activity_datasets = np.zeros((n_runs, n_neurons, T))
    odor_datasets = np.zeros((n_runs, odor_channels, T))

    print ('load traces ...')

    trace_variable = sio.loadmat(data_folder_name)
    trace_arr = trace_variable['traces']
    is_L = trace_variable['is_L']
    stimulate_seconds = trace_variable['stim_times']
    stims = trace_variable['stims']

    mean_value = np.nanmean(activity_datasets)
    min_value = np.nanmin(activity_datasets)
    max_value = np.nanmax(activity_datasets)
    std_value = np.nanstd(activity_datasets)
    print(f'mean: {mean_value}, min: {min_value}, max: {max_value}, std: {std_value}')

    for idata in trange(n_runs):
        ineuron = 0
        for ifile in range(N_length):
            if trace_arr[ifile][0].shape[1] == 42:
                data = trace_arr[ifile][0][0][idata]
                if data.shape[0] < 1:
                    activity_datasets[idata][ineuron][:] = np.nan
                else:
                    activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron += 1
                data = trace_arr[ifile][0][0][idata + 21]
                if data.shape[0] < 1:
                    activity_datasets[idata][ineuron][:] = np.nan
                else:
                    activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron += 1
            else:
                data = trace_arr[ifile][0][0][idata]
                if data.shape[0] < 1:
                    activity_datasets[idata][ineuron][:] = np.nan
                else:
                    activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron += 1

    activity_datasets = activity_datasets[:, :, T_start:]
    neuron_OI = get_neuron_index('AVFL', activity_neuron_list)
    for data_OI in range(n_runs):
        activity = activity_datasets[data_OI, neuron_OI, :]
        fig = plt.figure(figsize=(20, 2))
        activity = activity_datasets[data_OI, neuron_OI, :]
        plt.plot(activity, linewidth=1, c='b')
        activity = activity_datasets[data_OI, neuron_OI+1, :]
        plt.plot(activity, linewidth=1, c='r')
        plt.title(f'{data_OI} {neuron_OI} {activity_neuron_list[neuron_OI]}', fontsize=18)
        plt.savefig(f"graphs_data/{dataset_name}/Fig/AVFL/Fig_{data_OI:03d}_{neuron_OI:03d}.tif", dpi=80)
        plt.close()
    neuron_OI = get_neuron_index('I1L', activity_neuron_list)
    for data_OI in range(n_runs):
        activity = activity_datasets[data_OI, neuron_OI, :]
        fig = plt.figure(figsize=(20, 2))
        activity = activity_datasets[data_OI, neuron_OI, :]
        plt.plot(activity, linewidth=1, c='b')
        activity = activity_datasets[data_OI, neuron_OI+1, :]
        plt.plot(activity, linewidth=1, c='r')
        plt.title(f'{data_OI} {neuron_OI} {activity_neuron_list[neuron_OI]}', fontsize=18)
        plt.savefig(f"graphs_data/{dataset_name}/Fig/I1L/Fig_{data_OI:03d}_{neuron_OI:03d}.tif", dpi=80)
        plt.close()

    # add baseline
    activity_worm = activity_datasets + baseline
    # activity_with_zeros, missing_matrix = process_activity(activity_worm)
    # activity_worm = process_trace(activity_worm)

    time = np.arange(start=0, stop=T * step, step=step)
    odor_list = ['butanone', 'pentanedione', 'NaCL']
    for idata in range(n_runs):
        for it_stimu in range(stimulate_seconds.shape[0]):
            tim1_ind = time > stimulate_seconds[it_stimu][0]
            tim2_ind = time < stimulate_seconds[it_stimu][1]
            odor_on = np.multiply(tim1_ind.astype(int), tim2_ind.astype(int))
            stim_odor = stims[idata][it_stimu] - 1
            odor_datasets[idata][stim_odor][:] = odor_on

    odor_worms = odor_datasets[:, :, T_start:]

    os.makedirs(f"graphs_data/{dataset_name}/Fig/Fig/", exist_ok=True)
    os.makedirs(f"graphs_data/{dataset_name}/Fig/Kinograph/", exist_ok=True)

    print ('load connectome ...')

    chem_weights = torch.load(connectome_folder_name + 'chem_weights.pt')
    eassym_weights = torch.load(connectome_folder_name + 'eassym_weights.pt')
    chem_sparsity = torch.load(connectome_folder_name + 'chem_sparsity.pt')
    esym_sparsity = torch.load(connectome_folder_name + 'esym_sparsity.pt')
    map_Turuga_matrix = chem_weights+eassym_weights
    map_Turuga_matrix = map_Turuga_matrix.to(device=device)
    plot_worm_adjacency_matrix(to_numpy(map_Turuga_matrix), all_neuron_list, 'adjacency matrix Turuga 2022', f"graphs_data/{dataset_name}/full_Turuga_adjacency_matrix.png")

    print('load connectomes from other data ...')

    # Comparison with data from https://wormwiring.org/pages/adjacency.html
    # Cook 2019 Whole-animal connectomes of both Caenorhabditis

    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Cook_2019/SI_5_corrected_July_2020_bis.xlsx'
    sheet_name = 'male chemical'
    Cook_neuron_chem_names = pd.read_excel(file_path, sheet_name=sheet_name, usecols='C', skiprows=3, nrows=382, header=None)
    Cook_neuron_chem_names = [str(label) for label in Cook_neuron_chem_names.squeeze()]
    Cook_matrix_chem = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=3, nrows=382, usecols='D:NU', header=None)
    Cook_matrix_chem = np.array(Cook_matrix_chem)
    Cook_matrix_chem = np.nan_to_num(Cook_matrix_chem, nan=0.0)
    Cook_matrix_chem = torch.tensor(Cook_matrix_chem, dtype=torch.float32, device=device).t()
    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Cook_2019/SI_5_corrected_July_2020_bis.xlsx'
    sheet_name = 'male gap jn symmetric'
    Cook_neuron_elec_names = pd.read_excel(file_path, sheet_name=sheet_name, usecols='C', skiprows=3, nrows=586, header=None)
    Cook_neuron_elec_names = [str(label) for label in Cook_neuron_elec_names.squeeze()]
    Cook_matrix_elec = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=3, nrows=586, usecols='D:VQ', header=None)
    Cook_matrix_elec = np.array(Cook_matrix_elec)
    Cook_matrix_elec = np.nan_to_num(Cook_matrix_elec, nan=0.0)
    Cook_matrix_elec = torch.tensor(Cook_matrix_elec, dtype=torch.float32, device=device).t()
    map_Cook_matrix_chem , index = map_matrix(all_neuron_list, Cook_neuron_chem_names, Cook_matrix_chem)
    map_Cook_matrix_elec , index = map_matrix(all_neuron_list, Cook_neuron_elec_names, Cook_matrix_elec)
    map_Cook_matrix = map_Cook_matrix_chem + map_Cook_matrix_elec
    plot_worm_adjacency_matrix(to_numpy(map_Cook_matrix), all_neuron_list, 'adjacency matrix Cook 2019', f"graphs_data/{dataset_name}/full_Cook_adjacency_matrix.png")

    # Comparison with data from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020095
    data_Kaiser = scipy.io.loadmat('/groups/saalfeld/home/allierc/signaling/Celegans/Kaiser_2006/celegans277.mat')
    positions = data_Kaiser['celegans277positions']
    labels_raw = data_Kaiser['celegans277labels']
    Kaiser_neuron_names = [str(label[0]) for label in labels_raw.squeeze()]
    Kaiser_matrix = np.array(data_Kaiser['celegans277matrix'])
    Kaiser_matrix = torch.tensor(Kaiser_matrix, dtype=torch.float32, device=device)
    map_Kaiser_matrix , index = map_matrix(all_neuron_list, Kaiser_neuron_names, Kaiser_matrix)
    plot_worm_adjacency_matrix(to_numpy(map_Kaiser_matrix), all_neuron_list, 'adjacency matrix Kaiser 2006', f"graphs_data/{dataset_name}/full_Kaiser_adjacency_matrix.png")

    # Comparison with data from https://github.com/openworm/VarshneyEtAl2011
    # Structural Properties of the <i>Caenorhabditis elegans</i> Neuronal Network
    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Varshney_2011/ConnOrdered_040903.mat'
    mat_data = scipy.io.loadmat(file_path)
    chemical_connectome = mat_data['A_init_t_ordered']
    electrical_connectome = mat_data['Ag_t_ordered']
    neuron_names_raw = mat_data['Neuron_ordered']
    Varshney_matrix = np.array((chemical_connectome+electrical_connectome).todense())
    Varshney_matrix = torch.tensor(Varshney_matrix, dtype=torch.float32, device=device).t()
    Varshney_neuron_names = [str(cell[0][0]) for cell in neuron_names_raw]
    map_Varshney_matrix , index = map_matrix(all_neuron_list, Varshney_neuron_names, Varshney_matrix)
    plot_worm_adjacency_matrix(to_numpy(map_Varshney_matrix), all_neuron_list, 'adjacency matrix Varshney 2011', f"graphs_data/{dataset_name}/full_Varshney_adjacency_matrix.png")

    # Comparison with data from 'Connectomes across development reveal principles of brain maturation'
    # https://www.nature.com/articles/s41586-021-03778-4
    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Zhen_2021/41586_2021_3778_MOESM4_ESM.xlsx'
    sheet_name = 'Dataset7'
    Zhen_neuron_names = pd.read_excel(file_path, sheet_name=sheet_name, usecols='C', skiprows=4, nrows=224, header=None)
    Zhen_neuron_names = Zhen_neuron_names.squeeze()  # convert to Series for convenience
    Zhen_matrix = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4, nrows=224, usecols='D:GA', header=None)
    # Zhen_matrix = Zhen_matrix.T
    Zhen_matrix_7 = torch.tensor(np.array(Zhen_matrix), dtype=torch.float32, device=device)
    sheet_name = 'Dataset8'
    Zhen_matrix = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4, nrows=224, usecols='D:GA', header=None)
    # Zhen_matrix = Zhen_matrix.T
    Zhen_matrix_8 = torch.tensor(np.array(Zhen_matrix), dtype=torch.float32, device=device)
    Zhen_matrix = Zhen_matrix_7 + Zhen_matrix_8  # combine both datasets
    map_Zhen_matrix , index = map_matrix(all_neuron_list, Zhen_neuron_names, Zhen_matrix_7)
    plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix), all_neuron_list, 'adjacency matrix Mei Zhen 2021 (7)', f"graphs_data/{dataset_name}/full_Zhen_adjacency_matrix_7.png")
    map_Zhen_matrix , index = map_matrix(all_neuron_list, Zhen_neuron_names, Zhen_matrix_8)
    plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix), all_neuron_list, 'adjacency matrix Mei Zhen 2021 (8)', f"graphs_data/{dataset_name}/full_Zhen_adjacency_matrix_8.png")

    print('generate mask ...')

    mask_matrix = ((map_Zhen_matrix>0) | (map_Varshney_matrix>0) | (map_Kaiser_matrix>0) | (map_Cook_matrix>0) | (map_Turuga_matrix>0)) * 1.0
    torch.save(mask_matrix, f'./graphs_data/{dataset_name}/adjacency.pt')
    print (f'filling factor {torch.sum(mask_matrix)/mask_matrix.shape[0]**2:0.3f}')
    # zero_rows = torch.all(mask_matrix == 0, dim=1)
    # zero_columns = torch.all(mask_matrix == 0, dim=0)
    # mask_matrix[zero_rows] = 1
    # mask_matrix[:, zero_columns] = 1

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    sns.heatmap(to_numpy(mask_matrix), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(all_neuron_list)))
    ax.set_xticklabels(all_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(all_neuron_list)))
    ax.set_yticklabels(all_neuron_list, fontsize=6)
    plt.title('mask', fontsize=18)
    plt.xlabel('pre Neurons', fontsize=18)
    plt.ylabel('post Neurons', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/mask_adjacency_matrix.png", dpi=170)
    plt.close()


    # print('generate partial adjacency matrices ...')
    # # generate partial adjacency matrices for activity neurons
    # map_Cook_matrix_chem , index = map_matrix(activity_neuron_list, Cook_neuron_chem_names, Cook_matrix_chem)
    # map_Cook_matrix_elec , index = map_matrix(activity_neuron_list, Cook_neuron_elec_names, Cook_matrix_elec)
    # map_Cook_matrix = map_Cook_matrix_chem + map_Cook_matrix_elec
    # plot_worm_adjacency_matrix(to_numpy(map_Cook_matrix), activity_neuron_list, 'partial adjacency matrix Cook 2019', f"graphs_data/{dataset_name}/partial_Cook_adjacency_matrix.png")
    #
    # map_Varshney_matrix , index = map_matrix(activity_neuron_list, Varshney_neuron_names, Varshney_matrix)
    # plot_worm_adjacency_matrix(to_numpy(map_Varshney_matrix), activity_neuron_list, 'partial adjacency matrix Varshney 2011', f"graphs_data/{dataset_name}/partial_Varshney_adjacency_matrix.png")
    #
    # map_Zhen_matrix_7 , index = map_matrix(activity_neuron_list, Zhen_neuron_names, Zhen_matrix_7)
    # plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix7), activity_neuron_list, 'partial adjacency matrix Mei Zhen 2021', f"graphs_data/{dataset_name}/partial_Zhen_adjacency_matrix_7.png")
    # map_Zhen_matrix_8 , index = map_matrix(activity_neuron_list, Zhen_neuron_names, Zhen_matrix_8)
    # plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix8), activity_neuron_list, 'partial adjacency matrix Mei Zhen 2021', f"graphs_data/{dataset_name}/partial_Zhen_adjacency_matrix_8.png")
    #
    # map_Kaiser_matrix , index = map_matrix(activity_neuron_list, Kaiser_neuron_names, Kaiser_matrix)
    # plot_worm_adjacency_matrix(to_numpy(map_Kaiser_matrix), activity_neuron_list, 'partial adjacency matrix Kaiser 2006', f"graphs_data/{dataset_name}/partial full_Kaiser_adjacency_matrix.png")





    sensory_neuron_list = Cook_neuron_chem_names[20:103]
    with open(f"graphs_data/{dataset_name}/sensory_neuron_list.json", "w") as f:
        json.dump(sensory_neuron_list, f)
    inter_neuron_list = Cook_neuron_chem_names[103:184]
    with open(f"graphs_data/{dataset_name}/inter_neuron_list.json", "w") as f:
        json.dump(inter_neuron_list, f)
    motor_neuron_list = Cook_neuron_chem_names[184:292]
    with open(f"graphs_data/{dataset_name}/motor_neuron_list.json", "w") as f:
        json.dump(motor_neuron_list, f)
    larynx_neuron_list = Cook_neuron_chem_names[0:20]
    with open(f"graphs_data/{dataset_name}/larynx_neuron_list.json", "w") as f:
        json.dump(larynx_neuron_list, f)
    map_larynx_matrix , index = map_matrix(larynx_neuron_list, all_neuron_list, mask_matrix)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    sns.heatmap(to_numpy(map_larynx_matrix), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(larynx_neuron_list)))
    ax.set_xticklabels(larynx_neuron_list, fontsize=14, rotation=90)
    ax.set_yticks(range(len(larynx_neuron_list)))
    ax.set_yticklabels(larynx_neuron_list, fontsize=14)
    plt.title('larynx adjacency', fontsize=18)
    plt.xlabel('postsynaptic', fontsize=18)
    plt.ylabel('presynaptic', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/mask_larynx_adjacency_matrix.png", dpi=170)
    plt.close()

    # generate data for GNN training
    # create fully connected edges
    n_neurons = len(all_neuron_list)
    edge_index, edge_attr = dense_to_sparse(torch.ones((n_neurons)) - torch.eye(n_neurons))
    torch.save(edge_index.to(device), f'./graphs_data/{dataset_name}/edge_index.pt')
    activity_idx = []
    for k in range(len(activity_neuron_list)):
        neuron_OI = get_neuron_index(activity_neuron_list[k], all_neuron_list)
        activity_idx.append(neuron_OI)
    activity_idx = np.array(activity_idx)

    xc, yc = get_equidistant_points(n_points=n_neurons)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0))
    X1 = to_numpy(pos[perm])

    # type 0 larynx
    # type 1 sensory
    # type 2 inter
    # type 3 motor
    # type 4 other

    T1 = np.ones((n_neurons, 1)) * 4
    type_dict = {}
    for name in larynx_neuron_list:
        type_dict[name] = 0
    for name in sensory_neuron_list:
        type_dict[name] = 1
    for name in inter_neuron_list:
        type_dict[name] = 2
    for name in motor_neuron_list:
        type_dict[name] = 3

    # Default to type 4 ("other") if not found
    T1[activity_idx] = np.array([[type_dict.get(name, 4)] for name in activity_neuron_list])

    for run in range(config.training.n_runs):

        x_list = []
        y_list = []

        for it in trange(0, n_frames-2):
            x = np.zeros((n_neurons, 13))
            x[:, 0] = np.arange(n_neurons)
            x[:, 1:3] = X1
            x[:, 5:6] = T1
            x[:, 6] = 6
            x[activity_idx, 6] = activity_worm[run,:,it]
            x[:, 10:13] = odor_worms[run,:,it]
            x_list.append(x)

            y = (activity_worm[run,:,it+1]- activity_worm[run,:,it]) / delta_t
            y_list.append(y)

            if visualize & (run == 0) & (it % 2 == 0) & (it >= 0):
                plt.style.use('dark_background')

                plt.figure(figsize=(10, 10))
                plt.axis('off')
                values = x[:, 6]
                normed_vals = (values - 4) / (8 - 4)  # (min=4, max=8)

                black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])
                black_to_yellow = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])

                plt.scatter(X1[:, 0], X1[:, 1], s=700, c=normed_vals, cmap=black_to_green)

                plt.scatter(-0.45, 0.5, s=700, c=x[0, 10] + 0.1, cmap= black_to_yellow, vmin=0,vmax=1)
                plt.scatter(-0.4, 0.5, s=700, c=x[0, 11] + 0.1, cmap= black_to_yellow, vmin=0,vmax=1)
                plt.scatter(-0.35, 0.5, s=700, c=x[0, 12] + 0.1, cmap= black_to_yellow, vmin=0,vmax=1)

                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig/Fig_{run}_{it:03d}.tif", dpi=80)
                plt.close()

        x_list = np.array(x_list)
        y_list = np.array(y_list)
        np.save(f'graphs_data/{dataset_name}/x_list_{run}.npy', x_list)
        np.save(f'graphs_data/{dataset_name}/y_list_{run}.npy', y_list)

        activity = torch.tensor(x_list[:, :, 6:7], device=device)
        activity = activity.squeeze().t().cpu().numpy()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(221)
        plt.imshow(activity, aspect='auto', vmin =0, vmax=8, cmap='viridis')
        plt.title(f'dataset {idata}', fontsize=18)
        plt.xlabel('time', fontsize=18)
        plt.ylabel('neurons', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax = fig.add_subplot(222)
        plt.title(f'missing data', fontsize=18)
        test_im = activity * 0
        pos = np.argwhere(activity == 6)
        test_im[pos[:, 0], pos[:, 1]] = 1
        pos = np.argwhere(np.isnan(activity))
        test_im[pos[:, 0], pos[:, 1]] = 2
        pos = np.argwhere(np.isinf(activity))
        test_im[pos[:, 0], pos[:, 1]] = 3
        plt.imshow(test_im[:,500:], aspect='auto',vmin =0, vmax=3, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(223)
        plt.imshow(odor_worms[idata], aspect='auto', vmin =0, vmax=1, cmap='viridis', interpolation='nearest')
        plt.xlabel('time', fontsize=18)
        plt.ylabel('odor', fontsize=18)
        plt.title(f'odor stimuli', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/Fig/Kinograph/Fig_{run}.tif", dpi=80)  # 170.7)
        plt.close()



def load_shrofflab_celegans(
        file_path,
        *,
        replace_missing_cpm=None,
        device='cuda:0'
):
    """
    Load the Shrofflab C. elegans data from a CSV file and convert it to a PyTorch tensor.

    :param file_path: The path to the CSV file.
    :param replace_missing_cpm: If not None, replace missing cpm values (NaN) with this value.
    :param device: The PyTorch device to allocate the tensors on.
    :return: A tuple consisting of:
     * A :py:class:`TimeSeries` object containing the loaded data for each time point.
     * The names of the cells in the data.
    :raises ValueError: If the time series are not part of the same timeframe or if too many cells have abnormal time
    series lengths.
    """

    # Load the data from the CSV file and clean it a bit:
    # - drop rows with missing time values (occurs only at the end of the data)
    # - fill missing cpm values (don't interpolate because data is missing at the beginning or end)
    column_descriptors = {
        "x": CsvDescriptor(filename=file_path, column_name="x", type=np.float32, unit=u.micrometer),
        "y": CsvDescriptor(filename=file_path, column_name="y", type=np.float32, unit=u.micrometer),
        "z": CsvDescriptor(filename=file_path, column_name="z", type=np.float32, unit=u.micrometer),
        "t": CsvDescriptor(filename=file_path, column_name="time", type=np.float32, unit=u.day),
        "cell": CsvDescriptor(filename=file_path, column_name="cell", type=str, unit=u.dimensionless_unscaled),
        "cpm": CsvDescriptor(filename=file_path, column_name="log10 mean cpm", type=np.float32, unit=u.dimensionless_unscaled),
    }
    raw_data = load_csv_from_descriptors(column_descriptors)
    print(f"Loaded {raw_data.shape[0]} rows of data, dropping rows with missing time values...")
    raw_data.dropna(subset=["t"], inplace=True)
    print(f"Remaining: {raw_data.shape[0]} rows")
    if replace_missing_cpm is not None:
        print(f"Filling missing cpm values with {replace_missing_cpm}...")
        raw_data.fillna(replace_missing_cpm, inplace=True)

    # Find the indices where the data for each cell begins (time resets)
    time_reset = np.where(np.diff(raw_data["t"]) < 0)[0] + 1
    timeseries_boundaries = np.hstack([0, time_reset, raw_data.shape[0]])
    n_timepoints = np.diff(timeseries_boundaries).astype(int)
    n_normal_timepoints = np.median(n_timepoints).astype(int)
    start_time, end_time = np.min(raw_data["t"]), np.max(raw_data["t"]) + 1
    n_cells = len(n_timepoints)

    # Sanity checks to make sure the data is not too bad
    n_normal_data = np.count_nonzero(n_timepoints == n_normal_timepoints)
    cell_names = raw_data["cell"].values[timeseries_boundaries[:-1]]
    if (end_time - start_time) != n_normal_timepoints:
        raise ValueError("The time series are not part of the same timeframe.")
    if n_normal_data < 0.5 * n_cells:
        raise ValueError("Too many cells have abnormal time series lengths.")
    if n_normal_data != n_cells:
        abnormal_data = n_timepoints != n_normal_timepoints
        abnormal_cells = cell_names[abnormal_data]
        print(f"Warning: incomplete time series data for {abnormal_cells}")

    # Put values into a TimeSeries object
    relevant_fields = ["x", "y", "z", "cpm", "cell_id"]
    tensors_np = {name: np.nan * np.ones((n_cells * n_normal_timepoints)) for name in relevant_fields}
    time_idx = (raw_data["t"].to_numpy() - start_time).astype(int)
    cell_id = np.repeat(np.arange(n_cells), n_timepoints)
    raw_data.insert(0, "cell_id", cell_id)
    idx = np.ravel_multi_index((cell_id, time_idx), (n_cells, n_normal_timepoints))
    tensors = {}
    for name in relevant_fields:
        tensors_np[name][idx] = raw_data[name].to_numpy()
        split_tensors = np.squeeze(
            np.hsplit(tensors_np[name].reshape((n_cells, n_normal_timepoints)), n_normal_timepoints))
        tensors[name] = [torch.tensor(t, device=device) for t in split_tensors]

    time = torch.arange(start_time, end_time)
    data = [Data(
        time=time[i],
        cell_id=tensors["cell_id"][i],
        pos=torch.stack([tensors["x"][i], tensors["y"][i], tensors["z"][i]], dim=1),
        cpm=tensors["cpm"][i],
    ) for i in range(n_normal_timepoints)]
    time_series = TimeSeries(time, data)

    # Compute the velocity and the derivative of the gene expressions and add them to the time series
    velocity = time_series.compute_derivative('pos')
    d_cpm = time_series.compute_derivative('cpm')
    for i, data in enumerate(time_series):
        data.velocity = velocity[i]
        data.d_cpm = d_cpm[i]

    return time_series, cell_names


def load_celegans_gene_data(
        file_path,
        *,
        coordinate_system: Literal["cartesian", "polar"] = "cartesian",
        device='cuda:0'
):
    """
    Load C. elegans cell data from an HDF5 file (positions and gene expressions) and convert it to a PyTorch tensor.

    :param file_path: The path to the HDF5 file.
    :param coordinate_system: The coordinate system to use for the positions (either "cartesian" or "polar").
    :param device: The PyTorch device to allocate the tensors on.
    :return: A tuple consisting of:
     * A :py:class:`TimeSeries` object containing the loaded data for each time point.
     * A :py:class:`pandas.DataFrame` object containing information about the cells.
    """

    # Load cell information from the HDF5 file (metadata string) into pandas dataframe
    print(f"Loading data from '{file_path}'...")
    file = h5py.File(file_path, 'r')
    cell_info_raw = file["cellinf"][0][0].decode("utf-8")
    cell_info_raw = cell_info_raw.replace("false", "False").replace("true", "True")
    cell_info_raw = eval(cell_info_raw)

    names = [info.pop('name') for info in cell_info_raw]
    cell_info = pd.DataFrame(cell_info_raw, index=names)

    # There are two time series: one for the gene expressions (sparse) and one for the positions (dense)
    # Compute intersection of both time series and interpolate gene expressions where they are not defined
    gene_time = file["gene_time"][0]
    pos_time = file["pos_time"][0]
    min_t = max(gene_time[0], pos_time[0])
    max_t = min(gene_time[-1], pos_time[-1])
    time = np.arange(min_t, max_t + 1)
    pos_overlap = np.isin(pos_time, time)
    genes_overlap = np.isin(gene_time, time)

    # Assign positions
    match coordinate_system:
        case "cartesian":
            positions = file["pos_xyz"][pos_overlap]
        case "polar":
            positions = file["pos_rpz"][pos_overlap]
        case _:
            raise ValueError(f"Invalid coordinate system '{coordinate_system}'")

    # Interpolate gene expressions by piecewise linear spline
    gene_data = file["gene_CPM"][genes_overlap]
    t = gene_time[genes_overlap]
    f = make_interp_spline(t, gene_data, k=1, axis=0, check_finite=False)

    # Due to NaNs in the gene data, the interpolation is not perfect; make sure at least original data is present
    genes_are_present = np.isin(time, gene_time)
    interpolated_to_present_data = -np.ones_like(time, dtype=int)
    interpolated_to_present_data[genes_are_present] = np.arange(np.count_nonzero(genes_overlap))

    # Bundle everything in a TimeSeries object
    data = []
    for t in trange(len(time)):
        if genes_are_present[t]:
            interpolated_gene_data = gene_data[interpolated_to_present_data[t]]
        else:
            interpolated_gene_data = f(time[t])
        data.append(Data(
            time=time[t],
            pos=torch.tensor(positions[t], device=device),
            gene_cpm=torch.tensor(interpolated_gene_data.T, device=device),
        ))
    time_series = TimeSeries(torch.tensor(time, device=device), data)
    file.close()

    # Compute the velocity and the derivative of the gene expressions and add them to the time series
    velocity = time_series.compute_derivative('pos')
    d_cpm = time_series.compute_derivative('gene_cpm')
    for i, data in enumerate(time_series):
        data.velocity = velocity[i]
        data.d_gene_cpm = d_cpm[i]

    return time_series, cell_info


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
