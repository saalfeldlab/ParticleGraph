"""
A collection of functions for loading data from various sources.
"""
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
from ParticleGraph.utils import *
# from ParticleGraph.utils import choose_boundary_values

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
from ParticleGraph.generators import PDE_V
from cellpose import models, core, utils, io, models, metrics, denoise
import scipy.io as sio
import seaborn as sns
from torch_geometric.utils import dense_to_sparse
import pickle
import json
import scipy.io
import h5py

def extract_object_properties(segmentation_image, fluorescence_image=[]):
    # Label the objects in the segmentation image
    labeled_image = label(segmentation_image)

    # matplotlib.use("Qt5Agg")
    # fig = plt.figure(figsize=(13, 10.5))
    # plt.imshow(fluorescence_image)
    # plt.show()

    # Extract properties of the labeled objects
    object_properties = []
    for id, region in enumerate(regionprops(labeled_image, intensity_image=fluorescence_image)):
        # Get the cell ID
        cell_id = id
        # Calculate the position (centroid) of the object
        pos_x = region.centroid[0]
        pos_y = region.centroid[1]
        pos_x_weighted = region.weighted_centroid[0]
        pos_y_weighted = region.weighted_centroid[1]
        # Calculate the area of the object
        area = region.area

        if area > 8:
            # Calculate the perimeter of the object
            perimeter = region.perimeter
            # Calculate the aspect ratio of the bounding box
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-6)
            # Calculate the orientation of the object
            orientation = region.orientation
            # calculate sum intensity
            fluo_sum = region.mean_intensity * area

            object_properties.append((cell_id, pos_x_weighted, pos_y_weighted, area, perimeter, aspect_ratio, orientation, fluo_sum))

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


def load_2D_cell_data(config, device, visualize):

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

    delta_t = simulation_config.delta_t

    bc_pos, bc_dpos = choose_boundary_values('no')

    # Loading Data

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)

    files = glob.glob(f"{folder}/*")
    for f in files:
        if (f[-3:] != 'Fig') & (f[-2:] != 'GT') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (f != 'generation_code.py'):
            os.remove(f)
    files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)

    files = os.listdir(data_folder_name)
    files = [f for f in files if f.endswith('.tif')]
    files.sort()

    os.makedirs(f"{data_folder_name}/SEG", exist_ok=True)
    os.makedirs(f"{data_folder_name}/DN", exist_ok=True)

    model_path = image_data.cellpose_model
    model_custom = models.CellposeModel(gpu=True, pretrained_model=model_path)

    model_denoise = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")

    # model_cyto1 = models.CellposeModel(gpu=True, model_type='cyto3', nchan=2)
    # model_cyto2 = models.CellposeModel(gpu=True, model_type='cyto2', nchan=2)
    # model_cyto3 = models.CellposeModel(gpu=True, model_type='cyto2_cp3', nchan=2)

    print('generate segmentation masks with Cellpose ...')

    for it in trange(len(files)):
        im = tifffile.imread(data_folder_name + files[it])
        im = np.array(im)
        masks, flows, styles, imgs_dn = model_denoise.eval(im, diameter=75, channels=[0,0])
        # masks, flows, styles = model_custom.eval(im, diameter=120, flow_threshold=0.0, invert=False, normalize=True, channels=image_data.cellpose_channel)
        tifffile.imsave(data_folder_name + 'DN/' + files[it], imgs_dn[:,:,0].squeeze())
        # matplotlib.use("Qt5Agg")
        # fig = plt.subplots(figsize=(8, 8))
        # plt.imshow(masks)
        # plt.show()

    # if image_data.tracking_file != '':
    #     im_tracking = tifffile.imread(image_data.tracking_file)
    #     im_tracking = np.array(im_tracking)



    if False:

        if image_data.tracking_file != '':
            df = pd.read_csv(image_data.tracking_file)
            trackmate = dict()
            trackmate['x'] = np.array(df['POSITION_X'][3:]).astype(float)
            trackmate['y'] = np.array(df['POSITION_Y'][3:]).astype(float)
            trackmate['frame'] = np.array(df['FRAME'][3:]).astype(int)
            trackmate['track_ID'] = np.array(df['TRACK_ID'][3:])
            trackmate['track_ID'] = pd.Series(trackmate['track_ID']).fillna(-1).astype(int).to_numpy()


        n_cells = 0
        run = 0
        x_list = []
        y_list = []
        track_list = []
        full_vertice_list = []


        for it in range(0,n_frames-1):

            im_fluo = np.flipud(np.array(tifffile.imread(data_folder_name + files[it])))
            im_seg = np.flipud(np.array(tifffile.imread(data_folder_name + 'SEG/' + files[it])))

            im_dim = im_seg.shape
            if it == 0:
                trackmate['y'] = im_dim[0] - trackmate['y']

            object_properties = extract_object_properties(im_seg, im_fluo[:,:,image_data.membrane_channel])
            object_properties = np.array(object_properties, dtype=float)

            N = np.arange(object_properties.shape[0], dtype=np.float32)[:, None]
            X = object_properties[:,1:3]
            V = np.zeros((X.shape[0], 2))
            T = np.zeros((X.shape[0], 1))
            F = np.zeros((X.shape[0], 2))
            F [:, 1:2] = object_properties[:,7:8]
            AREA = object_properties[:,3:4]
            PERIMETER = object_properties[:,4:5]
            ASPECT = object_properties[:,5:6]
            ORIENTATION = object_properties[:,6:7]
            ID = n_cells + np.arange(object_properties.shape[0])[:, None]

            pos = np.argwhere(trackmate['frame'] == it)
            plt.scatter(trackmate['x'][pos], trackmate['y'][pos], s=10, c='w', alpha=0.75)
            X_trackmate = np.concatenate((trackmate['y'][pos], trackmate['x'][pos]), axis=1)
            trackID = trackmate['track_ID'][pos]

            # Calculate distances between each point in X and each point in X_trackmate
            distances = np.linalg.norm(X[:, None, :] - X_trackmate[None, :, :], axis=2)
            # Find the index of the closest point in X_trackmate for each point in X
            closest_indices = np.argmin(distances, axis=1)
            # Map the track_ID from trackmate to X using the closest indices
            X_track_ID = trackmate['track_ID'][pos][closest_indices]

            x = np.concatenate((N.astype(int), X, V, T, F, AREA, PERIMETER, ASPECT, ORIENTATION, X_track_ID, ID.astype(int)-1), axis=1)

            if (it>0) & (image_data.tracking_file != ''):

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

            y = torch.zeros((x.shape[0], 2), dtype=torch.float32, device=device)
            y_list.append(y)

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
        np.savez(f'graphs_data/{dataset_name}/full_vertice_list_{run}', *full_vertice_list)

        # torch.save(x_list, f'graphs_data/{dataset_name}/x_list_{run}.pt')
        # torch.save(y_list, f'graphs_data/{dataset_name}/y_list_{run}.pt')
        # torch.save(full_vertice_list, f'graphs_data/{dataset_name}/full_vertice_list{run}.pt')

        print(f'n_cells: {n_cells}')


def load_3D_cell_data(config, device, visualize):


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




def load_worm_data(config, device=None, visualize=None, step=None, cmap=None):

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

    # Loading Data from class Worm_Data_Loader(Dataset) in https://github.com/TuragaLab/wormvae

    print ('load connectome ...')
    chem_weights = torch.load(connectome_folder_name + 'chem_weights.pt')
    eassym_weights = torch.load(connectome_folder_name + 'eassym_weights.pt')
    chem_sparsity = torch.load(connectome_folder_name + 'chem_sparsity.pt')
    esym_sparsity = torch.load(connectome_folder_name + 'esym_sparsity.pt')

    with open(connectome_folder_name+"activity_neuron_list.pkl", "rb") as f:
        activity_neuron_list = pickle.load(f)
    with open(connectome_folder_name+"neuron_names.json", "r") as f:
        neuron_names = json.load(f)


    # plot matrixes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(221)
    ax = sns.heatmap(to_numpy(chem_weights), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.title('chemical weights', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = fig.add_subplot(222)
    ax = sns.heatmap(to_numpy(eassym_weights), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.title('electrical weights', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = fig.add_subplot(223)
    ax = sns.heatmap(to_numpy(chem_sparsity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.title('chemical sparsity', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = fig.add_subplot(224)
    ax = sns.heatmap(to_numpy(esym_sparsity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.title('electrical sparsity', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/all_connectome.png", dpi=170)
    plt.close()

    map_list = np.load(connectome_folder_name + 'map_list.npy', allow_pickle=True)

    subset_chem_weights = chem_weights[np.ix_(map_list, map_list)]
    subset_eassym_weights = eassym_weights[np.ix_(map_list, map_list)]
    subset_chem_sparsity = chem_sparsity[np.ix_(map_list, map_list)]
    subset_eassym_sparsity = esym_sparsity[np.ix_(map_list, map_list)]

    # subset_chem_weights_test = subset_chem_weights * 0
    # for k in trange(189):
    #     subset_chem_weights_test[k, :] = chem_weights[map_list[k],map_list]


    adjacency = torch.tensor(subset_chem_weights + subset_eassym_weights, dtype=torch.float32, device=device)
    torch.save(adjacency, f'./graphs_data/{dataset_name}/adjacency.pt')


    # plot matrixes
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(121)
    ax = sns.heatmap(to_numpy(adjacency>0), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    ax.set_xticks(range(len(activity_neuron_list)))
    ax.set_xticklabels(activity_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(activity_neuron_list)))
    ax.set_yticklabels(activity_neuron_list, fontsize=6)
    plt.imshow(to_numpy(chem_weights), aspect='auto', vmin=-1, vmax=1, cmap='viridis')
    plt.title('adjacency matrix', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    ax = fig.add_subplot(122)
    ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    ax.set_xticks(range(len(activity_neuron_list)))
    ax.set_xticklabels(activity_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(activity_neuron_list)))
    ax.set_yticklabels(activity_neuron_list, fontsize=6)
    plt.imshow(to_numpy(chem_weights), aspect='auto', vmin=-1, vmax=1, cmap='viridis')
    plt.title('weight matrix', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/adjacency_matrix.png", dpi=170)


    edge_index, edge_attr = dense_to_sparse(torch.ones((n_particles)) - torch.eye(n_particles))
    torch.save(edge_index.to(device), f'./graphs_data/{dataset_name}/edge_index.pt')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(221)
    ax = sns.heatmap(to_numpy(subset_chem_weights), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.imshow(to_numpy(chem_weights), aspect='auto', cmap='viridis')
    plt.title('chemical weights', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = fig.add_subplot(222)
    ax = sns.heatmap(to_numpy(subset_eassym_weights), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.title('electrical weights', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = fig.add_subplot(223)
    ax = sns.heatmap(to_numpy(subset_chem_sparsity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.title('chemical sparsity', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = fig.add_subplot(224)
    ax = sns.heatmap(to_numpy(subset_eassym_sparsity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},)
    plt.title('electrical sparsity', fontsize=18)
    plt.xlabel('neurons', fontsize=18)
    plt.ylabel('neurons', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/subset_connectome.png", dpi=170)
    plt.close()


    odor_channels = 3
    step = 0.25
    n_runs = 21

    # mat file attributes
    n_particles = 189
    T = 960
    N_length = 109
    T_start = 160
    activity_datasets = np.zeros((n_runs, n_particles, T))
    odor_datasets = np.zeros((n_runs, odor_channels, T))

    print ('load traces ...')

    trace_variable = sio.loadmat(data_folder_name)
    trace_arr = trace_variable['traces']
    print(f"traces shape: {trace_arr.shape}")
    is_L = trace_variable['is_L']
    neurons_name = trace_variable['neurons']
    stim_names = trace_variable["stim_names"]
    stimulate_seconds = trace_variable['stim_times']
    stims = trace_variable['stims']

    for idata in range(n_runs):
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
    # add baseline 2
    activity_worm = activity_datasets[:, :, T_start:] + 2

    activity_with_zeros, missing_matrix = process_activity(activity_worm)
    activity_worm = process_trace(activity_worm)

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

    for idata in trange(n_runs):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(221)
        plt.imshow(activity_worm[idata][:, 0:1000], aspect='auto', vmin =-5, vmax=5, cmap='viridis')
        plt.title(f'dataset {idata}', fontsize=18)
        plt.xlabel('time', fontsize=18)
        plt.ylabel('neurons', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax = fig.add_subplot(222)
        plt.title(f'missing data', fontsize=18)
        test_im = activity_worm[idata][:, 0:1000] * 0
        pos = np.argwhere(activity_worm[idata][:, 0:1000] == 0)
        test_im[pos[:, 0], pos[:, 1]] = 1
        pos = np.argwhere(np.isnan(activity_worm[idata][:, 0:1000]))
        test_im[pos[:, 0], pos[:, 1]] = 2
        pos = np.argwhere(np.isinf(activity_worm[idata][:, 0:1000]))
        test_im[pos[:, 0], pos[:, 1]] = 3
        plt.imshow(test_im[:,500:], aspect='auto',vmin =0, vmax=3, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(223)
        plt.imshow(odor_worms[idata][:, 0:1000], aspect='auto', vmin =0, vmax=1, cmap='viridis', interpolation='nearest')
        plt.xlabel('time', fontsize=18)
        plt.ylabel('odor', fontsize=18)
        plt.title(f'odor stimuli', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{idata}.tif", dpi=80)  # 170.7)
        plt.close()

    xc, yc = get_equidistant_points(n_points=n_particles)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0))
    X1 = to_numpy(pos[perm])

    for run in range(config.training.n_runs):

        x_list = []
        y_list = []

        for it in trange(0, n_frames-2):
            x = np.zeros((n_particles, 13))
            x[:, 0] = np.arange(n_particles)
            x[:, 1:3] = X1
            x[:, 6] = activity_worm[run,:,it]
            x[:, 10:13] = odor_worms[run,:,it]
            x_list.append(x)

            y = (activity_worm[run,:,it+1]- activity_worm[run,:,it]) / delta_t
            y_list.append(y)

            if visualize & (run == 0) & (it % 2 == 0) & (it >= 0):
                plt.style.use('dark_background')

                plt.figure(figsize=(10, 10))
                # plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=10, c=to_numpy(x[:, 6]),
                #             cmap='viridis', vmin=-10, vmax=10, edgecolors='k', alpha=1)
                plt.axis('off')
                plt.scatter(X1[:, 0], X1[:, 1], s=700, c=x[:, 6], cmap='viridis', vmin=-2, vmax=2)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"graphs_data/{dataset_name}/Fig/2D_Fig_{run}_{it}.tif", dpi=80)
                plt.close()

        x_list = np.array(x_list)
        y_list = np.array(y_list)
        np.save(f'graphs_data/{dataset_name}/x_list_{run}.npy', x_list)
        np.save(f'graphs_data/{dataset_name}/y_list_{run}.npy', y_list)


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
