"""
A collection of functions for loading data from various sources.
"""
from typing import Dict

import astropy.units as u
import pandas as pd
from tqdm import trange

from ParticleGraph.TimeSeries import TimeSeries
from ParticleGraph.field_descriptors import CsvDescriptor
from ParticleGraph.utils import *


def skip_to(file, start_line):
    with open(file) as f:
        pos = 0
        cur_line = f.readline()
        while cur_line != start_line:
            pos += 1
            cur_line = f.readline()

        return pos + 1


def convert_data(data, device, config, n_particle_types, n_frames):
    x_list = []
    y_list = []

    for it in trange(n_frames - 1):
        for n in range(n_particle_types):
            # if (n==9):
            #     p=1
            x = data[n][it, 1].clone().detach()
            y = data[n][it, 2].clone().detach()
            z = data[n][it, 3].clone().detach()
            vx = data[n][it, 4].clone().detach()
            vy = data[n][it, 5].clone().detach()
            vz = data[n][it, 6].clone().detach()

            tmp = torch.stack(
                [torch.tensor(n), x, y, vx, vy, torch.tensor(n), torch.tensor(0), torch.tensor(0), torch.tensor(0)])
            if n == 0:
                object_data = tmp[None, :]
            else:
                object_data = torch.cat((object_data, tmp[None, :]), 0)

            ax = data[n][it + 1, 4] - data[n][it, 4]
            ay = data[n][it + 1, 5] - data[n][it, 5]
            tmp = torch.stack([ax, ay]) / config.simulation.delta_t
            if n == 0:
                acc_data = tmp[None, :]
            else:
                acc_data = torch.cat((acc_data, tmp[None, :]), 0)

        x_list.append(object_data.to(device))
        y_list.append(acc_data.to(device))

    return x_list, y_list


def load_solar_system(config, device=None, visualize=False, folder=None, step=1000):
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
                   'europa',
                   'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea', 'titan', 'hyperion', 'moon',
                   'phobos', 'deimos', 'charon']

    # matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(12, 12))

    all_data = []

    for id, object in enumerate(object_list):

        print(f'object: {object}')
        filename = os.path.join(dataset_name, f'{object}.txt')

        df = skip_to(filename, "$$SOE\n")
        data = pd.read_csv(filename, header=None, skiprows=df, sep='\s+', nrows=n_step)
        tmp_x = data.iloc[:, 4:5].values
        tmp_y = data.iloc[:, 5:6].values
        tmp_z = data.iloc[:, 6:7].values
        tmp_vx = data.iloc[:, 7:8].values
        tmp_vy = data.iloc[:, 8:9].values
        tmp_vz = data.iloc[:, 9:10].values
        tmp_x = tmp_x[:, 0][:-1]
        tmp_y = tmp_y[:, 0][:-1]
        tmp_z = tmp_z[:, 0][:-1]
        tmp_vx = tmp_vx[:, 0][:-1]
        tmp_vy = tmp_vy[:, 0][:-1]
        tmp_vz = tmp_vz[:, 0][:-1]
        # convert string to float
        x = torch.ones((n_step - 1))
        y = torch.ones((n_step - 1))
        z = torch.ones((n_step - 1))
        vx = torch.ones((n_step - 1))
        vy = torch.ones((n_step - 1))
        vz = torch.ones((n_step - 1))
        for it in range(n_step - 1):
            x[it] = torch.tensor(float(tmp_x[it][0:-1]))
            y[it] = torch.tensor(float(tmp_y[it][0:-1]))
            z[it] = torch.tensor(float(tmp_z[it][0:-1]))
            vx[it] = torch.tensor(float(tmp_vx[it][0:-1]))
            vy[it] = torch.tensor(float(tmp_vy[it][0:-1]))
            vz[it] = torch.tensor(float(tmp_vz[it][0:-1]))

        object_data = torch.cat((torch.ones_like(x[:, None]) * id, x[:, None], y[:, None], z[:, None], vx[:, None],
                                 vy[:, None], vz[:, None], torch.ones_like(x[:, None]) * id,
                                 torch.zeros_like(x[:, None]), torch.zeros_like(x[:, None]),
                                 torch.zeros_like(x[:, None])), 1)

        all_data.append(object_data)

        plt.plot(to_numpy(y), to_numpy(x))
        plt.text(to_numpy(y[-1]), to_numpy(x[-1]), object, fontsize=6)

    x_list, y_list = convert_data(all_data, device, config, n_particle_types, n_frames + 1)

    dataset_name = config.dataset

    if visualize:
        for it in trange(n_frames - 1):
            if it % step == 0:
                fig = plt.figure(figsize=(12, 12))
                for id, object in enumerate(object_list):
                    plt.scatter(to_numpy(x_list[it][id, 1]), to_numpy(x_list[it][id, 2]), s=20)
                    if id < 10:
                        plt.text(to_numpy(x_list[it][id, 1]), to_numpy(x_list[it][id, 2]), object, fontsize=6)
                    if id == 9:
                        plt.arrow(to_numpy(x_list[it][id, 1]), to_numpy(x_list[it][id, 2]),
                                  to_numpy(y_list[it][id, 0]) * 1E14, to_numpy(y_list[it][id, 1]) * 1E14,
                                  head_width=0.5, head_length=0.7, fc='k', ec='k')
                plt.xlim([-0.5E10, 0.5E10])
                plt.ylim([-0.5E10, 0.5E10])
                plt.tight_layout()
                plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{it}.jpg", dpi=170.7)
                plt.close()

    for run in range(2):
        torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')


def load_shrofflab_celegans(
        file_path,
        *,
        replace_missing_cpm=None,
        device='cuda:0'
):
    """
    Load the Shrofflab C. elegans data from a CSV file and convert it to a PyTorch tensor.

    Args:
        file_path (str): The path to the CSV file.
        replace_missing_cpm (float): If not None, replace missing cpm values with this value.
        device (str): The PyTorch device to use for the tensor.

    Returns:
        tensor_list (List[torch.Tensor]): A list of PyTorch tensors containing the loaded data for each time point.
        time (np.ndarray): The time points corresponding to the data.
        cell_names (np.ndarray): The names of the cells in the data.

    Raises: ValueError: If the time series are not part of the same timeframe or if too many cells have abnormal time
    series lengths.
    """

    # Load the data from the CSV file and clean it a bit:
    # - drop rows with missing time values (occurs only at the end of the data)
    # - fill missing cpm values (don't interpolate because data is missing at the beginning or end)
    print(f"Loading data from {file_path}...")
    dtypes = {"time": np.float32, "x": np.float32, "y": np.float32,
              "z": np.float32, "cell": str, "log10 mean cpm": str}
    data = pd.read_csv(file_path, dtype=dtypes)
    print(f"Loaded {data.shape[0]} rows of data, dropping rows with missing time values...")
    data.dropna(subset=["time"], inplace=True)
    print(f"Remaining: {data.shape[0]} rows")
    if replace_missing_cpm is not None:
        print(f"Filling missing cpm values with {replace_missing_cpm}...")
        data.fillna(replace_missing_cpm, inplace=True)

    # Find the indices where the data for each cell begins (time resets)
    time_reset = np.where(np.diff(data["time"]) < 0)[0] + 1
    timeseries_boundaries = np.hstack([0, time_reset, data.shape[0]])
    n_timepoints = np.diff(timeseries_boundaries).astype(int)
    n_normal_timepoints = np.median(n_timepoints).astype(int)
    start_time, end_time = np.min(data["time"]), np.max(data["time"]) + 1
    n_cells = len(n_timepoints)

    # Sanity checks to make sure the data is not too bad
    n_normal_data = np.count_nonzero(n_timepoints == n_normal_timepoints)
    cell_names = data["cell"].values[timeseries_boundaries[:-1]]
    if (end_time - start_time) != n_normal_timepoints:
        raise ValueError("The time series are not part of the same timeframe.")
    if n_normal_data < 0.5 * n_cells:
        raise ValueError("Too many cells have abnormal time series lengths.")
    if n_normal_data != n_cells:
        abnormal_data = n_timepoints != n_normal_timepoints
        abnormal_cells = cell_names[abnormal_data]
        print(f"Warning: incomplete time series data for {abnormal_cells}")

    # Put values into a 3D tensor
    relevant_fields = ["x", "y", "z", "log10 mean cpm"]
    shape = (n_cells, n_normal_timepoints, len(relevant_fields))
    tensor = np.nan * np.ones((shape[0] * shape[1], shape[2]))
    time_idx = (data["time"].values - start_time).astype(int)
    cell_idx = np.repeat(np.arange(n_cells), n_timepoints)
    idx = np.ravel_multi_index((cell_idx, time_idx), (n_cells, n_normal_timepoints))
    for i, name in enumerate(relevant_fields):
        tensor[idx, i] = data[name].values
    tensor = np.transpose(tensor.reshape(shape), (1, 0, 2))

    # Compute the time derivatives and concatenate such that columns correspond to:
    # x, y, z, d/dt x, d/dt y, d/dt z, cpm, d/dt cpm
    tensor_gradient = tensor - np.roll(tensor, 1, 0)
    tensor_gradient[0, :, :] = np.nan
    tensor = np.concatenate([tensor[:, :, 0:3], tensor_gradient[:, :, 0:3],
                             tensor[:, :, 3:4], tensor_gradient[:, :, 3:4]], axis=2)

    # Put all the time points into a separate tensor
    tensor_list = []
    for i in range(n_normal_timepoints):
        cell_tensor = tensor[i]
        cell_ids = np.where(~np.isnan(cell_tensor[:, 0]))[0]
        cell_tensor = np.column_stack((cell_ids, cell_tensor[cell_ids, :]))
        tensor_list.append(torch.tensor(cell_tensor, device=device))

    time = np.arange(start_time, end_time)

    return tensor_list, time, cell_names


def ensure_local_path_exists(path):
    """
    Ensure that the local path exists. If it doesn't, create the directory structure.

    Args:
        path (str): The path to be checked and created if necessary.

    Returns:
        str: The absolute path of the created directory.
    """

    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(os.getcwd(), path)


def load_csv_from_descriptors(
        column_descriptors: Dict[str, CsvDescriptor],
        **kwargs
) -> Dict[str, torch.Tensor]:
    different_files = set(descriptor.filename for descriptor in column_descriptors.values())
    columns = []

    for file in different_files:
        dtypes = {descriptor.column_name: descriptor.type for descriptor in column_descriptors.values()
                  if descriptor.filename == file}
        print(f"Loading data from '{file}':")
        for column_name, type in dtypes.items():
            print(f"  - column {column_name} as {type}")
        columns.append(pd.read_csv(file, dtype=dtypes, usecols=list(dtypes.keys()), **kwargs))

    entire_data = pd.concat(columns, axis='columns')

    tensors = {}
    for name, descriptor in column_descriptors.items():
        tensors[name] = torch.tensor(entire_data[descriptor.column_name].values)

    return tensors
