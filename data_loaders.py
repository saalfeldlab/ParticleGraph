"""
A collection of functions for loading data from various sources.
"""

import numpy as np
import torch
import pandas as pd


def load_shrofflab_celegans(file_path):
    """
    Load the Shrofflab C. elegans data from a CSV file and convert it to a PyTorch tensor.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        torch.Tensor: A PyTorch tensor containing the loaded data.
    """

    dtypes = {"time": np.float32, "x": np.float32, "y": np.float32, "z": np.float32, "cell": str, "log10 mean cpm": str}
    data = pd.read_csv(file_path, dtype=dtypes)
    data, time = clean_and_prepare(data)

    tensor = torch.from_numpy(data)

    return tensor, time


def clean_and_prepare(data):
    """
    Separate the data for each cell into its own dataset.

    Args:
        array (numpy.ndarray): A numpy array containing the data for all cells.

    Returns:
        tensor (numpy.ndarray): tensor where each page is the data for one timepoint.
    """

    # Find the indices where the data for each cell begins (time resets)
    timeseries_start = np.hstack(
        [0, np.where(np.diff(data["time"]) < 0)[0] + 1])
    cutoff_index = timeseries_start[-1]
    data = data.iloc[:cutoff_index]
    n_datapoints = np.diff(timeseries_start).astype(int)
    n_normal_datapoints = np.median(n_datapoints).astype(int)
    timeseries_start = timeseries_start[:-1]

    # Sanity checks to make sure the data is not too bad
    n_cells = len(n_datapoints)
    n_normal_data = np.count_nonzero(n_datapoints == n_normal_datapoints)
    start_time, end_time = np.min(data["time"]), np.max(data["time"]) + 1
    if (end_time - start_time) != n_normal_datapoints:
        raise ValueError("The time series are not part of the same timeframe.")
    if n_normal_data < 0.5 * n_cells:
        raise ValueError("Too many cells have abnormal time series lengths.")
    if n_normal_data != n_cells:
        abnormal_data = timeseries_start[n_datapoints != n_normal_datapoints]
        abnormal_cells = data["cell"][abnormal_data].values
        print(f"Warning: some cells have abnormal time series lengths and will be padded by NaN: {abnormal_cells}")

    # Put values into a 3D tensor
    tensor = np.nan * np.ones((n_cells * n_normal_datapoints, 5))
    time_idx = (data["time"].values - start_time).astype(int)
    cell_idx = np.repeat(np.arange(n_cells), n_datapoints)
    time = np.arange(start_time, end_time)
    idx = np.ravel_multi_index((cell_idx, time_idx), (n_cells, n_normal_datapoints))
    tensor[idx, 0] = data["time"].values
    tensor[idx, 1] = data["x"].values
    tensor[idx, 2] = data["y"].values
    tensor[idx, 3] = data["z"].values
    tensor[idx, 4] = data["log10 mean cpm"].values
    tensor = tensor.reshape((n_cells, n_normal_datapoints, 5))
    np.transpose(tensor, (0, 2, 1))

    return tensor, time
