"""
This script demonstrates how to load the Shroff Lab C. elegans data set and how to access the data.
"""
import os

import torch

from ParticleGraph.data_loaders import load_shrofflab_celegans

# Load the data set (insert the path to the data set on your system)
# The data consists of a list of time points, each containing a :py:class:`torch_geometric.data.Data` object, and
# a list of cell names
root = "<path-to-data>"
path = os.path.join(root, "log10_mean-and-smoothed_lin-32.csv")
data, cell_names = load_shrofflab_celegans(path, device="cpu")
first_datum = data[0]

# The time series behaves more or less like a list
print(f"Data was acquired at {len(data)} time points between {data.time[0]} and {data.time[-1]}")
print(f"Data was recorded for {len(cell_names)} cells - first 10 cell names: {cell_names[:10]}")

# There are 5 fields in the data objects: cell_id, pos, velocity, cpm (gene data: counts per million), and d_cpm
print(f"Cell IDs for the first time point: {first_datum.cell_id}")
print(f"Shape of positions at first time point: {first_datum.pos.shape}")
print(f"Shape of velocity at first time point: {first_datum.velocity.shape}")
print(f"Shape of gene expression (counts per million) at first time point: {first_datum.cpm.shape}")
print(f"Shape of derivative of gene expression at first time point: {first_datum.d_cpm.shape}")

# Data that is not available is represented by NaN
actual_cell_names = cell_names[torch.logical_not(torch.isnan(first_datum.cell_id))]
print(f"Number of cells actually present in the first time point: {len(actual_cell_names)} of {len(cell_names)}")
