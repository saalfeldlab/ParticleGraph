import os

import torch

from ParticleGraph.data_loaders import load_shrofflab_celegans

root = "<path-to-data>"
path = os.path.join(root, "log10_mean-and-smoothed_lin-32.csv")
data, cell_names = load_shrofflab_celegans(path, device="cpu")
first_datum = data[0]

print(f"Data was acquired at {len(data)} time points between {data.time[0]} and {data.time[-1]}")
print(f"Data was recorded for {len(cell_names)} cells - first 10 cell names: {cell_names[:10]}")
print(f"Cell IDs for the first time point: {first_datum.cell_id}")
print(f"Shape of positions at first time point: {first_datum.pos.shape}")
print(f"Shape of velocity at first time point: {first_datum.velocity.shape}")
print(f"Shape of gene expression (counts per million) at first time point: {first_datum.cpm.shape}")
print(f"Shape of derivative of gene expression at first time point: {first_datum.d_cpm.shape}")

actual_cell_names = cell_names[torch.logical_not(torch.isnan(first_datum.cell_id))]
print(f"Number of cells actually present in the first time point: {len(actual_cell_names)} of {len(cell_names)}")
