"""
This script demonstrates how to load a large C. elegans data set with cell positions and gene expression data and how
to access the data.
"""
import os.path

from ParticleGraph.data_loaders import load_celegans_gene_data

# Load the data set (insert the path to the data set on your system)
# The data consists of a list of time points, each containing a :py:class:`torch_geometric.data.Data` object and
# a :py:class:`pandas.DataFrame` object containing information about the cells
root = "/groups/saalfeld/home/allierc/signaling/Celegans"
path = os.path.join(root, "position_genes.h5")
time_series, cell_info = load_celegans_gene_data(path, device="cpu")

# The info about the cells is stored in a pandas DataFrame
print(f"Number of cells: {len(cell_info)}")
print(cell_info.describe())

cell_name = cell_info.index[5]
cell_type = cell_info['type'].iloc[5]
print(f"First few gene names: {cell_info.index[:6]}")
print(f"Cell type for {cell_name}: {cell_type}")

# Since the location and gene data are acquired at different time intervals, the time series only contains
# the time in which both data are available (gene data is linearly interpolated)
print(f"Number of time points: {len(time_series)}")
print(f"Time points: {time_series.time}")

# The data objects contain the fields 'pos' and 'gene_cpm' and their derivatives ('velocity' and 'd_gene_cpm')
time_point = time_series[0]
print(f"Time point fields: {time_point.node_attrs()}")
print(f"Number of cells in time point 0: {time_point.num_nodes}")

print(f"Position shape: {time_point.pos.shape}")
print(f"Velocity shape: {time_point.velocity.shape}")
print(f"Gene expression shape: {time_point.gene_cpm.shape}")
print(f"Gene expression derivative shape: {time_point.d_gene_cpm.shape}")
