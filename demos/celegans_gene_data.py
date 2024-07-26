import os.path

from ParticleGraph.data_loaders import load_celegans_gene_data

root = "/groups/saalfeld/home/allierc/signaling/Celegans"
path = os.path.join(root, "position_genes.h5")

time_series, cell_info = load_celegans_gene_data(path, device="cpu")
print(f"Number of cells: {len(cell_info)}")
print(cell_info.describe())
print()

cell_name = cell_info.index[5]
cell_type = cell_info['type'].iloc[5]
print(f"First few gene names: {cell_info.index[:6]}")
print(f"Cell type for {cell_name}: {cell_type}")

print(f"Number of time points: {len(time_series)}")
print(f"Time points: {time_series.time}")

time_point = time_series[0]
print(f"Time point fields: {time_point.node_attrs()}")
print(f"Number of cells in time point 0: {time_point.num_nodes}")

print(f"Position shape: {time_point.pos.shape}")
print(f"Gene expression shape: {time_point.gene_cpm.shape}")
