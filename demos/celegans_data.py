import os

from ParticleGraph.data_loaders import load_shrofflab_celegans

root = "<path-to-data>"
path = os.path.join(root, "log10_mean-and-smoothed_lin-32.csv")
data, time, cell_names = load_shrofflab_celegans(path, device="cpu")

print(f"Data was acquired at {len(time)} time points between {time[0]} and {time[-1]}")
print(f"Data was recorded for {len(cell_names)} cells - first 10 cell names: {cell_names[:10]}")
print(f"Shape of data at first time point: {data[0].shape}")




