import torch

from ParticleGraph.data_loaders import load_wanglab_salivary_gland
from ParticleGraph.utils import bundle_fields

path = "/groups/wang/wanglab/GNN/240104-SMG-HisG-PNA-Cy3-001-SIL/1 - Denoised_Statistics/1 - Denoised_Position.csv"

time_series, global_ids = load_wanglab_salivary_gland(path, device="cpu")

frame = 100
frame_data = time_series[frame]

# IDs are in the range 0, ..., N-1; global ids are stored separately
print(f"Data fields: {frame_data.node_attrs()}")
print(f"Number of particles in frame {frame}: {frame_data.num_nodes}")
print(f"Local ids in frame {frame}: {frame_data.track_id}")
print(f"Global ids in frame {frame}: {global_ids[frame_data.track_id]}")

# summarize some of the fields in a particular dataset
X = bundle_fields(frame_data, "track_id", "pos", "velocity")

# compute the acceleration and a mask to filter out NaN values
acceleration, mask = time_series.compute_derivative("velocity", id_name="track_id")
Y = acceleration[frame]
Y = Y[mask[frame], :]
print(f"NaNs in sanitized acceleration: {torch.isnan(Y).sum()}")

# Sanity-check one to one correspondence between X and Y
#   pred = GNN(X)
#   loss = pred[mask] - Y[mask]

# stack all the accelerations / masks
acceleration = torch.vstack(acceleration)
mask = torch.hstack(mask)
std = torch.std(acceleration[mask, :], dim=0)

# get velocity for all time steps
velocity = torch.vstack([frame.velocity for frame in time_series])

# a TimeSeries object can be sliced like a list
every_second_frame = time_series[::2]
first_ten_frames = time_series[:10]
last_ten_frames_reversed = time_series[-1:-11:-1]
