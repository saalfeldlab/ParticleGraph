"""
This script demonstrates how to load simulated agent data and how to access the data.
"""
from ParticleGraph.data_loaders import load_agent_data

# Load the data set (insert the path to the data set on your system)
# The data consists of a list of time points, each containing a :py:class:`torch_geometric.data.Data` object, and
# a grid of the signal that the agents are responding to
root = "/groups/saalfeld/home/allierc/signaling/Agents/data/mechanistic, noise, chemo/chemo_full_1"
time_series, signal = load_agent_data(root, device="cpu")

# The time series behaves more or less like a list
print(f"Number of time points: {len(time_series)}")

time_point = time_series[0]
print(f"Time point fields: {time_point.node_attrs()}")
print(f"Number of agents in time point 0: {time_point.num_nodes}")

# There are 6 fields in the data objects: pos, velocity, internal, orientation, reversal_timer, and state
print(f"Position shape: {time_point.pos.shape}")
print(f"Velocity shape: {time_point.velocity.shape}")
print(f"Internal variable shape: {time_point.internal.shape}")
print(f"Orientation shape: {time_point.orientation.shape}")
print(f"Internal reversal timer shape: {time_point.reversal_timer.shape}")
print(f"Internal state shape: {time_point.state.shape}")

# The signal is a 2D grid, i.e., the signal is fixed in time
print(f"Signal shape: {signal.shape}")
