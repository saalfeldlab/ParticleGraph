# import networkx as nx
import umap
# matplotlib.use("Qt5Agg")

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.embedding_cluster import *
# from ParticleGraph.Plot3D import *
from GNN_particles_Ntype import *
from ParticleGraph.data_loaders import *

if __name__ == '__main__':

    time_series, global_ids = load_wanglab_salivary_gland('/groups/wang/wanglab/GNN/240104-SMG-HisG-PNA-Cy3-001-SIL/1 - Denoised_Statistics/1 - Denoised_Position.csv', device='cpu')

    frame = 100
    frame_data = time_series[frame]
    print(frame_data.node_attrs())

    points = to_numpy(frame_data.pos)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, color='blue')

    # IDs are in the range 0, ..., N-1; global ids are stored separately
    print(f"fields of frames: {frame_data.node_attrs()}")
    print(f"number of particles in frame {frame}: {frame_data.num_nodes}")
    print(f"local ids: {frame_data.track_id}")
    print(f"global ids: {global_ids[frame_data.track_id]}")

    # summarize some of the fields in a particular dataset, i.e. input of GNN
    X = bundle_fields(frame_data, "track_id", "pos", "velocity")

    # compute the acceleration and a mask to filter out NaN values for all frames
    velocity, mask = time_series.compute_derivative('pos', id_name='track_id')
    print(velocity[frame])
    print(torch.count_nonzero(torch.isnan(velocity[100])))
    Y = velocity[frame]
    Y = Y[mask[frame], :]

    print(time_series.time)

    acceleration, mask = time_series.compute_derivative('velocity', id_name='track_id')
    Y = acceleration[frame]
    Y = Y[mask[frame], :]

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




    #
    # config_list = ['arbitrary_3_dropout_10_GD']
    #
    # for config_file in config_list:
    #     # Load parameters from config file
    #     config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    #     # print(config.pretty())
    #
    #     device = set_device(config.training.device)
    #     print(f'device {device}')
    #
    #     # data_generate(config, device=device, visualize=True, run_vizualized=1, style='color', alpha=1, erase=True, bSave=True, step=10) #config.simulation.n_frames // 7)
    #     # data_generate_particle_field(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 20)
    #     data_train(config, device)
    #     # data_test(config, visualize=True, style='color', verbose=False, best_model=20, run=1, step=config.simulation.n_frames // 40, test_simulation=False, sample_embedding=True, device=device)



