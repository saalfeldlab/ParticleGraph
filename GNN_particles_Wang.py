# import networkx as nx
import umap
# matplotlib.use("Qt5Agg")

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.embedding_cluster import *
# from ParticleGraph.Plot3D import *
from GNN_particles_Ntype import *

if __name__ == '__main__':

    time_series = load_wanglab_salivary_gland('/groups/wang/wanglab/GNN/240104-SMG-HisG-PNA-Cy3-001-SIL/1 - Denoised_Statistics/1 - Denoised_Position.csv')

    frame = 100

    n_particles = time_series[frame].num_nodes

    x = torch.zeros((n_particles, 7),device='cuda:0')
    x[:, 0] = time_series[frame].track_id
    x[:, 1:4] = time_series[frame].pos
    x[:, 4:7] = time_series[frame].velocity

    y = torch.zeros((n_particles, 3), device='cuda:0')
    acceleration = time_series.compute_derivative('velocity', id_name='track_id')

    y = acceleration[frame]

    acceleration = torch.vstack(acceleration)
    mask = torch.logical_not(torch.any(torch.isnan(acceleration), dim=1))
    std = torch.std(acceleration[mask, :], dim=0)









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



