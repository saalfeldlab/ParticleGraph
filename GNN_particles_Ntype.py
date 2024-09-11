from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.graph_data_generator import data_generate
from ParticleGraph.models import data_train, data_test
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.utils import *

if __name__ == '__main__':

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    config_list = ['arbitrary_3', 'arbitrary_3_continuous', 'arbitrary_3_3', 'arbitrary_16', 'arbitrary_32',
                   'arbitrary_64', 'arbitrary_3_field_video_bison_quad', 'arbitrary_16', 'arbitrary_16_noise_0_3',
                   'arbitrary_16_noise_0_4', 'arbitrary_16_noise_0_5' \
                                             'arbitrary_3_dropout_30', 'arbitrary_3_dropout_10',
                   'arbitrary_3_dropout_10_no_ghost', 'arbitrary_3_field_boats', 'gravity_16', 'gravity_continuous',
                   'Coulomb_3_256', 'gravity_16_noise_0_4', 'Coulomb_3_noise_0_4', 'Coulomb_3_noise_0_3', \
                   'gravity_16_noise_0_3', 'gravity_16_dropout_10', 'gravity_16_dropout_30',
                   'Coulomb_3_dropout_10_no_ghost', 'Coulomb_3_dropout_10', 'boids_16_256', 'boids_32_256',
                   'boids_64_256', 'boids_16_noise_0_3', 'boids_16_noise_0_4', 'boids_16_dropout_10', \
                   'boids_16_dropout_10_no_ghost', 'wave_slit_ter', 'wave_boat_ter', 'RD_RPS', 'signal_N_100_2_a']

    for config_file in config_list:
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        device = set_device(config.training.device)
        print(f'device {device}')
        data_generate(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 100)
        # data_train(config, config_file, False, device)
        # data_test (config=config, config_file=config_file, visualize=True, style='color', verbose=False, best_model='0_7500', run=0, step=1, save_velocity=True, device=device) #config.simulation.n_frames // 3, test_simulation=False, sample_embedding=False, device=device)    # config.simulation.n_frames // 7
