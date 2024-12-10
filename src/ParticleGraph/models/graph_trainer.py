import time

import matplotlib.pyplot as plt
import torch
import seaborn as sns
import torch.nn.functional as F
import random

from GNN_particles_Ntype import *
from ParticleGraph.models.utils import *
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Ghost_Particles import *
from geomloss import SamplesLoss
from ParticleGraph.sparsify import EmbeddingCluster, sparsify_cluster
from ParticleGraph.models.Smooth_Particle import Smooth_Particle

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit

from ParticleGraph.generators.cell_utils import *
from ParticleGraph.fitting_models import linear_model
from torch_geometric.utils import dense_to_sparse
import torch.optim as optim

def data_train(config=None, config_file=None, erase=False, best_model=None, device=None):
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    has_mesh = (config.graph_model.mesh_model_name != '')
    has_signal = (config.graph_model.signal_model_name != '')
    has_particle_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)
    has_cell_division = config.simulation.has_cell_division
    do_tracking = config.training.do_tracking
    has_state = (config.simulation.state_type != 'discrete')
    has_WBI = 'WBI' in config.dataset
    has_mouse_city = 'mouse_city' in config.dataset
    has_time_window = config.training.time_window > 0

    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    if 'Agents' in config.graph_model.particle_model_name:
        data_train_agents(config, config_file, best_model, device)
    elif has_mouse_city:
        data_train_mouse_city(config, config_file, erase, best_model, device)
    elif has_WBI:
        data_train_WBI(config, config_file, erase, best_model, device)
    elif has_particle_field:
        data_train_particle_field(config, config_file, erase, best_model, device)
    elif has_mesh:
        data_train_mesh(config, config_file, erase, best_model, device)
    elif has_signal:
        if ('PDE_N2' in config.graph_model.signal_model_name) | ('PDE_N4' in config.graph_model.signal_model_name) | (
                'PDE_N5' in config.graph_model.signal_model_name):
            data_train_synaptic2(config, config_file, erase, best_model, device)
        elif 'PDE_N3' in config.graph_model.signal_model_name:
            data_train_synaptic3(config, config_file, erase, best_model, device)
        else:
            data_train_synaptic(config, config_file, erase, best_model, device)
    elif do_tracking & has_cell_division:
        data_train_cell(config, config_file, erase, best_model, device)
    elif do_tracking:
        data_train_tracking(config, config_file, erase, best_model, device)
    elif has_cell_division:
        data_train_cell(config, config_file, erase, best_model, device)
    elif has_state:
        data_train_particles_with_states(config, config_file, erase, best_model, device)
    elif 'PDE_GS' in config.graph_model.particle_model_name:
        data_solar_system(config, config_file, erase, best_model, device)
    else:
        data_train_particle(config, config_file, erase, best_model, device)


def data_train_particle(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    train_norm = train_config.train_norm
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    time_window = train_config.time_window
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    rotation_augmentation = train_config.rotation_augmentation
    translation_augmentation = train_config.translation_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    recursive_loop = train_config.recursive_loop
    smooth_particle = train_config.smooth_particle
    predict_density = train_config.predict_density
    coeff_predict_density = train_config.coeff_predict_density


    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    has_ghost = train_config.n_ghosts > 0
    has_bounding_box = 'PDE_F' in model_config.particle_model_name
    n_ghosts = train_config.n_ghosts
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    has_state = (config.simulation.state_type != 'discrete')
    coeff_entropy_loss = train_config.coeff_entropy_loss
    entropy_loss = KoLeoLoss()

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    print('Load data ...')
    x_list = []
    y_list = []
    run_lengths = list()
    time.sleep(0.5)
    for run in trange(n_runs):
        x = np.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
        run_lengths.append(len(x))
    x = torch.tensor(x_list[0][0], dtype=torch.float32, device=device)
    y = torch.tensor(y_list[0][0], dtype=torch.float32, device=device)
    if n_runs >= 100:
        n_steps = n_runs // 10
    else:
        n_steps = 1
    time.sleep(0.5)
    for run in trange(0, n_runs, n_steps):
        for k in range(run_lengths[run]):
            if (k % 10 == 0) | (n_frames < 1000):
                try:
                    x = torch.cat((x, torch.tensor(x_list[run][k], dtype=torch.float32, device=device)), 0)
                except:
                    print(f'Error in run {run} frame {k}')
                y = torch.cat((y, torch.tensor(y_list[run][k], dtype=torch.float32, device=device)), 0)
        time.sleep(0.5)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    x = []
    y = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    if best_model != None:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    x = torch.tensor(x_list[1][0], dtype=torch.float32, device=device)
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')

    if has_ghost:
        ghosts_particles = Ghost_Particles(config, n_particles, vnorm, device)
        optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    Niter = n_frames * data_augmentation_loop // batch_size
    print(f'plot every {Niter // 50} iterations')

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    list_loss_density = []
    time.sleep(1)
    for epoch in range(start_epoch, n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        total_loss_density = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            dataset_batch = []
            for batch in range(batch_size):

                run = 1 + np.random.randint(n_runs - 1)
                k = time_window + np.random.randint(run_lengths[run] - 1 - time_window - recursive_loop)
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()

                if translation_augmentation:
                    displacement = torch.randn(1, dimension, dtype=torch.float32, device=device) * 5
                    displacement = displacement.repeat(x.shape[0], 1)
                    x[:, 1:dimension + 1] = x[:, 1:dimension + 1] + displacement

                if batch == 0:
                    data_id = torch.ones((x.shape[0],1), dtype=torch.int) * run
                else:
                    data_id = torch.cat((data_id, torch.ones((x.shape[0],1), dtype=torch.int) * run), dim = 0)

                if has_ghost:
                    x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                    if ghosts_particles.boids:
                        distance = torch.sum(
                            bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                        dist_np = to_numpy(distance)
                        ind_np = torch.min(distance, axis=1)[1]
                        x_ghost[:, 3:5] = x[ind_np, 3:5].clone().detach()
                    x = torch.cat((x, x_ghost), 0)

                    with torch.no_grad():
                        model.a[run, n_particles:n_particles + n_ghosts] = model.a[
                            run, ghosts_particles.embedding_index].clone().detach()  # sample ghost embedding

                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edges = adj_t.nonzero().t().contiguous()

                if time_window == 0:
                    dataset = data.Data(x=x[:, :], edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)
                else:
                    xt = []
                    for t in range(time_window):
                        x_ = torch.tensor(x_list[run][k - t], dtype=torch.float32, device=device)
                        if translation_augmentation:
                            x_[:, 1:dimension + 1] = x_[:, 1:dimension + 1] + displacement
                        xt.append(x_[:, :])

                    dataset = data.Data(x=xt, edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)

                y = torch.tensor(y_list[run][k], dtype=torch.float32, device=device).clone().detach()
                if predict_density:
                    y = torch.cat((y, torch.tensor(x_list[run][k+1][:,-1][:,None], dtype=torch.float32, device=device).clone().detach()), dim=1)
                if recursive_loop> 0:
                    for m in range(recursive_loop):
                        y_= model.recursive_param[m] * torch.tensor(y_list[run][k+m+1], dtype=torch.float32, device=device).clone().detach()
                        y = torch.cat((y, y_), dim = 1)

                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)

                y[:,0:dimension] = y[:,0:dimension] / ynorm

                if rotation_augmentation:
                    new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                    new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                    y[:, 0] = new_x
                    y[:, 1] = new_y

                if batch == 0:
                    y_batch = y
                else:
                    y_batch = torch.cat((y_batch, y), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()

            for batch in batch_loader:
               pred = model(batch, data_id=data_id, training=True, vnorm=vnorm, phi=phi)

            if has_ghost:
                loss = ((pred[mask_ghost] - y_batch)).norm(train_norm)
            elif predict_density:
                loss_density = coeff_predict_density * (pred[:, dimension] - y_batch[:, dimension]).norm(train_norm)
                loss = (pred[:, 0:dimension] - y_batch[:, 0:dimension]).norm(train_norm) + loss_density
            else:
                loss = (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()

            # flag = True
            # if (recursive_loop > 0) | predict_density :
            #     for name, p in model.named_parameters():
            #         if torch.isnan(p.grad).any():
            #     print('grad nan')
            #     print(torch.unique(data_id))
            #     logger.info('grad nan')

            if has_ghost:
                optimizer_ghost_particles.step()

            total_loss += loss.item()
            if predict_density:
                total_loss_density += loss_density.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % (Niter // 50) == 0)) | (N == 0)):
                plot_training(config=config, dataset_name=dataset_name, log_dir=log_dir,
                              epoch=epoch, N=N, x=x, model=model, n_nodes=0, n_node_types=0, index_nodes=0,
                              dataset_num=1,
                              index_particles=index_particles, n_particles=n_particles,
                              n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
            if ((epoch == 0) & (N % (Niter // 200) == 0)):
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50,
                                   memory_percentage_threshold=0.6)



        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))
        if predict_density:
            print("Epoch {}. Loss density: {:.6f}".format(epoch, total_loss_density / (N + 1) / n_particles / batch_size))
            logger.info("Epoch {}. Loss density: {:.6f}".format(epoch, total_loss_density / (N + 1) / n_particles / batch_size))
            list_loss_density.append(total_loss_density / (N + 1) / n_particles / batch_size)
            torch.save(list_loss_density, os.path.join(log_dir, 'loss_density.pt'))

        scheduler.step()
        print(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}')
        logger.info(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}')

        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()},
                       os.path.join(log_dir, 'models', f'best_ghost_particles_with_{n_runs - 1}_graphs_{epoch}.pt'))

        fig = plt.figure(figsize=(22, 4))
        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='k')
        if predict_density:
            plt.plot(list_loss_density, color='b')

        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        if (simulation_config.n_interactions < 100) & (has_bounding_box == False) & ('PDE_K' not in model_config.particle_model_name):
            ax = fig.add_subplot(1, 5, 2)
            embedding = get_embedding(model.a, 1)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0],
                            embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)

            ax = fig.add_subplot(1, 5, 3)
            func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                                model_MLP=model.lin_edge, model_a=model.a,
                                                                n_nodes=0,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                                cmap=cmap, dimension=dimension, device=device)

            labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                              train_config.cluster_distance_threshold, type_list,
                                                              n_particle_types, embedding_cluster)

            accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')
            logger.info(f'accuracy: {np.round(accuracy, 3)}    n_clusters: {n_clusters}')

            ax = fig.add_subplot(1, 5, 4)
            for n in np.unique(new_labels):
                pos = np.array(np.argwhere(new_labels == n).squeeze().astype(int))
                if pos.size > 0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=5)
            plt.xlabel('proj 0', fontsize=12)
            plt.ylabel('proj 1', fontsize=12)
            plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top',
                     transform=ax.transAxes, fontsize=10)

            ax = fig.add_subplot(1, 5, 5)
            model_a_ = model.a[1].clone().detach()
            for n in range(n_clusters):
                pos = np.argwhere(labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    median_center = model_a_[pos, :]
                    median_center = torch.median(median_center, dim=0).values
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                    model_a_[pos, :] = median_center
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=10, c='k')

            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)
            plt.xticks(fontsize=10.0)
            plt.yticks(fontsize=10.0)

            if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq - 1) & (
                    epoch < n_epochs - sparsity_freq):
                # Constrain embedding domain
                with torch.no_grad():
                    model.a[1] = model_a_.clone().detach()
                print(f'regul_embedding: replaced')
                logger.info(f'regul_embedding: replaced')

                # Constrain function domain
                if train_config.sparsity == 'replace_embedding_function':

                    logger.info(f'replace_embedding_function')
                    y_func_list = func_list * 0

                    ax, fig = fig_init()
                    for n in np.unique(new_labels):
                        pos = np.argwhere(new_labels == n)
                        pos = pos.squeeze()
                        if pos.size > 0:
                            target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                            y_func_list[pos] = target_func
                        plt.plot(to_numpy(target_func) * to_numpy(ynorm), linewidth=2, alpha=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}_before training function.tif")
                    plt.close()

                    lr_embedding = 1E-12
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    for sub_epochs in range(20):
                        loss = 0
                        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                        pred = []
                        optimizer.zero_grad()
                        for n in range(n_particles):
                            embedding_ = model.a[1, n, :].clone().detach() * torch.ones(
                                (1000, model_config.embedding_dim), device=device)
                            match model_config.particle_model_name:
                                case 'PDE_ParticleField_A' | 'PDE_A':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                case 'PDE_ParticleField_B' | 'PDE_B':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                case 'PDE_G':
                                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                             rr[:, None] / max_radius, 0 * rr[:, None],
                                                             0 * rr[:, None],
                                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                case 'PDE_E':
                                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                             rr[:, None] / max_radius, embedding_, embedding_), dim=1)
                                case 'PDE_K':
                                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                             rr[:, None] / max_radius, embedding_, embedding_), dim=1)
                                case 'PDE_K1':
                                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                             rr[:, None] / max_radius), dim=1)
                            pred.append(model.lin_edge(in_features.float()))
                        pred = torch.stack(pred)
                        loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                        logger.info(f'    loss: {np.round(loss.item() / n_particles, 3)}')
                        loss.backward()
                        optimizer.step()

                if train_config.fix_cluster_embedding:
                    lr_embedding = 1E-12
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

            else:
                if epoch > n_epochs - sparsity_freq:
                    lr_embedding = train_config.learning_rate_embedding_end
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                    lr = train_config.learning_rate_start
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()


def data_solar_system(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    target_batch_size = train_config.batch_size
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    data_augmentation_loop = train_config.data_augmentation_loop
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    n_runs = train_config.n_runs
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)

    vnorm = torch.tensor(1, device=device)
    ynorm = torch.tensor(1.0E-6, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_0_0.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    x = x_list[1][0].clone().detach()
    n_particles = x.shape[0]
    config.simulation.n_particles = n_particles
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    Niter = n_frames * data_augmentation_loop // batch_size
    print(f'plot every {Niter // 100} iterations')

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in range(Niter):

            run = 1 + np.random.randint(n_runs - 1)

            dataset_batch = []
            for batch in range(batch_size):

                k = np.random.randint(n_frames - 10)

                x = x_list[run][k].clone().detach()

                dataset = data.Data(x=x[:, :], edge_index=model.edges)
                dataset_batch.append(dataset)

                y = y_list[run][k].clone().detach() / ynorm
                if batch == 0:
                    y_batch = y
                else:
                    y_batch = torch.cat((y_batch, y), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=1)

            loss = (pred - y_batch).norm(2)
            # loss = ((pred - y_batch) / y_batch.norm(2)).norm(2)

            loss.backward()
            optimizer.step()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % (Niter // 100) == 0)) | (N == 0)):
                print(f'loss: {loss.item():.4e}')
                fig, ax = fig_init()
                plt.scatter(to_numpy(y_batch[:, 0]), to_numpy(pred[:, 0]), s=40, c='k', alpha=0.5)
                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.xticks(fontsize=18.0)
                plt.yticks(fontsize=18.0)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/func_{dataset_name}_{epoch}_{N}.tif", dpi=87)
                plt.close()

                fig, ax = fig_init()
                gt_mass_log = to_numpy(model.log10_mass.squeeze())
                pred_mass_log = to_numpy(model.a[1, :, 0].squeeze())
                plt.scatter(gt_mass_log, pred_mass_log, s=40, c='k', alpha=0.5)
                plt.xlim([0.4, 1.1])
                plt.ylim([0.4, 1.1])
                plt.xticks(fontsize=18.0)
                plt.yticks(fontsize=18.0)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/embedding/embedding_{dataset_name}_{epoch}_{N}.tif", dpi=87)
                plt.close()

                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(22, 4))
        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()


def data_train_cell(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    rotation_augmentation = train_config.rotation_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    has_inert_model = simulation_config.cell_inert_model_coeff > 0
    do_tracking = train_config.do_tracking
    has_state = (simulation_config.state_type != 'discrete')
    max_radius = simulation_config.max_radius

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    edge_p_p_list = []
    vertices_pos_list = []

    n_particles_max = 0
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        if x[-1][-1, 0] > n_particles_max:
            n_particles_max = x[-1][-1, 0] + 1
        if run > 0:
            y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
            edge_p_p = np.load(f'graphs_data/graphs_{dataset_name}/edge_p_p_list_{run}.npz')
            x_list.append(x)
            y_list.append(y)
            edge_p_p_list.append(edge_p_p)
            # if has_inert_model:
            #     vertices_pos = np.load(f'graphs_data/graphs_{dataset_name}/vertices_pos_list_{run}.npz')
            #     vertices_pos_list.append(vertices_pos)
        else:
            # first dataset is not loaded to spare memory
            # first dataset is not used for training but for validation
            small_tensor = torch.zeros((1, 1), dtype=torch.float32, device=device)
            x_list.append(small_tensor)
            y_list.append(small_tensor)
            # vertices_pos_list.append(small_tensor)
            edge_p_p_list.append(to_numpy(small_tensor))

    n_particles_max = int(to_numpy(n_particles_max))
    n_particles = n_particles_max
    config.simulation.n_particles = n_particles_max
    type_list = torch.load(f'graphs_data/graphs_{dataset_name}/type_list_1.pt', map_location=device)
    if n_particles_max > type_list.shape[0]:
        type_list = torch.cat(
            (type_list, torch.zeros((n_particles_max - type_list.shape[0], 1), dtype=torch.int64, device=device)))

    x = x_list[1][0].clone().detach()
    y = y_list[1][0].clone().detach()
    config.simulation.n_particles_max = n_particles_max

    for run in range(1, n_runs):
        for k in trange(n_frames):
            if (k % 10 == 0) | (n_frames < 1000):
                x = torch.cat((x, x_list[run][k].clone().detach()), 0)
                y = torch.cat((y, y_list[run][k].clone().detach()), 0)
        print(x_list[run][k].shape)
        time.sleep(0.5)

    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    np.save(os.path.join(log_dir, 'n_particles_max.npy'), n_particles_max)
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    if do_tracking | has_state:
        n_particles_max = 0
        id_list = []
        type_list = []
        for k in range(n_frames + 1):
            type = x_list[1][k][:, 5]
            type_list.append(type)
            if k == 0:
                type_stack = type
            else:
                type_stack = torch.cat((type_stack, type), 0)
            ids = x_list[1][k][:, -1]
            id_list.append(ids)
            n_particles_max += len(type)
            if do_tracking:
                x_list[1][k][:, 3:5] = 0
        config.simulation.n_particles_max = n_particles_max
        Initial_node_id = id_list
    else:
        for k in range(n_frames + 1):
            type = x_list[1][k][:, 5]
            if k == 0:
                type_stack = type
            else:
                type_stack = torch.cat((type_stack, type), 0)

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    if best_model != None:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x = x_list[1][n_frames - 1].clone().detach()

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    Niter = n_frames * data_augmentation_loop // batch_size
    print(f'plot every {Niter // 10} iterations')

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    time.sleep(1)
    for epoch in range(start_epoch, n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(n_runs - 1)

            dataset_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 2)

                x = x_list[run][k].clone().detach()

                edges = edge_p_p_list[run][f'arr_{k}']
                edges = torch.tensor(edges, dtype=torch.int64, device=device)
                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)

                y = y_list[run][k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)
                y = y / ynorm
                if batch == 0:
                    y_batch = y[:, 0:2]
                else:
                    y_batch = torch.cat((y_batch, y[:, 0:2]), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)

            optimizer.zero_grad()

            for i, batch in enumerate(batch_loader):
                pred = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi)

            if rotation_augmentation:
                new_x = cos_phi * pred[:, 0] - sin_phi * pred[:, 1]
                new_y = sin_phi * pred[:, 0] + cos_phi * pred[:, 1]
                pred[:, 0] = new_x
                pred[:, 1] = new_y

            if do_tracking:
                x_next = x_list[run][k + 1]
                x_pos_next = x_next[:, 1:3].clone().detach()
                if model_config.prediction == '2nd_derivative':
                    x_pos_pred = (x[:, 1:3] + delta_t * (x[:, 3:5] + delta_t * pred * ynorm))
                else:
                    x_pos_pred = (x[:, 1:3] + delta_t * pred * ynorm)
                distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_value = result.values
                pos_pre = min_value
                indices = result.indices
                pos = torch.argwhere(min_value < 0.5E-5)
                if model_config.prediction == '2nd_derivative':
                    loss = torch.sum(pos_pre[pos]) * 1E8
                else:
                    loss = torch.sum(pos_pre) * 1E5
            else:
                loss = (pred - y_batch).norm(2)  # + model.a.norm(1) * 1E-3

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 10) & (N % (Niter // 10) == 0)) | (N == 0)):
                if do_tracking | has_state:
                    id_list = []
                    for k in range(n_frames + 1):
                        ids = x_list[1][k][:, -1]
                        id_list.append(ids)
                    plot_training_state(config=config, id_list=id_list, dataset_name=dataset_name, log_dir=log_dir,
                                        epoch=epoch, N=N, model=model, n_particle_types=n_particle_types,
                                        type_list=type_list, type_stack=type_stack, ynorm=ynorm, cmap=cmap,
                                        device=device)
                else:
                    plot_training_cell(config=config, dataset_name=dataset_name, log_dir=log_dir,
                                       epoch=epoch, N=N, model=model, n_particle_types=n_particle_types,
                                       type_list=type_list, ynorm=ynorm, cmap=cmap, device=device)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 20,
                                   memory_percentage_threshold=0.6)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if do_tracking:
            id_list_ = []
            for k in range(0, n_frames + 1):
                ids = x_list[1][k][:, -1]
                if k == 0:
                    id_list_ = ids
                else:
                    id_list_ = torch.cat((id_list_, ids), 0)
            print(f'N tracks: {len(torch.unique(id_list_))}')
            logger.info(f'N tracks: {len(torch.unique(id_list_))}')
            tracking_index_list = []
            if epoch == 0:
                distance_threshold = 1E-3
            elif epoch == 1:
                distance_threshold = 1E-3
            else:
                distance_threshold = 1E-4
            for k in trange(n_frames):
                x = x_list[1][k].clone().detach()
                x_list[1][k + 1][:, -1] = Initial_node_id[k + 1]

                edges = edge_p_p_list[1][f'arr_{k}']
                edges = torch.tensor(edges, dtype=torch.int64, device=device)
                dataset = data.Data(x=x[:, :], edge_index=edges)

                pred = model(dataset, training=True, vnorm=vnorm, phi=torch.zeros(1, device=device))

                x_next = x_list[1][k + 1]
                x_pos_next = x_next[:, 1:3].clone().detach()
                if config.graph_model.prediction == '2nd_derivative':
                    x_pos_pred = (x[:, 1:3] + delta_t * (x[:, 3:5] + delta_t * pred * ynorm))
                else:
                    x_pos_pred = (x[:, 1:3] + delta_t * pred * ynorm)
                distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_distance_value = result.values
                min_index = result.indices

                first_cell_id = to_numpy(x[:, 0])
                next_cell_id = to_numpy(x_next[min_index, 0])

                tracking_distance = torch.sqrt(min_distance_value)

                bVisu = False

                if bVisu:
                    fig = plt.figure(figsize=(8, 8))
                    plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=10, c='k')
                pos = torch.argwhere(tracking_distance < distance_threshold)
                if len(pos) > 0:  # reduce dimension of latent space, the connected cell inherit the cell_id from the previous frame
                    if bVisu:
                        plt.scatter(to_numpy(x_pos_pred[pos, 0]), to_numpy(x_pos_pred[pos, 1]), s=10, c='g', alpha=0.5)

                    list_first = np.arange(len(x))[to_numpy(pos).astype(int)].squeeze()
                    list_next = to_numpy(min_index[to_numpy(pos).astype(int)]).squeeze()
                    list = np.concatenate((list_first[:, None], list_next[:, None]), axis=1)

                    node_id_first = to_numpy(x_list[1][k][list_first, -1])
                    node_id_next = to_numpy(x_list[1][k + 1][list_next, -1])

                    with torch.no_grad():
                        model.a[node_id_next] = model.a[node_id_first].clone().detach()

                    x_list[1][k + 1][list_next, -1] = x_list[1][k][list_first, -1].clone().detach()
                    if (model_config.prediction == '2nd_derivative'):
                        new_velocity = (x_list[1][k + 1][list_next, 1:3].clone().detach() - x_list[1][k][list_first,
                                                                                            1:3].clone().detach()) / delta_t
                        x_list[1][k + 1][list_next, 3:5] = new_velocity.clone().detach()

                tracking_index_list.append(np.sum((first_cell_id == next_cell_id) * 1.0) / len(x) * 100)

                if bVisu:
                    pos = torch.argwhere(tracking_distance >= distance_threshold)
                    if len(pos) > 0:
                        plt.scatter(to_numpy(x_pos_pred[pos, 0]), to_numpy(x_pos_pred[pos, 1]), s=10, c='r', alpha=0.5)
                    for n in range(len(x)):
                        plt.text(to_numpy(x[n, 1]) + 0.005, to_numpy(x[n, 2]) - 0.005,
                                 str(int(to_numpy(x_list[1][k][n, -1]))), fontsize=6)
                    plt.xlim([-0.2, 1.2])
                    plt.ylim([-0.2, 1.2])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_recons/tracking_{epoch}_{k}.tif")
                    plt.close()

                # fig = plt.figure(figsize=(8, 8))
                # plt.scatter(to_numpy(x[:, 1]),to_numpy(x[:, 2]),s=10,c='k')
                # plt.scatter(to_numpy(x_pos_next[:, 0]),to_numpy(x_pos_next[:, 1]),s=10,c='k',alpha=0.5)
                # pos = np.argwhere(first_cell_id!=next_cell_id)
                # plt.scatter(to_numpy(x[pos, 1]),to_numpy(x[pos, 2]),s=10,c='r',alpha=1)
                # plt.scatter(to_numpy(x_pos_pred[pos, 0]),to_numpy(x_pos_pred[pos, 1]),s=10,c='r',alpha=0.5)

            fig = plt.figure(figsize=(22, 4))

            ax = fig.add_subplot(1, 5, 1)
            id_list = []
            id_list_ = []
            for k in range(0, n_frames + 1):
                ids = x_list[1][k][:, -1]
                id_list.append(ids)
                if k == 0:
                    id_list_ = ids
                else:
                    id_list_ = torch.cat((id_list_, ids), 0)
                if k % 10 == 0:
                    for n in range(n_particle_types):
                        pos = torch.argwhere(type_list[k] == n)
                        if len(pos) > 0:
                            plt.scatter(to_numpy(model.a[to_numpy(ids[pos]), 0]),
                                        to_numpy(model.a[to_numpy(ids[pos]), 1]), s=5, color=cmap.color(n), alpha=0.5)

            print(f'N tracks: {len(torch.unique(id_list_))}')
            logger.info(f'N tracks: {len(torch.unique(id_list_))}')

            ax = fig.add_subplot(1, 5, 2)
            func_list, true_type_list, short_model_a_list, proj_interaction = analyze_edge_function_state(rr=[],
                                                                                                          config=config,
                                                                                                          model=model,
                                                                                                          id_list=id_list,
                                                                                                          type_list=type_list,
                                                                                                          ynorm=ynorm,
                                                                                                          cmap=cmap,
                                                                                                          visualize=True,
                                                                                                          device=device)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
            plt.close()

        # embedding = proj_interaction
        # labels, n_clusters, new_labels = sparsify_cluster_state(config.training.cluster_method,
        #                                                         proj_interaction, embedding,
        #                                                         config.training.cluster_distance_threshold,
        #                                                         true_type_list,
        #                                                         n_particle_types, embedding_cluster)


def data_train_mouse_city(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    n_epochs = train_config.n_epochs
    n_particle_types = simulation_config.n_particle_types
    time_step = simulation_config.time_step
    delta_t = simulation_config.delta_t * time_step
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    rotation_augmentation = train_config.rotation_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    time_step = simulation_config.time_step
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    has_inert_model = simulation_config.cell_inert_model_coeff > 0
    do_tracking = train_config.do_tracking
    distance_threshold = train_config.distance_threshold
    epoch_distance_replace = train_config.epoch_distance_replace
    has_state = (simulation_config.state_type != 'discrete')

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    os.makedirs(f'./{log_dir}/tmp_training/loss', exist_ok=True)

    x_list = []
    x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device)
    x_list.append(x)
    edge_p_p_list = []
    edge_p_p = np.load(f'graphs_data/graphs_{dataset_name}/edge_p_p_list_0.npz')
    edge_p_p_list.append(edge_p_p)
    n_frames = len(x)
    config.simulation.n_frames = n_frames

    vnorm = torch.ones(1, dtype=torch.float32, device=device)
    ynorm = torch.ones(1, dtype=torch.float32, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    if do_tracking | has_state:
        n_particles_max = 0
        id_list = []
        type_list = []
        for k in range(n_frames):
            type = x_list[0][k][:, 5]
            type_list.append(type)
            if k == 0:
                type_stack = type
            else:
                type_stack = torch.cat((type_stack, type), 0)
            ids = x_list[0][k][:, -1]
            id_list.append(ids)
            n_particles_max += len(type)
        config.simulation.n_particles_max = n_particles_max
    np.save(os.path.join(log_dir, 'n_particles_max.npy'), n_particles_max)

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    if best_model != None:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    coeff_entropy_loss = train_config.coeff_entropy_loss
    entropy_loss = KoLeoLoss()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop} iterations per epoch')

    Niter = n_frames * data_augmentation_loop
    print(f'plot every {Niter // 20 // target_batch_size} iterations')

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    total_list_loss = []

    for epoch in range(start_epoch, n_epochs + 1):

        list_loss = []
        list_frame = []

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        print(f'batch_size: {batch_size}')
        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size
        time.sleep(1)

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 0
            optimizer.zero_grad()

            loss = torch.zeros(1, dtype=torch.float32, device=device)

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 2)
                x = x_list[run][k].clone().detach()
                edges = edge_p_p_list[run][f'arr_{k}']
                edges = torch.tensor(edges, dtype=torch.int64, device=device)
                dataset = data.Data(x=x[:, :], edge_index=edges)

                pred = model(dataset, data_id=run, training=True, vnorm=vnorm, phi=phi, has_field=False)

                if rotation_augmentation:
                    new_x = cos_phi * pred[:, 0] - sin_phi * pred[:, 1]
                    new_y = sin_phi * pred[:, 0] + cos_phi * pred[:, 1]
                    pred[:, 0] = new_x
                    pred[:, 1] = new_y

                if do_tracking:
                    x_next = x_list[run][k + 1]
                    x_pos_next = x_next[:, 1:3].clone().detach()
                    if model_config.prediction == '2nd_derivative':
                        x_pos_pred = (x[:, 1:3] + delta_t * (x[:, 3:5] + delta_t * pred * ynorm))
                    else:
                        x_pos_pred = (x[:, 1:3] + delta_t * pred * ynorm)
                    distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                    result = distance.min(dim=1)
                    min_value = result.values
                    indices = result.indices
                    loss += torch.sum(min_value) * 1E5
                else:
                    loss += (pred - y_batch).norm(2)

                # if coeff_entropy_loss > 0:
                #     idx = torch.randperm(len(model.a))
                #     model_a = model.a[idx[0:10000]].clone().detach()
                #     loss += coeff_entropy_loss * entropy_loss(model_a)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() / 1E5 / 6 / Niter
            list_loss.append(loss.item() / 1E5 / 6)
            list_frame.append(k)

            visualize_embedding = True
            if visualize_embedding & ((N % (Niter // 20) == 0) | (N == 0)):
                id_list = []
                frame_list = []
                for k in range(n_frames):
                    ids = x_list[0][k][:, -1]
                    frames = torch.ones_like(ids) * k
                    if k == 0:
                        id_list = ids
                        frame_list = frames
                    else:
                        id_list = torch.cat((id_list, ids), 0)
                        frame_list = torch.cat((frame_list, frames), 0)

                plot_training_mouse(config=config, id_list=id_list, frame_list=frame_list, dataset_name=dataset_name,
                                    log_dir=log_dir,
                                    epoch=epoch, N=N, model=model, n_particle_types=n_particle_types,
                                    type_stack=type_stack, ynorm=ynorm, cmap=cmap, device=device)

                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(1, 1, 1)
                plt.scatter(np.array(list_frame), np.array(list_loss), s=5, c='k', alpha=0.07, edgecolors='none')
                ax.set_yscale('log')
                plt.savefig(f"./{log_dir}/tmp_training/loss/loss_{epoch}.tif")
                plt.close()

                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 20,
                                   memory_percentage_threshold=0.6)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        total_list_loss.append(total_loss)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(total_list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/Loss.tif")
        plt.close()

        torch.save(total_list_loss, os.path.join(log_dir, 'loss.pt'))


def data_train_mesh(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    n_node_types = simulation_config.n_node_types
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    sparsity_freq = train_config.sparsity_freq

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    logger.info(f'Graph files N: {n_runs}')

    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(n_runs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()
    for run in range(n_runs):
        for k in range(n_frames):
            h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
    hnorm = torch.std(h)
    torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
    print(f'hnorm: {to_numpy(hnorm)}')
    logger.info(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    h = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_17.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x_mesh = x_mesh_list[1][n_frames - 1].clone().detach()
    type_list = x_mesh[:, 5:6].clone().detach()
    n_nodes = x_mesh.shape[0]
    print(f'N nodes: {n_nodes}')
    logger.info(f'N nodes: {n_nodes}')

    index_nodes = []
    x_mesh = x_mesh_list[1][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == n)
        index_nodes.append(index.squeeze())

    print("Start training mesh ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        old_batch_size = batch_size
        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        if epoch == 1:
            repeat_factor = batch_size // old_batch_size
            mask_mesh = mask_mesh.repeat(repeat_factor, 1)

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size
        if (batch_size == 1):
            Niter = Niter // 4

        for N in range(Niter):

            run = 1 + np.random.randint(n_runs - 1)

            dataset_batch = []
            for batch in range(batch_size):
                k = np.random.randint(n_frames - 1)
                x_mesh = x_mesh_list[run][k].clone().detach()

                if batch == 0:
                    data_id = torch.ones((x_mesh.shape[0],1), dtype=torch.int) * run
                else:
                    data_id = torch.cat((data_id, torch.ones((x_mesh.shape[0],1), dtype=torch.int) * run), dim = 0)

                if train_config.noise_level > 0:
                    x_mesh[:, 6:7] = x_mesh[:, 6:7] + train_config.noise_level * torch.randn_like(x_mesh[:, 6:7])
                dataset = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
                dataset_batch.append(dataset)
                y = y_mesh_list[run][k].clone().detach() / hnorm
                if batch == 0:
                    y_batch = y
                else:
                    y_batch = torch.cat((y_batch, y), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id = data_id)

            loss = ((pred - y_batch) * mask_mesh).norm(2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = ('Wave' in model_config.mesh_model_name)
            if visualize_embedding & (((epoch < 10) & (N % (Niter // 50) == 0)) | (N == 0)):
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

                plot_training(config=config, dataset_name=dataset_name,
                              log_dir=log_dir,
                              epoch=epoch, N=N, x=x_mesh, model=model, n_nodes=n_nodes, n_node_types=n_node_types,
                              index_nodes=index_nodes, dataset_num=1,
                              index_particles=[], n_particles=[],
                              n_particle_types=[], ynorm=ynorm, cmap=cmap, axis=True, device=device)

                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_nodes / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_nodes / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_nodes / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(22, 4))

        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 5, 2)
        embedding = get_embedding(model.a, 1)
        for n in range(n_node_types):
            plt.scatter(embedding[index_nodes[n], 0],
                        embedding[index_nodes[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        if (simulation_config.n_interactions < 100):

            ax = fig.add_subplot(1, 5, 3)
            func_list = []
            popt_list = []
            for n in range(n_nodes):
                embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                if model_config.mesh_model_name == 'RD_RPS_Mesh':
                    embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                    u = torch.tensor(np.linspace(0, 1, 100)).to(device)
                    u = u[:, None]
                    r = u
                    in_features = torch.cat((u, u, u, u, u, u, embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    h = h[:, 0]
                elif model_config.mesh_model_name == 'RD_RPS_Mesh_bis':
                    embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                    u = torch.tensor(np.linspace(0, 10, 100)).to(device)
                    u = u[:, None]
                    r = u
                    in_features = torch.cat((u, u, u, embedding_), dim=1)
                    h = model.lin_phi_L(in_features.float())
                    h = h[:, 0]
                else:
                    r = torch.tensor(np.linspace(-150, 150, 100)).to(device)
                    in_features = torch.cat((r[:, None], embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    popt, pcov = curve_fit(linear_model, to_numpy(r.squeeze()), to_numpy(h.squeeze()))
                    popt_list.append(popt)
                    h = h[:, 0]
                func_list.append(h)
                if (n % 24):
                    plt.plot(to_numpy(r),
                             to_numpy(h) * to_numpy(hnorm), linewidth=1,
                             color='k', alpha=0.05)
            func_list = torch.stack(func_list)
            coeff_norm = to_numpy(func_list)
            popt_list = np.array(popt_list)

            if 'RD_RPS_Mesh' in model_config.mesh_model_name:
                trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm)
                proj_interaction = trans.transform(coeff_norm)
            else:
                proj_interaction = popt_list
                proj_interaction[:, 1] = proj_interaction[:, 0]

            if (replace_with_cluster) & ((epoch + 1) % sparsity_freq == 0):

                labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction,
                                                                  embedding, train_config.cluster_distance_threshold,
                                                                  type_list, n_node_types, embedding_cluster)

                accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
                print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')
                logger.info(f'accuracy: {np.round(accuracy, 3)}    n_clusters: {n_clusters}')

                ax = fig.add_subplot(1, 5, 4)
                for n in np.unique(new_labels):
                    pos = np.array(np.argwhere(new_labels == n).squeeze().astype(int))
                    if pos.size > 0:
                        plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=5)
                plt.xlabel('proj 0', fontsize=12)
                plt.ylabel('proj 1', fontsize=12)
                plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top',
                         transform=ax.transAxes, fontsize=10)

                ax = fig.add_subplot(1, 5, 5)
                model_a_ = model.a[1].clone().detach()
                for n in range(n_clusters):
                    pos = np.argwhere(labels == n).squeeze().astype(int)
                    pos = np.array(pos)
                    if pos.size > 0:
                        median_center = model_a_[pos, :]
                        median_center = torch.median(median_center, dim=0).values
                        plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                        model_a_[pos, :] = median_center
                        plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=10, c='k')

                plt.xlabel('ai0', fontsize=12)
                plt.ylabel('ai1', fontsize=12)
                plt.xticks(fontsize=10.0)
                plt.yticks(fontsize=10.0)

                match train_config.sparsity:
                    case 'replace_embedding':
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
                        if train_config.fix_cluster_embedding:
                            lr_embedding = 1E-8
                            lr = train_config.learning_rate_end
                            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                            logger.info(f'Learning rates: {lr}, {lr_embedding}')
                    case 'replace_embedding_function':
                        logger.info(f'replace_embedding_function')
                        # Constrain function domain
                        y_func_list = func_list * 0
                        for n in range(n_nodes):
                            pos = np.argwhere(new_labels == n)
                            pos = pos.squeeze()
                            if pos.size > 0:
                                target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                                y_func_list[pos] = target_func
                        lr_embedding = 1E-8
                        lr = train_config.learning_rate_end
                        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                        for sub_epochs in range(20):
                            loss = 0
                            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                            pred = []
                            optimizer.zero_grad()
                            for n in range(n_nodes):
                                embedding_ = model.a[1, n, :].clone().detach() * torch.ones(
                                    (1000, model_config.embedding_dim), device=device)
                                match model_config.particle_model_name:
                                    case 'PDE_A':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                    case 'PDE_A_bis':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_),
                                            dim=1)
                                    case 'PDE_B':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_G':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_E':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                                pred.append(model.lin_edge(in_features.float()))

                            pred = torch.stack(pred)
                            loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                            logger.info(f'    loss: {np.round(loss.item() / n_nodes, 3)}')
                            loss.backward()
                            optimizer.step()
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
            else:
                if epoch > n_epochs - sparsity_freq:
                    lr_embedding = train_config.learning_rate_embedding_end
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                    lr = train_config.learning_rate_start
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()


def data_train_particle_field(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training particle field data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_siren = 'siren' in model_config.field_type
    has_siren_time = 'siren_with_time' in model_config.field_type
    has_cell_division = simulation_config.has_cell_division
    rotation_augmentation = train_config.rotation_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    n_ghosts = train_config.n_ghosts
    has_large_range = train_config.large_range
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    sparsity_freq = train_config.sparsity_freq

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    edge_p_p_list = []
    edge_p_f_list = []
    edge_f_f_list = []
    edge_f_p_list = []
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        edge_p_p = torch.load(f'graphs_data/graphs_{dataset_name}/edge_p_p_list{run}.pt', map_location=device)
        edge_f_p = torch.load(f'graphs_data/graphs_{dataset_name}/edge_f_p_list{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
        edge_p_p_list.append(edge_p_p)
        edge_f_p_list.append(edge_f_p)
    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(n_runs):
        for k in trange(n_frames):
            if (k % 10 == 0) | (n_frames < 1000):
                x = torch.cat((x, x_list[run][k].clone().detach()), 0)
                y = torch.cat((y, y_list[run][k].clone().detach()), 0)
        print(x_list[run][k].shape)
        time.sleep(0.5)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(n_runs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()
    for run in range(n_runs):
        for k in range(n_frames):
            h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
    hnorm = torch.std(h)
    torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
    print(f'hnorm: {to_numpy(hnorm)}')
    logger.info(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    x = []
    y = []
    h = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    if best_model != None:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x = x_list[1][n_frames - 1].clone().detach()
    n_particles = x.shape[0]
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')
    config.simulation.n_particles = n_particles

    if has_siren:

        image_width = int(np.sqrt(n_nodes))
        if has_siren_time:
            model_f = Siren_Network(image_width=image_width, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                    first_omega_0=80, hidden_omega_0=80.)
        else:
            model_f = Siren_Network(image_width=image_width, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80,
                                    hidden_omega_0=80.)
        model_f.to(device=device)
        model_f.train()
        optimizer_f = torch.optim.Adam(lr=1e-5, params=model_f.parameters())
        # net = f"./log/try_{config_file}/models/best_model_f_with_1_graphs_20.pt"
        # state_dict = torch.load(net, map_location=device)
        # model_f.load_state_dict(state_dict['model_state_dict'])

    if has_ghost:
        ghosts_particles = Ghost_Particles(config, n_particles, vnorm, device)
        optimizer_ghost_particles = torch.optim.Adam(lr=1e-4, params=ghosts_particles.parameters())

        mu = ghosts_particles.mu
        optimizer_ghost_particles = torch.optim.Adam([mu], lr=1e-4)
        var = ghosts_particles.var
        optimizer_ghost_particles.add_param_group({'params': [var], 'lr': 1e-4})

        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = []
    time.sleep(1)

    for epoch in range(n_epochs + 1):

        batch_size = get_batch_size(epoch)

        f_p_mask = []
        for k in range(batch_size):
            if k == 0:
                f_p_mask = np.zeros((n_nodes, 1))
                f_p_mask = np.concatenate((f_p_mask, np.ones((n_particles, 1))), axis=0)
            else:
                f_p_mask = np.concatenate((f_p_mask, np.zeros((n_nodes, 1))), axis=0)
                f_p_mask = np.concatenate((f_p_mask, np.ones((n_particles, 1))), axis=0)
        f_p_mask = np.argwhere(f_p_mask == 1)
        f_p_mask = f_p_mask[:, 0]

        logger.info(f'batch_size: {batch_size}')
        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(n_runs - 1)

            dataset_batch_p_p = []
            dataset_batch_f_p = []
            time_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 2)
                x = x_list[run][k].clone().detach()
                x_mesh = x_mesh_list[run][k].clone().detach()

                match model_config.field_type:
                    case 'tensor':
                        x_mesh[:, 6:7] = model.field[run]
                    case 'siren':
                        x_mesh[:, 6:7] = model_f() ** 2
                    case 'siren_with_time':
                        x_mesh[:, 6:7] = model_f(time=k / n_frames) ** 2
                x_particle_field = torch.concatenate((x_mesh, x), dim=0)

                if has_ghost:
                    x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                    if ghosts_particles.boids:
                        distance = torch.sum(
                            bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                        dist_np = to_numpy(distance)
                        ind_np = torch.min(distance, axis=1)[1]
                        x_ghost[:, 3:5] = x[ind_np, 3:5].clone().detach()
                    x = torch.cat((x, x_ghost), 0)

                    with torch.no_grad():
                        model.a[run, n_particles:n_particles + n_ghosts] = model.a[
                            run, ghosts_particles.embedding_index].clone().detach()  # sample ghost embedding

                edges = edge_p_p_list[run][k]
                dataset_p_p = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch_p_p.append(dataset_p_p)

                edges = edge_f_p_list[run][k]
                dataset_f_p = data.Data(x=x_particle_field[:, :], edge_index=edges)
                dataset_batch_f_p.append(dataset_f_p)

                y = y_list[run][k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)

                y = y / ynorm

                if rotation_augmentation:
                    new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                    new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                    y[:, 0] = new_x
                    y[:, 1] = new_y
                if batch == 0:
                    y_batch = y[:, 0:2]
                else:
                    y_batch = torch.cat((y_batch, y[:, 0:2]), dim=0)

                if has_ghost:
                    if batch == 0:
                        var_batch = torch.mean(ghosts_particles.var[run, k], dim=0)
                        var_batch = var_batch[:, None]
                    else:
                        var = torch.mean(ghosts_particles.var[run, k], dim=0)
                        var_batch = torch.cat((var_batch, var[:, None]), dim=0)

            batch_loader_p_p = DataLoader(dataset_batch_p_p, batch_size=batch_size, shuffle=False)
            batch_loader_f_p = DataLoader(dataset_batch_f_p, batch_size=batch_size, shuffle=False)

            optimizer.zero_grad()

            if has_siren:
                optimizer_f.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()

            for batch in batch_loader_f_p:
                pred_f_p = model(batch, data_id = run, training=True, vnorm=vnorm, phi=phi, has_field=True)
            for batch in batch_loader_p_p:
                pred_p_p = model(batch, data_id = run, training=True, vnorm=vnorm, phi=phi, has_field=False)

            pred_f_p = pred_f_p[f_p_mask]

            if has_ghost:
                loss = ((pred_p_p[mask_ghost] + 0 * pred_f_p - y_batch)).norm(2) + var_batch.mean() + model.field.norm(
                    2)
            else:
                loss = (pred_p_p + pred_f_p - y_batch).norm(2)  # + model.field.norm(2)

            loss.backward()
            optimizer.step()
            if has_siren:
                optimizer_f.step()
            if has_ghost:
                optimizer_ghost_particles.step()

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % (Niter // 50) == 0)) | (N == 0)):
                plot_training_particle_field(config=config, has_siren=has_siren, has_siren_time=has_siren_time,
                                             model_f=model_f, dataset_name=dataset_name, n_frames=n_frames,
                                             model_name=model_config.particle_model_name, log_dir=log_dir,
                                             epoch=epoch, N=N, x=x, x_mesh=x_mesh, model=model, n_nodes=0,
                                             n_node_types=0, index_nodes=0, dataset_num=1,
                                             index_particles=index_particles, n_particles=n_particles,
                                             n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True,
                                             device=device)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                if (has_siren):
                    torch.save({'model_state_dict': model_f.state_dict(),
                                'optimizer_state_dict': optimizer_f.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
            if ((epoch == 0) & (N % (Niter // 200) == 0)):
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                if (has_siren):
                    torch.save({'model_state_dict': model_f.state_dict(),
                                'optimizer_state_dict': optimizer_f.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if has_siren:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_f.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()},
                       os.path.join(log_dir, 'models', f'best_ghost_particles_with_{n_runs - 1}_graphs_{epoch}.pt'))

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(22, 4))
        # white background
        # plt.style.use('classic')

        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 5, 2)
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()

        ax = fig.add_subplot(1, 5, 3)
        func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                            model_MLP=model.lin_edge, model_a=model.a,
                                                            n_nodes=0,
                                                            dataset_number=1,
                                                            n_particles=n_particles, ynorm=ynorm,
                                                            type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                            cmap=cmap, dimension=dimension, device=device)

        labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                          train_config.cluster_distance_threshold, type_list,
                                                          n_particle_types, embedding_cluster)

        accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')
        logger.info(f'accuracy: {np.round(accuracy, 3)}    n_clusters: {n_clusters}')

        ax = fig.add_subplot(1, 5, 4)
        for n in np.unique(new_labels):
            pos = np.array(np.argwhere(new_labels == n).squeeze().astype(int))
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=5)
        plt.xlabel('proj 0', fontsize=12)
        plt.ylabel('proj 1', fontsize=12)
        plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top',
                 transform=ax.transAxes, fontsize=10)

        ax = fig.add_subplot(1, 5, 5)
        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                model_a_[pos, :] = median_center
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=10, c='k')
        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)

        if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):

            # Constrain embedding domain
            with torch.no_grad():
                model.a[1] = model_a_.clone().detach()
            print(f'regul_embedding: replaced')
            logger.info(f'regul_embedding: replaced')

            # Constrain function domain
            if train_config.sparsity == 'replace_embedding':

                logger.info(f'replace_embedding_function')
                y_func_list = func_list * 0

                ax, fig = fig_init()
                for n in np.unique(new_labels):
                    pos = np.argwhere(new_labels == n)
                    pos = pos.squeeze()
                    if pos.size > 0:
                        target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                        y_func_list[pos] = target_func
                    plt.plot(to_numpy(target_func) * to_numpy(ynorm), linewidth=2, alpha=1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}_before training function.tif")

                lr_embedding = 1E-12
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                for sub_epochs in range(20):
                    loss = 0
                    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                    pred = []
                    optimizer.zero_grad()
                    for n in range(n_particles):
                        embedding_ = model.a[1, n, :].clone().detach() * torch.ones(
                            (1000, model_config.embedding_dim), device=device)
                        match model_config.particle_model_name:
                            case 'PDE_ParticleField_A':
                                in_features = torch.cat(
                                    (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                     rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                            case 'PDE_ParticleField_B':
                                in_features = torch.cat(
                                    (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                     rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                        pred.append(model.lin_edge(in_features.float()))
                    pred = torch.stack(pred)
                    loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                    logger.info(f'    loss: {np.round(loss.item() / n_particles, 3)}')
                    loss.backward()
                    optimizer.step()

            if train_config.fix_cluster_embedding:
                lr_embedding = 1E-12
            else:
                lr_embedding = train_config.learning_rate_embedding_start
            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
            logger.info(f'Learning rates: {lr}, {lr_embedding}')

        else:
            if epoch > n_epochs - sparsity_freq:
                lr_embedding = train_config.learning_rate_embedding_end
                lr = train_config.learning_rate_end
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')
            else:
                lr_embedding = train_config.learning_rate_embedding_start
                lr = train_config.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')


def data_train_potential_energy(config, config_file, erase, best_model, device):
    model = SIREN(in_features=1, out_features=1, hidden_features=256, hidden_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(10000):
        x = torch.linspace(-10, 10, 1000).unsqueeze(1)
        U_pred = model(x)
        F_pred = -torch.autograd.grad(U_pred.sum(), x, create_graph=True)[0]
        F_true = 2 * x + 2

        loss = criterion(F_pred, F_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


def data_train_synaptic(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    n_particle_types = simulation_config.n_particle_types
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    data_augmentation_loop = train_config.data_augmentation_loop
    recursive_loop = train_config.recursive_loop
    target_batch_size = train_config.batch_size
    delta_t = simulation_config.delta_t
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    embedding_cluster = EmbeddingCluster(config)
    cmap = CustomColorMap(config=config)
    n_runs = train_config.n_runs
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    has_siren = 'siren' in model_config.excitation_type
    has_siren_time = 'siren_with_time' in model_config.excitation_type

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')

    x_list = []
    y_list = []
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    if best_model != None:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)

    model.train()

    table = PrettyTable(["Modules", "Parameters"])
    n_total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        n_total_params += param
    print(table)
    print(f"Total Trainable Params: {n_total_params}")
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x = x_list[1][n_frames - 1].clone().detach()
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles
    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        else:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    type_list = get_type_list(x, dimension)

    if has_siren:
        im = imread(f"graphs_data/{simulation_config.excitation_value_map}")
        image_width = im.shape[1]
        if has_siren_time:
            model_f = Siren_Network(image_width=image_width, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                    first_omega_0=80, hidden_omega_0=80.)
        else:
            model_f = Siren_Network(image_width=image_width, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80,
                                    hidden_omega_0=80.)
        model_f.to(device=device)
        model_f.train()
        optimizer_exc = torch.optim.Adam(lr=train_config.learning_rate_NNR, params=model_f.parameters())

        # net = f"./log/try_{config_file}/models/best_model_f_with_9_graphs_14.pt"
        # state_dict = torch.load(net, map_location=device)
        # model_f.load_state_dict(state_dict['model_state_dict'])

        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
        logger.info(f'Learning rates: {lr}, {lr_embedding}')
        print(f'has_siren, learning rates: {lr}, {lr_embedding}')

    else:
        model_f = []

    model.edges = torch.load(f'./graphs_data/graphs_{dataset_name}/edge_index.pt', map_location=device)
    adjacency = torch.load(f'./graphs_data/graphs_{dataset_name}/adjacency.pt', map_location=device)

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    Niter = int(n_frames * data_augmentation_loop // batch_size * n_runs / 10)
    print(f'plot every {Niter // 100} iterations')

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    time.sleep(2)
    for epoch in range(14, n_epochs + 1):

        if (epoch == 20) & (train_config.coeff_anneal_L1 > 0):
            train_config.coeff_L1 = train_config.coeff_anneal_L1
            logger.info(f'coeff_L1: {train_config.coeff_L1}')

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        total_loss = 0

        Niter = int(n_frames * data_augmentation_loop // batch_size * n_runs / 10)
        print(f'Niter = {Niter}')
        logger.info(f'Niter = {Niter}')

        excitation = torch.zeros((n_particles, 1), device=device)

        for N in trange(Niter):

            run = 1 + np.random.randint(n_runs - 1)
            k = np.random.randint(n_frames - 6)

            optimizer.zero_grad()

            for batch in range(batch_size):

                if ('PDE_N2' in config.graph_model.signal_model_name) | (
                        'PDE_N5' in config.graph_model.signal_model_name):
                    in_features = torch.zeros((n_particles, 1), device=device)
                    func_phi = model.lin_phi(in_features.float())
                    func_edge = model.lin_edge(in_features.float())
                elif 'PDE_N3' in config.graph_model.signal_model_name:
                    in_features = torch.cat((torch.zeros((n_particles, 1), device=device), model.a[1, :]), dim=1)
                    func_phi = model.lin_phi(in_features.float())
                    in_features = torch.zeros((n_particles, 1), device=device)
                    func_edge = model.lin_edge(in_features.float())
                    u = x_list[run][k].clone().detach()
                    u = u[:, 6:7]
                    diff = torch.relu(model.lin_edge(u) - model.lin_edge(u + 0.1)).norm(2)
                else:
                    func_edge = model.lin_edge(torch.zeros(1, device=device))
                    in_features = torch.cat((torch.zeros((n_particles, 1), device=device), model.a[1, :]), dim=1)
                    func_phi = model.lin_phi(in_features.float())

                match recursive_loop:

                    case 1:
                        x = x_list[run][k].clone().detach()
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred = model(dataset, data_id=run, has_field=has_field)
                        y = y_list[run][k].clone().detach()
                        y = y / ynorm

                        loss = (pred - y).norm(2) + model.W.norm(1) * train_config.coeff_L1 + func_phi.norm(
                            2) + func_edge.norm(2) + diff * 10 * (epoch == 0)

                    case 2:
                        x = x_list[run][k].clone().detach()
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred1 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += pred1 * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred2 = model(dataset, data_id=run, has_field=has_field)
                        y1 = y_list[run][k].clone().detach() / ynorm
                        y2 = y_list[run][k + 1].clone().detach() / ynorm

                        loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + model.W.norm(
                            1) * train_config.coeff_L1 + func_phi.norm(2) + func_edge.norm(2) + diff * 10 * (epoch == 0)

                    case 3:
                        x = x_list[run][k].clone().detach()
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred1 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += pred1 * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred2 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred3 = model(dataset, data_id=run, has_field=has_field)

                        y1 = y_list[run][k].clone().detach() / ynorm
                        y2 = y_list[run][k + 1].clone().detach() / ynorm
                        y3 = y_list[run][k + 2].clone().detach() / ynorm

                        loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + (pred3 - y3).norm(2) + model.W.norm(
                            1) * train_config.coeff_L1 + func_phi.norm(2) + func_edge.norm(2) + diff * 10 * (epoch == 0)

                    case 5:

                        x = x_list[run][k].clone().detach()
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred1 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += pred1 * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred2 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred3 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2 + pred3) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred4 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2 + pred3 + pred4) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred5 = model(dataset, data_id=run, has_field=has_field)

                        y1 = y_list[run][k].clone().detach() / ynorm
                        y2 = y_list[run][k + 1].clone().detach() / ynorm
                        y3 = y_list[run][k + 2].clone().detach() / ynorm
                        y4 = y_list[run][k + 3].clone().detach() / ynorm
                        y5 = y_list[run][k + 4].clone().detach() / ynorm

                        loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + (pred3 - y3).norm(2) + (pred4 - y4).norm(
                            2) + (pred5 - y5).norm(2) + model.W.norm(1) * train_config.coeff_L1 + func_f.norm(2)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % (Niter // 50) == 0)) | (N == 0)):
                plot_training_signal(config, dataset_name, model, adjacency, ynorm, log_dir, epoch, N, index_particles,
                                     n_particles, n_particle_types, type_list, cmap, has_siren, has_siren_time, model_f,
                                     n_frames, device)
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50,
                                   memory_percentage_threshold=0.6)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if has_siren:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_exc.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))
            list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(20, 8))

        ax = fig.add_subplot(2, 5, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(2, 5, 2)
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
        if 'asymmetric' in config.simulation.adjacency_matrix:
            A = model.vals
        else:
            A = torch.zeros(n_particles, n_particles, device=device, requires_grad=False, dtype=torch.float32)
            A[i, j] = model.vals
            A.T[i, j] = model.vals

        ax = fig.add_subplot(2, 5, 3)
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.001, vmax=0.001)
        # ax.invert_yaxis()
        plt.title('True connectivity matrix', fontsize=12)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        ax = fig.add_subplot(2, 5, 4)
        ax = sns.heatmap(to_numpy(A), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-1, vmax=1)
        # ax.invert_yaxis()
        plt.title('Learned connectivity matrix', fontsize=12)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=8)

        ax = fig.add_subplot(2, 5, 5)
        gt_weight = to_numpy(adjacency)
        pred_weight = to_numpy(A)
        plt.scatter(gt_weight, pred_weight, s=0.1, c='k', alpha=0.01)
        plt.xlabel('true weight', fontsize=12)
        plt.ylabel('learned weight', fontsize=12)
        plt.title('comparison')

        x_data = gt_weight.flatten()
        y_data = pred_weight.flatten()
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')

        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')
        plt.text(-1, 100, f'R^2: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}', fontsize=10)

        ax = fig.add_subplot(2, 5, 6)

        func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                            model_MLP=model.lin_phi, model_a=model.a,
                                                            n_nodes=0,
                                                            dataset_number=1,
                                                            n_particles=n_particles, ynorm=ynorm,
                                                            type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                            cmap=cmap, dimension=dimension, device=device)

        labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                          train_config.cluster_distance_threshold, type_list,
                                                          n_particle_types, embedding_cluster)

        accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')
        logger.info(f'accuracy: {np.round(accuracy, 3)}    n_clusters: {n_clusters}')

        ax = fig.add_subplot(2, 5, 7)
        for n in np.unique(new_labels):
            pos = np.array(np.argwhere(new_labels == n).squeeze().astype(int))
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=5)
        plt.xlabel('proj 0', fontsize=12)
        plt.ylabel('proj 1', fontsize=12)
        plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top',
                 transform=ax.transAxes, fontsize=10)

        ax = fig.add_subplot(2, 5, 8)
        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                model_a_[pos, :] = median_center
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=10, c='k')

        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)

        ax = fig.add_subplot(2, 5, 9)
        plt.hist(np.abs(to_numpy(A)), bins=100)

        if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):
            # Constrain embedding domain
            with torch.no_grad():
                model.a[1] = model_a_.clone().detach()
            print(f'regul_embedding: replaced')
            logger.info(f'regul_embedding: replaced')

            # Constrain function domain
            if train_config.sparsity == 'replace_embedding_function':

                logger.info(f'replace_embedding_function')
                y_func_list = func_list * 0

                ax = fig.add_subplot(2, 5, 9)
                for n in np.unique(new_labels):
                    pos = np.argwhere(new_labels == n)
                    pos = pos.squeeze()
                    if pos.size > 0:
                        target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                        y_func_list[pos] = target_func
                    plt.plot(to_numpy(target_func) * to_numpy(ynorm), linewidth=2, alpha=1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()

                lr_embedding = 1E-12
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                for sub_epochs in range(20):
                    rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
                    pred = []
                    optimizer.zero_grad()
                    for n in range(n_particles):
                        embedding_ = model.a[1, n, :].clone().detach() * torch.ones((1000, model_config.embedding_dim),
                                                                                    device=device)
                        in_features = get_in_features(rr, embedding_, config.graph_model.signal_model_name,
                                                      config.simulation.max_radius)
                        pred.append(model.lin_phi(in_features.float()))
                    pred = torch.stack(pred)
                    loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                    logger.info(f'    loss: {np.round(loss.item() / n_particles, 3)}')
                    loss.backward()
                    optimizer.step()

            if train_config.fix_cluster_embedding:
                lr_embedding = 1E-12
            else:
                lr_embedding = train_config.learning_rate_embedding_start
            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
            logger.info(f'Learning rates: {lr}, {lr_embedding}')
        else:
            if epoch > n_epochs - sparsity_freq:
                lr_embedding = train_config.learning_rate_embedding_end
                lr = train_config.learning_rate_end
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')
            else:
                lr_embedding = train_config.learning_rate_embedding_start
                lr = train_config.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()


def data_train_synaptic2(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training with data {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    n_particle_types = simulation_config.n_particle_types
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    data_augmentation_loop = train_config.data_augmentation_loop
    omega = model_config.omega
    recursive_loop = train_config.recursive_loop
    target_batch_size = train_config.batch_size
    delta_t = simulation_config.delta_t
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    embedding_cluster = EmbeddingCluster(config)
    cmap = CustomColorMap(config=config)
    n_runs = train_config.n_runs
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = True
    else:
        has_field = False

    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Loading graph files N: {n_runs} ...')
    logger.info(f'Graph files N: {n_runs}')

    x_list = []
    y_list = []
    for run in trange(n_runs):
        x = np.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    x = x_list[0][n_frames - 1]
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)
    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    if has_field:
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)

    if best_model != None:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')

        if has_field:
            net = f'./log/try_{config_file}/models/best_model_f_with_{n_runs - 1}_graphs_{best_model}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])

    else:
        start_epoch = 0

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    model.train()

    if has_field:
        optimizer_f = torch.optim.Adam(lr=train_config.learning_rate_NNR, params=model_f.parameters())
        model_f.train()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    adjacency = torch.load(f'./graphs_data/graphs_{dataset_name}/adjacency.pt', map_location=device)
    model.edges = torch.load(f'./graphs_data/graphs_{dataset_name}/edge_index.pt', map_location=device)
    print(f'{to_numpy(torch.sum(model.edges))} edges')

    if simulation_config.connectivity_mask:
        model.mask = model.mask * (adjacency != 0) * 1.0
        if simulation_config.connectivity_filling_factor > 0:
            supp = (torch.rand(adjacency.shape, device=device) < simulation_config.connectivity_filling_factor) * 1.0
            model.mask = torch.max(model.mask, supp)

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    Niter = int(n_frames * data_augmentation_loop // batch_size * n_runs / 10)

    print(f'plot every {Niter // 100} iterations')

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    time.sleep(2)
    for epoch in range(start_epoch, n_epochs + 1):

        if (epoch < train_config.n_epochs_init):
            coeff_L1 = train_config.first_coeff_L1
            coeff_diff = train_config.coeff_diff
        else:
            coeff_L1 = train_config.coeff_L1
            coeff_diff = 0
        logger.info(f'coeff_L1: {coeff_L1} coeff_diff: {coeff_diff}')

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        total_loss = 0

        Niter = int(n_frames * data_augmentation_loop // batch_size * n_runs / 10)
        print(f'Niter = {Niter}')
        logger.info(f'Niter = {Niter}')

        for N in trange(Niter):

            run = np.random.randint(n_runs)
            k = np.random.randint(n_frames - 5)

            if has_field:
                optimizer_f.zero_grad()
            optimizer.zero_grad()

            for batch in range(batch_size):

                in_features = torch.cat((torch.zeros((n_particles, 1), device=device), model.a), dim=1)
                func_phi = model.lin_phi(in_features.float())
                x = torch.tensor(x_list[run][k], device=device)
                if model_config.signal_model_name == 'PDE_N4':
                    in_features = torch.zeros((n_particles, dimension + 1), device=device)
                    func_edge = model.lin_edge(in_features.float())
                    in_features = torch.cat((x[:, 6:7], model.a), dim=1)
                    in_features_next = torch.cat((x[:, 6:7] + 0.1, model.a), dim=1)
                    diff = torch.relu(model.lin_edge(in_features) - model.lin_edge(in_features_next)).norm(2)
                elif model_config.signal_model_name == 'PDE_N5':
                    in_features = torch.zeros((n_particles, 2 * dimension + 1), device=device)
                    func_edge = model.lin_edge(in_features.float())
                    in_features = torch.cat((x[:, 6:7], model.a, model.a), dim=1)
                    in_features_next = torch.cat((x[:, 6:7] + 0.1, model.a, model.a), dim=1)
                    diff = torch.relu(model.lin_edge(in_features) - model.lin_edge(in_features_next)).norm(2)
                else:
                    in_features = torch.zeros((n_particles, 1), device=device)
                    func_edge = model.lin_edge(in_features.float())
                    diff = torch.relu(model.lin_edge(x[:, 6:7]) - model.lin_edge(x[:, 6:7] + 0.1)).norm(2)

                if has_field:
                    if 'visual' in field_type:
                        x[:n_nodes, 8:9] = model_f(time=k / n_frames) ** 2
                        x[n_nodes:n_particles, 8:9] = 1
                    else:
                        x[:, 8:9] = model_f(time=k / n_frames) ** 2

                match recursive_loop:
                    case 1:
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred = model(dataset, data_id=run, has_field=has_field)
                        y = torch.tensor(y_list[run][k], device=device) / ynorm
                        loss = (pred - y).norm(2) + model.W.norm(1) * coeff_L1 + func_phi.norm(2) + func_edge.norm(
                            2) + diff * coeff_diff
                    case 2:
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred1 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += pred1 * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred2 = model(dataset, data_id=run, has_field=has_field)
                        y1 = torch.tensor(y_list[run][k], device=device) / ynorm
                        y2 = torch.tensor(y_list[run][k + 1], device=device) / ynorm
                        loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + model.W.norm(1) * coeff_L1 + func_phi.norm(
                            2) + func_edge.norm(2) + diff * coeff_diff
                    case 3:
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred1 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += pred1 * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred2 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred3 = model(dataset, data_id=run, has_field=has_field)
                        y1 = torch.tensor(y_list[run][k], device=device) / ynorm
                        y2 = torch.tensor(y_list[run][k + 1], device=device) / ynorm
                        y3 = torch.tensor(y_list[run][k + 2], device=device) / ynorm
                        loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + (pred3 - y3).norm(2) + model.W.norm(
                            1) * coeff_L1 + func_phi.norm(2) + func_edge.norm(2) + diff * coeff_diff
                    case 5:
                        dataset = data.Data(x=x, edge_index=model.edges)
                        pred1 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += pred1 * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred2 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred3 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2 + pred3) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred4 = model(dataset, data_id=run, has_field=has_field)
                        x_ = x.clone().detach()
                        x_[:, 6:7] += (pred1 + pred2 + pred3 + pred4) * delta_t
                        dataset = data.Data(x=x_, edge_index=model.edges)
                        pred5 = model(dataset, data_id=run, has_field=has_field)
                        y1 = y_list[run][k].clone().detach() / ynorm
                        y2 = y_list[run][k + 1].clone().detach() / ynorm
                        y3 = y_list[run][k + 2].clone().detach() / ynorm
                        y4 = y_list[run][k + 3].clone().detach() / ynorm
                        y5 = y_list[run][k + 4].clone().detach() / ynorm

                        loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + (pred3 - y3).norm(2) + (pred4 - y4).norm(
                            2) + (pred5 - y5).norm(2) + model.W.norm(1) * coeff_L1 + func_f.norm(2)

            loss.backward()

            optimizer.step()
            if has_field:
                optimizer_f.step()

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % (Niter // 50) == 0)) | (N == 0)):
                plot_training_signal(config, dataset_name, model, adjacency, ynorm, log_dir, epoch, N, n_particles,
                                     n_particle_types, type_list, cmap, device)
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                if (has_field):
                    if 'visual' in field_type:
                        tmp = torch.reshape(x[:n_nodes, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
                    else:
                        tmp = torch.reshape(x[:, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
                    tmp = to_numpy(torch.sqrt(tmp))
                    tmp = np.rot90(tmp, k=1)
                    fig = plt.figure(figsize=(12, 12))
                    plt.imshow(tmp, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.tif", dpi=80)
                    plt.close()
                    torch.save({'model_state_dict': model_f.state_dict(),
                                'optimizer_state_dict': optimizer_f.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if has_field:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_f.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(20, 8))

        ax = fig.add_subplot(2, 5, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(2, 5, 2)
        for n in range(n_particle_types):
            pos = torch.argwhere(type_list == n).squeeze()
            plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=0.1, color=cmap.color(n))
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        A = model.W.clone().detach() * model.mask.clone().detach()

        ax = fig.add_subplot(2, 5, 3)
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.001, vmax=0.001)
        plt.title('True connectivity matrix', fontsize=12)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        ax = fig.add_subplot(2, 5, 4)
        ax = sns.heatmap(to_numpy(A), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-1, vmax=1)
        plt.title('Learned connectivity matrix', fontsize=12)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=8)

        plt.tight_layout()

        ax = fig.add_subplot(2, 5, 5)
        gt_weight = to_numpy(adjacency)
        pred_weight = to_numpy(A)
        plt.scatter(gt_weight, pred_weight, s=0.1, c='k', alpha=0.01)
        plt.xlabel('true weight', fontsize=12)
        plt.ylabel('learned weight', fontsize=12)
        plt.title('comparison')

        x_data = gt_weight.flatten()
        y_data = pred_weight.flatten()
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')
        logger.info(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')

        ax = fig.add_subplot(2, 5, 6)
        embedding = to_numpy(model.a.squeeze())

        if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
            model_MLP = model.lin_edge
        else:
            model_MLP = model.lin_phi

        func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                            model_MLP=model_MLP, model_a=model.a,
                                                            n_nodes=0,
                                                            dataset_number=1,
                                                            n_particles=n_particles, ynorm=ynorm,
                                                            type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                            cmap=cmap, dimension=dimension, device=device)

        labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                          train_config.cluster_distance_threshold, type_list,
                                                          n_particle_types, embedding_cluster)

        accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')
        logger.info(f'accuracy: {np.round(accuracy, 3)}    n_clusters: {n_clusters}')

        ax = fig.add_subplot(2, 5, 7)
        for n in np.unique(new_labels):
            pos = np.array(np.argwhere(new_labels == n).squeeze().astype(int))
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=5)
        plt.xlabel('proj 0', fontsize=12)
        plt.ylabel('proj 1', fontsize=12)
        plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top',
                 transform=ax.transAxes, fontsize=10)

        ax = fig.add_subplot(2, 5, 8)
        model_a_ = model.a.clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            if pos.size > 0:
                pos = np.array(pos)
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                model_a_[pos, :] = median_center
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=10, c='k')
        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)

        if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):
            # Constrain embedding domain

            model.a.values = model_a_.clone().detach()
            print(f'regul_embedding: replaced')
            logger.info(f'regul_embedding: replaced')

            # Constrain function domain
            if train_config.sparsity == 'replace_embedding_function':

                logger.info(f'replace_embedding_function')
                y_func_list = func_list * 0

                ax = fig.add_subplot(2, 5, 9)
                for n in np.unique(new_labels):
                    pos = np.argwhere(new_labels == n)
                    pos = pos.squeeze()
                    if pos.size > 0:
                        target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                        y_func_list[pos] = target_func
                    plt.plot(to_numpy(target_func) * to_numpy(ynorm), linewidth=2, alpha=1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()

                lr_embedding = 1E-12
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                for sub_epochs in range(20):
                    rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
                    pred = []
                    optimizer.zero_grad()
                    for n in range(n_particles):
                        embedding_ = model.a[n, :].clone().detach() * torch.ones((1000, model_config.embedding_dim),
                                                                                 device=device)
                        in_features = get_in_features(rr, embedding_, config.graph_model.signal_model_name,
                                                      config.simulation.max_radius)
                        if (model_config.signal_model_name == 'PDE_N4'):
                            pred.append(model.lin_edge(in_features.float()))
                        else:
                            pred.append(model.lin_phi(in_features.float()))
                    pred = torch.stack(pred)
                    loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                    logger.info(f'    loss: {np.round(loss.item() / n_particles, 3)}')
                    loss.backward()
                    optimizer.step()
            if train_config.fix_cluster_embedding:
                lr_embedding = 1E-12
            else:
                lr_embedding = train_config.learning_rate_embedding_start
            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
            logger.info(f'Learning rates: {lr}, {lr_embedding}')
        else:
            if epoch > n_epochs - sparsity_freq:
                lr_embedding = train_config.learning_rate_embedding_end
                lr = train_config.learning_rate_end
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')
            else:
                lr_embedding = train_config.learning_rate_embedding_start
                lr = train_config.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')
        if (epoch == 20) & (train_config.coeff_anneal_L1 > 0):
            coeff_L1 = train_config.coeff_anneal_L1
            logger.info(f'coeff_L1: {coeff_L1}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()


def data_train_agents(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    rotation_augmentation = train_config.rotation_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    has_ghost = train_config.n_ghosts > 0
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    has_state = (config.simulation.state_type != 'discrete')

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_3.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    print('Load data ...')

    time_series, signal = load_agent_data(dataset_name, device=device)

    velocities = [t.velocity for t in time_series]
    velocities.pop(0)  # the first element is always NaN
    velocities = torch.stack(velocities)
    if torch.any(torch.isnan(velocities)):
        raise ValueError('Discovered NaN in velocities. Aborting.')
    velocities = bc_dpos(velocities)

    if model_config.prediction == 'first_derivative':
        vnorm = torch.std(velocities[:, :, 0]) / 10
        ynorm = vnorm
    else:
        vnorm = torch.std(velocities[:, :, 0]) / 10
        ynorm = vnorm / 10

    positions = torch.stack([t.pos for t in time_series])
    min = torch.min(positions[:, :, 0])
    max = torch.max(positions[:, :, 0])
    mean = torch.mean(positions[:, :, 0])
    std = torch.std(positions[:, :, 0])
    print(f"min: {min}, max: {max}, mean: {mean}, std: {std}")

    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    x = []
    y = []

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{config_file}/models/best_model_with_1_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    n_particles = config.simulation.n_particles
    print(f'N particles: {n_particles}')
    logger.info(f'N particles:  {n_particles}')

    if os.path.exists(f'./log/try_{config_file}/edge_p_p_list.npz'):
        print('Load list of edges index ...')
        edge_p_p_list = np.load(f'./log/try_{config_file}/edge_p_p_list.npz')
    else:
        print('Create list of edges index ...')
        edge_p_p_list = []
        for k in trange(n_frames):
            time_point = time_series[k]
            x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
            x = torch.column_stack((torch.arange(0, n_particles, device=device), x))

            nbrs = NearestNeighbors(n_neighbors=simulation_config.n_neighbors, algorithm='auto').fit(
                to_numpy(x[:, 1:dimension + 1]))
            distances, indices = nbrs.kneighbors(to_numpy(x[:, 1:dimension + 1]))
            edge_index = []
            for i in range(indices.shape[0]):
                for j in range(1, indices.shape[1]):  # Start from 1 to avoid self-loop
                    edge_index.append((i, indices[i, j]))
            edge_index = np.array(edge_index)
            edge_index = torch.tensor(edge_index, device=device).t().contiguous()
            edge_p_p_list.append(to_numpy(edge_index))
        np.savez(f'./log/try_{config_file}/edge_p_p_list', *edge_p_p_list)

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    Niter = n_frames * data_augmentation_loop // batch_size
    print(f'plot every {Niter // 50} iterations')

    list_loss = []
    time.sleep(1)

    for epoch in range(n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            dataset_batch = []
            for batch in range(batch_size):

                k = np.random.randint(2, n_frames - 2)

                time_point = time_series[k]
                x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
                x = torch.column_stack((torch.arange(0, n_particles, device=device), x))
                x[:, 1:5] = x[:, 1:5] / 1000

                edges = edge_p_p_list[f'arr_{k}']
                edges = torch.tensor(edges, dtype=torch.int64, device=device)
                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)

                if model_config.prediction == 'first_derivative':
                    time_point = time_series[k + 1]
                    y = bc_dpos(time_point.velocity.clone().detach() / 1000)
                else:
                    time_point = time_series[k + 1]
                    v_prev = bc_dpos(time_point.velocity.clone().detach() / 1000)
                    time_point = time_series[k - 1]
                    v_next = bc_dpos(time_point.velocity.clone().detach() / 1000)
                    y = (v_next - v_prev)

                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)

                y = y / ynorm

                if rotation_augmentation:
                    new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                    new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                    y[:, 0] = new_x
                    y[:, 1] = new_y
                if batch == 0:
                    y_batch = y[:, 0:2]
                else:
                    y_batch = torch.cat((y_batch, y[:, 0:2]), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=0, training=True, vnorm=vnorm, phi=phi, frame=k)

            loss = (pred - y_batch).norm(2)

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % (Niter // 50) == 0)) | (N == 0)):

                if has_state:
                    ax, fig = fig_init()
                    embedding = torch.reshape(model.a[0], (n_particles * n_frames, model_config.embedding_dim))
                    plt.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), s=0.1, alpha=0.01, c='k')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{epoch}_{N}.tif", dpi=87)
                else:
                    ax, fig = fig_init()
                    embedding = model.a[0]
                    # plt.hist(to_numpy(embedding[:, 0]), bins=1000)
                    plt.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), s=1, alpha=0.1, c='k')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{epoch}_{N}.tif", dpi=87)

                fig, ax = fig_init()
                plt.scatter(to_numpy(y[:, 0]), to_numpy(pred[:, 0]), s=0.1, c='k', alpha=0.1)
                # plt.scatter(to_numpy(y[:, 1]), to_numpy(pred[:, 1]), s=0.1, alpha=0.1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/particle/Fig_{epoch}_{N}.tif", dpi=87)

                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))


def data_train_WBI(config, config_file, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    rotation_augmentation = train_config.rotation_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    has_state = (config.simulation.state_type != 'discrete')

    l_dir, log_dir, logger = create_log_dir(config, config_file, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(n_runs):
        for k in trange(n_frames - 2):
            if (k % 10 == 0) | (n_frames < 1000):
                x = torch.cat((x, x_list[run][k].clone().detach()), 0)
                y = torch.cat((y, y_list[run][k].clone().detach()), 0)
        print(x_list[run][k].shape)
        time.sleep(0.5)

    ynorm = torch.std(y)

    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    logger.info(f'ynorm: {to_numpy(ynorm)}')

    x = []
    y = []

    print('Create GNN model ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_0_0.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    x = x_list[0][0].clone().detach()
    n_particles = x.shape[0]
    config.simulation.n_particles = n_particles
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')

    print('Load local connectivity ...')
    edges = torch.load(f'./graphs_data/graphs_{dataset_name}/edge_index.pt', map_location=device)
    print('Local connectivity loaded ...')

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    Niter = n_frames * data_augmentation_loop // batch_size
    print(f'plot every {Niter // 50} iterations')

    list_loss = []
    time.sleep(1)

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    for epoch in range(n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in range(Niter):

            run = 0

            dataset_batch = []
            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)

                x = x_list[run][k].clone().detach()

                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)

                y = y_list[run][k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)

                y = y / ynorm

                if rotation_augmentation:
                    new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                    new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                    y[:, 0] = new_x
                    y[:, 1] = new_y
                if batch == 0:
                    y_batch = y[:, 0:1]
                else:
                    y_batch = torch.cat((y_batch, y[:, 0:1]), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=run)

            loss = (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % (Niter // 50) == 0)) | (N == 0)):
                embedding = get_embedding(model.a, 0)
                fig = plt.figure(figsize=(8, 8))
                plt.scatter(embedding[:, 0], embedding[:, 1], s=1, c=to_numpy(type_list[:, 0]), cmap='tab20', vmin=0,
                            vmax=255)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif", dpi=87)
                plt.close()

                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50,
                                   memory_percentage_threshold=0.6)

            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(22, 4))
        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15,
              ratio=1, run=1, plot_data=False, test_simulation=False, sample_embedding=False, fixed=False, bounce=False, time_ratio=1, device=[]):
    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_ghost = config.training.n_ghosts > 0
    has_bounding_box = 'PDE_F' in model_config.particle_model_name
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames * time_ratio
    delta_t = simulation_config.delta_t / time_ratio
    time_window = training_config.time_window
    smooth_particle = training_config.smooth_particle

    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    dimension = simulation_config.dimension
    has_siren_time = 'siren_with_time' in model_config.field_type
    has_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)

    do_tracking = training_config.do_tracking
    has_state = (config.simulation.state_type != 'discrete')
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_synaptic_field = True
    else:
        has_synaptic_field = False

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"

    n_sub_population = n_particles // n_particle_types

    first_cell_id_particles = []
    for n in range(n_particle_types):
        index = np.arange(n_particles * n // n_particle_types, n_particles * (n + 1) // n_particle_types)
        first_cell_id_particles.append(index)

    print('load data ...')
    if only_mesh:
        vnorm = torch.tensor(1.0, device=device)
        ynorm = torch.tensor(1.0, device=device)
        hnorm = torch.load(f'./log/try_{config_file}/hnorm.pt', map_location=device).to(device)
        x_mesh_list = []
        y_mesh_list = []
        time.sleep(0.5)
        for run in trange(n_runs):
            x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
            x_mesh_list.append(x_mesh)
            h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
            y_mesh_list.append(h)
        h = y_mesh_list[0][0].clone().detach()
        x_list = x_mesh_list
        y_list = y_mesh_list
        x = x_list[run][0].clone().detach()
    elif has_field:
        x_list = []
        y_list = []
        x_mesh_list = []
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        hnorm = torch.load(f'./log/try_{config_file}/hnorm.pt', map_location=device).to(device)
        x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device))
        y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device))
        ynorm = torch.load(f'./log/try_{config_file}/ynorm.pt', map_location=device).to(device)
        vnorm = torch.load(f'./log/try_{config_file}/vnorm.pt', map_location=device).to(device)
        x = x_list[0][0].clone().detach()
        n_particles = x.shape[0]
        config.simulation.n_particles = n_particles
        index_particles = get_index_particles(x, n_particle_types, dimension)
        x_mesh = x_mesh_list[0][0].clone().detach()
    else:
        x_list = []
        y_list = []
        if (model_config.signal_model_name == 'PDE_N'):
            x = np.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.npy')
            x_list.append(torch.tensor(x, device=device))
            y_list.append(torch.tensor(y, device=device))
        else:
            if os.path.exists(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt'):
                x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
                y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
                x_list.append(x)
                y_list.append(y)
            else:
                x = np.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.npy')
                x = torch.tensor(x, dtype=torch.float32, device=device)
                y = np.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.npy')
                y = torch.tensor(y, dtype=torch.float32, device=device)
                x_list.append(x)
                y_list.append(y)

                x = x_list[0][0].clone().detach()
                if 'PDE_F' not in model_config.particle_model_name:
                    n_particles = int(x.shape[0] / ratio)
                    config.simulation.n_particles = n_particles
                n_frames = len(x_list[0]) * time_ratio
                index_particles = get_index_particles(x, n_particle_types, dimension)
                if n_particle_types > 1000:
                    index_particles = []
                    for n in range(3):
                        index = np.arange(n_particles * n // 3, n_particles * (n + 1) // 3)
                        index_particles.append(index)
                        n_particle_types = 3

        ynorm = torch.load(f'./log/try_{config_file}/ynorm.pt', map_location=device).to(device)
        vnorm = torch.load(f'./log/try_{config_file}/vnorm.pt', map_location=device).to(device)

    if do_tracking | has_state:
        for k in range(len(x_list[0])):
            type = x_list[0][k][:, 5]
            if k == 0:
                type_list = type
            else:
                type_list = torch.concatenate((type_list, type))
        n_particles_max = len(type_list)
        config.simulation.n_particles_max = n_particles_max

    if ratio > 1:
        new_nparticles = int(n_particles * ratio)
        model.a = nn.Parameter(
            torch.tensor(np.ones((n_runs, int(new_nparticles), 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        n_particles = new_nparticles
        index_particles = get_index_particles(x, n_particle_types, dimension)
    if sample_embedding:
        model_a_ = nn.Parameter(
            torch.tensor(np.ones((int(n_particles), model.embedding_dim)), device=device, requires_grad=False,
                         dtype=torch.float32))
        for n in range(n_particles):
            t = to_numpy(x[n, 5]).astype(int)
            index = first_cell_id_particles[t][np.random.randint(n_sub_population)]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((model.n_dataset, int(n_particles), model.embedding_dim)), device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_
    if has_ghost:
        model_ghost = Ghost_Particles(config, n_particles, vnorm, device)
        net = f"./log/try_{config_file}/models/best_ghost_particles_with_{n_runs - 1}_graphs_20.pt"
        state_dict = torch.load(net, map_location=device)
        model_ghost.load_state_dict(state_dict['model_state_dict'])
        model_ghost.eval()
        x_removed_list = torch.load(f'graphs_data/graphs_{dataset_name}/x_removed_list_0.pt', map_location=device)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    if has_mesh:
        hnorm = torch.load(f'./log/try_{config_file}/hnorm.pt', map_location=device).to(device)

        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt', map_location=device)
        mask_mesh = mesh_data['mask']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']

        xy = to_numpy(mesh_data['mesh_pos'])
        x_ = xy[:, 0]
        y_ = xy[:, 1]
        mask = to_numpy(mask_mesh)
        mask_mesh = (x_ > np.min(x_) + 0.02) & (x_ < np.max(x_) - 0.02) & (y_ > np.min(y_) + 0.02) & (
                y_ < np.max(y_) - 0.02)
        mask_mesh = torch.tensor(mask_mesh, dtype=torch.bool, device=device)

        # plt.scatter(x_, y_, s=2, c=to_numpy(mask_mesh))
    if has_adjacency_matrix:
        if model_config.signal_model_name == 'PDE_N':
            mat = scipy.io.loadmat(simulation_config.connectivity_file)
            adjacency = torch.tensor(mat['A'], device=device)
            adj_t = adjacency > 0
            edge_index = adj_t.nonzero().t().contiguous()
            edge_attr_adjacency = adjacency[adj_t]
        else:
            adjacency = torch.load(f'./graphs_data/graphs_{dataset_name}/adjacency.pt', map_location=device)
            adjacency_ = adjacency.t().clone().detach()
            adj_t = torch.abs(adjacency_) > 0
            edge_index = adj_t.nonzero().t().contiguous()
    if 'PDE_N' in model_config.signal_model_name:
        adjacency = torch.load(f'./graphs_data/graphs_{dataset_name}/adjacency.pt', map_location=device)
        edge_index = torch.load(f'./graphs_data/graphs_{dataset_name}/edge_index.pt', map_location=device)
        edge_index_, edge_attr = dense_to_sparse(adjacency)
        first_dataset = data.Data(x=x.clone().detach(), pos=x[:, 1:3].clone().detach(),
                                  edge_index=edge_index.clone().detach(), edge_attr=edge_attr.clone().detach())
        if ('modulation' in model_config.field_type) | ('visual' in model_config.field_type):
            print('load b_i movie ...')
            im = imread(f"graphs_data/{simulation_config.node_value_map}")
            A1 = torch.zeros((n_particles, 1), device=device)
        else:
            print('create graph positions ...')
            G_first = to_networkx(first_dataset.clone().detach(), remove_self_loops=True, to_undirected=True)
            first_positions = nx.spring_layout(G_first, weight='weight', seed=42, k=1)
            X1_first = torch.zeros((n_particles, 2), device=device)
            for node, pos in first_positions.items():
                X1_first[node, :] = torch.tensor([pos[0], pos[1]], device=device)
            print('done ...')

        neuron_index = torch.randint(0, n_particles, (6,))
        neuron_gt_list = []
        neuron_pred_list = []

    if verbose:
        print(f'Test data ... {model_config.particle_model_name} {model_config.mesh_model_name}')
        print('log_dir: {}'.format(log_dir))
        print(f'network: {net}')
        print(table)
        print(f"Total Trainable Params: {total_params}")

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm

    if smooth_particle:
        model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension, device=device)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if test_simulation:
        model, bc_pos, bc_dpos = choose_model(config, device=device)
    else:
        if has_mesh:
            mesh_model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            mesh_model.load_state_dict(state_dict['model_state_dict'])
            mesh_model.eval()
        else:
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            mesh_model = None
            if 'PDE_K' in model_config.particle_model_name:
                model.connection_matrix = torch.load(f'graphs_data/graphs_{dataset_name}/connection_matrix_list.pt',
                                                     map_location=device)
                timeit = np.load(f'graphs_data/graphs_{dataset_name}/times_train_springs_example.npy',
                                 allow_pickle=True)
                timeit = timeit[run][0]
                time_id = 0
        if has_field:
            model_f_p = model

            image_width = int(np.sqrt(n_nodes))
            if has_siren_time:
                model_f = Siren_Network(image_width=image_width, in_features=3, out_features=1, hidden_features=128,
                                        hidden_layers=5, outermost_linear=True, device=device, first_omega_0=80,
                                        hidden_omega_0=80.)
                net = f'./log/try_{config_file}/models/best_model_f_with_1_graphs_{best_model}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])
                model_f.to(device=device)
                model_f.eval()
                table = PrettyTable(["Modules", "Parameters"])
                total_params = 0
                for name, parameter in model_f.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    param = parameter.numel()
                    table.add_row([name, param])
                    total_params += param
                if verbose:
                    print(table)
                    print(f"Total Trainable Params: {total_params}")
            else:
                t = model.field[run].reshape(image_width, image_width)
                t = torch.rot90(t)
                t = torch.flipud(t)
                t = t.reshape(image_width * image_width, 1)
                with torch.no_grad():
                    model.a = a_.clone().detach()
                    model.field[run] = t.clone().detach()

    rmserr_list = []
    pred_err_list = []
    gloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    geomloss_list = []
    angle_list = []
    time.sleep(1)

    if time_window > 0:
        start_it = time_window
        stop_it = n_frames-1
    else:
        start_it = 0
        stop_it = n_frames-1

    x = x_list[0][start_it].clone().detach()
    n_particles = x.shape[0]

    for it in trange(start_it, stop_it):

        if it < n_frames - 4:
            x0 = x_list[0][it//time_ratio].clone().detach()
            x0_next = x_list[0][(it+1)//time_ratio].clone().detach()
            y0 = y_list[0][it//time_ratio].clone().detach()
        if has_mesh:
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset_mesh = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)

        # error calculations
        if 'PDE_N' in model_config.signal_model_name:
            rmserr = torch.sqrt(torch.mean(torch.sum(bc_dpos(x[:, 6:7] - x0[:, 6:7]) ** 2, axis=1)))
            neuron_gt_list.append(x0[neuron_index, 6:7])
            neuron_pred_list.append(x[neuron_index, 6:7])
        elif model_config.mesh_model_name == 'WaveMesh':
            rmserr = torch.sqrt(torch.mean((x[mask_mesh.squeeze(), 6:7] - x0[mask_mesh.squeeze(), 6:7]) ** 2))
        elif model_config.mesh_model_name == 'RD_RPS_Mesh':
            rmserr = torch.sqrt(
                torch.mean(torch.sum((x[mask_mesh.squeeze(), 6:9] - x0[mask_mesh.squeeze(), 6:9]) ** 2, axis=1)))
        elif has_bounding_box:
            rmserr = torch.sqrt(
                torch.mean(torch.sum(bc_dpos(x[:, 1:dimension + 1] - x0[:, 1:dimension + 1]) ** 2, axis=1)))
        else:
            if (do_tracking) | (x.shape[0]!=x0.shape[0]):
                rmserr = torch.zeros(1, device=device)
            else:
                rmserr = torch.sqrt(torch.mean(torch.sum(bc_dpos(x[:, 1:dimension + 1] - x0[:, 1:dimension + 1]) ** 2, axis=1)))
            if x.shape[0] > 5000:
                geomloss = gloss(x[0:5000, 1:3], x0[0:5000, 1:3])
            else:
                geomloss = gloss(x[:, 1:3], x0[:, 1:3])
            geomloss_list.append(geomloss.item())
        rmserr_list.append(rmserr.item())

        # update calculations
        if model_config.mesh_model_name == 'DiffMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0, )
            x[:, 6:7] += pred * hnorm * delta_t
        elif model_config.mesh_model_name == 'WaveMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=1)
            x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
            x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif 'RD_RPS_Mesh' in model_config.mesh_model_name == 'RD_RPS_Mesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=1)
                x[mask_mesh.squeeze(), 6:9] += pred[mask_mesh.squeeze()] * hnorm * delta_t
                x[mask_mesh.squeeze(), 6:9] = torch.clamp(x[mask_mesh.squeeze(), 6:9], 0, 1)
        elif has_field:
            match model_config.field_type:
                case 'tensor':
                    x_mesh[:, 6:7] = model.field[run]
                case 'siren':
                    x_mesh[:, 6:7] = model_f() ** 2
                case 'siren_with_time':
                    x_mesh[:, 6:7] = model_f(time=it / n_frames) ** 2
            x_particle_field = torch.concatenate((x_mesh, x), dim=0)

            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset_p_p = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

            distance = torch.sum(
                bc_dpos(x_particle_field[:, None, 1:dimension + 1] - x_particle_field[None, :, 1:dimension + 1]) ** 2,
                dim=2)
            adj_t = ((distance < (max_radius / 2) ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            pos = torch.argwhere((edge_index[1, :] >= n_nodes) & (edge_index[0, :] < n_nodes))
            pos = to_numpy(pos[:, 0])
            edge_index = edge_index[:, pos]
            dataset_f_p = data.Data(x=x_particle_field, pos=x_particle_field[:, 1:3], edge_index=edge_index,
                                    field=x_particle_field[:, 6:7])

            with torch.no_grad():
                y0 = model(dataset_p_p, data_id=1, training=False, vnorm=vnorm, phi=torch.zeros(1, device=device),
                           has_field=False)
                y1 = model_f_p(dataset_f_p, data_id=1, training=False, vnorm=vnorm, phi=torch.zeros(1, device=device),
                               has_field=True)[n_nodes:]
                y = y0 + y1

            if model_config.prediction == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm
                x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t)
        else:

            data_id = torch.ones((n_particles, 1), dtype=torch.int) * run

            x_ = x
            if has_ghost:
                x_ghost = model_ghost.get_pos(dataset_id=run, frame=it, bc_pos=bc_pos)
                x_ = torch.cat((x_, x_ghost), 0)

            # compute connectivity and prediction
            if has_adjacency_matrix:
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                if test_simulation:
                    y = y0 / ynorm
                else:
                    with torch.no_grad():
                        pred = model(dataset, data_id=1)
                        y = pred
            else:
                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()

                if time_window > 0:
                    xt = []
                    for t in range(time_window):
                        x_ = x_list[0][it//time_ratio - t].clone().detach()
                        xt.append(x_[:, :])
                    dataset = data.Data(x=xt, edge_index=edge_index)
                else:
                    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                if test_simulation:
                    y = y0 / ynorm
                else:
                    with torch.no_grad():
                        pred = model(dataset, data_id=data_id, training=False, vnorm=vnorm, phi=torch.zeros(1, device=device))
                        y = pred[:,0:dimension]

                if has_ghost:
                    y = y[mask_ghost]

            if plot_data:
                y = y0.clone().detach() / ynorm

            loss = (pred [:,0:dimension] * ynorm - y0).norm(2)
            pred_err_list.append(to_numpy(torch.sqrt(loss)))

            if model_config.prediction == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, dimension + 1:2 * dimension + 1] = x[:, dimension + 1:2 * dimension + 1] + y  # speed update
            else:
                y = y * vnorm
                if 'PDE_N' in model_config.signal_model_name:
                    x[:, 6:7] += y * delta_t  # signal update
                else:
                    x[:, dimension + 1:2 * dimension + 1] = y

            x[:, 1:dimension + 1] = bc_pos(x[:, 1:dimension + 1] + x[:, dimension + 1:2 * dimension + 1] * delta_t)  # position update

            if smooth_particle:
                density = model_density(x=x, has_field=False)
                x[:,-1] = density.squeeze()

            if bounce:
                # Bounce on walls
                bouncing_pos = torch.argwhere((x[:, 1] <= 0.1) | (x[:, 1] >= 0.9)).squeeze()
                if bouncing_pos.numel() > 0:
                    x[bouncing_pos, 3] = - 0.6 * x[bouncing_pos, 3]
                bouncing_pos = torch.argwhere((x[:, 2] <= 0.1) | (x[:, 2] >= 0.9)).squeeze()
                if bouncing_pos.numel() > 0:
                    x[bouncing_pos, 4] = - 0.6 * x[bouncing_pos, 4]

            if time_window:
                fixed_pos = torch.argwhere(x[:,5]!=0)
                x_list[0][it//time_ratio+1,fixed_pos.squeeze(),1:2 * dimension + 1] = x[fixed_pos.squeeze(), 1:2 * dimension + 1].clone().detach()

            if fixed:
                fixed_pos = torch.argwhere(x[:,5]==0)
                x[fixed_pos.squeeze(), 1:2 * dimension + 1] = x_list[0][it+1,fixed_pos.squeeze(),1:2 * dimension + 1].clone().detach()

        # vizualization
        if (it % step == 0) & (it >= 0) & visualize:

            num = f"{it:06}"

            if 'latex' in style:
                plt.rcParams['text.usetex'] = True
                rc('font', **{'family': 'serif', 'serif': ['Palatino']})

            if 'black' in style:
                plt.style.use('dark_background')

            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            ax.tick_params(axis='both', which='major', pad=15)

            if has_mesh:
                pts = x[:, 1:3].detach().cpu().numpy()
                tri = Delaunay(pts)
                colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                if model_config.mesh_model_name == 'DiffMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                if model_config.mesh_model_name == 'WaveMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                if model_config.mesh_model_name == 'RD_Gray_Scott_Mesh':
                    fig = plt.figure(figsize=(12, 6))
                    ax = fig.add_subplot(1, 2, 1)
                    colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                    ax = fig.add_subplot(1, 2, 2)
                    colors = torch.sum(x[tri.simplices, 7], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                if (model_config.mesh_model_name == 'RD_RPS_Mesh') | (
                        model_config.mesh_model_name == 'RD_RPS_Mesh_bis'):
                    H1_IM = torch.reshape(x[:, 6:9], (100, 100, 3))
                    plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.axis('off')
            elif ('visual' in field_type) & ('PDE_N' in model_config.signal_model_name):
                if plot_data:

                    plt.close()

                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    if ('modulation' in field_type):
                        A1[:, 0:1] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)
                    if ('visual' in field_type):
                        A1[:n_nodes, 0:1] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)
                        A1[n_nodes:n_particles, 0:1] = 1

                fig = plt.figure(figsize=(8, 12))
                plt.subplot(211)
                plt.title(r'$b_i$', fontsize=48)
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=8, c=to_numpy(A1[:, 0]), cmap='viridis', vmin=0,
                            vmax=2)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(212)
                plt.title(r'$x_i$', fontsize=48)
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=8, c=to_numpy(x[:, 6:7]), cmap='viridis',
                            vmin=-10,
                            vmax=10)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{num}.tif", dpi=80)
                plt.close()
            elif 'PDE_N' in model_config.signal_model_name:

                plt.close()

                y, msg = y = model(dataset, data_id=1, return_all=True)

                plt.style.use('dark_background')
                matplotlib.rcParams['savefig.pad_inches'] = 0

                if False:  # plot_data:
                    plt.figure(figsize=(10, 10))
                    edge_colors = to_numpy(msg) / 10
                    edge_colors = np.minimum(1, edge_colors)
                    edge_colors = np.maximum(-1, edge_colors)
                    cmap = plt.cm.viridis
                    edge_colors = cmap((edge_colors + 1) / 2)
                    nx.draw(G_first, pos=first_positions, with_labels=False, edge_color=edge_colors, node_size=0,
                            alpha=0.5, cmap='bwr')
                    # plt.scatter(to_numpy(X1_first[:, 0]), to_numpy(X1_first[:, 1]), s=200, c=to_numpy(x[:, 6]), cmap='viridis', vmin=-10,vmax=10, edgecolors='None',alpha=0.9)
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim([-1.03, 1.03])
                    plt.ylim([-1.03, 1.03])
                    plt.savefig(f"./{log_dir}/tmp_recons/Edges_{config_file}_{num}.tif", dpi=170.7)
                    plt.close()
                    im = imread(f"./{log_dir}/tmp_recons/Edges_{config_file}_{num}.tif")
                    plt.figure(figsize=(10, 10))
                    plt.imshow(im)
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(3, 3, 1)
                    plt.imshow(im[800:1000, 800:1000, :])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_recons/Edges_{config_file}_{num}.tif", dpi=80)
                    plt.close()

                plt.figure(figsize=(10, 10))
                plt.scatter(to_numpy(X1_first[:, 0]), to_numpy(X1_first[:, 1]), s=30, c=to_numpy(x[:, 6]),
                            cmap='viridis', vmin=-10, vmax=10, edgecolors='k', alpha=1)
                ax = plt.gca()
                ax.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=170.7)
                plt.close()
                im = imread(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif")
                plt.figure(figsize=(10, 10))
                plt.imshow(im)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(3, 3, 1)
                plt.imshow(im[800:1000, 800:1000, :])
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=80)
                plt.close()
            elif 'PDE_K' in model_config.particle_model_name:

                plt.close()
                fig = plt.figure(figsize=(12, 12))
                plt.scatter(x[:, 2].detach().cpu().numpy(),
                            x[:, 1].detach().cpu().numpy(), s=20, color='r')
                if it < n_frames - 1:
                    x0_ = x_list[0][it + 1].clone().detach()
                    plt.scatter(x0_[:, 2].detach().cpu().numpy(),
                                x0_[:, 1].detach().cpu().numpy(), s=40, color='w', alpha=1, edgecolors='None')

                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
            elif do_tracking:
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=20, c='k')
                plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=20, c='r')
                try:
                    x1 = x_list[0][it + 1].clone().detach()
                    plt.scatter(to_numpy(x1[:, 2]), to_numpy(x1[:, 1]), s=20, c='g')
                except:
                    pass
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
            else:
                s_p = 25
                index_particles = get_index_particles(x, n_particle_types, dimension)
                for n in range(n_particle_types):
                    if 'bw' in style:
                        plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                                    x[index_particles[n], 1].detach().cpu().numpy(), s=s_p, color='w')
                    else:
                        plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                                    x[index_particles[n], 1].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
                plt.xlim([0, 1])
                plt.ylim([0, 1])

            if 'latex' in style:
                plt.xlabel(r'$x$', fontsize=78)
                plt.ylabel(r'$y$', fontsize=78)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
            if 'frame' in style:
                plt.xlabel('x', fontsize=48)
                plt.ylabel('y', fontsize=48)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
                plt.text(0, 1.1, f'   ', ha='left', va='top', transform=ax.transAxes, fontsize=48)
                ax.tick_params(axis='both', which='major', pad=15)
                # cbar = plt.colorbar(shrink=0.5)
                # cbar.ax.tick_params(labelsize=32)
            if 'arrow' in style:
                for m in range(x.shape[0]):
                    if x[m, 4] != 0 :
                        if 'speed' in style:
                            # plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(y0[m, 1]) * delta_t * 50, dy=to_numpy(y0[m, 0]) * delta_t * 50, head_width=0.004, length_includes_head=False, color='g')
                            # plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(x[m, 4]) * delta_t * 50, dy=to_numpy(x[m, 3]) * delta_t * 50, head_width=0.004, length_includes_head=False, color='w')
                            # angle = compute_signed_angle(x[m, 3:5], y0[m, 0:2])
                            # angle_list.append(angle)

                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(x[m, 4]) * delta_t * 2, dy=to_numpy(x[m, 3]) * delta_t * 2, head_width=0.004, length_includes_head=True, color='g')

                        if 'acc_true' in style:
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(y0[m, 1])/5E3, dy=to_numpy(y0[m, 0])/5E3, head_width=0.004, length_includes_head=True, color='r')
                        if 'acc_learned' in style:
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(pred[m, 1]*ynorm.squeeze()) / 5E3, dy=to_numpy(pred[m, 0]*ynorm.squeeze()) / 5E3, head_width=0.004, length_includes_head=True, color='r')
                plt.xlim([0,1])
                plt.ylim([0,1])
                # plt.xlim([0.4,0.6])
                # plt.ylim([0.4,0.6])

                # plt.text(0,1.05,f'true acc {to_numpy(y0[900,0:2])} {to_numpy(pred[900,0:2]*ynorm)}',fontsize=12)
                # plt.text(0,1.025,f'true speed {to_numpy(x_list[0][it//time_ratio][900,3:5] + y0[900,0:2] * delta_t)} {to_numpy(x_list[0][it//time_ratio][900,3:5] + pred[900,0:2] * ynorm * delta_t)}',fontsize=12)
            if 'no_ticks' in style:
                plt.xticks([])
                plt.yticks([])
            if 'PDE_G' in model_config.particle_model_name:
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
            if 'PDE_GS' in model_config.particle_model_name:

                object_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune',
                               'pluto', 'io',
                               'europa', 'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea',
                               'titan', 'hyperion', 'moon',
                               'phobos', 'deimos', 'charon']

                masses = torch.tensor(
                    [1.989e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26, 1.31e22,
                     8.93e22, 4.80e22, 1.48e23, 1.08e23, 3.75e19, 1.08e20,
                     6.18e20, 1.10e21, 2.31e21, 1.35e23, 5.62e18, 7.35e22, 1.07e16, 1.48e15, 1.52e21],
                    device=device)

                pos = x[:, 1:dimension + 1]  # - x[0,1:dimension + 1]
                distance = torch.sqrt(torch.sum(bc_dpos(pos[:, None, :] - pos[None, 0, :]) ** 2, dim=2))
                unit_vector = pos / distance

                if it == 0:
                    log_coeff = torch.log(distance[1:])
                    log_coeff_min = torch.min(log_coeff)
                    log_coeff_max = torch.max(log_coeff)
                    log_coeff_diff = log_coeff_max - log_coeff_min
                    d_log = [log_coeff_min, log_coeff_max, log_coeff_diff]

                    log_coeff = torch.log(masses)
                    log_coeff_min = torch.min(log_coeff)
                    log_coeff_max = torch.max(log_coeff)
                    log_coeff_diff = log_coeff_max - log_coeff_min
                    m_log = [log_coeff_min, log_coeff_max, log_coeff_diff]

                    m_ = torch.log(masses) / m_log[2]

                distance_ = (torch.log(distance) - d_log[0]) / d_log[2]
                pos = distance_ * unit_vector
                pos = to_numpy(pos)
                pos[0] = 0

                for n in range(25):
                    plt.scatter(pos[n, 1], pos[n, 0], s=200 * to_numpy(m_[n] ** 3), color=cmap.color(n))
                    # plt.text(pos[n,1], pos[n, 0], object_list[n], fontsize=8)
                plt.xlim([-1.2, 1.2])
                plt.ylim([-1.2, 1.2])
                plt.xticks([])
                plt.yticks([])
            if 'PDE_K' in model_config.particle_model_name:
                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)


            if not ('PDE_N' in model_config.signal_model_name):
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{num}.tif", dpi=80)
                plt.close()

            if smooth_particle:
                fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                plt.scatter(x[:, 2].detach().cpu().numpy(),
                            x[:, 1].detach().cpu().numpy(), s=20, c=x[:, -1].detach().cpu().numpy(), vmin=0, vmax=1)
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Density_{config_file}_{num}.tif", dpi=80)
                plt.close()



            if has_ghost:

                x0 = x_list[0][it + 1].clone().detach()
                x_ghost_pos = bc_pos(x_ghost[:, 1:3])
                x_removed = x_removed_list[it]
                x_all = torch.cat((x, x_removed), 0)

                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                for n in range(n_particle_types):
                    plt.scatter(x0[index_particles[n], 1].detach().cpu().numpy(),
                                x0[index_particles[n], 2].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
                if 'frame' in style:
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    plt.xticks(fontsize=48.0)
                    plt.yticks(fontsize=48.0)
                else:
                    plt.xticks([])
                    plt.yticks([])
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Ghost1_{config_file}_{it}.tif", dpi=170.7)
                plt.close()
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.scatter(x_ghost_pos[:, 0].detach().cpu().numpy(),
                            x_ghost_pos[:, 1].detach().cpu().numpy(), s=s_p, color='g')
                if 'frame' in style:
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    plt.xticks(fontsize=48.0)
                    plt.yticks(fontsize=48.0)
                else:
                    plt.xticks([])
                    plt.yticks([])
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Ghost2_{config_file}_{it}.tif", dpi=170.7)
                plt.close()
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.scatter(x_removed[:, 1].detach().cpu().numpy(),
                            x_removed[:, 2].detach().cpu().numpy(), s=s_p, color='r')
                if 'frame' in style:
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    plt.xticks(fontsize=48.0)
                    plt.yticks(fontsize=48.0)
                else:
                    plt.xticks([])
                    plt.yticks([])
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Ghost3_{config_file}_{it}.tif", dpi=170.7)
                plt.close()

    print('prediction error {:.3e}+/-{:.3e}'.format(np.mean(pred_err_list), np.std(pred_err_list)))

    print('average rollout RMSE {:.3e}+/-{:.3e}'.format(np.mean(rmserr_list), np.std(rmserr_list)))
    if has_mesh:
        h = x_mesh_list[0][0][:, 6:7]
        for k in range(n_frames):
            h = torch.cat((h, x_mesh_list[0][k][:, 6:7]), 0)
        h = to_numpy(h)
        print(h.shape)
        print('average u {:.3e}+/-{:.3e}'.format(np.mean(h), np.std(h)))

    if 'PDE_N' in model_config.signal_model_name:
        neuron_gt_list = torch.cat(neuron_gt_list, 0)
        neuron_pred_list = torch.cat(neuron_pred_list, 0)

        neuron_gt_list = torch.reshape(neuron_gt_list, (n_frames - 1, 6))
        neuron_pred_list = torch.reshape(neuron_pred_list, (n_frames - 1, 6))

        plt.style.use('dark_background')
        matplotlib.rcParams['savefig.pad_inches'] = 0

        plt.figure(figsize=(10, 10))
        plt.plot(neuron_gt_list[:, 0].detach().cpu().numpy(), c='w', linewidth=8, label='original', alpha=0.5)
        plt.plot(neuron_pred_list[:, 0].detach().cpu().numpy(), linewidth=4, c='w', label='prediction')
        plt.legend(fontsize=24)
        plt.plot(neuron_gt_list[:, 1:6].detach().cpu().numpy(), c='w', linewidth=8, alpha=0.5)
        plt.plot(neuron_pred_list[:, 1:6].detach().cpu().numpy(), linewidth=4)
        plt.xlim([0, 500])
        plt.xlabel('time index', fontsize=48)
        plt.ylabel(r'$x_i$', fontsize=48)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/neuron_signals.tif", dpi=170.7)
        plt.close()

    else:

        if geomloss_list == []:
            geomloss_list = [0, 0]
        r = [np.mean(rmserr_list), np.std(rmserr_list), np.mean(geomloss_list), np.std(geomloss_list)]
        print('average rollout Sinkhorn div. {:.3e}+/-{:.3e}'.format(np.mean(geomloss_list), np.std(geomloss_list)))
        np.save(f"./{log_dir}/rmserr_geomloss_{config_file}.npy", r)

        if False:
            rmserr_list = np.array(rmserr_list)
            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            x_ = np.arange(len(rmserr_list))
            y_ = rmserr_list
            plt.scatter(x_, y_, c='k')
            plt.xticks(fontsize=48)
            plt.yticks(fontsize=48)
            plt.xlabel(r'$Epochs$', fontsize=78)
            plt.ylabel(r'$RMSE$', fontsize=78)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/rmserr_{config_file}_plot.tif", dpi=170.7)

        if False:

            x0_next = x_list[0][it].clone().detach()

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            temp1 = torch.cat((x, x0_next), 0)
            temp2 = torch.tensor(np.arange(n_particles), device=device)
            temp3 = torch.tensor(np.arange(n_particles) + n_particles, device=device)
            temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
            temp4 = torch.t(temp4)
            distance4 = torch.sqrt(torch.sum((x[:, 1:3] - x0_next[:, 1:3]) ** 2, 1))
            p = torch.argwhere(distance4 < 0.3)

            temp1_ = temp1[:, [2, 1]].clone().detach()
            pos = dict(enumerate(np.array((temp1_).detach().cpu()), 0))
            dataset = data.Data(x=temp1_, edge_index=torch.squeeze(temp4[:, p]))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, ax=ax, edge_color='r', width=4)
            for n in range(n_particle_types):
                plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                            x[index_particles[n], 1].detach().cpu().numpy(), s=100, color=cmap.color(n))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel(r'$x$', fontsize=78)
            plt.ylabel(r'$y$', fontsize=78)
            formatx = '%.1f'
            formaty = '%.1f'
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, axis='both', which='major', pad=15)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
            ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
            plt.xticks(fontsize=48.0)
            plt.yticks(fontsize=48.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/rmserr_{config_file}_{it}.tif", dpi=170.7)
            plt.close()

            fig = plt.figure(figsize=(12, 12))
            for n in range(n_particle_types):
                plt.scatter(x0_next[index_particles[n], 2].detach().cpu().numpy(),
                            x0_next[index_particles[n], 1].detach().cpu().numpy(), s=50, color=cmap.color(n))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel(r'$x$', fontsize=78)
            plt.ylabel(r'$y$', fontsize=78)
            formatx = '%.2f'
            formaty = '%.2f'
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
            ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
            plt.xticks(fontsize=48.0)
            plt.yticks(fontsize=48.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/GT_{config_file}_{it}.tif", dpi=170.7)
            plt.close()

    if len(angle_list)>0:
        angle = torch.stack(angle_list)
        fig = plt.figure(figsize=(12, 12))
        plt.hist(to_numpy(angle), bins=1000, color='w')
        plt.xlabel('angle', fontsize=48)
        plt.ylabel('count', fontsize=48)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim([-90,90])
        plt.savefig(f"./{log_dir}/results/angle.tif", dpi=170.7)
        plt.close



    # run = 0
    # k = 32
    # x = x_list[run][k]
    # y = y_list[run][k]
    #
    # x_next = x_list[run][k + 1].clone().detach()
    # x_prev = x_list[run][k - 1].clone().detach()
    # print('')
    # print(x[1200])
    # print((x[1200, 1:3] - x_prev[1200, 1:3]) / delta_t)
    # print(x[1200, 3:5])
    # print(x_next[1200])
    # print((x_next[1200, 1:3] - x[1200, 1:3]) / delta_t)
    # print(x_next[1200, 3:5])
    # acc = ((x_next[1200, 1:3] - x[1200, 1:3]) / delta_t - (x[1200, 1:3] - x_prev[1200, 1:3]) / delta_t) / delta_t
    # print(acc)
    # print((x_next[1200, 1:3] - 2 * x[1200, 1:3] + x_prev[1200, 1:3]) / delta_t ** 2)
    # print(y[1200])




