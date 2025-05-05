import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random

from GNN_particles_Ntype import *
from ParticleGraph.models.utils import *
from ParticleGraph.utils import *
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Ghost_Particles import *
from geomloss import SamplesLoss
from ParticleGraph.sparsify import EmbeddingCluster, sparsify_cluster
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit

from ParticleGraph.generators.cell_utils import *
from ParticleGraph.fitting_models import linear_model
from torch_geometric.utils import dense_to_sparse
import torch.optim as optim
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from ParticleGraph.denoise_data import *
from scipy.spatial import KDTree
from sklearn import neighbors, metrics
import torch.nn.functional as F

def data_train(config=None, erase=False, best_model=None, device=None):
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
    has_mouse_city = ('mouse_city' in config.dataset) | ('rat_city' in config.dataset)
    sub_sampling = config.simulation.sub_sampling
    rotation_augmentation = config.training.rotation_augmentation

    if rotation_augmentation & (sub_sampling > 1):
        assert(False), 'rotation_augmentation does not work with sub_sampling > 1'

    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    if 'Agents' in config.graph_model.particle_model_name:
        data_train_agents(config, best_model, device)
    elif has_mouse_city:
        data_train_rat_city(config, erase, best_model, device)
    elif has_WBI:
        data_train_WBI(config, erase, best_model, device)
    elif has_particle_field:
        data_train_particle_field(config, erase, best_model, device)
    elif has_mesh:
        data_train_mesh(config, erase, best_model, device)
    elif has_signal:
        data_train_synaptic2(config, erase, best_model, device)
    elif do_tracking & has_cell_division:
        data_train_cell(config, erase, best_model, device)
    elif do_tracking:
        data_train_tracking(config, erase, best_model, device)
    elif has_cell_division:
        data_train_cell(config, erase, best_model, device)
    elif has_state:
        data_train_particle(config, erase, best_model, device)
    elif 'PDE_GS' in config.graph_model.particle_model_name:
        data_solar_system(config, erase, best_model, device)
    else:
        data_train_particle(config, erase, best_model, device)


def data_train_particle(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting

    print(f'training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    time_window = train_config.time_window
    time_step = train_config.time_step
    recursive_loop = train_config.recursive_loop

    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    recursive_loop = train_config.recursive_loop
    coeff_continuous = train_config.coeff_continuous
    coeff_permutation = train_config.coeff_permutation
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
    particle_batch_ratio = train_config.particle_batch_ratio
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs

    log_dir, logger = create_log_dir(config, erase)
    print(f'graph files N: {n_runs}')
    logger.info(f'graph files N: {n_runs}')
    time.sleep(0.5)
    print('load data ...')
    x_list = []
    y_list = []
    edge_p_p_list = []
    # edge_saved = os.path.exists(f'graphs_data/{dataset_name}/edge_p_p_list_0.npz')
    edge_saved = False

    run_lengths = list()
    time.sleep(0.5)
    n_particles_max = 0

    for run in trange(n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        if np.isnan(x).any() | np.isnan(y).any():
            print('Pb isnan')
        if x[0].shape[0] > n_particles_max:
            n_particles_max = x[0].shape[0]
        x_list.append(x)
        y_list.append(y)
        if edge_saved:
            edge_p_p = np.load(f'graphs_data/{dataset_name}/edge_p_p_list_{run}.npz')
            edge_p_p_list.append(edge_p_p)
        run_lengths.append(len(x))
    x = torch.tensor(x_list[0][0], dtype=torch.float32, device=device)
    y = torch.tensor(y_list[0][0], dtype=torch.float32, device=device)
    time.sleep(0.5)
    for run in trange(0, n_runs,  max(n_runs // 10, 1)):
        for k in range(run_lengths[run]):
            if (k % 10 == 0) | (n_frames < 1000):
                try:
                    x = torch.cat((x, torch.tensor(x_list[run][k], dtype=torch.float32, device=device)), 0)
                except:
                    print(f'Error in run {run} frame {k}')
                y = torch.cat((y, torch.tensor(y_list[run][k], dtype=torch.float32, device=device)), 0)
        time.sleep(0.5)
    if torch.isnan(x).any() | torch.isnan(y).any():
        print('Pb isnan')
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

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    if (best_model != None) & (best_model != ''):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    logger.info(f"total Trainable Params: {n_total_params}")
    logger.info(f'learning rates: {lr}, {lr_embedding}')
    model.train()

    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    x = torch.tensor(x_list[plot_config.data_embedding][0], dtype=torch.float32, device=device)
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
    if simulation_config.state_type == 'sequence':
        ind_a = torch.tensor(np.arange(1, n_particles*100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]

    print("start training particles ...")
    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    time.sleep(1)

    for epoch in range(start_epoch, n_epochs + 1):

        logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        batch_size = int(get_batch_size(epoch))
        logger.info(f'batch_size: {batch_size}')

        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        if particle_batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / particle_batch_ratio)
        else:
            Niter = n_frames * data_augmentation_loop // batch_size
        plot_frequency = int(Niter // 20)

        if epoch==0:
            print(f'{Niter} iterations per epoch')
            logger.info(f'{Niter} iterations per epoch')
            print(f'plot every {plot_frequency} iterations')

        time.sleep(1)
        total_loss = 0
        # start = time.time()

        for N in trange(Niter):

            dataset_batch = []
            ids_batch = []
            ids_index = 0
            for batch in range(batch_size):

                run = 1 + np.random.randint(n_runs - 1)
                k = time_window + np.random.randint(run_lengths[run] - 1 - time_window - time_step - recursive_loop)
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
                if recursive_loop > 0:
                    x_next = torch.tensor(x_list[run][k + recursive_loop], dtype=torch.float32, device=device).clone().detach()
                else:
                    x_next = torch.tensor(x_list[run][k + time_step], dtype=torch.float32, device=device).clone().detach()

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

                if edge_saved:
                    edges = edge_p_p_list[run][f'arr_{k}']
                    edges = torch.tensor(edges, dtype=torch.int64, device=device)
                else:

                    # positions = x_list[run][k][:, 1:3]
                    # tree = neighbors.KDTree(positions)
                    # receivers_list = tree.query_radius(positions, r=max_radius)
                    # num_nodes = len(positions)
                    # senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
                    # receivers = np.concatenate(receivers_list, axis=0)
                    # edges = torch.tensor(np.array([senders, receivers]), dtype=torch.int64, device=device)

                    distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
                    edges = adj_t.nonzero().t().contiguous()

                if particle_batch_ratio < 1:
                    ids = np.random.permutation(x.shape[0])[:int(x.shape[0] * particle_batch_ratio)]
                    ids = np.sort(ids)
                    mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                    edges = edges[:, mask]

                if time_window == 0:
                    dataset = data.Data(x=x[:, :], edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)
                else:
                    xt = []
                    for t in range(time_window):
                        x_ = torch.tensor(x_list[run][k - t], dtype=torch.float32, device=device)
                        xt.append(x_[:, :])
                    dataset = data.Data(x=xt, edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)

                if recursive_loop > 0 :
                    y = torch.tensor(y_list[run][k+recursive_loop], dtype=torch.float32, device=device).clone().detach()
                elif time_step == 1:
                    y = torch.tensor(y_list[run][k], dtype=torch.float32, device=device).clone().detach()
                elif time_step > 1:
                    y = torch.tensor(x_list[run][k + time_step, :, 1:dimension + 1], device=device).clone().detach()

                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)
                if time_step > 1:
                    y[:,0:dimension] = y[:,0:dimension] / ynorm

                if train_config.shared_embedding:
                    run = 1

                if batch == 0:
                    data_id = torch.ones((y.shape[0],1), dtype=torch.int) * run
                    y_batch = y
                    k_batch = torch.ones((x.shape[0],1), dtype=torch.int, device = device) * k
                    if particle_batch_ratio < 1:
                        ids_batch = ids
                else:
                    data_id = torch.cat((data_id, torch.ones((y.shape[0],1), dtype=torch.int) * run), dim = 0)
                    y_batch = torch.cat((y_batch, y), dim=0)
                    k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device = device) * k), dim=0)
                    if particle_batch_ratio < 1:
                        ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                ids_index += x.shape[0]

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=data_id, training=True, k=k_batch)

            if recursive_loop > 0:
                for loop in range(recursive_loop):
                    ids_index = 0
                    for batch in range(batch_size):
                        x = dataset_batch[batch].x.clone().detach()

                        X1 = x[:, 1:dimension + 1]
                        V1 = x[:, dimension + 1:2 * dimension + 1]
                        if model_config.prediction == '2nd_derivative':
                            V1 += pred[ids_index:ids_index+x.shape[0]] * ynorm * delta_t
                        else:
                            V1 = pred[ids_index:ids_index+x.shape[0]] * ynorm
                        x[:, 1:dimension + 1] = bc_pos(X1 + V1 * delta_t)
                        x[:, dimension + 1:2 * dimension + 1] = V1
                        dataset_batch[batch].x = x

                        ids_index += x.shape[0]

                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        pred = model(batch, data_id=data_id, training=True, k=k_batch)

            if has_ghost:
                loss = ((pred[mask_ghost] - y_batch)).norm(2)
            elif simulation_config.state_type == 'sequence':
                loss = (pred - y_batch).norm(2)
                loss = loss + train_config.coeff_model_a * (model.a[run, ind_a + 1] - model.a[run, ind_a]).norm(2)
            else:
                if time_step == 1:
                    if particle_batch_ratio < 1:
                        loss = (pred[ids_batch] - y_batch[ids_batch]).norm(2)
                    else:
                        loss = (pred - y_batch).norm(2)
                elif time_step > 1:
                    if model_config.prediction == '2nd_derivative':
                        x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * (x[:, dimension + 1:2*dimension + 1] + delta_t * time_step * pred * ynorm))
                    else:
                        x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * pred * ynorm)

                    if particle_batch_ratio < 1:
                        loss = loss + (x_batch[ids_batch] - y_batch[ids_batch]).norm(2)
                    else:
                        loss = loss + ( x_batch - y_batch).norm(2)

            if (epoch>0) & (coeff_continuous>0):
                rr = torch.linspace(0, max_radius, 1000, dtype=torch.float32, device=device)
                for n in np.random.permutation(n_particles)[:n_particles//100]:
                    embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                    in_features = get_in_features(rr=rr + simulation_config.max_radius/200, embedding=embedding_, model=model, model_name=config.graph_model.particle_model_name, max_radius=simulation_config.max_radius)
                    func1 = model.lin_edge(in_features)
                    in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config.graph_model.particle_model_name, max_radius=simulation_config.max_radius)
                    func0 = model.lin_edge(in_features)
                    grad = func1-func0
                    loss = loss + coeff_continuous * grad.norm(2)

            # matplotlib.use("Qt5Agg")
            # fig = plt.figure()
            # plt.scatter(to_numpy(y_batch), to_numpy(pred), s=1, c='k',alpha=0.1)

            loss.backward()
            optimizer.step()

            if has_ghost:
                optimizer_ghost_particles.step()

            # if False:
                for name, param in model.lin_edge.named_parameters():
                    if param.requires_grad:
                        print(f"Gradient of {name}: {param.grad}")
            #     for name, param in model.lin_edge.named_parameters():
            #         if param.requires_grad:
            #             print(f"{name}: {param.data}")
            # end = time.time()
            # print(f"iter time: {end - start:.4f} seconds")

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % plot_frequency == 0)) | (N == 0)):
                plot_training(config=config, log_dir=log_dir,
                              epoch=epoch, N=N, x=x, model=model, n_nodes=0, n_node_types=0, index_nodes=0,
                              dataset_num=1,
                              index_particles=index_particles, n_particles=n_particles,
                              n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)
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

        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        if (has_bounding_box == False) & ('PDE_K' not in model_config.particle_model_name) & ('PDE_MLPs' not in model_config.particle_model_name) & ('PDE_F' not in model_config.particle_model_name) & ('PDE_WF' not in model_config.particle_model_name):
            ax = fig.add_subplot(1, 5, 2)
            embedding = get_embedding(model.a, 1)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0],
                            embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)

            ax = fig.add_subplot(1, 5, 3)
            func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                                model_MLP=model.lin_edge, model=model,
                                                                n_nodes=0,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                                cmap=cmap, update_type='NA', device=device)

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
                    plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}_before training function.tif")
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
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()


def data_solar_system(config, erase, best_model, device):
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

    log_dir, logger = create_log_dir(config, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
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
    # net = f"{log_dir}/models/best_model_with_1_graphs_0_0.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
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
                plt.savefig(f"./{log_dir}/tmp_training/function/func_{epoch}_{N}.tif", dpi=87)
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
                plt.savefig(f"./{log_dir}/tmp_training/embedding/embedding_{epoch}_{N}.tif", dpi=87)
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
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()


def data_train_cell(config, erase, best_model, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

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
    min_radius = simulation_config.min_radius
    time_step = simulation_config.time_step

    log_dir, logger = create_log_dir(config, erase)
    print(f'graph files N: {n_runs}')
    logger.info(f'graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    edge_p_p_list = []
    vertices_pos_list = []
    # edge_saved = os.path.exists(f'graphs_data/{dataset_name}/edge_p_p_list_0.npz')
    edge_saved = False

    print('load data ...')
    for run in trange(n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        x_list.append(x)
        y_list.append(y)
        if edge_saved:
            edge_p_p = np.load(f'graphs_data/{dataset_name}/edge_p_p_list_{run}.npz')
            edge_p_p_list.append(edge_p_p)
            
    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()

    for run in range(n_runs):
        for k in trange(n_frames):
            if (k % 10 == 0) | (n_frames < 1000):
                x = torch.cat((x, x_list[run][k].clone().detach()), 0)
                y = torch.cat((y, y_list[run][k].clone().detach()), 0)
        time.sleep(0.5)

    posnorm, bounding_box = norm_position(x, dimension, device)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    if do_tracking:
        vnorm = torch.tensor([1.0], dtype=torch.float32, device=device)
        ynorm = torch.tensor([1.0], dtype=torch.float32, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    torch.save(posnorm, os.path.join(log_dir, 'posnorm.pt'))
    torch.save(bounding_box, os.path.join(log_dir, 'bounding_box.pt'))
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
            if do_tracking:
                x_list[0][k][:, 1+dimension:1+2*dimension] = 0
        config.simulation.n_particles_max = n_particles_max + 1
        n_particles = n_particles_max / n_frames
    else:
        for k in range(n_frames + 1):
            type = x_list[0][k][:, 5]
            if k == 0:
                type_stack = type
            else:
                type_stack = torch.cat((type_stack, type), 0)

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    if best_model != None:
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"

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

    x = x_list[0][n_frames - 1].clone().detach()

    print("start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    Niter = n_frames * data_augmentation_loop // batch_size
    print(f'plot every {Niter // 100} iterations')

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    time.sleep(1)
    for epoch in range(start_epoch, n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        Niter=2

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = np.random.randint(n_runs)

            dataset_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - time_step - 2)

                x = x_list[run][k].clone().detach()

                if edge_saved:
                    edges = edge_p_p_list[run][f'arr_{k}']
                else:
                    distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    edges = adj_t.nonzero().t().contiguous()

                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)

                # viz_3d_cell(x, edges, posnorm, 100)

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
                pred = model(batch, data_id=run, training=True, phi=phi)

            if rotation_augmentation:
                new_x = cos_phi * pred[:, 0] - sin_phi * pred[:, 1]
                new_y = sin_phi * pred[:, 0] + cos_phi * pred[:, 1]
                pred[:, 0] = new_x
                pred[:, 1] = new_y

            if do_tracking:
                x_next = x_list[run][k + time_step]
                x_pos_next = x_next[:, 1:dimension + 1].clone().detach()
                if model_config.prediction == '2nd_derivative':
                    x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * (x[:, dimension + 1:2*dimension + 1] + delta_t * time_step * pred * ynorm))
                else:
                    x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * pred * ynorm)
                distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_value = result.values
                indices = result.indices
                loss = min_value.norm(2) * 1E3 / posnorm
            else:
                loss = (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()

            # model.lin_edge.layers[0].weight.grad
            # fig = plt.figure(figsize=(8, 8))
            # plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=10, c='b')
            # plt.scatter(to_numpy(x_next[:, 1]), to_numpy(x_next[:, 2]), s=10, c='r')

            total_loss += loss.item()

            visualize_embedding = True
            if ((epoch < 10) & (N % (Niter // 100) == 0)) | (N == 0):
                if visualize_embedding:

                    # if do_tracking | has_state:
                    #     id_list = []
                    #     for k in range(n_frames):
                    #         ids = x_list[0][k][:, -1]
                    #         id_list.append(ids)
                    #     plot_training_state(config=config, id_list=id_list, log_dir=log_dir,
                    #                         epoch=epoch, N=N, model=model, n_particle_types=n_particle_types,
                    #                         type_list=type_list, type_stack=type_stack, ynorm=ynorm, cmap=cmap,
                    #                         device=device)
                    # else:
                    #     plot_training_cell(config=config, log_dir=log_dir,
                    #                        epoch=epoch, N=N, model=model, n_particle_types=n_particle_types,
                    #                        type_list=type_list, ynorm=ynorm, cmap=cmap, device=device)

                    fig, ax = fig_init(fontsize=24)
                    plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), s=0.1, color='g',alpha=0.1, edgecolor='none')
                    plt.xlabel(r'$a_{i0}$', fontsize=48)
                    plt.ylabel(r'$a_{i1}$', fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
                    plt.close()

                    fig, ax = fig_init(fontsize=24)
                    g = to_numpy(model.a.grad)
                    plt.scatter(g[:, 0], g[:, 1], s=0.1, c='g')
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/grad_{epoch}_{N}.tif", dpi=87)
                    plt.close()

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


        # embedding = proj_interaction
        # labels, n_clusters, new_labels = sparsify_cluster_state(config.training.cluster_method,
        #                                                         proj_interaction, embedding,
        #                                                         config.training.cluster_distance_threshold,
        #                                                         true_type_list,
        #                                                         n_particle_types, embedding_cluster)


def data_train_rat_city(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(config.training.seed)

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    n_epochs = train_config.n_epochs
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
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
    time_step = train_config.time_step
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    coeff_model_a = train_config.coeff_model_a

    do_tracking = train_config.do_tracking

    has_state = (simulation_config.state_type != 'discrete')

    log_dir, logger = create_log_dir(config, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    os.makedirs(f'./{log_dir}/tmp_training/loss', exist_ok=True)

    x_list = []
    x = torch.load(f'graphs_data/{dataset_name}/x_list_0.pt', map_location=device, weights_only=False)
    x_list.append(x)
    edge_p_p_list = []
    edge_p_p = np.load(f'graphs_data/{dataset_name}/edge_p_p_list_0.npz')
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

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    if best_model != None:
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    coeff_entropy_loss = train_config.coeff_entropy_loss
    entropy_loss = KoLeoLoss()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')

    print("start training ...")

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

        plot_frequency = int(Niter // 100)
        print(f'{Niter} iterations per epoch')
        logger.info(f'{Niter} iterations per epoch')
        print(f'plot every {plot_frequency} iterations')

        Niter = 2

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 0
            optimizer.zero_grad()

            k = 1 + np.random.randint(n_frames - 2*time_step)
            # k = np.random.randint(n_frames // 2 - 2) * 2
            x = x_list[run][k].clone().detach()

            loss = (model.a[:, 1:] - model.a[:, :-1]).norm(2) * coeff_model_a

            edges = edge_p_p_list[run][f'arr_{k}']
            edges = torch.tensor(edges, dtype=torch.int64, device=device)
            dataset = data.Data(x=x[:, :], edge_index=edges)

            pred = model(dataset, data_id=run, training=True, phi=phi, has_field=False, k=k)

            if rotation_augmentation:
                new_x = cos_phi * pred[:, 0] - sin_phi * pred[:, 1]
                new_y = sin_phi * pred[:, 0] + cos_phi * pred[:, 1]
                pred[:, 0] = new_x
                pred[:, 1] = new_y

            if (do_tracking): # | (epoch==0):
                x_next = x_list[run][k + time_step]
                x_pos_next = x_next[:, 1:3].clone().detach()
                if model_config.prediction == '2nd_derivative':
                    x_pos_pred = (x[:, 1:3] + (delta_t * time_step) * (x[:, 3:5] + delta_t * time_step * pred * ynorm))
                else:
                    x_pos_pred = (x[:, 1:3] + (delta_t * time_step) * pred * ynorm)
                distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_value = result.values
                indices = result.indices
                loss += torch.sum(min_value) * 1E5
            else:
                x_next = x_list[run][k + time_step]
                sorted_indices = torch.argsort(x_next[:, 8])
                x_next_sorted = x_next[sorted_indices,1:3]

                pred_pos_next = x[:,1:3] + delta_t * time_step * pred
                sorted_indices = torch.argsort(x[:, 8])
                pred_pos_next_sorted = pred_pos_next[sorted_indices]

                loss += (pred_pos_next_sorted - x_next_sorted).norm(2) * 1E5

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
            if visualize_embedding & ((N % plot_frequency == 0) | (N == 0)):

                plot_training_mouse(config=config, log_dir=log_dir, epoch=epoch, N=N, model=model)

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


def data_train_mesh(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    n_epochs = train_config.n_epochs
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
    dimension = simulation_config.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = simulation_config.n_particle_types

    log_dir, logger = create_log_dir(config, erase)
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
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
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
    mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    if 'WaveMeshSmooth' in model_config.mesh_model_name:
        with torch.no_grad():
            distance = torch.sum((mesh_data['mesh_pos'][:, None, :] - mesh_data['mesh_pos'][None, :, :]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
            edge_index_mesh = adj_t.nonzero().t().contiguous()

    h = []

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    if best_model != None:
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('update variables ...')
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

    print("start training mesh ...")


    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        old_batch_size = batch_size
        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        Niter = n_frames * data_augmentation_loop // batch_size
        plot_frequency = int(Niter // 50)
        if epoch==0:
            print(f'{Niter} iterations per epoch')
            logger.info(f'{Niter} iterations per epoch')
            print(f'plot every {plot_frequency} iterations')

        if epoch == 1:
            repeat_factor = batch_size // old_batch_size
            mask_mesh = mask_mesh.repeat(repeat_factor, 1)

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size
        if (batch_size == 1):
            Niter = Niter // 4

        for N in trange(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

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

                    # fig = plt.figure(figsize=(8, 8))
                    # plt.ion()
                    # pos = torch.argwhere(edge_index_mesh[0,:]==4850)
                    # plt.scatter(to_numpy(x_mesh[:, 1]), to_numpy(x_mesh[:, 2]), s=10, c=to_numpy(x_mesh[:, 6]))
                    # plt.scatter(to_numpy(x_mesh[edge_index_mesh[1, pos], 1]), to_numpy(x_mesh[edge_index_mesh[1, pos], 2]), s=10, c='r')
                    # plt.show()

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
                pred = model(batch, data_id = data_id, training = True, phi = phi)

                # fig = plt.figure(figsize=(8, 8))
                # plt.ion()
                # pos = torch.argwhere(edge_index_mesh[0,:]==4850)
                # plt.scatter(to_numpy(x_mesh[:, 1]), to_numpy(x_mesh[:, 2]), s=10, c=to_numpy(model.density))
                # plt.scatter(to_numpy(x_mesh[edge_index_mesh[1, pos], 1]), to_numpy(x_mesh[edge_index_mesh[1, pos], 2]), s=10, c='r')
                # plt.show()

            loss = ((pred - y_batch) * mask_mesh).norm(2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = ('Wave' in model_config.mesh_model_name)
            if visualize_embedding & (((epoch < 10) & (N % plot_frequency == 0)) | (N == 0)):
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

                plot_training(config=config, log_dir=log_dir, epoch=epoch, N=N, x=x_mesh, model=model, n_nodes=n_nodes, n_node_types=n_node_types,
                              index_nodes=index_nodes, dataset_num=1, index_particles=[], n_particles=[],
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
            else:
                r = torch.tensor(np.linspace(-150, 150, 100)).to(device)
                in_features = torch.cat((r[:, None], embedding_), dim=1)
                h = model.lin_phi(in_features.float())
                popt, pcov = curve_fit(linear_model, to_numpy(r.squeeze()), to_numpy(h.squeeze()))
                popt_list.append(popt)
                h = h[:, 0]
            func_list.append(h)
            if (n % 100):
                plt.plot(to_numpy(r),
                         to_numpy(h) * to_numpy(hnorm), linewidth=1,
                         color='k', alpha=0.05)
        func_list = torch.stack(func_list)
        coeff_norm = to_numpy(func_list)
        popt_list = np.array(popt_list)

        if 'RD_RPS_Mesh' in model_config.mesh_model_name:
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
        elif 'WaveMeshSmooth' in model_config.mesh_model_name:
            embedding = get_embedding(model.a, 1)
            proj_interaction = []
            labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                              train_config.cluster_distance_threshold, type_list,
                                                              n_particle_types, embedding_cluster)

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
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()


def data_train_particle_field(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'training particle field data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

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
    rotation_augmentation = train_config.rotation_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    n_ghosts = train_config.n_ghosts

    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    sparsity_freq = train_config.sparsity_freq

    log_dir, logger = create_log_dir(config, erase)
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
        x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device, weights_only=False)
        y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device, weights_only=False)
        edge_p_p = torch.load(f'graphs_data/{dataset_name}/edge_p_p_list{run}.pt', map_location=device, weights_only=False)
        edge_f_p = torch.load(f'graphs_data/{dataset_name}/edge_f_p_list{run}.pt', map_location=device, weights_only=False)
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
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device, weights_only=False)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt', map_location=device, weights_only=False)
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
    mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_1.pt', map_location=device, weights_only=False)
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
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
    model.ynorm = ynorm
    model.vnorm = vnorm

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"total trainable Params: {n_total_params}")
    logger.info(f'learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
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
        # net = f"{log_dir}/models/best_model_f_with_1_graphs_20.pt"
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
                pred_f_p = model(batch, data_id = run, training=True, phi=phi, has_field=True)
            for batch in batch_loader_p_p:
                pred_p_p = model(batch, data_id = run, training=True, phi=phi, has_field=False)

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
                                             model_f=model_f, n_frames=n_frames,
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
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()

        ax = fig.add_subplot(1, 5, 3)
        func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                            model_MLP=model.lin_edge, model=model,
                                                            n_nodes=0,
                                                            dataset_number=1,
                                                            n_particles=n_particles, ynorm=ynorm,
                                                            type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                            cmap=cmap, update_type='NA', device=device)

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
                plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}_before training function.tif")

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


def data_train_potential_energy(config, erase, best_model, device):
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


def data_train_synaptic2(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'training with data {model_config.particle_model_name} {model_config.mesh_model_name}')

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
    particle_batch_ratio = train_config.particle_batch_ratio
    batch_size = get_batch_size(0)
    embedding_cluster = EmbeddingCluster(config)
    cmap = CustomColorMap(config=config)
    n_runs = train_config.n_runs
    noise_level = train_config.noise_level
    field_type = model_config.field_type
    coeff_lin_modulation = train_config.coeff_lin_modulation
    coeff_model_b = train_config.coeff_model_b
    coeff_sign = train_config.coeff_sign
    time_step = train_config.time_step
    n_ghosts = int(train_config.n_ghosts)
    has_ghost = n_ghosts > 0

    if field_type != '':
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = True
    else:
        n_nodes = simulation_config.n_particles
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = False

    has_Siren = has_field & ('learnable_short_term_plasticity' not in field_type)

    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    recursive_parameters = train_config.recursive_parameters.copy()

    log_dir, logger = create_log_dir(config, erase)
    print(f'loading graph files N: {n_runs} ...')
    logger.info(f'Graph files N: {n_runs}')

    x_list = []
    y_list = []
    for run in trange(1,n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    x = x_list[0][n_frames - 10]

    activity = torch.tensor(x_list[0][:, :, 6:7],device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()
    activity = activity.t()

    xnorm = torch.round(1.5*torch.std(distrib).to(device))
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    print(f'xnorm: {to_numpy(xnorm)}')
    logger.info(f'xnorm: {to_numpy(xnorm)}')

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

    if train_config.denoise:
        print('denoise data ...')
        for k in range(len(x_list)):
            x_, y_ = denoise_data(config, x_list[k], y_list[k], device)
            x_list[k] = x_
            y_list[k] = y_

        activity = torch.tensor(x_list[0][:, :, 6:7], device=device)
        activity = activity.squeeze()
        activity = activity.t()

        plt.figure(figsize=(15, 10))
        n = np.random.permutation(n_particles)
        for i in range(25):
            plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        plt.xlabel('time', fontsize=64)
        plt.ylabel('$x_{i}$', fontsize=64)
        plt.xlim([0,n_frames])
        # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.title(r'$x_i$ samples',fontsize=48)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/denoised_activity.tif', dpi=300)
        plt.close()

    if model_config.embedding_init !='':
        print('compute init embedding ...')
        for j in trange(n_frames):
            if j == 0:
                time_series = np.array(x_list[0][j][:,6:7])
            else:
                time_series = np.concatenate((time_series, x_list[0][j][:,6:7]), axis=1)
        time_series = np.array(time_series)

        match model_config.embedding_init:
            case 'umap':
                trans = umap.UMAP(n_neighbors=50, n_components=2 , transform_queue_size=0, random_state=config.training.seed).fit(time_series)
                projections = trans.transform(time_series)
            case 'pca':
                pca = PCA(n_components=2)
                projections = pca.fit_transform(time_series)
            case 'svd':
                svd = TruncatedSVD(n_components=2)
                projections = svd.fit_transform(time_series)
            case 'tsne':
                tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                projections = tsne.fit_transform(time_series)

        fig = plt.figure(figsize=(8, 8))
        for n in range(n_particle_types):
            pos = torch.argwhere(type_list == n).squeeze()
            plt.scatter(projections[to_numpy(pos), 0], projections[to_numpy(pos), 1], s=10, color=cmap.color(n))
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Embedding_init.tif")
        plt.close()
    else:
        projections = None

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(model_config=config, device=device, projections=projections)
    if has_ghost:
        model_missing_activity = nn.ModuleList([
            Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                  hidden_features=model_config.hidden_dim_nnr,
                  hidden_layers=model_config.n_layers_nnr, first_omega_0=omega, hidden_omega_0=omega,
                  outermost_linear=model_config.outermost_linear_nnr)
            for n in range(n_runs)
        ])
        model_missing_activity.to(device=device)
        optimizer_f = torch.optim.Adam(lr=train_config.learning_rate_NNR, params=model_missing_activity.parameters())
        model_missing_activity.train()
    elif has_Siren:
        if ('Siren_short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
            model_f = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                            hidden_features=model_config.hidden_dim_nnr,
                            hidden_layers=model_config.n_layers_nnr, first_omega_0=omega, hidden_omega_0=omega,
                            outermost_linear=model_config.outermost_linear_nnr)
        else:
            model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                    first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        optimizer_f = torch.optim.Adam(lr=train_config.learning_rate_NNR, params=model_f.parameters())
        model_f.train()
    if (best_model != None) & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'load {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')

        if has_Siren:
            net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{best_model}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])
    else:
        start_epoch = 0
    lr = train_config.learning_rate_start
    if train_config.learning_rate_update_start==0:
        lr_update = train_config.learning_rate_start
    else:
        lr_update = train_config.learning_rate_update_start
    lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start
    lr_modulation = train_config.learning_rate_modulation_start

    print(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')
    logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, lr_modulation=lr_modulation)

    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)

    # matrix = to_numpy(adjacency)
    # rank = get_matrix_rank(matrix)
    # print(f"the rank of the matrix {adjacency.shape} is: {rank}")
    # logger.info(f"the rank of the matrix {adjacency.shape} is: {rank}")
    # centers, density = compute_spectral_density(matrix)
    # plt.figure(figsize=(8, 8))
    # plt.plot(centers, density,c='k')
    # plt.xlabel('Eigenvalue')
    # plt.ylabel('Density')
    # plt.title('Spectral Density')
    # plt.savefig(f"./{log_dir}/tmp_training/Spectral_density_Aij.tif")

    if train_config.with_connectivity_mask:
        if has_ghost:
            mask_ = (adjacency >0) * 1.0
            model.mask[:n_particles,:n_particles] = mask_
        else:
            model.mask = (adjacency >0) * 1.0
    if coeff_sign > 0:
        index_weight = []
        for i in range(n_particles):
            index_weight.append(torch.argwhere(model.mask[:, i] >0).squeeze())
    if has_ghost:
        edges, edge_attr = dense_to_sparse(torch.ones((n_particles + n_ghosts)) - torch.eye(n_particles + n_ghosts))
        edges = edges.to(device=device)
        model.mask[n_particles:, :] = 0
    else:
        edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    edges_all = edges.clone().detach()

    print(f'{edges.shape[1]} edges')

    if 'PDE_N3' in model_config.signal_model_name:
        ind_a = torch.tensor(np.arange(1, n_particles*100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]
    if has_field:
        modulation = torch.tensor(x_list[0], device=device)
        modulation = modulation[:, :, 8:9].squeeze()
        modulation = modulation.t()
        modulation = modulation.clone().detach()
        d_modulation = (modulation[:, 1:] - modulation[:, :-1]) / delta_t
        modulation_norm = torch.tensor(1.0E-2, device=device)

    coeff_L1 = train_config.coeff_L1
    coeff_diff = train_config.coeff_diff
    coeff_diff_update = train_config.coeff_diff_update
    logger.info(f'coeff_L1: {coeff_L1} coeff_diff: {coeff_diff} coeff_diff_update: {coeff_diff_update}')
    print(f'coeff_L1: {coeff_L1} coeff_diff: {coeff_diff} coeff_diff_update: {coeff_diff_update}')

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss = []
    time.sleep(0.2)

    for epoch in range(start_epoch, n_epochs + 1):

        if (epoch == train_config.epoch_reset) | ((epoch>0) & (epoch % train_config.epoch_reset_freq == 0)):
            with torch.no_grad():
                model.W.copy_(model.W * 0)
                model.a.copy_(model.a * 0)
            logger.info(f'reset W model.a at epoch : {epoch}')
            print(f'reset W model.a at epoch : {epoch}')
        if epoch==train_config.n_epochs_init:
            coeff_diff = coeff_diff_update / 100
            coeff_diff_update = coeff_diff_update / 100
            logger.info(f'coeff_L1: {coeff_L1} coeff_diff: {coeff_diff} coeff_diff_update: {coeff_diff_update}')
            print(f'coeff_L1: {coeff_L1} coeff_diff: {coeff_diff} coeff_diff_update: {coeff_diff_update}')

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        if particle_batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / particle_batch_ratio * 0.2)
        else:
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2 // max(recursive_loop,1))

        plot_frequency = int(Niter // 20)
        print(f'{Niter} iterations per epoch')
        logger.info(f'{Niter} iterations per epoch')
        print(f'plot every {plot_frequency} iterations')

        total_loss = 0
        k = 0

        for N in trange(Niter):

            if has_Siren | has_ghost:
                optimizer_f.zero_grad()
            optimizer.zero_grad()

            dataset_batch = []
            ids_batch = []
            ids_index = 0

            loss = 0
            run = np.random.randint(n_runs - 1)

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 5 - batch_size - time_step)

                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                if not(torch.isnan(x).any()):
                    if has_field:
                        if 'visual' in field_type:
                            x[:n_nodes, 8:9] = model_f(time=k / n_frames) ** 2
                            x[n_nodes:n_particles, 8:9] = 1
                        elif 'learnable_short_term_plasticity' in field_type:
                            alpha = (k % model.embedding_step) / model.embedding_step
                            x[:, 8] = alpha * model.b[:, k // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,k // model.embedding_step] ** 2
                            loss = loss + (model.b[:, 1:] - model.b[:, :-1]).norm(2) * coeff_model_b
                        elif ('Siren_short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
                            t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                            t[:, 0, :] = torch.tensor(k / n_frames, dtype=torch.float32, device=device)
                            if 'derivative' in field_type:
                                m = model_f(t).squeeze() ** 2
                                x[:, 8] = m
                                m_next = model_f(t + 1.0E-3).squeeze() ** 2
                                grad = (m_next - m) / 1.0E-3
                                in_modulation = torch.cat((x[:, 6:7].clone().detach(), m[:, None]), dim=1)
                                pred_modulation = model.lin_modulation(in_modulation)
                                loss += (grad - pred_modulation.squeeze()).norm(2) * coeff_lin_modulation
                            else:
                                x[:, 8] = model_f(t) ** 2
                        else:
                            x[:, 8:9] = model_f(time=k / n_frames) ** 2
                    else:
                        x[:, 8:9] = torch.ones_like(x[:, 0:1])
                    if has_ghost:
                        t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                        t[:, 0, :] = torch.tensor(k / n_frames, dtype=torch.float32, device=device)
                        missing_activity = model_missing_activity[run](t).squeeze()
                        x_ghost = torch.zeros((n_ghosts, x.shape[1]), device=device)
                        x_ghost[:, 6:7] = missing_activity[:, None]
                        x_ghost[:, 0] = n_particles + torch.arange(n_ghosts, device=device)
                        x = torch.cat((x, x_ghost), 0)
                        ids = np.arange(n_particles)
                        edges = edges_all.clone().detach()
                        mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                        edges = edges[:, mask]

                    # regularisations
                    in_features = get_in_features_update(rr=None, model=model, device=device)

                    func_phi = model.lin_phi(in_features.float())

                    # sparsity on Wij and phi(0)=0
                    loss += model.W[:n_particles,:n_particles].norm(1) * coeff_L1 + func_phi.norm(2)
                    if has_ghost and (train_config.coeff_L1_ghost>0):
                        loss += model.W[:n_particles, n_particles:].norm(1) * train_config.coeff_L1_ghost
                        # fig = plt.figure(figsize=(8, 8))
                        # plt.imshow(to_numpy(model.W[:n_particles, n_particles:]))
                        # plt.savefig('W_ghost.tif')
                        # plt.close()

                    # lin.edge monotonic positive
                    if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N7'):
                        in_features_prev = torch.cat((x[:n_particles, 6:7] - xnorm/150, model.a[:n_particles]), dim=1)
                        in_features = torch.cat((x[:n_particles, 6:7], model.a[:n_particles]), dim=1)
                        in_features_next = torch.cat((x[:n_particles, 6:7] + xnorm/150, model.a[:n_particles]), dim=1)
                        if model.embedding_trial:
                            in_features_prev = torch.cat((in_features_prev, model.b[0].repeat(n_particles, 1)), dim=1)
                            in_features = torch.cat((in_features, model.b[0].repeat(n_particles, 1)), dim=1)
                            in_features_next = torch.cat((in_features_next, model.b[0].repeat(n_particles, 1)), dim=1)
                    elif model_config.signal_model_name == 'PDE_N5':
                        if model.embedding_trial:
                            in_features_prev = torch.cat((x[:n_particles, 6:7] - xnorm / 150, model.a[:n_particles], model.b[0].repeat(n_particles, 1), model.a[:n_particles], model.b[0].repeat(n_particles, 1)), dim=1)
                            in_features = torch.cat((x[:n_particles, 6:7], model.a[:n_particles], model.b[0].repeat(n_particles, 1), model.a[:n_particles], model.b[0].repeat(n_particles, 1)), dim=1)
                            in_features_next = torch.cat((x[:n_particles, 6:7] + xnorm/150, model.a[:n_particles], model.b[0].repeat(n_particles, 1), model.a[:n_particles], model.b[0].repeat(n_particles, 1)), dim=1)
                        else:
                            in_features_prev = torch.cat((x[:n_particles, 6:7] - xnorm / 150, model.a[:n_particles], model.a[:n_particles]), dim=1)
                            in_features = torch.cat((x[:n_particles, 6:7], model.a[:n_particles], model.a[:n_particles]), dim=1)
                            in_features_next = torch.cat((x[:n_particles, 6:7] + xnorm/150, model.a[:n_particles], model.a[:n_particles]), dim=1)
                    else:
                        in_features = x[:n_particles, 6:7]
                        in_features_next = x[:n_particles, 6:7] + xnorm/150
                        in_features_prev = x[:n_particles, 6:7] - xnorm / 150

                    if coeff_diff > 0:
                        if model_config.lin_edge_positive:
                            msg_1 = model.lin_edge(in_features_prev) ** 2
                            msg0 = model.lin_edge(in_features)**2
                            msg1 = model.lin_edge(in_features_next)**2
                        else:
                            msg_1 = model.lin_edge(in_features_prev)
                            msg0 = model.lin_edge(in_features)
                            msg1 = model.lin_edge(in_features_next)
                        loss = loss + torch.relu(msg0 - msg1).norm(2) * coeff_diff

                    if coeff_sign > 0:
                        
                        W_sign = torch.tanh(5 * model.W)
                        loss_contribs = []
                        for i in range(n_particles):
                            indices = index_weight[i]
                            if len(indices) > 0:
                                values = W_sign[indices, i]
                                std = torch.std(values, unbiased=False)
                                loss_contribs.append(std)
                        if loss_contribs:
                            loss = loss + torch.stack(loss_contribs).norm(2) * coeff_sign


                    if (model.update_type == 'generic') & (coeff_diff_update>0):
                        in_feature_update = torch.cat((torch.zeros((n_particles,1), device=device), model.a[:n_particles], msg0, torch.ones((n_particles,1), device=device)), dim=1)
                        in_feature_update_next = torch.cat((torch.zeros((n_particles, 1), device=device), model.a[:n_particles], msg1, torch.ones((n_particles, 1), device=device)), dim=1)
                        if 'positive' in train_config.diff_update_regul:
                            loss = loss + torch.relu(model.lin_phi(in_feature_update) - model.lin_phi(in_feature_update_next)).norm(2) * coeff_diff_update
                        if 'TV' in train_config.diff_update_regul:
                            in_feature_update_next_bis = torch.cat((torch.zeros((n_particles, 1), device=device), model.a[:n_particles], msg1, torch.ones((n_particles, 1), device=device)*1.1), dim=1)
                            loss = loss + (model.lin_phi(in_feature_update) - model.lin_phi(in_feature_update_next_bis)).norm(2) * coeff_diff_update
                        if 'second_derivative' in train_config.diff_update_regul:
                            in_feature_update_prev = torch.cat((torch.zeros((n_particles, 1), device=device), model.a[:n_particles], msg_1, torch.ones((n_particles, 1), device=device)), dim=1)
                            loss = loss + (model.lin_phi(in_feature_update_prev) + model.lin_phi(in_feature_update_next) - 2 * model.lin_phi(in_feature_update)).norm(2) * coeff_diff_update

                    if particle_batch_ratio < 1:
                         ids = np.random.permutation(x.shape[0])[:int(x.shape[0] * particle_batch_ratio)]
                         ids = np.sort(ids)
                         edges = edges_all.clone().detach()
                         mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                         edges = edges[:, mask]

                    if recursive_loop > 1:
                        y = torch.tensor(y_list[run][k+recursive_loop], device=device) / ynorm
                    elif time_step == 1:
                        y = torch.tensor(y_list[run][k], device=device) / ynorm
                    elif time_step >1:
                        y = torch.tensor(x_list[run][k + time_step,:,6:7], device=device).clone().detach()

                    if not(torch.isnan(y).any()):

                        dataset = data.Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.ones((x.shape[0],1), dtype=torch.int) * run
                            x_batch = x[:, 6:7]
                            y_batch = y
                            k_batch = torch.ones((x.shape[0],1), dtype=torch.int, device = device) * k
                            if has_ghost | (particle_batch_ratio < 1) :
                                ids_batch = ids
                        else:
                            data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 6:7]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            k_batch = torch.cat((k_batch, torch.ones((x.shape[0],1), dtype=torch.int, device = device) * k), dim = 0)
                            if has_ghost | (particle_batch_ratio < 1) :
                                ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                        ids_index += x.shape[0]

            if not(dataset_batch==[]):
                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)

                for batch in batch_loader:
                    pred = model(batch, data_id=data_id, k=k_batch)

                if recursive_loop > 1:
                    for loop in range(recursive_loop):

                        ids_index = 0
                        for batch in range(batch_size):
                            x = dataset_batch[batch].x.clone().detach()
                            u = x[:, 6:7]
                            du = pred[ids_index:ids_index+x.shape[0]]
                            x[:, 6:7] = u + du * delta_t
                            x[:, 7:8] = du
                            dataset_batch[batch].x = x
                            ids_index += x.shape[0]

                        batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                        for batch in batch_loader:
                            if ('PDE_N3' in model_config.signal_model_name):
                                pred = model(batch, k=k_batch)
                            else:
                                pred = model(batch)

                if time_step == 1:
                    if has_ghost:
                        loss = loss + (pred[ids_batch] - y_batch).norm(2)
                    elif particle_batch_ratio < 1:
                        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)
                    else:
                        loss = loss + (pred - y_batch).norm(2)
                elif time_step > 1:
                    if has_ghost:
                        loss = loss + (x_batch[ids_batch] + pred[ids_batch] * delta_t * time_step - y_batch).norm(2) / time_step
                    elif particle_batch_ratio < 1:
                        loss = loss + (x_batch[ids_batch] + pred[ids_batch] * delta_t * time_step - y_batch[ids_batch]).norm(2) / time_step
                    else:
                        loss = loss + ( x_batch + pred * delta_t * time_step - y_batch).norm(2) / time_step

                    # if run == 0:
                    #     plt.figure(figsize=(15, 10))
                    #     n = np.random.permutation(n_particles)
                    #     for i in range(5):
                    #         plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
                    #         plt.scatter(k, to_numpy(x[n[i].astype(int), 6:7]), color='k', s=50)
                    #         plt.scatter(k+time_step, to_numpy(y_batch[n[i].astype(int)]), color='k', s=50)
                    #         plt.scatter(k+time_step, to_numpy(x_batch[n[i].astype(int)] + pred[n[i].astype(int)] * delta_t * time_step), color='r', s=100, marker='x')
                    #     plt.xlabel('time', fontsize=64)
                    #     plt.ylabel('$x_{i}$', fontsize=64)
                    #     plt.xlim([k-64, k+64])
                    #     # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
                    #     plt.xticks(fontsize=28)
                    #     plt.yticks(fontsize=28)
                    #     plt.title(r'$x_i$ samples', fontsize=48)
                    #     plt.tight_layout()
                    #     plt.savefig(f'activity.tif', dpi=300)
                    #     plt.close()

                if ('PDE_N3' in model_config.signal_model_name):
                    loss = loss + train_config.coeff_model_a * (model.a[ind_a+1] - model.a[ind_a]).norm(2)

                loss.backward()
                optimizer.step()
                if has_Siren | has_ghost:
                    optimizer_f.step()
                total_loss += loss.item()

                with torch.no_grad():
                    model.W.copy_(model.W * model.mask)

                visualize_embedding = True
                if visualize_embedding & ((N % plot_frequency == 0) | (N == 0)):
                    with torch.no_grad():
                        plot_training_signal(config, model, adjacency, xnorm, log_dir, epoch, N, n_particles,
                                             n_particle_types, type_list, cmap, device)

                        if has_field:
                            plot_training_signal_field(x, n_nodes, n_nodes_per_axis, recursive_loop, k, time_step, x_list, run, model, field_type, model_f,
                                                       edges, y_list, ynorm, delta_t, n_frames, log_dir, epoch, N,
                                                       recursive_parameters, modulation, device)
                            if 'learnable_short_term_plasticity' not in field_type:
                                torch.save({'model_state_dict': model_f.state_dict(),'optimizer_state_dict': optimizer_f.state_dict()}, os.path.join(log_dir,
                                            'models',f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                        if has_ghost:
                            torch.save({'model_state_dict': model_missing_activity.state_dict(),'optimizer_state_dict': optimizer_f.state_dict()}, os.path.join(log_dir,
                                            'models',f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

                        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles ))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles ))
        logger.info(f'recursive_parameters: {recursive_parameters[0]:.2f}')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if has_Siren:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_f.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append(total_loss / (N + 1) / n_particles )
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))


        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(2, 5, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(2, 5, 2)
        for n in range(n_particle_types):
            pos = torch.argwhere(type_list == n)
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
        pred_weight = to_numpy(A[:n_particles, :n_particles])
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

        ax.text(0.01, 0.99, f'$R^2$ {r_squared:0.3f}   slope {lin_fit[0]:0.3f}', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left')

        all_func_values = []
        ax = fig.add_subplot(2, 5, 10)
        rr = torch.linspace(-xnorm, xnorm, 1000, device=device)
        for n in range(n_particles):
            embedding_ = model.a[n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
            in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=model_config.signal_model_name, max_radius=simulation_config.max_radius)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            if model_config.lin_edge_positive:
                func = func ** 2
            all_func_values.append(func)
            if (n % 2 == 0):
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=2, alpha=0.25)
        all_func_values = torch.cat(all_func_values)
        y_min, y_max = all_func_values.min().item(), all_func_values.max().item()
        plt.ylim([y_min-0.1, y_max*1.1])

        if 'PDE_N3' not in model_config.signal_model_name:

            ax = fig.add_subplot(2, 5, 6)
            embedding = to_numpy(model.a.squeeze())

            if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5') | (model_config.signal_model_name == 'PDE_N9'):
                model_MLP = model.lin_edge
                update_type = 'NA'
            else:
                model_MLP = model.lin_phi
                update_type = model.update_type

            func_list_, proj_interaction = analyze_edge_function(rr=torch.linspace(-xnorm, xnorm, 1000, device=device), vizualize=False, config=config,
                                                                model_MLP=model_MLP, model=model,
                                                                n_nodes=0,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                                cmap=cmap, update_type = update_type, device=device)

            model_MLP = model.lin_phi
            update_type = model.update_type

            func_list, proj_interaction_ = analyze_edge_function(rr=torch.linspace(-xnorm, xnorm, 1000, device=device), vizualize=True, config=config,
                                                                model_MLP=model_MLP, model=model,
                                                                n_nodes=0,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                                cmap=cmap, update_type = update_type, device=device)

            labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                              train_config.cluster_distance_threshold, type_list,
                                                              n_particle_types, embedding_cluster)

            if has_ghost:
                labels = labels[:n_particles]
                new_labels = new_labels[:n_particles]
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
            ax.text(0.01, 0.99, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left')

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
                with torch.no_grad():
                    model.a.copy_(model_a_)
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
                    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, lr_modulation=lr_modulation)
                    for sub_epochs in trange(20):
                        rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
                        pred = []
                        optimizer.zero_grad()
                        for n in range(n_particles):
                            embedding_ = model.a[n, :].clone().detach() * torch.ones((1000, model_config.embedding_dim),device=device)
                            in_features = get_in_features_update(rr=rr[:,None], model=model, device=device)
                            pred.append(model.lin_phi(in_features.float()))
                        pred = torch.stack(pred)
                        loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                        logger.info(f'    loss: {np.round(loss.item() / n_particles, 3)}')
                        loss.backward()
                        optimizer.step()
                if train_config.fix_cluster_embedding:
                    lr = 1E-12
                    lr_embedding = 1E-12
                    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, lr_modulation=lr_modulation)
                    logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')
            else:
                lr = train_config.learning_rate_start
                lr_embedding = train_config.learning_rate_embedding_start
                optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, lr_modulation=lr_modulation)
                logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')

            if (epoch == 20) & (train_config.coeff_anneal_L1 > 0):
                coeff_L1 = train_config.coeff_anneal_L1
                logger.info(f'coeff_L1: {coeff_L1}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()


def data_train_agents(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

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

    log_dir, logger = create_log_dir(config, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"{log_dir}/models/best_model_with_1_graphs_3.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    print('Load data ...')

    time_series, signal = _load_agent_data(dataset_name, device=device)

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
    logger.info(f"total trainable Params: {n_total_params}")
    logger.info(f'learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"{log_dir}/models/best_model_with_1_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    n_particles = config.simulation.n_particles
    print(f'N particles: {n_particles}')
    logger.info(f'N particles:  {n_particles}')

    if os.path.exists(f'{log_dir}/edge_p_p_list.npz'):
        print('Load list of edges index ...')
        edge_p_p_list = np.load(f'{log_dir}/edge_p_p_list.npz')
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
        np.savez(f'{log_dir}/edge_p_p_list', *edge_p_p_list)

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
                pred = model(batch, data_id=0, training=True, phi=phi, frame=k)

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


def data_train_WBI(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

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

    log_dir, logger = create_log_dir(config, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
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
    # net = f"{log_dir}/models/best_model_with_1_graphs_0_0.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"total trainable Params: {n_total_params}")
    logger.info(f'learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
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
    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
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
                plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
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
              ratio=1, run=1, test_mode='', sample_embedding=False, particle_of_interest=1, device=[]):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    n_ghosts = config.training.n_ghosts
    has_ghost = config.training.n_ghosts > 0
    has_bounding_box = 'PDE_F' in model_config.particle_model_name
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    time_window = training_config.time_window
    time_step = training_config.time_step
    sub_sampling = simulation_config.sub_sampling
    bounce_coeff = simulation_config.bounce_coeff
    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension
    has_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)

    do_tracking = training_config.do_tracking
    has_state = (config.simulation.state_type != 'discrete')
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))

    log_dir = 'log/' + config.config_file
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
    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"

    n_sub_population = n_particles // n_particle_types
    first_cell_id_particles = []
    for n in range(n_particle_types):
        index = np.arange(n_particles * n // n_particle_types, n_particles * (n + 1) // n_particle_types)
        first_cell_id_particles.append(index)

    print(f'load data run {run} ...')
    if only_mesh:
        vnorm = torch.tensor(1.0, device=device)
        ynorm = torch.tensor(1.0, device=device)
        hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)
        x_mesh_list = []
        y_mesh_list = []
        time.sleep(0.5)
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
        x_list = x_mesh_list
        y_list = y_mesh_list
    elif has_field:
        x_list = []
        y_list = []
        x_mesh_list = []
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)
        x_list.append(torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device))
        y_list.append(torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device))
        ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True).to(device)
        vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True).to(device)
        x = x_list[0][0].clone().detach()
        n_particles = x.shape[0]
        config.simulation.n_particles = n_particles
        index_particles = get_index_particles(x, n_particle_types, dimension)
        x_mesh = x_mesh_list[0][0].clone().detach()
    else:
        x_list = []
        y_list = []
        if (model_config.signal_model_name == 'PDE_N'):
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
            x_list.append(torch.tensor(x, device=device))
            y_list.append(torch.tensor(y, device=device))
        elif (model_config.particle_model_name == 'PDE_M'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            x_list.append(x)
        else:
            if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
                x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
                y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
                x_list.append(x)
                y_list.append(y)
            else:
                x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
                x = torch.tensor(x, dtype=torch.float32, device=device)
                y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
                y = torch.tensor(y, dtype=torch.float32, device=device)
                x_list.append(x)
                y_list.append(y)

                x = x_list[0][0].clone().detach()
                if ('PDE_MLPs' not in model_config.particle_model_name) & ('PDE_F' not in model_config.particle_model_name) & ('PDE_WF' not in model_config.particle_model_name):
                    n_particles = int(x.shape[0] / ratio)
                    config.simulation.n_particles = n_particles
                n_frames = len(x_list[0])
                index_particles = get_index_particles(x, n_particle_types, dimension)
                if n_particle_types > 1000:
                    index_particles = []
                    for n in range(3):
                        index = np.arange(n_particles * n // 3, n_particles * (n + 1) // 3)
                        index_particles.append(index)
                        n_particle_types = 3

        ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
        vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True )

    if do_tracking | has_state:
        for k in range(len(x_list[0])):
            type = x_list[0][k][:, 2*dimension+1]
            if k == 0:
                type_list = type
            else:
                type_list = torch.concatenate((type_list, type))
        n_particles_max = len(type_list) + 1
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
    if has_ghost & ('PDE_N' not in model_config.signal_model_name):
        model_ghost = Ghost_Particles(config, n_particles, vnorm, device)
        net = f"{log_dir}/models/best_ghost_particles_with_{n_runs - 1}_graphs_20.pt"
        state_dict = torch.load(net, map_location=device)
        model_ghost.load_state_dict(state_dict['model_state_dict'])
        model_ghost.eval()
        x_removed_list = torch.load(f'graphs_data/{dataset_name}/x_removed_list_0.pt', map_location=device)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    if has_mesh:
        hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)

        mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_{run}.pt', map_location=device)
        mask_mesh = mesh_data['mask']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']

        xy = to_numpy(mesh_data['mesh_pos'])
        x_ = xy[:, 0]
        y_ = xy[:, 1]
        mask = to_numpy(mask_mesh)
        mask_mesh = (x_ > np.min(x_) + 0.02) & (x_ < np.max(x_) - 0.02) & (y_ > np.min(y_) + 0.02) & (y_ < np.max(y_) - 0.02)
        mask_mesh = torch.tensor(mask_mesh, dtype=torch.bool, device=device)

        if 'WaveMeshSmooth' in model_config.mesh_model_name:
            with torch.no_grad():
                distance = torch.sum((mesh_data['mesh_pos'][:, None, :] - mesh_data['mesh_pos'][None, :, :]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
                edge_index_mesh = adj_t.nonzero().t().contiguous()


        # plt.scatter(x_, y_, s=2, c=to_numpy(mask_mesh))
    if has_adjacency_matrix:
        if model_config.signal_model_name == 'PDE_N':
            mat = scipy.io.loadmat(simulation_config.connectivity_file)
            adjacency = torch.tensor(mat['A'], device=device)
            adj_t = adjacency > 0
            edge_index = adj_t.nonzero().t().contiguous()
            edge_attr_adjacency = adjacency[adj_t]
        else:
            adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
            adjacency_ = adjacency.t().clone().detach()
            adj_t = torch.abs(adjacency_) > 0
            edge_index = adj_t.nonzero().t().contiguous()

    if 'PDE_N' in model_config.signal_model_name:
        has_adjacency_matrix = True
        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        edge_index = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
        if ('modulation' in model_config.field_type) | ('visual' in model_config.field_type):
            print('load b_i movie ...')
            im = imread(f"graphs_data/{simulation_config.node_value_map}")
            A1 = torch.zeros((n_particles, 1), device=device)
        # else:
        #     print('create graph positions ...')
        #     G_first = to_networkx(first_dataset.clone().detach(), remove_self_loops=True, to_undirected=True)
        #     first_positions = nx.spring_layout(G_first, weight='weight', seed=42, k=1)
        #     X1_first = torch.zeros((n_particles, 2), device=device)
        #     for node, pos in first_positions.items():
        #         X1_first[node, :] = torch.tensor([pos[0], pos[1]], device=device)
        #     print('done ...')

        # neuron_index = torch.randint(0, n_particles, (6,))
        neuron_gt_list = []
        neuron_pred_list = []
        modulation_gt_list = []
        modulation_pred_list = []

        if os.path.exists(f'./graphs_data/{dataset_name}/X1.pt') > 0:
            X1_first = torch.load(f'./graphs_data/{dataset_name}/X1.pt', map_location=device)
            X_msg = torch.load(f'./graphs_data/{dataset_name}/X_msg.pt', map_location=device)
        else:
            xc, yc = get_equidistant_points(n_points=n_particles)
            X1_first = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
            perm = torch.randperm(X1_first.size(0))
            X1_first = X1_first[perm]
            torch.save(X1_first, f'./graphs_data/{dataset_name}/X1_first.pt')
            xc, yc = get_equidistant_points(n_points=n_particles**2)
            X_msg = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
            perm = torch.randperm(X_msg.size(0))
            X_msg = X_msg[perm]
            torch.save(X_msg, f'./graphs_data/{dataset_name}/X_msg.pt')

    if verbose:
        print(f'test data ... {model_config.particle_model_name} {model_config.mesh_model_name}')
        print('log_dir: {}'.format(log_dir))
        print(f'network: {net}')
        print(table)
        print(f"total trainable Params: {total_params}")

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    model.particle_of_interest = particle_of_interest
    if 'PDE_MLPs_A' in config.graph_model.particle_model_name:
        model.model = config.graph_model.particle_model_name + '_eval'

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if 'test_simulation' in 'test_mode':
        model, bc_pos, bc_dpos = choose_model(config, device=device)
    else:
        if has_mesh:
            mesh_model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            mesh_model.load_state_dict(state_dict['model_state_dict'])
            mesh_model.eval()
        else:
            state_dict = torch.load(net, map_location=device, weights_only=True)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            mesh_model = None
            if 'PDE_K' in model_config.particle_model_name:
                model.connection_matrix = torch.load(f'graphs_data/{dataset_name}/connection_matrix_list.pt',
                                                     map_location=device)
                timeit = np.load(f'graphs_data/{dataset_name}/times_train_springs_example.npy',
                                 allow_pickle=True)
                timeit = timeit[run][0]
                time_id = 0
            if 'PDE_N' in model_config.signal_model_name:
                if ('Siren_short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
                    model_f = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega, hidden_omega_0=model_config.omega,
                                    outermost_linear=model_config.outermost_linear_nnr)
                    net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])
                    model_f.to(device=device)
                    model_f.eval()
                if ('modulation' in model_config.field_type) | ('visual' in model_config.field_type):
                    model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                            out_features=model_config.output_size_nnr,
                                            hidden_features=model_config.hidden_dim_nnr,
                                            hidden_layers=model_config.n_layers_nnr, outermost_linear=True,
                                            device=device,
                                            first_omega_0=model_config.omega, hidden_omega_0=model_config.omega)
                    net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])
                    model_f.to(device=device)
                    model_f.eval()
                if has_ghost & ('PDE_N' in model_config.signal_model_name):
                    print('load missing activity models ...')
                    model_missing_activity = nn.ModuleList([
                        Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                              hidden_features=model_config.hidden_dim_nnr,
                              hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega, hidden_omega_0=model_config.omega,
                              outermost_linear=model_config.outermost_linear_nnr)
                        for n in range(n_runs)
                    ])
                    model_missing_activity.to(device=device)
                    net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{best_model}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_missing_activity.load_state_dict(state_dict['model_state_dict'])

        if has_field:
            model_f_p = model
            image_width = int(np.sqrt(n_nodes))
            if 'siren_with_time' in model_config.field_type:
                model_f = Siren_Network(image_width=image_width, in_features=3, out_features=1, hidden_features=128,
                                        hidden_layers=5, outermost_linear=True, device=device, first_omega_0=80,
                                        hidden_omega_0=80.)
                net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
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
    x_inference_list = []

    for it in trange(start_it, start_it+200):   #  start_it+200): # min(9600+start_it,stop_it-time_step)):

        check_and_clear_memory(device=device, iteration_number=it, every_n_iterations=25, memory_percentage_threshold=0.6)
        # print(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        # print(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        # with torch.no_grad():
        if it < n_frames - 4:
            x0 = x_list[0][it].clone().detach()
            x0_next = x_list[0][(it+time_step)].clone().detach()
            if not(model_config.particle_model_name == 'PDE_M'):
                y0 = y_list[0][it].clone().detach()
        if has_mesh:
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset_mesh = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
        if (time_step>1) | do_tracking:
            x = x0.clone().detach()

        # error calculations
        if 'PDE_N' in model_config.signal_model_name:
            rmserr = torch.sqrt(torch.mean(torch.sum(bc_dpos(x[:n_particles, 6:7] - x0[:, 6:7]) ** 2, axis=1)))
            neuron_gt_list.append(x0[:, 6:7])
            neuron_pred_list.append(x[:n_particles, 6:7].clone().detach())
            if ('Siren_short_term_plasticity' in field_type) | ('modulation' in field_type):
                modulation_gt_list.append(x0[:, 8:9])
                modulation_pred_list.append(x[:, 8:9].clone().detach())
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

        if config.training.shared_embedding:
            data_id = torch.ones((n_particles, 1), dtype=torch.int)
        else:
            data_id = torch.ones((n_particles, 1), dtype=torch.int) * run

        # update calculations
        if model_config.mesh_model_name == 'DiffMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0, )
            x[:, 6:7] += pred * hnorm * delta_t
        elif model_config.mesh_model_name == 'WaveMeshSmooth':
            pred = mesh_model(dataset_mesh, data_id=data_id, training=False, phi=torch.zeros(1, device=device))
            with torch.no_grad():
                x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
                x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif model_config.mesh_model_name == 'WaveMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=data_id)
            x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
            x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif 'RD_RPS_Mesh' in model_config.mesh_model_name == 'RD_RPS_Mesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=data_id)
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
                y0 = model(dataset_p_p, data_id=1, training=False, phi=torch.zeros(1, device=device),
                           has_field=False)
                y1 = model_f_p(dataset_f_p, data_id=1, training=False, phi=torch.zeros(1, device=device),
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

            if has_ghost & ('PDE_N' not in model_config.signal_model_name):
                x_ = x
                x_ghost = model_ghost.get_pos(dataset_id=run, frame=it, bc_pos=bc_pos)
                x_ = torch.cat((x_, x_ghost), 0)

            # compute connectivity and prediction
            if has_adjacency_matrix:
                if 'PDE_N' in model_config.signal_model_name:
                    if 'visual' in field_type:
                        x[:n_nodes, 8:9] = model_f(time=it / n_frames) ** 2
                        x[n_nodes:n_particles, 8:9] = 1
                    elif 'learnable_short_term_plasticity' in field_type:
                        alpha = (k % model.embedding_step) / model.embedding_step
                        x[:, 8] = alpha * model.b[:, it // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,it // model.embedding_step] ** 2
                    elif ('Siren_short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
                        t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                        t[:, 0, :] = torch.tensor(it / n_frames, dtype=torch.float32, device=device)
                        x[:, 8] = model_f(t).squeeze() ** 2
                    elif 'modulation' in field_type:
                        x[:, 8:9] = model_f(time=it / n_frames) ** 2
                    if has_ghost & ('PDE_N' in model_config.signal_model_name):
                        t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                        t[:, 0, :] = torch.tensor(it / n_frames, dtype=torch.float32, device=device)
                        missing_activity = model_missing_activity[run](t).squeeze()
                        x_ghost = torch.zeros((n_ghosts, x.shape[1]), device=device)
                        x_ghost[:, 6:7] = missing_activity[:, None]
                        x_ghost[:, 0] = n_particles + torch.arange(n_ghosts, device=device)
                        x = torch.cat((x[:n_particles], x_ghost), 0)
                        edges, edge_attr = dense_to_sparse(torch.ones((n_particles + n_ghosts)) - torch.eye(n_particles + n_ghosts))
                        edges = edges.to(device=device)
                        ids = np.arange(n_particles)
                        mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                        edge_index = edges[:, mask]
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                if 'test_simulation' in test_mode:
                    y = y0 / ynorm
                else:
                    with torch.no_grad():
                        pred = model(dataset, data_id=data_id)
                        y = pred
            else:
                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()

                if time_window > 0:
                    xt = []
                    for t in range(time_window):
                        x_ = x_list[0][it - t].clone().detach()
                        xt.append(x_[:, :])
                    dataset = data.Data(x=xt, edge_index=edge_index)
                else:
                    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                if 'test_simulation' in test_mode:
                    y = y0 / ynorm
                    pred = y
                else:
                    pred = model(dataset, data_id=data_id, training=False, k=it)
                    y = pred

                if has_ghost:
                    y = y[mask_ghost]

            with torch.no_grad():
                if sub_sampling > 1:
                    # predict position, does not work with rotation_augmentation
                    if time_step == 1:
                        x_next = bc_pos(y[:,0:dimension])
                    elif time_step == 2:
                        x_next = bc_pos(y[:,dimension:2*dimension])

                    x[:, dimension + 1:2 * dimension + 1] = (x_next - x[:, 1:dimension + 1]) / delta_t
                    x[:, 1:dimension + 1] = x_next
                    loss = (x[:, 1:dimension + 1] - x0_next[:, 1:dimension + 1]).norm(2)
                    pred_err_list.append(to_numpy(torch.sqrt(loss)))
                elif do_tracking:

                    # x[:, dimension + 1:2 * dimension + 1] = y
                    # x[:, 1:dimension + 1] = bc_pos(x[:, 1:dimension + 1] + x[:, dimension + 1:2 * dimension + 1] * delta_t)

                    x_pos_next = x0_next[:, 1:dimension + 1].clone().detach()
                    if pred.shape[1] != dimension:
                        pred = torch.cat((pred, torch.zeros(pred.shape[0], 1, device=pred.device)), dim= 1)
                    if model_config.prediction == '2nd_derivative':
                        x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * (x[:, dimension + 1:2*dimension + 1] + delta_t * time_step * pred * ynorm))
                    else:
                        x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * pred * ynorm)
                    distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                    result = distance.min(dim=1)
                    min_value = result.values
                    indices = result.indices
                    loss = torch.std(torch.sqrt(min_value))
                    pred_err_list.append(to_numpy(torch.sqrt(loss)))
                    if 'inference' in test_mode:
                        x[:,dimension+1:2*dimension+1] = pred.clone().detach() / (delta_t * time_step)
                elif ('PDE_N' in model_config.signal_model_name) & (time_step>1):
                    x[:n_particles, 6:7] += y[:n_particles] * delta_t * time_step
                    loss = (x[:n_particles, 6:7] - x0_next[:n_particles, 6:7]).norm(2) * 0
                    pred_err_list.append(to_numpy(torch.sqrt(loss)))
                else:
                    loss = (pred[:n_particles, 0:dimension] * ynorm - y0).norm(2)

                    # matplotlib.use("Qt5Agg")
                    # fig = plt.figure()
                    # plt.scatter(to_numpy(y0), to_numpy(ynorm*pred[:n_particles, 0:dimension]))
                    # plt.xlabel('y0')
                    # plt.ylabel('pred')
                    # plt.savefig(f'pred_{it}.png')
                    # plt.close()

                    pred_err_list.append(to_numpy(torch.sqrt(loss)))
                    if model_config.prediction == '2nd_derivative':
                        y = y * ynorm * delta_t
                        x[:n_particles, dimension + 1:2 * dimension + 1] = x[:n_particles, dimension + 1:2 * dimension + 1] + y[:n_particles]  # speed update
                    else:
                        y = y * vnorm
                        if 'PDE_N' in model_config.signal_model_name:
                            x[:n_particles, 6:7] += y[:n_particles] * delta_t  # signal update
                        else:
                            x[:n_particles, dimension + 1:2 * dimension + 1] = y[:n_particles]

                    x[:, 1:dimension + 1] = bc_pos(x[:, 1:dimension + 1] + x[:, dimension + 1:2 * dimension + 1] * delta_t)  # position update

                if 'bounce_all_v0' in test_mode:
                    gap = 0.005
                    bouncing_pos = torch.argwhere((x[:, 1] <= 0.1-gap) | (x[:, 1] >= 0.9+gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 3] = - 0.7 * x[bouncing_pos, 3]
                        x[bouncing_pos, 1] += x[bouncing_pos, 3] * delta_t
                    bouncing_pos = torch.argwhere((x[:, 2] <= 0.1-gap) | (x[:, 2] >= 0.9+gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 4] = - 0.7 * x[bouncing_pos, 4]
                        x[bouncing_pos, 2] += x[bouncing_pos, 4] * delta_t
                if 'bounce_all' in test_mode:
                    gap = 0.005
                    bouncing_pos = torch.argwhere((x[:, 1] <= 0.1 + gap) | (x[:, 1] >= 0.9 - gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 3] = - 0.7 * bounce_coeff* x[bouncing_pos, 3]
                        x[bouncing_pos, 1] += x[bouncing_pos, 3] * delta_t * 10
                    bouncing_pos = torch.argwhere((x[:, 2] <= 0.1 + gap) | (x[:, 2] >= 0.9 - gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 4] = - 0.7 * bounce_coeff * x[bouncing_pos, 4]
                        x[bouncing_pos, 2] += x[bouncing_pos, 4] * delta_t * 10

                if (time_window>1) & ('plot_data' not in test_mode):
                    moving_pos = torch.argwhere(x[:,5]!=0)
                    x_list[0][it+1,moving_pos.squeeze(),1:2 * dimension + 1] = x[moving_pos.squeeze(), 1:2 * dimension + 1].clone().detach()
                if 'fixed' in test_mode:
                    fixed_pos = torch.argwhere(x[:,5]==0)
                    x[fixed_pos.squeeze(), 1:2 * dimension + 1] = x_list[0][it+1,fixed_pos.squeeze(),1:2 * dimension + 1].clone().detach()
                if 'inference' in test_mode:
                    x_inference_list.append(x)

            # vizualization
            if 'plot_data' in test_mode:
                x = x_list[0][it].clone().detach()

            if (it % step == 0) & (it >= 0) & visualize:

                # print(f'acceleration {torch.max(y[:, 0])}')
                # print(f'speed {torch.max(x[:, 3])}')

                num = f"{it:06}"

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'black' in style:
                    plt.style.use('dark_background')
                    mc = 'w'
                else:
                    plt.style.use('default')
                    mc = 'k'

                fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                ax.tick_params(axis='both', which='major', pad=15)

                if has_mesh:
                    pts = x[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                    if model_config.mesh_model_name == 'DiffMesh':
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                    if 'WaveMesh' in model_config.mesh_model_name:
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
                    if 'plot_data' in test_mode:

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

                    # y = model(dataset, data_id=1, return_all=True)

                    matplotlib.rcParams['savefig.pad_inches'] = 0

                    if has_ghost:
                        plt.figure(figsize=(25, 25))
                        plt.scatter(to_numpy(X1_first[:n_particles, 0]), 1.1 + to_numpy(X1_first[:n_particles, 1]), s=1000, c=to_numpy(x[:n_particles, 6]), vmin=-2, vmax = 2, cmap='viridis', edgecolors='k', alpha=1)
                        plt.scatter(to_numpy(X1_first[:n_ghosts, 0]) + 1.1, 1.1 + to_numpy(X1_first[:n_ghosts, 1]), s=1000, c=4*to_numpy(x[n_particles:n_ghosts+n_particles, 6]), vmin=-2, vmax = 2, cmap='viridis', edgecolors='k', alpha=1)
                        plt.scatter(to_numpy(X1_first[:n_particles, 0]), to_numpy(X1_first[:n_particles, 1]), s=1000, c=to_numpy(x0[:n_particles, 6]), vmin=-2, vmax = 2, cmap='viridis', edgecolors='k', alpha=1)
                        plt.scatter(to_numpy(X1_first[:n_particles, 0]) + 1.1, to_numpy(X1_first[:n_particles, 1]), s=1000, c=to_numpy(y[:n_particles]* delta_t), vmin=-2, vmax = 2, cmap='viridis', edgecolors='k', alpha=1)

                        plt.axis('off')
                        plt.text(0.05, 1.0, r'neuron activity                             extra activity (*4)', fontsize=48, color='w', ha='left', va='top',
                                 transform=plt.gca().transAxes)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=170)
                        plt.close()
                    else:
                        plt.figure(figsize=(10, 10))
                        plt.scatter(to_numpy(X1_first[:, 0]), to_numpy(X1_first[:, 1]), s=200, c=to_numpy(x[:, 6]),
                                    cmap='viridis', vmin=-10, vmax=10, edgecolors='k', alpha=1)
                        plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=170)
                        plt.close()

                        im = imread(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif")
                        plt.figure(figsize=(10, 10))
                        plt.imshow(im)
                        plt.axis('off')
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

                    # plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c='w', alpha=0.25)
                    plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=10, c='w', alpha=0.5)
                    plt.scatter(to_numpy(x_pos_pred[:, 1]), to_numpy(x_pos_pred[:, 0]), s=10, c='r')
                    x1 = x_list[0][it + time_step].clone().detach()
                    plt.scatter(to_numpy(x1[:, 2]), to_numpy(x1[:, 1]), s=10, c='g')

                    plt.xticks([])
                    plt.yticks([])

                    if 'zoom' in style:
                        for m in range(x.shape[0]):
                                plt.arrow(x=to_numpy(x0[m, 2]), y=to_numpy(x0[m, 1]), dx=to_numpy(x[m, dimension+2]) * delta_t,
                                          dy=to_numpy(x[m, dimension+1]) * delta_t, head_width=0.004,
                                          length_includes_head=True, color='g')
                        plt.xlim([300,400])
                        plt.ylim([300,400])
                    else:
                        plt.xlim([0,700])
                        plt.ylim([0,700])
                    plt.tight_layout()
                else:
                    s_p = 10
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

                    if particle_of_interest > 1:

                        xc = x[particle_of_interest, 2].detach().cpu().numpy()
                        yc = x[particle_of_interest, 1].detach().cpu().numpy()
                        pos = torch.argwhere(edge_index[1,:]==particle_of_interest)
                        pos = pos[:, 0]
                        if 'zoom' in style:
                            plt.scatter(x[edge_index[0,pos], 2].detach().cpu().numpy(),
                                        x[edge_index[0,pos], 1].detach().cpu().numpy(), s=s_p * 20 , color=mc, alpha=0.25)
                        else:
                            plt.scatter(x[edge_index[0,pos], 2].detach().cpu().numpy(),
                                        x[edge_index[0,pos], 1].detach().cpu().numpy(), s=s_p * 5 , color=mc, alpha=0.25, )


                        for k in range(pos.shape[0]):
                            plt.arrow(x[edge_index[1,pos[k]], 2].detach().cpu().numpy(),
                                        x[edge_index[1,pos[k]], 1].detach().cpu().numpy(),  dx=to_numpy(model.msg[k,1]) * delta_t/20,
                                      dy=to_numpy(model.msg[k,0]) * delta_t/20, head_width=0.004, length_includes_head=True, color=mc,alpha=0.25)

                        plt.arrow(x=to_numpy(x[particle_of_interest, 2]), y=to_numpy(x[particle_of_interest, 1]), dx=to_numpy(x[particle_of_interest, 4]) * delta_t * 100,
                                  dy=to_numpy(x[particle_of_interest, 3]) * delta_t * 100, head_width=0.004, length_includes_head=True, color=cmap.color(to_numpy(x[particle_of_interest, 5]).astype(int)))

                    if 'zoom' in style:
                        plt.xlim([xc-0.1, xc+0.1])
                        plt.ylim([yc-0.1, yc+0.1])
                        plt.xticks([])
                        plt.yticks([])

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
                if 'name' in style:
                    plt.title(f"{os.path.basename(log_dir)}", fontsize=24)

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

                # save figure
                if not ('PDE_N' in model_config.signal_model_name):
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{run}_{num}.tif", dpi=100)
                    plt.close()

                if ('PDE_N' in model_config.signal_model_name) & (it % 200 == 0) & (it > 0):

                    if has_ghost & ('PDE_N' in model_config.signal_model_name):
                        n = [1, 3, 10, 15, 26, 27, 52, 62, 92, 120]
                    else:
                        n = [20, 30, 100, 150, 260, 270, 520, 620, 720, 820]

                    neuron_gt_list_ = torch.cat(neuron_gt_list, 0)
                    neuron_pred_list_ = torch.cat(neuron_pred_list, 0)
                    neuron_gt_list_ = torch.reshape(neuron_gt_list_,
                                                    (neuron_gt_list_.shape[0] // n_particles, n_particles))
                    neuron_pred_list_ = torch.reshape(neuron_pred_list_,
                                                      (neuron_pred_list_.shape[0] // n_particles, n_particles))

                    plt.figure(figsize=(20, 10))
                    if 'latex' in style:
                        plt.rcParams['text.usetex'] = True
                        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
                    ax = plt.subplot(121)
                    plt.plot(neuron_gt_list_[:, n[0]].detach().cpu().numpy(), c=mc, linewidth=8, label='true',
                             alpha=0.25)
                    plt.plot(neuron_pred_list_[:, n[0]].detach().cpu().numpy(), linewidth=4, c='k',
                             label='learned')
                    plt.legend(fontsize=24)
                    plt.plot(neuron_gt_list_[:, n[1:10]].detach().cpu().numpy(), c=mc, linewidth=8, alpha=0.25)
                    plt.plot(neuron_pred_list_[:, n[1:10]].detach().cpu().numpy(), linewidth=4)
                    plt.xlim([0, 1400])
                    plt.xlabel(r'time-points', fontsize=48)
                    plt.ylabel(r'$x_i$', fontsize=48)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.ylim([-30, 30])
                    # plt.text(40, 26, f'time: {it}', fontsize=34)
                    ax = plt.subplot(122)
                    plt.scatter(to_numpy(neuron_gt_list_[-1, :]), to_numpy(neuron_pred_list_[-1, :]), s=10, c=mc)
                    plt.xlim([-30, 30])
                    plt.ylim([-30, 30])
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    x_data = to_numpy(neuron_gt_list_[-1, :])
                    y_data = to_numpy(neuron_pred_list_[-1, :])
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    plt.xlabel(r'true $x_i$', fontsize=48)
                    plt.ylabel(r'learned $x_i$', fontsize=48)
                    plt.text(-28.5, 26, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-28.5, 22, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                    plt.tight_layout()
                    plt.savefig(f'./{log_dir}/results/comparison_xi_{it}.png', dpi=80)
                    plt.close()

                    if ('Siren_short_term_plasticity' in field_type) | ('modulation' in field_type):

                        modulation_gt_list_ = torch.cat(modulation_gt_list, 0)
                        modulation_pred_list_ = torch.cat(modulation_pred_list, 0)
                        modulation_gt_list_ = torch.reshape(modulation_gt_list_,
                                                            (modulation_gt_list_.shape[0] // n_particles, n_particles))
                        modulation_pred_list_ = torch.reshape(modulation_pred_list_,
                                                              (modulation_pred_list_.shape[0] // n_particles,
                                                               n_particles))

                        plt.figure(figsize=(20, 10))
                        if 'latex' in style:
                            plt.rcParams['text.usetex'] = True
                            rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                        ax = plt.subplot(122)
                        plt.scatter(to_numpy(modulation_gt_list_[-1, :]), to_numpy(modulation_pred_list_[-1, :]), s=10,
                                    c=mc)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                        x_data = to_numpy(modulation_gt_list_[-1, :])
                        y_data = to_numpy(modulation_pred_list_[-1, :])
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        plt.xlabel(r'true modulation', fontsize=48)
                        plt.ylabel(r'learned modulation', fontsize=48)
                        # plt.text(0.05, 0.9 * lin_fit[0], f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                        # plt.text(0.05, 0.8 * lin_fit[0], f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                        ax = plt.subplot(121)
                        plt.plot(modulation_gt_list_[:, n[0]].detach().cpu().numpy(), c='k', linewidth=8, label='true',
                                 alpha=0.25)
                        plt.plot(modulation_pred_list_[:, n[0]].detach().cpu().numpy() / lin_fit[0], linewidth=4, c='k',
                                 label='learned')
                        plt.legend(fontsize=24)
                        plt.plot(modulation_gt_list_[:, n[1:10]].detach().cpu().numpy(), c='k', linewidth=8, alpha=0.25)
                        plt.plot(modulation_pred_list_[:, n[1:10]].detach().cpu().numpy() / lin_fit[0], linewidth=4)
                        plt.xlim([0, 1400])
                        plt.xlabel(r'time-points', fontsize=48)
                        plt.ylabel(r'modulation', fontsize=48)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                        plt.ylim([0, 2])
                        # plt.text(40, 26, f'time: {it}', fontsize=34)
                        plt.tight_layout()
                        plt.tight_layout()
                        plt.savefig(f'./{log_dir}/results/comparison_modulation_{it}.png', dpi=80)
                        plt.close()

                if ('feature' in style) & ('PDE_MLPs_A' in config.graph_model.particle_model_name):
                    if 'PDE_MLPs_A_bis' in model.model:
                        fig = plt.figure(figsize=(22, 5))
                    else:
                        fig = plt.figure(figsize=(22, 6))
                    for k in range(model.new_features.shape[1]):
                        ax = fig.add_subplot(1, model.new_features.shape[1], k + 1)
                        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c=to_numpy(model.new_features[:, k]), s=5,
                                    cmap='viridis')
                        ax.set_title(f'new_features {k}')
                        # cbar = plt.colorbar()
                        # cbar.ax.tick_params(labelsize=6)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_recons/Features_{config_file}_{run}_{num}.tif", dpi=100)
                    plt.close()

                if 'boundary' in style:
                    fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                    t = torch.min(x[:, 7:],-1).values
                    plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=25, c=to_numpy(t),vmin=-1,vmax=1)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_recons/Boundary_{config_file}_{num}.tif", dpi=80)
                    plt.close()

                if has_ghost & ('PDE_N' not in model_config.signal_model_name):
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

    if 'inference' in test_mode:
        torch.save(x_inference_list, f"./{log_dir}/x_inference_list_{run}.pt")

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

        torch.save(neuron_gt_list, f"./{log_dir}/neuron_gt_list.pt")
        torch.save(neuron_pred_list, f"./{log_dir}/neuron_pred_list.pt")

    else:
        if False:
            # geomloss_list == []:
            geomloss_list = [0, 0]
            r = [np.mean(rmserr_list), np.std(rmserr_list), np.mean(geomloss_list), np.std(geomloss_list)]
            print('average rollout Sinkhorn div. {:.3e}+/-{:.3e}'.format(np.mean(geomloss_list), np.std(geomloss_list)))
            np.save(f"./{log_dir}/rmserr_geomloss_{config_file}.npy", r)

        if False:
            rmserr_list = np.array(rmserr_list)
            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            x_ = np.arange(len(rmserr_list))
            y_ = rmserr_list
            plt.scatter(x_, y_, c=mc)
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

