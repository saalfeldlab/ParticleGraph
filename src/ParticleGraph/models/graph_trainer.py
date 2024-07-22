import matplotlib.pyplot as plt
import torch

from GNN_particles_Ntype import *
from ParticleGraph.generators.utils import *
from ParticleGraph.models.utils import *
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Ghost_Particles import *
from geomloss import SamplesLoss
from ParticleGraph.embedding_cluster import EmbeddingCluster, sparsify_cluster

import random

def data_train(config, config_file, device):

    # matplotlib.use("Qt5Agg")

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    has_mesh = (config.graph_model.mesh_model_name != '')
    has_signal = (config.graph_model.signal_model_name != '')
    has_particle_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)
    has_cell_division = config.simulation.has_cell_division
    has_no_tracking = config.training.has_no_tracking
    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    if has_particle_field:
        data_train_particle_field(config, config_file, device)
    elif has_mesh:
        data_train_mesh(config, config_file, device)
    elif has_signal:
        data_train_signal(config, config_file, device)
    elif has_no_tracking & has_cell_division:
        data_train_cell_tracking(config, config_file, device)
    elif has_no_tracking:
        data_train_tracking(config, config_file, device)
    elif has_cell_division:
        data_train_cell(config, config_file, device)
    else:
        data_train_particles(config, config_file, device)


def data_train_particles(config, config_file, device):

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
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
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
    has_state = (config.simulation.state_type != 'discrete')

    l_dir, log_dir, logger = create_log_dir(config, config_file)
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

    x = []
    y = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # print('Loading existing model ...')
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_0.pt"
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
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')

    if has_state:
        index_l = []
        index = 0
        for k in range(n_frames):
            new_index = torch.arange(index, index + n_particles)
            index_l.append(new_index)
            x_list[1][k][:, 0] = new_index
            index += n_particles
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

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)
        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in range(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(n_runs - 1)

            dataset_batch = []
            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)

                x = x_list[run][k].clone().detach()

                if has_ghost:
                    x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                    if ghosts_particles.boids:
                        distance = torch.sum(bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                        dist_np = to_numpy(distance)
                        ind_np = torch.min(distance,axis=1)[1]
                        x_ghost[:,3:5] = x[ind_np, 3:5].clone().detach()
                    x = torch.cat((x, x_ghost), 0)

                    with torch.no_grad():
                        model.a[run,n_particles:n_particles+n_ghosts] = model.a[run,ghosts_particles.embedding_index].clone().detach()   # sample ghost embedding

                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                t = torch.Tensor([max_radius ** 2])
                edges = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)

                y = y_list[run][k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)

                y = y / ynorm

                if data_augmentation:
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
                pred = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi)

            if has_ghost:
                loss = ((pred[mask_ghost] - y_batch)).norm(2)
            else:
                if not (has_large_range):
                    loss = (pred - y_batch).norm(2)
                else:
                    loss = ((pred - y_batch) / (y_batch)).norm(2) / 1E9

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30 ) & (N%(Niter//50) == 0)) | (N==0)):
                if has_state:
                    model_a = model.a[1].clone().detach()
                    fig, ax = fig_init()
                    for k in trange(config.simulation.n_frames - 2):
                        embedding = to_numpy(model_a[k * n_particles:(k + 1) * n_particles, :].clone().detach())
                        index_particles = get_index_particles(x_list[1][k].clone().detach(), n_particle_types,
                                                              dimension)
                        for n in range(n_particle_types):
                            plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=1,
                                        color=cmap.color(n), alpha=0.025)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif", dpi=80)
                    plt.close()

                    type_list = torch.stack(x_list[1]).clone().detach()
                    t = to_numpy(type_list[:, :, 5])
                    type_list = to_numpy(type_list[:, :, 5].flatten())
                    type_list = type_list[0:len(type_list) - n_particles]

                    fig, ax = fig_init()
                    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                    for n in trange(5000):
                        sample = np.random.randint(0, len(type_list))
                        type = type_list[sample].astype(int)
                        embedding_ = model_a[sample, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                     device=device)
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_), dim=1)
                        with torch.no_grad():
                            func = model.lin_edge(in_features.float())
                        func = func[:, 0]
                        plt.plot(to_numpy(rr),
                                 to_numpy(func) * to_numpy(ynorm),
                                 color=cmap.color(type), linewidth=2, alpha=0.1)
                    plt.xlim([0, max_radius])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_{epoch}_{N}.tif", dpi=80)
                    plt.close()
                else:
                    plot_training(config=config, dataset_name=dataset_name, log_dir=log_dir,
                                  epoch=epoch, N=N, x=x, model=model, n_nodes=0, n_node_types=0, index_nodes=0, dataset_num=1,
                                  index_particles=index_particles, n_particles=n_particles,
                                  n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            loss.backward()
            optimizer.step()

            if has_ghost:
                optimizer_ghost_particles.step()
                # if (N > 0) & (N % 1000 == 0) & (train_config.ghost_method == 'MLP'):
                #     fig = plt.figure(figsize=(8, 8))
                #     plt.imshow(to_numpy(ghosts_particles.data[run, :, 120, :].squeeze()))
                #     fig.savefig(f"{log_dir}/tmp_training/embedding/ghosts_{N}.jpg", dpi=300)
                #     plt.close()

            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()}, os.path.join(log_dir, 'models', f'best_ghost_particles_with_{n_runs - 1}_graphs_{epoch}.pt'))

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
        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)

        if (simulation_config.n_interactions < 100):


            ax = fig.add_subplot(1, 5, 3)
            func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                                model_MLP=model.lin_edge, model_a=model.a,
                                                                n_nodes = 0,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                type_list=to_numpy(x[:, 1+2*dimension]),
                                                                cmap=cmap, dimension=dimension, device=device)

            labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding, train_config.cluster_distance_threshold, index_particles, n_particle_types, embedding_cluster)

            if not(has_state):
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
                plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top', transform=ax.transAxes,fontsize=10)

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

                if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq-1) & (epoch < n_epochs - sparsity_freq):
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
                                    case 'PDE_ParticleField_A'|'PDE_A':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                    case 'PDE_ParticleField_B'|'PDE_B':
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


def data_train_tracking(config, config_file, device):

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
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    has_ghost = train_config.n_ghosts > 0
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames
    sequence_length = len(config.training.sequence)
    has_sequence = (config.training.sequence != ['none'])

    l_dir, log_dir, logger = create_log_dir(config, config_file)
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

    x = []
    y = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    print('Loading existing model ...')
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_0.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start * 200
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')

    print('Update variables ...')
    x = x_list[1][n_frames - 1].clone().detach()
    n_particles = x.shape[0]
    config.simulation.n_particles = n_particles
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')

    index_l = []
    index = 0
    for k in range(n_frames):
        new_index = torch.arange(index, index + n_particles)
        index_l.append(new_index)
        x_list[1][k][:, 0] = new_index
        index += n_particles

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop} iterations per epoch')

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        Niter = n_frames * data_augmentation_loop

        for N in range(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(n_runs - 1)

            k = np.random.randint(n_frames)
            # k = int(N % n_frames)

            x = x_list[run][k].clone().detach()

            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            t = torch.Tensor([max_radius ** 2])
            edges = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x[:, :], edge_index=edges)

            # y = y_list[run][k].clone().detach()
            # if noise_level > 0:
            #     y = y * (1 + torch.randn_like(y) * noise_level)
            # y = y[:, 0:2] / ynorm

            optimizer.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()

            pred = model(dataset, training=True, vnorm=vnorm, phi=phi)
            if data_augmentation:
                new_x = cos_phi * pred[:, 0] - sin_phi * pred[:, 1]
                new_y = sin_phi * pred[:, 0] + cos_phi * pred[:, 1]
                pred[:, 0] = new_x
                pred[:, 1] = new_y

            x_next = x_list[run][k+1]
            x_pos_next = x_next[:,1:3].clone().detach()
            x_pos_pred = (x[:,1:3] + delta_t * pred)

            # y_ = bc_dpos(x_next - x[:, 1:3]) / delta_t /ynorm
            # pred_ = (x_pred - x[:, 1:3]) / delta_t
            # loss = (pred - y).norm(2)
            # loss = (pred_ - y_).norm(2)

            distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
            result = distance.min(dim=1)
            min_value = result.values
            min_index = result.indices
            pos_pre = min_value / delta_t**2

            # cell_index = x[:, 0].to(torch.int64).clone().detach()
            # cell_index_next = x_next[min_index,0].to(torch.int64)
            # embedding_loss = (model.a[cell_index] - model.a[cell_index_next]).norm(2)

            loss = torch.sum(pos_pre) # * config.training.coeff_loss1 + embedding_loss * config.training.coeff_loss2

            # loss1 = min_value.norm(2) / (vnorm**2) * config.training.coeff_loss1
            # loss1 = (x_pred[:, None, 1:3] - y[None, :, :]).norm(2)
            # loss2 = 0 * pred.norm(2) / (vnorm**2) * config.training.coeff_loss2

            visualize_embedding = True
            if visualize_embedding & (((epoch < 3 ) & (N % (Niter//100) == 0)) | (N==0)):

                fig = plt.figure(figsize=(8, 8))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=10, c='k', alpha=0.05)
                plt.scatter(to_numpy(x_pos_next[:, 0]), to_numpy(x_pos_next[:, 1]), s=10, c='r', alpha=0.1)
                plt.scatter(to_numpy(x_pos_pred[:, 0]), to_numpy(x_pos_pred[:, 1]), s=10, c='b', alpha=0.1)
                plt.xlim([0.2, 0.8])
                plt.ylim([0.2, 0.8])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/particle/{dataset_name}_{epoch}_{N}.tif")
                plt.close()

                model_a = model.a.clone().detach()
                fig, ax = fig_init()
                for k in trange(config.simulation.n_frames - 2):
                    embedding = to_numpy(model_a[k * n_particles:(k + 1) * n_particles, :].clone().detach())
                    index_particles = get_index_particles(x_list[1][k].clone().detach(), n_particle_types, dimension)
                    for n in range(n_particle_types):
                        plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=1,
                                    color=cmap.color(n), alpha=0.025)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif", dpi=80)
                plt.close()

                type_list = torch.stack(x_list[1]).clone().detach()
                t = to_numpy(type_list[:, :, 5])
                type_list = to_numpy(type_list[:, :, 5].flatten())
                type_list = type_list[0:len(type_list) - n_particles]

                fig, ax = fig_init()
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                for n in trange(5000):
                    sample = np.random.randint(0, len(type_list))
                    type = type_list[sample].astype(int)
                    embedding_ = model_a[sample, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             rr[:, None] / max_radius, embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(type), linewidth=2, alpha=0.1)
                plt.xlim([0, max_radius])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_{epoch}_{N}.tif", dpi=80)
                plt.close()

                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if epoch>18:

            lr_embedding = train_config.learning_rate_embedding_end
            lr = train_config.learning_rate_end
            logger.info(f'Learning rates: {lr}, {lr_embedding}')

        elif has_sequence:

            if epoch%sequence_length == 0:

                fig = plt.figure(figsize=(8, 8))

                print('from cell to track training')
                logger.info('from cell to track training')

                tracking_index = 0
                tracking_index_list=[]
                for k in trange(n_frames):
                    x = x_list[1][k].clone().detach()
                    distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)

                    pred = model(dataset, training=True, vnorm=vnorm, phi=torch.zeros(1, device=device))

                    x_next = x_list[run][k + 1]
                    x_next = x_next[:, 1:3].clone().detach()
                    x_pred = (x[:, 1:3] + delta_t * pred)

                    distance = torch.sum(bc_dpos(x_pred[:, None, :] - x_next[None, :, :]) ** 2, dim=2)
                    result = distance.min(dim=1)
                    min_value = result.values
                    min_index = result.indices
                    plt.scatter(np.arange(len(min_index)), to_numpy(min_index), s=10, c='k', alpha=0.05)
                    tracking_index += np.sum((to_numpy(min_index) - np.arange(len(min_index))==0)) / n_frames / n_particles *100
                    x_list[1][k+1][min_index, 0:1] = x_list[1][k][:, 0:1].clone().detach()

                    tracking_index_list.append(len(x_pred) - np.sum((to_numpy(min_index) - np.arange(len(min_index)) == 0)))

                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/proxy_tracking_{dataset_name}_{epoch}_{N}.tif", dpi=87)
                plt.close()

                print(f'tracking index: {tracking_index}')
                logger.info(f'tracking index: {tracking_index}')

                print(f'tracking errors: {np.sum(tracking_index_list)}')
                logger.info(f'tracking errors: {np.sum(tracking_index_list)}')

                fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
                plt.plot(np.arange(n_frames), tracking_index_list, color='k', linewidth=2)
                plt.ylabel(r'tracking errors', fontsize=78)
                plt.xlabel(r'frame', fontsize=78)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/tracking_error_{config_file}_{epoch}_{N}.tif", dpi=170.7)
                plt.close()

                x_ = torch.stack(x_list[1])
                x_ = torch.reshape(x_, (x_.shape[0] * x_.shape[1], x_.shape[2]))
                x_ = x_[0:(n_frames-1)*n_particles]
                x_ = to_numpy(x_[:,0])
                indexes = np.unique(x_)

                np.save(f"./{log_dir}/tmp_training/indexes_{dataset_name}_{epoch}_{N}.npy", x_)

                for k in indexes:
                    pos = np.argwhere(x_ == k)
                    if len(pos>0):
                        pos=pos[:,0]
                        model_a = torch.median(model.a[pos,:],dim=0).values
                        model_a = model_a.clone().detach()
                        model_a = model_a.repeat(len(pos),1)
                        with torch.no_grad():
                            model.a[pos,:] = model_a

                print(f'{len(np.unique(x_))} tracks,  first track index: {np.min(x_)},  last track index: {np.max(x_)}')
                logger.info(f'{len(np.unique(x_))} tracks,  first track index: {np.min(x_)},  last track index: {np.max(x_)}')

                lr_embedding = train_config.learning_rate_embedding_start
                lr = train_config.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

            elif epoch%sequence_length == 1:

                print('from track to cell training')
                logger.info('from track to cell training')

                np.save(f"./{log_dir}/tmp_training/indexes_{dataset_name}_{epoch}_{N}.npy", x_)

                for k in indexes:
                    pos = np.argwhere(x_ == k)
                    if len(pos > 0):
                        pos = pos[:, 0]
                        model_a = model.a[pos[0], :]
                        model_a = model_a.clone().detach()
                        model_a = model_a.repeat(len(pos), 1)
                        with torch.no_grad():
                            model.a[pos, :] = model_a

                for k in range(n_frames):
                    x_list[1][k][:, 0] = index_l[k].clone().detach()

            fig = plt.figure(figsize=(8, 8))
            for k in range(0,n_frames-2,n_frames//10):
                embedding = to_numpy(model.a[k*n_particles:(k+1)*n_particles,:].clone().detach())
                for n in range(n_particle_types):
                    plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=20, c=cmap.color(n))
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/all_particle_{dataset_name}_{epoch}.tif", dpi=87)
            plt.close()

        lr_embedding = train_config.learning_rate_embedding_start * 200
        lr = train_config.learning_rate_start
        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
        logger.info(f'Learning rates: {lr}, {lr_embedding}')

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_corrected.pt'))


def data_train_cell_tracking(config, config_file, device):

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
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    has_ghost = train_config.n_ghosts > 0
    n_ghosts = train_config.n_ghosts
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames

    l_dir, log_dir, logger = create_log_dir(config, config_file)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    edge_p_p_list = []
    n_particles_max = 0
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        if x[len(x)-1].shape[0] > n_particles_max:
            n_particles_max = x[len(x)-1].shape[0]
        if run>0:
            y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
            edge_p_p = np.load(f'graphs_data/graphs_{dataset_name}/edge_p_p_list_{run}.npz')
            x_list.append(x)
            y_list.append(y)
            edge_p_p_list.append(edge_p_p)
        else:
            # first dataset is not loaded to spare memory
            # first dataset is not used for training but for validation
            small_tensor = torch.zeros((1, 1), dtype=torch.float32, device=device)
            x_list.append(small_tensor)
            y_list.append(small_tensor)
            edge_p_p_list.append(to_numpy(small_tensor))
    x = x_list[1][0].clone().detach()
    y = y_list[1][0].clone().detach()
    config.simulation.n_particles_max = n_particles_max
    for run in range(1,n_runs):
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

    x = []
    y = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    print('Loading existing model ...')
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_0.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start * 200
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')

    print('Update variables ...')
    x = x_list[1][n_frames - 1].clone().detach()
    n_particles = x.shape[0]
    config.simulation.n_particles = n_particles
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')

    index_l = []
    index = 0
    for k in range(n_frames):
        pos = torch.argwhere(x_list[1][k][:, 6]==0)   # list of dead cells
        if len(pos)>0:
            x_list[1][k][pos,1] = 20
            x_list[1][k][pos,2] = 20

        new_index = torch.arange(index, index + len(x_list[1][k]))
        index_l.append(new_index)
        x_list[1][k][:, 0] = new_index
        index += len(x_list[1][k])

    if has_ghost:

        ghosts_particles = Ghost_Particles(config, n_particles, vnorm, device)
        optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop} iterations per epoch')

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        Niter = n_frames * data_augmentation_loop

        for N in range(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(n_runs - 1)
            k = np.random.randint(n_frames)
            x = x_list[run][k].clone().detach()

            if has_ghost:
                x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                if ghosts_particles.boids:
                    distance = torch.sum(bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                    ind_np = torch.min(distance,axis=1)[1]
                    x_ghost[:,3:5] = x[ind_np, 3:5].clone().detach()
                x = torch.cat((x, x_ghost), 0)

                with torch.no_grad():
                    model.a[run,n_particles:n_particles+n_ghosts] = model.a[run,ghosts_particles.embedding_index].clone().detach()   # sample ghost embedding

            edges = edge_p_p_list[run][f'arr_{k}']
            edges = torch.tensor(edges, dtype=torch.int64, device=device)
            dataset = data.Data(x=x[:, :], edge_index=edges)

            # y = y_list[run][k].clone().detach()
            # if noise_level > 0:
            #     y = y * (1 + torch.randn_like(y) * noise_level)
            # y = y[:, 0:2] / ynorm

            optimizer.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()

            pred = model(dataset, training=True, vnorm=vnorm, phi=phi, has_field=True)
            if data_augmentation:
                new_x = cos_phi * pred[:, 0] - sin_phi * pred[:, 1]
                new_y = sin_phi * pred[:, 0] + cos_phi * pred[:, 1]
                pred[:, 0] = new_x
                pred[:, 1] = new_y

            if has_ghost:
                loss = ((pred[mask_ghost] - y)).norm(2)

            x_next = x_list[run][k+1]
            x_next = x_next[:,1:3].clone().detach()
            x_pred = (x[:,1:3] + delta_t * pred)

            # y_ = bc_dpos(x_next - x[:, 1:3]) / delta_t /ynorm
            # pred_ = (x_pred - x[:, 1:3]) / delta_t
            # loss = (pred - y).norm(2)
            # loss = (pred_ - y_).norm(2)

            distance = torch.sum(bc_dpos(x_pred[:, None, :] - x_next[None, :, :]) ** 2, dim=2)
            result = distance.min(dim=1)
            min_value = result.values
            min_index = result.indices
            pos_pre = min_value / delta_t**2

            loss = torch.sum(pos_pre)

            # loss1 = min_value.norm(2) / (vnorm**2) * config.training.coeff_loss1
            # loss1 = (x_pred[:, None, 1:3] - y[None, :, :]).norm(2)
            # loss2 = 0 * pred.norm(2) / (vnorm**2) * config.training.coeff_loss2

            visualize_embedding = True
            if visualize_embedding & (((epoch < 3 ) & (N % (Niter//100) == 0)) | (N==0)):
                print(N)
                fig = plt.figure(figsize=(8, 8))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=10, c='k', alpha=0.05)
                plt.scatter(to_numpy(x_next[:, 0]), to_numpy(x_next[:, 1]), s=10, c='r', alpha=0.1)
                plt.scatter(to_numpy(x_pred[:, 0]), to_numpy(x_pred[:, 1]), s=10, c='b', alpha=0.1)
                plt.xlim([0.2, 0.8])
                plt.ylim([0.2, 0.8])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/particle/{dataset_name}_{epoch}_{N}.tif")
                plt.close()

                x_ = x_list[1][0].clone().detach()
                index_particles = get_index_particles(x_, n_particle_types, dimension)
                plot_training_cell(config=config, dataset_name=dataset_name, log_dir=log_dir,
                              epoch=epoch, N=N, model=model, index_particles=index_particles, n_particle_types=n_particle_types, type_list=type_list, ynorm=ynorm, cmap=cmap, device=device)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            loss.backward()
            optimizer.step()

            if has_ghost:
                optimizer_ghost_particles.step()
                # if (N > 0) & (N % 1000 == 0) & (train_config.ghost_method == 'MLP'):
                #     fig = plt.figure(figsize=(8, 8))
                #     plt.imshow(to_numpy(ghosts_particles.data[run, :, 120, :].squeeze()))
                #     fig.savefig(f"{log_dir}/tmp_training/embedding/ghosts_{N}.jpg", dpi=300)
                #     plt.close()

            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        t, r, a = get_gpu_memory_map(device)
        logger.info(f"GPU memory: total {t} reserved {r} allocated {a}")

        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()}, os.path.join(log_dir, 'models', f'best_ghost_particles_with_{n_runs - 1}_graphs_{epoch}.pt'))

        if epoch>18:

            lr_embedding = train_config.learning_rate_embedding_end
            lr = train_config.learning_rate_end
            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
            logger.info(f'Learning rates: {lr}, {lr_embedding}')

        else:

            if epoch%2 == 0:

                fig = plt.figure(figsize=(8, 8))

                print('from cell to track training')
                logger.info('from cell to track training')

                tracking_index = 0
                tracking_index_list = []
                for k in trange(n_frames):
                    x = x_list[1][k].clone().detach()
                    distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)

                    pred = model(dataset, training=True, vnorm=vnorm, phi=torch.zeros(1, device=device))

                    x_next = x_list[run][k + 1]
                    x_next = x_next[:, 1:3].clone().detach()
                    x_pred = (x[:, 1:3] + delta_t * pred)

                    distance = torch.sum(bc_dpos(x_pred[:, None, :] - x_next[None, :, :]) ** 2, dim=2)
                    result = distance.min(dim=1)
                    min_value = result.values
                    min_index = result.indices
                    plt.scatter(np.arange(len(min_index)), to_numpy(min_index), s=10, c='k', alpha=0.05)
                    tracking_index += np.sum((to_numpy(min_index) - np.arange(len(min_index))==0)) / n_frames / n_particles *100
                    x_list[1][k+1][min_index, 0:1] = x_list[1][k][:, 0:1].clone().detach()

                    tracking_index_list.append(len(x_pred) - np.sum((to_numpy(min_index) - np.arange(len(min_index)) == 0)))

                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/proxy_tracking_{dataset_name}_{epoch}_{N}.tif", dpi=87)
                plt.close()

                fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
                plt.plot(np.arange(n_frames), tracking_index_list, color='k', linewidth=2)
                plt.ylabel(r'tracking errors', fontsize=78)
                plt.xlabel(r'frame', fontsize=78)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/tracking_error_{config_file}_{epoch}_{N}.tif", dpi=170.7)
                plt.close()

                print(f'tracking errors: {np.sum(tracking_index_list)}')
                logger.info(f'tracking errors: {np.sum(tracking_index_list)}')

                print(f'tracking index: {tracking_index}')
                logger.info(f'tracking index: {tracking_index}')

                for k in range(n_frames-1):
                    if k==0:
                        x_ = x_list[1][k].clone().detach()
                    else:
                        x_ = torch.cat((x_,x_list[1][k]),0)
                x_ = to_numpy(x_[:,0])
                indexes = np.unique(x_)

                np.save(f"./{log_dir}/tmp_training/indexes_{dataset_name}_{epoch}_{N}.npy", x_)

                for k in indexes:
                    pos = np.argwhere(x_ == k)
                    if len(pos>0):
                        pos=pos[:,0]
                        model_a = torch.median(model.a[pos,:],dim=0).values
                        model_a = model_a.clone().detach()
                        model_a = model_a.repeat(len(pos),1)
                        with torch.no_grad():
                            model.a[pos,:] = model_a

                print(f'{len(np.unique(x_))} tracks,  first track index: {np.min(x_)},  last track index: {np.max(x_)}')
                logger.info(f'{len(np.unique(x_))} tracks,  first track index: {np.min(x_)},  last track index: {np.max(x_)}')

                lr_embedding = train_config.learning_rate_embedding_start
                lr = train_config.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

            else:

                print('from track to cell training')
                logger.info('from track to cell training')

                np.save(f"./{log_dir}/tmp_training/indexes_{dataset_name}_{epoch}_{N}.npy", x_)

                for k in indexes:
                    pos = np.argwhere(x_ == k)
                    if len(pos > 0):
                        pos = pos[:, 0]
                        model_a = model.a[pos[0], :]
                        model_a = model_a.clone().detach()
                        model_a = model_a.repeat(len(pos), 1)
                        with torch.no_grad():
                            model.a[pos, :] = model_a

                for k in range(n_frames):
                    x_list[1][k][:, 0] = index_l[k].clone().detach()

        # fig = plt.figure(figsize=(8, 8))
        # for k in range(0,n_frames-2,n_frames//10):
        #     embedding = to_numpy(model.a[k*n_particles:(k+1)*n_particles,:].clone().detach())
        #     for n in range(n_particle_types):
        #         plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=20, c=cmap.color(n))
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/all_particle_{dataset_name}_{epoch}_{N}.tif", dpi=87)
        # plt.close()

        lr_embedding = train_config.learning_rate_embedding_start * 200
        lr = train_config.learning_rate_start
        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
        logger.info(f'Learning rates: {lr}, {lr_embedding}')

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_corrected.pt'))


def data_train_cell(config, config_file, device):

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
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
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

    l_dir, log_dir, logger = create_log_dir(config, config_file)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    T1_list = []
    edge_p_p_list = []
    n_particles_max = 0
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        if x[-1][-1,0] > n_particles_max:
            n_particles_max = x[-1][-1,0]+1
        if run>0:
            y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
            edge_p_p = np.load(f'graphs_data/graphs_{dataset_name}/edge_p_p_list_{run}.npz')
            T1 = torch.load(f'graphs_data/graphs_{dataset_name}/T1_list_{run}.pt', map_location=device)
            x_list.append(x)
            y_list.append(y)
            edge_p_p_list.append(edge_p_p)
            T1_list.append(T1)
        else:
            # first dataset is not loaded to spare memory
            # first dataset is not used for training but for validation
            small_tensor = torch.zeros((1, 1), dtype=torch.float32, device=device)
            x_list.append(small_tensor)
            y_list.append(small_tensor)
            T1_list.append(small_tensor)
            edge_p_p_list.append(to_numpy(small_tensor))
    n_particles_max= int(to_numpy(n_particles_max))
    x = x_list[1][0].clone().detach()
    y = y_list[1][0].clone().detach()
    config.simulation.n_particles_max = n_particles_max
    for run in range(1,n_runs):
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

    x = []
    y = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_20.pt"
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
    x = x_list[1][n_frames - 1].clone().detach()
    n_particles = len(T1_list[1])
    config.simulation.n_particles = n_particles
    print(f'N particles: {config.simulation.n_particles} to {len(T1_list[1])} ')
    logger.info(f'N particles: {config.simulation.n_particles} to {len(T1_list[1])} ')

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

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in range(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(n_runs - 1)

            dataset_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 2)

                x = x_list[run][k].clone().detach()

                if has_ghost:
                    x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                    if ghosts_particles.boids:
                        distance = torch.sum(bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                        dist_np = to_numpy(distance)
                        ind_np = torch.min(distance,axis=1)[1]
                        x_ghost[:,3:5] = x[ind_np, 3:5].clone().detach()
                    x = torch.cat((x, x_ghost), 0)
                    with torch.no_grad():
                        model.a[run,n_particles:n_particles+n_ghosts] = model.a[run,ghosts_particles.embedding_index].clone().detach()   # sample ghost embedding

                edges = edge_p_p_list[run][f'arr_{k}']
                edges = torch.tensor(edges, dtype=torch.int64, device=device)
                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)

                y = y_list[run][k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)

                y = y / ynorm

                if data_augmentation:
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
                pred = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi, has_field=True)

            if has_ghost:
                loss = ((pred[mask_ghost] - y_batch)).norm(2)
            else:
                # mask_cell_alive = np.argwhere(to_numpy(x[:,6:7]) == 1)
                # mask_cell_alive = mask_cell_alive[:, 0].astype(int)
                # loss = ((pred[mask_cell_alive] - y_batch[mask_cell_alive])).norm(2)
                loss = ((pred - y_batch)).norm(2)

            visualize_embedding = True
            if visualize_embedding & (((epoch < 3 ) & (N%(Niter//100) == 0)) | (N==0)):
                plot_training_cell(config=config, dataset_name=dataset_name, log_dir=log_dir,
                              epoch=epoch, N=N, model=model, n_particle_types=n_particle_types, type_list=T1_list[1], ynorm=ynorm, cmap=cmap, device=device)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                t, r, a = get_gpu_memory_map(device)
                logger.info(f"GPU memory: total {t} reserved {r} allocated {a}")

            loss.backward()
            optimizer.step()

            if has_ghost:
                optimizer_ghost_particles.step()
                # if (N > 0) & (N % 1000 == 0) & (train_config.ghost_method == 'MLP'):
                #     fig = plt.figure(figsize=(8, 8))
                #     plt.imshow(to_numpy(ghosts_particles.data[run, :, 120, :].squeeze()))
                #     fig.savefig(f"{log_dir}/tmp_training/embedding/ghosts_{N}.jpg", dpi=300)
                #     plt.close()

            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()}, os.path.join(log_dir, 'models', f'best_ghost_particles_with_{n_runs - 1}_graphs_{epoch}.pt'))


def data_train_mesh(config, config_file, device):

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

    l_dir, log_dir, logger = create_log_dir(config, config_file)
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
                pred = model(batch, data_id=run)

            loss = ((pred - y_batch) * mask_mesh).norm(2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = False
            if visualize_embedding & (((epoch < 30 ) & (N%(Niter//50) == 0)) | (N==0)):
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

                plot_training(config=config, dataset_name=dataset_name,
                              log_dir=log_dir,
                              epoch=epoch, N=N, x=x_mesh, model=model, n_nodes=n_nodes, n_node_types=n_node_types, index_nodes=index_nodes, dataset_num=1,
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

            labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding, train_config.cluster_distance_threshold, index_nodes, n_node_types, embedding_cluster)

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
            plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top', transform=ax.transAxes,fontsize=10)

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

            if (replace_with_cluster) & ((epoch+1) % sparsity_freq == 0):
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


def data_train_particle_field(config, config_file, device):

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
    data_augmentation = train_config.data_augmentation
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

    l_dir, log_dir, logger = create_log_dir(config, config_file)
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
    # print('Loading existing model ...')
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_5.pt"
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
            model_f = Siren_Network(image_width=image_width, in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                        hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device, first_omega_0=80, hidden_omega_0=80.)
        else:
            model_f = Siren_Network(image_width=image_width, in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                        hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80, hidden_omega_0=80.)
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

        f_p_mask=[]
        for k in range(batch_size):
            if k==0:
                f_p_mask=np.zeros((n_nodes,1))
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

        for N in range(Niter):

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
                        x_mesh [:,6:7] = model.field[run]
                    case 'siren':
                        x_mesh[:, 6:7] = model_f()**2
                    case 'siren_with_time':
                        x_mesh[:, 6:7] = model_f(time=k/n_frames)**2
                x_particle_field = torch.concatenate((x_mesh, x), dim=0)

                if has_ghost:
                    x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                    if ghosts_particles.boids:
                        distance = torch.sum(bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                        dist_np = to_numpy(distance)
                        ind_np = torch.min(distance,axis=1)[1]
                        x_ghost[:,3:5] = x[ind_np, 3:5].clone().detach()
                    x = torch.cat((x, x_ghost), 0)

                    with torch.no_grad():
                        model.a[run,n_particles:n_particles+n_ghosts] = model.a[run,ghosts_particles.embedding_index].clone().detach()   # sample ghost embedding

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

                if data_augmentation:
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
                        var_batch = torch.mean(ghosts_particles.var[run,k],dim=0)
                        var_batch = var_batch[:,None]
                    else:
                        var = torch.mean(ghosts_particles.var[run,k],dim=0)
                        var_batch = torch.cat((var_batch, var[:, None]), dim=0)

            batch_loader_p_p = DataLoader(dataset_batch_p_p, batch_size=batch_size, shuffle=False)
            batch_loader_f_p = DataLoader(dataset_batch_f_p, batch_size=batch_size, shuffle=False)

            optimizer.zero_grad()

            if has_siren:
                optimizer_f.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()

            for batch in batch_loader_f_p:
                pred_f_p = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi, has_field=True)
            for batch in batch_loader_p_p:
                pred_p_p = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi, has_field=False)

            pred_f_p = pred_f_p[f_p_mask]

            if has_ghost:
                loss = ((pred_p_p[mask_ghost] + 0 * pred_f_p - y_batch)).norm(2) + var_batch.mean() + model.field.norm(2)
            else:
                loss = (pred_p_p + pred_f_p - y_batch).norm(2) # + model.field.norm(2)

            loss.backward()
            optimizer.step()
            if has_siren:
                optimizer_f.step()
            if has_ghost:
                optimizer_ghost_particles.step()

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30 ) & (N%(Niter//50) == 0)) | (N==0)):
                plot_training_particle_field(config=config, has_siren=has_siren, has_siren_time=has_siren_time, model_f=model_f, dataset_name=dataset_name, n_frames=n_frames, model_name=model_config.particle_model_name, log_dir=log_dir,
                              epoch=epoch, N=N, x=x, x_mesh=x_mesh, model_field=model.field, model=model, n_nodes=0, n_node_types=0, index_nodes=0, dataset_num=1,
                              index_particles=index_particles, n_particles=n_particles,
                              n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
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
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()}, os.path.join(log_dir, 'models', f'best_ghost_particles_with_{n_runs - 1}_graphs_{epoch}.pt'))

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
                                                          train_config.cluster_distance_threshold, index_particles,
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
            if train_config.sparsity=='replace_embedding':

                logger.info(f'replace_embedding_function')
                y_func_list = func_list * 0

                ax, fig = fig_init()
                for n in np.unique(new_labels):
                    pos = np.argwhere(new_labels == n)
                    pos = pos.squeeze()
                    if pos.size > 0:
                        target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                        y_func_list[pos] = target_func
                    plt.plot (to_numpy(target_func) * to_numpy(ynorm), linewidth=2 , alpha=1)
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


def data_train_signal(config, config_file, device):

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
    has_mesh = (config.graph_model.mesh_model_name != '')
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    sparsity_freq = train_config.sparsity_freq
    delta_t = simulation_config.delta_t
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs

    l_dir, log_dir, logger = create_log_dir(config, config_file)
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
    # net = f"./log/try_{config_file}/models/best_model_with_99_graphs_9.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

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

    if 'mat' in simulation_config.connectivity_file:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
    else:
        adjacency = torch.load(simulation_config.connectivity_file, map_location=device)
    adj_t = adjacency > 0
    edge_index = adj_t.nonzero().t().contiguous()

    if config_file == 'signal_N_100_2_b':
        for n in trange(20000):
            i = np.random.randint(n_particles)
            j = np.random.randint(n_particles)
            if adjacency[i,j]==0:
                edge_index = torch.cat((edge_index, torch.tensor([[i], [j]], device=device)), 1)
                edge_index = torch.cat((edge_index, torch.tensor([[j], [i]], device=device)), 1)
    if config_file == 'signal_N_100_2_c':
        for n in trange(40000):
            i = np.random.randint(n_particles)
            j = np.random.randint(n_particles)
            if adjacency[i,j]==0:
                edge_index = torch.cat((edge_index, torch.tensor([[i], [j]], device=device)), 1)
                edge_index = torch.cat((edge_index, torch.tensor([[j], [i]], device=device)), 1)
    if (config_file == 'signal_N_100_2_d') | ('signal_N_100_2_asym' in config_file):
        print('Fully connected...')
        for i in trange(n_particles):
                i_s = torch.ones(n_particles, device=device) * i
                j_s = torch.arange(n_particles, device=device)
                ij_s = torch.cat((i_s[:,None], j_s[:,None]), dim=1).t()
                if i==0:
                    edge_index = ij_s
                else:
                    edge_index = torch.cat((edge_index, ij_s), dim=1)
                edge_index = edge_index.to(dtype=torch.int64)

    model.edges = edge_index
    logger.info(f'edge_index.shape {edge_index.shape} ')

    print("Start training ...")
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
            if has_mesh:
                mask_mesh = mask_mesh.repeat(repeat_factor, 1)
            if has_ghost:
                mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
                mask_ghost = np.tile(mask_ghost, batch_size)
                mask_ghost = np.argwhere(mask_ghost == 1)
                mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0

        Niter = n_frames * data_augmentation_loop // batch_size
        if (has_mesh) & (batch_size == 1):
            Niter = Niter // 4
        print(f'Niter = {Niter}')
        logger.info(f'Niter = {Niter}')

        for N in trange(Niter):

            run = 1 + np.random.randint(n_runs - 1)
            k = np.random.randint(n_frames - 6)

            optimizer.zero_grad()

            match recursive_loop:

                case 1:

                    x = x_list[run][k].clone().detach()
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred = model(dataset, data_id=run)
                    y = y_list[run][k].clone().detach()
                    y = y / ynorm
                    y = y[:, 0:2]
                    loss = (pred - y).norm(2)

                case 2:

                    x = x_list[run][k].clone().detach()
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred1 = model(dataset, data_id=run)
                    x[:, 6:7] += pred1 * delta_t
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred2 = model(dataset, data_id=run)

                    y = (y_list[run][k].clone().detach() + y_list[run][k + 1].clone().detach())
                    y = y / ynorm
                    y = y[:, 0:2]


                    func_f = model.lin_edge(torch.zeros(1,device=device))
                    in_features = torch.cat((torch.zeros((n_particles,1),device=device),  model.a[1, :]), dim=1)
                    func_phi = model.lin_phi(in_features.float())


                    loss = (pred1 + pred2 - y).norm(2) / 2 + model.vals.norm(1) * config.training.coeff_L1 + func_f.norm(2) + func_phi.norm(2)

                        
                case 3:

                    x = x_list[run][k].clone().detach()
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred1 = model(dataset, data_id=run)
                    x[:, 6:7] += pred1 * delta_t
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred2 = model(dataset, data_id=run)
                    x[:, 6:7] += pred2 * delta_t
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred3 = model(dataset, data_id=run)

                    y1 = y_list[run][k].clone().detach()/ ynorm
                    y2 = y_list[run][k+1].clone().detach()/ ynorm
                    y3 = y_list[run][k+2].clone().detach()/ ynorm
                    y1 = y1[:, 0:2]
                    y2 = y2[:, 0:2]
                    y3 = y3[:, 0:2]

                    loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + (pred3 - y3).norm(2)

                case 5:

                    x = x_list[run][k].clone().detach()
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred1 = model(dataset, data_id=run)
                    x[:, 6:7] += pred1 * delta_t
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred2 = model(dataset, data_id=run)
                    x[:, 6:7] += pred2 * delta_t
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred3 = model(dataset, data_id=run)
                    x[:, 6:7] += pred3 * delta_t
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred4 = model(dataset, data_id=run)
                    x[:, 6:7] += pred4 * delta_t
                    dataset = data.Data(x=x[:, :], edge_index=model.edges)
                    pred5 = model(dataset, data_id=run)

                    y1 = y_list[run][k].clone().detach()/ ynorm
                    y2 = y_list[run][k+1].clone().detach()/ ynorm
                    y3 = y_list[run][k+2].clone().detach()/ ynorm
                    y4 = y_list[run][k+3].clone().detach()/ ynorm
                    y5 = y_list[run][k+4].clone().detach()/ ynorm
                    y1 = y1[:, 0:2]
                    y2 = y2[:, 0:2]
                    y3 = y3[:, 0:2]
                    y4 = y4[:, 0:2]
                    y5 = y5[:, 0:2]

                    loss = (pred1 - y1).norm(2) + (pred2 - y2).norm(2) + (pred3 - y3).norm(2)+ (pred4 - y4).norm(2) + (pred5 - y5).norm(2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = False
            if visualize_embedding & (((epoch == 0) & (N < 10000) & (N % 1000 == 0)) | (N==0)):
                plot_training_signal(config, dataset, model, adjacency, log_dir, epoch, N, index_particles, n_particles, n_particle_types, device)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if train_config.regul_matrix:
            logger.info('regul_matrix')
            percentile = 0.95
            A = model.A.clone().detach()
            A = torch.reshape(A, (n_particles*n_particles,1))
            A = torch.abs(A)
            A = A * (A > torch.quantile(A, percentile))
            A = torch.reshape(A, (n_particles,n_particles))
            i, j = torch.triu_indices(n_particles,n_particles, requires_grad=False, device=device)
            Aij = to_numpy(A[i,j].clone().detach())
            with torch.no_grad():
                model.vals = nn.Parameter(torch.tensor(Aij, requires_grad=True, dtype=torch.float32, device=device))
            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)


        fig = plt.figure(figsize=(22, 4))

        ax = fig.add_subplot(1, 6, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        A = torch.zeros(n_particles, n_particles, device=device, requires_grad=False, dtype=torch.float32)
        if 'asymmetric' in config.simulation.adjacency_matrix:
            A = model.vals
        else:
            i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
            A[i,j] = model.vals
            A.T[i,j] = model.vals

        ax = fig.add_subplot(1, 6, 3)
        gt_weight = to_numpy(adjacency[adj_t])
        pred_weight = to_numpy(A[adj_t])
        plt.scatter(gt_weight, pred_weight, s=0.1,c='k')
        plt.xlabel('gt weight', fontsize=12)
        plt.ylabel('predicted weight', fontsize=12)
        ax = fig.add_subplot(1, 6, 4)
        gt_weight = to_numpy(adjacency)
        pred_weight = to_numpy(A)
        plt.scatter(gt_weight, pred_weight, s=0.1,c='k')
        plt.xlabel('all gt weight', fontsize=12)
        plt.ylabel('all predicted weight', fontsize=12)
        ax = fig.add_subplot(1, 6, 5)
        plt.imshow(to_numpy(adjacency), cmap='viridis', vmin=0, vmax=0.01)
        ax = fig.add_subplot(1, 6, 6)
        plt.imshow(np.abs(to_numpy(A)), cmap='viridis', vmin=0, vmax=0.001)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, ratio=1, run=1, test_simulation=False, sample_embedding = False, device=[]):
    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_ghost = config.training.n_ghosts > 0
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    dimension = simulation_config.dimension
    has_siren_time = 'siren_with_time' in model_config.field_type
    has_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)
    if best_model == -1:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs-1}_graphs.pt"
    else:
        net = f"./log/try_{config_file}/models/best_model_with_{n_runs-1}_graphs_{best_model}.pt"

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    if verbose:
        print(f'Test data ... {model_config.particle_model_name} {model_config.mesh_model_name}')
        print('log_dir: {}'.format(log_dir))
        print(f'network: {net}')
        print(table)
        print(f"Total Trainable Params: {total_params}")

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
                t = t.reshape(image_width * image_width,1)
                with torch.no_grad():
                    model.a = a_.clone().detach()
                    model.field[run] = t.clone().detach()

    first_embedding = model.a[1].data.clone().detach()

    n_sub_population = n_particles // n_particle_types

    first_index_particles = []
    for n in range(n_particle_types):
        index = np.arange(n_particles * n // n_particle_types, n_particles * (n + 1) // n_particle_types)
        first_index_particles.append(index)

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
        index_particles = get_index_particles(x, n_particle_types,dimension)
        x_mesh = x_mesh_list[0][0].clone().detach()
    else:
        x_list = []
        y_list = []
        x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device))
        y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device))
        ynorm = torch.load(f'./log/try_{config_file}/ynorm.pt', map_location=device).to(device)
        vnorm = torch.load(f'./log/try_{config_file}/vnorm.pt', map_location=device).to(device)
        x = x_list[0][0].clone().detach()
        n_particles = int(x.shape[0] / ratio)
        config.simulation.n_particles = n_particles
        index_particles = get_index_particles(x, n_particle_types, dimension)
        if n_particle_types>1000:
            index_particles = []
            for n in range(3):
                index = np.arange(n_particles * n // 3, n_particles * (n + 1) // 3)
                index_particles.append(index)
                n_particle_types = 3

    if ratio > 1:
        new_nparticles = int(n_particles * ratio)
        model.a = nn.Parameter(
            torch.tensor(np.ones((n_runs, int(new_nparticles), 2)), device=device, dtype=torch.float32, requires_grad=False))
        n_particles = new_nparticles
        index_particles = get_index_particles(x, n_particle_types, dimension)
    if sample_embedding:
        model_a_ = nn.Parameter(
            torch.tensor(np.ones((int(n_particles), model.embedding_dim)),device=device,requires_grad=False, dtype=torch.float32))
        for n in range(n_particles):
            t = to_numpy(x[n,5]).astype(int)
            index = first_index_particles[t][np.random.randint(n_sub_population)]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((model.n_dataset,int(n_particles), model.embedding_dim)),device=device,requires_grad=False, dtype=torch.float32))
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
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    rmserr_list= []
    gloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    geomloss_list=[]
    time.sleep(1)
    for it in trange(n_frames+1):

        x0 = x_list[0][it].clone().detach()
        y0 = y_list[0][it].clone().detach()

        if model_config.signal_model_name == 'PDE_N':
            rmserr = torch.sqrt(torch.mean(torch.sum(bc_dpos(x[:, 6:7] - x0[:, 6:7]) ** 2, axis=1)))
        elif model_config.mesh_model_name == 'WaveMesh':
            rmserr = torch.sqrt(torch.mean((x[mask_mesh.squeeze(), 6:7] - x0[mask_mesh.squeeze(), 6:7]) ** 2))
            # errors = (x[mask_mesh.squeeze(), 6:7] - x0[mask_mesh.squeeze(), 6:7]) ** 2
            # percentile_90th = torch.quantile(errors, 0.9)
            # errors = errors[errors < percentile_95th]
            # rmserr_95 = torch.sqrt(torch.mean(errors))
            #
            fig, ax = plt.subplots()
            plt.scatter(to_numpy(x0[mask_mesh.squeeze(), 6:7]), to_numpy(x[mask_mesh.squeeze(), 6:7]), s=2, c='r')
            # fig, ax = plt.subplots()
            # errors = (x[mask_mesh.squeeze(), 6:7] - x0[mask_mesh.squeeze(), 6:7]) ** 2
            # plt.plot(to_numpy(torch.sqrt(errors)), c='k')

        elif model_config.mesh_model_name == 'RD_RPS_Mesh':
            rmserr = torch.sqrt(torch.mean(torch.sum((x[mask_mesh.squeeze(), 6:9] - x0[mask_mesh.squeeze(), 6:9]) ** 2, axis=1)))
        else:
            rmserr = torch.sqrt(torch.mean(torch.sum(bc_dpos(x[:, 1:3] - x0[:, 1:3]) ** 2, axis=1)))
            if x.shape[0]>5000:
                geomloss = gloss(x[0:5000, 1:3], x0[0:5000, 1:3])
            else:
                geomloss = gloss(x[:, 1:3], x0[:, 1:3])
            geomloss_list.append(geomloss.item())

        rmserr_list.append(rmserr.item())

        if has_mesh:
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset_mesh = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)

        if model_config.mesh_model_name == 'DiffMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0, )
            x[:, 6:7] += pred * hnorm * delta_t
        elif model_config.mesh_model_name == 'WaveMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=1)
            x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
            x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif (model_config.mesh_model_name == 'RD_RPS_Mesh') | (model_config.mesh_model_name=='RD_RPS_Mesh_bis'):
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

            distance = torch.sum(bc_dpos(x[:, None, 1:dimension+1] - x[None, :, 1:dimension+1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset_p_p = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

            distance = torch.sum(bc_dpos(x_particle_field[:, None, 1:dimension+1] - x_particle_field[None, :, 1:dimension+1]) ** 2, dim=2)
            adj_t = ((distance < (max_radius/2) ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            pos = torch.argwhere((edge_index[1,:]>=n_nodes) & (edge_index[0,:]<n_nodes))
            pos = to_numpy(pos[:,0])
            edge_index = edge_index[:,pos]
            dataset_f_p = data.Data(x=x_particle_field, pos=x_particle_field[:, 1:3], edge_index=edge_index, field=x_particle_field[:,6:7])

            with torch.no_grad():
                y0 = model(dataset_p_p,data_id=1, training=False, vnorm=vnorm, phi=torch.zeros(1, device=device),has_field=False)
                y1 = model_f_p(dataset_f_p,data_id=1, training=False, vnorm=vnorm,phi=torch.zeros(1, device=device),has_field=True)[n_nodes:]
                y = y0 + y1

            if model_config.prediction == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm
                x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t)
        else:

            x_ = x
            if has_ghost:
                x_ghost = model_ghost.get_pos(dataset_id=run, frame=it, bc_pos=bc_pos)
                x_ = torch.cat((x_, x_ghost), 0)

            if has_adjacency_matrix:
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                if test_simulation:
                    y = y0 / ynorm
                else:
                    with torch.no_grad():
                        y = model(dataset, data_id=1)
            else:
                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                if test_simulation:
                    # y = y0 / ynorm
                    dataset.x[:,5] = x0[:,5]
                    index_particles = get_index_particles(dataset.x, n_particle_types, dimension)
                    y = model(dataset) / ynorm
                else:
                    with torch.no_grad():
                        y = model(dataset, data_id=1, training=False, vnorm=vnorm,
                                  phi=torch.zeros(1, device=device))  # acceleration estimation
                if has_ghost:
                    y = y[mask_ghost]

            if model_config.prediction == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm
                if model_config.signal_model_name == 'PDE_N':
                    x[:, 6:7] += y * delta_t    # signal update
                else:
                    x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t)  # position update

        if (it % step == 0) & (it >= 0) & visualize:

            if 'latex' in style:
                plt.rcParams['text.usetex'] = True
                rc('font', **{'family': 'serif', 'serif': ['Palatino']})

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
                if (model_config.mesh_model_name == 'RD_RPS_Mesh') | (model_config.mesh_model_name=='RD_RPS_Mesh_bis'):
                    H1_IM = torch.reshape(x[:, 6:9], (100, 100, 3))
                    plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.axis('off')
            elif model_config.signal_model_name == 'PDE_N':

                matplotlib.rcParams['savefig.pad_inches'] = 0
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=200, c=to_numpy(x[:, 6]), cmap='viridis', vmin=0,
                            vmax=3)
                plt.xlim([-1.2, 1.2])
                plt.ylim([-1.2, 1.2])
                # plt.xlabel('x', fontsize=48)
                # plt.ylabel('y', fontsize=48)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
                ax.tick_params(axis='both', which='major', pad=15)
                plt.text(0, 1.1, f'   ', ha='left', va='top', transform=ax.transAxes, fontsize=48)
                plt.tight_layout()
            else:
                s_p = 100
                if simulation_config.has_cell_division:
                    s_p = 25
                for n in range(n_particle_types):
                    plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                                x[index_particles[n], 1].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
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
            if 'no_ticks' in style:
                plt.xticks([])
                plt.yticks([])
            if not(('RD_RPS_Mesh' in model_config.mesh_model_name)|(model_config.signal_model_name == 'PDE_N')):
                plt.xlim([0, 1])
                plt.ylim([0, 1])
            if 'PDE_G' in model_config.particle_model_name:
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{it}.tif", dpi=170.7)
            # plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{10000+it}.tif", dpi=42.675)
            plt.close()

            if has_ghost:

                x0 = x_list[0][it+1].clone().detach()
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

    print('average rollout RMS {:.3e}+/-{:.3e}'.format(np.mean(rmserr_list), np.std(rmserr_list)))
    if has_mesh:
        h = x_mesh_list[0][0][:,6:7]
        for k in range(n_frames):
            h = torch.cat((h, x_mesh_list[0][k][:,6:7]), 0)
        h = to_numpy(h)
        print(h.shape)
        print('average u {:.3e}+/-{:.3e}'.format(np.mean(h), np.std(h)))

    elif model_config.signal_model_name != 'PDE_N':

        r = [np.mean(rmserr_list), np.std(rmserr_list), np.mean(geomloss_list), np.std(geomloss_list)]
        print('average rollout Sinkhorn div. {:.3e}+/-{:.3e}'.format(np.mean(geomloss_list), np.std(geomloss_list)))
        np.save(f"./{log_dir}/rmserr_geomloss_{config_file}.npy", r)

        if True:
            rmserr_list = np.array(rmserr_list)
            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            x_ = np.arange(len(rmserr_list))
            y_ = rmserr_list
            plt.scatter(x_,y_,c='k')
            plt.xticks(fontsize=48)
            plt.yticks(fontsize=48)
            plt.xlabel(r'$Epochs$', fontsize=78)
            plt.ylabel(r'$RMSE$', fontsize=78)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/rmserr_{config_file}_plot.tif", dpi=170.7)

        if True:

            x0_next = x_list[0][it].clone().detach()

            fig = plt.figure(figsize=(12, 12))
            ax=fig.add_subplot(1,1,1)
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
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False,ax=ax,edge_color='r', width=4)
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