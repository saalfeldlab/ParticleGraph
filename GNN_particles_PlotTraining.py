import time

from matplotlib.ticker import FormatStrFormatter
# import networkx as nx
from sklearn import metrics

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.data_loaders import *
from ParticleGraph.embedding_cluster import *
from ParticleGraph.models import Division_Predictor
from ParticleGraph.models.Ghost_Particles import Ghost_Particles
from ParticleGraph.models.utils import *
from ParticleGraph.utils import *
from matplotlib import rc


# matplotlib.use("Qt5Agg")
# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

def data_plot_training(config, mode, device):
    print('')

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    radius = simulation_config.max_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    has_ghost = train_config.n_ghosts > 0
    has_particle_dropout = train_config.particle_dropout > 0
    target_batch_size = train_config.batch_size
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    replace_with_cluster = 'replace' in train_config.sparsity
    visualize_embedding = False
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    aggr_type = config.graph_model.aggr_type

    embedding_cluster = EmbeddingCluster(config)

    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)

    x_list = []
    y_list = []
    print('Load data ...')
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)

    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in trange(n_frames):
            if (k%10 == 0) | (n_frames<1000):
                x = torch.cat((x,x_list[run][k].clone().detach()),0)
                y = torch.cat((y,y_list[run][k].clone().detach()),0)
        print(x_list[run][k].shape)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    vnorm = vnorm[4]
    ynorm = ynorm[4]
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    if has_mesh:
        x_mesh_list = []
        y_mesh_list = []
        for run in trange(NGraphs):
            x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
            x_mesh_list.append(x_mesh)
            h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
            y_mesh_list.append(h)
        h = y_mesh_list[0][0].clone().detach()
        for run in range(NGraphs):
            for k in range(n_frames):
                h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(f'hnorm: {to_numpy(hnorm)}')
        time.sleep(0.5)

        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)

        mask_mesh = mesh_data['mask']
        # mesh_pos = mesh_data['mesh_pos']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']
        # face = mesh_data['face']

        mask_mesh = mask_mesh.repeat(batch_size, 1)

    h=[]
    x=[]
    y=[]

    print('done ...')

    model, bc_pos, bc_dpos = choose_training_model(config, device)


    if  has_cell_division:
        model_division = Division_Predictor(config, device)
        optimizer_division, n_total_params_division = set_trainable_division_parameters(model_division, lr=1E-3)
        logger.info(f"Total Trainable Divsion Params: {n_total_params_division}")
        logger.info(f'Learning rates: 1E-3')

    x = x_list[1][0].clone().detach()
    type_list = x[:, 5:6].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    time.sleep(0.5)


    # matplotlib.use("Qt5Agg")
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0
    # style = {
    #     "pgf.rcfonts": False,
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,
    #     "font.family": "sans-serif"
    # }
    # matplotlib.rcParams.update(style)
    # plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "HelveticaNeue", "Helvetica-Neue", "Helvetica", "Arial",
    #                                    "Liberation"]

    epoch_list = [20]
    for epoch in epoch_list:

        net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        # n_particle_types = 3
        # index_particles = []
        # for n in range(n_particle_types):
        #     index_particles.append(
        #         np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
        # type = torch.zeros(int(n_particles / n_particle_types), device=device)
        # for n in range(1, n_particle_types):
        #     type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
        # x[:,5]=type

        # n_particles = int(n_particles * (1-train_config.dropout))
        # types = to_numpy(x[:, 5])
        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        # embedding = get_embedding(model.a, 1, index_particles, n_particles, n_particle_types)
        # for n in range(n_particle_types):
        #     pos = np.argwhere(types == n)
        #     plt.scatter(embedding[pos, 0],
        #                 embedding[pos, 1], color=cmap.color(n), s=50)
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        # plt.xticks(fontsize=32.0)
        # plt.yticks(fontsize=32.0)
        # plt.xlim([0,2])
        # plt.ylim([0, 2])
        # plt.tight_layout()

        matplotlib.use("Qt5Agg")
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=50)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.xlim([0,2])
        plt.ylim([0,2])
        plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        if model_config.particle_model_name == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif model_config.particle_model_name == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

        print(n_particles)

        func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=True, config=config,
                                                                model_lin_edge=model.lin_edge, model_a=model.a,
                                                                dataset_number=1,
                                                                n_particles=int(n_particles*(1-train_config.particle_dropout)), ynorm=ynorm,
                                                                types=to_numpy(x[:, 5]),
                                                                cmap=cmap, device=device)
        # plt.xlabel(r'$d_{ij}$', fontsize=64)
        # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim([-0.04, 0.03])
        plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()

        match train_config.cluster_method:
            case 'kmeans':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans')
            case 'kmeans_auto_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
            case 'kmeans_auto_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                proj_interaction = embedding
            case 'distance_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
            case 'distance_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=0.05)
                proj_interaction = embedding
            case 'distance_both':
                new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
                labels, n_clusters = embedding_cluster.get(new_projection, 'distance')

        fig_ = plt.figure(figsize=(12, 12))
        axf = fig_.add_subplot(1, 1, 1)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        axf.yaxis.set_major_locator(plt.MaxNLocator(3))
        axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for n in range(n_clusters):
            pos = np.argwhere(labels == n)
            pos = np.array(pos)
            if pos.size > 0:
                print(f'cluster {n}  {len(pos)}')
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=100,alpha=0.1)
        label_list = []
        for n in range(n_particle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)
        plt.xlabel(r'UMAP-proj 0', fontsize=64)
        plt.ylabel(r'UMAP-proj 1', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/UMAP_{dataset_name}_{epoch}.tif", dpi=300)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        new_labels = labels.copy()
        for n in range(n_particle_types):
            new_labels[labels == label_list[n]] = n
            pos = np.argwhere(labels == label_list[n])
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                            color=cmap.color(n), s=0.1)
        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')

        fig = plt.figure(figsize=(12, 12))
        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                model_a_[pos, :] = median_center
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
        for n in np.unique(new_labels):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), color='k', s=5)
        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)
        plt.close()

        model_a_first = model.a.clone().detach()

        with torch.no_grad():
            model.a[1] = model_a_.clone().detach()

        if True:

            # matplotlib.use("Qt5Agg")

            fig_ = plt.figure(figsize=(12, 12))
            axf = fig_.add_subplot(1, 1, 1)
            axf.xaxis.set_major_locator(plt.MaxNLocator(3))
            axf.yaxis.set_major_locator(plt.MaxNLocator(3))
            axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            csv_ = []
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=400, alpha=0.1)
                csv_.append(embedding[index_particles[n], :])
            plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
            plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            # csv_ = np.array(csv_)
            plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif", dpi=300)
            # np.save(f"./{log_dir}/tmp_training/embedding_{dataset_name}.npy", csv_)
            # csv_= np.reshape(csv_,(csv_.shape[0]*csv_.shape[1],2))
            # np.savetxt(f"./{log_dir}/tmp_training/embedding_{dataset_name}.txt", csv_)
            plt.close()

            p = config.simulation.params
            if len(p) > 1:
                p = torch.tensor(p, device=device)
            else:
                p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt',map_location=device)


            rmserr_list = []
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1,1,1)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            csv_ = []
            csv_.append(to_numpy(rr))
            for n in range(int(n_particles*(1-train_config.particle_dropout))):
                embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                match config.graph_model.particle_model_name:
                    case 'PDE_A':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_), dim=1)
                    case 'PDE_A_bis':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_, embedding_), dim=1)
                    case 'PDE_B' | 'PDE_B_bis':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    case 'PDE_G':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    case 'PDE_GS':
                        in_features = torch.cat((rr[:, None] / max_radius, embedding_), dim=1)
                    case 'PDE_E':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_, embedding_), dim=-1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                csv_.append(to_numpy(func))
                true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int),:].squeeze(), p[to_numpy(type_list[n]).astype(int),:].squeeze())
                rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func) ** 2)))
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
            plt.xticks(fontsize=32)
            plt.yticks(fontsize=32)
            plt.xlabel(r'$d_{ij}$', fontsize=64)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
            plt.xlim([0, max_radius])
            # plt.ylim([-0.15, 0.15])
            # plt.ylim([-0.04, 0.03])
            plt.ylim([-0.1, 0.1])
            # plt.ylim([-0.03, 0.03])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/func_all_{dataset_name}_{epoch}.tif",dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')
            np.save(f"./{log_dir}/tmp_training/func_all_{dataset_name}_{epoch}.npy", csv_)
            np.savetxt(f"./{log_dir}/tmp_training/func_all_{dataset_name}_{epoch}.txt", csv_)
            plt.close()

            # func_list = []
            # fig = plt.figure(figsize=(12, 12))
            # ax = fig.add_subplot(1,1,1)
            # # ax.xaxis.get_major_formatter()._usetex = False
            # # ax.yaxis.get_major_formatter()._usetex = False
            # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            # for n in range(n_particle_types):
            #     pos = np.argwhere(new_labels == n).squeeze().astype(int)
            #     if pos.size > 0:
            #         embedding_ = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            #         match config.graph_model.particle_model_name:
            #             case 'PDE_A':
            #                 in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
            #                                          rr[:, None] / max_radius, embedding_), dim=1)
            #             case 'PDE_A_bis':
            #                 in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
            #                                          rr[:, None] / max_radius, embedding_, embedding_), dim=1)
            #             case 'PDE_B' | 'PDE_B_bis':
            #                 in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
            #                                          rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
            #                                          0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            #             case 'PDE_G':
            #                 in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
            #                                          rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
            #                                          0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            #             case 'PDE_GS':
            #                 in_features = torch.cat((rr[:, None] / max_radius, embedding_), dim=1)
            #             case 'PDE_E':
            #                 in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
            #                                          rr[:, None] / max_radius, embedding_, embedding_), dim=-1)
            #         with torch.no_grad():
            #             func = model.lin_edge(in_features.float())
            #         func = func[:, 0]
            #
            #         func_list.append(func)
            #         plt.plot(to_numpy(rr),
            #                  to_numpy(func) * to_numpy(ynorm),
            #                  color=cmap.color(n), linewidth=8)
            #         # plt.plot(to_numpy(rr),
            #         #          to_numpy(model.psi(rr, p[n], p[n])),
            #         #          color=cmap.color(n), linewidth=1)
            # # plt.xlabel(r'$d_{ij}$', fontsize=64)
            # # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
            # plt.xticks(fontsize=32)
            # plt.yticks(fontsize=32)
            # plt.xlim([0, max_radius])
            # # plt.ylim([-0.15, 0.15])
            # # plt.ylim([-0.04, 0.03])
            # plt.ylim([-0.1, 0.1])
            # plt.tight_layout()
            # plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif",dpi=170.7)
            # plt.close()


            # func_list = torch.stack(func_list) * ynorm
            # true_func_list = []
            # for n in range(n_particles):
            #     pos = np.argwhere(new_labels == n).squeeze().astype(int)
            #     if pos.size > 0:
            #         true_func_list.append(model.psi(rr, p[n], p[n]))
            # true_func_list = torch.stack(true_func_list)
            # rmserr = torch.sqrt(torch.mean((func_list - true_func_list) ** 2))
            # print(f'function unique RMS error: {np.round(rmserr.item(), 7)}')


            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1,1,1)
            # ax.xaxis.get_major_formatter()._usetex = False
            # ax.yaxis.get_major_formatter()._usetex = False
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            csv_ = []
            csv_.append(to_numpy(rr))
            for n in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
                csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
            # plt.xlabel(r'$d_{ij}$', fontsize=64)
            # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
            plt.xticks(fontsize=32)
            plt.yticks(fontsize=32)
            plt.xlim([0, max_radius])
            # plt.ylim([-0.15, 0.15])
            plt.ylim([-0.04, 0.03])
            # plt.ylim([-0.1, 0.1])

            plt.xlabel(r'$d_{ij}$', fontsize=64)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/true_func_{dataset_name}.tif",dpi=170.7)
            plt.close()
            np.save(f"./{log_dir}/tmp_training/true_func_{dataset_name}_{epoch}.npy", csv_)
            np.savetxt(f"./{log_dir}/tmp_training/true_func_{dataset_name}_{epoch}.txt", csv_)

def data_plot_training_asym(config, mode, device):
    print('')

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    radius = simulation_config.max_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    replace_with_cluster = 'replace' in train_config.sparsity
    visualize_embedding = False
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    aggr_type = config.graph_model.aggr_type

    embedding_cluster = EmbeddingCluster(config)

    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)

    x_list = []
    y_list = []
    print('Load data ...')
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)

    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in trange(n_frames):
            if (k%10 == 0) | (n_frames<1000):
                x = torch.cat((x,x_list[run][k].clone().detach()),0)
                y = torch.cat((y,y_list[run][k].clone().detach()),0)
        print(x_list[run][k].shape)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    vnorm = vnorm[4]
    ynorm = ynorm[4]
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    if has_mesh:
        x_mesh_list = []
        y_mesh_list = []
        for run in trange(NGraphs):
            x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
            x_mesh_list.append(x_mesh)
            h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
            y_mesh_list.append(h)
        h = y_mesh_list[0][0].clone().detach()
        for run in range(NGraphs):
            for k in range(n_frames):
                h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(f'hnorm: {to_numpy(hnorm)}')
        time.sleep(0.5)

        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)

        mask_mesh = mesh_data['mask']
        # mesh_pos = mesh_data['mesh_pos']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']
        # face = mesh_data['face']

        mask_mesh = mask_mesh.repeat(batch_size, 1)

    h=[]
    x=[]
    y=[]

    print('done ...')

    model, bc_pos, bc_dpos = choose_training_model(config, device)


    if  has_cell_division:
        model_division = Division_Predictor(config, device)
        optimizer_division, n_total_params_division = set_trainable_division_parameters(model_division, lr=1E-3)
        logger.info(f"Total Trainable Divsion Params: {n_total_params_division}")
        logger.info(f'Learning rates: 1E-3')

    x = x_list[1][0].clone().detach()
    type_list = x[:, 5:6].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    time.sleep(0.5)


    matplotlib.use("Qt5Agg")
    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0
    # style = {
    #     "pgf.rcfonts": False,
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,
    #     "font.family": "sans-serif"
    # }
    # matplotlib.rcParams.update(style)
    # plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "HelveticaNeue", "Helvetica-Neue", "Helvetica", "Arial",
    #                                    "Liberation"]

    epoch_list = [20]
    for epoch in epoch_list:

        net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        n_particle_types = 3
        index_particles = []
        for n in range(n_particle_types):
            index_particles.append(
                np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
        type = torch.zeros(int(n_particles / n_particle_types), device=device)
        for n in range(1, n_particle_types):
            type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
        x[:,5]=type

        # n_particles = int(n_particles * (1-train_config.dropout))
        # types = to_numpy(x[:, 5])
        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        # embedding = get_embedding(model.a, 1)
        # for n in range(n_particle_types):
        #     pos = np.argwhere(types == n)
        #     plt.scatter(embedding[pos, 0],
        #                 embedding[pos, 1], color=cmap.color(n), s=50)
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        # plt.xticks(fontsize=32.0)
        # plt.yticks(fontsize=32.0)
        # plt.xlim([0,2])
        # plt.ylim([0, 2])
        # plt.tight_layout()

        matplotlib.use("Qt5Agg")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=400)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        if model_config.particle_model_name == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif model_config.particle_model_name == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=True, config=config,
                                                                model_lin_edge=model.lin_edge, model_a=model.a,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                types=to_numpy(x[:, 5]),
                                                                cmap=cmap, device=device)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim([-0.04, 0.03])
        plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()


        match train_config.cluster_method:
            case 'kmeans_auto_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
            case 'kmeans_auto_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                proj_interaction = embedding
            case 'distance_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
            case 'distance_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=1.5)
                proj_interaction = embedding
            case 'distance_both':
                new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
                labels, n_clusters = embedding_cluster.get(new_projection, 'distance')

        fig = plt.figure(figsize=(12, 12))
        for n in range(n_clusters):
            pos = np.argwhere(labels == n)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=5)
        label_list = []
        for n in range(n_particle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)

        plt.xlabel('proj 0', fontsize=12)
        plt.ylabel('proj 1', fontsize=12)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        new_labels = labels.copy()
        for n in range(n_particle_types):
            new_labels[labels == label_list[n]] = n
            pos = np.argwhere(labels == label_list[n])
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                            color=cmap.color(n), s=0.1)
        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')

        fig = plt.figure(figsize=(12, 12))
        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                model_a_[pos, :] = median_center
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
        for n in np.unique(new_labels):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), color='k', s=5)
        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)
        plt.close()

        model_a_first = model.a.clone().detach()

        with torch.no_grad():
            model.a[1] = model_a_.clone().detach()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model_a_first, 1)
        csv_ = embedding
        np.save(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.txt", csv_)
        if n_particle_types > 1000:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                        cmap='viridis')
        else:
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n),
                            s=400, alpha=0.1)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif", dpi=170.7)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        func_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particle_types):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            embedding_1 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            for m in range(n_particle_types):
                pos = np.argwhere(new_labels == m).squeeze().astype(int)
                embedding_2 = model.a[1, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                 device=device)

                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                func_list.append(func)
                csv_.append(to_numpy(func))
                plt.plot(to_numpy(rr),
                         to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(n), linewidth=8)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim([-0.03, 0.03])
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif", dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.txt", csv_)
        plt.close()

        p = config.simulation.params
        p = config.simulation.params
        if len(p) > 0:
            p = torch.tensor(p, device=device)
        else:
            p = torch.load(f'graphs_data/graphs_{dataset_name}/p.pt')
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        true_func_list=[]
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particle_types):
            for m in range(n_particle_types):
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[3*n + m], p[n*3 +m])), color=cmap.color(n), linewidth=8)
                true_func_list.append(model.psi(rr, p[3*n + m], p[n*3 +m]))
                csv_.append(to_numpy(model.psi(rr, p[3*n + m], p[n*3 +m]).squeeze()))
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=64)
        # xticks with sans serif font
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim([-0.03, 0.03])
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{dataset_name}.tif", dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/true_func_{dataset_name}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/true_func_{dataset_name}_{epoch}.txt", csv_)
        plt.close()

        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)

        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2,axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')

def data_plot_training_continuous(config, mode, device):
    print('')

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension=simulation_config.dimension
    n_epochs = train_config.n_epochs
    radius = simulation_config.max_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    replace_with_cluster = 'replace' in train_config.sparsity
    visualize_embedding = False
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    aggr_type = config.graph_model.aggr_type

    embedding_cluster = EmbeddingCluster(config)

    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)

    x_list = []
    y_list = []
    print('Load data ...')
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)

    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in trange(n_frames):
            if (k%10 == 0) | (n_frames<1000):
                x = torch.cat((x,x_list[run][k].clone().detach()),0)
                y = torch.cat((y,y_list[run][k].clone().detach()),0)
        print(x_list[run][k].shape)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    vnorm = vnorm[4]
    ynorm = ynorm[4]
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    if has_mesh:
        x_mesh_list = []
        y_mesh_list = []
        for run in trange(NGraphs):
            x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
            x_mesh_list.append(x_mesh)
            h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
            y_mesh_list.append(h)
        h = y_mesh_list[0][0].clone().detach()
        for run in range(NGraphs):
            for k in range(n_frames):
                h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(f'hnorm: {to_numpy(hnorm)}')
        time.sleep(0.5)

        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)

        mask_mesh = mesh_data['mask']
        # mesh_pos = mesh_data['mesh_pos']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']
        # face = mesh_data['face']

        mask_mesh = mask_mesh.repeat(batch_size, 1)

    h=[]
    x=[]
    y=[]

    print('done ...')

    model, bc_pos, bc_dpos = choose_training_model(config, device)


    if  has_cell_division:
        model_division = Division_Predictor(config, device)
        optimizer_division, n_total_params_division = set_trainable_division_parameters(model_division, lr=1E-3)
        logger.info(f"Total Trainable Divsion Params: {n_total_params_division}")
        logger.info(f'Learning rates: 1E-3')

    x = x_list[1][0].clone().detach()
    type_list = x[:, 5:6].clone().detach()
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    time.sleep(0.5)


    matplotlib.use("Qt5Agg")
    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0
    # style = {
    #     "pgf.rcfonts": False,
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,
    #     "font.family": "sans-serif"
    # }
    # matplotlib.rcParams.update(style)
    # plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "HelveticaNeue", "Helvetica-Neue", "Helvetica", "Arial",
    #                                    "Liberation"]

    epoch_list = [20]
    for epoch in epoch_list:

        net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        n_particle_types = 3
        index_particles = []
        for n in range(n_particle_types):
            index_particles.append(
                np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
        type = torch.zeros(int(n_particles / n_particle_types), device=device)
        for n in range(1, n_particle_types):
            type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
        x[:,5]=type

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        embedding = get_embedding(model.a, 1)
        csv_ = embedding
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=400, alpha=0.1)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.tif",dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/embedding_{dataset_name}_{epoch}.txt", csv_)
        plt.close()


        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        func_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particles):
            embedding = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            csv=to_numpy(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n//1600), linewidth=8,alpha=0.1)
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        # plt.ylim([-0.15, 0.15])
        plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.06])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.tif",dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/func_{dataset_name}_{epoch}.txt", csv_)
        plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        if os.path.exists(f'graphs_data/graphs_{dataset_name}/model_p.pt'):
            p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt')
        else:
            p = config.simulation.params
        true_func_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particles):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n//1600), linewidth=8,alpha=0.1)
            true_func_list.append(model.psi(rr, p[n], p[n]))
            csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
        plt.xlabel(r'$d_{ij}$', fontsize=64)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        # plt.ylim([-0.15, 0.15])
        plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.06])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/true_func_{dataset_name}.tif",dpi=170.7)
        np.save(f"./{log_dir}/tmp_training/true_func_{dataset_name}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/tmp_training/true_func_{dataset_name}_{epoch}.txt", csv_)
        plt.close()

        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)

        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2,axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print(f'all function RMS error: {np.round(np.mean(rmserr_list), 7)}+/-{np.round(np.std(rmserr_list), 7)}')

def data_plot_training_particle_field(config, mode, device):
    print('')

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    n_nodes = simulation_config.n_nodes
    n_node_types = simulation_config.n_node_types
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    has_large_range = train_config.large_range
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)


    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)

    x_list = []
    y_list = []
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_1.pt', map_location=device))
    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_1.pt', map_location=device)
    x_mesh_list.append(x_mesh)
    y_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_1.pt', map_location=device)
    y_mesh_list.append(y_mesh)
    hnorm = torch.load(f'./log/try_{dataset_name}/hnorm.pt', map_location=device).to(device)

    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    index_particles = []
    x = x_list[0][0].clone().detach()
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    if has_ghost:
        ghosts_particles = Ghost_Particles(config, n_particles, device)
        if train_config.ghost_method == 'MLP':
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.data], lr=5E-4)
        else:
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    index_nodes = []
    x_mesh = x_mesh_list[0][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == -n - 1)
        index_nodes.append(index.squeeze())

    for epoch in trange (0, 20):

        net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_{epoch}.pt"
        state_dict = torch.load(net,map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(3, 3, 1)
        embedding = get_embedding(model.a, 1)
        embedding = embedding[n_nodes:]
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=10)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)
        plt.title('Particle embedding', fontsize=12)

        match train_config.cluster_method:
            case 'kmeans_auto_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                proj_interaction = embedding
            case 'distance_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=0.1)
                proj_interaction = embedding
        print(f'n_clusters: {n_clusters}')

        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[n_nodes + pos, :]
                median_center = torch.median(median_center, dim=0).values
                model_a_[n_nodes + pos, :] = median_center

        with torch.no_grad():
            model.a[1][n_nodes:, :] = model_a_[n_nodes:].clone().detach()

        ax = fig.add_subplot(3, 3, 2)
        embedding = get_embedding(model.a, 1)
        embedding = embedding[n_nodes:]
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=25)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)
        plt.title('Clustered particle embedding', fontsize=12)

        ax = fig.add_subplot(3, 3, 3)
        embedding = get_embedding(model.a, 1)
        embedding = embedding[:n_nodes, :]
        for n in range(n_node_types):
                plt.scatter(embedding[index_nodes[n], 0],
                            embedding[index_nodes[n], 1], color=cmap.color(n), s=25)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)
        plt.title('Field embedding', fontsize=12)

        ax = fig.add_subplot(3, 3, 4)
        plt.title('Pos rate', fontsize=12)
        uu = torch.tensor(np.linspace(500, 7500, 200)).to(device)
        popt_list = []
        for n in range(n_nodes):
            embedding_ = model.a[1, n, :] * torch.ones((200, 2), device=device)
            in_features = torch.cat((uu[:, None], uu[:, None]*0, embedding_), dim=1)
            h = model.lin_phi(in_features.float())
            h = h[:, 0]
            popt, pcov = curve_fit(linear_model, to_numpy(uu.squeeze()), to_numpy(h.squeeze()))
            popt_list.append(popt)
            if n%50:
                t = x_mesh[n,5:6]
                cc = cmap.color(int(to_numpy(-t)))
                plt.plot(to_numpy(uu), to_numpy(h), color=cc, linewidth=2,alpha=0.01)
        ax = fig.add_subplot(3, 3, 5)
        plt.title('Pos rate', fontsize=12)
        t = np.array(popt_list)
        t = t[:, 0]
        t = np.reshape(t, (100, 100))
        plt.imshow(t, cmap='viridis')
        plt.xticks([])
        plt.yticks([])

        ax = fig.add_subplot(3, 3, 7)
        plt.title('Neg rate', fontsize=12)
        popt_list0 = []
        popt_list1 = []
        r = torch.tensor(np.linspace(0, 1, 200)).to(device)
        for n in range(n_nodes):
            embedding_ = model.a[1, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
            in_features = torch.cat((r[:, None], embedding_), dim=1)
            with torch.no_grad():
                h = model.lin_mesh(in_features.float())
                h = h[:, 0]
                popt_list0.append(torch.mean(h))
                popt, pcov = curve_fit(linear_model, to_numpy(r.squeeze()), to_numpy(h.squeeze()))
                popt_list1.append(popt)
                if n % 50:
                    t = x_mesh[n, 5:6]
                    cc = cmap.color(int(to_numpy(-t)))
                    plt.plot(to_numpy(r),
                             to_numpy(h),
                             color=cc, linewidth=2,alpha=0.0025)

        ax = fig.add_subplot(3, 3, 8)
        plt.title('Neg rate', fontsize=12)
        t = torch.stack(popt_list0)
        t = to_numpy(t)
        t = t[:, None]
        t = np.reshape(t, (100, 100))
        plt.imshow(t, cmap='viridis')
        plt.xticks([])
        plt.yticks([])

        ax = fig.add_subplot(3, 3, 9)
        t = np.array(popt_list1)
        t = t[:, 0]
        t = np.reshape(t, (100, 100))
        plt.imshow(t, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        fig.savefig(f"./{log_dir}/tmp_training/Analysis_{dataset_name}_{epoch}.tif",dpi=170.7)
        plt.close()


if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')

    config_list = ['arbitrary_16','arbitrary_16_noise_1E-1','arbitrary_16_noise_0_2','arbitrary_16_noise_0_3']

    # config_list = ['arbitrary_3_dropout_10_no_ghost','arbitrary_3_dropout_10','arbitrary_3_dropout_20','arbitrary_3_dropout_30','arbitrary_3_dropout_40']

    for config_file in config_list:

        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        # print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        cmap = CustomColorMap(config=config)  # create colormap for given model_config

        data_plot_training(config, mode='figures' , device=device)



