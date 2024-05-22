from GNN_particles_Ntype import *

def data_generate(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):

    has_particle_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)
    has_mesh = (config.graph_model.mesh_model_name != '')

    if has_particle_field:
        data_generate_particle_field(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=False, step=step,
                                     alpha=0.2, ratio=1,
                                     scenario='none', device=None, bSave=True)
    elif has_mesh:
        data_generate_mesh(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=erase, step=step,
                                        alpha=0.2, ratio=1,
                                        scenario=scenario, device=device, bSave=bSave)

    else:
        data_generate_particle(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=erase, step=step,
                                        alpha=0.2, ratio=1,
                                        scenario=scenario, device=device, bSave=bSave)


def data_generate_particle(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2,
                           ratio=1, scenario='none', device=None, bSave=True):
    print('')

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    has_signal = (config.graph_model.signal_model_name != '')
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    has_cell_division = simulation_config.has_cell_division
    n_frames = simulation_config.n_frames
    cycle_length = None
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    if config.data_folder_name != 'none':
        print(f'Generating from data ...')
        generate_from_data(config=config, device=device, visualize=visualize, folder=folder, step=step)
        return

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/generated_data/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/generated_data/*')
    for f in files:
        os.remove(f)

    model, bc_pos, bc_dpos = choose_model(config, device=device)
    particle_dropout_mask = np.arange(n_particles)
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    if has_adjacency_matrix:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    for run in range(config.training.n_runs):

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1, cycle_length, cycle_length_distrib = init_particles(config, device=device, cycle_length=cycle_length)
        index_particles = []
        for n in range(n_particle_types):
            pos = torch.argwhere(T1 == n)
            pos = to_numpy(pos[:, 0].squeeze()).astype(int)
            index_particles.append(pos)

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            # calculate cell division
            if (it >= 0) & has_cell_division & (n_particles < 20000):
                pos = torch.argwhere(A1.squeeze() > cycle_length_distrib)
                y_division = (A1.squeeze() > cycle_length_distrib).clone().detach() * 1.0
                # cell division
                if len(pos) > 1:
                    n_add_nodes = len(pos)
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    y_division = torch.concatenate((y_division, torch.zeros((n_add_nodes), device=device)), 0)
                    n_particles = n_particles + n_add_nodes
                    N1 = torch.arange(n_particles, device=device)
                    N1 = N1[:, None]
                    separation = 1E-3 * torch.randn((n_add_nodes, 2), device=device)
                    X1 = torch.cat((X1, X1[pos, :] + separation), dim=0)
                    X1[pos, :] = X1[pos, :] - separation
                    phi = torch.randn(n_add_nodes, dtype=torch.float32, requires_grad=False,
                                      device=device) * np.pi * 2
                    cos_phi = torch.cos(phi)
                    sin_phi = torch.sin(phi)
                    new_x = cos_phi * V1[pos, 0] + sin_phi * V1[pos, 1]
                    new_y = -sin_phi * V1[pos, 0] + cos_phi * V1[pos, 1]
                    V1[pos, 0] = new_x
                    V1[pos, 1] = new_y
                    V1 = torch.cat((V1, -V1[pos, :]), dim=0)
                    T1 = torch.cat((T1, T1[pos, :]), dim=0)
                    H1 = torch.cat((H1, H1[pos, :]), dim=0)
                    A1[pos, :] = 0
                    A1 = torch.cat((A1, A1[pos, :]), dim=0)
                    nd = torch.ones(len(pos), device=device) + 0.05 * torch.randn(len(pos), device=device)
                    cycle_length_distrib = torch.cat(
                        (cycle_length_distrib, cycle_length[to_numpy(T1[pos, 0])].squeeze() * nd), dim=0)
                    y_timer = A1.squeeze().clone().detach()
                    index_particles = []
                    for n in range(n_particles):
                        pos = torch.argwhere(T1 == n)
                        pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                        index_particles.append(pos)

            x = torch.concatenate(
                (N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                 H1.clone().detach(), A1.clone().detach()), 1)

            # compute connectivity rule
            if has_adjacency_matrix:
                adj_t = adjacency > 0
                edge_index = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
            else:
                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

            # model prediction
            with torch.no_grad():
                y = model(dataset)

            # append list
            if (it >= 0) & bSave:

                if has_cell_division:
                    x_list.append(x.clone().detach())
                    y_ = torch.concatenate((y, y_timer[:, None], y_division[:, None]), 1)
                    y_list.append(y_.clone().detach())
                else:
                    if has_particle_dropout:
                        x_ = x[particle_dropout_mask].clone().detach()
                        x_[:, 0] = torch.arange(len(x_), device=device)
                        x_list.append(x_)
                        x_ = x[inv_particle_dropout_mask].clone().detach()
                        x_[:, 0] = torch.arange(len(x_), device=device)
                        x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                        y_list.append(y[particle_dropout_mask].clone().detach())
                    else:
                        x_list.append(x.clone().detach())
                        y_list.append(y.clone().detach())

            # Particle update
            if has_signal:
                H1[:, 1] = y.squeeze()
                H1[:, 0] = H1[:, 0] + H1[:, 1] * delta_t
            else:
                if model_config.prediction == '2nd_derivative':
                    V1 += y * delta_t
                else:
                    V1 = y
                X1 = bc_pos(X1 + V1 * delta_t)
            A1 = A1 + delta_t

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'bw' in style:

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    s_p = 100
                    if simulation_config.has_cell_division:
                        s_p = 25
                    if False:  # config.simulation.non_discrete_level>0:
                        plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=s_p, color='k')
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                        s=s_p, color='k')
                    if training_config.particle_dropout > 0:
                        plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                    x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                    alpha=0.75)
                        plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                 x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if 'PDE_G' in model_config.particle_model_name:
                        plt.xlim([-2, 2])
                        plt.ylim([-2, 2])
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                    elif 'frame' in style:
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

                if 'color' in style:

                    if model_config.particle_model_name == 'PDE_O':
                        fig = plt.figure(figsize=(12, 12))
                        plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=100,
                                    c=np.sin(to_numpy(H1[:, 2])), vmin=-1, vmax=1, cmap='viridis')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Lut_Fig_{run}_{it}.jpg",
                                    dpi=170.7)
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        # plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=5, c='b')
                        plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=10, c='lawngreen',
                                    alpha=0.75)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Rot_{run}_Fig{it}.jpg",
                                    dpi=170.7)
                        plt.close()

                    elif model_config.signal_model_name == 'PDE_N':

                        matplotlib.rcParams['savefig.pad_inches'] = 0
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=200, c=to_numpy(H1[:, 0]), cmap='cool',
                                    vmin=0, vmax=3)
                        plt.xlim([-1.5, 1.5])
                        plt.ylim([-1.5, 1.5])
                        plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=24)
                        # cbar = plt.colorbar(shrink=0.5)
                        # cbar.ax.tick_params(labelsize=32)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{10000 + it}.tif",
                                    dpi=42.675)
                        plt.close()

                    elif (model_config.particle_model_name == 'PDE_A') & (dimension == 3):

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(111, projection='3d')
                        for n in range(n_particle_types):
                            ax.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                       to_numpy(x[index_particles[n], 3]), s=50, color=cmap.color(n))
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.set_zlim([0, 1])
                        pl.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                        plt.close()

                    else:
                        # matplotlib.use("Qt5Agg")

                        matplotlib.rcParams['savefig.pad_inches'] = 0
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        ax.xaxis.get_major_formatter()._usetex = False
                        ax.yaxis.get_major_formatter()._usetex = False
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        s_p = 100
                        if simulation_config.has_cell_division:
                            s_p = 25
                        for n in range(n_particle_types):
                                plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                            s=s_p, color=cmap.color(n))
                        if training_config.particle_dropout > 0:
                            plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                        x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                        alpha=0.75)
                            plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                     x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        if 'PDE_G' in model_config.particle_model_name:
                            plt.xlim([-2, 2])
                            plt.ylim([-2, 2])
                        if 'latex' in style:
                            plt.xlabel(r'$x$', fontsize=64)
                            plt.ylabel(r'$y$', fontsize=64)
                            plt.xticks(fontsize=32.0)
                            plt.yticks(fontsize=32.0)
                        elif 'frame' in style:
                            plt.xlabel('x', fontsize=32)
                            plt.ylabel('y', fontsize=32)
                            plt.xticks(fontsize=32.0)
                            plt.yticks(fontsize=32.0)
                            ax.tick_params(axis='both', which='major', pad=15)
                            plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
                        else:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.tif", dpi=170.7)
                        # plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{10000+it}.tif", dpi=42.675)
                        plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            if has_particle_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
                np.save(f'graphs_data/graphs_{dataset_name}/particle_dropout_mask.npy', particle_dropout_mask)
                np.save(f'graphs_data/graphs_{dataset_name}/inv_particle_dropout_mask.npy', inv_particle_dropout_mask)
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(cycle_length, f'graphs_data/graphs_{dataset_name}/cycle_length.pt')
            torch.save(cycle_length_distrib, f'graphs_data/graphs_{dataset_name}/cycle_length_distrib.pt')
            torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')

            # if model_config.signal_model_name == 'PDE_N' & (run == run_vizualized):
            #     matplotlib.rcParams['savefig.pad_inches'] = 0
            #     fig = plt.figure(figsize=(12, 12))
            #     signal=[]
            #     for k in range(len(x_list)):
            #         signal.append(x_list[k][:,6:7])
            #     signal = torch.stack(signal)
            #     signal = to_numpy(signal.squeeze())
            #     plt.imshow(signal, aspect='auto', cmap='viridis')
            #     plt.xticks([])
            #     plt.yticks([])


def data_generate_mesh(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):
    print('')

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/generated_data/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/generated_data/*')
    for f in files:
        os.remove(f)
    mesh_model = choose_mesh_model(config, device=device)

    for run in range(config.training.n_runs):

        X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, model_mesh=mesh_model, device=device)
        torch.save(mesh_data, f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt')
        mask_mesh = mesh_data['mask'].squeeze()
        if 'pics' in simulation_config.node_type_map:
            i0 = imread(f'graphs_data/{simulation_config.node_type_map}')
            values = i0[(to_numpy(X1_mesh[:, 0]) * 255).astype(int), (to_numpy(X1_mesh[:, 1]) * 255).astype(int)]
            values = np.reshape(values,len(X1_mesh))
            mesh_model.coeff = torch.tensor(values, device=device, dtype=torch.float32)[:, None]

        time.sleep(0.5)
        x_mesh_list=[]
        y_mesh_list=[]
        for it in trange(simulation_config.start_frame, n_frames + 1):

            x_mesh = torch.concatenate(
                (N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
                 T1_mesh.clone().detach(), H1_mesh.clone().detach()), 1)
            x_mesh_list.append(x_mesh.clone().detach())

            dataset_mesh = data.Data(x=x_mesh, edge_index=mesh_data['edge_index'],
                                     edge_attr=mesh_data['edge_weight'], device=device)


            match config.graph_model.mesh_model_name:
                case 'DiffMesh':
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                        H1[mask_mesh, 1:2] = pred[mask_mesh]
                    H1_mesh[mask_mesh, 0:1] += pred[mask_mesh, 0:1] * delta_t
                    new_pred = torch.zeros_like(pred)
                    new_pred[mask_mesh] = pred[mask_mesh]
                    pred = new_pred
                case 'WaveMesh':
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                    H1_mesh[mask_mesh, 1:2] += pred[mask_mesh, :] * delta_t
                    H1_mesh[mask_mesh, 0:1] += H1_mesh[mask_mesh, 1:2] * delta_t
                    # x_ = to_numpy(x_mesh)
                    # plt.scatter(x_[:, 1], x_[:, 2], c=to_numpy(H1_mesh[:, 0]))
                case 'RD_Gray_Scott_Mesh' | 'RD_FitzHugh_Nagumo_Mesh' | 'RD_RPS_Mesh' | 'RD_RPS_Mesh_bis':
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                        H1_mesh[mesh_data['mask'].squeeze(), :] += pred[mesh_data['mask'].squeeze(), :] * delta_t
                        H1_mesh[mask_mesh.squeeze(), 6:9] = torch.clamp(H1_mesh[mask_mesh.squeeze(), 6:9], 0, 1)
                        H1 = H1_mesh.clone().detach()
                case 'PDE_O_Mesh':
                    pred = []

            y_mesh_list.append(pred)

            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'graph' in style:

                    fig = plt.figure(figsize=(12, 12))
                    match model_config.mesh_model_name:
                        case 'RD_RPS_Mesh':
                            H1_IM = torch.reshape(x_mesh[:, 6:9], (100, 100, 3))
                            plt.imshow(to_numpy(H1_IM), vmin=0, vmax=1)
                        case 'Wave_Mesh' | 'DiffMesh':
                            pts = x_mesh[:, 1:3].detach().cpu().numpy()
                            tri = Delaunay(pts)
                            colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                          vmax=2500)
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_g_color_{it}.tif", dpi=300)
                    plt.close()

                if 'color' in style:

                    # matplotlib.use("Qt5Agg")

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.xaxis.get_major_formatter()._usetex = False
                    ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                    pts = x_mesh[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                    match model_config.mesh_model_name:
                        case 'DiffMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                        case 'WaveMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                            fmt = lambda x, pos: '{:.1f}'.format((x)/100, pos)
                            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                        case 'RD_Gray_Scott_Mesh':
                            fig = plt.figure(figsize=(12, 6))
                            ax = fig.add_subplot(1, 2, 1)
                            colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.axis('off')
                        case 'RD_RPS_Mesh' | 'RD_RPS_Mesh_bis':
                            H1_IM = torch.reshape(H1, (100, 100, 3))
                            plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                            fmt = lambda x, pos: '{:.1f}'.format((x)/100, pos)
                            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            # plt.xticks([])
                            # plt.yticks([])
                            # plt.axis('off')`
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                    elif 'frame' in style:
                        plt.xlabel('x', fontsize=32)
                        plt.ylabel('y', fontsize=32)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                        ax.tick_params(axis='both', which='major', pad=15)
                        plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
                    else:
                        plt.xticks([])
                        plt.yticks([])

                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.tif", dpi=170.7)
                    # plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{10000+it}.tif", dpi=42.675)
                    plt.close()

        if bSave:
            torch.save(x_mesh_list, f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt')
            torch.save(y_mesh_list, f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt')


def data_generate_particle_field(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):
    print('')

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    delta_t = simulation_config.delta_t
    has_signal = (config.graph_model.signal_model_name != '')
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_cell_division = simulation_config.has_cell_division
    n_frames = simulation_config.n_frames
    cycle_length = None
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    if config.data_folder_name != 'none':
        generate_from_data(config=config, device=device, visualize=visualize, folder=folder, step=step)
        return

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/generated_data/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/generated_data/*')
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    model_p_p, bc_pos, bc_dpos = choose_model(config, device=device)
    model_f_p = model_p_p

    model_f_f = choose_mesh_model(config, device=device)

    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        particle_dropout_mask = np.arange(n_particles)
    if has_adjacency_matrix:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    for run in range(config.training.n_runs):

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        x_mesh_list = []
        y_mesh_list = []
        edge_p_p_list = []
        edge_p_f_list = []
        edge_f_f_list = []
        edge_f_p_list = []


        # initialize particle and mesh states
        X1, V1, T1, H1, A1, N1, cycle_length, cycle_length_distrib = init_particles(config, device=device,cycle_length=cycle_length)
        X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, model_mesh=model_f_f, device=device)

        # matplotlib.use("Qt5Agg")
        # fig = plt.figure(figsize=(12, 12))
        # im = torch.reshape(H1_mesh[:,0:1],(100,100))
        # plt.imshow(to_numpy(im))
        # plt.colorbar()

        torch.save(mesh_data, f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt')
        mask_mesh = mesh_data['mask'].squeeze()
        index_particles = []
        for n in range(n_particle_types):
            pos = torch.argwhere(T1 == n)
            pos = to_numpy(pos[:, 0].squeeze()).astype(int)
            index_particles.append(pos)

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            if model_config.field_type == 'siren_with_time':

                if 'video' in simulation_config.node_value_map:
                    im = imread(f"graphs_data/{simulation_config.node_value_map}") / 255 * 5000
                    im = np.reshape(im[it], (n_nodes_per_axis * n_nodes_per_axis))
                    H1_mesh[:, 0:1] = torch.tensor(im[:,None], dtype=torch.float32, device=device)
                else:
                    H1_mesh = rotate_init_mesh(it, config, device=device)
                    im = torch.reshape(H1_mesh[:, 0:1], (n_nodes_per_axis, n_nodes_per_axis))
                # io.imsave(f"graphs_data/graphs_{dataset_name}/generated_data/rotated_image_{it}.tif", to_numpy(im))

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach()), 1)

            x_mesh = torch.concatenate(
                (N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
                 T1_mesh.clone().detach(), H1_mesh.clone().detach(), A1_mesh.clone().detach()), 1)
            x_particle_field = torch.concatenate((x_mesh, x), dim=0)

            # compute connectivity rules
            dataset_mesh = data.Data(x=x_mesh, edge_index=mesh_data['edge_index'],
                                     edge_attr=mesh_data['edge_weight'], device=device)

            distance = torch.sum(bc_dpos(x[:, None, 1:dimension+1] - x[None, :, 1:dimension+1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset_p_p = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])
            if not(has_particle_dropout):
                edge_p_p_list.append(edge_index)

            distance = torch.sum(bc_dpos(x_particle_field[:, None, 1:dimension+1] - x_particle_field[None, :, 1:dimension+1]) ** 2, dim=2)
            adj_t = ((distance < (max_radius/2) ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            pos = torch.argwhere((edge_index[1,:]>=n_nodes) & (edge_index[0,:]<n_nodes))
            pos = to_numpy(pos[:,0])
            edge_index = edge_index[:,pos]
            dataset_f_p = data.Data(x=x_particle_field, pos=x_particle_field[:, 1:3], edge_index=edge_index, field=x_particle_field[:,6:7])
            if not (has_particle_dropout):
                edge_f_p_list.append(edge_index)

            # model prediction
            with torch.no_grad():
                y0 = model_p_p(dataset_p_p)
                y1 = model_f_p(dataset_f_p)[n_nodes:]
                y = y0 + y1

            # append list
            if (it >= 0) & bSave:
                if has_particle_dropout:

                    x_ = x[inv_particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                    x_ = x[particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_list.append(x_)
                    y_list.append(y[particle_dropout_mask].clone().detach())

                    distance = torch.sum(bc_dpos(x_[:, None, 1:dimension + 1] - x_[None, :, 1:dimension + 1]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    edge_p_p_list.append(edge_index)

                    x_particle_field = torch.concatenate((x_mesh, x_), dim=0)

                    distance = torch.sum(bc_dpos(
                        x_particle_field[:, None, 1:dimension + 1] - x_particle_field[None, :, 1:dimension + 1]) ** 2, dim=2)
                    adj_t = ((distance < (max_radius / 2) ** 2) & (distance > min_radius ** 2)).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    pos = torch.argwhere((edge_index[1, :] >= n_nodes) & (edge_index[0, :] < n_nodes))
                    pos = to_numpy(pos[:, 0])
                    edge_index = edge_index[:, pos]
                    edge_f_p_list.append(edge_index)

                else:
                    x_list.append(x.clone().detach())
                    y_list.append(y.clone().detach())

            # Particle update
            if model_config.prediction == '2nd_derivative':
                V1 += y * delta_t
            else:
                V1 = y
            X1 = bc_pos(X1 + V1 * delta_t)

            A1 = A1 + delta_t

            # Mesh update
            x_mesh_list.append(x_mesh.clone().detach())
            pred = x_mesh[:,6:7]
            y_mesh_list.append(pred)

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'graph' in style:

                    fig = plt.figure(figsize=(10, 10))

                    if model_config.mesh_model_name == 'RD_RPS_Mesh':
                        H1_IM = torch.reshape(x_mesh[:, 6:9], (100, 100, 3))
                        plt.imshow(to_numpy(H1_IM), vmin=0, vmax=1)
                    elif (model_config.mesh_model_name == 'Wave_Mesh') | (model_config.mesh_model_name =='DiffMesh') :
                        pts = x_mesh[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                        if model_config.mesh_model_name == 'WaveMesh' :
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                          vmax=2500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=5000)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    elif model_config.particle_model_name == 'PDE_G':
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=40, color=cmap.color(n))
                    elif model_config.particle_model_name == 'PDE_E':
                        for n in range(n_particle_types):
                            g = 40
                            if simulation_config.params[n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n),
                                        alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_g_color_{it}.tif", dpi=300)
                    plt.close()

                if 'bw' in style:

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    s_p = 50
                    if simulation_config.has_cell_division:
                        s_p = 25
                    if False:  # config.simulation.non_discrete_level>0:
                        plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=s_p, color='k')
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                        s=s_p, color='k')
                    if training_config.particle_dropout > 0:
                        plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                    x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                    alpha=0.75)
                        plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                 x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

                if 'color' in style:

                    # matplotlib.use("Qt5Agg")
                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    # ax.xaxis.get_major_formatter()._usetex = False
                    # ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    # if (has_mesh | (simulation_config.boundary == 'periodic')):
                    #     ax = plt.axes([0, 0, 1, 1], frameon=False)
                    # else:
                    #     ax = plt.axes([-2, -2, 2, 2], frameon=False)
                    # ax.get_xaxis().set_visible(False)
                    # ax.get_yaxis().set_visible(False)
                    # plt.autoscale(tight=True)
                    if has_mesh:
                        pts = x_mesh[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                        match model_config.mesh_model_name:
                            case 'DiffMesh':
                                plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                              facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                            case 'WaveMesh':
                                plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                              facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                            case 'RD_Gray_Scott_Mesh':
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
                            case 'RD_RPS_Mesh':
                                fig = plt.figure(figsize=(12, 12))
                                H1_IM = torch.reshape(H1, (100, 100, 3))
                                plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                                plt.xticks([])
                                plt.yticks([])
                                plt.axis('off')
                    else:
                        s_p = 100
                        if simulation_config.has_cell_division:
                            s_p = 25
                        if False:  # config.simulation.non_discrete_level>0:
                            plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=s_p, color='k')
                        else:
                            for n in range(n_particle_types):
                                plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                            s=s_p, color=cmap.color(n))
                        if training_config.particle_dropout > 0:
                            plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                        x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                        alpha=0.75)
                            plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                     x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                    # plt.xlim([-2,2])
                    # plt.ylim([-2,2])
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=64)
                        plt.ylabel(r'$y$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                    elif 'frame' in style:
                        plt.xlabel('x', fontsize=64)
                        plt.ylabel('$', fontsize=64)
                        plt.xticks(fontsize=32.0)
                        plt.yticks(fontsize=32.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

                    if False:  # not(has_mesh):
                        fig = plt.figure(figsize=(12, 12))
                        s_p = 25
                        if simulation_config.has_cell_division:
                            s_p = 10
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=s_p, color='k')
                        if (simulation_config.boundary == 'periodic'):
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                        else:
                            plt.xlim([-4, 4])
                            plt.ylim([-4, 4])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_bw_{it}.jpg", dpi=170.7)
                        plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            if has_particle_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
                np.save(f'graphs_data/graphs_{dataset_name}/particle_dropout_mask.npy', particle_dropout_mask)
                np.save(f'graphs_data/graphs_{dataset_name}/inv_particle_dropout_mask.npy', inv_particle_dropout_mask)
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(x_mesh_list, f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt')
            torch.save(y_mesh_list, f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt')
            torch.save(edge_p_p_list, f'graphs_data/graphs_{dataset_name}/edge_p_p_list{run}.pt')
            torch.save(edge_f_p_list, f'graphs_data/graphs_{dataset_name}/edge_f_p_list{run}.pt')
            torch.save(cycle_length, f'graphs_data/graphs_{dataset_name}/cycle_length.pt')
            torch.save(cycle_length_distrib, f'graphs_data/graphs_{dataset_name}/cycle_length_distrib.pt')
            torch.save(model_p_p.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')
