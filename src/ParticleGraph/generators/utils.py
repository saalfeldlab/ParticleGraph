
from ParticleGraph.generators import *
from ParticleGraph.utils import *
from ParticleGraph.data_loaders import load_solar_system, load_LG_ODE, load_WaterRampsWall, load_2D_cell_data, load_3D_cell_data
from time import sleep
from scipy.spatial import Delaunay
from tifffile import imread, imsave
from torch_geometric.utils import get_mesh_laplacian
from tqdm import trange
from torch_geometric.utils import dense_to_sparse
from scipy import stats
import seaborn as sns

def generate_from_data(config, device, visualize=True, step=None, cmap=None):

    data_folder_name = config.data_folder_name
    image_data = config.image_data

    if data_folder_name == 'graphs_data/solar_system':
        load_solar_system(config, device, visualize, step)
    elif 'LG-ODE' in data_folder_name:
        load_LG_ODE(config, device, visualize, step)
    elif 'WaterDropSmall' in data_folder_name:
        load_WaterDropSmall(config, device, visualize, step, cmap)
    elif 'WaterRamps' in data_folder_name:
        load_WaterRampsWall(config, device, visualize, step, cmap)
    elif image_data.file_type != 'none':
        if image_data.file_type == '3D masks meshes':
            load_3D_cell_data(config, device, visualize)
        if image_data.file_type == '2D fluo':
            load_2D_cell_data(config, device, visualize)
    else:
        raise ValueError(f'Unknown data folder name {data_folder_name}')


def choose_model(config=[], W=[], device=[]):
    particle_model_name = config.graph_model.particle_model_name
    model_signal_name = config.graph_model.signal_model_name
    aggr_type = config.graph_model.aggr_type
    n_particles = config.simulation.n_particles
    delta_t = config.simulation.delta_t
    n_particle_types = config.simulation.n_particle_types
    short_term_plasticity_mode = config.simulation.short_term_plasticity_mode

    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)

    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius

    params = config.simulation.params
    p = torch.tensor(params, dtype=torch.float32, device=device).squeeze()

    # create GNN depending in type specified in config file
    match particle_model_name:
        case 'PDE_A' | 'PDE_ParticleField_A' | 'PDE_Cell_A' :
            if config.simulation.non_discrete_level>0:
                p = torch.ones(n_particle_types, 4, device=device) + torch.rand(n_particle_types, 4, device=device)
                pp=[]
                n_particle_types = len(params)
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
                for n in range(n_particle_types):
                    if n==0:
                        pp=p[n].repeat(n_particles//n_particle_types,1)
                    else:
                        pp=torch.cat((pp,p[n].repeat(n_particles//n_particle_types,1)),0)
                p=pp.clone().detach()
                p=p+torch.randn(n_particles,4,device=device) * config.simulation.non_discrete_level
            sigma = config.simulation.sigma
            p = p if n_particle_types == 1 else torch.squeeze(p)
            func_p = config.simulation.func_params
            embedding_step = config.simulation.n_frames // 100
            model = PDE_A(aggr_type=aggr_type, p=p, func_p = func_p, sigma=sigma, bc_dpos=bc_dpos, dimension=dimension, embedding_step=embedding_step)
        case 'PDE_B' | 'PDE_ParticleField_B' | 'PDE_Cell_B' | 'PDE_Cell_B_area':  # comprised between 10 and 50
            model = PDE_B(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_B_mass':
            final_cell_mass = torch.tensor(config.simulation.final_cell_mass, device=device)
            model = PDE_B_mass(aggr_type=aggr_type, p=p, final_mass = final_cell_mass, bc_dpos=bc_dpos)
        case 'PDE_B_bis':
            model = PDE_B_bis(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos)
        case 'PDE_G':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            model = PDE_G(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_GS':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            model = PDE_GS(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_E':
            model = PDE_E(aggr_type=aggr_type, p=p,
                          clamp=config.training.clamp, pred_limit=config.training.pred_limit,
                          prediction=config.graph_model.prediction, bc_dpos=bc_dpos)
        case 'PDE_F':
            model = PDE_F(aggr_type=aggr_type, p=torch.tensor(params, dtype=torch.float32, device=device), bc_dpos=bc_dpos,
                          dimension=dimension, delta_t=delta_t, max_radius=max_radius, field_type=config.graph_model.field_type)
        case 'PDE_K':
            p = params
            edges = np.random.choice(p[0], size=(n_particles, n_particles), p=p[1])
            edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)
            connection_matrix = torch.tensor(edges, dtype=torch.float32, device=device)
            model = PDE_K(aggr_type=aggr_type, connection_matrix=connection_matrix, bc_dpos=bc_dpos)

        case 'PDE_O':
            model = PDE_O(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, beta=config.simulation.beta)
        case 'Maze':
            model = PDE_B(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos)
        case _:
            model = PDE_Z(device=device)


    match config.simulation.phi:
        case 'tanh':
            phi=torch.tanh
        case 'relu':
            phi=torch.relu
        case 'sigmoid':
            phi=torch.sigmoid
        case _:
            phi=torch.sigmoid


    match model_signal_name:
        case 'PDE_N2':
            model = PDE_N2(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N3':
            model = PDE_N3(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N4':
            model = PDE_N4(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N5':
            model = PDE_N5(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N6':
            model = PDE_N6(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)
        case 'PDE_N7':
            model = PDE_N7(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)
        case 'PDE_N9':
            model = PDE_N4(aggr_type=aggr_type, p=p, W=W, phi=phi)


    return model, bc_pos, bc_dpos


def choose_mesh_model(config, X1_mesh, device):
    mesh_model_name = config.graph_model.mesh_model_name
    n_node_types = config.simulation.n_node_types
    aggr_type = config.graph_model.mesh_aggr_type
    _, bc_dpos = choose_boundary_values(config.simulation.boundary)

    params = config.simulation.params
    delta_t = config.simulation.delta_t
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius

    if mesh_model_name =='':
        mesh_model = []
    else:
        # c = initialize_random_values(n_node_types, device)
        # if not('pics' in config.simulation.node_coeff_map):
        #     for n in range(n_node_types):
        #         c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])

        if config.simulation.node_coeff_map !='' :
            i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
        else:
            i0 = imread(f'graphs_data/pattern_Null.tif')
        i0 = np.flipud(i0)
        values = i0[(to_numpy(X1_mesh[:, 1]) * 255).astype(int), (to_numpy(X1_mesh[:, 0]) * 255).astype(int)]
        values = np.reshape(values,len(X1_mesh))
        torch.tensor(values, device=device, dtype=torch.float32)[:, None]


        match mesh_model_name:
            case 'RD_Gray_Scott_Mesh':
                mesh_model = RD_Gray_Scott(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'RD_FitzHugh_Nagumo_Mesh':
                mesh_model = RD_FitzHugh_Nagumo(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'RD_RPS_Mesh':
                mesh_model = RD_RPS(aggr_type=aggr_type, bc_dpos=bc_dpos, coeff=values)
            case 'RD_RPS_Mesh_bis':
                mesh_model = RD_RPS(aggr_type=aggr_type, bc_dpos=bc_dpos)
            case 'DiffMesh' | 'WaveMesh':
                mesh_model = PDE_Laplacian(aggr_type=aggr_type, bc_dpos=bc_dpos, coeff=values)
            case 'WaveSmoothParticle':
                mesh_model = PDE_S(aggr_type=aggr_type, bc_dpos=bc_dpos, p=torch.tensor(params, dtype=torch.float32, device=device),
                          dimension=dimension, delta_t=delta_t, max_radius=max_radius, field_type=config.graph_model.field_type)
            case 'Chemotaxism_Mesh':
                c = initialize_random_values(n_node_types, device)
                for n in range(n_node_types):
                    c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
                mesh_model = PDE_Laplacian(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'PDE_O_Mesh':
                c = initialize_random_values(n_node_types, device)
                for n in range(n_node_types):
                    c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
                mesh_model = PDE_Laplacian(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case _:
                mesh_model = PDE_Z(device=device)


        i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
        i0 = np.flipud(i0)
        values = i0[(to_numpy(X1_mesh[:, 1]) * 255).astype(int), (to_numpy(X1_mesh[:, 0]) * 255).astype(int)]
        values = np.reshape(values,len(X1_mesh))
        mesh_model.coeff = torch.tensor(values, device=device, dtype=torch.float32)[:, None]

    return mesh_model


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)


def init_particles(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    n_frames = config.simulation.n_frames
    n_particles = simulation_config.n_particles * ratio
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if 'PDE_F' in config.graph_model.particle_model_name:
        pos = torch.rand(n_particles, dimension, device=device)
        if simulation_config.pos_init == 'square':
            pos = pos * 0.5 + 0.25
    elif (simulation_config.boundary == 'periodic'):
        pos = torch.rand(n_particles, dimension, device=device)
        if n_particles <= 10:
            if 'PDE_K' in config.graph_model.particle_model_name:
                pos = pos * 0.5 + 0.25
            else:
                pos = pos * 0.1 + 0.45
        elif n_particles<=100:
            if 'PDE_K' in config.graph_model.particle_model_name:
                pos = pos * 0.4 + 0.2
            else:
                pos = pos * 0.2 + 0.4
        elif n_particles<=500:
            pos = pos * 0.5 + 0.25
    elif 'PDE_N' in config.graph_model.signal_model_name:
        xc, yc = get_equidistant_points(n_points=n_particles)
        pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
        perm = torch.randperm(pos.size(0))
        pos = pos[perm]

    else:
        pos = torch.randn(n_particles, dimension, device=device) * 0.5
    dpos = dpos_init * torch.randn((n_particles, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    type = torch.zeros(int(n_particles / n_particle_types), device=device)
    for n in range(1, n_particle_types):
        type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
    if type.shape[0] < n_particles:
        type = torch.cat((type, n * torch.ones(n_particles - type.shape[0], device=device)), 0)
    if (simulation_config.params == 'continuous') | (config.simulation.non_discrete_level > 0):  # TODO: params is a list[list[float]]; this can never happen?
        type = torch.tensor(np.arange(n_particles), device=device)

    if 'PDE_F_special' in config.graph_model.particle_model_name:
        n_wall_particles = 1000
        real_n_particles = n_particles - n_wall_particles
        pos = torch.rand(real_n_particles, dimension, device=device)
        pos[:, 0] = pos[:, 0] * 0.2 + 0.75
        pos[:, 1] = pos[:, 1] * 0.95 + 0.025
        n_particles_wall = n_wall_particles // 4
        wall_pos = torch.linspace(0, 1, n_particles_wall, device=device)
        wall0 = torch.zeros(n_particles_wall, 2, device=device)
        wall0[:,0] = wall_pos
        wall1 = torch.zeros(n_particles_wall, 2, device=device)
        wall1[:,0] = wall_pos
        wall1[:,1] = 1
        wall2 = torch.zeros(n_particles_wall, 2, device=device)
        wall2[:,1] = wall_pos
        wall3 = torch.zeros(n_particles_wall, 2, device=device)
        wall3[:,1] = wall_pos
        wall3[:,0] = 1
        pos = torch.cat((wall0,wall1,wall2,wall3,pos), dim=0)

        pos = pos + torch.randn((n_particles,dimension), device=device) * 0.001

        dpos [0:n_wall_particles, :] = 0
        type = torch.cat((torch.zeros(n_wall_particles, device=device),torch.ones(real_n_particles, device=device)),0)


    if (config.graph_model.signal_model_name == 'PDE_N6') | (config.graph_model.signal_model_name == 'PDE_N7'):
        features = torch.cat((torch.rand((n_particles, 1), device=device), 0.1 * torch.randn((n_particles, 1), device=device),
                              torch.ones((n_particles, 1), device=device), torch.zeros((n_particles, 1), device=device)), 1)
    elif 'excitation_single' in config.graph_model.field_type:
        features = torch.zeros((n_particles, 2), device=device)
    else:
        # features = torch.cat((torch.rand((n_particles, 1), device=device), 0.1 * torch.randn((n_particles, 1), device=device)), 1)
        features = torch.cat((torch.randn((n_particles, 1), device=device) * 5 , 0.1 * torch.randn((n_particles, 1), device=device)), 1)


    type = type[:, None]
    particle_id = torch.arange(n_particles, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_particles,1), device=device)

    if 'pattern' in scenario:
        i0 = imread(f'graphs_data/pattern_0.tif')
        type = np.round(i0[(to_numpy(pos[:, 0]) * 255).astype(int), (to_numpy(pos[:, 1]) * 255).astype(int)] / 255 * n_particle_types-1).astype(int)
        type = torch.tensor(type, device=device)
        type = type[:, None]
    if 'uniform' in scenario :
        type = torch.ones(n_particles, device=device) * int(scenario.split()[-1])
        type =  type[:, None]
    if 'stripes' in scenario:
        l = n_particles//n_particle_types
        for n in range(n_particle_types):
            index = np.arange(n*l, (n+1)*l)
            pos[index, 1:2] = torch.rand(l, 1, device=device) * (1/n_particle_types) + n/n_particle_types


    return pos, dpos, type, features, age, particle_id


def get_index(n_particles, n_particle_types):
    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    return index_particles


def get_time_series(x_list, cell_id, feature):

    match feature:
        case 'velocity_x':
            feature = 3
        case 'velocity_y':
            feature = 4
        case 'type' | 'state':
            feature = 5
        case 'age':
            feature = 8
        case 'mass':
            feature = 10

        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))

    return to_numpy(torch.stack(time_series))


def init_mesh(config, device):

    simulation_config = config.simulation
    model_config = config.graph_model

    n_nodes = simulation_config.n_nodes
    n_particles = simulation_config.n_particles
    node_value_map = simulation_config.node_value_map
    field_grid = model_config.field_grid
    max_radius = simulation_config.max_radius

    n_nodes_per_axis = int(np.sqrt(n_nodes))
    xs = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    ys = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_nodes_per_axis ** 2, 1))
    y_mesh = torch.reshape(y_mesh, (n_nodes_per_axis ** 2, 1))
    mesh_size = 1 / n_nodes_per_axis
    pos_mesh = torch.zeros((n_nodes, 2), device=device)
    pos_mesh[0:n_nodes, 0:1] = x_mesh[0:n_nodes]
    pos_mesh[0:n_nodes, 1:2] = y_mesh[0:n_nodes]

    i0 = imread(f'graphs_data/{node_value_map}')
    if len(i0.shape) == 2:
        # i0 = i0[0,:, :]
        i0 = np.flipud(i0)
        values = i0[(to_numpy(pos_mesh[:, 1]) * 255).astype(int), (to_numpy(pos_mesh[:, 0]) * 255).astype(int)]

    mask_mesh = (x_mesh > torch.min(x_mesh) + 0.02) & (x_mesh < torch.max(x_mesh) - 0.02) & (y_mesh > torch.min(y_mesh) + 0.02) & (y_mesh < torch.max(y_mesh) - 0.02)

    if 'grid' not in field_grid:
        if 'pattern_Null.tif' in simulation_config.node_value_map:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 24
        else:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 8

    match config.graph_model.mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            features_mesh = torch.zeros((n_nodes, 2), device=device)
            features_mesh[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            features_mesh[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case 'RD_FitzHugh_Nagumo_Mesh':
            features_mesh = torch.zeros((n_nodes, 2), device=device) + torch.rand((n_nodes, 2), device=device) * 0.1
        case 'RD_RPS_Mesh' | 'RD_RPS_Mesh_bis':
            features_mesh = torch.rand((n_nodes, 3), device=device)
            s = torch.sum(features_mesh, dim=1)
            for k in range(3):
                features_mesh[:, k] = features_mesh[:, k] / s
        case 'DiffMesh' | 'WaveMesh' | 'Particle_Mesh_A' | 'Particle_Mesh_B' | 'WaveSmoothParticle':
            features_mesh = torch.zeros((n_nodes, 2), device=device)
            features_mesh[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case 'PDE_O_Mesh':
            features_mesh = torch.zeros((n_particles, 5), device=device)
            features_mesh[0:n_particles, 0:1] = x_mesh[0:n_particles]
            features_mesh[0:n_particles, 1:2] = y_mesh[0:n_particles]
            features_mesh[0:n_particles, 2:3] = torch.randn(n_particles, 1, device=device) * 2 * np.pi  # theta
            features_mesh[0:n_particles, 3:4] = torch.ones(n_particles, 1, device=device) * np.pi / 200  # d_theta
            features_mesh[0:n_particles, 4:5] = features_mesh[0:n_particles, 3:4]  # d_theta0
            pos_mesh[:, 0] = features_mesh[:, 0] + (3 / 8) * mesh_size * torch.cos(features_mesh[:, 2])
            pos_mesh[:, 1] = features_mesh[:, 1] + (3 / 8) * mesh_size * torch.sin(features_mesh[:, 2])
        case '' :
            features_mesh = torch.zeros((n_nodes, 2), device=device)

    # i0 = imread(f'graphs_data/{node_type_map}')
    # values = i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(y_mesh[:, 0]) * 255).astype(int)]
    # if np.max(values) > 0:
    #     values = np.round(values / np.max(values) * (simulation_config.n_node_types-1))
    # type_mesh = torch.tensor(values, device=device)
    # type_mesh = type_mesh[:, None]

    type_mesh = torch.zeros((n_nodes, 1), device=device)

    node_id_mesh = torch.arange(n_nodes, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((n_nodes, 2), device=device)

    x_mesh = torch.concatenate((node_id_mesh.clone().detach(), pos_mesh.clone().detach(), dpos_mesh.clone().detach(),
                                type_mesh.clone().detach(), features_mesh.clone().detach()), 1)

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    print('removal of skinny faces ...')
    sleep(0.5)
    for k in trange(face.shape[0]):
        # compute edge distances
        x1 = pos[face[k, 0], :]
        x2 = pos[face[k, 1], :]
        x3 = pos[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
    edge_index_mesh, edge_weight_mesh = get_mesh_laplacian(pos=pos_3d, face=face, normalization="None")
    edge_weight_mesh = edge_weight_mesh.to(dtype=torch.float32)
    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}

    if (config.graph_model.particle_model_name == 'PDE_ParticleField_A')  | (config.graph_model.particle_model_name == 'PDE_ParticleField_B'):
        type_mesh = 0 * type_mesh

    a_mesh = torch.zeros_like(type_mesh)
    type_mesh = type_mesh.to(dtype=torch.float32)

    if 'Smooth' in config.graph_model.mesh_model_name:
        distance = torch.sum((pos_mesh[:, None, :] - pos_mesh[None, :, :]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= 0)).float() * 1
        mesh_data['edge_index'] = adj_t.nonzero().t().contiguous()


    return pos_mesh, dpos_mesh, type_mesh, features_mesh, a_mesh, node_id_mesh, mesh_data


def init_synapse_map(config, x, edge_attr_adjacency, device):

    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
    G = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  # NOT IMPLEMENTED

        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,

        # Log
        verbose=True)

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=500)
    positions = np.array(list(positions.values()))
    X1 = torch.tensor(positions, dtype=torch.float32, device=device)
    X1 = X1 - torch.mean(X1, 0)

    torch.save(X1, f'./graphs_data/graphs_{dataset_name}/X1.pt')

    x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                           H1.clone().detach(), A1.clone().detach()), 1)

    # pos = nx.spring_layout(G, weight='weight', seed=42, k=1)
    # for k,p in pos.items():
    #     X1[k,:] = torch.tensor([p[0],p[1]], device=device)
    
    
def init_adjacency(connectivity_file, connectivity_distribution, connectivity_filling_factor, T1, n_particles, n_particle_types, dataset_name, device):

    if 'adjacency.pt' in connectivity_file:
        adjacency = torch.load(connectivity_file, map_location=device)

    elif 'mat' in connectivity_file:
        mat = scipy.io.loadmat(connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)

    elif 'zarr' in connectivity_file:
        print('loading zarr ...')
        dataset = xr.open_zarr(connectivity_file)
        trained_weights = dataset["trained"]  # alpha * sign * N
        print(f'weights {trained_weights.shape}')
        untrained_weights = dataset["untrained"]  # sign * N
        values = trained_weights[0:n_particles,0:n_particles]
        values = np.array(values)
        values = values / np.max(values)
        adjacency = torch.tensor(values, dtype=torch.float32, device=device)
        values=[]

    elif 'tif' in connectivity_file:
        adjacency = constructRandomMatrices(n_neurons=n_particles, density=1.0, connectivity_mask=f"./graphs_data/{connectivity_file}" ,device=device)
        n_particles = adjacency.shape[0]
        config.simulation.n_particles = n_particles

    elif 'values' in connectivity_file:
        parts = connectivity_file.split('_')
        w01 = float(parts[-2])
        w10 = float(parts[-1])
        adjacency =[[0, w01], [w10, 0]]
        adjacency = np.array(adjacency)
        adjacency = torch.tensor(adjacency, dtype=torch.float32, device=device)

    else:

        if 'Gaussian' in connectivity_distribution:
            adjacency = torch.randn((n_particles, n_particles), dtype=torch.float32, device=device)
            adjacency = adjacency / np.sqrt(n_particles)
            print(f"Gaussian   1/sqrt(N)  {1/np.sqrt(n_particles)}    std {torch.std(adjacency.flatten())}")

        elif 'Lorentz' in connectivity_distribution:

            s = np.random.standard_cauchy(n_particles**2)
            s[(s < -25) | (s > 25)] = 0

            if n_particles < 2000:
                s = s / n_particles**0.7
            elif n_particles <4000:
                s = s / n_particles**0.675
            elif n_particles < 8000:
                s = s / n_particles**0.67
            elif n_particles == 8000:
                s = s / n_particles**0.66
            elif n_particles > 8000:
                s = s / n_particles**0.5
            print(f"Lorentz   1/sqrt(N)  {1/np.sqrt(n_particles):0.3f}    std {np.std(s):0.3f}")

            adjacency = torch.tensor(s, dtype=torch.float32, device=device)
            adjacency = torch.reshape(adjacency, (n_particles, n_particles))

        elif 'uniform' in connectivity_distribution:
            adjacency = torch.rand((n_particles, n_particles), dtype=torch.float32, device=device)
            adjacency = adjacency - 0.5

        i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
        adjacency[i, i] = 0

    if connectivity_filling_factor != 1:
        mask = torch.rand(adjacency.shape) >  connectivity_filling_factor
        adjacency[mask] = 0
        mask = (adjacency != 0).float()
        # edge_index_, edge_attr_ = dense_to_sparse(adjacency)
        if n_particles>10000:
            edge_index = large_tensor_nonzero(mask)
            print (f'edge_index {edge_index.shape}')
        else:
            edge_index = mask.nonzero().t().contiguous()

    else:
        adj_matrix = torch.ones((n_particles)) - torch.eye(n_particles)
        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        mask = (adj_matrix != 0).float()

    if 'structured' in connectivity_distribution:
        parts = connectivity_distribution.split('_')
        float_value1 = float(parts[-2])  # repartition pos/neg
        float_value2 = float(parts[-1])  # filling factor

        matrix_sign = torch.tensor(stats.bernoulli(float_value1).rvs(n_particle_types ** 2) * 2 - 1,
                                   dtype=torch.float32, device=device)
        matrix_sign = matrix_sign.reshape(n_particle_types, n_particle_types)

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_0.png', dpi=300)
        plt.close()

        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        adjacency = torch.abs(adjacency)
        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        sign_matrix = matrix_sign[xy_grid[..., 0], xy_grid[..., 1]]
        adjacency *= sign_matrix

        plt.imshow(to_numpy(sign_matrix))
        plt.savefig(f"graphs_data/{dataset_name}/large_connectivity_sign.tif", dpi=130)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_1.png', dpi=300)
        plt.close()

        flat_sign_matrix = sign_matrix.flatten()
        num_elements = len(flat_sign_matrix)
        num_ones = int(num_elements * float_value2)
        indices = np.random.choice(num_elements, num_ones, replace=False)
        flat_sign_matrix[:] = 0
        flat_sign_matrix[indices] = 1
        sign_matrix = flat_sign_matrix.reshape(sign_matrix.shape)

        adjacency *= sign_matrix

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_2.png', dpi=300)
        plt.close()

    edge_index = edge_index.to(device=device)

    return edge_index, adjacency, mask



