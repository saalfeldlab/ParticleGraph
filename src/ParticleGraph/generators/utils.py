from ParticleGraph.generators import *
from ParticleGraph.utils import *
from time import sleep
from scipy.spatial import Delaunay
from tifffile import imread, imwrite as imsave
from torch_geometric.utils import get_mesh_laplacian
from tqdm import trange
from torch_geometric.utils import dense_to_sparse
from scipy import stats
import seaborn as sns
from scipy.spatial import cKDTree
import subprocess
import os
import glob
import importlib.util
import re
from ParticleGraph.models import PDE_Diffusiophoresis


def load_pde_variant(variant_name, generators_path=None):
    """
    Dynamically load a PDE variant class from a file.

    Parameters
    ----------
    variant_name : str
        Name like 'Diffusiophoresis_Mesh_1' or 'Diffusiophoresis_Mesh_GrayScott'
    generators_path : str, optional
        Path to generators directory. If None, uses default location.

    Returns
    -------
    class
        The PDE class, or None if not found
    """
    if generators_path is None:
        generators_path = os.path.dirname(os.path.abspath(__file__))

    # Extract variant suffix: 'Diffusiophoresis_Mesh_1' -> '1', 'Diffusiophoresis_Mesh_GrayScott' -> 'GrayScott', 'PDE_Diffusiophoresis_GrayScott' -> 'GrayScott'
    match = re.match(r'Diffusiophoresis_Mesh_(.+)', variant_name)
    if not match:
        match = re.match(r'PDE_Diffusiophoresis_(.+)', variant_name)
    if not match:
        return None

    variant_suffix = match.group(1)

    # Look for file: PDE_Diffusiophoresis_1.py, PDE_Diffusiophoresis_GrayScott.py, etc.
    file_name = f"PDE_Diffusiophoresis_{variant_suffix}.py"
    file_path = os.path.join(generators_path, file_name)

    if not os.path.exists(file_path):
        print(f"Warning: PDE variant file not found: {file_path}")
        return None

    # Use standard import machinery — required for PyG's JIT propagate compilation.
    module_name = f"ParticleGraph.generators.PDE_Diffusiophoresis_{variant_suffix}"
    import sys
    import importlib
    importlib.invalidate_caches()

    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    # Get the PDE class (expected name: PDE_Diffusiophoresis_{suffix})
    class_name = f"PDE_Diffusiophoresis_{variant_suffix}"
    if hasattr(module, class_name):
        return getattr(module, class_name)

    # Fallback: look for any class starting with PDE_Diffusiophoresis
    for name in dir(module):
        if name.startswith('PDE_Diffusiophoresis'):
            return getattr(module, name)

    print(f"Warning: No PDE class found in {file_path}")
    return None


def load_pde_d_variant(variant_name, generators_path=None):
    """
    Dynamically load a PDE_D variant class from a file.

    Parameters
    ----------
    variant_name : str
        Name like 'PDE_ParticleField_D_Boids' or 'PDE_D_Chemotaxis'
    generators_path : str, optional
        Path to generators directory. If None, uses default location.

    Returns
    -------
    class
        The PDE_D variant class, or None if not found
    """
    if generators_path is None:
        generators_path = os.path.dirname(os.path.abspath(__file__))

    # Extract variant suffix from various naming patterns:
    # 'PDE_ParticleField_D_Boids' -> 'Boids'
    # 'PDE_Cell_D_Boids' -> 'Boids'
    # 'PDE_D_Boids' -> 'Boids'
    for pattern in [r'PDE_ParticleField_D_(.+)', r'PDE_Cell_D_(.+)', r'PDE_D_(.+)']:
        match = re.match(pattern, variant_name)
        if match:
            variant_suffix = match.group(1)
            break
    else:
        return None

    # Look for file: PDE_D_Boids.py, etc.
    file_name = f"PDE_D_{variant_suffix}.py"
    file_path = os.path.join(generators_path, file_name)

    if not os.path.exists(file_path):
        print(f"Warning: PDE_D variant file not found: {file_path}")
        return None

    # Use standard import machinery — required for PyG's JIT propagate compilation.
    # spec_from_file_location creates an incomplete module spec that breaks PyG's
    # Jinja-based propagate module generation (AttributeError on _propagate).
    module_name = f"ParticleGraph.generators.PDE_D_{variant_suffix}"
    import sys
    import importlib
    importlib.invalidate_caches()  # Handle recently-created files

    if module_name in sys.modules:
        # Reload to pick up any modifications (e.g., LLM-edited files)
        module = importlib.reload(sys.modules[module_name])
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # Fallback: file-based import for files outside standard package structure
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    # Get the PDE_D class (expected name: PDE_D_{suffix})
    class_name = f"PDE_D_{variant_suffix}"
    if hasattr(module, class_name):
        print(f"loaded PDE_D variant: {class_name} from {file_name}")
        return getattr(module, class_name)

    # Fallback: look for any class starting with PDE_D_
    for name in dir(module):
        if name.startswith('PDE_D_') and not name.startswith('PDE_D__'):
            print(f"loaded PDE_D variant: {name} from {file_name}")
            return getattr(module, name)

    print(f"Warning: No PDE_D class found in {file_path}")
    return None


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
        case _ if particle_model_name.startswith('PDE_D_') or particle_model_name.startswith('PDE_ParticleField_D_') or particle_model_name.startswith('PDE_Cell_D_'):
            # PDE_D variant (e.g., PDE_D_Boids, PDE_ParticleField_D_Chemotaxis)
            params_mesh = config.simulation.params_mesh
            p_mesh = torch.tensor(params_mesh, dtype=torch.float32, device=device).squeeze()
            if params is not None and params[0] != [-1] and len(params[0]) >= 4:
                particle_params = torch.tensor(params, dtype=torch.float32, device=device)
            else:
                particle_params = None
            sigma = config.simulation.sigma
            pde_d_class = load_pde_d_variant(particle_model_name)
            if pde_d_class is not None:
                model = pde_d_class(aggr_type=aggr_type, p=p_mesh, particle_params=particle_params,
                                    bc_dpos=bc_dpos, dimension=dimension, sigma=sigma)
            else:
                raise ValueError(f"Failed to load PDE_D variant: {particle_model_name}")
        case 'PDE_D' | 'PDE_ParticleField_D' | 'PDE_Cell_D' :
            params_mesh = config.simulation.params_mesh
            p_mesh = torch.tensor(params_mesh, dtype=torch.float32, device=device).squeeze()
            # Per-type particle params from simulation.params (like PDE_A)
            # Layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
            # Block 4 code change: Enable attraction-repulsion even with n_particle_types=1
            # Previously required n_particle_types > 1, now any valid params activates it
            if params is not None and params[0] != [-1] and len(params[0]) >= 8:
                particle_params = torch.tensor(params, dtype=torch.float32, device=device)
            else:
                particle_params = None
            sigma = config.simulation.sigma
            model = PDE_D(aggr_type=aggr_type, p=p_mesh, particle_params=particle_params,
                          bc_dpos=bc_dpos, dimension=dimension, sigma=sigma)
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

        case 'PDE_F' |'PDE_F_A' | 'PDE_F_B' :
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
            i0 = np.flipud(i0)
            values = i0[(to_numpy(X1_mesh[:, 1]) * 255).astype(int), (to_numpy(X1_mesh[:, 0]) * 255).astype(int)]
            values = np.reshape(values,len(X1_mesh))
            values = torch.tensor(values, device=device, dtype=torch.float32)[:, None]
        else:
            values = torch.ones((X1_mesh.shape[0],1), device=device)


        match mesh_model_name:
            case 'RD_Gray_Scott_Mesh':
                mesh_model = RD_Gray_Scott(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'RD_FitzHugh_Nagumo_Mesh':
                mesh_model = RD_FitzHugh_Nagumo(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'RD_Mesh':
                mesh_model = RD_RPS(aggr_type=aggr_type, bc_dpos=bc_dpos, coeff=values)
            case 'Diffusiophoresis_Mesh':
                params_mesh = config.simulation.params_mesh
                p = torch.tensor(params_mesh, dtype=torch.float32, device=device).squeeze()
                mesh_model = PDE_Diffusiophoresis(aggr_type=aggr_type, bc_dpos=bc_dpos, p=p)
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
                # Try dynamic loading for PDE variants (e.g., Diffusiophoresis_Mesh_1, Diffusiophoresis_Mesh_GrayScott, PDE_Diffusiophoresis_GrayScott)
                if mesh_model_name.startswith('Diffusiophoresis_Mesh_') or mesh_model_name.startswith('PDE_Diffusiophoresis_'):
                    pde_class = load_pde_variant(mesh_model_name)
                    if pde_class is not None:
                        params_mesh = config.simulation.params_mesh
                        p = torch.tensor(params_mesh, dtype=torch.float32, device=device).squeeze()
                        mesh_model = pde_class(aggr_type=aggr_type, bc_dpos=bc_dpos, p=p)
                        print(f"Loaded PDE variant: {mesh_model_name}")
                    else:
                        raise ValueError(f"Failed to load PDE variant: {mesh_model_name}")
                else:
                    mesh_model = PDE_Z(device=device)




    return mesh_model


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)


def init_particles(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    model_config = config.graph_model
    n_frames = config.simulation.n_frames
    n_particles = simulation_config.n_particles * ratio
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if 'PDE_F' in config.graph_model.particle_model_name:
        pos = torch.rand(n_particles, dimension, device=device)
        if simulation_config.pos_init == 'square':
            pos = pos * 0.5 + 0.25
    elif "diffusiophoresis" in model_config.field_type:
        if simulation_config.pos_init == 'random':
            print('random particles across full domain [0.05, 0.95]')
            pos = torch.rand(n_particles, dimension, device=device) * 0.9 + 0.05
        else:
            print('equidistant particles at center [0.5, 0.5] with radius 0.25')
            xc, yc = get_equidistant_points(n_points=n_particles)
            pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device)
            pos = pos * 0.25 + 0.5  # Scale to radius 0.25, center at [0.5, 0.5]
    elif (simulation_config.boundary == 'periodic') | ('PDE_K' in config.graph_model.particle_model_name):
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

    if simulation_config.bounce:
        n_wall_particles = n_particles // n_particle_types
        n_particles_wall = n_wall_particles // 4
        wall_pos = torch.linspace(0.1, 0.9, n_particles_wall, device=device)
        wall0 = torch.zeros(n_particles_wall, 2, device=device)
        wall0[:,0] = wall_pos
        wall0[:,1] = 0.1
        wall1 = torch.zeros(n_particles_wall, 2, device=device)
        wall1[:,0] = wall_pos
        wall1[:,1] = 0.9
        wall2 = torch.zeros(n_particles_wall, 2, device=device)
        wall2[:,0] = 0.1
        wall2[:,1] = wall_pos
        wall3 = torch.zeros(n_particles_wall, 2, device=device)
        wall3[:,0] = 0.9
        wall3[:,1] = wall_pos
        pos_ = torch.cat((wall0,wall1,wall2,wall3), dim=0)
        pos_ = pos_ + torch.randn((n_wall_particles,dimension), device=device) * 0.001

        dpos [0:n_wall_particles] = 0
        pos [0:n_wall_particles:] = pos_

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


def init_neurons(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    n_frames = config.simulation.n_frames
    n_neurons = simulation_config.n_neurons * ratio
    n_neuron_types = simulation_config.n_neuron_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init


    xc, yc = get_equidistant_points(n_points=n_neurons)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0))
    pos = pos[perm]

    dpos = dpos_init * torch.randn((n_neurons, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))

    type = torch.zeros(int(n_neurons / n_neuron_types), device=device)

    for n in range(1, n_neuron_types):
        type = torch.cat((type, n * torch.ones(int(n_neurons / n_neuron_types), device=device)), 0)
    if type.shape[0] < n_neurons:
        type = torch.cat((type, n * torch.ones(n_neurons - type.shape[0], device=device)), 0)

    if (config.graph_model.signal_model_name == 'PDE_N6') | (config.graph_model.signal_model_name == 'PDE_N7'):
        features = torch.cat((torch.rand((n_neurons, 1), device=device), 0.1 * torch.randn((n_neurons, 1), device=device),
                              torch.ones((n_neurons, 1), device=device), torch.zeros((n_neurons, 1), device=device)), 1)
    elif 'excitation_single' in config.graph_model.field_type:
        features = torch.zeros((n_neurons, 2), device=device)
    else:
        features = torch.cat((torch.randn((n_neurons, 1), device=device) * 5 , 0.1 * torch.randn((n_neurons, 1), device=device)), 1)

    type = type[:, None]
    particle_id = torch.arange(n_neurons, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_neurons,1), device=device)

    return pos, dpos, type, features, age, particle_id


def random_rotation_matrix(device='cpu'):
    # Random Euler angles
    roll = torch.rand(1, device=device) * 2 * torch.pi
    pitch = torch.rand(1, device=device) * 2 * torch.pi
    yaw = torch.rand(1, device=device) * 2 * torch.pi

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices around each axis
    R_x = torch.tensor([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ], device=device).squeeze()

    R_y = torch.tensor([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], device=device).squeeze()

    R_z = torch.tensor([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ], device=device).squeeze()

    # Combined rotation matrix: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R


def stratified_sphere_points(n_points, radius=1.0, device='cpu'):
    # Estimate number of shells (radial layers)
    n_shells = int(torch.ceil(torch.tensor(n_points ** (1/3))).item())
    points = []

    total_points = 0
    for i in range(n_shells):
        r_lower = i / n_shells
        r_upper = (i + 1) / n_shells
        r_mean = (r_lower + r_upper) / 2

        # Fraction of points proportional to shell volume
        shell_volume = r_upper**3 - r_lower**3
        n_shell_points = int(shell_volume * n_points)

        if n_shell_points == 0:
            continue

        # Stratified indices within shell
        indices = torch.arange(n_shell_points, dtype=torch.float32, device=device) + 0.5

        # Spherical coordinates for points uniformly distributed on shell surface
        phi = torch.acos(1 - 2 * indices / n_shell_points)  # polar angle [0, pi]
        theta = 2 * torch.pi * indices * ((1 + 5 ** 0.5) / 2)  # golden angle for good azimuthal spacing

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        shell_points = torch.stack([x, y, z], dim=1) * (r_mean * radius)
        points.append(shell_points)

        total_points += n_shell_points

    # If not enough points generated due to rounding, fill with random points inside the sphere
    if total_points < n_points:
        remaining = n_points - total_points

        u = torch.rand(remaining, device=device)
        r = radius * u.pow(1/3)  # Correct radius distribution for uniform volume density

        phi = torch.acos(1 - 2 * torch.rand(remaining, device=device))
        theta = 2 * torch.pi * torch.rand(remaining, device=device)

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        random_points = torch.stack([x, y, z], dim=1) * r.unsqueeze(1)
        points.append(random_points)

    all_points = torch.cat(points, dim=0)
    return all_points[:n_points]


def get_equidistant_3D_points(n_points=1024):
    """
    Generate equidistant points within a unit sphere using improved 3D distribution.

    Args:
        n_points: Number of points to generate

    Returns:
        x, y, z: Arrays of coordinates for points within unit sphere
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5

    # Radial distribution for uniform density in sphere volume
    # Use cube root for 3D volume distribution
    r = np.cbrt(indices / n_points)

    # Use Fibonacci spiral for uniform surface distribution
    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices

    # For uniform distribution on sphere surface (not clustered at poles)
    # y should be uniform in [-1, 1], not cos(phi)
    y = 1 - 2 * indices / n_points

    # Calculate radius in xy-plane
    radius_xy = np.sqrt(1 - y * y)

    # Convert to Cartesian coordinates
    x = radius_xy * np.cos(theta) * r
    y = y * r
    z = radius_xy * np.sin(theta) * r

    return x, y, z


def handle_collisions(positions, velocities, min_distance=0.01):
    """
    Prevent particle overlap by implementing soft repulsion
    
    Parameters
    ----------
    positions : torch.Tensor
        Particle positions [n_particles, dimension]
    velocities : torch.Tensor
        Particle velocities [n_particles, dimension]
    min_distance : float
        Minimum allowed distance between particles
        
    Returns
    -------
    torch.Tensor
        Adjusted positions
    torch.Tensor
        Adjusted velocities
    """
    n_particles = positions.shape[0]
    device = positions.device
    dimension = positions.shape[1]
    
    # Use a grid-based approach for efficiency with many particles
    # For 9600 particles, checking all pairs would be very slow
    
    # Compute grid cell size based on min_distance
    cell_size = min_distance * 2
    grid_size = int(1.0 / cell_size) + 1
    
    # Initialize grid
    grid = {}
    
    # Assign particles to grid cells
    for i in range(n_particles):
        # Get grid indices for this particle
        cell_x = int(positions[i, 0] / cell_size)
        cell_y = int(positions[i, 1] / cell_size)
        
        # Periodic boundary for cells
        cell_x = cell_x % grid_size
        cell_y = cell_y % grid_size
        
        # Add particle to grid
        cell_idx = (cell_x, cell_y)
        if cell_idx not in grid:
            grid[cell_idx] = []
        grid[cell_idx].append(i)
    
    # Check collisions only with neighboring cells
    displacements = torch.zeros_like(positions)
    
    for cell_idx, particles in grid.items():
        cell_x, cell_y = cell_idx
        
        # Check neighboring cells (including self)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Get neighboring cell with periodic boundary
                neighbor_x = (cell_x + dx) % grid_size
                neighbor_y = (cell_y + dy) % grid_size
                
                neighbor_idx = (neighbor_x, neighbor_y)
                if neighbor_idx in grid:
                    # Check collisions with particles in this neighboring cell
                    for i in particles:
                        for j in grid[neighbor_idx]:
                            if i != j:  # Don't check against self
                                # Compute displacement with periodic boundary
                                d_pos = positions[j] - positions[i]
                                
                                # Apply periodic boundary conditions
                                for d in range(dimension):
                                    if d_pos[d] > 0.5:
                                        d_pos[d] -= 1.0
                                    elif d_pos[d] < -0.5:
                                        d_pos[d] += 1.0
                                
                                # Compute squared distance
                                dist_sq = torch.sum(d_pos**2)
                                
                                # Check if particles are too close
                                if dist_sq < min_distance**2:
                                    # Compute actual distance
                                    dist = torch.sqrt(dist_sq)
                                    
                                    # Compute normalized direction
                                    if dist > 1e-6:  # Avoid division by zero
                                        d_pos_norm = d_pos / dist
                                    else:
                                        # If particles are exactly overlapping, use a random direction
                                        d_pos_norm = torch.randn(dimension, device=device)
                                        d_pos_norm = d_pos_norm / torch.norm(d_pos_norm)
                                    
                                    # Compute overlap
                                    overlap = min_distance - dist
                                    
                                    # Compute displacement to resolve collision
                                    # Each particle moves half the overlap distance
                                    displacement = 0.5 * overlap * d_pos_norm
                                    
                                    # Accumulate displacements
                                    displacements[i] -= displacement
                                    displacements[j] += displacement
    
    # Apply accumulated displacements
    positions += displacements
    
    # Adjust velocities based on collisions
    # Particles moving toward each other should slow down
    for i in range(n_particles):
        if torch.norm(displacements[i]) > 1e-6:
            v_parallel = torch.dot(velocities[i], displacements[i]) / torch.norm(displacements[i])
            if v_parallel < 0:  # Moving toward collision
                # Reduce component of velocity in collision direction
                v_dir = displacements[i] / torch.norm(displacements[i])
                velocities[i] -= 0.5 * v_parallel * v_dir
    
    # Ensure positions stay within bounds [0, 1]
    positions = torch.clamp(positions, min=0.0, max=1.0)
    
    return positions, velocities


def find_neighbors_with_radius(pos, h, max_neighbors=32):
    """Find neighbors within radius h using scipy KDTree"""
    device = pos.device
    n_particles = pos.shape[0]

    # Convert to numpy for scipy
    pos_np = pos.cpu().numpy()

    # Build KDTree
    tree = cKDTree(pos_np)

    # Find neighbors for each particle
    neighbor_lists = []
    for i in range(n_particles):
        # Query ball returns indices within radius
        neighbors = tree.query_ball_point(pos_np[i], r=h)
        # Remove self and limit to max_neighbors
        neighbors = [n for n in neighbors if n != i]
        if len(neighbors) > max_neighbors:
            # Keep closest neighbors
            dists = np.linalg.norm(pos_np[neighbors] - pos_np[i], axis=1)
            sorted_idx = np.argsort(dists)[:max_neighbors]
            neighbors = [neighbors[idx] for idx in sorted_idx]
        neighbor_lists.append(neighbors)

    return neighbor_lists


def MLS_gradient_velocity(query_pos, neighbor_pos, neighbor_vel, h):
    """
    MLS reconstruction following Müller 2004
    Returns velocity and velocity gradient (C matrix) at query position
    """
    device = query_pos.device
    n_neighbors = neighbor_pos.shape[0]

    if n_neighbors < 4:
        # Insufficient neighbors - return zero gradient
        return torch.zeros(2, device=device), torch.zeros(2, 2, device=device)

    # Relative positions
    dx = neighbor_pos - query_pos.unsqueeze(0)  # [n_neighbors, 2]
    r = torch.norm(dx, dim=1)  # [n_neighbors]

    # Müller 2004 kernel weights
    h_sq = h * h
    r_sq = r * r
    valid = r < h
    weights = torch.zeros_like(r)
    weights[valid] = (315.0 / (64.0 * np.pi * h ** 9)) * (h_sq - r_sq[valid]) ** 3

    if torch.sum(valid) < 4:
        return torch.zeros(2, device=device), torch.zeros(2, 2, device=device)

    # Polynomial basis P(x) = [1, x, y] for linear MLS
    P = torch.cat([
        torch.ones(n_neighbors, 1, device=device),  # 1
        dx  # [x, y]
    ], dim=1)  # [n_neighbors, 3]

    # Weight matrix Ξ
    Xi = torch.diag(weights)  # [n_neighbors, n_neighbors]

    # Moment matrix M = P^T Ξ P
    M = P.T @ Xi @ P  # [3, 3]

    # Check conditioning and use SVD if needed
    try:
        cond_num = torch.linalg.cond(M)
        if cond_num > 1e12 or torch.isnan(cond_num):
            # Use SVD for robust inversion
            U, S, Vh = torch.linalg.svd(M)
            S_inv = torch.where(S > 1e-15, 1.0 / S, 0.0)
            M_inv = (Vh.T * S_inv.unsqueeze(0)) @ Vh
        else:
            M_inv = torch.linalg.inv(M)
    except:
        # Fallback to SVD
        U, S, Vh = torch.linalg.svd(M)
        S_inv = torch.where(S > 1e-15, 1.0 / S, 0.0)
        M_inv = (Vh.T * S_inv.unsqueeze(0)) @ Vh

    # Reconstruct velocity components
    vel_x = neighbor_vel[:, 0]  # [n_neighbors]
    vel_y = neighbor_vel[:, 1]  # [n_neighbors]

    # MLS coefficients: c = M^(-1) P^T Ξ u
    coeffs_x = M_inv @ P.T @ Xi @ vel_x  # [3]
    coeffs_y = M_inv @ P.T @ Xi @ vel_y  # [3]

    # Reconstructed velocity at query point: [c0, c1, c2] for v = c0 + c1*x + c2*y
    # At query point (x=0, y=0 in local coords), velocity = c0
    velocity = torch.stack([coeffs_x[0], coeffs_y[0]])  # [2]

    # Velocity gradient (C matrix): ∂v/∂x = [c1_x, c2_x; c1_y, c2_y]
    C_matrix = torch.stack([
        torch.stack([coeffs_x[1], coeffs_x[2]]),  # [∂vx/∂x, ∂vx/∂y]
        torch.stack([coeffs_y[1], coeffs_y[2]])  # [∂vy/∂x, ∂vy/∂y]
    ])  # [2, 2]

    return velocity, C_matrix


def MLS_C(features, h=0.0125, max_neighbors=32):
    """
    Main MLS function compatible with SIREN interface
    Input: features = torch.cat((pos, velocity, frame), dim=1)
    Output: C_mls.reshape(-1, 2, 2)
    """
    device = features.device
    n_particles = features.shape[0]

    pos = features[:, :2]  # [n_particles, 2]
    velocity = features[:, 2:4]  # [n_particles, 2]
    neighbor_lists = find_neighbors_with_radius(pos, h, max_neighbors)

    # Initialize output
    C_mls = torch.zeros(n_particles, 2, 2, device=device)

    # Statistics tracking
    neighbor_counts = torch.tensor([len(neighbors) + 1 for neighbors in neighbor_lists])  # +1 for self
    svd_count = 0
    insufficient_count = 0

    # Process each particle
    for i in range(n_particles):
        # Get neighbors (including self)
        neighbor_indices = neighbor_lists[i] + [i]  # Add self to neighbors

        if len(neighbor_indices) < 4:
            insufficient_count += 1
            continue

        # Convert to tensor
        neighbor_idx_tensor = torch.tensor(neighbor_indices, device=device)
        neighbor_pos = pos[neighbor_idx_tensor]
        neighbor_vel = velocity[neighbor_idx_tensor]

        # MLS reconstruction
        _, C_matrix = MLS_gradient_velocity(
            pos[i], neighbor_pos, neighbor_vel, h
        )
        C_mls[i] = C_matrix

    # Print statistics
    # print(f"MLS Statistics (h={h:.4f}):")
    print(f"  Neighbors: min={neighbor_counts.min()}, max={neighbor_counts.max()}, "
          f"mean={neighbor_counts.float().mean():.1f}, std={neighbor_counts.float().std():.1f}")
    # print(f"  <4 neighbors: {insufficient_count}/{n_particles} "
    #       f"({100 * insufficient_count / n_particles:.1f}%)")
    # print(f"  4-15 neighbors: {torch.sum((neighbor_counts >= 4) & (neighbor_counts <= 15)).item()}/{n_particles} "
    #       f"({100 * torch.sum((neighbor_counts >= 4) & (neighbor_counts <= 15)) / n_particles:.1f}%)")
    # print(f"  >20 neighbors: {torch.sum(neighbor_counts > 20).item()}/{n_particles} "
    #       f"({100 * torch.sum(neighbor_counts > 20) / n_particles:.1f}%)")

    return C_mls.reshape(-1, 2, 2)



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

    if 'grid' in field_grid:
        pos_mesh = pos_mesh
    else:
        if 'pattern_Null.tif' in simulation_config.node_value_map:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 24
        else:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 8

    if "diffusiophoresis" in model_config.field_type:
        pos_mesh = pos_mesh * 1.0

    match config.graph_model.mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            node_value = torch.zeros((n_nodes, 2), device=device)
            node_value[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            node_value[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case 'RD_FitzHugh_Nagumo_Mesh':
            node_value = torch.zeros((n_nodes, 2), device=device) + torch.rand((n_nodes, 2), device=device) * 0.1
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' :
            node_value = torch.rand((n_nodes, 3), device=device)
            s = torch.sum(node_value, dim=1)
            for k in range(3):
                node_value[:, k] = node_value[:, k] / s
        case 'Diffusiophoresis_Mesh':
            node_value = torch.rand((n_nodes, 2), device=device)
            s = torch.sum(node_value, dim=1)
            for k in range(2):
                node_value[:, k] = node_value[:, k] / s
        case _ if config.graph_model.mesh_model_name.startswith('Diffusiophoresis_Mesh_') or config.graph_model.mesh_model_name.startswith('PDE_Diffusiophoresis'):
            # PDE variants (Gray-Scott, etc.) - initialize with small random perturbations
            # Gray-Scott: U≈1 (substrate), V≈0 (autocatalyst) with localized seeds
            if 'GrayScott' in config.graph_model.mesh_model_name:
                # Initialize U=1, V=0 everywhere, then add localized seeds of V
                node_value = torch.zeros((n_nodes, 2), device=device)
                node_value[:, 0] = 1.0  # U = 1 (substrate at equilibrium)
                node_value[:, 1] = 0.0  # V = 0 (no autocatalyst initially)
                # Add random localized seeds of autocatalyst
                n_seeds = max(1, n_nodes // 100)  # ~1% of nodes as seeds
                seed_indices = torch.randperm(n_nodes)[:n_seeds]
                node_value[seed_indices, 0] = 0.5  # Deplete U at seed locations
                node_value[seed_indices, 1] = 0.25  # Add V at seed locations
            elif 'Schnakenberg' in config.graph_model.mesh_model_name:
                # Schnakenberg: initialize near steady state u*=a+b, v*=b/(a+b)^2
                # with small random perturbations to break symmetry
                a_param = config.simulation.params_mesh[0][2]
                b_param = config.simulation.params_mesh[0][3]
                u_star = a_param + b_param
                v_star = b_param / (u_star ** 2)
                node_value = torch.zeros((n_nodes, 2), device=device)
                node_value[:, 0] = u_star + 0.01 * torch.randn(n_nodes, device=device)
                node_value[:, 1] = v_star + 0.01 * torch.randn(n_nodes, device=device)
            elif 'GM' in config.graph_model.mesh_model_name:
                # Gierer-Meinhardt: initialize near homogeneous steady state
                # Steady state: a* = sigma_a/mu_a, h* = rho*(sigma_a/mu_a)^2/mu_h
                # with small random perturbations for symmetry breaking
                # params_mesh[0]: [Da, rho, mu_a, sigma_a, kappa, time_scale]
                # params_mesh[1]: [Dh, mu_h, sigma_h, ...]
                rho_param = config.simulation.params_mesh[0][1]
                mu_a_param = config.simulation.params_mesh[0][2]
                sigma_a_param = config.simulation.params_mesh[0][3]
                mu_h_param = config.simulation.params_mesh[1][1]
                a_star = sigma_a_param / max(mu_a_param, 1e-6)
                h_star = rho_param * a_star**2 / max(mu_h_param, 1e-6)
                node_value = torch.zeros((n_nodes, 2), device=device)
                node_value[:, 0] = max(a_star, 0.1) + 0.01 * torch.randn(n_nodes, device=device)
                node_value[:, 1] = max(h_star, 0.1) + 0.01 * torch.randn(n_nodes, device=device)
            elif 'FHN' in config.graph_model.mesh_model_name:
                # FitzHugh-Nagumo: initialize near resting state with localized perturbations
                # FHN resting state: u* ≈ intersection of u-nullcline and v-nullcline
                # For standard params (a=0.75, b=1.0): u* ≈ -1.2, v* ≈ (u*+a)/b ≈ -0.45
                # Use random init near resting state + localized excitation seeds
                # Literature: Ermakova et al. (2009) PLoS ONE 4:e4454
                a_param = config.simulation.params_mesh[0][1]  # a parameter
                b_param = config.simulation.params_mesh[0][2]  # b parameter
                # Approximate resting state (leftmost intersection of nullclines)
                u_rest = -1.2  # Approximate for typical a, b values
                v_rest = (u_rest + a_param) / max(b_param, 0.1)
                node_value = torch.zeros((n_nodes, 2), device=device)
                node_value[:, 0] = u_rest + 0.01 * torch.randn(n_nodes, device=device)  # u near rest
                node_value[:, 1] = v_rest + 0.01 * torch.randn(n_nodes, device=device)  # v near rest
                # Add localized excitation seeds (push u above threshold to trigger waves)
                n_seeds = max(3, n_nodes // 200)  # ~0.5% of nodes as seeds
                seed_indices = torch.randperm(n_nodes)[:n_seeds]
                node_value[seed_indices, 0] = 1.5  # Excited state (well above threshold)
                node_value[seed_indices, 1] = v_rest  # v unchanged at seeds
            else:
                # Default for other variants: normalized random
                node_value = torch.rand((n_nodes, 2), device=device)
                s = torch.sum(node_value, dim=1)
                for k in range(2):
                    node_value[:, k] = node_value[:, k] / s
        case 'DiffMesh' | 'WaveMesh' | 'Particle_Mesh_A' | 'Particle_Mesh_B' | 'WaveSmoothParticle':
            node_value = torch.zeros((n_nodes, 2), device=device)
            node_value[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case 'PDE_O_Mesh':
            node_value = torch.zeros((n_particles, 5), device=device)
            node_value[0:n_particles, 0:1] = x_mesh[0:n_particles]
            node_value[0:n_particles, 1:2] = y_mesh[0:n_particles]
            node_value[0:n_particles, 2:3] = torch.randn(n_particles, 1, device=device) * 2 * np.pi  # theta
            node_value[0:n_particles, 3:4] = torch.ones(n_particles, 1, device=device) * np.pi / 200  # d_theta
            node_value[0:n_particles, 4:5] = node_value[0:n_particles, 3:4]  # d_theta0
            pos_mesh[:, 0] = node_value[:, 0] + (3 / 8) * mesh_size * torch.cos(node_value[:, 2])
            pos_mesh[:, 1] = node_value[:, 1] + (3 / 8) * mesh_size * torch.sin(node_value[:, 2])
        case '' :
            node_value = torch.zeros((n_nodes, 2), device=device)

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
                                type_mesh.clone().detach(), node_value.clone().detach()), 1)

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    # removal of skinny faces
    sleep(0.5)
    for k in range(face.shape[0]):
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

    # Add periodic wrap-around edges when boundary='periodic'
    # This connects left↔right and top↔bottom mesh boundary nodes so the Laplacian
    # treats the domain as a torus, enabling seamless Turing pattern formation.
    if hasattr(config.simulation, 'boundary') and config.simulation.boundary == 'periodic':
        # Estimate typical interior Laplacian edge weight from existing edges
        # Use median of negative weights (off-diagonal Laplacian entries)
        neg_weights = edge_weight_mesh[edge_weight_mesh < 0]
        if len(neg_weights) > 0:
            typical_weight = neg_weights.median().item()
        else:
            typical_weight = -1.0 / (mesh_size ** 2)

        # Node indices on a regular n×n grid: node(i,j) = i * n + j
        # where i is row (y-axis), j is column (x-axis)
        n = n_nodes_per_axis
        periodic_src = []
        periodic_dst = []

        # Left↔Right: connect column 0 to column n-1 (same row)
        for i in range(n):
            left_node = i * n + 0          # column 0
            right_node = i * n + (n - 1)   # column n-1
            periodic_src.extend([left_node, right_node])
            periodic_dst.extend([right_node, left_node])

        # Top↔Bottom: connect row 0 to row n-1 (same column)
        for j in range(n):
            top_node = 0 * n + j            # row 0
            bottom_node = (n - 1) * n + j   # row n-1
            periodic_src.extend([top_node, bottom_node])
            periodic_dst.extend([bottom_node, top_node])

        periodic_edge_index = torch.tensor([periodic_src, periodic_dst], dtype=torch.long, device=device)
        periodic_weights = torch.full((len(periodic_src),), typical_weight, dtype=torch.float32, device=device)

        # Append periodic edges to existing Laplacian
        edge_index_mesh = torch.cat([edge_index_mesh, periodic_edge_index], dim=1)
        edge_weight_mesh = torch.cat([edge_weight_mesh, periodic_weights])

        # CRITICAL: Update diagonal (self-loop) entries to maintain row-sum = 0.
        # Each new off-diagonal edge with weight w (negative) added to node i
        # requires the diagonal entry L(i,i) to be updated by -w (positive).
        # Without this correction, the Laplacian rows don't sum to 0, causing
        # unbounded mass injection and NaN divergence (confirmed iters 13-14).
        diagonal_correction = torch.zeros(n_nodes, dtype=torch.float32, device=device)
        periodic_src_tensor = torch.tensor(periodic_src, dtype=torch.long, device=device)
        diagonal_correction.scatter_add_(0, periodic_src_tensor, -periodic_weights)

        # Find and update existing self-loop weights
        self_loop_mask = edge_index_mesh[0] == edge_index_mesh[1]
        self_loop_nodes = edge_index_mesh[0, self_loop_mask]
        edge_weight_mesh[self_loop_mask] += diagonal_correction[self_loop_nodes]

        print(f"Added {len(periodic_src)} periodic Laplacian edges (weight={typical_weight:.4f})")
        print(f"Updated {self_loop_mask.sum().item()} diagonal entries (max correction={diagonal_correction.abs().max().item():.4f})")

    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}

    if (config.graph_model.particle_model_name == 'PDE_ParticleField_A')  | (config.graph_model.particle_model_name == 'PDE_ParticleField_B'):
        type_mesh = 0 * type_mesh

    a_mesh = torch.zeros_like(type_mesh)
    type_mesh = type_mesh.to(dtype=torch.float32)

    return pos_mesh, dpos_mesh, type_mesh, node_value, a_mesh, node_id_mesh, mesh_data


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
    
    
def init_connectivity(connectivity_file, connectivity_distribution, connectivity_filling_factor, T1, n_particles, n_particle_types, dataset_name, device):

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

import subprocess

def generate_lossless_video_ffv1(output_dir, run=0, framerate=10, output_name="_ffv1.mkv", config_indices=None):
    """
    Generate a truly lossless compressed video using ffmpeg's FFV1 codec.

    Parameters:
        output_dir (str): Path to directory containing Fig/Fig_*.png.
        run (int): Run index to use in filename pattern.
        framerate (int): Desired video framerate.
        output_name (str): Name of output .mkv file.
    """
    fig_dir = os.path.join(output_dir, "Fig")
    input_pattern = os.path.join(fig_dir, f"Fig_{run}_%06d.png")
    output_path = os.path.join(output_dir, f"input_{config_indices}{output_name}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-c:v", "ffv1",
        "-level", "3",
        "-g", "1",  # No GOP (intra-frame only)
        output_path,
    ]

    print(f"Generating lossless video (FFV1): {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Lossless video (FFV1) saved to: {output_path}")

def generate_lossless_video_libx264(output_dir, run=0, framerate=10, output_name="_libx264.mkv", config_indices=None):
    """
    Generate a lossless H.264 video using libx264 from a sequence of PNG images.

    Parameters:
        output_dir (str): Path to directory containing Fig/Fig_*.png.
        run (int): Run index to use in filename pattern.
        framerate (int): Desired video framerate.
        output_name (str): Output video file name (.mkv recommended).
    """
    fig_dir = os.path.join(output_dir, "Fig")
    input_pattern = os.path.join(fig_dir, f"Fig_{run}_%06d.png")
    output_path = os.path.join(output_dir, f"input_{config_indices}{output_name}")

    command = [
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-preset", "veryslow",
        "-crf", "0",  # lossless mode
        "-pix_fmt", "yuv444p",  # preserve full chroma info
        output_path
    ]

    print(f"Generating lossless video (libx264): {' '.join(command)}")
    subprocess.run(command, check=True)
    print(f"Lossless video (libx264) saved to: {output_path}")

def generate_compressed_video_mp4(output_dir, run=0, framerate=10, output_name=".mp4", config_indices=None, crf=23):
    """
    Generate a compressed video using ffmpeg's libx264 codec in MP4 format.
    Automatically handles odd dimensions by scaling to even dimensions.

    Parameters:
        output_dir (str): Path to directory containing Fig/Fig_*.png.
        run (int): Run index to use in filename pattern.
        framerate (int): Desired video framerate.
        output_name (str): Name of output .mp4 file.
        crf (int): Constant Rate Factor for quality (0-51, lower = better quality, 23 is default).
    """
    import os
    import subprocess

    fig_dir = os.path.join(output_dir, "Fig")
    input_pattern = os.path.join(fig_dir, f"Fig_{run}_%06d.png")
    output_path = os.path.join(output_dir, f"input_{config_indices}{output_name}")

    # Video filter to ensure even dimensions (required for yuv420p)
    # This scales the video so both width and height are divisible by 2
    video_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",  # Suppress verbose output
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-vf", video_filter,  # Apply video filter for even dimensions
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",  # Encoding speed/compression efficiency tradeoff
        "-pix_fmt", "yuv420p",  # Ensures compatibility with most players
        output_path,
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video saved: {output_path}")


