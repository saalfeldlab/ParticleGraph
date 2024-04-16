import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Qt5Agg")
from ParticleGraph.generators import PDE_ParticleField, PDE_A, PDE_B, PDE_B_bis, PDE_E, PDE_G, PDE_GS, PDE_N, PDE_Z, RD_Gray_Scott, RD_FitzHugh_Nagumo, RD_RPS, \
    Laplacian_A, PDE_O
from ParticleGraph.utils import choose_boundary_values
from ParticleGraph.data_loaders import load_solar_system


def generate_from_data(config, device, visualize=True, folder=None, step=None):

    data_folder_name = config.data_folder_name

    match data_folder_name:
        case 'graphs_data/solar_system':
            load_solar_system(config, device, visualize, folder, step)
        case _:
            raise ValueError(f'Unknown data folder name {data_folder_name}')


def choose_model(config, device):
    particle_model_name = config.graph_model.particle_model_name
    model_signal_name = config.graph_model.signal_model_name
    aggr_type = config.graph_model.aggr_type
    n_particles = config.simulation.n_particles
    n_node_types = config.simulation.n_node_types
    n_nodes = config.simulation.n_nodes
    n_particle_types = config.simulation.n_particle_types
    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)
    dimension = config.simulation.dimension

    params = config.simulation.params

    match particle_model_name:
        case 'PDE_ParticleField':
            pos_rate = torch.ones(n_node_types, device=device)*8E-4
            for n in range(n_node_types):
                pos_rate[n] = torch.tensor(config.simulation.pos_rate[n])
            neg_rate = torch.ones(n_node_types, device=device)*8E-4
            for n in range(n_node_types):
                neg_rate[n] = torch.tensor(config.simulation.pos_rate[n])
            p = torch.rand(n_particle_types, 4, device=device) * 100  # comprised between 10 and 50
            if params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            else:
                print(p)
            model = PDE_ParticleField(aggr_type=aggr_type,  pos_rate=pos_rate, neg_rate=neg_rate, beta=config.simulation.beta, delta_t=config.simulation.delta_t,  p=torch.squeeze(p), bc_dpos=bc_dpos, n_particles=n_particles, n_nodes=n_nodes)
        case 'PDE_A':
            p = torch.ones(n_particle_types, 4, device=device) + torch.rand(n_particle_types, 4, device=device)
            if config.simulation.non_discrete_level>0:
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
            elif params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            else:
                print(p)
            sigma = config.simulation.sigma
            p = p if n_particle_types == 1 else torch.squeeze(p)
            model = PDE_A(aggr_type=aggr_type, p=torch.squeeze(p), sigma=sigma, bc_dpos=bc_dpos, dimension=dimension)
            # matplotlib.use("Qt5Agg")
            # rr = torch.tensor(np.linspace(0, 0.075, 1000)).to(device)
            # for n in range(n_particles):
            #     func= model.psi(rr,p[n])
            #     plt.plot(rr.detach().cpu().numpy(),func.detach().cpu().numpy(),c='k',alpha=0.01)
        case 'PDE_B':
            p = torch.rand(n_particle_types, 3, device=device) * 100  # comprised between 10 and 50
            if params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            else:
                print(p)
            model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)
        case 'PDE_B_bis':
            p = torch.rand(n_particle_types, 3, device=device) * 100  # comprised between 10 and 50
            if params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            else:
                print(p)
            model = PDE_B_bis(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)
        case 'PDE_G':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            if len(params) > 1:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p), clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_GS':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            if len(params) > 1:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_GS(aggr_type=aggr_type, p=torch.squeeze(p), clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_E':
            p = initialize_random_values(n_particle_types, device)
            if len(params) > 0:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_E(aggr_type=aggr_type, p=torch.squeeze(p),
                          clamp=config.training.clamp, pred_limit=config.training.pred_limit,
                          prediction=config.graph_model.prediction, bc_dpos=bc_dpos)
        case 'PDE_O':
            p = initialize_random_values(n_particle_types, device)
            if len(params) > 0:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_O(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos, beta=config.simulation.beta)
        case 'Maze':
            p = torch.rand(n_particle_types, 3, device=device) * 100  # comprised between 10 and 50
            for n in range(n_particle_types):
                p[n] = torch.tensor(params[n])
            model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)
        case _:
            model = PDE_Z(device=device)

    match model_signal_name:

        case 'PDE_N':
            p = torch.rand(n_particle_types, 2, device=device) * 100  # comprised between 10 and 50
            if params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_N(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)



    return model, bc_pos, bc_dpos


def choose_mesh_model(config, device):
    mesh_model_name = config.graph_model.mesh_model_name
    n_node_types = config.simulation.n_node_types
    aggr_type = config.graph_model.mesh_aggr_type
    _, bc_dpos = choose_boundary_values(config.simulation.boundary)

    c = initialize_random_values(n_node_types, device)
    for n in range(n_node_types):
        c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])

    beta = config.simulation.beta

    match mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            mesh_model = RD_Gray_Scott(aggr_type=aggr_type, c=torch.squeeze(c), beta=beta, bc_dpos=bc_dpos)
        case 'RD_FitzHugh_Nagumo_Mesh':
            mesh_model = RD_FitzHugh_Nagumo(aggr_type=aggr_type, c=torch.squeeze(c), beta=beta, bc_dpos=bc_dpos)
        case 'RD_RPS_Mesh':
            mesh_model = RD_RPS(aggr_type=aggr_type, c=torch.squeeze(c), beta=beta, bc_dpos=bc_dpos)
        case 'DiffMesh' | 'WaveMesh':
            mesh_model = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=beta, bc_dpos=bc_dpos)
        case 'Chemotaxism_Mesh':
            c = initialize_random_values(n_node_types, device)
            for n in range(n_node_types):
                c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
            mesh_model = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=beta, bc_dpos=bc_dpos)
        case 'PDE_O_Mesh':
            c = initialize_random_values(n_node_types, device)
            for n in range(n_node_types):
                c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
            mesh_model = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=beta, bc_dpos=bc_dpos)
        case _:
            raise ValueError(f'Unknown model {model_name}')

    return mesh_model


# TODO: this seems to be used to provide default values in case no parameters are given?
def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)
