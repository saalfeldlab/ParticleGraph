import torch
import numpy as np

from ParticleGraph.generators import PDE_A, PDE_B, PDE_E, PDE_G, PDE_Z, RD_Gray_Scott, RD_FitzHugh_Nagumo, RD_RPS, \
    Laplacian_A, PDE_O
from ParticleGraph.utils import choose_boundary_values


def choose_model(model_config, device):
    model_name = model_config['model']
    n_particle_types = model_config['nparticle_types']
    aggr_type = model_config['aggr_type']
    has_mesh = 'Mesh' in model_config['model']
    n_node_types = model_config['nnode_types']

    # create boundary functions for position and velocity respectively
    bc_pos, bc_dpos = choose_boundary_values(model_config['boundary'])

    if has_mesh:
        p = initialize_random_values(n_particle_types, device)
        if len(model_config['p']) > 0:
            for n in range(n_particle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = PDE_Z()
        c = initialize_random_values(n_particle_types, device)
        for n in range(n_particle_types):
            c[n] = torch.tensor(model_config['c'][n])

        match model_name:
            case 'RD_Gray_Scott_Mesh':
                mesh = RD_Gray_Scott(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                     bc_dpos=bc_dpos)
            case 'RD_FitzHugh_Nagumo_Mesh':
                mesh = RD_FitzHugh_Nagumo(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                          bc_dpos=bc_dpos)
            case 'RD_RPS_Mesh':
                mesh = RD_RPS(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'], bc_dpos=bc_dpos)
            case 'DiffMesh' | 'WaveMesh':
                mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                   bc_dpos=bc_dpos)
            case _:
                raise ValueError(f'Unknown model {model_name}')
    else:
        mesh = []
        match model_name:
            case 'PDE_A':
                print(f'Generate PDE_A')
                p = torch.ones(n_particle_types, 4, device=device) + torch.rand(n_particle_types, 4, device=device)
                if len(model_config['p']) > 0:
                    for n in range(n_particle_types):
                        p[n] = torch.tensor(model_config['p'][n])
                if n_particle_types == 1:
                    model = PDE_A(aggr_type=aggr_type, p=p, sigma=model_config['sigma'], bc_dpos=bc_dpos)
                else:
                    model = PDE_A(aggr_type=aggr_type, p=torch.squeeze(p), sigma=model_config['sigma'], bc_dpos=bc_dpos)
            case 'PDE_B':
                print(f'Generate PDE_B')
                p = torch.rand(n_particle_types, 3, device=device) * 100  # comprised between 10 and 50
                if len(model_config['p']) > 0:
                    for n in range(n_particle_types):
                        p[n] = torch.tensor(model_config['p'][n])
                model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)
            case 'PDE_G':
                if model_config['p'][0] == -1:
                    p = np.linspace(0.5, 5, n_particle_types)
                    p = torch.tensor(p, device=device)
                if len(model_config['p']) > 1:
                    for n in range(n_particle_types):
                        p[n] = torch.tensor(model_config['p'][n])
                model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p), clamp=model_config['clamp'],
                              pred_limit=model_config['pred_limit'], bc_dpos=bc_dpos)
            case 'PDE_E':
                p = initialize_random_values(n_particle_types, device)
                if len(model_config['p']) > 0:
                    for n in range(n_particle_types):
                        p[n] = torch.tensor(model_config['p'][n])
                model = PDE_E(aggr_type=aggr_type, p=torch.squeeze(p),
                              clamp=model_config['clamp'], pred_limit=model_config['pred_limit'],
                              prediction=model_config['prediction'], bc_dpos=bc_dpos)
            case 'PDE_O':
                p = initialize_random_values(n_particle_types, device)
                if len(model_config['p']) > 0:
                    for n in range(n_particle_types):
                        p[n] = torch.tensor(model_config['p'][n])
                model = PDE_O(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos, beta=model_config['beta'])
            case 'Maze':
                print(f'Generate PDE_B')
                p = torch.rand(n_particle_types, 3, device=device) * 100  # comprised between 10 and 50
                for n in range(n_particle_types):
                    p[n] = torch.tensor(model_config['p'][n])
                model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)
                c = initialize_random_values(n_node_types, device)
                for n in range(n_node_types):
                    c[n] = torch.tensor(model_config['c'][n])
                mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],
                                   bc_dpos=bc_dpos)
            case _:
                raise ValueError(f'Unknown model {model_name}')

    return model, mesh, bc_pos, bc_dpos


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)
