import torch
from prettytable import PrettyTable

from ParticleGraph.models import Interaction_Particles, Mesh_Laplacian, Mesh_RPS
from ParticleGraph.utils import choose_boundary_values


def choose_training_model(model_config, device):
    model_name = model_config['model']
    n_particle_types = model_config['nparticle_types']
    aggr_type = model_config['aggr_type']
    has_mesh = 'Mesh' in model_config['model']
    n_node_types = model_config['nnode_types']

    bc_pos, bc_dpos = choose_boundary_values(model_config['boundary'])

    match model_name:
        case 'PDE_A' | 'PDE_B':
            model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
        case 'PDE_E':
            model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
        case 'PDE_G':
            model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
        case 'DiffMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
        case 'WaveMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
        case 'RD_RPS_Mesh':
            model = Mesh_RPS(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
        case _:
            raise ValueError(f'Unknown model {model_name}')

    return model, bc_pos, bc_dpos


def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size

    return get_batch_size


def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 2 else batch_size

    return get_batch_size


def set_trainable_parameters(model, lr_embedding, lr):
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.a)
    _, *parameters = trainable_params

    embedding = model.a
    optimizer = torch.optim.Adam([embedding], lr=lr_embedding)
    for parameter in parameters:
        optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params
