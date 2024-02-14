import numpy as np
import torch

from ParticleGraph.utils import to_numpy


def init_particles(model_config, device):
    nparticles = model_config['nparticles']
    nparticle_types = model_config['nparticle_types']
    v_init = model_config['v_init']

    cycle_length = torch.clamp(torch.abs(torch.ones(nparticle_types, 1, device=device) * 400 + torch.randn(nparticle_types, 1, device=device) * 150),min=100, max=700)

    if model_config['boundary'] == 'periodic':
        pos = torch.rand(nparticles, 2, device=device)
    else:
        pos = torch.randn(nparticles, 2, device=device) * 0.5
    dpos = v_init * torch.randn((nparticles, 2), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    type = torch.zeros(int(nparticles / nparticle_types), device=device)
    for n in range(1, nparticle_types):
        type = torch.cat((type, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
    type = type[:, None]
    if model_config['p'] == 'continuous':
        type = torch.tensor(np.arange(nparticles), device=device)
        type = type[:, None]
    features = torch.zeros((nparticles, 2), device=device)
    cycle_length_distrib = cycle_length[to_numpy(type[:, 0]).astype(int)]
    cycle_duration = torch.rand(nparticles, device=device)
    cycle_duration = cycle_duration[:, None]
    cycle_duration = cycle_duration * cycle_length_distrib
    particle_id = torch.arange(nparticles, device=device)
    particle_id = particle_id[:, None]

    return pos, dpos, type, features, cycle_duration, particle_id
