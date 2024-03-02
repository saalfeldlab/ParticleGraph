import numpy as np
import torch
import torch.nn as nn

class Ghost_Particles(torch.nn.Module):

    def __init__(self, model_config, n_particles, device):
        super(Ghost_Particles, self).__init__()
        self.n_ghosts = model_config.training.n_ghosts
        self.n_frames = model_config.simulation.n_frames
        self.n_dataset = model_config.training.n_runs
        self.device = device

        self.ghost_pos = nn.Parameter(torch.rand((self.n_dataset, self.n_frames, self.n_ghosts, 2), device=device, requires_grad=True))
        self.N1 = torch.arange(n_particles,n_particles+self.n_ghosts, device=device, requires_grad=False)
        self.V1 = torch.zeros((self.n_ghosts,2), device=device, requires_grad=False)
        self.T1 = torch.zeros(self.n_ghosts, device=device, requires_grad=False)
        self.H1 = torch.zeros((self.n_ghosts,2), device=device, requires_grad=False)
        self.A1 = torch.zeros(self.n_ghosts, device=device, requires_grad=False)

    def get_pos (self, dataset_id, frame):

        return torch.concatenate((self.N1[:,None], self.ghost_pos[dataset_id, frame:frame+1,:,:].squeeze(),self.V1,self.T1[:,None],self.H1,self.A1[:,None]), 1)

