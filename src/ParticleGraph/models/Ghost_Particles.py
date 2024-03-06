import numpy as np
import torch
import torch.nn as nn

class Renderer(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, num_layers=3, out_dim=2):
        super().__init__()
        act_fn = nn.ReLU()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            # layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return torch.clamp(out, min=0, max=1)

class Ghost_Particles(torch.nn.Module):

    def __init__(self, model_config, n_particles, device):
        super(Ghost_Particles, self).__init__()
        self.n_ghosts = model_config.training.n_ghosts
        self.n_frames = model_config.simulation.n_frames
        self.n_dataset = model_config.training.n_runs
        self.device = device

        self.ghost_pos = nn.Parameter(torch.rand((self.n_dataset, self.n_frames, self.n_ghosts, 2), device=device, requires_grad=True))

        self.data = nn.Parameter(0*torch.randn((self.n_dataset, 32, self.n_ghosts, 128), device=device), requires_grad=True)
        
        self.N1 = torch.arange(n_particles,n_particles+self.n_ghosts, device=device, requires_grad=False)
        self.V1 = torch.zeros((self.n_ghosts,2), device=device, requires_grad=False)
        self.T1 = torch.zeros(self.n_ghosts, device=device, requires_grad=False)
        self.H1 = torch.zeros((self.n_ghosts,2), device=device, requires_grad=False)
        self.A1 = torch.zeros(self.n_ghosts, device=device, requires_grad=False)

        self.renderer = Renderer(in_dim=32, hidden_dim=32, num_layers=2, out_dim=2)
        self.renderer = self.renderer.to(device)
        
        

    def get_pos (self, dataset_id, frame):

        return torch.concatenate((self.N1[:,None], self.ghost_pos[dataset_id, frame:frame+1,:,:].squeeze(),self.V1,self.T1[:,None],self.H1,self.A1[:,None]), 1)
    
    def get_pos_t(self, dataset_id, frame):

        t0 = np.floor(128*frame/self.n_frames).astype(int)
        t1 = t0 + 1
        alpha = 128*frame/self.n_frames - t0

        sample = self.data[dataset_id, :, :, t0] * alpha + self.data[dataset_id, :, :, t1] * (1-alpha)
        sample = sample.permute(1, 0)
        
        pos = self.renderer(sample)
        
        return torch.concatenate((self.N1[:,None], pos,self.V1,self.T1[:,None],self.H1,self.A1[:,None]), 1)
    
    
    

