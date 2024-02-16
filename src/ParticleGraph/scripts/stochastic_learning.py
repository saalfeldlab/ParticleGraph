
import torch
import numpy as np
from torch.distributions import Categorical
from torch.distributions import Categorical
import torch.nn as nn


class DeepNormal(nn.Module):

    def __init__(self, n_inputs, n_hidden):
        super().__init__()

        # Shared parameters
        self.shared_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
        )

        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
            nn.Softplus(),  # enforces positivity
        )

    def forward(self, x):
        # Shared embedding
        shared = self.shared_layer(x)

        # Parametrization of the mean
        μ = self.mean_layer(shared)

        # Parametrization of the standard deviation
        σ = self.std_layer(shared)

        return torch.distributions.Normal(μ, σ)

def compute_loss(model, x, y):
    normal_dist = model(x)
    neg_log_likelihood = -normal_dist.log_prob(y)
    return torch.mean(neg_log_likelihood)



if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')


    probs = torch.rand(100)
    probs = probs / torch.sum(probs)
    probs = probs.to('cuda:0')
    gt_sampler = Categorical(probs)

    learned_distribution = torch.zeros(10, dtype=torch.float32, device='cuda:0', requires_grad=True)
    with torch.no_grad():
        learned_distribution = torch.rand(10, dtype=torch.float32, device='cuda:0')
        learned_distribution = learned_distribution / torch.sum(learned_distribution)

    learned_distribution.requires_grad = True
    m = Categorical(learned_distribution)
    
    optimizer = torch.optim.Adam([learned_distribution], lr=0.01)

    for n in range(1000):

        draw = torch.randint(0, 10, (1,), device='cuda:0')
        loss = m.log_prob(draw.detach()) - gt_sampler.log_prob(draw.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        


        print(f'loss: {loss.item()}')


        
    

    

    
    