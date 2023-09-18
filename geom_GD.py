import torch
from geomloss import SamplesLoss # See also ImagesLoss, VolumesLoss
import torch
import torch.nn as nn

class Embedding_freq(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding_freq, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device {device}')

num_t_freq = 2
embedding_t = Embedding(1, num_t_freq)
x=torch.ones(1)*0.25
print(embedding_t(x))

# Create some large point clouds in 3D
x = torch.randn(1000, 3, requires_grad=True).cuda()
y = torch.randn(2000, 3).cuda()
# Define a Sinkhorn (~Wasserstein) loss between sampled measures
S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

Sxy = S_e(x, y) # By default, use constant weights = 1/number of samples
g_x, = torch.autograd.grad(Sxy, [x])