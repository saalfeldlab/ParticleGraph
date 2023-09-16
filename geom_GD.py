import torch
from geomloss import SamplesLoss # See also ImagesLoss, VolumesLoss

# Create some large point clouds in 3D
x = torch.randn(100000, 3, requires_grad=True).cuda()
y = torch.randn(200000, 3).cuda()
# Define a Sinkhorn (~Wasserstein) loss between sampled measures
S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

Sxy = S_e(x, y) # By default, use constant weights = 1/number of samples
g_x, = torch.autograd.grad(Sxy, [x])