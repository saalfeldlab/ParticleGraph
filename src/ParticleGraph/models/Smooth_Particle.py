# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file
import os

import torch
from torch import nn

import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import torch_geometric.data as data

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
import matplotlib

from ParticleGraph.utils import choose_boundary_values
from ParticleGraph.config import ParticleGraphConfig



class Smooth_Particle(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the speed of the particles (dimension 2)
    """

    def __init__(self, config=[], aggr_type='mean', bc_dpos=[], dimension=[]):
        super(Smooth_Particle, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.smooth_radius = config.training.smooth_radius
        self.smooth_function = config.training.smooth_function

        tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dimension)
        mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid), 1)
        self.mgrid = mgrid.to(device)


    def forward(self, x=[], has_field=False):


        distance = torch.sum(bc_dpos(x[:, None, 1:self.dimension + 1] - x[None, :, 1:self.dimension + 1]) ** 2, dim=2)
        adj_t = ((distance <  self.smooth_radius ** 2) & (distance > 0)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        xp = x
        self.edge_index = edge_index


        if has_field:
            field = xp[:,6:7]
        else:
            field = torch.ones_like(xp[:,0:1])

        out = self.propagate(edge_index, pos=xp[:, 1:self.dimension+1], field=field)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, field_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)
        distance = torch.sqrt(distance_squared)

        d = self.W(distance, self.smooth_radius, self.smooth_function)

        return d[:,None]

    def W(self, d, s, function):

        match function:
            case 'gaussian':
                w_density = 1/(np.pi*s**2)*torch.exp(-d**2/s**2) / 1E3
            case triangular:
                w_density = 4/(np.pi*s**8)*(s**2-d**2)**3 / 1E3


        return w_density


class Smooth_Particle_Field(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the speed of the particles (dimension 2)
    """

    def __init__(self, config=[], aggr_type='mean', bc_dpos=[], dimension=[]):
        super(Smooth_Particle_Field, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.smooth_radius = config.training.smooth_radius
        self.smooth_function = config.training.smooth_function

        tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dimension)
        mgrid = torch.cat((torch.ones((mgrid.shape[0], 1)), mgrid), 1)
        self.mgrid = mgrid.to(device)


    def forward(self, x=[], has_field=False):


        distance = torch.sum(bc_dpos(x[:, None, 1:self.dimension + 1] - mgrid[None, :, 1:self.dimension + 1]) ** 2, dim=2)
        adj_t = ((distance <  self.smooth_radius ** 2) & (distance > 0)).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()
        xp = torch.cat((mgrid, x[:, 0:self.dimension + 1]), 0)
        edge_index[0,:] = edge_index[0,:] + mgrid.shape[0]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        self.edge_index = edge_index

        if has_field:
            field = xp[:,6:7]
        else:
            field = torch.ones_like(xp[:,0:1])

        out = self.propagate(edge_index, pos=xp[:, 1:self.dimension+1], field=field)

        density = out[0:self.mgrid.shape[0],0]
        grad = out[0:self.mgrid.shape[0],1:]

        return density, grad


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, field_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)
        distance = torch.sqrt(distance_squared)

        d, d_d = self.W(distance, self.smooth_radius, self.smooth_function)
        pos = torch.argwhere(distance==0)
        if pos.numel() > 0:
            d[pos] = 0
            d_d[pos] = 0

        d_d = d_d/distance
        d_d = d_d[:,None].repeat(1,2) * self.bc_dpos(pos_j - pos_i)

        t = torch.cat((d[:,None], d_d), 1)

        return t

    def W(self, d, s, function):

        match function:
            case 'gaussian':
                w_density = 1/(np.pi*s**2)*torch.exp(-d**2/s**2) / 1E3
                wp_density = 2/(np.pi*s**4)*d*torch.exp(-d**2/s**2) / 1E6
            case triangular:
                w_density = 4/(np.pi*s**8)*(s**2-d**2)**3 / 1E3
                wp_density = 24/(np.pi*s**8)*d*(s**2-d**2)**2 / 1E6


        return w_density, wp_density


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


if __name__ == '__main__':


    device = 'cuda:0'
    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    x_list = np.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_falling_water_ramp/x_list_2.npy')
    x_list = torch.tensor(x_list, dtype=torch.float32, device=device)

    plt.style.use('dark_background')

    bc_pos, bc_dpos = choose_boundary_values('no')
    config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/test_smooth_particle.yaml')
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    smooth_radius  = config.training.smooth_radius

    model_density = Smooth_Particle(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)
    # model_density = Smooth_Particle_Field(config=config, aggr_type='mean', bc_dpos=bc_dpos, dimension=dimension)
    mgrid = model_density.mgrid.clone().detach()


    for k in trange(0,len(x_list),10):

        x = x_list[k].squeeze()
        density = model_density(x=x, has_field=False)
        # density, grad = model_density(x=x, has_field=False)


        bViz = 2
        if (bViz == 1):
            xp = torch.cat((mgrid, x[:, 0:dimension + 1]), 0)
            edges = model_density.edge_index

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(x[:, 2].detach().cpu().numpy(),
                        x[:, 1].detach().cpu().numpy(), s=10, c='w')
            plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                        mgrid[:, 1].detach().cpu().numpy(), s=1, c='r')
            pixel = 5020
            plt.scatter(mgrid[pixel, 2].detach().cpu().numpy(),
                        mgrid[pixel, 1].detach().cpu().numpy(), s=40, c='g')
            pos = torch.argwhere(edges[1,:] == pixel).squeeze()
            plt.scatter(xp[edges[0,pos], 2].detach().cpu().numpy(), xp[edges[0,pos], 1].detach().cpu().numpy(), s=10, c='b')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/field_grid_{k}.png")
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                        mgrid[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/field_density_{k}.png")
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(mgrid[:, 2].detach().cpu().numpy(),
                        mgrid[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
            idx = torch.arange(0, mgrid.shape[0], 1)
            idx = torch.randperm(idx.shape[0])
            idx = idx[0:2000]
            for n in idx:
                plt.arrow(mgrid[n, 2].detach().cpu().numpy(), mgrid[n, 1].detach().cpu().numpy(),
                          grad[n, 1].detach().cpu().numpy(), grad[n, 0].detach().cpu().numpy(), head_width=0.005, color='w', alpha=0.5)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/field_grad_{k}.png")
            plt.close()

        if (bViz == 2):

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(x[:, 2].detach().cpu().numpy(),
                        x[:, 1].detach().cpu().numpy(), s=10, c=density.detach().cpu().numpy(), vmin=0, vmax=1)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig(f"tmp/particle_density_{k}.png")
            plt.close()









def test_siren():


    cameraman = ImageFitting(256)
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                      hidden_layers=3, outermost_linear=True, first_omega_0=80, hidden_omega_0=80.)
    img_siren.cuda()

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in trange(total_steps):
        model_output, coords = img_siren(model_input)
        # model_output = gradient(model_output, coords)
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(256, 256).detach().numpy())
            axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy())
            axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy())
            plt.savefig(f"tmp/output_{step}.png")
            plt.close()

        optim.zero_grad()
        loss.backward()
        optim.step()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(model_output.cpu().view(256, 256).detach().numpy())
    axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy())
    axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy())
    plt.show()






    model_siren = Siren_Network(image_width=256, in_features=2, out_features=1, hidden_features=256, hidden_layers=3,
                                outermost_linear=True)
    model_siren = model_siren.to(device=device)
    optimizer = torch.optim.Adam(lr=1e-4, params=model_siren.parameters())

    i0 = imread('data/pics_boat.tif')

    y = torch.tensor(i0, dtype=torch.float32, device=device)
    y = y.flatten()
    y = y[:, None]

    coords = get_mgrid(256, dim=2)
    coords = coords.to('cuda:0')

    print(coords.device, y.device)

    for epoch in trange(10000):
        optimizer.zero_grad()

        x = model_siren() ** 2

        loss = (x - y).norm(2)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            pred = model_siren() ** 2
            pred = torch.reshape(pred, (256, 256))
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(pred.detach().cpu().numpy())
            # plt.scatter(y.detach().cpu().numpy(),x.detach().cpu().numpy(),c='k',s=1)
            plt.savefig(f"tmp/output2_{epoch}.png")
            plt.close()
