# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file
import os

import numpy as np
import torch
import torch.nn as nn

# from ParticleGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt
from tifffile import imread, imsave
from tqdm import trange
from ParticleGraph.utils import *
from ParticleGraph.config import ParticleGraphConfig
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


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





def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)
def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
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


class MLP(nn.Module):

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None, initialisation=None):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)

        if initialisation == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else :
            nn.init.normal_(layer.weight, std=0.1)
            nn.init.zeros_(layer.bias)

        self.layers.append(layer)

        if activation=='tanh':
            self.activation = F.tanh
        else:
            self.activation = F.relu

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


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



if __name__ == '__main__':

    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import matplotlib.pyplot as plt
    import torch



    matplotlib.use("Qt5Agg")


    config_list = ['signal_N6_a1']


    for config_file_ in config_list:

        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        device = set_device(config.training.device)
        print(f'device  {device}')
        print(f'folder  {config.dataset}')

        dataset_name = config.dataset

        simulation_config = config.simulation
        train_config = config.training
        model_config = config.graph_model
        n_frames = config.simulation.n_frames
        n_particles = config.simulation.n_particles
        n_runs = config.training.n_runs
        n_particle_types = config.simulation.n_particle_types
        delta_t = config.simulation.delta_t
        p = config.simulation.params
        omega = model_config.omega
        cmap = CustomColorMap(config=config)
        dimension = config.simulation.dimension
        max_radius = config.simulation.max_radius
        field_type = model_config.field_type

        x_list = []
        y_list = []
        for run in trange(1):
            if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
                x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
                y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
                x = to_numpy(torch.stack(x))
                y = to_numpy(torch.stack(y))
            else:
                x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
                y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
            x_list.append(x)
            y_list.append(y)

        activity = torch.tensor(x_list[0], device=device)
        activity = activity[:, :, 8:9].squeeze()
        activity = activity.t()

        # plt.figure(figsize=(15, 10))
        # n = np.random.permutation(n_particles)
        # for i in range(10):
        #     plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        # plt.xlabel('time', fontsize=64)
        # plt.ylabel('$x_{i}$', fontsize=64)
        # plt.xticks([0, 20000], fontsize=48)
        # plt.yticks(fontsize=48)
        # plt.tight_layout()
        # plt.show()


        # img_siren = Siren(in_features=2, out_features=1, hidden_features=256,
        #                   hidden_layers=3, outermost_linear=True, first_omega_0=80, hidden_omega_0=80.)
        # img_siren.cuda()
        #
        # total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
        # steps_til_summary = 10
        #
        # optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
        # ground_truth = activity[0:256,0:256].flatten()
        # ground_truth = ground_truth[None, :, None]
        # model_input = get_mgrid(256, 2).cuda()
        # model_input = model_input[None,:,:]
        #
        # for step in trange(total_steps):
        #     model_output, coords = img_siren(model_input)
        #     loss = ((model_output - ground_truth) ** 2).mean()
        #
        #     if not step % steps_til_summary:
        #         print("Step %d, Total loss %0.6f" % (step, loss))
        #         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        #         axes[0].imshow(model_output.cpu().view(256, 256).detach().numpy())
        #         axes[1].imshow(ground_truth.cpu().view(256, 256).detach().numpy())
        #         plt.show()
        #
        #     optim.zero_grad()
        #     loss.backward()
        #     optim.step()




        nlayers = 32

        # model = MLP(input_size=1, output_size=1, nlayers=nlayers, hidden_size=512, device=device)

        model = Siren(in_features=1, out_features=1, hidden_features=256,hidden_layers=3, first_omega_0=80, hidden_omega_0=80., outermost_linear=True)

        # model = TimeCoordinatedSineMLP(input_size=1, hidden_size=256, output_size=1, num_layers=3, omega_0=80)

        model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=1E-4)
        model.train()

        indices = np.arange(0, n_frames+1,100).astype(int)
        t = torch.linspace(0, n_frames+1,(n_frames+1)//100, dtype=torch.float32, device=device) / 1000
        t = t[None,:,None]

        y = activity[0][indices]
        y = y[None,0:1000,None]


        for epoch in trange(10000):
            optimizer.zero_grad()
            pred = model(t)[0]
            loss = (pred- y).norm(2)
            loss.backward()
            optimizer.step()

            if epoch%1000==0:

                fig = plt.figure()
                plt.plot(to_numpy(t.squeeze()), to_numpy(y.squeeze()), linewidth=2)
                plt.plot(to_numpy(t.squeeze()), to_numpy(pred.squeeze()), linewidth=2)
                plt.tight_layout()
                plt.show()































