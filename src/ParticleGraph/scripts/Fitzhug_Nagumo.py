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
import torch.optim as optim



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
        super(Siren, self).__init__()

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

        output = self.net(coords)
        return output

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


class model_duo(nn.Module):

    def __init__(self, device=None):

        super(model_duo, self).__init__()

        self.device = device
        
        self.mlp0 = MLP(input_size=3, output_size=1, nlayers=5, hidden_size=128, device=self.device)
        self.mlp1 = MLP(input_size=2, output_size=1, nlayers=3, hidden_size=16, device=self.device)
        self.siren = Siren(in_features=1, out_features=1, hidden_features=128, hidden_layers=3, outermost_linear=True, first_omega_0=80, hidden_omega_0=80.)
        self.siren = self.siren.to(self.device)

    def forward(self, x):
        
        return self.siren(x)




if __name__ == '__main__':

    device = set_device('auto')
    print(f'device  {device}')

    # Parameters
    a = 0.7
    b = 0.8
    epsilon = 0.18

    # Simulation settings
    T = 1000       # total time
    dt = 0.1      # time step
    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)

    # Initialize variables
    v = np.zeros(n_steps)
    w = np.zeros(n_steps)
    I_ext = np.zeros(n_steps)

    # Initial conditions
    v[0] = -1.0
    w[0] = 1.0

    # External excitation: periodic pulse every 30s lasting 1s
    pulse_interval = 80.0  # seconds
    pulse_duration = 1.0   # seconds
    pulse_amplitude = 0.8  # strength of excitation

    for i, t in enumerate(time):
        if (t % pulse_interval) < pulse_duration:
            I_ext[i] = pulse_amplitude

    # Rollout using Euler method
    for i in range(n_steps - 1):
        dv = v[i] - (v[i]**3)/3 - w[i] + I_ext[i]
        dw = epsilon * (v[i] + a - b * w[i])
        v[i+1] = v[i] + dt * dv
        w[i+1] = w[i] + dt * dw

    # Plotting
    fig = plt.figure(figsize=(12, 6))
    # Time series
    plt.subplot(2, 1, 1)
    plt.plot(time, v, label='v (membrane potential)')
    plt.plot(time, w, label='w (recovery variable)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('FitzHugh-Nagumo with Periodic Excitation')
    plt.legend()
    plt.grid(True)
    # Plot I_ext
    plt.subplot(2, 1, 2)
    plt.plot(time, I_ext, color='red', label='External input $I_{ext}$')
    plt.xlabel('Time')
    plt.ylabel('Input current')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./tmp/time_series.png', dpi=170)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(v, w)
    plt.xlabel('v')
    plt.ylabel('w')
    plt.title('Phase Portrait')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./tmp/phase_portrait.png', dpi=170)
    plt.close()

    v_true = torch.tensor(v, dtype=torch.float32, device=device)
    w_true = torch.tensor(w, dtype=torch.float32, device=device)
    I_ext = torch.tensor(I_ext, dtype=torch.float32, device=device)

    model = model_duo(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1E-3)
    model.train()

    batch_size = 1000
    t = torch.linspace(0, n_steps + 1, (n_steps + 1), dtype=torch.float32, device=device) / n_steps
    t = t[None, 0:n_steps, None]

    for epoch in trange(100000):

        time = np.random.randint(1, n_steps-1, batch_size).astype(int)

        optimizer.zero_grad()

        w = model.siren(t[:,time,:]/n_steps)[0]

        loss = F.mse_loss(w, w_true[time,None])


        # dv_pred = model.mlp0(torch.cat((v_true[time,None], w, I_ext[time,None]), dim=1))
        # dw_pred = model.mlp1(torch.cat((v_true[time,None], w), dim=1))
        #
        # y_dv = (v_true[time+1,None]-v_true[time-1,None]) / (2*dt)
        # y_dw = (model.siren(t[:,time+1,:]/n_steps)[0] - model.siren(t[:,time-1,:]/n_steps)[0]) / (2*dt)
        #
        # loss = F.mse_loss(dv_pred, y_dv) + F.mse_loss(dw_pred, y_dw)

        loss.backward()
        optimizer.step()

        if epoch % 10000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            with torch.no_grad():
                model.eval()

                time = np.arange(0,n_steps).astype(int)
                w = model.siren(t[:,time,:]/n_steps)[0]
                fig = plt.figure(figsize=(12, 6))
                plt.plot(to_numpy(w))
                plt.plot(to_numpy(w_true), linestyle='--')
                plt.show()






