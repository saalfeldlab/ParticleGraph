import glob
import json
import logging
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.data as data
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import umap
import yaml  # need to install pyyaml
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from sklearn import metrics
from sklearn.cluster import KMeans
from tifffile import imread
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils.convert import to_networkx
from tqdm import trange
from matplotlib import rc
import os
from ParticleGraph.utils import to_numpy, cc
import numpy as np


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


class LorentzPredictor(nn.Module):
    def __init__(self, input_size, output_size, pupil=None):
        super(LorentzPredictor, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, 3)
        self.fc3 = nn.Linear(3, output_size)
        self.fc4 = nn.Linear(output_size, output_size)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class LorenzDataset:

    def __init__(self, s, r, b, dt, num_steps, device):

        self.device = device

        self.dt = dt
        self.num_steps = num_steps
        self.s = s
        self.r = r
        self.b = b

        self.xyz = torch.zeros((num_steps + 1, 3), device=self.device)  # Need one more for the initial values
        self.xyz[0] = torch.tensor([0., 1., 1.05], device=device)  # Set initial values
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):

            x, y, z = self.xyz[i]
            x_dot = self.s * (y - x)
            y_dot = self.r * x - y - x * z
            z_dot = x * y - self.b * z

            newx = self.xyz[i,0] + x_dot * self.dt
            newy = self.xyz[i, 1] + y_dot * self.dt
            newz = self.xyz[i, 2] + z_dot * self.dt

            self.xyz[i+1,0] = newx
            self.xyz[i+1, 1] = newy
            self.xyz[i+1, 2] = newz


    def len(self):
        # Count number of points in xyz
        n_points = self.xyz.shape[0]
        return n_points

    def getitem_past_x(self, t, K):
        # Return window from t to t+K-1
        return self.xyz[t-K:t, 0]

    def getitem_future_x(self, t, K):
        # Return window from t to t+K-1
        return self.xyz[t:t+K, 0]

    def getitem_past_future_x(self, t, K_past, K_future):
        # Return window from t to t+K-1
        return self.xyz[t-K_past:t, 0], self.xyz[t:t+K_future, 0]

    def getitem_window_t1_x(self, t, K):
        # Return window from t to t+K-1
        return self.xyz[t-K:t, 0], self.xyz[t-k+1:t+1, 0]

    def getvariance(self):

        return torch.var(self.xyz[0,:])



if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('use of https://github.com/gpeyre/.../ml_10_particle_system.ipynb')
    print('')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    data = LorenzDataset(s=10, r=28, b=2.667, dt=0.01, num_steps=10000, device=device)

    xyzs = to_numpy(data.xyz)
    plt.ion()
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    model = LorentzPredictor(input_size=10, output_size=1000)
    model.apply(LorentzPredictor.init_weights)
    model.to(device)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print (f'Total parameters: {total_params}')

    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training loop sliding window size K
    batch_size = 1000
    k = 10
    T = 1000
    variance = data.getvariance()

    for epoch in range(1000):

            loss1=0
            loss2=0

            for n in range(batch_size):

                # draw an integer number between 10 and 1000
                t = np.random.randint(1000, data.len()-T)
                past_sample, future_sample = data.getitem_past_future_x(t=t, K_past=k, K_future=T)
                pred = model(past_sample)
                # loss1 += F.mse_loss(pred, future_sample)/(T*variance)
                # loss2 += 1/2*(F.mse_loss(pred, future_sample)/(T*variance))**2
                loss1 += -torch.log(1-F.mse_loss(pred, future_sample)/(T*variance))

            loss=loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch}, loss1 {np.round(loss1.item()/batch_size,4)}')

    past_sample, future_sample = data.getitem_past_future_x(t=100, K_past=k, K_future=T)
    pred = model(pred)


    import matplotlib.pyplot as plt

    fig= plt.figure(figsize=(8,20))
    plt.ion()
    ax = fig.add_subplot(3, 1, 1)
    plt.plot(to_numpy(future_sample))
    ax = fig.add_subplot(3, 1, 2)
    plt.plot(to_numpy(future_sample))
    ax = fig.add_subplot(3, 1, 3)
    pred = model(past_sample)
    plt.plot(to_numpy(pred))

    # compare with T+1
    # inference larger T






