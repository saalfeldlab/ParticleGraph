import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import glob
import torch_geometric as pyg
import torch_geometric.data as data
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import time
from shutil import copyfile
from prettytable import PrettyTable
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from geomloss import SamplesLoss
import torch_geometric.transforms as T
import pandas
import trackpy
# from numpy import vstack
# from sklearn.metrics import confusion_matrix, recall_score, f1_score
# from torch_geometric.utils import degree
import umap
from tifffile import imwrite, imread
import pygame
from tools import *
from random import uniform
import colorsys
from matrix import *
from math import pi,sin,cos
from torch_geometric.utils import degree
from scipy.spatial import Delaunay
import logging

def distmat_square(X, Y):
    return torch.sum(bc_diff(X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
def kernel(X, Y):
    return -torch.sqrt(distmat_square(X, Y))
def MMD(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.sum(kernel(X, X)) / n ** 2 + \
        torch.sum(kernel(Y, Y)) / m ** 2 - \
        2 * torch.sum(kernel(X, Y)) / (n * m)
    return a.item()
def normalize99(Y, lower=1, upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return x01, x99
def norm_velocity(xx, device):
    mvx = torch.mean(xx[:, 3])
    mvy = torch.mean(xx[:, 4])
    vx = torch.std(xx[:, 3])
    vy = torch.std(xx[:, 4])
    nvx = np.array(xx[:, 3].detach().cpu())
    vx01, vx99 = normalize99(nvx)
    nvy = np.array(xx[:, 4].detach().cpu())
    vy01, vy99 = normalize99(nvy)

    return torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)
def norm_acceleration(yy, device):
    max = torch.mean(yy[:, 0])
    may = torch.mean(yy[:, 1])
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = normalize99(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = normalize99(nay)

    return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)

class cc:

    def __init__(self, model_config):
        self.model_config = model_config
        self.model = model_config['model']
        if model_config['cmap'] == 'tab10':
            self.nmap = 8
        else:
            self.nmap = model_config['nparticle_types']

    def color(self,index):
        if self.model=='ElecParticles':
            if index == 0:
                index = (0, 0, 1)
            elif index== 1:
                index = (0, 0.5, 0.75)
            elif index == 2:
                index = (1, 0, 0)
            return (index)
        else:
            #color_map = plt.cm.get_cmap(self.model_config['cmap'])
            color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
            index = color_map(index/self.nmap)

        return index

class Laplacian_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], c=[], beta=[], clamp=[]):
        super(Laplacian_A, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = beta
        self.clamp = clamp

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        c = self.c[x[:, 5].detach().cpu().numpy()]
        c = c[:,None]

        heat_flow = self.beta * c * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        heat_flow = heat_flow - x[:,7:8]*1E-3*0

        # if (torch.min(heat_flow)<-0.5):
        #     pos = torch.argwhere(heat_flow<-0.5).detach().cpu().numpy().astype(int)
        #     k=pos[0,0]
        #     print(x[pos, 1], x[pos, 2], x[pos, 6])
        #     print(k)
        #     pos = torch.argwhere(edge_index[0, :] == k).detach().cpu().numpy().astype(int)
        #     print(edge_index[:,pos])
        #
        #     coeff = edge_attr[pos.squeeze()]
        #     coeff = coeff[:, None]
        #     print(coeff)
        #     print(coeff * x[edge_index[1, pos], 6])
        # if (torch.max(x[:, 6:7] + x[:, 7:8] + heat_flow[:,0:1]) > 5.05):
        #     pos = torch.argwhere(x[:, 6:7] + x[:, 7:8] + heat_flow[:,0:1]>5.05).detach().cpu().numpy().astype(int)
        #     k=pos[0,0]
        #     print(x[pos, 1], x[pos, 2], x[pos, 6])
        #     print(k)
        #     pos = torch.argwhere(edge_index[0, :] == k).detach().cpu().numpy().astype(int)
        #     print(edge_index[:,pos])
        #
        #     coeff = edge_attr[pos.squeeze()]
        #     coeff = coeff[:, None]
        #     print(coeff)
        #     print(coeff * x[edge_index[1, pos], 6])

        return heat_flow

    def message(self, x_i, x_j, edge_attr):

        # edge_attr = torch.clamp(edge_attr,min=-2*torch.std(edge_attr),max=2*torch.std(edge_attr))
        heat = edge_attr * x_j[:, 6]

        return heat[:, None]

    def psi(self, r, p):
        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

        return psi[:, None]
class MLP(nn.Module):

    def __init__(self, input_size, output_size, nlayers, hidden_size, device):

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
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
class Boid:
	def __init__(self, x, y):
		self.position = Vector(x, y)
		vec_x = uniform(-1, 1)
		vec_y = uniform(-1, 1)
		self.velocity = Vector(vec_x, vec_y)
		self.velocity.normalize()
		#set a random magnitude
		self.velocity = self.velocity * uniform(1.5, 4)
		self.acceleration = Vector()
		self.color = (255, 255,255)
		self.temp = self.color
		self.secondaryColor = (70, 70, 70)
		self.max_speed = 5
		self.max_length = 1
		self.size = 2
		self.stroke = 5
		self.angle = 0
		self.hue = 0
		self.toggles = {"separation":True, "alignment":True, "cohesion":True}
		self.values = {"separation":0.1, "alignment":0.1, "cohesion":0.1}
		self.radius = 40
	def limits(self, width , height):
		if self.position.x > width:
			self.position.x = 0
		elif self.position.x < 0:
			self.position.x = width

		if self.position.y > height:
			self.position.y = 0
		elif self.position.y < 0:
			self.position.y = height

	def behaviour(self, flock):
		self.acceleration.reset()

		if self.toggles["separation"] == True:
			avoid = self.separation(flock)
			avoid = avoid * self.values["separation"]
			self.acceleration.add(avoid)

		if self.toggles["cohesion"]== True:
			coh = self.cohesion(flock)
			coh = coh * self.values["cohesion"]
			self.acceleration.add(coh)

		if self.toggles["alignment"] == True:
			align = self.alignment(flock)
			align = align * self.values["alignment"]
			self.acceleration.add(align)

	def separation(self, flockMates):
		total = 0
		steering = Vector()

		for mate in flockMates:
			dist = getDistance(self.position, mate.position)
			if mate is not self and dist < self.radius:
				temp = SubVectors(self.position,mate.position)
				temp = temp/(dist ** 2 + 1E-9)
				steering.add(temp)
				total += 1
		if total > 0:
			steering = steering / total
			# steering = steering - self.position
			steering.normalize()
			steering = steering * self.max_speed
			steering = steering - self.velocity
			steering.limit(self.max_length)

		return steering

	def alignment(self, flockMates):
		total = 0
		steering = Vector()
		# hue = uniform(0, 0.5)
		for mate in flockMates:
			dist = getDistance(self.position, mate.position)
			if mate is not self and dist < self.radius:
				vel = mate.velocity.Normalize()
				steering.add(vel)
				mate.color = hsv_to_rgb( self.hue ,1, 1)
				total += 1
		if total > 0:
			steering = steering / total
			steering.normalize()
			steering = steering * self.max_speed
			steering = steering - self.velocity.Normalize()
			steering.limit(self.max_length)
		return steering
	def cohesion(self, flockMates):
		total = 0
		steering = Vector()
		for mate in flockMates:
			dist = getDistance(self.position, mate.position)
			if mate is not self and dist < self.radius:
				steering.add(mate.position)
				total += 1
		if total > 0:
			steering = steering / total
			steering = steering - self.position
			steering.normalize()
			steering = steering * self.max_speed
			steering = steering - self.velocity
			steering.limit(self.max_length)

		return steering

	def update(self):

		self.position = self.position + self.velocity
		self.velocity = self.velocity + self.acceleration
		self.velocity.limit(self.max_speed)
		self.angle = self.velocity.heading() + pi/2

	def Draw(self, distance, scale):
		ps = []
		points = [None for _ in range(3)]

		points[0] = [[0],[-self.size],[0]]
		points[1] = [[self.size//2],[self.size//2],[0]]
		points[2] = [[-self.size//2],[self.size//2],[0]]

		for point in points:
			rotated = matrix_multiplication(rotationZ(self.angle) , point)
			z = 1/(distance - rotated[2][0])

			projection_matrix = [[z, 0, 0], [0, z, 0]]
			projected_2d = matrix_multiplication(projection_matrix, rotated)

			x = int(projected_2d[0][0] * scale) + self.position.x
			y = int(projected_2d[1][0] * scale) + self.position.y
			ps.append((x, y))

		ps.append(ps[0])
		return ps

class Particles_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[], prediction=[]):
        super(Particles_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.tau = tau
        self.prediction = prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        newv = self.tau * self.propagate(edge_index, x=(x, x))


        if self.prediction == '2nd_derivative':
            oldv = x[:, 3:5]
            acc = newv - oldv
            return acc
        else:
            return newv

    def message(self, x_i, x_j):
        r = torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)  # squared distance
        pp = self.p[x_i[:, 5].detach().cpu().numpy(), :]
        psi = - pp[:, 2] * torch.exp(-r ** pp[:, 0] / (2 * sigma ** 2)) + pp[:, 3] * torch.exp(-r ** pp[:, 1] / (2 * sigma ** 2))
        return psi[:, None] * bc_diff(x_i[:, 1:3] - x_j[:, 1:3])

    def psi(self, r, p):
        return r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(
            -r ** (2 * p[1]) / (2 * sigma ** 2)))
class Particles_E(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[], clamp=[], pred_limit=[], prediction=[]):
        super(Particles_E, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.tau = tau
        self.clamp = clamp
        self.pred_limit = pred_limit
        self.prediction = prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1))
        r = torch.clamp(r, min=self.clamp)
        r = torch.concatenate((r[:, None], r[:, None]), -1)

        p1 = self.p[x_i[:, 5].detach().cpu().numpy()]
        p1 = p1.squeeze()
        p1 = torch.concatenate((p1[:, None], p1[:, None]), -1)

        p2 = self.p[x_j[:, 5].detach().cpu().numpy()]
        p2 = p2.squeeze()
        p2 = torch.concatenate((p2[:, None], p2[:, None]), -1)

        acc = p1 * p2 * bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / r ** 3
        acc = torch.clamp(acc, max=self.pred_limit)

        return acc

    def psi(self, r, p1, p2):
        r_ = torch.clamp(r, min=self.clamp)
        acc = p1 * p2 * r / r_ ** 2
        acc = torch.clamp(acc, max=self.pred_limit)
        return acc  # Elec particles
class Particles_G(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=[], p=[], tau=[], clamp=[], pred_limit=[]):
        super(Particles_G, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.tau = tau
        self.clamp = clamp
        self.pred_limit = pred_limit

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.tau * self.propagate(edge_index, x=(x, x))
        return acc

    def message(self, x_i, x_j):
        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1))
        r = torch.clamp(r, min=self.clamp)
        r = torch.concatenate((r[:, None], r[:, None]), -1)

        p = self.p[x_j[:, 5].detach().cpu().numpy()]
        p = p.squeeze()
        p = torch.concatenate((p[:, None], p[:, None]), -1)

        acc = p * bc_diff(x_j[:, 1:3] - x_i[:, 1:3]) / r ** 3

        return torch.clamp(acc, max=self.pred_limit)

    def psi(self, r, p):
        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

        return psi[:, None]

class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(InteractionParticles, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.ndataset = model_config['nrun'] - 1
        self.upgrade_type = model_config['upgrade_type']
        self.prediction = model_config['prediction']
        self.upgrade_type = model_config['upgrade_type']
        self.nlayers_update  = model_config['nlayers_update']
        self.hidden_size_update = model_config['hidden_size_update']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

        self.lin_update = MLP(input_size=self.output_size + self.embedding + 2, output_size=self.output_size, nlayers=self.nlayers_update,
                            hidden_size=self.hidden_size_update, device=self.device)

    def forward(self, data, data_id, step, vnorm, cos_phi, sin_phi):

        self.data_id = data_id
        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pred = self.propagate(edge_index, x=(x, x))

        if self.upgrade_type == 'linear':
            embedding = self.a[self.data_id, x[:, 0].detach().cpu().numpy(), :]
            pred = self.lin_update(torch.cat((pred, x[:, 3:5], embedding), dim=-1))

        if step == 2:
            deg = pyg_utils.degree(edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)  # test, if degree = 0 acc =0
            return deg * pred
        else:
            return pred

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / self.radius
        x_i_vx = x_i[:, 3:4] / self.vnorm[4]
        x_i_vy = x_i[:, 4:5] / self.vnorm[4]
        x_j_vx = x_j[:, 3:4] / self.vnorm[4]
        x_j_vy = x_j[:, 4:5] / self.vnorm[4]

        if (self.data_augmentation) & (self.step == 1):
            new_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_x
            delta_pos[:, 1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        embedding = self.a[self.data_id, x_i[:, 0].detach().cpu().numpy(), :]

        if self.prediction == '2nd_derivative':
            in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)
        else:
            if self.prediction == 'first_derivative_L':
                in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)
            if self.prediction == 'first_derivative_S':
                in_features = torch.cat((delta_pos, r, embedding), dim=-1)

        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        return -(r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(-r ** (2 * p[1]) / (2 * sigma ** 2))))
class GravityParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(GravityParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.upgrade_type = model_config['upgrade_type']
        self.ndataset = model_config['nrun'] - 1
        self.clamp = model_config['clamp']
        self.pred_limit = model_config['pred_limit']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id, step, vnorm, cos_phi, sin_phi):

        self.data_id = data_id
        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x, x))

        if step == 2:
            deg = pyg_utils.degree(edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)
            return deg * acc
        else:
            return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / self.radius
        x_i_vx = x_i[:, 3:4] / self.vnorm[4]
        x_i_vy = x_i[:, 4:5] / self.vnorm[5]
        x_j_vx = x_j[:, 3:4] / self.vnorm[4]
        x_j_vy = x_j[:, 4:5] / self.vnorm[5]

        if (self.data_augmentation) & (self.step == 1):
            new_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_x
            delta_pos[:, 1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        embedding = self.a[self.data_id, x_j[:, 0].detach().cpu().numpy(), :]  # depends on other
        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

        return psi[:, None]
class ElecParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(ElecParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.ndataset = model_config['nrun'] - 1
        self.upgrade_type = model_config['upgrade_type']
        self.clamp = model_config['clamp']
        self.pred_limit = model_config['pred_limit']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id, step, vnorm, cos_phi, sin_phi):

        self.data_id = data_id
        self.vnorm = vnorm
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        acc = self.propagate(edge_index, x=(x, x))

        if step == 2:
            deg = pyg_utils.degree(edge_index[0], data.num_nodes)
            deg = (deg > 0)
            deg = (deg > 0).type(torch.float32)
            deg = torch.concatenate((deg[:, None], deg[:, None]), axis=1)
            return deg * acc
        else:
            return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum(bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) ** 2, axis=1)) / self.radius  # squared distance
        r = r[:, None]

        delta_pos = bc_diff(x_i[:, 1:3] - x_j[:, 1:3]) / self.radius
        x_i_vx = x_i[:, 3:4] / self.vnorm[4]
        x_i_vy = x_i[:, 4:5] / self.vnorm[5]
        x_j_vx = x_j[:, 3:4] / self.vnorm[4]
        x_j_vy = x_j[:, 4:5] / self.vnorm[5]

        if (self.data_augmentation) & (self.step == 1):
            new_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_x
            delta_pos[:, 1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        embedding0 = self.a[self.data_id, x_i[:, 0].detach().cpu().numpy(), :]
        embedding1 = self.a[self.data_id, x_j[:, 0].detach().cpu().numpy(), :]  # depends on other
        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, embedding0, embedding1), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p1, p2):
        r_ = torch.clamp(r, min=self.clamp)
        acc = p1 * p2 * r / r_ ** 3
        acc = torch.clamp(acc, max=self.pred_limit)
        return -acc  # Elec particles
class MeshDiffusion(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):

        super(MeshDiffusion, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']
        self.nparticles = model_config['nparticles']
        self.radius = model_config['radius']
        self.data_augmentation = model_config['data_augmentation']
        self.noise_level = model_config['noise_level']
        self.embedding = model_config['embedding']
        self.dataset_name = model_config['dataset']
        graph_files = glob.glob(f"graphs_data/graphs_particles_{self.dataset_name}/x_list*")
        NGraphs = len(graph_files)

        self.ndataset = NGraphs - 1
        self.upgrade_type = model_config['upgrade_type']
        self.prediction = model_config['prediction']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                            hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.ndataset, int(self.nparticles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):

        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        heat = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        return heat

    def message(self, x_i, x_j, edge_attr):

        embedding = self.a[self.data_id, x_i[:, 0].detach().cpu().numpy(), :]

        in_features = torch.cat((edge_attr[:,None], x_j[:, 6:7]-x_i[:, 6:7], embedding), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        return r * (-p[2] * torch.exp(-r ** (2 * p[0]) / (2 * sigma ** 2)) + p[3] * torch.exp(
            -r ** (2 * p[1]) / (2 * sigma ** 2)))

def data_generate(model_config,bVisu=True, bDetails=False, bErase=False, step=5):
    print('')
    print('Generating data ...')

    dataset_name = model_config['dataset']
    folder = f'./graphs_data/graphs_particles_{dataset_name}/'
    os.makedirs(folder, exist_ok=True)

    # files = glob.glob(f"./tmp_data/*")
    # for f in files:
    #     os.remove(f)

    if bErase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            os.remove(f)

    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    json_ = json.dumps(model_config)
    f = open(f"{folder}/model_config.json", "w")
    f.write(json_)
    f.close()

    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    noise_level = model_config['noise_level']
    v_init = model_config['v_init']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    rr = torch.tensor(np.linspace(0, radius * 2, 1000))
    rr = rr.to(device)
    if bMesh:
        particle_value_map = model_config['particle_value_map']
        particle_type_map = model_config['particle_type_map']


    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if model_config['model'] == 'Particles_A':
        print(f'Generate Particles_A')
        p = torch.ones(nparticle_types, 4, device=device) + torch.rand(nparticle_types, 4, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        if nparticle_types == 1:
            model = Particles_A(aggr_type=aggr_type, p=p, tau=model_config['tau'],
                                prediction=model_config['prediction'])
        else:
            model = Particles_A(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                                prediction=model_config['prediction'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if model_config['model'] == 'GravityParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
    if model_config['model'] == 'ElecParticles':
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
                print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
                torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        model = Particles_E(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'],
                            prediction=model_config['prediction'])
        psi_output = []
        for n in range(nparticle_types):
            for m in range(nparticle_types):
                psi_output.append(model.psi(rr, torch.squeeze(p[n]), torch.squeeze(p[m])))
    if bMesh:
        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])
        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])
        model_mesh = Laplacian_A(aggr_type=aggr_type, c=torch.squeeze(c), beta=model_config['beta'],clamp=model_config['clamp'])
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, torch.squeeze(p[n])))
            print(f'p{n}: {np.round(torch.squeeze(p[n]).detach().cpu().numpy(), 4)}')
            torch.save(torch.squeeze(p[n]), f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')

    torch.save({'model_state_dict': model.state_dict()}, f'graphs_data/graphs_particles_{dataset_name}/model.pt')

    for run in range(model_config['nrun']):

        x_list=[]
        y_list=[]
        h_list=[]

        if (model_config['model'] == 'WaveMesh') | (model_config['boundary'] == 'periodic'):
            X1 = torch.rand(nparticles, 2, device=device)
        else:
            X1 = torch.randn(nparticles, 2, device=device) * 0.5
        V1 = v_init * torch.randn((nparticles, 2), device=device)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = T1[:, None]
        ####### TO BE CHANGED #############################
        # h = torch.zeros((nparticles, 1), device=device)
        H1 = torch.zeros((nparticles, 2), device=device)
        H1[:,0:1] = torch.ones((nparticles, 1), device=device) + torch.randn((nparticles, 1), device=device) / 2

        if bMesh:

            x_width = int(np.sqrt(nparticles))
            xs = torch.linspace(0, 1, steps=x_width)
            ys = torch.linspace(0, 1, steps=x_width)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            x = torch.reshape(x, (x_width**2, 1))
            y = torch.reshape(y, (x_width**2, 1))
            x_width = 1/x_width/8
            X1[0:nparticles,0:1] = x[0:nparticles]
            X1[0:nparticles,1:2] = y[0:nparticles]
            X1=X1+torch.randn(nparticles, 2, device=device) * x_width
            X1=torch.clamp(X1,min=0,max=1)

            # X1=X1*0
            # n_width = int(np.sqrt(nparticles))
            # for k in range(n_width):
            #     for n in range(n_width):
            #         X1[k + n * n_width,0]=k/n_width + (n%2) / n_width / 2
            #         X1[k + n * n_width, 1] = n / n_width
            # X1 = X1 + torch.randn(nparticles, 2, device=device) * 1/n_width/8
            # X1 = torch.clamp(X1, min=0, max=1)
            # plt.ion()
            # plt.scatter(X1[:,0].detach().cpu().numpy(),X1[:,1].detach().cpu().numpy(),s=10)

            i0 = imread(f'graphs_data/{particle_value_map}')
            values = i0[(X1[:, 0].detach().cpu().numpy() * 255).astype(int), (X1[:, 1].detach().cpu().numpy() * 255).astype(int)]
            H1[:,0] = torch.tensor(values / 255 * 5000, device=device)
            torchsum0 = torch.sum(H1)
            # plt.scatter(X1[:, 0].detach().cpu().numpy(), X1[:, 1].detach().cpu().numpy(), s=1,
            #             c=H1[:, 0].detach().cpu().numpy())

            i0 = imread(f'graphs_data/{particle_type_map}')
            values = i0[(X1[:, 0].detach().cpu().numpy() * 255).astype(int), (X1[:, 1].detach().cpu().numpy()*255).astype(int)]
            T1 = torch.tensor(values, device=device)
            T1 = T1[:, None]
            # plt.scatter(X1[:, 0].detach().cpu().numpy(), X1[:, 1].detach().cpu().numpy(), s=10,
            #             c=T1[:, 0].detach().cpu().numpy())

        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:, None]

        time.sleep(0.5)

        noise_current = 0 * torch.randn((nparticles, 2), device=device)
        noise_prev_prev = 0 * torch.randn((nparticles, 2), device=device)

        for it in tqdm(range(model_config['start_frame'], nframes)):

            if it==0:
                V1=torch.clamp(V1,min=-torch.std(V1),max=+torch.std(V1))

            noise_prev_prev = noise_prev_prev.clone().detach()
            noise_prev = noise_current.clone().detach()
            noise_current = torch.randn((nparticles, 2), device=device) * noise_level

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), H1.clone().detach()), 1)
            x_noise = x.clone().detach()

            if (it >= 0) & (noise_level > 0):
                x_noise = x.clone().detach()
                x_noise[:, 1:3] = x[:, 1:3] + noise_current
                x_noise[:, 3:5] = x[:, 3:5] + noise_current - noise_prev
            if (it>=0):
                x_list.append(x_noise.clone().detach())

            if bMesh:
                dataset = data.Data(x=x_noise, pos=x_noise[:, 1:3])
                transform_0 = T.Compose([T.Delaunay()])
                dataset_face = transform_0(dataset).face
                mesh_pos = torch.cat((x_noise[:, 1:3], torch.ones((x_noise.shape[0], 1), device=device)), dim=1)
                edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face,
                                                                       normalization="None")  # "None", "sym", "rw"
                dataset_mesh = data.Data(x=x_noise, edge_index=edge_index, edge_attr=edge_weight, device=device)

            distance = torch.sum(bc_diff(x_noise[:, None, 1:3] - x_noise[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x_noise, pos=x_noise[:, 1:3], edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset)
            if (it >= 0) & (noise_level == 0):
                y_list.append(y.clone().detach())
            if (it >= 0) & (noise_level > 0):
                y_noise = y[:, 0:2] + noise_current - 2 * noise_prev + noise_prev_prev
                y_list.append(y_noise.clone().detach())

            if model_config['prediction'] == '2nd_derivative':
                V1 += y[:, 0:2]
            else:
                V1 = y[:, 0:2]

            if not(bMesh):
                X1 = bc_pos(X1 + V1)

            if model_config['model'] == 'DiffMesh':
                if it >= 0:
                    mask = torch.argwhere ((X1[:,0]>0.1)&(X1[:,0]<0.9)&(X1[:,1]>0.1)&(X1[:,1]<0.9)).detach().cpu().numpy().astype(int)
                    mask = mask[:, 0:1]
                    with torch.no_grad():
                        pred = model_mesh(dataset_mesh)
                        H1[mask,1:2] = pred[mask]
                    H1[mask,0:1] += H1[mask,1:2]
                    h_list.append(pred)

            if model_config['model'] == 'WaveMesh':
                if it >= 0:
                    # mask = torch.argwhere ((X1[:,0]>0.005)&(X1[:,0]<0.995)&(X1[:,1]>0.005)&(X1[:,1]<0.995)).detach().cpu().numpy().astype(int)
                    # mask = mask[:, 0:1]
                    # invmask = torch.argwhere ((X1[:,0]<=0.025)|(X1[:,0]>=0.975)|(X1[:,1]<=0.025)|(X1[:,1]>=0.975)).detach().cpu().numpy().astype(int)
                    # invmask = invmask[:, 0:1]
                    with torch.no_grad():
                        pred = model_mesh(dataset_mesh)
                        H1[:,1:2] += pred[:]
                    H1[:,0:1] += H1[:,1:2]
                    h_list.append(pred)

            if (run == 0) & (it % step == 0) & (it >= 0) & bVisu:

                fig = plt.figure(figsize=(11.8, 12))
                # plt.ion()
                ax = fig.add_subplot(2, 2, 1)
                if model_config['model'] == 'GravityParticles':
                    for n in range(nparticle_types):
                        g = p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 7.5
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                    x[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                    alpha=0.75,color=cmap.color(n))
                elif bMesh:
                    pts = x_noise[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = torch.sum(x_noise[tri.simplices, 6], axis=1) / 3.0
                    if model_config['model'] == 'WaveMesh':
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(), facecolors=colors.detach().cpu().numpy(),edgecolors='k',vmin=-1500,vmax=1500)
                    else:
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=2500)

                    # plt.scatter(x_noise[:, 1].detach().cpu().numpy(),x_noise[:, 2].detach().cpu().numpy(), s=10, alpha=0.75,
                    #                 c=x[:, 6].detach().cpu().numpy(), cmap='gist_gray',vmin=-5000,vmax=5000)
                    # ax.set_facecolor([0.5,0.5,0.5])
                elif model_config['model'] == 'ElecParticles':
                    for n in range(nparticle_types):
                        g = np.abs(p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)
                        if model_config['p'][n][0]<=0:
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='r', alpha=0.5)
                        else:
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='b', alpha=0.5)
                else:
                    for n in range(nparticle_types):
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(), x[index_particles[n], 2].detach().cpu().numpy(), s=3, color=cmap.color(n))
                if bMesh | (model_config['boundary']=='periodic'):
                    plt.text(0, 1.08, f'frame: {it}')
                    plt.text(0, 1.03, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                else:
                    plt.text(-1.25, 1.5, f'frame: {it}')
                    plt.text(-1.25, 1.4, f'{x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])

                ax = fig.add_subplot(2, 2, 2)
                plt.scatter(x_noise[:, 1].detach().cpu().numpy(), x_noise[:, 2].detach().cpu().numpy(), s=1, color='k',alpha=0.75)
                if bDetails: # model_config['radius']<0.01:
                    pos = dict(enumerate(np.array(x_noise[:, 1:3].detach().cpu()), 0))
                    if bMesh:
                        vis = to_networkx(dataset_mesh, remove_self_loops=True, to_undirected=True)
                    else:
                        distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                        adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                        edge_index2 = adj_t2.nonzero().t().contiguous()
                        dataset2 = data.Data(x=x, edge_index=edge_index2)
                        vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False,alpha=0.3)
                if bMesh | (model_config['boundary']=='periodic'):
                    plt.xlim([-0.1,1.1])
                    plt.ylim([-0.1,1.1])
                else:
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])

                if bDetails:
                    ax = fig.add_subplot(2, 2, 3)
                    if model_config['model'] == 'GravityParticles':
                        for n in range(nparticle_types):
                            g = p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 7.5 * 4
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                        alpha=0.75,
                                        color=cmap.color(n))  # , facecolors='none', edgecolors='k')
                    elif bMesh:
                        pts = x_noise[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x_noise[tri.simplices, 6], axis=1) / 3.0

                        if model_config['model'] == 'WaveMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-1500,
                                          vmax=1500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=2500)
                        # ax.set_facecolor([0.5,0.5,0.5])
                    elif model_config['model'] == 'ElecParticles':
                        for n in range(nparticle_types):
                            g = np.abs(p[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20) * 4
                            if model_config['p'][n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='r', alpha=0.5)
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c='b', alpha=0.5)
                    elif model_config['model'] == 'Particles_A':
                        for n in range(nparticle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=50, alpha=0.75,
                                        color=cmap.color(n))

                    if bMesh | (model_config['boundary']=='periodic'):
                        plt.xlim([0.3, 0.7])
                        plt.ylim([0.3, 0.7])
                    else:
                        plt.xlim([-0.25, 0.25])
                        plt.ylim([-0.25, 0.25])

                    if not(bMesh):
                        for k in range(nparticles):
                            plt.arrow(x=x[k, 1].detach().cpu().item(),y=x[k, 2].detach().cpu().item(),
                                      dx=x[k, 3].detach().cpu().item()*model_config['arrow_length'], dy=x[k, 4].detach().cpu().item()*model_config['arrow_length'],color='k')

                    ax = fig.add_subplot(2, 2, 4)
                    if not(bMesh):
                        if len(x_list)>30:
                            x_all =torch.stack(x_list)
                            for k in range(nparticles):
                                xc = x_all[-30:-1, k, 1].detach().cpu().numpy().squeeze()
                                yc = x_all[-30:-1, k, 2].detach().cpu().numpy().squeeze()
                                plt.scatter(xc,yc,s=0.05, color='k',alpha=0.75)
                        elif len(x_list)>6:
                            x_all =torch.stack(x_list)
                            for k in range(nparticles):
                                xc = x_all[:, k, 1].detach().cpu().numpy().squeeze()
                                yc = x_all[:, k, 2].detach().cpu().numpy().squeeze()
                                plt.scatter(xc,yc,s=0.05, color='k',alpha=0.75)
                        plt.xlim([-1.3, 1.3])
                        plt.ylim([-1.3, 1.3])

                plt.tight_layout()
                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

        torch.save(x_list, f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
        torch.save(h_list, f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt')

        bDetails = False


    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)
    print('Normalization ...')
    arr = np.arange(0, NGraphs)
    x_list=[]
    y_list=[]
    for run in tqdm(arr):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt',map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt',map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))
    x = torch.stack(x_list)
    x = torch.reshape(x,(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    y = torch.stack(y_list)
    y = torch.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3]))
    vnorm = norm_velocity(x, device)
    ynorm = norm_acceleration(y, device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    print (vnorm)
    print (ynorm)
    if bMesh:
        h_list=[]
        for run in arr:
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt',map_location=device)
            h_list.append(torch.stack(h))
        h = torch.stack(h_list)
        h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(hnorm)

def data_generate_boid(model_config, bVisu=True, bDetails=True, bErase=False, step=1):

    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_data/*")
    # for f in files:
    #     os.remove(f)

    print('')
    print('Generating data ...')
    print('use of https://github.com/Josephbakulikira/simple-Flocking-simulation-python-pygame')
    
    dataset_name = model_config['dataset']
    folder = f'./graphs_data/graphs_particles_{dataset_name}/'
    os.makedirs(folder, exist_ok=True)

    if bErase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            os.remove(f)
    
    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))
    
    json_ = json.dumps(model_config)
    f = open(f"{folder}/model_config.json", "w")
    f.write(json_)
    f.close()
    
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    noise_level = model_config['noise_level']
    v_init = model_config['v_init']
    rr = torch.tensor(np.linspace(0, radius * 2, 1000))
    rr = rr.to(device)
    
    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))
    
    fps = 60
    scale = 40
    Distance = 5
    speed = 0.0005
    size =1000

    for run in range(model_config['nrun']):

        x_list = []
        y_list = []
        h_list = []

        X1 = torch.randn(nparticles, 2, device=device) * 0.5
        V1 = v_init * torch.randn((nparticles, 2), device=device)
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = T1[:, None]
        H1 = torch.ones((nparticles, 2), device=device) + torch.randn((nparticles, 1), device=device) / 2
        N1 = torch.arange(nparticles, device=device)
        N1 = N1[:, None]

        time.sleep(0.5)

        noise_current = 0 * torch.randn((nparticles, 2), device=device)
        noise_prev_prev = 0 * torch.randn((nparticles, 2), device=device)
        noise_prev = 0 * torch.randn((nparticles, 2), device=device)

        flock = []
        for i in range(nparticles):
            flock.append(Boid(np.random.randint(20, size - 20), np.random.randint(20, size - 20)))
        for n, boid in enumerate(flock):
            if nparticle_types == 1:
                p = model_config['p']
            else:
                p = model_config['p'][int(T1[n].detach().cpu().numpy())]
            boid.toggles = {"separation": True, "alignment": True, "cohesion": True}
            boid.values = {"separation": p[0] / 100, "alignment": p[1] / 100, "cohesion": p[2] / 100}

        for it in tqdm(range(nframes)):

            if bVisu & (it % step == 0):
                fig = plt.figure(figsize=(11.8, 12))
                #plt.ion()
                ax = fig.add_subplot(2, 2, 1)
                plt.xlim([0, size])
                plt.ylim([0, size])
            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), H1.clone().detach()), 1)
            for n, boid  in enumerate(flock):
                x[n,1]=torch.tensor(boid.position.x/1000, device=device)
                x[n,2]=torch.tensor(boid.position.y/1000, device=device)
                x[n,3]=torch.tensor(boid.velocity.x/1000, device=device)
                x[n,4]=torch.tensor(boid.velocity.y/1000, device=device)

            if (it >= 0) & (noise_level == 0):
                x_list.append(x)
                # torch.save(x, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')
            if (it >= 0) & (noise_level > 0):
                x_noise = x
                x_noise[:, 1:3] = x[:, 1:3] + noise_current
                x_noise[:, 3:5] = x[:, 3:5] + noise_current - noise_prev
                x_list.append(x_noise)
                # torch.save(x_noise, f'graphs_data/graphs_particles_{dataset_name}/x_{run}_{it}.pt')

            for n,boid in enumerate (flock):
                boid.radius = scale
                boid.limits(size, size)
                boid.behaviour(flock)
                boid.update()
                if bVisu & (it % step == 0):
                    ps = boid.Draw(Distance, scale)
                    ps = np.array(ps)
                    plt.plot(ps[:, 0], ps[:, 1], c=cmap.color(T1[n].detach().cpu().numpy()), alpha=0.5)

            y = torch.zeros((nparticles,2),device=device)
            for n, boid  in enumerate(flock):
                y[n,0]=torch.tensor(boid.acceleration.x/1000, device=device)
                y[n,1]=torch.tensor(boid.acceleration.y/1000, device=device)
            if (it >= 0) & (noise_level == 0):
                y_list.append(y)
            if (it >= 0) & (noise_level > 0):
                y_noise = y[:, 0:2] + noise_current - 2 * noise_prev + noise_prev_prev
                y_list.append(y_noise)

            if bVisu & (it%step==0):
                if bDetails:
                    ax = fig.add_subplot(2, 2, 2)
                    plt.scatter(x[:, 1].detach().cpu().numpy(), x[:, 2].detach().cpu().numpy(), s=1, color='k',
                                alpha=0.75)
                    pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                    distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                    adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                    edge_index2 = adj_t2.nonzero().t().contiguous()
                    dataset2 = data.Data(x=x, edge_index=edge_index2)
                    vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.3)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])

                    ax = fig.add_subplot(2, 2, 3)

                    for n in range(nparticle_types):
                        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                    x[index_particles[n], 2].detach().cpu().numpy(), s=50, alpha=0.75,
                                    color=cmap.color(n))
                    plt.xlim([0.3, 0.7])
                    plt.ylim([0.3, 0.7])
                    for k in range(nparticles):
                            plt.arrow(x=x[k, 1].detach().cpu().item(),y=x[k, 2].detach().cpu().item(),
                                      dx=x[k, 3].detach().cpu().item()*model_config['arrow_length'], dy=x[k, 4].detach().cpu().item()*model_config['arrow_length'],color='k')

                plt.tight_layout()
                plt.savefig(f"./tmp_data/Fig_{ntry}_{it}.tif")
                plt.close()

        for k in range(1,len(x_list)):
            prev=x_list[k-1]
            next=x_list[k]
            v = bc_diff(next[:,1:3]-prev[:,1:3])
            acc = v - prev[:,3:5]
            x_list[k][:,3:5]=v
            y_list[k-1][:,0:2]=acc

        # x_list[0][:,1:3]+x_list[0][:,3:5]+y_list[0]-x_list[1][:,1:3]

        torch.save(x_list, f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
        torch.save(h_list, f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt')

        bDetails = False
        bVisu = False
def data_train(model_config, bSparse=False):

    print('')

    model = []
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    data_augmentation = model_config['data_augmentation']
    embedding = model_config['embedding']
    batch_size = model_config['batch_size']
    batch_size = 1
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    sparsity = model_config['sparsity']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    print('log_dir: {}'.format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir,'models'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir,'tmp_training'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    files = glob.glob(f"{log_dir}/tmp_training/*")
    for f in files:
        os.remove(f)
    files = glob.glob(f"{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(model_config)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs - 1}')
    logger.info(f'Graph files N: {NGraphs - 1}')


    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
    if (model_config['model'] == 'Particles_A'):
        model = InteractionParticles(model_config, device)
    if (model_config['model'] == 'DiffMesh'):
        model = MeshDiffusion(model_config, device)
    if (model_config['model'] == 'WaveMesh'):
        model = MeshDiffusion(model_config, device)

    # net = f"./log/try_126/models/best_model_with_9_graphs_13.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lra = 1E-3
    lr = 1E-3

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    it = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if it == 0:
            optimizer = torch.optim.Adam([model.a], lr=lra)
        else:
            optimizer.add_param_group({'params': parameter, 'lr': lr})
        it += 1
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    logger.info(f'Learning rates: {lr}, {lra}')

    net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    logger.info(f'network: {net}')
    Nepochs = 25  ######################## 22
    logger.info(f'N epochs: {Nepochs}')
    print('')

    model.train()
    best_loss = np.inf
    list_loss = []
    embedding_center = []
    regul_embedding=0
    data_augmentation_loop = 20
    print(f'data_augmentation_loop: {data_augmentation_loop}')
    logger.info(f'data_augmentation_loop: {data_augmentation_loop}')
    print('Load data ...')
    x_list=[]
    y_list=[]
    time.sleep(0.5)
    for run in tqdm(np.arange(0, NGraphs)):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt',map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt',map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))
    ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{ntry}/vnorm.pt', map_location=device).to(device)
    logger.info(ynorm)
    logger.info(vnorm)
    if bMesh:
        h_list=[]
        for run in tqdm(np.arange(0, NGraphs)):
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt',map_location=device)
            h_list.append(torch.stack(h))
        hnorm = torch.load(f'./log/try_{ntry}/hnorm.pt', map_location=device).to(device)
        x = x_list[0][0].clone().detach()
        index_particles = []
        for n in range(model_config['nparticle_types']):
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
            index_particles.append(index.squeeze())
        logger.info(hnorm)

    print('Start training ...')
    logger.info("Start training ...")
    time.sleep(0.5)
    for epoch in range(Nepochs + 1):

        if epoch == 1:
            batch_size = model_config['batch_size']
            print(f'batch_size: {batch_size}')
            logger.info(f'batch_size: {batch_size}')

        if epoch == 5:
            if data_augmentation:
                data_augmentation_loop = 200
                print(f'data_augmentation_loop: {data_augmentation_loop}')
                logger.info(f'data_augmentation_loop: {data_augmentation_loop}')
        if epoch == 10:
            lra = 1E-3
            lr = 5E-4
            table = PrettyTable(["Modules", "Parameters"])
            it = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if it == 0:
                    optimizer = torch.optim.Adam([model.a], lr=lra)
                else:
                    optimizer.add_param_group({'params': parameter, 'lr': lr})
                it += 1
            print(f'Learning rates: {lr}, {lra}')
            logger.info(f'Learning rates: {lr}, {lra}')
        if epoch == 24:
            print('not training embedding ...')
            logger.info('not training embedding ...')
            model.a.requires_grad = False
            regul_embedding = 0

        total_loss = 0

        for N in range(0, nframes * data_augmentation_loop // batch_size):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            for batch in range(batch_size):

                k = np.random.randint(nframes - 1)
                x = x_list[run][k].clone().detach()

                if bMesh:
                    dataset = data.Data(x=x, pos=x[:, 1:3])
                    transform_0 = T.Compose([T.Delaunay()])
                    dataset_face = transform_0(dataset).face
                    mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
                    edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face,
                                                                           normalization="None")  # "None", "sym", "rw"
                    dataset = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)
                    dataset_batch.append(dataset)
                    y = h_list[run][k].clone().detach()/hnorm
                    if batch == 0:
                        y_batch = y
                    else:
                        y_batch = torch.cat((y_batch, y), axis=0)
                else:
                    distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                    adj_t = (distance < radius ** 2).float() * 1
                    t = torch.Tensor([radius ** 2])
                    edges = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x[:, :], edge_index=edges)
                    dataset_batch.append(dataset)
                    y = y_list[run][k].clone().detach()
                    if model_config['prediction'] == '2nd_derivative':
                        y = y / ynorm[4]
                    else:
                        y = y / vnorm[4]
                    if data_augmentation:
                        new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                        new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                        y[:, 0] = new_x
                        y[:, 1] = new_y
                    if batch == 0:
                        y_batch = y
                    else:
                        y_batch = torch.cat((y_batch, y), axis=0)

                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                optimizer.zero_grad()

                for batch in batch_loader:
                    if bMesh:
                        pred = model(batch, data_id=run - 1)
                    else:
                        pred = model(batch, data_id=run - 1, step=1, vnorm=vnorm, cos_phi=cos_phi, sin_phi=sin_phi)

            if regul_embedding>0:
                regul_term_embedding = (model.a[run-1] - embedding_center[0].clone().detach()) ** 2
                for k in range(1,model_config['ninteractions']):
                        regul_term_embedding = regul_term_embedding * (model.a[run-1]-embedding_center[k].clone().detach())**2
                regul_term_embedding = regul_embedding * torch.sqrt(torch.mean(regul_term_embedding))
                loss = (pred - y_batch).norm(2) + regul_term_embedding
            else:
                loss = (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # optimizer.zero_grad()
            # t = torch.sum(model.a[run])
            # loss = (pred - y_batch).norm(2) + t
            # loss.backward()
            # optimizer.step()
            # total_loss += loss.item()


        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        if (total_loss / nparticles / batch_size / N < best_loss):
            best_loss = total_loss / N / nparticles / batch_size
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs.pt'))
            print("Epoch {}. Loss: {:.6f} saving model  ".format(epoch,total_loss / N / nparticles / batch_size))
            logger.info("Epoch {}. Loss: {:.6f} saving model  ".format(epoch,total_loss / N / nparticles / batch_size))
        else:
            print("Epoch {}. Loss: {:.6f}".format(epoch,total_loss / N / nparticles / batch_size))
            logger.info("Epoch {}. Loss: {:.6f}".format(epoch,total_loss / N / nparticles / batch_size))

        list_loss.append(total_loss / N / nparticles / batch_size)

        fig = plt.figure(figsize=(16, 8))
        # plt.ion()

        ax = fig.add_subplot(2, 4, 1)
        plt.plot(list_loss, color='k')
        plt.ylim([0, 0.003])
        plt.xlim([0, 50])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        embedding = []
        for n in range(model.a.shape[0]):
            embedding.append(model.a[n])
        embedding = torch.stack(embedding).detach().cpu().numpy()
        embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
        embedding_particle = []
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                embedding_particle.append(embedding[index_particles[n]+m*nparticles, :])

        ax = fig.add_subplot(2, 4, 2)
        if (embedding.shape[1] > 2):
            ax = fig.add_subplot(2, 4, 2, projection='3d')
            for n in range(nparticle_types):
                ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2], color=cmap.color(n), s=1)
        else:
            if (embedding.shape[1] > 1):
                for m in range(model.a.shape[0]):
                    for n in range(nparticle_types):
                        plt.scatter(embedding_particle[n+m*nparticle_types][:, 0], embedding_particle[n+m*nparticle_types][:, 1], color=cmap.color(n), s=3)
                plt.xlabel('Embedding 0', fontsize=12)
                plt.ylabel('Embedding 1', fontsize=12)
            else:
                for n in range(nparticle_types):
                    plt.hist(embedding_particle[n][:, 0], width=0.01, alpha=0.5,color=cmap.color(n))

        ax = fig.add_subplot(2, 4, 3)
        if model_config['model'] == 'ElecParticles':
            acc_list = []
            for m in range(model.a.shape[0]):
                for k in range(nparticle_types):
                    for n in index_particles[k]:
                        rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                        embedding0 = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                        embedding1 = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                        in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                                 rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding0, embedding1), dim=1)
                        acc = model.lin_edge(in_features.float())
                        acc = acc[:, 0]
                        acc_list.append(acc)
                        if n % 5 == 0:
                            plt.plot(rr.detach().cpu().numpy(),
                                     acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'], linewidth=1,
                                     color=cmap.color(k),alpha=0.25)
            acc_list = torch.stack(acc_list)
            plt.xlim([0, 0.05])
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            coeff_norm = acc_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
        elif model_config['model'] == 'GravityParticles':
            acc_list = []
            for n in range(nparticles):
                rr = torch.tensor(np.linspace(0, radius * 1.3, 1000)).to(device)
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                acc_list.append(acc)
                plt.plot(rr.detach().cpu().numpy(),acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],color=cmap.color(x[n,5].detach().cpu().numpy()), linewidth=1,alpha=0.25)
            acc_list = torch.stack(acc_list)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim([1E-3, 0.2])
            plt.ylim([1, 1E7])
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            coeff_norm = acc_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
        elif model_config['model'] == 'Particles_A':
            acc_list = []
            for n in range(nparticles):
                rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                ### TO BE CHANGED ###
                if model_config['prediction'] == '2nd_derivative':
                    in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                else:
                    if model_config['prediction'] == 'first_derivative_L':
                        in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
                    if model_config['prediction'] == 'first_derivative_S':
                        in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None], rr[:, None] / model_config['radius'], embedding), dim=1)

                acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                acc_list.append(acc)
                if n%5==0:
                    plt.plot(rr.detach().cpu().numpy(),acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],color=cmap.color(x[n,5].detach().cpu().numpy()), linewidth=1,alpha=0.25)
            plt.xlabel('Distance [a.u]', fontsize=12)
            plt.ylabel('MLP [a.u]', fontsize=12)
            acc_list = torch.stack(acc_list)
            coeff_norm = acc_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
        elif bMesh:
            f_list = []
            for n in range(nparticles):
                r0 = torch.tensor(np.linspace(4, 5, 1000)).to(device)
                r1 = torch.tensor(np.linspace(-250, 250, 1000)).to(device)
                embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((r0[:, None], r1[:, None], embedding), dim=1)
                h = model.lin_edge(in_features.float())
                h = h[:, 0]
                f_list.append(h)
                if n % 5 == 0:
                    plt.plot(r1.detach().cpu().numpy(),
                             h.detach().cpu().numpy() * hnorm.detach().cpu().numpy(), linewidth=1,
                             color='k',alpha=0.05)
            f_list = torch.stack(f_list)
            coeff_norm = f_list.detach().cpu().numpy()
            trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int),
                              n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)
            particle_types = x_list[0][0, :, 5].clone().detach().cpu().numpy()
            ax = fig.add_subplot(2, 4, 4)
            for n in range(nparticle_types):
                plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1], s=5)
            plt.xlabel('UMAP 0', fontsize=12)
            plt.ylabel('UMAP 1', fontsize=12)

            kmeans = KMeans(init="random", n_clusters=nparticle_types, n_init=1000, max_iter=10000, random_state=13)
            kmeans.fit(proj_interaction)
            for n in range(nparticle_types):
                plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)
                pos = np.argwhere(kmeans.labels_ == n).squeeze().astype(int)

        ax = fig.add_subplot(2, 4, 4)
        for n in range(nparticle_types):
            plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                        color=cmap.color(n), s=5, alpha=0.75)
        plt.xlabel('UMAP 0', fontsize=12)
        plt.ylabel('UMAP 1', fontsize=12)

        kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=5000, max_iter=10000,random_state=13)
        kmeans.fit(proj_interaction)
        for n in range(nparticle_types):
            tmp = kmeans.labels_[index_particles[n]]
            sub_group = np.round(np.median(tmp))
            accuracy = len(np.argwhere(tmp == sub_group)) / len(tmp) * 100
            print(f'Sub-group {n} accuracy: {np.round(accuracy, 3)}')
            logger.info(f'Sub-group {n} accuracy: {np.round(accuracy, 3)}')
        for n in range(model_config['ninteractions']):
            plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)

        if (epoch==9) | (epoch==14) | (epoch==19) | (epoch==24):

            model_a_=model.a.clone().detach()
            model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
            embedding_center=[]
            for k in range(model_config['ninteractions']):
                pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, axis=0).values
                embedding_center.append(median_center.clone().detach())
                model_a_[pos, :] = torch.median(median_center, axis=0).values
            model_a_ = torch.reshape(model_a_, (model.a.shape[0],model.a.shape[1], model.a.shape[2]))

            # Constrain embedding with UMAP of plots clustering
            if sparsity=='replace':
                with torch.no_grad():
                    for n in range(model.a.shape[0]):
                        model.a[n]=model_a_[0].clone().detach()
                print(f'regul_embedding: replaced')
                logger.info(f'regul_embedding: replaced')
            elif 'regul' in sparsity:
                regul_embedding = float(sparsity[-4:])
                print(f'regul_embedding: {regul_embedding}')
                logger.info(f'regul_embedding: {regul_embedding}')

            if (epoch % 10 == 0) & (epoch > 0):
                best_loss = total_loss / N / nparticles / batch_size
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs.pt'))
                xx, rmserr_list = data_test(model_config, bVisu=True, bPrint=False, step=int(nframes//20), folder_out=f'{log_dir}/tmp_recons/')
                model.train()
            # if (epoch > 9):
            #     ax = fig.add_subplot(2, 4, 5)
            #     for n in range(nparticle_types):
            #         plt.scatter(xx[index_particles[n], 1], xx[index_particles[n], 2], s=1,color='k')
            #     ax = plt.gca()
            #     ax.axes.xaxis.set_ticklabels([])
            #     ax.axes.yaxis.set_ticklabels([])
            #     plt.xlim([0,1])
            #     plt.ylim([0,1])
            #     ax.axes.get_xaxis().set_visible(False)
            #     ax.axes.get_yaxis().set_visible(False)
            #     plt.axis('off')
            #     ax = fig.add_subplot(2, 4, 6)
            #     plt.plot(np.arange(len(rmserr_list)), rmserr_list, label='RMSE', color='r')
            #     plt.ylim([0, 0.1])
            #     plt.xlim([0, nframes])
            #     plt.tick_params(axis='both', which='major', labelsize=10)
            #     plt.xlabel('Frame [a.u]', fontsize=14)
            #     ax.set_ylabel('RMSE [a.u]', fontsize=14, color='r')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{ntry}_{epoch}.tif")
        plt.close()

def data_test(model_config, bVisu=False, bPrint=True, index_particles=0, prev_nparticles=0, new_nparticles=0,prev_index_particles=0,best_model=0,step=5, bTest='', folder_out='tmp_recons',initial_map=''):
    # files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/tmp_recons/*")
    # for f in files:
    #     os.remove(f)
    if bPrint:
        print('')
        print('Plot validation test ... ')

    model = []
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh=(model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')

    if index_particles == 0:
        index_particles = []
        np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
        for n in range(model_config['nparticle_types']):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    if (model_config['model'] == 'Particles_A'):
        model = InteractionParticles(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
        p_mass = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_mass[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
        p_elec = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            p_elec[n] = torch.load(f'graphs_data/graphs_particles_{dataset_name}/p_{n}.pt')
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = torch.concatenate((T1[:, None], T1[:, None]), 1)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if bMesh:

        p = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        if len(model_config['p']) > 0:
            for n in range(nparticle_types):
                p[n] = torch.tensor(model_config['p'][n])
        model = Particles_G(aggr_type=aggr_type, p=torch.squeeze(p), tau=model_config['tau'],
                            clamp=model_config['clamp'], pred_limit=model_config['pred_limit'])

        c = torch.ones(nparticle_types, 1, device=device) + torch.rand(nparticle_types, 1, device=device)
        for n in range(nparticle_types):
            c[n] = torch.tensor(model_config['c'][n])
        model_mesh = MeshDiffusion(model_config, device)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))
    if best_model == -1:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    else:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"
    if bPrint:
        print('Graph files N: ', NGraphs - 1)
        print(f'network: {net}')
    if bTest!='integration':
        if  bMesh:
            state_dict = torch.load(net, map_location=device)
            model_mesh.load_state_dict(state_dict['model_state_dict'])
            model_mesh.eval()
        else:
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

    if new_nparticles > 0:  # nparticles larger than initially

        ratio_particles = int(new_nparticles / prev_nparticles)
        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio_particles}')
        print('')

        if ratio_particles>1:
            embedding = model.a.data.clone().detach()
            new_embedding = []

            for n in range(nparticle_types):
                for m in range(ratio_particles):
                    if (n == 0) & (m == 0):
                        new_embedding = embedding[0,prev_index_particles[n]]
                    else:
                        new_embedding = torch.cat((new_embedding, embedding[0,prev_index_particles[n]]), axis=0)

            model.a = nn.Parameter(
                torch.tensor(np.ones((model.ndataset,int(prev_nparticles) * ratio_particles, model_config['embedding'])), device=device, requires_grad=False, dtype=torch.float32))
            model.a.data[0] = new_embedding.float()
            nparticles = new_nparticles
            model_config['nparticles'] = new_nparticles

    ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{ntry}/vnorm.pt', map_location=device).to(device)
    if bMesh:
        hnorm = torch.load(f'./log/try_{ntry}/hnorm.pt', map_location=device).to(device)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if bPrint:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    print('log_dir: {}'.format(log_dir))

    x_recons=[]
    y_recons=[]
    x_list=[]
    y_list=[]
    for run in range(2):
        x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt',map_location=device)
        y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt',map_location=device)
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))

    x = x_list[0][0].clone().detach()
    x00 = x_list[0][0].clone().detach()

    if bMesh:
        index_particles = []
        T1 = []
        for n in range(model_config['nparticle_types']):
            index=np.argwhere(x[:,5].detach().cpu().numpy()==n)
            index_particles.append(index.squeeze())

    if 'Boids' in model_config['description']:
        T1 = torch.zeros(int(nparticles / nparticle_types), device=device)
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types), device=device)), 0)
        T1 = T1[:, None]
        model_a_ = model.a[0].clone().detach()
        t = []
        for k in range(model_config['ninteractions']):
            pos = np.argwhere(T1.detach().cpu().numpy() == k).squeeze().astype(int)
            temp = model_a_[pos[:,0], :].clone().detach()
            print(torch.median(temp, axis=0).values)
            model_a_[pos[:,0], :] = torch.median(temp, axis=0).values.repeat((len(pos), 1))
            t.append(torch.median(temp, axis=0).values)
        if initial_map !='':
            i0 = imread(f'graphs_data/{initial_map}')
            values = i0[(x[:, 1].detach().cpu().numpy() * 255).astype(int), (x[:, 2].detach().cpu().numpy() * 255).astype(int)]
            T1 = torch.tensor(values, device=device)
            T1 = T1[:, None]
        for k in range(model_config['ninteractions']):
            pos = np.argwhere(T1.detach().cpu().numpy() == k).squeeze().astype(int)
            with torch.no_grad():
                model.a[0,pos[:,0], :] = t[k]

        fps = 60
        scale = 40
        Distance = 5
        speed = 0.0005
        size = 1000
        flock = []
        for i in range(nparticles):
            flock.append(Boid(np.random.randint(20, size - 20), np.random.randint(20, size - 20)))

    if bPrint:
        print('')
        print(f'x: {x.shape}')
        print(f'index_particles: {index_particles[0].shape}')
        print('')
    time.sleep(0.5)

    rmserr_list = []
    discrepency_list = []

    for it in tqdm(range(nframes - 1)):

        x0 = x_list[0][it].clone().detach()
        x0_next = x_list[0][it+1].clone().detach()
        y0 = y_list[0][it].clone().detach()

        if (it%10==0) & (bTest=='prediction'):
            x[:, 1:5] = x0[:, 1:5].clone().detach()

        if model_config['model'] == 'DiffMesh':
            x[:,1:5]=x0[:,1:5].clone().detach()
            dataset = data.Data(x=x, pos=x[:, 1:3])
            transform_0 = T.Compose([T.Delaunay()])
            dataset_face = transform_0(dataset).face
            mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
            edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)
            dataset_mesh = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)
            with torch.no_grad():
                pred = model_mesh(dataset_mesh, data_id=0,)
            x[:,6:7] += pred * hnorm
        elif model_config['model'] == 'WaveMesh':
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset = data.Data(x=x, pos=x[:, 1:3])
            transform_0 = T.Compose([T.Delaunay()])
            dataset_face = transform_0(dataset).face
            mesh_pos = torch.cat((x[:, 1:3], torch.ones((x.shape[0], 1), device=device)), dim=1)
            edge_index, edge_weight = pyg_utils.get_mesh_laplacian(pos=mesh_pos, face=dataset_face)
            dataset_mesh = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight, device=device)
            with torch.no_grad():
                pred = model_mesh(dataset_mesh, data_id=0, )
            x[:, 7:8] += pred * hnorm
            x[:, 6:7] += x[:, 7:8]
        else:
            distance = torch.sum(bc_diff(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            dataset = data.Data(x=x, edge_index=edge_index)

            if bTest == 'integration':
                if model_config['prediction'] == '2nd_derivative':
                    y = y0 / ynorm[4]
                else:
                    y = y0 / vnorm[4]
            else:
                with torch.no_grad():
                    y = model(dataset, data_id=0, step=2, vnorm=vnorm, cos_phi=0, sin_phi=0)  # acceleration estimation

            if model_config['prediction'] == '2nd_derivative':
                y = y * ynorm[4]
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm[4]
                x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5])  # position update

            x_recons.append(x.clone().detach())
            y_recons.append(y.clone().detach())

        if bMesh:
            mask = torch.argwhere((x[:, 1] < 0.025) | (x[:, 1] > 0.975) | (x[:, 2] < 0.025) | (x[:, 2] > 0.975)).detach().cpu().numpy().astype(int)
            mask = mask[:, 0:1]
            x[mask, 6:8]=0
            rmserr = torch.sqrt(torch.mean(torch.sum((x[:, 6:7] - x0_next[:, 6:7]) ** 2, axis=1)))
            rmserr_list.append(rmserr.item())
        else:
            rmserr = torch.sqrt(torch.mean(torch.sum(bc_diff(x[:, 1:3] - x0_next[:, 1:3]) ** 2, axis=1)))
            rmserr_list.append(rmserr.item())
            discrepency = MMD(x[:, 1:3], x0[:, 1:3])
            discrepency_list.append(discrepency)

        if (it % step == 0) & (it>=0) &  bVisu:

            if bMesh:
                dataset2 = dataset_mesh
            else:
                distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, axis=2)
                adj_t2 = ((distance2 < radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                edge_index2 = adj_t2.nonzero().t().contiguous()
                dataset2 = data.Data(x=x, edge_index=edge_index2)

            fig = plt.figure(figsize=(21, 8))
            # plt.ion()

            for k in range(5):
                if k==0:
                    ax = fig.add_subplot(2, 5, 1)
                    x_ = x00
                    sc = 1
                elif k == 1:
                    ax = fig.add_subplot(2, 5, 2)
                    x_ = x0
                    sc = 1
                elif k == 2:
                    ax = fig.add_subplot(2, 5, 7)
                    x_ = x
                    sc = 1
                elif k == 3:
                    ax = fig.add_subplot(2, 5, 3)
                    x_ = x0
                    sc = 5
                elif k == 4:
                    ax = fig.add_subplot(2, 5, 8)
                    x_ = x
                    sc = 5

                if ((k==1)|(k==2))&('Boids' in model_config['description']):
                    for n, boid in enumerate(flock):
                        boid.position.x = x_[n, 1].detach().cpu().numpy() * 1000
                        boid.position.y = x_[n, 2].detach().cpu().numpy() * 1000
                        boid.velocity.x = x_[n, 3].detach().cpu().numpy() * 1000
                        boid.velocity.y = x_[n, 4].detach().cpu().numpy() * 1000
                        boid.angle = np.arctan(-boid.velocity.x / (boid.velocity.y+1E-10))+np.pi
                        if boid.velocity.y<0:
                            boid.angle=boid.angle + np.pi
                        ps = boid.Draw(Distance, scale)
                        ps = np.array(ps)
                        plt.plot(ps[:, 0], ps[:, 1], c=cmap.color(T1[n].detach().cpu().numpy()), alpha=0.5)
                elif (k==0) & (bMesh):
                    plt.scatter(x0_next[:, 6].detach().cpu().numpy(),x[:, 6].detach().cpu().numpy(),s=1, alpha=0.25, c='k')
                    plt.xlabel('True temperature [a.u.]', fontsize="14")
                    plt.ylabel('Model temperature [a.u]', fontsize="14")
                elif model_config['model'] == 'GravityParticles':
                    for n in range(nparticle_types):
                        g = p_mass[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 10 * sc
                        plt.scatter(x_[index_particles[n], 1].detach().cpu(), x_[index_particles[n], 2].detach().cpu(), s=g, alpha=0.75,color=cmap.color(n))  # , facecolors='none', edgecolors='k')
                elif model_config['model'] == 'ElecParticles':
                    for n in range(nparticle_types):
                        g = np.abs(p_elec[T1[index_particles[n], 0].detach().cpu().numpy()].detach().cpu().numpy() * 20)*sc
                        if model_config['p'][n][0]<=0:
                            plt.scatter(x_[index_particles[n], 1].detach().cpu().numpy(),
                                        x_[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                        c='r',alpha=0.5)  # , facecolors='none', edgecolors='k')
                        else:
                            plt.scatter(x_[index_particles[n], 1].detach().cpu().numpy(),
                                        x_[index_particles[n], 2].detach().cpu().numpy(), s=g,
                                        c='b',alpha=0.5)  # , facecolors='none', edgecolors='k')
                elif bMesh:
                    pts = x_[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = torch.sum(x_[tri.simplices, 6], axis=1) / 3.0
                    if model_config['model'] == 'WaveMesh':
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(), facecolors=colors.detach().cpu().numpy(),edgecolors='k',vmin=-5000,vmax=5000)
                    else:
                        plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                      facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=5000)
                else:
                    for n in range(nparticle_types):
                        plt.scatter(x_[index_particles[n], 1].detach().cpu(), x_[index_particles[n], 2].detach().cpu(),s=sc,color=cmap.color(n))
                if (k > 2) & (bMesh==False):
                    for n in range(nparticles):
                        plt.arrow(x=x_[n, 1].detach().cpu().item(),y=x_[n, 2].detach().cpu().item(),
                                  dx=x_[n, 3].detach().cpu().item()*model_config['arrow_length'], dy=x_[n, 4].detach().cpu().item()*model_config['arrow_length'],color='k')
                if k<3:
                    if ((k==1)|(k==2))&('Boids' in model_config['description']):
                        plt.xlim([0,size])
                        plt.ylim([0,size])
                    elif (k == 0) & (bMesh):
                        plt.xlim([-5000, 5000])
                        plt.ylim([-5000, 5000])
                    elif (model_config['boundary'] == 'no'):
                        plt.xlim([-1.3, 1.3])
                        plt.ylim([-1.3, 1.3])
                    else:
                        plt.xlim([0,1])
                        plt.ylim([0,1])
                else:
                    if bMesh | ('Boids' in model_config['description']):
                        plt.xlim([0.3, 0.7])
                        plt.ylim([0.3, 0.7])
                    else:
                        plt.xlim([-0.25, 0.25])
                        plt.ylim([-0.25, 0.25])

            ax = fig.add_subplot(2, 5, 4)
            plt.plot(np.arange(len(rmserr_list)), rmserr_list, label='RMSE', c='k')
            plt.ylim([0, 0.1])
            plt.xlim([0, nframes])
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.xlabel('Frame [a.u]', fontsize="14")
            ax.set_ylabel('RMSE [a.u]', fontsize="14", color='k')
            if bMesh:
                plt.ylim([0, 5000])
            else:
                ax2 = ax.twinx()
                plt.plot(np.arange(len(discrepency_list)), discrepency_list,label='Maximum Mean Discrepencies', c='b')
                ax2.set_ylabel('MMD [a.u]', fontsize="14", color='b')
                ax2.set_ylim([0, 2E-3])

            # ax = fig.add_subplot(2, 5, 5)
            # plt.scatter(y0[:,0].detach().cpu().numpy(),y[:,0].detach().cpu().numpy(),s=1,color='b')
            # plt.scatter(y0[:,1].detach().cpu().numpy(),y[:,1].detach().cpu().numpy(), s=1, color='r')
            # plt.xlabel('Y true [a.u]', fontsize="14")
            # plt.ylabel('Y pred [a.u]', fontsize="14")
            # if model_config['prediction']=='2nd_derivative':
            #     plt.xlim([-ynorm[4].detach().cpu().numpy(),ynorm[4].detach().cpu().numpy()])
            #     plt.ylim([-ynorm[4].detach().cpu().numpy(), ynorm[4].detach().cpu().numpy()])
            # else:
            #     plt.xlim([-ynorm[4].detach().cpu().numpy(),ynorm[4].detach().cpu().numpy()])
            #     plt.ylim([-ynorm[4].detach().cpu().numpy(), ynorm[4].detach().cpu().numpy()])
            #
            # ax = fig.add_subplot(2, 5, 10)
            # plt.hist(y0[:,0].detach().cpu().numpy(),200,alpha=0.25,color='b')
            # plt.hist(y[:,0].detach().cpu().numpy(),200,alpha=0.25,color='r')
            # if model_config['prediction']=='2nd_derivative':
            #     plt.xlim([-ynorm[4].detach().cpu().numpy(),ynorm[4].detach().cpu().numpy()])
            # else:
            #     plt.xlim([-ynorm[4].detach().cpu().numpy(),ynorm[4].detach().cpu().numpy()])

            ax = fig.add_subplot(2, 5, 6)
            pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
            vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.2)
            if model_config['boundary'] == 'no':
                plt.xlim([-1.3, 1.3])
                plt.ylim([-1.3, 1.3])
            else:
                plt.xlim([0,1])
                plt.ylim([0,1])

            ax = fig.add_subplot(2, 5, 9)

            if not(bMesh):
                temp1 = torch.cat((x, x0_next), 0)
                temp2 = torch.tensor(np.arange(nparticles), device=device)
                temp3 = torch.tensor(np.arange(nparticles) + nparticles, device=device)
                temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
                temp4 = torch.t(temp4)
                distance3 = torch.sqrt(torch.sum((x[:, 1:3] - x0_next[:, 1:3]) ** 2, 1))
                p = torch.argwhere(distance3 < 0.3)
                pos = dict(enumerate(np.array((temp1[:, 1:3]).detach().cpu()), 0))
                dataset = data.Data(x=temp1[:, 1:3], edge_index=torch.squeeze(temp4[:, p]))
                vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False)
                if model_config['boundary'] == 'no':
                    plt.xlim([-1.3, 1.3])
                    plt.ylim([-1.3, 1.3])
                else:
                    plt.xlim([0,1])
                    plt.ylim([0,1])

            plt.tight_layout()

            plt.savefig(f"./{folder_out}/Fig_{ntry}_{it}.tif")

            plt.close()

    print(f'RMSE: {np.round(rmserr.item(), 4)}')
    if bPrint:
        print(f'ntry: {ntry}')
        # print(f'MMD: {np.round(discrepency, 4)}')

    torch.save(x_recons, f'{log_dir}/x_list.pt')
    torch.save(y_recons, f'{log_dir}/y_list.pt')

    return x.detach().cpu().numpy(), rmserr_list
def data_test_generate(model_config, bVisu=True, bDetails=False, step=5):


    # scenario A
    # X1[:, 0] = X1[:, 0] / nparticle_types
    # for n in range(nparticle_types):
    #     X1[index_particles[n], 0] = X1[index_particles[n], 0] + n / nparticle_types

    # scenario B
    # X1[index_particles[0], :] = X1[index_particles[0], :]/2 + 1/4

    # scenario C
    # i0 = imread('graphs_data/pattern_1.tif')
    # pos = np.argwhere(i0 == 255)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[0],:] = torch.tensor(pos[l[index_particles[0]],:]/255,dtype=torch.float32,device=device)
    # pos = np.argwhere(i0 == 0)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[1],:] = torch.tensor(pos[l[index_particles[0]],:]/255,dtype=torch.float32,device=device)

    # scenario D
    # i0 = imread('graphs_data/pattern_3.tif')
    # pos = np.argwhere(i0 == 255)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[0],:] = torch.tensor(pos[l[0:1600*3],:]/255,dtype=torch.float32,device=device)
    # pos = np.argwhere(i0 == 128)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[1],:] = torch.tensor(pos[l[0:1600*3],:]/255,dtype=torch.float32,device=device)
    # pos = np.argwhere(i0 == 0)
    # l = np.arange(pos.shape[0])
    # l = np.random.permutation(l)
    # X1[index_particles[2],:] = torch.tensor(pos[l[0:1600*3],:]/255,dtype=torch.float32,device=device)

    # scenario E
    # i0 = imread('graphs_data/pattern_5.tif')
    # values=i0[(X1[:,0].detach().cpu().numpy()*256).astype(int),(X1[:,1].detach().cpu().numpy()*256).astype(int)]
    # H1 = torch.tensor(values/255*1.5,device=device)
    # H1 = H1[:,None]


    return prev_nparticles, new_nparticles, prev_index_particles, index_particles
def data_plot(model_config, epoch, bPrint, best_model=0):
    print('')

    # for loop in range(25):
    #     print(f'Loop: {loop}')

    model = []
    ntry = model_config['ntry']
    radius = model_config['radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

    graph_files = glob.glob(f"graphs_data/graphs_particles_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print('Graph files N: ', NGraphs - 1)
    time.sleep(0.5)

    # arr = np.arange(0, NGraphs)
    # x_list=[]
    # y_list=[]
    # for run in arr:
    #     x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt')
    #     y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt')
    #     x_list.append(torch.stack(x))
    #     y_list.append(torch.stack(y))
    # x = torch.stack(x_list)
    # x = torch.reshape(x,(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    # y = torch.stack(y_list)
    # y = torch.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3]))
    # vnorm = norm_velocity(x, device)
    # ynorm = norm_acceleration(y, device)
    # torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    # torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    # print (vnorm,ynorm)

    x_list = []
    y_list = []
    x_stat = []
    y_stat = []
    distance_list = []
    deg_list = []
    if False:        # analyse tmp_recons
        x = torch.load(f'{log_dir}/x_list.pt')
        y = torch.load(f'{log_dir}/y_list.pt')
        for k in np.arange(0, len(x) - 1, 4):
            distance = torch.sum(bc_diff(x[k][:, None, 1:3] - x[k][None, :, 1:3]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, edge_index=edge_index)
            distance = np.sqrt(distance[edge_index[0, :], edge_index[1, :]].detach().cpu().numpy())
            deg = degree(dataset.edge_index[0], dataset.num_nodes)
            deg_list.append(deg.detach().cpu().numpy())
            distance_list.append([np.mean(distance), np.std(distance)])
            x_stat.append(torch.concatenate((torch.mean(x[k][:, 3:5], axis=0), torch.std(x[k][:, 3:5], axis=0)),axis=-1).detach().cpu().numpy())
            y_stat.append(torch.concatenate((torch.mean(y[k], axis=0), torch.std(y[k], axis=0)), axis=-1).detach().cpu().numpy())
        x_list.append(torch.stack(x))
        y_list.append(torch.stack(y))
    else:
        for run in tqdm(range(NGraphs)):
            x = torch.load(f'graphs_data/graphs_particles_{dataset_name}/x_list_{run}.pt',map_location=device)
            y = torch.load(f'graphs_data/graphs_particles_{dataset_name}/y_list_{run}.pt',map_location=device)
            if run==0:
                for k in np.arange(0, len(x) - 1, 4):

                    distance = torch.sum(bc_diff(x[k][:, None, 1:3] - x[k][None, :, 1:3]) ** 2, axis=2)
                    t = torch.Tensor([radius ** 2])  # threshold
                    adj_t = (distance < radius ** 2).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x, edge_index=edge_index)
                    distance = np.sqrt(distance[edge_index[0, :], edge_index[1, :]].detach().cpu().numpy())
                    deg = degree(dataset.edge_index[0], dataset.num_nodes)
                    deg_list.append(deg.detach().cpu().numpy())
                    distance_list.append([np.mean(distance), np.std(distance)])
                    x_stat.append(torch.concatenate((torch.mean(x[k][:, 3:5], axis=0), torch.std(x[k][:, 3:5], axis=0)),axis=-1).detach().cpu().numpy())
                    y_stat.append(torch.concatenate((torch.mean(y[k], axis=0), torch.std(y[k], axis=0)), axis=-1).detach().cpu().numpy())
            x_list.append(torch.stack(x))
            y_list.append(torch.stack(y))

    x = torch.stack(x_list)
    x = torch.reshape(x,(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
    y = torch.stack(y_list)
    y = torch.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3]))
    vnorm = norm_velocity(x, device)
    ynorm = norm_acceleration(y, device)
    print(vnorm, ynorm)
    print(vnorm[4], ynorm[4])

    x_stat = np.array(x_stat)
    y_stat = np.array(y_stat)

    fig = plt.figure(figsize=(20, 5))
    plt.ion()
    ax = fig.add_subplot(1, 5, 4)

    deg_list = np.array(deg_list)
    distance_list = np.array(distance_list)
    plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0] + deg_list[:, 1], c='k')
    plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0], c='r')
    plt.plot(np.arange(deg_list.shape[0]) * 4, deg_list[:, 0] - deg_list[:, 1], c='k')
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Degree [a.u]', fontsize="14")
    ax = fig.add_subplot(1, 5, 1)
    plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0] + distance_list[:, 1], c='k')
    plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0], c='r')
    plt.plot(np.arange(distance_list.shape[0]) * 4, distance_list[:, 0] - distance_list[:, 1], c='k')
    plt.ylim([0, model_config['radius']])
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Distance [a.u]', fontsize="14")
    ax = fig.add_subplot(1, 5, 2)
    plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0] + x_stat[:, 2], c='k')
    plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0], c='r')
    plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 0] - x_stat[:, 2], c='k')
    plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1] + x_stat[:, 3], c='k')
    plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1], c='r')
    plt.plot(np.arange(x_stat.shape[0]) * 4, x_stat[:, 1] - x_stat[:, 3], c='k')
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Velocity [a.u]', fontsize="14")
    ax = fig.add_subplot(1, 5, 3)
    plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0] + y_stat[:, 2], c='k')
    plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0], c='r')
    plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 0] - y_stat[:, 2], c='k')
    plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1] + y_stat[:, 3], c='k')
    plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1], c='r')
    plt.plot(np.arange(y_stat.shape[0]) * 4, y_stat[:, 1] - y_stat[:, 3], c='k')
    plt.xlim([0, nframes])
    plt.xlabel('Frame [a.u]', fontsize="14")
    plt.ylabel('Acceleration [a.u]', fontsize="14")
    plt.tight_layout()
    plt.show()

    if bMesh:
        h_list=[]
        for run in tqdm(range(NGraphs)):
            h = torch.load(f'graphs_data/graphs_particles_{dataset_name}/h_list_{run}.pt',map_location=device)
            h_list.append(torch.stack(h))
        h = torch.stack(h_list)
        h = torch.reshape(h, (h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
        hnorm = torch.std(h)
        torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
        print(hnorm)
        model = MeshDiffusion(model_config, device)
    if model_config['model'] == 'GravityParticles':
        model = GravityParticles(model_config, device)
    if model_config['model'] == 'ElecParticles':
        model = ElecParticles(model_config, device)
    if (model_config['model'] == 'Particles_A'):
        model = InteractionParticles(model_config, device)
        print(f'Training InteractionParticles')

    if best_model==-1:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs-1}_graphs.pt"
    else:
        net = f"./log/try_{ntry}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"
    state_dict = torch.load(net,map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    lra = 1E-3
    lr = 1E-3

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    it = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if it == 0:
            optimizer = torch.optim.Adam([model.a], lr=lra)
        else:
            optimizer.add_param_group({'params': parameter, 'lr': lr})
        it += 1
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f'Learning rates: {lr}, {lra}')
    print('')
    print(f'network: {net}')
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=weight_decay)
    model.eval()
    best_loss = np.inf

    print('')
    time.sleep(0.5)
    print('Plotting ...')

    fig = plt.figure(figsize=(16, 8))
    plt.ion()

    if bMesh:
        x = x_list[0][0].clone().detach()
        index_particles = []
        for n in range(model_config['nparticle_types']):
            index=np.argwhere(x[:,5].detach().cpu().numpy()==n)
            index_particles.append(index.squeeze())
    embedding = []
    for n in range(model.a.shape[0]):
        embedding.append(model.a[n])
    embedding = torch.stack(embedding).detach().cpu().numpy()
    embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
    embedding_particle = []
    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            embedding_particle.append(embedding[index_particles[n] + m * nparticles, :])
    ax = fig.add_subplot(2, 4, 2)
    if (embedding.shape[1] > 2):
        ax = fig.add_subplot(2, 4, 2, projection='3d')
        for n in range(nparticle_types):
            ax.scatter(embedding_particle[n][:, 0], embedding_particle[n][:, 1], embedding_particle[n][:, 2],
                       color=cmap.color(n), s=1)   #
    else:
        if (embedding.shape[1] > 1):
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types):
                    plt.scatter(embedding_particle[n + m * nparticle_types][:, 0],embedding_particle[n + m * nparticle_types][:, 1], color=cmap.color(n), s=3)
            plt.xlabel('Embedding 0', fontsize=12)
            plt.ylabel('Embedding 1', fontsize=12)
        else:
            for n in range(nparticle_types):
                plt.hist(embedding_particle[n][:, 0], 100, alpha=0.5, color=cmap.color(n))

    rr = torch.tensor(np.linspace(0, radius, 1000)).to(device)
    ax = fig.add_subplot(2, 4, 3)
    if model_config['model'] == 'ElecParticles':
        acc_list = []
        for m in range(model.a.shape[0]):
            for k in range(nparticle_types):
                for n in index_particles[k]:
                    embedding = model.a[m, n, :] * torch.ones((1000, model_config['embedding']), device=device)
                    in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                             rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding, embedding), dim=1)
                    acc = model.lin_edge(in_features.float())
                    acc = acc[:, 0]
                    acc_list.append(acc)
                    if n % 5 == 0:
                        plt.plot(rr.detach().cpu().numpy(),
                                 acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                                 linewidth=1,
                                 color=cmap.color(k ), alpha=0.25)
        acc_list = torch.stack(acc_list)
        plt.xlim([0, 0.05])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
        coeff_norm = acc_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                          random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)
    elif model_config['model'] == 'GravityParticles':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            plt.plot(rr.detach().cpu().numpy(),
                     acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                     color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
        acc_list = torch.stack(acc_list)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1E-3, 0.2])
        plt.ylim([1, 1E7])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
        coeff_norm = acc_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                          random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)
    elif model_config['model'] == 'Particles_A':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            if model_config['prediction'] == '2nd_derivative':
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            else:
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            if n % 5 == 0:
                plt.plot(rr.detach().cpu().numpy(),
                         acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                         color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
        acc_list = torch.stack(acc_list)
        coeff_norm = acc_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,
                          random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)
    elif bMesh:
        h_list = []
        for n in range(nparticles):
            r0 = torch.tensor(np.linspace(4, 5, 1000)).to(device)
            r1 = torch.tensor(np.linspace(-250, 250, 1000)).to(device)
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((r0[:, None], r1[:, None], embedding), dim=1)
            h = model.lin_edge(in_features.float())
            h = h[:, 0]
            h_list.append(h)
            if n % 5 == 0:
                plt.plot(r1.detach().cpu().numpy(),h.detach().cpu().numpy() * hnorm.detach().cpu().numpy(), linewidth=1,color='k', alpha=0.05)
        h_list = torch.stack(h_list)
        coeff_norm = h_list.detach().cpu().numpy()
        trans = umap.UMAP(n_neighbors=np.round(nparticles / model_config['ninteractions']).astype(int), n_components=2,random_state=42, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
        proj_interaction = np.squeeze(proj_interaction)

    ax = fig.add_subplot(2, 4, 4)

    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=1000, max_iter=10000,random_state=13)
    kmeans.fit(proj_interaction)
    for n in range(nparticle_types):
        tmp=kmeans.labels_[index_particles[n]]
        sub_group = np.round(np.median(tmp))
        accuracy=len(np.argwhere(tmp==sub_group))/len(tmp)*100
        print(f'Sub-group {n} accuracy: {np.round(accuracy,3)}')

    for n in range(nparticle_types):
        if proj_interaction.ndim == 1:
            plt.hist(proj_interaction[index_particles[n]], width=0.01, alpha=0.5, color=cmap.color(n))
        if proj_interaction.ndim==2:
            plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                        color=cmap.color(n), s=5)
            plt.xlabel('UMAP 0', fontsize=12)
            plt.ylabel('UMAP 1', fontsize=12)

    for n in range(model_config['ninteractions']):
        plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)

    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    t=[]
    for k in range(model_config['ninteractions']):
        pos = np.argwhere(kmeans.labels_ == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        print(torch.median(temp, axis=0).values)
        model_a_[pos, :] = torch.median(temp, axis=0).values.repeat((len(pos),1))
        t.append(torch.median(temp, axis=0).values)
    model_a_ = torch.reshape(model_a_, (model.a.shape[0], model.a.shape[1], model.a.shape[2]))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_[0]
    embedding = []
    for n in range(model.a.shape[0]):
        embedding.append(model.a[n])
    embedding = torch.stack(embedding).detach().cpu().numpy()
    embedding = np.reshape(embedding, [embedding.shape[0] * embedding.shape[1], embedding.shape[2]])
    embedding_particle = []
    for m in range(model.a.shape[0]):
        for n in range(nparticle_types):
            embedding_particle.append(embedding[index_particles[n] + m * nparticles, :])

    ax = fig.add_subplot(2, 4, 6)
    if (embedding.shape[1] > 2):
        ax = fig.add_subplot(2, 4, 6, projection='3d')
        for m in range(model.a.shape[0]):
            for n in range(nparticle_types):
                ax.scatter(model.a[m][index_particles[n], 0].detach().cpu().numpy(), model.a[m][index_particles[n], 1].detach().cpu().numpy(), model.a[m][index_particles[n], 1].detach().cpu().numpy(),
                           color=cmap.color(n), s=20)
    else:
        if (embedding.shape[1] > 1):
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types-1,-1,-1):
                    plt.scatter(model.a[m][index_particles[n], 0].detach().cpu().numpy(),model.a[m][index_particles[n], 1].detach().cpu().numpy(),
                                color=cmap.color(n), s=20)
            plt.xlabel('Embedding 0', fontsize=12)
            plt.ylabel('Embedding 1', fontsize=12)
        else:
            for m in range(model.a.shape[0]):
                for n in range(nparticle_types-1,-1,-1):
                    plt.hist(model.a[m][index_particles[n], 0].detach().cpu().numpy(), width=0.01, alpha=0.5, color=cmap.color(n))

    ax = fig.add_subplot(2, 4, 7)
    if model_config['model'] == 'ElecParticles':
        t = model.a.detach().cpu().numpy()
        tmean = np.ones((model_config['nparticle_types'],model_config['embedding']))
        for n in range(model_config['nparticle_types']):
            tmean[n] = np.mean(t[:,index_particles[n],:],axis=(0,1))
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                embedding0 = torch.tensor(tmean[m],device=device) * torch.ones((1000, model_config['embedding']), device=device)
                embedding1 = torch.tensor(tmean[n],device=device) * torch.ones((1000, model_config['embedding']), device=device)
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding0, embedding1), dim=1)
                acc = model.lin_edge(in_features.float())
                acc = acc[:, 0]
                plt.plot(rr.detach().cpu().numpy(),acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],linewidth=1,color='k')
        plt.xlim([0,0.02])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    elif model_config['model'] == 'GravityParticles':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                     rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            plt.plot(rr.detach().cpu().numpy(),
                     acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                     color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
        acc_list = torch.stack(acc_list)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1E-3, 0.2])
        plt.ylim([1, 1E7])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    elif model_config['model'] == 'Particles_A':
        acc_list = []
        for n in range(nparticles):
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            if model_config['prediction'] == '2nd_derivative':
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            else:
                in_features = torch.cat((-rr[:, None] / model_config['radius'], 0 * rr[:, None],
                                         rr[:, None] / model_config['radius'], embedding), dim=1)
            acc = model.lin_edge(in_features.float())
            acc = acc[:, 0]
            acc_list.append(acc)
            if n % 5 == 0:
                plt.plot(rr.detach().cpu().numpy(),
                         acc.detach().cpu().numpy() * ynorm[4].detach().cpu().numpy() / model_config['tau'],
                         color=cmap.color(x[n, 5].detach().cpu().numpy() ), linewidth=1, alpha=0.25)
    elif bMesh:
        for n in range(nparticles):
            r0 = torch.tensor(np.linspace(4, 5, 1000)).to(device)
            r1 = torch.tensor(np.linspace(-100, 100, 1000)).to(device)
            embedding = model.a[0, n, :] * torch.ones((1000, model_config['embedding']), device=device)
            in_features = torch.cat((r0[:, None], r1[:, None], embedding), dim=1)
            h = model.lin_edge(in_features.float())
            h = h[:, 0]
            if n % 5 == 0:
                plt.plot(r1.detach().cpu().numpy(),h.detach().cpu().numpy() * hnorm.detach().cpu().numpy(), linewidth=1,color='k', alpha=0.05)

    plt.xlabel('Distance [a.u]', fontsize=12)
    plt.ylabel('MLP [a.u]', fontsize=12)

    ax = fig.add_subplot(2, 4, 8)
    if model_config['model'] == 'Particles_A':
        p = model_config['p']
        p = torch.tensor(p,device=device)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, p[n]))
        for n in range(nparticle_types-1,-1,-1):
            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1)
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    if model_config['model'] == 'GravityParticles':
        p = model_config['p']
        p = torch.tensor(p,device=device)
        psi_output = []
        for n in range(nparticle_types):
            psi_output.append(model.psi(rr, p[n]))
        for n in range(nparticle_types-1,-1,-1):
            plt.plot(rr.detach().cpu().numpy(), np.array(psi_output[n].cpu()), linewidth=1, color=cmap.color(n))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1E-3, 0.2])
        plt.ylim([1, 1E7])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)
    if model_config['model'] == 'ElecParticles':
        p = model_config['p']
        p = torch.tensor(p,device=device)
        psi_output = []
        for m in range(nparticle_types):
            for n in range(nparticle_types):
                temp=model.psi(rr, p[n],p[m])
                plt.plot(rr.detach().cpu().numpy(), np.array(temp.cpu()), linewidth=1,c='k')
        plt.xlim([0, 0.02])
    if bMesh:
        for n in range(nparticle_types):
            plt.scatter(x[index_particles[n],1].detach().cpu().numpy(), x[index_particles[n],2].detach().cpu().numpy(), color=cmap.color(kmeans.labels_[index_particles[n]]), s=10)

    plt.tight_layout()
    plt.show()

def load_model_config(id=48):
    model_config_test = []

    if id == 44:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': -1000,
                             'arrow_length':10,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'description':'Gravity',
                             'sparsity':'replace'}
    if id == 45:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 2000,
                             'sigma': .005,
                             'tau': 1E-9/100,
                             'v_init': 5E-5/100,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': -1000,
                             'cmap':'tab10',
                             'arrow_length':100,
                             'description':'Gravity',
                             'sparsity':'replace'}
    if id == 46:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 2000,
                             'sigma': .005,
                             'tau': 1E-9/100,
                             'v_init': 5E-5/100,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': -1000,
                             'cmap':'tab10',
                             'arrow_length':100,
                             'description':'Gravity',
                             'sparsity':'replace'}
    if id == 47:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_44',
                             'nparticles': 960,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'GravityParticles',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'nrun': 2,
                             'clamp': 0.002,
                             'pred_limit': 1E9,
                             'start_frame': -1000,
                             'arrow_length':10,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'description':'Gravity sparsity regul_1E-4',
                             'sparsity':'regul_1E-4'}

    # particles
    if id == 74:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_74',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': 'first_derivative_S',
                             'upgrade_type': 'none',
                             'p': [[1.0413, 1.5615, 1.6233, 1.6012], [1.8308, 1.9055, 1.7667, 1.0855],
                                   [1.785, 1.8579, 1.7226, 1.0584]],
                             'nrun': 2,
                             'start_frame': 20,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description': 'regul_1E-4 3 interaction particles',
                             'sparsity':'replace'}
    if id == 75:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': [[1.0413, 1.5615, 1.6233, 1.6012], [1.8308, 1.9055, 1.7667, 1.0855],
                                   [1.785, 1.8579, 1.7226, 1.0584]],
                             'nrun': 2,
                             'start_frame': 20,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description': 'pred=first derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14',
                             'sparsity':'replace'}
    if id == 76:
        model_config_test = {'ntry': id,
                             'input_size': 8,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': [[1.0413, 1.5615, 1.6233, 1.6012], [1.8308, 1.9055, 1.7667, 1.0855],
                                   [1.785, 1.8579, 1.7226, 1.0584]],
                             'nrun': 2,
                             'start_frame': 20,
                             'cmap':'tab10',
                             'arrow_length':20,
                             'description': 'pred=second derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14',
                             'sparsity':'replace'}
    if id == 77:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 8,
                             'ninteractions': 8,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': 'first_derivative_S',
                             'upgrade_type': 'none',
                             'p': [[1.2425, 1.3355, 1.3397, 1.3929], [1.629, 1.4932, 1.5311, 1.8677],[1.9852, 1.1892, 1.1544, 1.993],[1.6898, 1.1336, 1.4869, 1.7767],[1.8847, 1.5448, 1.8063, 1.3873],[1.496, 1.4064, 1.9045, 1.733],[1.5108, 1.9904, 1.1665, 1.6975],[1.6153, 1.8557, 1.2758, 1.0684]],
                             'nrun': 2,
                             'start_frame': 0,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description': '8 types pred=first derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14',
                             'sparsity':'replace'}
    if id == 78:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 8,
                             'ninteractions': 8,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': 'first_derivative_S',
                             'upgrade_type': 'none',
                             'p': [],
                             'nrun': 2,
                             'start_frame': 0,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description': '8 types pred=first derivative Particles_A is a first derivative simulation, interaction is function of r.exp-r^2 interaction is type dependent best_model:14',
                             'sparsity':'replace'}
    if id == 79:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 2,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.075,
                             'dataset': f'231001_{id}',
                             'nparticles': 4800,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 200,
                             'sigma': .005,
                             'tau': 0.1,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 1,
                             'model': 'Particles_A',
                             'prediction': 'first_derivative_S',
                             'upgrade_type': 'none',
                             'p': [[1.0413, 1.5615, 1.6233, 1.6012], [1.8308, 1.9055, 1.7667, 1.0855],
                                   [1.785, 1.8579, 1.7226, 1.0584]],
                             'nrun': 2,
                             'start_frame': 20,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'sparsity':'regul_1E-4',
                             'description': 'regul_1E-4 3 interaction particles'}

    # elctrostatic
    if id == 84:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 2000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 'none',
                             'nrun': 10,
                             'clamp': 0.005,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':40,
                             'cmap':'tab20c',
                             'arrow_length':10,
                             'sparsity':'replace'}
    if id == 85:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 5E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 'none',
                             'nrun': 10,
                             'clamp': 0.005,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':10,
                             'cmap':'tab20b',
                             'sparsity':'replace'}
    if id == 86:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 2000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'no',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 'none',
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':40,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'sparsity':'replace'}
    if id == 87:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 'none',
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':5,
                             'cmap':'tab10',
                             'description':'Periodic Particles_E is a second derivative simulation, acceleration is function of electrostatic law qiqj/r2 interaction is type-type dependent best_model:22',
                             'sparsity':'replace'}
    if id == 88:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_{id}',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 'none',
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':20,
                             'cmap':'tab10',
                             'description':'Periodic Particles_E is a second derivative simulation, acceleration is function of electrostatic law qiqj/r2 interaction is type-type dependent best_model:22',
                             'sparsity':'replace'}
    if id == 89:
        model_config_test = {'ntry': id,
                             'input_size': 11,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.15,
                             'dataset': f'231001_87',
                             'nparticles': 960,
                             'nparticle_types': 3,
                             'ninteractions': 3,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-9,
                             'v_init': 1E-4,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'ElecParticles',
                             'prediction': '2nd_derivative',
                             'p': [[2], [1], [-1]],
                             'upgrade_type': 'none',
                             'nrun': 10,
                             'clamp': 0.002,
                             'pred_limit': 1E10,
                             'start_frame': 0,
                             'arrow_length':5,
                             'cmap':'tab10',
                             'description':'sparsity regul_1E-4 Periodic Particles_E is a second derivative simulation, acceleration is function of electrostatic law qiqj/r2 interaction is type-type dependent best_model:22',
                             'sparsity':'regul_1E-4'}

    # 4 types boundary periodic N=960 mesh diffusion
    if id == 121:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 16,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 3840,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'DiffMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'c': np.linspace(1, 12, 4).tolist(),
                             'beta': 1E-4,
                             'nrun': 2,
                             'clamp': 0.01,
                             'pred_limit': 1E9,
                             'start_frame': -300,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'sparsity':'replace'
                             }
    # 4 types boundary periodic N=960 mesh wave
    if id == 122:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 6000,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(1, 1, 4).tolist(),
                             'c': [0.1, 0.2, 0.5, 1],
                             'beta': 1E-5,
                             'nrun': 2,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'description':'Wave equation fixed particles 4 beta coefficients',
                             'sparsity':'replace'
                             }
    if id == 123:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 64,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 6000,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'DiffMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 4).tolist(),
                             'c': np.linspace(1, 12, 4).tolist(),
                             'beta': 1E-4,
                             'nrun': 2,
                             'clamp': 0.01,
                             'pred_limit': 1E9,
                             'start_frame': -300,
                             'cmap':'tab20b',
                             'arrow_length':10,
                             'description':'Heat equation fixed particles 4 conductivities',
                             'sparsity':'replace'
                             }
    if id == 124:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 16,
                             'n_mp_layers': 5,
                             'noise_level': 5E-4,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 4225,
                             'nparticle_types': 5,
                             'ninteractions': 5,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 5).tolist(),
                             'c': [0,0.2,0.9,1,0.3],
                             'particle_value_map': 'pattern_6.tif',     # 'particle_value_map': 'pattern_6.tif',
                             'particle_type_map': 'pattern_8.tif',
                             'beta': 1E-2,
                             'nrun': 10,
                             'clamp': 0,
                             'pred_limit': 1E9,
                             'start_frame': 0,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'description': 'Wave equation brownian particles 5 coefficients',
                             'sparsity':'replace'
                             }
    if id == 125:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 16,
                             'n_mp_layers': 5,
                             'noise_level': 5E-4,
                             'radius': 0.3,
                             'dataset': f'231001_{id}',
                             'nparticles': 4225,
                             'nparticle_types': 5,
                             'ninteractions': 5,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'DiffMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 5).tolist(),
                             'c': [0,0.2,0.9,1,0.3],
                             'particle_value_map': 'pattern_9.tif',
                             'particle_type_map': 'pattern_8.tif',
                             'beta': 1E-2,
                             'nrun': 2,
                             'clamp': 0,
                             'pred_limit': 1E9,
                             'start_frame': 0,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'description': 'Diffusion equation brownian particles 5 coefficients',
                             'sparsity':'replace'
                             }
    if id == 126:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 16,
                             'n_mp_layers': 5,
                             'noise_level': 5E-4,
                             'radius': 0.3,
                             'dataset': f'231001_126',
                             'nparticles': 4225,
                             'nparticle_types': 5,
                             'ninteractions': 5,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 5).tolist(),
                             'c': [0,0.2,0.9,1,0.3],
                             'particle_value_map': 'pattern_10.tif',     # 'particle_value_map': 'pattern_6.tif',
                             'particle_type_map': 'pattern_8.tif',
                             'beta': 1E-2,
                             'nrun': 10,
                             'clamp': 0,
                             'pred_limit': 1E9,
                             'start_frame': 0,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'description': 'Wave equation brownian particles 5 coefficients',
                             'sparsity':'none'
                             }
    if id == 127:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 16,
                             'n_mp_layers': 5,
                             'noise_level': 5E-4,
                             'radius': 0.3,
                             'dataset': f'231001_126',
                             'nparticles': 4225,
                             'nparticle_types': 5,
                             'ninteractions': 5,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 5).tolist(),
                             'c': [0,0.2,0.9,1,0.3],
                             'particle_value_map': 'pattern_10.tif',     # 'particle_value_map': 'pattern_6.tif',
                             'particle_type_map': 'pattern_8.tif',
                             'beta': 1E-2,
                             'nrun': 10,
                             'clamp': 0,
                             'pred_limit': 1E9,
                             'start_frame': 0,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'description': 'Wave equation brownian particles 5 coefficients',
                             'sparsity':'regul_1E-4'
                             }
    if id == 128:
        model_config_test = {'ntry': id,
                             'input_size': 4,
                             'output_size': 1,
                             'hidden_size': 16,
                             'n_mp_layers': 5,
                             'noise_level': 5E-4,
                             'radius': 0.3,
                             'dataset': f'231001_126',
                             'nparticles': 4225,
                             'nparticle_types': 5,
                             'ninteractions': 5,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 5E-5,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'WaveMesh',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'none',
                             'p': np.linspace(0.2, 5, 5).tolist(),
                             'c': [0,0.2,0.9,1,0.3],
                             'particle_value_map': 'pattern_10.tif',     # 'particle_value_map': 'pattern_6.tif',
                             'particle_type_map': 'pattern_8.tif',
                             'beta': 1E-2,
                             'nrun': 10,
                             'clamp': 0,
                             'pred_limit': 1E9,
                             'start_frame': 0,
                             'cmap':'tab10',
                             'arrow_length':10,
                             'description': 'Wave equation brownian particles 5 coefficients',
                             'sparsity':'regul_1E-1'
                             }

    if id == 142:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.04,
                             'dataset': f'231001_{id}',
                             'nparticles': 900,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'add',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'linear',
                             'nlayers_update': 3,
                             'hidden_size_update': 64,
                             'p': [[50,10,40],[50,30,40],[50,50,40],[50,80,40]],   # separation alignement cohesion
                             'nrun': 10,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab10',
                             'arrow_length':5,
                             'description':'Boids acceleration pred 4 different alignement 10 30 50 80',
                             'sparsity':'replace'
                             }
    if id == 143:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.04,
                             'dataset': f'231001_{id}',
                             'nparticles': 1800,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'linear',
                             'nlayers_update': 3,
                             'hidden_size_update': 64,
                             'p': [[50,10,40],[50,30,40],[50,50,40],[50,80,40]],    # separation alignement cohesion
                             'nrun': 10,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab10',
                             'arrow_length':5,
                             'description':'Boids acceleration pred aggr mean boid speed/4 4 different alignement 10 30 50 80',
                             'sparsity':'replace'
                             }
    if id == 144:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.04,
                             'dataset': f'231001_{id}',
                             'nparticles': 1800,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'linear',
                             'nlayers_update': 3,
                             'hidden_size_update': 64,
                             'p': [[50,10,0],[10,30,40],[20,50,40],[35,80,40]],        # separation alignement cohesion
                             'nrun': 10,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab10',
                             'arrow_length':5,
                             'description':'Boids acceleration pred aggr mean boid speed/4 4 different params',
                             'sparsity':'replace'
                             }
    if id == 145:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.04,
                             'dataset': f'231001_{id}',
                             'nparticles': 1800,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'linear',
                             'nlayers_update': 3,
                             'hidden_size_update': 64,
                             'p': [[50,10,0],[40,20,20],[20,30,20],[35,40,20]],        # separation alignement cohesion
                             'nrun': 10,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab10',
                             'arrow_length':5,
                             'description':'Boids acceleration pred aggr mean boid speed/4 4 different params',
                             'sparsity':'replace'
                             }
    if id == 146:
        model_config_test = {'ntry': id,
                             'input_size': 9,
                             'output_size': 2,
                             'hidden_size': 128,
                             'n_mp_layers': 5,
                             'noise_level': 0,
                             'radius': 0.04,
                             'dataset': f'231001_{id}',
                             'nparticles': 1800,
                             'nparticle_types': 4,
                             'ninteractions': 4,
                             'nframes': 1000,
                             'sigma': .005,
                             'tau': 1E-10,
                             'v_init': 0,
                             'aggr_type': 'mean',
                             'boundary': 'periodic',  # periodic   'no'  # no boundary condition
                             'data_augmentation': True,
                             'batch_size': 8,
                             'embedding': 2,
                             'model': 'Particles_A',
                             'prediction': '2nd_derivative',
                             'upgrade_type': 'linear',
                             'nlayers_update': 3,
                             'hidden_size_update': 64,
                             'p': [[50,10,0],[40,20,20],[20,30,20],[35,40,20]],        # separation alignement cohesion
                             'nrun': 10,
                             'clamp': 1E-3,
                             'pred_limit': 1E9,
                             'start_frame': 0.,
                             'cmap':'tab10',
                             'arrow_length':5,
                             'description':'Boids acceleration pred aggr mean boid speed/4 4 different params',
                             'sparsity':'replace'
                             }

    for key, value in model_config_test.items():
        print(key, ":", value)

    return model_config_test

if __name__ == '__main__':

    print('')
    print('version 1.6 231120')
    print('use of https://github.com/gpeyre/.../ml_10_particle_system.ipynb')
    print('')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    scaler = StandardScaler()
    S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    gtestlist = [126,47,89,74] #[123, 140, 141, 73, 123] # [75,84,85]

    for gtest in gtestlist:

        model_config = load_model_config(id=gtest)

        # if (gtest>=0) and (gtest<10):
        #     model_config = load_model_config(id=44)
        # if (gtest>=10) and (gtest<20):
        #     model_config = load_model_config(id=45)
        # if (gtest>=20) and (gtest<30):
        #     model_config = load_model_config(id=46)
        # model_config['ntry']=gtest
        cmap = cc(model_config=model_config)

        sigma = model_config['sigma']
        aggr_type = model_config['aggr_type']

        if model_config['boundary'] == 'no':  # change this for usual BC
            def bc_pos(X):
                return X
            def bc_diff(D):
                return D
        else:
            def bc_pos(X):
                return torch.remainder(X, 1.0)
            def bc_diff(D):
                return torch.remainder(D - .5, 1.0) - .5


        # if 'Boids' in model_config['description']:
        #     data_generate_boid(model_config, bVisu=True, bDetails=True, bErase=False, step=10)
        # else:
        #     data_generate(model_config, bVisu=True, bDetails=True, bErase=False, step=10)
        data_train(model_config)
        # x, rmserr_list = data_test(model_config, bVisu=True, bPrint=True, best_model=13, step=5, bTest='', initial_map='')
        # data_plot(model_config, epoch=-1, bPrint=True, best_model=-1)
        # prev_nparticles, new_nparticles, prev_index_particles, index_particles = data_test_generate(model_config, bVisu=True, bDetails=True, step=10)
        # x, rmserr_list = data_test(model_config, bVisu = True, bPrint=True, index_particles=index_particles, prev_nparticles=prev_nparticles, new_nparticles=new_nparticles, prev_index_particles=prev_index_particles, best_model=-1, step=100)


