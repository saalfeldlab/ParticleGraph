# from pysr import PySRRegressor

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
import math
import torch_geometric.utils as pyg_utils
import torch.nn as nn
from torch.nn import functional as F
from prettytable import PrettyTable
from shutil import copyfile

class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=2, hidden=128):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hidden, device=device))
        if num_layers > 2:
            for i in range(1, num_layers - 1):
                layer = nn.Linear(hidden, hidden, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden, out_feats, device=device)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x


class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionParticles, self).__init__(aggr='mean')  # "Add" aggregation.

        self.lin_edge = MLP(in_feats=8, out_feats=2, num_layers=3, hidden=16)
        self.a = nn.Parameter(torch.tensor(np.ones((int(nparticles), 1)), device='cuda:0'))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x[:, 4] = self.a[x[:, 6].detach().cpu().numpy(), 0]
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x,x))
        return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type= x_i[:,4:5]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]
        # x_j_type = x_j[:, 5]

        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:]), dim=-1)

        return self.lin_edge(in_features)
    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)


def psi(r,p):
    sigma = .005;
    return -p[2]*torch.exp(-r**p[0] / (2 * sigma ** 2)) + p[3]* torch.exp(-r**p[1] / (2 * sigma ** 2))


if __name__ == "__main__":

    X = 2 * np.random.randn(100, 2)
    y = 2 * np.exp(X[:, 0]*3) - 4 * np.exp(X[:, 1]*4)

    p1 = [1.39, 1.8, 0.094, 0.91]
    p1[0] = p1[0] + 1
    p1[1] = p1[1] + 1


    # model = PySRRegressor(
    #     niterations=1000,  # < Increase me for better results
    #     binary_operators=["+", "*", "-", "/"],
    #     unary_operators=[
    #         "exp",
    #         "inv(x) = 1/x",
    #         # ^ Custom operator (julia syntax)
    #     ],
    #     extra_sympy_mappings={"inv": lambda x: 1 / x},
    #     # ^ Define operator for SymPy as well
    #     loss="loss(prediction, target) = (prediction - target)^2",
    #     # ^ Custom loss function (julia syntax)
    # )
    #
    # model.fit(X, y)
    # print(model)
    # result=model.equations.lambda_format.T


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flist = ['ReconsGraph']
    for folder in flist:
        files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/Graph/{folder}/*")
        for f in files:
            os.remove(f)


    datum = '230824'
    print(datum)

    nparticles=2000
    niter=200
    d=2
    sigma = .005;
    radius=0.075

    model = InteractionParticles()

    ntry=515
    state_dict = torch.load(f"./log/try_{ntry}/models/best_model_with_1_graphs.pt")
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    ynorm=torch.load(f'./log/try_{ntry}/ynorm.pt')
    vnorm=torch.load(f'./log/try_{ntry}/vnorm.pt')

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")

    print(torch.mean(model.a[0:int(nparticles/2)]))
    print(torch.std(model.a[0:int(nparticles / 2)]))
    print(torch.mean(model.a[int(nparticles/2):nparticles]))
    print(torch.std(model.a[int(nparticles / 2):nparticles]))

    # plt.ion()
    # plt.hist(model.a.detach().cpu().numpy(),100)
    fig = plt.figure(figsize=(25, 16))
    plt.plot(model.a.detach().cpu().numpy(), '.',color='k')
    plt.show()


    x=torch.load(f'graphs_data/graphs_particles_{datum}/x_0_0.pt')
    x00=torch.load(f'graphs_data/graphs_particles_{datum}/x_0_0.pt')
    y=torch.load(f'graphs_data/graphs_particles_{datum}/y_0_0.pt')

    rmserr_list=[]
    rmserr_list0=[]
    rmserr_list1=[]

    c1 = np.array([220, 50, 32]) / 255
    c2 = np.array([0, 114, 178]) / 255

    for it in tqdm(range(niter-1)):

        x0 = torch.load(f'graphs_data/graphs_particles_{datum}/x_0_{it+1}.pt')

        distance = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
        t = torch.Tensor([radius ** 2])  # threshold
        adj_t = (distance < radius ** 2).float() * 1
        edge_index = adj_t.nonzero().t().contiguous()

        dataset = data.Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            y = model(dataset)      # acceleration estimation

        #y = y * (ynorm[1] - ynorm[0]) + ynorm[0]
        #y = y * (ynorm[1] - ynorm[0])

        y[:, 0] = y[:, 0] * ynorm[4]
        y[:, 1] = y[:, 1] * ynorm[5]

        x[:, 2:4] = x[:, 2:4] + y   # speed update

        x[:,0:2] = x[:,0:2] + x[:,2:4]  # position update

        if (it%10==0):

            fig = plt.figure(figsize=(25, 16))

            ax = fig.add_subplot(2, 3, 1)
            plt.scatter(x00[0:1000, 0].detach().cpu(), x00[0:1000, 1].detach().cpu(), s=3, color=c1)
            plt.scatter(x00[1000:, 0].detach().cpu(), x00[1000:, 1].detach().cpu(), s=3, color=c2)
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'Distribution at t0 is 1.0x1.0')

            ax = fig.add_subplot(2,3,2)
            plt.scatter(x0[0:1000, 0].detach().cpu(), x0[0:1000, 1].detach().cpu(), s=3, color=c1)
            plt.scatter(x0[1000:, 0].detach().cpu(), x0[1000:, 1].detach().cpu(), s=3, color=c2)
            ax = plt.gca()
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'True', fontsize=30)
            # plt.text(-0.25, 1.38, f'Frame: {min(niter,it)}')
            # plt.text(-0.25, 1.33, f'Physics simulation', fontsize=10)

            ax = fig.add_subplot(2,3,4)
            pos=dict(enumerate(np.array(x[:,0:2].detach().cpu()), 0))
            vis = to_networkx(dataset,remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=10, linewidths=0, with_labels=False)
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, f'Frame: {it}')
            plt.text(-0.25, 1.33, f'Graph: {x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)

            ax = fig.add_subplot(2,3,5)
            plt.scatter(x[0:1000, 0].detach().cpu(), x[0:1000, 1].detach().cpu(), s=3, color=c1)
            plt.scatter(x[1000:, 0].detach().cpu(), x[1000:, 1].detach().cpu(), s=3, color=c2)
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'Model', fontsize=30)
            # plt.text(-0.25, 1.38, f'Frame: {it}')
            # plt.text(-0.25, 1.33, f'GNN prediction', fontsize=10)


            rmserr = torch.sqrt(torch.mean(torch.sum((x - x0) ** 2, axis=1)))
            rmserr_list.append(rmserr.item())
            rmserr0 = torch.sqrt(torch.mean(torch.sum((x[0:int(nparticles / 2), :] - x0[0:int(nparticles / 2), :]) ** 2, axis=1)))
            rmserr_list0.append(rmserr0.item())
            rmserr1 = torch.sqrt(torch.mean(torch.sum((x[int(nparticles / 2):nparticles, :] - x0[int(nparticles / 2):nparticles, :]) ** 2, axis=1)))
            rmserr_list1.append(rmserr1.item())

            ax = fig.add_subplot(2, 3, 3)
            plt.plot(rmserr_list, 'k', label='RMSE')
            plt.plot(rmserr_list0, color=c1, label='RMSE0')
            plt.plot(rmserr_list1, color=c2, label='RMSE1')
            plt.ylim([0, 0.1])
            plt.xlim([0, 200])
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.xlabel('Frame [a.u]', fontsize="10")
            plt.ylabel('RMSE [a.u]', fontsize="10")
            plt.legend(fontsize="10")

            ax = fig.add_subplot(2, 3, 6)
            temp1 = torch.cat((x, x0), 0)
            pos = dict(enumerate(np.array(temp1[:, 0:2].detach().cpu()), 0))
            temp2 = torch.tensor(np.arange(nparticles),device=device)
            temp3 = torch.tensor(np.arange(nparticles)+nparticles,device=device)
            temp4 = torch.concatenate((temp2[:,None],temp3[:,None]),1)
            temp4 = torch.t(temp4)
            dataset = data.Data(x=temp1, edge_index=temp4)
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False)
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.18, f'Frame: {it}')
            plt.text(-0.25, 1.13, 'Prediction RMSE: {:.4f}'.format(rmserr.detach()), fontsize=10)

            plt.savefig(f"./ReconsGraph/Fig_{it}.tif")
            plt.close()

