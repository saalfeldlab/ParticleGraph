import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

import glob
import torch_geometric as pyg
import torch_geometric.data as data
import math
import torch_geometric.utils as pyg_utils
import torch.nn as nn
from torch.nn import functional as F
from shutil import copyfile
from prettytable import PrettyTable
import time
import networkx as nx
from torch_geometric.utils.convert import to_networkx


def normalize99(Y, lower=1,upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return x01, x99


def display_frame(t=20):
    s = t/(niter//save_per-1)
    plt.scatter(Zsvg[:,0,t], Zsvg[:,1,t], color=[s,0,1-s])
    plt.axis('equal')
    plt.axis([0,1,0,1])

def distmat_square(X,Y):
    return torch.sum( bc_diff(X[:,None,:] - Y[None,:,:])**2, axis=2 )

def distmat_square2(X, Y):
    X_sq = (X ** 2).sum(axis=-1)
    Y_sq = (Y ** 2).sum(axis=-1)
    cross_term = X.matmul(Y.T)
    return X_sq[:, None] + Y_sq[None, :] - 2 * cross_term

def kernel(X,Y):
    return -torch.sqrt( distmat_square(X,Y) )

def MMD(X,Y):
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.sum( kernel(X,X) )/n**2 + \
      torch.sum( kernel(Y,Y) )/m**2 - \
      2*torch.sum( kernel(X,Y) )/(n*m)
    return a.item()

def psi(r,p):
    sigma = .05;
    return -p[2]*torch.exp(-r**p[0] / (2 * sigma ** 2)) + p[3]* torch.exp(-r**p[1] / (2 * sigma ** 2))
def Speed(X,Y,p):
    sigma = .05;

    temp=distmat_square(X,Y)
    return 0.25/X.shape[0] * 1/sigma**2 * torch.sum(psi(distmat_square(X,Y),p)[:,:,None] * bc_diff( X[:,None,:] - Y[None,:,:] ), axis=1 )

def Edge_index(X,Y):

    return torch.sum( bc_diff(X[:,None,:] - Y[None,:,:])**2, axis=2 )


class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, nlayers, device):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layernorm = False

        for i in range(nlayers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == nlayers - 1 else hidden_size, device=device, dtype=torch.float64
            ))
            if i != nlayers - 1:
                self.layers.append(torch.nn.ReLU())
                # self.layers.append(torch.nn.Dropout(p=0.0))
        if self.layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size, device=device, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.lin_edge = MLP(in_feats=11, out_feats=2, num_layers=3, hidden=32)
        # self.lin_node = MLP(in_feats=4, out_feats=1, num_layers=2, hidden=16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x,x))
        return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type= x_i[:,4]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]

        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:,None].repeat(1,4)), dim=-1)

        return self.lin_edge(in_features)
    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)

class EdgeNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        return self.new_edges

    def message(self, x_i, x_j, edge_feature):

        r = torch.sqrt(torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type= x_i[:,4]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]

        d = r

        self.new_edges = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:,None].repeat(1,4)), dim=-1)

        return d

class InteractionNetworkEmb(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, nlayers, embedding, device):
        super().__init__(aggr='add')  # "Add" aggregation.

        self.nlayers = nlayers
        self.device = device
        self.embedding = embedding

        self.lin_edge = MLP(input_size=3*self.embedding, hidden_size=3*self.embedding, output_size=self.embedding, nlayers= self.nlayers, device=self.device)
        self.lin_node = MLP(input_size=2*self.embedding, hidden_size=2*self.embedding, output_size=self.embedding, nlayers= self.nlayers, device=self.device)


    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        node_out = x + node_out
        edge_out = edge_feature + self.new_edges

        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):

        x = torch.cat((edge_feature, x_i, x_j ), dim=-1)

        x = self.lin_edge(x)
        self.new_edges = x

        return x

class ReccurentGNN(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(self,model_config, device):
        super().__init__()

        self.hidden_size = model_config['hidden_size']
        self.embedding = model_config['embedding']
        self.nlayers = model_config['n_mp_layers']
        self.device = device
        self.noise_level = model_config['noise_level']

        self.edge_init = EdgeNetwork()

        self.layer = torch.nn.ModuleList([InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device) for _ in range(self.nlayers)])
        self.node_out = MLP(input_size=self.embedding, hidden_size=self.hidden_size, output_size=2, nlayers=3, device=self.device)

        self.embedding_node = MLP(input_size=8, hidden_size=self.embedding, output_size=self.embedding, nlayers=3, device=self.device)
        self.embedding_edges = MLP(input_size=11, hidden_size=self.embedding, output_size=self.embedding, nlayers=3, device=self.device)

    def forward(self, data):

        node_feature = torch.cat((data.x[:,0:4],data.x[:,4:5].repeat(1,4)), dim=-1)

        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]),requires_grad=False, device='cuda:0') * self.noise_level
        node_feature= node_feature+noise
        edge_feature = self.edge_init(node_feature, data.edge_index, edge_feature=data.edge_attr)

        node_feature = node_feature.to(dtype=torch.float64)
        edge_feature = edge_feature.to(dtype=torch.float64)

        node_feature = self.embedding_node(node_feature)
        edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer[i](node_feature, data.edge_index, edge_feature=edge_feature)

        pred = self.node_out(node_feature)

        return pred




if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    # flist = ['ReconsGraph']
    # for folder in flist:
    #     files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/Graph/{folder}/*")
    #     for f in files:
    #         os.remove(f)



    niter = 400
    d = 2  # dimension
    radius = 0.075

    p=torch.load('./p_list_simu_N5.pt')


    # datum = '230721'
    # print(datum)
    # ntypes_list=[5]
    # n_list=[250]

    # datum = '230724'
    # print(datum)
    # ntypes_list=[5]
    # n_list=[500]

    datum = '230802'
    print(datum)
    ntypes_list=[2]
    n_list=[2000]

    ntypes=ntypes_list[0]
    nparticles=n_list[0]

    boundary = 'no'  # no boundary condition
    # boundary = 'per' # periodic
    if boundary == 'no':
        tau = 1 / 1000  # time step
    else:
        tau = 1 / 200
    if boundary == 'no':  # change this for usual BC
        def bc_pos(X):
            return X
        def bc_diff(D):
            return D
    else:
        def bc_pos(X):
            return torch.remainder(X, 1.0)
        def bc_diff(D):
            return torch.remainder(D - .5, 1.0) - .5

    step = 2

    # train
    if step==1:

    # train data

        graph_files = glob.glob(f"../Graph/graphs_data/graphs_particles_{datum}/edge*")
        NGraphs=int(len(graph_files)/niter)
        print ('Graph files N: ',NGraphs)
        print('Normalize ...')
        time.sleep(0.5)

        arr = np.arange(0,NGraphs-1,2)
        for run in tqdm(arr):
            x=torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/x_list_{run}.pt')
            acc=torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')
            if run == 0:
                xx = x
                aacc = acc
            else:
                xx = torch.concatenate((x, xx))
                aacc = torch.concatenate((aacc, acc))

        mvx = torch.mean(xx[:,:,0,:])
        mvy = torch.mean(xx[:,:,1,:])
        vx = torch.std(xx[:,:,0,:])
        vy = torch.std(xx[:,:,1,:])
        nvx = np.array(xx[:,:,0,:].detach().cpu())
        vx01, vx99 = normalize99(nvx)
        nvy = np.array(xx[:,:,1,:].detach().cpu())
        vy01, vy99 = normalize99(nvy)
        vnorm = torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)


        print(f'v_x={mvx} +/- {vx}')
        print(f'v_y={mvy} +/- {vy}')
        print(f'vx01={vx01} vx99={vx99}')
        print(f'vy01={vy01} vy99={vy99}')

        max = torch.mean(aacc[:,:,0,:])
        may = torch.mean(aacc[:,:,1,:])
        ax = torch.std(aacc[:,:,0,:])
        ay = torch.std(aacc[:,:,1,:])
        nax = np.array(aacc[:,:,0,:].detach().cpu())
        ax01, ax99 = normalize99(nax)
        nay = np.array(aacc[:,:,1,:].detach().cpu())
        ay01, ay99 = normalize99(nay)

        ynorm = torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)

        print(f'acc_x={max} +/- {ax}')
        print(f'acc_y={may} +/- {ay}')
        print(f'ax01={ax01} ax99={ax99}')
        print(f'ay01={ay01} ay99={ay99}')

        best_loss = np.inf

        batch_size = 1

        model_config = {'ntry': 513,
                        'embedding': 128,
                        'hidden_size': 32,
                        'n_mp_layers': 4,
                        'noise_level': 0}

        ntry=model_config['ntry']

        print(f"ntry :{ntry}")

        l_dir = os.path.join('..','Graph','log')
        log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
        print('log_dir: {}'.format(log_dir))

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

        copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))
        torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
        torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

        grid_list = [1, 2, 5, 10, 20]

        for gridsearch in grid_list:

            model = ReccurentGNN(model_config=model_config, device=device)

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

            optimizer= torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            model.train()

            for epoch in range(12):

                model.train()
                total_loss = []

                for N in tqdm(range(10000)):

                    run = 1 + np.random.randint(gridsearch)

                    x_list=torch.load(f'graphs_data/graphs_particles_{datum}/x_list_{run}.pt')
                    acc_list=torch.load(f'graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')

                    acc_list[:,:, 0,:] = acc_list[:,:, 0,:] / ynorm[4]
                    acc_list[:,:, 1,:] = acc_list[:,:, 1,:] / ynorm[5]

                    optimizer.zero_grad()
                    loss = 0

                    for loop in range(batch_size):

                        k = np.random.randint(niter - 1)

                        edges = torch.load(f'graphs_data/graphs_particles_{datum}/edge_{run}_{k}.pt')

                        x=torch.squeeze(x_list[k,:,:,:])
                        x = torch.permute(x, (2, 0, 1))
                        x = torch.reshape (x,(nparticles*ntypes,5))
                        dataset = data.Data(x=x, edge_index=edges)
                        y = torch.squeeze(acc_list[k,:,:,:])
                        y = torch.permute(y, (2, 0, 1))
                        y = torch.reshape (y,(nparticles*ntypes,2))

                        pred = model(dataset)

                        datafit = (pred-y).norm(2)

                        loss += datafit

                        total_loss.append(datafit.item())

                    loss.backward()
                    optimizer.step()

                    if (epoch==0) & (N%10==0):
                        print("   N {} Loss: {:.4f}".format(N, np.mean(total_loss) / nparticles))
                    if N%250==0:
                        print("   N {} Loss: {:.4f}".format(N,np.mean(total_loss)/nparticles ))

                print("Epoch {}. Loss: {:.4f}".format(epoch, np.mean(total_loss)/nparticles ))

                if (np.mean(total_loss) < best_loss):
                    best_loss = np.mean(total_loss)
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_trained_with_{gridsearch}_graph.pt'))
                    print('\t\t Saving model')


            # fig = plt.figure(figsize=(16, 8))
            # plt.ion()
            # ax = fig.add_subplot(1, 2, 1)
            # for N in range(ntypes):
            #     plt.scatter(x[:, 0].detach().cpu().numpy(), x[:, 1].detach().cpu().numpy(),s=5)
            # plt.axis([-0.1, 1.1, -0.1, 1.1])
            # ax = plt.gca()
            # ax.axes.xaxis.set_ticklabels([])
            # ax.axes.yaxis.set_ticklabels([])
            # ax = fig.add_subplot(1, 2, 2)
            #
            # distance = distmat_square(x, x)
            # t = torch.Tensor([0.075 * 0.075])  # threshold
            # adj_t = (distance < 0.075 * 0.075).float() * 1
            # edge_index = adj_t.nonzero().t().contiguous()
            #
            # dataset = data.Data(x=x, edge_index=edges)
            # vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            # pos = dict(enumerate(np.array(x[:, 0:2].detach().cpu()), 0))
            # nx.draw(vis, pos=pos, ax=ax, node_size=10, linewidths=0)

    # test
    if step==2:


        # model_config = {'ntry': 502,
        #                 'embedding': 32,
        #                 'n_mp_layers': 4,
        #                 'hidden_size': 32,
        #                 'noise_level': 0}
        #
        model_config = {'ntry': 503,
                        'embedding': 128,
                        'n_mp_layers': 4,
                        'hidden_size': 32,
                        'noise_level': 0}
        #
        # model_config = {'ntry': 504,
        #                 'embedding': 64,
        #                 'n_mp_layers': 4,
        #                 'hidden_size': 32,
        #                 'noise_level': 0}
        #
        # model_config = {'ntry': 505,
        #                 'embedding': 32,
        #                 'n_mp_layers': 4,
        #                 'hidden_size': 32,
        #                 'noise_level': 0}

        datum = '230724'
        print(datum)

        ntry=model_config['ntry']
        print(f"ntry :{ntry}")
        print(f"embedding: {model_config['embedding']}")


        model = ReccurentGNN(model_config=model_config, device=device)
        state_dict = torch.load(f"../Graph/log/try_{ntry}/models/best_model.pt")
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()


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

        GT = torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/x_list_0.pt')
        GT0 = torch.squeeze (GT[0])

        ynorm = torch.load(f'../Graph/log/try_{ntry}/ynorm.pt')
        vnorm = torch.load(f'../Graph/log/try_{ntry}/vnorm.pt')

        run=0
        acc_list = torch.load(f'graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')
        acc_list[:, :, 0, :] = acc_list[:, :, 0, :] / ynorm[4]
        acc_list[:, :, 1, :] = acc_list[:, :, 1, :] / ynorm[5]

        edges = torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/edge_{run}_0.pt')


        print(f"ntry :{ntry}")
        print(f"run :{run}")
        dataset = data.Data(x=GT0.cuda(), edge_index=edges.cuda())
        print('num_nodes: ', GT0.shape[0])
        nparticles = GT0.shape[0]
        print('dataset.num_node_features: ', dataset.num_node_features)

        x = GT0
        x = torch.permute(x, (2, 0, 1))
        x = torch.reshape(x, (nparticles * ntypes, 5))

        GT0 = x.clone()

        for it in tqdm(range(200)):

            x0 = torch.squeeze (GT[it+1])
            x0 = torch.permute(x0, (2, 0, 1))
            x0 = torch.reshape(x0, (nparticles * ntypes, 5))

            distance = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset)  # acceleration estimation

            # y = torch.squeeze(acc_list[0, :, :, :])
            # y = torch.permute(y, (2, 0, 1))
            # y = torch.reshape(y, (nparticles * ntypes, 2))

            y[:, 0] = y[:, 0] * ynorm[4]
            y[:, 1] = y[:, 1] * ynorm[5]
            x[:, 2:4] = x[:, 2:4] + y  # speed update
            x[:, 0:2] = x[:, 0:2] + x[:, 2:4]  # position update

            fig = plt.figure(figsize=(25, 16))
            # plt.ion()
            ax = fig.add_subplot(2, 3, 1)
            for k in range(ntypes):
                plt.scatter(GT0[k*nparticles:(k+1)*nparticles, 0].detach().cpu(), GT0[k*nparticles:(k+1)*nparticles, 1].detach().cpu(), s=3)

            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'Distribution at t0 is 1.0x1.0')

            ax = fig.add_subplot(2, 3, 2)
            for k in range(ntypes):
                plt.scatter(x0[k*nparticles:(k+1)*nparticles, 0].detach().cpu(), x0[k*nparticles:(k+1)*nparticles, 1].detach().cpu(), s=3)
            ax = plt.gca()
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'True', fontsize=30)
            # plt.text(-0.25, 1.38, f'Frame: {min(niter,it)}')
            # plt.text(-0.25, 1.33, f'Physics simulation', fontsize=10)

            ax = fig.add_subplot(2, 3, 4)
            pos = dict(enumerate(x[:, 0:2].detach().cpu().numpy(), 0))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw(vis, pos=pos, ax=ax, node_size=10, linewidths=0)
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, f'Frame: {it}')
            plt.text(-0.25, 1.33, f'Graph: {x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)

            ax = fig.add_subplot(2, 3, 5)
            for k in range(ntypes):
                plt.scatter(x[k*nparticles:(k+1)*nparticles, 0].detach().cpu(), x[k*nparticles:(k+1)*nparticles, 1].detach().cpu(), s=3)
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'Model', fontsize=30)

            plt.savefig(f"../Graph/ReconsGraph/Fig_{it}.tif")
            plt.close()







