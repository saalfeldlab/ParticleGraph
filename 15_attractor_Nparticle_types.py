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

class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, model_config, device):

        super(InteractionParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.device = device
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.hidden_size = model_config['hidden_size']
        self.nlayers = model_config['n_mp_layers']

        self.noise_level = model_config['noise_level']

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers, hidden_size=self.hidden_size, device=self.device)

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
        x_i_type = x_i[:,4:5]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]

        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type), dim=-1)   # [:,None].repeat(1,4)

        return self.lin_edge(in_features)
    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    flist = ['ReconsGraph']
    for folder in flist:
        files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/Graph/{folder}/*")
        for f in files:
            os.remove(f)

    ntypes_list=[2]
    n_list=[2000]


    ntypes=ntypes_list[0]
    nparticles=n_list[0]


    niter = 200
    d = 2  # dimension
    radius = 0.075

    p=torch.load('./p_list_simu_N5.pt')

    datum = '230802'
    print(datum)

    p[0] = torch.tensor([1.23, 1.59, 0.1, 0.87])
    p[1] = torch.tensor([1.78, 1.6, 0.65, 0.38])

    block_type = torch.zeros(nparticles,1,device=device)
    for k in range(1,ntypes):
        block_type=torch.cat((block_type,k*torch.ones(nparticles,1,device=device)),1)
    block_type = block_type[:, None, :]
    block_type = block_type.repeat(niter, 1, 1, 1)

    # folder=f'graphs_data/graphs_particles_{datum}'
    folder=f'./graphs_data/graphs_particles_{datum}'

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

    # generate data
    if step==0:

        copyfile(os.path.realpath(__file__), os.path.join(f'graphs_data/graphs_particles_{datum}/', 'generating_code.py'))

        for n in n_list:
            for ntypes in ntypes_list:
                for run in range(30,200):

                    x_list=[]
                    speed_list=[]
                    edge_list=[]
                    acc_list=[]

                    print(f'run: {run} n types: {ntypes}  n points: {n}')

                    X = torch.rand(n,d,ntypes,device=device)
                    t = torch.tensor(np.linspace(-1.5,1.5,1000))

                    ZSvg = torch.zeros((n, d, ntypes, niter),device=device)
                    New_speed= torch.ones((n, d, ntypes),device=device)

                    rr = torch.tensor(np.linspace(0, 0.015, 100),device=device)
                    # Psi = psi(rr,p)

                    time.sleep(0.5)
                    for it in tqdm(range(niter)):

                        Xall=torch.permute(X, (0, 2, 1))
                        Xall = torch.reshape(Xall, (ntypes * n, d))

                        for N in range(ntypes):
                            New_speed[:,:,N]= - tau * Speed(X[:,:,N],Xall,p[N,:])

                        for N in range(ntypes):
                            ZSvg[:,:,N,it]=X[:,:,N].clone().detach()
                            X[:, :, N] = bc_pos(X[:,:,N] + New_speed[:,:,N])

                        distance = distmat_square(Xall, Xall)
                        t = torch.Tensor([0.075 * 0.075])  # threshold
                        adj_t = (distance < 0.075 * 0.075).float() * 1
                        edge_index = adj_t.nonzero().t().contiguous()

                        if run==-1:
                            fig = plt.figure(figsize=(16, 8))
                            #plt.ion()
                            ax = fig.add_subplot(1, 2, 1)
                            for N in range(ntypes):
                                plt.scatter(ZSvg[:, 0, N, it].detach().cpu().numpy(), ZSvg[:, 1, N, it].detach().cpu().numpy(), s=5)
                            plt.axis([-0.1, 1.1, -0.1, 1.1])
                            ax = plt.gca()
                            ax.axes.xaxis.set_ticklabels([])
                            ax.axes.yaxis.set_ticklabels([])
                            ax = fig.add_subplot(1, 2, 2)
                            dataset = data.Data(x=Xall, edge_index=edge_index)
                            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                            pos = dict(enumerate(np.array(Xall[:, 0:2].detach().cpu()), 0))
                            nx.draw(vis, pos=pos, ax=ax, node_size=10, linewidths=0)
                            plt.savefig(f"graphs_data/graphs_particles_{datum}/Fig/Fig_{it}.tif")

                        torch.save(edge_index,f'graphs_data/graphs_particles_{datum}/edge_{run}_{it}.pt')

                        x_list.append(X.detach().cpu().numpy())
                        speed_list.append(New_speed.detach().cpu().numpy())

                        edge_list.append(edge_index.detach().cpu().numpy())
                        if it==0:
                            acc_list.append(New_speed.detach().cpu().numpy()*0)
                        else:
                            acc_list.append(New_speed.detach().cpu().numpy() - prev_Speed)
                        prev_Speed = New_speed.detach().cpu().numpy()

                    ZSvg=ZSvg.detach().cpu().numpy()

                    x_list = torch.tensor(x_list,device = device)
                    speed_list = torch.tensor(speed_list, device=device)
                    x_list = torch.cat((x_list,speed_list),2)
                    x_list = torch.cat((x_list, block_type), 2)
                    acc_list = torch.tensor(acc_list,device = device)

                    torch.save(x_list, f'graphs_data/graphs_particles_{datum}/x_list_{run}.pt')
                    torch.save(acc_list, f'graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')

                    if run==0:

                        print ('Test data ...')
                        for it in tqdm(range(niter - 1)):

                            x1 = torch.squeeze(x_list[it + 1])
                            x1 = torch.permute(x1, (2, 0, 1))
                            x1 = torch.reshape(x1, (nparticles * ntypes, 5))

                            x0 = torch.squeeze(x_list[it])
                            x0 = torch.permute(x0, (2, 0, 1))
                            x0 = torch.reshape(x0, (nparticles * ntypes, 5))

                            y1 = torch.squeeze(acc_list[it + 1, :, :, :])
                            y1 = torch.permute(y1, (2, 0, 1))
                            y1 = torch.reshape(y1, (nparticles * ntypes, 2))

                            diff0 = torch.sum(x1[:,2:4] - (x1[:,0:2] - x0[:,0:2])) / nparticles
                            diff1 = torch.sum(y1[:, :] - (x1[:, 2:4] - x0[:, 2:4])) / nparticles

                            if torch.abs(diff0+diff1)>1E-5:
                                print('pb frame {it}')

                            # training

    # train
    if step==1:

    # train data

        graph_files = glob.glob(f'./graphs_data/graphs_particles_{datum}/edge*')
        NGraphs=int(len(graph_files)/niter)
        print ('Graph files N: ',NGraphs)
        print('Normalize ...')
        time.sleep(0.5)

        arr = np.arange(0,NGraphs-1,2)
        for run in tqdm(arr):
            x=torch.load(f'./graphs_data/graphs_particles_{datum}/x_list_{run}.pt')
            acc=torch.load(f'./graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')
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

        grid_list = [1,2,5,10,20]

        for gridsearch in grid_list:

            best_loss = np.inf

            print(f"gridsearch: {gridsearch}")

            model_config = {'ntry': 512,
                        'input_size': 8,
                        'output_size': 2,
                        'hidden_size': 32,
                        'n_mp_layers': 3,
                        'noise_level': 0}

            ntry = model_config['ntry']

            l_dir = os.path.join('.', 'log')
            log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
            print('log_dir: {}'.format(log_dir))

            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

            copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

            torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
            torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

            print(f"ntry :{ntry}")

            model = InteractionParticles(model_config,device)
            # state_dict = torch.load(f"./log/try_{ntry}/models/model_epoch_7.pt")
            # model.load_state_dict(state_dict['model_state_dict'])

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
            print('')

            for epoch in range(12):         #training of 3 hours 12E4 grads

                model.train()
                total_loss = []

                for N in tqdm(range(5000)):

                    run = 1 + np.random.randint(gridsearch)

                    x_list=torch.load(f'./graphs_data/graphs_particles_{datum}/x_list_{run}.pt')
                    acc_list=torch.load(f'./graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')

                    acc_list[:,:, 0,:] = acc_list[:,:, 0,:] / ynorm[4]
                    acc_list[:,:, 1,:] = acc_list[:,:, 1,:] / ynorm[5]

                    optimizer.zero_grad()
                    loss = 0

                    for loop in range(10):

                        k = np.random.randint(niter - 1)

                        edges = torch.load(f'./graphs_data/graphs_particles_{datum}/edge_{run}_{k}.pt')

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

                print("Epoch {}. Loss: {:.4f}".format(epoch, np.mean(total_loss)/nparticles ))

                if (np.mean(total_loss) < best_loss):
                    best_loss = np.mean(total_loss)
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_trained_with_{gridsearch}_graph.pt'))
                    print('\t\t Saving model')
                # torch.save({'model_state_dict': model.state_dict(),
                #             'optimizer_state_dict': optimizer.state_dict()},
                #            os.path.join(log_dir, 'models', f'model_epoch_{epoch}.pt'))

    # test
    if step==2:

        # model_config = {'ntry': 501,
        #             'input_size': 11,
        #             'output_size': 2,
        #             'hidden_size': 32,
        #             'n_mp_layers': 3,
        #             'noise_level': 0}

        # model_config = {'ntry': 506,
        #             'input_size': 11,
        #             'output_size': 2,
        #             'hidden_size': 32,
        #             'n_mp_layers': 3,
        #             'noise_level': 0}
        #
        # model_config = {'ntry': 507,
        #             'input_size': 11,
        #             'output_size': 2,
        #             'hidden_size': 32,
        #             'n_mp_layers': 3,
        #             'noise_level': 0}
        #
        # model_config = {'ntry': 508,
        #             'input_size': 11,
        #             'output_size': 2,
        #             'hidden_size': 32,
        #             'n_mp_layers': 3,
        #             'noise_level': 0}

        # model_config = {'ntry': 510,
        #                 'input_size': 8,
        #                 'output_size': 2,
        #                 'hidden_size': 32,
        #                 'n_mp_layers': 3,
        #                 'datum' : '230725',
        #                 'noise_level': 0}

        model_config = {'ntry': 511,
                        'input_size': 8,
                        'output_size': 2,
                        'hidden_size': 32,
                        'n_mp_layers': 3,
                        'datum' : '230802',
                        'noise_level': 0}

        # model_config = {'ntry': 512,
        #                 'input_size': 8,
        #                 'output_size': 2,
        #                 'hidden_size': 32,
        #                 'n_mp_layers': 3,
        #                 'datum' : '230802',
        #                 'noise_level': 0}

        datum=model_config['datum']

        ntry = model_config['ntry']
        print(f'ntry: {ntry}')

        grid_list = [1,2,5,10,20]

        for gridsearch in grid_list:
            print(datum)
            print(f"hidden_size: {model_config['hidden_size']}")

            model = InteractionParticles(model_config=model_config,device=device)
            state_dict = torch.load(f"./log/try_{ntry}/models/best_model_trained_with_{gridsearch}_graph.pt")
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

            ynorm = torch.load(f'./log/try_{ntry}/ynorm.pt')
            vnorm = torch.load(f'./log/try_{ntry}/vnorm.pt')



            GT = torch.load(f'./graphs_data/graphs_particles_{datum}/x_list_0.pt')
            GT0 = torch.squeeze (GT[0])

            acc_list = torch.load(f'graphs_data/graphs_particles_{datum}/acc_list_0.pt')
            acc_list[:, :, 0, :] = acc_list[:, :, 0, :] / ynorm[4]
            acc_list[:, :, 1, :] = acc_list[:, :, 1, :] / ynorm[5]


            edges = torch.load(f'./graphs_data/graphs_particles_{datum}/edge_0_0.pt')

            dataset = data.Data(x=GT0.cuda(), edge_index=edges.cuda())
            print('num_nodes: ', GT0.shape[0])
            nparticles = GT0.shape[0]
            print('dataset.num_node_features: ', dataset.num_node_features)

            x = GT0
            x = torch.permute(x, (2, 0, 1))
            x = torch.reshape(x, (nparticles * ntypes, 5))

            GT0 = x.clone()

            for it in tqdm(range(niter - 1)):

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


                if it%10==0:

                    fig = plt.figure(figsize=(25, 16))
                    # plt.ion()
                    ax = fig.add_subplot(2, 3, 1)
                    for k in range(ntypes):
                        plt.scatter(GT0[k * nparticles:(k + 1) * nparticles, 0].detach().cpu(),
                                    GT0[k * nparticles:(k + 1) * nparticles, 1].detach().cpu(), s=3)

                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.38, 'Distribution at t0 is 1.0x1.0')

                    ax = fig.add_subplot(2, 3, 2)
                    for k in range(ntypes):
                        plt.scatter(x0[k * nparticles:(k + 1) * nparticles, 0].detach().cpu(),
                                    x0[k * nparticles:(k + 1) * nparticles, 1].detach().cpu(), s=3)
                    ax = plt.gca()
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.38, 'True', fontsize=30)

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
                        plt.scatter(x[k * nparticles:(k + 1) * nparticles, 0].detach().cpu(),
                                    x[k * nparticles:(k + 1) * nparticles, 1].detach().cpu(), s=3)
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.38, 'Model', fontsize=30)

                    ax = fig.add_subplot(2, 3, 6)
                    temp1 = torch.cat((x, x0), 0)
                    pos = dict(enumerate(np.array(temp1[:, 0:2].detach().cpu()), 0))
                    temp2 = torch.tensor(np.arange(nparticles * ntypes), device=device)
                    temp3 = torch.tensor(np.arange(nparticles * ntypes) + nparticles * ntypes, device=device)
                    temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
                    temp4 = torch.t(temp4)
                    dataset = data.Data(x=temp1, edge_index=temp4)
                    vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                    nx.draw(vis, pos=pos, ax=ax, node_size=0, linewidths=0)
                    plt.xlim([-0.3, 1.3])
                    plt.ylim([-0.3, 1.3])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.axis('off')
                    plt.text(-0.25, 1.38, f'Frame: {it}')
                    rmserr = torch.sqrt(torch.mean(torch.sum((x - x0) ** 2, axis=1)))
                    plt.text(-0.25, 1.33, 'Prediction RMSE: {:.4f}'.format(rmserr.detach()), fontsize=10)

                    plt.savefig(f"./ReconsGraph/Gridsearch_{gridsearch}_Fig_{it}.tif")
                    plt.close()







