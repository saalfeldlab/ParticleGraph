import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
from tqdm import tqdm
import ipywidgets as widgets
from matplotlib.animation import FuncAnimation
import os
import glob

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 500  # number of points per classes
    niter = 100

    NN=0
    flag=True

    datum='230329'

    folder=f'graphs_2_particles_{datum}'

    if not (os.path.exists(folder)):
        os.mkdir(folder)
    else:
        flist = ['Fig']
        for folder in flist:
            files = glob.glob(f"/{folder}/*")
            for f in files:
                os.remove(f)

    for run in range(10):

        print(NN)

        d = 2 # dimension
        X1 = torch.rand(n,d)
        X1 = X1.to(device);

        X2 = torch.rand(n,d)
        X2 = X2.to(device);

        boundary = 'no' # no boundary condition
        boundary = 'per' # periodic
        if boundary=='no': # change this for usual BC
            def bc_pos(X): return X
            def bc_diff(D): return D
        else:
            def bc_pos(X): return torch.remainder(X,1.0)
            def bc_diff(D): return torch.remainder(D-.5,1.0)-.5

        t = torch.tensor(np.linspace(-1.5,1.5,1000))

        if boundary=='no':
            tau = 1/500 # time step
        else:
            tau = 1/200

        save_per = 1 # periodicity of saving
        Zsvg1 = torch.zeros((n,d,niter//save_per)) # to store all the intermediate time
        Zsvg2 = torch.zeros((n, d, niter // save_per))  # to store all the intermediate time

        p1=torch.rand(1,4)
        p1 = torch.squeeze(p1)
        p1[0] = p1[0] + 1
        p1[1] = p1[1] + 1
        p2=torch.rand(1,4)
        p2 = torch.squeeze(p2)
        p2[0] = p2[0] + 1
        p2[1] = p2[1] + 1

        # p= torch.tensor([1.0,1.99,0.96,0.47])  # circle1
        # p1= torch.tensor([1.0,1.99,0.65,0.14])  # circle2
        # p2 = torch.tensor([1.0, 1.0, 0.5, 1])  # first
        # p = torch.tensor([1.3,1.44,0.83,0.16])  # lines
        # p= torch.tensor([1.01,1.3,0.86,0.63])  # 2*circles
        # p = torch.tensor([1.54, 1.43, 0.83, 0.3]) # cubes
        # p = torch.tensor([1.56 1.82 0.96 0.31]) # losanges

        p1= torch.tensor([1.23,1.59,0.1,0.87])
        p2 = torch.tensor([1.78, 1.6, 0.65, 0.38])

        p1=p1.to(device)
        p1=torch.round(p1, decimals=2)
        p2=p2.to(device)
        p2=torch.round(p2, decimals=2)

        X1.requires_grad = False
        X2.requires_grad = False

        rr = torch.tensor(np.linspace(0, 0.015, 100))
        rr = rr.to(device)
        psi1 = psi(rr,p1)
        psi2 = psi(rr, p2)

        # for it in tqdm(range(niter)):
        #     if np.mod(it,save_per)==0:
        #       Zsvg[:,:,it//save_per] = X.clone().detach() #
        #     L = -1 / X.shape[0] * torch.sum(psi(distmat_square(X, X), p), axis=(0, 1))
        #     [g] = torch.autograd.grad(L, [X])
        #     X = bc_pos( X - tau * g )

        for it in tqdm(range(niter)):

            if np.mod(it,save_per)==0:
              Zsvg1[:,:,it//save_per] = X1.clone().detach() # for later display
              Zsvg2[:, :, it // save_per] = X2.clone().detach()  # for later display

            Speed1=Speed(X1,torch.cat((X1,X2),0),p1)
            Speed2 = Speed(X2, torch.cat((X1, X2), 0), p2)
            X1 = bc_pos(X1 - tau * Speed1)
            X2 = bc_pos(X2 - tau * Speed2)

            distance=distmat_square(torch.cat((X1,X2),0),torch.cat((X1,X2),0))
            t = torch.Tensor([0.05*0.05]) # threshold
            adj_t = (distance < 0.05*0.05).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            edge_index=edge_index.detach().cpu()
            torch.save(edge_index,f'graphs_2_particles_{datum}/edge_index_{NN}.pt')

            temp1=torch.cat((X1.clone().detach(),X2.clone().detach()),0)
            temp2=torch.cat((Speed1.clone().detach(),Speed2.clone().detach()),0)
            X=torch.concatenate((temp1,temp2),1)
            X=X.detach().cpu()
            torch.save(X,f'graphs_2_particles_{datum}/X_{NN}.pt')

            label=torch.ones((X1.shape[0]*2))
            label[0:X1.shape[0]]=0
            torch.save(label,f'graphs_2_particles_{datum}/label_{NN}.pt')

            NN = NN+1

        c1 = np.array([220, 50, 32])/255
        c2 = np.array([0, 114, 178])/255

        def animate(t):
            fig.clf()
            plt.scatter(Zsvg1[:, 0, t], Zsvg1[:, 1, t]+0.08, s=3, color=c1)
            plt.scatter(Zsvg2[:, 0, t], Zsvg2[:, 1, t]+0.08, s=3, color=c2)
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.axis('equal')
            plt.axis([-0.3, 1.3, -0.3, 1.3])
            # plt.tight_layout()
            plt.text(-0.25, 1.33, 'sigma:.05 N:2000 niter:200')
            plt.text(-0.25, 1.25, f'p1: {np.array(p1.cpu())}',color=c1)
            plt.text(-0.25,1.2,f'p2: {np.array(p2.cpu())}', color=c2)
            plt.text(-0.25, 1.38, f'frame: {t}')

            ax = fig.add_subplot(5, 5, 21)
            plt.plot(np.array(psi1.cpu()),color=c1,linewidth=1)
            plt.plot(np.array(psi2.cpu()), color=c2,linewidth=1)
            plt.plot(np.array(0*np.linspace(0, 1, 100)), color=[0,0,0], linewidth=0.5)

            plt.tick_params(axis='x', labelsize=6)
            plt.tick_params(axis='y', labelsize=6)
            plt.ylabel('Interaction Kernel Psi(r)', fontsize=6)
            plt.xlabel('r [a.u]', fontsize=6)

        if run==0:
            fig = plt.figure(figsize=(9, 9))
            for t in tqdm(range(0,niter,1)):
               animate(t)
               plt.savefig(f"./temp/Fig_{NN}_{t}.tif")

        # ani = FuncAnimation(fig, animate, frames=niter, interval=0.1)
        # plt.show()



    # print('Save avi')

    # ani.save('try_animation.avi', fps=10, dpi=80)


