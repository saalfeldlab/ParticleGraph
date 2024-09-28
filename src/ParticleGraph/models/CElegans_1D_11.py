import torch
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread, imsave
from torch import nn
import torch.nn.functional as F
import os
from tqdm import trange
from Siren_Network import *
import GPUtil
import pandas as pd

def laplacian_1d(x):
    # Ensure the tensor is 1D
    assert x.dim() == 1, "Input tensor must be 1D"

    # Compute second-order finite differences
    laplacian = torch.zeros_like(x)
    laplacian[1:-1] = x[:-2] - 2 * x[1:-1] + x[2:]

    return laplacian

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

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def set_device(device=None):
    if device is None or device == 'auto':
        if torch.cuda.is_available():
            # Get the list of available GPUs and their free memory
            gpus = GPUtil.getGPUs()
            if gpus:
                # Find the GPU with the maximum free memory
                device_id = max(range(len(gpus)), key=lambda x: gpus[x].memoryFree)
                device = f'cuda:{device_id}'
            else:
                device = 'cpu'
        else:
            device = 'cpu'
    return device

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    return tensor.detach().cpu().numpy()

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

class CElegans_Laplacian(nn.Module):

    def __init__(self, config=None, device=None):

        super(CElegans_Laplacian, self).__init__()

        self.device = config['device']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.hidden_dim = config['hidden_dim']
        self.n_layers = config['n_layers']
        self.n_datasets = config['n_datasets']
        self.n_frames = config['n_frames']

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                           hidden_size=self.hidden_dim, device=self.device)


        self.alpha = nn.Parameter(torch.tensor(np.ones((int(self.n_datasets),1)), device=self.device,requires_grad=True, dtype=torch.float32))
        self.alpha_ = nn.Parameter(torch.tensor(np.ones((int(self.n_datasets), 99)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.a = nn.Parameter(torch.tensor(np.ones((int(self.n_datasets), 1)), device=self.device, requires_grad=True, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(np.ones((int(self.n_datasets), 1)), device=self.device, requires_grad=True, dtype=torch.float32))
        self.a_ = nn.Parameter(torch.tensor(np.ones((int(self.n_datasets), 99)), device=self.device, requires_grad=True, dtype=torch.float32))
        self.b_ = nn.Parameter(torch.tensor(np.ones((int(self.n_datasets), 99)), device=self.device, requires_grad=True, dtype=torch.float32))


    def forward(self, x, data_id, frame):

        u = x[:,0:1]
        du = x[:,1:2]
        laplacian_u = x[:,2:3]
        a = self.a_[data_id, :]
        b = self.b_[data_id, :]

        embedding = self.a[data_id, :].repeat(x.shape[0],1)
        alpha = self.alpha_[data_id, :]

        # frame = torch.tensor(frame, dtype=torch.float32, device=self.device)
        # in_features = torch.cat((x[:,0:2], embedding),1) #, frame.repeat(99,1)), 1)

        pred = alpha * laplacian_u.squeeze() + a * u.squeeze() + 0 * b * du.squeeze()

        return pred




if __name__ == '__main__':

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass
    device = set_device('auto')
    print(f'device {device}')

    path = '/groups/saalfeld/home/allierc/signaling/Celegans/Celegans_data'
    width = 99
    n_datasset = 12
    run = 11
    os.makedirs(f'/groups/saalfeld/home/allierc/signaling/Celegans/{run}', exist_ok=True)
    os.makedirs(f'/groups/saalfeld/home/allierc/signaling/Celegans/{run}/models', exist_ok=True)
    os.makedirs(f'/groups/saalfeld/home/allierc/signaling/Celegans/{run}/tmp_training', exist_ok=True)
    os.makedirs(f'/groups/saalfeld/home/allierc/signaling/Celegans/{run}/tmp_training/embedding', exist_ok=True)
    os.makedirs(f'/groups/saalfeld/home/allierc/signaling/Celegans/{run}/tmp_training/function', exist_ok=True)
    os.makedirs(f'/groups/saalfeld/home/allierc/signaling/Celegans/{run}/tmp_training/scatter', exist_ok=True)


    y=[]
    n_length = {}
    for k in range(n_datasset):
        data = np.array(pd.read_csv(f'{path}/wrm_{k+1}.csv'))
        y.append(torch.tensor(data, dtype=torch.float32, device=device))
        n_length[k] = data.shape[0]
    y=torch.stack(y)

    config = {}
    config['device'] = device
    config['input_size'] = 4
    config['output_size'] = 1
    config['hidden_dim'] = 256
    config['n_layers'] = 8
    config['n_datasets'] = 12
    config['n_frames'] = n_length[0]

    model = CElegans_Laplacian(config=config, device=device)

    # dy = torch.zeros_like(y)
    # ddy = torch.zeros_like(y)
    # Laplace_y = torch.zeros_like(y)
    # 
    # dy[:,1:,:] = y[:, 1:, :] - y[:, :-1, :]
    # ddy[:, 2:, :] = dy[:,2:,:] - dy[:,1:-1,:]
    # for k in trange(n_datasset-1):
    #     for i in range(1,n_length[k]-1):
    #         Laplace_y[k,i] = laplacian_1d(y[k,i])
    # 
    # torch.save(y, f'{path}/y.pt')
    # torch.save(dy, f'{path}/dy.pt')
    # torch.save(ddy, f'{path}/ddy.pt')
    # torch.save(Laplace_y, f'{path}/Laplace_y.pt')

    y = torch.load(f'{path}/y.pt', map_location=device)
    dy = torch.load(f'{path}/dy.pt', map_location=device)
    ddy = torch.load(f'{path}/ddy.pt', map_location=device)
    Laplace_y = torch.load(f'{path}/Laplace_y.pt', map_location=device)

    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    for epoch in range(200000):
        optimizer.zero_grad()
        flag_gd = False

        loss = 0
        batch_size = 32
        for batch in range(batch_size):

            k = 0 # np.random.randint(0, n_datasset)
            n = np.random.randint(10, n_length[k]-10)

            u = torch.cat((y[k,n][:,None], dy[k,n][:,None], Laplace_y[k,n][:,None]), 1)

            pred = model(u, k, n/n_length[k])
            target = ddy[k,n+1]

            l = (pred - target).norm(2)

            if not(torch.isnan(l).any()):
                loss += l
                flag_gd = True

        if flag_gd:
            loss.backward()
            optimizer.step()

        if epoch%1000 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()/batch_size}")
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       f"/groups/saalfeld/home/allierc/signaling/Celegans/{run}/models/model_{epoch}.pt")

            target_list=[]
            pred_list=[]
            k = 0
            for n in range(100, n_length[k]-100,50):
                u = torch.cat((y[k,n][:,None], dy[k,n][:,None], Laplace_y[k,n][:,None]), 1)
                pred = model(u, k, n/n_length[k])
                target = ddy[k, n+1]
                target_list.append(target)
                pred_list.append(pred)
            target = torch.stack(target_list)
            pred = torch.stack(pred_list)
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(to_numpy(target),to_numpy(pred), c='k', s=1, alpha=0.05)
            # plt.xlim([-1,1])
            # plt.ylim([-1,1])
            plt.savefig(f"/groups/saalfeld/home/allierc/signaling/Celegans/{run}/tmp_training/scatter/output_{epoch}.png", dpi=100)
            plt.close()

            # fig = plt.figure(figsize=(8, 8))
            # rr = torch.tensor(np.linspace(-20, 20, 1000), dtype=torch.float32, device=device)
            # for d in range(config['n_datasets'] ):
            #     embedding_ = model.a[d, :] * torch.ones((1000, 2), device=device)
            #     # frame = torch.tensor(0.5, dtype=torch.float32, device=device) * torch.ones((1000, 1), device=device)
            #     in_features = torch.cat((rr[:,None], torch.zeros_like(rr[:,None]), embedding_), 1)
            #     func = model.lin_phi(in_features)
            #     plt.plot(to_numpy(rr), to_numpy(func))
            # plt.tight_layout()
            # plt.savefig(f"/groups/saalfeld/home/allierc/signaling/Celegans/8/tmp_training/function/output_u_{epoch}.png", dpi=100)
            # plt.close()

            # fig = plt.figure(figsize=(8, 8))
            # plt.plot(to_numpy(model.alpha),'.',c='k', markersize=10)
            # plt.tight_layout()
            # plt.savefig(f"/groups/saalfeld/home/allierc/signaling/Celegans/8/tmp_training/embedding/alpha_{epoch}.png", dpi=100)
            # plt.close()

            # fig = plt.figure(figsize=(8, 8))
            # embedding_ = model.a
            # for d in range(config['n_datasets']):
            #     plt.scatter(to_numpy(embedding_[d,0:1]), to_numpy(embedding_[d,1:2]), s=100)
            # plt.tight_layout()
            # plt.savefig(f"/groups/saalfeld/home/allierc/signaling/Celegans/8/tmp_training/embedding/embedding_{epoch}.png", dpi=100)
            # plt.close()
















