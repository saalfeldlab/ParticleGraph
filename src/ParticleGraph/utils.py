import numpy as np
import torch
import GPUtil
import matplotlib.pyplot as plt


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    return tensor.detach().cpu().numpy()


def set_device(device=None):
    if device is None or device == 'auto':
        if torch.cuda.is_available():
            # Get the list of available GPUs and their free memory
            GPUs = GPUtil.getGPUs()
            if GPUs:
                # Find the GPU with the maximum free memory
                device_id = max(range(len(GPUs)), key=lambda x: GPUs[x].memoryFree)
                device = f'cuda:{device_id}'
            else:
                device = 'cpu'
        else:
            device = 'cpu'
    return device


def normalize99(Y, lower=1, upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01 + 1e-10)
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

        self.bMesh = 'Mesh' in model_config['model']

    def color(self, index):

        if self.model == 'PDE_E':

            if index == 0:
                index = (0, 0, 1)
            elif index == 1:
                index = (0, 0.5, 0.75)
            elif index == 2:
                index = (1, 0, 0)
            elif index == 3:
                index = (0.75, 0, 0)
            else:
                index = (0, 0, 0)
            return (index)
        elif self.bMesh:
            if index == 0:
                index = (0, 0, 0)
            else:
                color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
                index = color_map(index / self.nmap)

        else:
            # color_map = plt.cm.get_cmap(self.model_config['cmap'])
            color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
            index = color_map(index / self.nmap)

        return index
