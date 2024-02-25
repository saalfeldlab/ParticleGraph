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


def symmetric_cutoff(x, percent=1):
    """
    Minimum and maximum values if a certain percentage of the data is cut off from both ends.
    """
    x_lower = np.percentile(x, percent)
    x_upper = np.percentile(x, 100 - percent)
    return x_lower, x_upper


def norm_velocity(xx, device):
    vx = torch.std(xx[:, 3])
    vy = torch.std(xx[:, 4])
    nvx = np.array(xx[:, 3].detach().cpu())
    vx01, vx99 = symmetric_cutoff(nvx)
    nvy = np.array(xx[:, 4].detach().cpu())
    vy01, vy99 = symmetric_cutoff(nvy)

    return torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)


def norm_acceleration(yy, device):
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = symmetric_cutoff(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = symmetric_cutoff(nay)

    return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)


def choose_boundary_values(bc_name):
    def identity(x):
        return x

    def periodic(x):
        return torch.remainder(x, 1.0)  # in [0, 1)

    def shifted_periodic(x):
        return torch.remainder(x - 0.5, 1.0) - 0.5  # in [-0.5, 0.5)

    match bc_name:
        case 'no':
            return identity, identity
        case 'periodic':
            return periodic, shifted_periodic
        case _:
            raise ValueError(f'Unknown boundary condition {bc_name}')


def grads2D(params):

    params_sx = torch.roll(params, -1, 0)
    params_sy = torch.roll(params, -1, 1)

    sx = -(params - params_sx)
    sy = -(params - params_sy)

    sx[-1, :] = 0
    sy[:, -1] = 0

    return [sx,sy]


def tv2d(params):
    nb_voxel = (params.shape[0]) * (params.shape[1])
    t=params.detach().cpu()
    sx,sy= grads2D(t)

    tvloss = torch.sqrt(sx.cuda() ** 2 + sy.cuda() ** 2 + 1e-8).sum()
    # tvloss += torch.nn.functional.relu(-params).norm(1) / 15
    return tvloss / (nb_voxel)


class CustomColorMap:
    def __init__(self, config):
        self.cmap_name = config.plotting.colormap
        self.model_name = config.graph_model.particle_model_name

        if self.cmap_name == 'tab10':
            self.nmap = 8
        else:
            self.nmap = config.simulation.n_particles

        self.has_mesh = 'Mesh' in self.model_name

    def color(self, index):

        if self.model_name == 'PDE_E':
            match index:
                case 0:
                    color = (0, 0, 1)
                case 1:
                    color = (0, 0.5, 0.75)
                case 2:
                    color = (1, 0, 0)
                case 3:
                    color = (0.75, 0, 0)
                case _:
                    color = (0, 0, 0)
        elif self.has_mesh:
            if index == 0:
                color = (0, 0, 0)
            else:
                color_map = plt.colormaps.get_cmap(self.cmap_name)
                color = color_map(index / self.nmap)
        else:
            color_map = plt.colormaps.get_cmap(self.cmap_name)
            color = color_map(index)

        return color
