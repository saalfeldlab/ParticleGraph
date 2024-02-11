import numpy as np
import torch
import GPUtil


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
    Get min and max value of x if a certain percentage of the data is cut off from both ends.

    Args:
        x (np.ndarray): The data.
        percent (float): The percentage of data to cut off from both ends.
    """
    x_lower = np.percentile(x, percent)
    x_upper = np.percentile(x, 100 - percent)
    return x_lower, x_upper


def norm_velocity(xx, device):
    vx = torch.std(xx[:, 3])
    vy = torch.std(xx[:, 4])
    nvx = np.array(xx[:, 3].detach().cpu())
    vx01, vx99 = symmetric_cutoff(nvx, percent=1)
    nvy = np.array(xx[:, 4].detach().cpu())
    vy01, vy99 = symmetric_cutoff(nvy, percent=1)

    return torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)


def norm_acceleration(yy, device):
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = symmetric_cutoff(nax, percent=1)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = symmetric_cutoff(nay, percent=1)

    return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)
