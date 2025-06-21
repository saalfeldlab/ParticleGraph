import glob
import logging
import os

import GPUtil
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from skimage.metrics import structural_similarity as ssim
from torch_geometric.data import Data
from torchvision.transforms import CenterCrop
import gc
from torch import cuda
import subprocess
import re

def sort_key(filename):
            # Extract the numeric parts using regular expressions
            if filename.split('_')[-2] == 'graphs':
                return 0
            else:
                return 1E7 * int(filename.split('_')[-2]) + int(filename.split('_')[-1][:-3])


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    return tensor.detach().cpu().numpy()


def set_device(device: str = 'auto'):
    """
    Set the device to use for computations. If 'auto' is specified, the device is chosen automatically:
     * if GPUs are available, the GPU with the most free memory is chosen
     * if MPS is available, MPS is used
     * otherwise, the CPU is used
    :param device: The device to use for computations. Automatically chosen if 'auto' is specified (default).
    :return: The torch.device object that is used for computations.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # Unset CUDA_VISIBLE_DEVICES

    if device == 'auto':
        if torch.cuda.is_available():
            try:
                # Use nvidia-smi to get free memory of each GPU
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                # Parse the output
                free_mem_list = []
                for line in result.strip().split('\n'):
                    index_str, mem_str = line.strip().split(',')
                    index = int(index_str)
                    free_mem = float(mem_str) * 1024 * 1024  # Convert MiB to bytes
                    free_mem_list.append((index, free_mem))
                # Ensure the device count matches
                num_gpus = torch.cuda.device_count()
                if num_gpus != len(free_mem_list):
                    print(f"Mismatch in GPU count between PyTorch ({num_gpus}) and nvidia-smi ({len(free_mem_list)})")
                    device = 'cpu'
                    print(f"Using device: {device}")
                else:
                    # Find the GPU with the most free memory
                    max_free_memory = -1
                    best_device_id = -1
                    for index, free_mem in free_mem_list:
                        if free_mem > max_free_memory:
                            max_free_memory = free_mem
                            best_device_id = index
                    if best_device_id == -1:
                        raise ValueError("Could not determine the GPU with the most free memory.")

                    device = f'cuda:{best_device_id}'
                    torch.cuda.set_device(best_device_id)  # Set the chosen device globally
                    total_memory_gb = torch.cuda.get_device_properties(best_device_id).total_memory / 1024 ** 3
                    free_memory_gb = max_free_memory / 1024 ** 3
                    print(
                        f"Using device: {device}, name: {torch.cuda.get_device_name(best_device_id)}, "
                        f"total memory: {total_memory_gb:.2f} GB, free memory: {free_memory_gb:.2f} GB")
            except Exception as e:
                print(f"Failed to get GPU information: {e}")
                device = 'cpu'
                print(f"Using device: {device}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"Using device: {device}")
        else:
            device = 'cpu'
            print(f"Using device: {device}")
    return device


def set_size(x, particles, mass_distrib_index):
    # particles = index_particles[n]

    #size = 5 * np.power(3, ((to_numpy(x[index_particles[n] , -2]) - 200)/100)) + 10
    size = np.power((to_numpy(x[particles, mass_distrib_index])), 1.2) / 1.5

    return size


def get_gpu_memory_map(device=None):
    t = np.round(torch.cuda.get_device_properties(device).total_memory / 1E9, 2)
    r = np.round(torch.cuda.memory_reserved(device) / 1E9, 2)
    a = np.round(torch.cuda.memory_allocated(device) / 1E9, 2)

    return t, r, a


def symmetric_cutoff(x, percent=1):
    """
    Minimum and maximum values if a certain percentage of the data is cut off from both ends.
    """
    x_lower = np.percentile(x, percent)
    x_upper = np.percentile(x, 100 - percent)
    return x_lower, x_upper


def norm_area(xx, device):

    pos = torch.argwhere(xx[:, -1]<1.0)
    ax = torch.std(xx[pos, -1])

    return torch.tensor([ax], device=device)


def norm_velocity(xx, dimension, device):
    if dimension == 2:
        vx = torch.std(xx[:, 3])
        vy = torch.std(xx[:, 4])
        nvx = np.array(xx[:, 3].detach().cpu())
        vx01, vx99 = symmetric_cutoff(nvx)
        nvy = np.array(xx[:, 4].detach().cpu())
        vy01, vy99 = symmetric_cutoff(nvy)
    else:
        vx = torch.std(xx[:, 4])
        vy = torch.std(xx[:, 5])
        vz = torch.std(xx[:, 6])
        nvx = np.array(xx[:, 4].detach().cpu())
        vx01, vx99 = symmetric_cutoff(nvx)
        nvy = np.array(xx[:, 5].detach().cpu())
        vy01, vy99 = symmetric_cutoff(nvy)
        nvz = np.array(xx[:, 6].detach().cpu())
        vz01, vz99 = symmetric_cutoff(nvz)

    # return torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)

    return torch.tensor([vx], device=device)


def norm_position(xx, dimension, device):
    if dimension == 2:
        bounding_box = get_2d_bounding_box(xx[:, 1:3]* 1.1)
        posnorm = max(bounding_box.values())

        return torch.tensor(posnorm, dtype=torch.float32, device=device), torch.tensor([bounding_box['x_max']/posnorm, bounding_box['y_max']/posnorm], dtype=torch.float32, device=device)
    else:

        bounding_box = get_3d_bounding_box(xx[:, 1:4]* 1.1)
        posnorm = max(bounding_box.values())

        return torch.tensor(posnorm, dtype=torch.float32, device=device), torch.tensor([bounding_box['x_max']/posnorm, bounding_box['y_max']/posnorm, bounding_box['z_max']/posnorm], dtype=torch.float32, device=device)


def norm_acceleration(yy, device):
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = symmetric_cutoff(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = symmetric_cutoff(nay)

    # return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)

    return torch.tensor([ax], device=device)


def choose_boundary_values(bc_name):
    def identity(x):
        return x

    def periodic(x):
        return torch.remainder(x, 1.0)

    def periodic_wall(x):
        y = torch.remainder(x[:,0:1], 1.0)
        return torch.cat((y,x[:,1:2]), 1)

    def shifted_periodic(x):
        return torch.remainder(x - 0.5, 1.0) - 0.5

    def shifted_periodic_wall(x):
        y = torch.remainder(x[:,0:1] - 0.5, 1.0) - 0.5
        return torch.cat((y,x[:,1:2]), 1)


    match bc_name:
        case 'no':
            return identity, identity
        case 'periodic':
            return periodic, shifted_periodic
        case 'wall':
            return periodic_wall, shifted_periodic_wall
        case _:
            raise ValueError(f'unknown boundary condition {bc_name}')


def grads2D(params):
    params_sx = torch.roll(params, -1, 0)
    params_sy = torch.roll(params, -1, 1)

    sx = -(params - params_sx)
    sy = -(params - params_sy)

    sx[-1, :] = 0
    sy[:, -1] = 0

    return [sx, sy]


def tv2D(params):
    nb_voxel = (params.shape[0]) * (params.shape[1])
    sx, sy = grads2D(params)
    tvloss = torch.sqrt(sx.cuda() ** 2 + sy.cuda() ** 2 + 1e-8).sum()
    return tvloss / nb_voxel


def density_laplace(y, x):
    grad = density_gradient(y, x)
    return density_divergence(grad, x)


def density_divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def density_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compute_angle(tensor1, tensor2):
    # Ensure the tensors are 2D
    assert tensor1.shape == tensor2.shape == (2,), "Tensors must be 2D vectors"

    # Compute the dot product
    dot_product = torch.dot(tensor1, tensor2)

    # Compute the magnitudes (norms) of the tensors
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    # Compute the cosine of the angle
    cos_angle = dot_product / (norm1 * norm2)

    # Compute the angle in radians
    angle = torch.acos(cos_angle)

    return angle * 180 / np.pi


def compute_signed_angle(tensor1, tensor2):
    # Ensure the tensors are 2D
    assert tensor1.shape == tensor2.shape == (2,), "Tensors must be 2D vectors"

    # Compute the dot product
    dot_product = torch.dot(tensor1, tensor2)

    # Compute the magnitudes (norms) of the tensors
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    # Compute the cosine of the angle
    cos_angle = dot_product / (norm1 * norm2)

    # Compute the angle in radians
    angle = torch.acos(cos_angle)

    # Compute the sign of the angle using the cross product
    cross_product = tensor1[0] * tensor2[1] - tensor1[1] * tensor2[0]
    sign = torch.sign(cross_product)

    # Return the signed angle
    return angle * sign * 180 / np.pi


def get_r2_numpy_corrcoef(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2


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

        if ('PDE_F' in self.model_name) | ('PDE_M' in self.model_name) | ('PDE_MLPs' in self.model_name):
            match index:
                case 0:
                    color = (0.75, 0.75, 0.75)
                case 1:
                    color = (0, 0.5, 0.75)
                case 2:
                    color = (1, 0, 0)
                case 3:
                    color = (0.5, 0.75, 0)
                case 4:
                    color = (0, 0.75, 0)
                case 5:
                    color = (0.5, 0, 0.25)
                case _:
                    color = (1, 1, 1)
        elif self.model_name == 'PDE_E':
            match index:
                case 0:
                    color = (1, 1, 1)
                case 1:
                    color = (0, 0.5, 0.75)
                case 2:
                    color = (1, 0, 0)
                case 3:
                    color = (0.75, 0, 0)
                case _:
                    color = (0.5, 0.5, 0.5)
        elif self.has_mesh:
            if index == 0:
                color = (0, 0, 0)
            else:
                color_map = plt.colormaps.get_cmap(self.cmap_name)
                color = color_map(index / self.nmap)
        else:
            color_map = plt.colormaps.get_cmap(self.cmap_name)
            if self.cmap_name == 'tab20':
                color = color_map(index % 20)
            else:
                color = color_map(index)

        return color


def load_image(path, crop_width=None, device='cpu'):
    target = imageio.v2.imread(path).astype(np.float32)
    target = target / np.max(target)
    target = torch.tensor(target).unsqueeze(0).to(device)
    if crop_width is not None:
        target = CenterCrop(crop_width)(target)
    return target


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def calculate_psnr(img1, img2, max_value=255):
    """Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def calculate_ssim(img1, img2):
    ssim_score = ssim(img1, img2, data_range=img2.max() - img2.min())
    return ssim_score


def add_pre_folder(config_file_):

    if 'arbitrary' in config_file_:
        config_file = os.path.join('arbitrary', config_file_)
        pre_folder = 'arbitrary/'
    elif 'boids' in config_file_:
        config_file = os.path.join('boids', config_file_)
        pre_folder = 'boids/'
    elif 'Coulomb' in config_file_:
        config_file = os.path.join('Coulomb', config_file_)
        pre_folder = 'Coulomb/'
    elif 'fluids' in config_file_:
        config_file = os.path.join('fluids', config_file_)
        pre_folder = 'fluids/'
    elif 'gravity' in config_file_:
        config_file = os.path.join('gravity', config_file_)
        pre_folder = 'gravity/'
    elif 'springs' in config_file_:
        config_file = os.path.join('springs', config_file_)
        pre_folder = 'springs/'
    elif 'CElegans' in config_file_:
        config_file = os.path.join('CElegans', config_file_)
        pre_folder = 'CElegans/'
    elif 'fly' in config_file_:
        config_file = os.path.join('fly', config_file_)
        pre_folder = 'fly/'
    elif 'signal' in config_file_:
        config_file = os.path.join('signal', config_file_)
        pre_folder = 'signal/'
    elif 'falling_water_ramp' in config_file_:
        config_file = os.path.join('falling_water_ramp', config_file_)
        pre_folder = 'falling_water_ramp/'
    elif 'multimaterial' in config_file_:
        config_file = os.path.join('multimaterial', config_file_)
        pre_folder = 'multimaterial/'
    elif 'RD_RPS' in config_file_:
        config_file = os.path.join('reaction_diffusion', config_file_)
        pre_folder = 'reaction_diffusion/'
    elif 'wave' in config_file_:
        config_file = os.path.join('wave', config_file_)
        pre_folder = 'wave/'
    elif ('cell' in config_file_) | ('cardio' in config_file_) | ('U2OS' in config_file_):
        config_file = os.path.join('cell', config_file_)
        pre_folder = 'cell/'
    elif 'mouse' in config_file_:
        config_file = os.path.join('mouse_city', config_file_)
        pre_folder = 'mouse_city/'
    elif 'rat' in config_file_:
        config_file = os.path.join('rat_city', config_file_)
        pre_folder = 'rat_city/'

    return config_file, pre_folder


def get_log_dir(config=[]):

    if 'PDE_A' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/arbitrary/')
    elif 'PDE_B' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/boids/')
    elif 'PDE_E' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/Coulomb/')
    elif 'PDE_F' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/fluids/')
    elif 'PDE_G' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/gravity/')
    elif 'PDE_K' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/springs/')
    elif 'PDE_N' in config.graph_model.signal_model_name:
        l_dir = os.path.join('./log/signal/')
    elif 'PDE_M' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/multimaterial/')
    elif 'PDE_MLPs' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/multimaterial/')
    elif 'RD_RPS' in config.graph_model.mesh_model_name:
        l_dir = os.path.join('./log/reaction_diffusion/')
    elif 'Wave' in config.graph_model.mesh_model_name:
        l_dir = os.path.join('./log/wave/')
    elif 'cell' in config.dataset:
        l_dir = os.path.join('./log/cell/')
    elif 'mouse' in config.dataset:
        l_dir = os.path.join('./log/mouse/')
    elif 'rat' in config.dataset:
        l_dir = os.path.join('./log/rat/')
    elif 'celegans' in config.dataset:
        l_dir = os.path.join('./log/celegans/')

    return l_dir


def create_log_dir(config=[], erase=True):

    log_dir = os.path.join('.', 'log', config.config_file)
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/particle'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/field'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/matrix'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/prediction'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/lin_phi'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/lin_edge'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)
    if config.training.n_ghosts > 0:
        os.makedirs(os.path.join(log_dir, 'tmp_training/ghost'), exist_ok=True)

    if erase:
        files = glob.glob(f"{log_dir}/results/*")
        for f in files:
            if ('all' not in f) & ('field' not in f):
                os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/particle/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/field/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/matrix/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/function/lin_edge/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/function/lin_phis/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/embedding/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/ghost/*")
        for f in files:
            os.remove(f)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    return log_dir, logger


def bundle_fields(data: Data, *names: str) -> torch.Tensor:
    tensors = []
    for name in names:
        tensor = data[name]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        tensors.append(tensor)
    return torch.concatenate(tensors, dim=-1)


def fig_init(fontsize=48, formatx='%.2f', formaty='%.2f'):
    # from matplotlib import rc, font_manager
    # from numpy import arange, cos, pi
    # from matplotlib.pyplot import figure, axes, plot, xlabel, ylabel, title, \
    #     grid, savefig, show
    # sizeOfFont = 12
    # fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
    #                   'weight': 'normal', 'size': sizeOfFont}
    # ticks_font = font_manager.FontProperties(family='sans-serif', style='normal',
    #                                          size=sizeOfFont, weight='normal', stretch='normal')
    # rc('text', usetex=True)
    # rc('font', **fontProperties)
    # figure(1, figsize=(6, 4))
    # ax = axes([0.1, 0.1, 0.8, 0.7])
    # t = arange(0.0, 1.0 + 0.01, 0.01)
    # s = cos(2 * 2 * pi * t) + 2
    # plot(t, s)
    # for label in ax.get_xticklabels():
    #     label.set_fontproperties(ticks_font)
    # for label in ax.get_yticklabels():
    #     label.set_fontproperties(ticks_font)
    # xlabel(r'\textbf{time (s)}')
    # ylabel(r'\textit{voltage (mV)}', fontsize=16, family='Helvetica')
    # title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
    #       fontsize=16, color='r')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks([])
    plt.yticks([])
    # ax.xaxis.get_major_formatter()._usetex = False
    # ax.yaxis.get_major_formatter()._usetex = False
    ax.tick_params(axis='both', which='major', pad=15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
    ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return fig, ax


def get_time_series(x_list, cell_id, feature):
    match feature:
        case 'mass':
            feature = 10
        case 'velocity_x':
            feature = 3
        case 'velocity_y':
            feature = 4
        case "type":
            feature = 5
        case "stage":
            feature = 9
        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))
    return to_numpy(torch.stack(time_series))


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def get_sorted_image_files(pic_folder, pic_format):
    # Check if the folder exists
    if not os.path.exists(pic_folder):
        raise FileNotFoundError(f"The folder `{pic_folder}` does not exist.")

    # Get the list of image files with the specified format
    image_files = glob.glob(os.path.join(pic_folder, f"*.{pic_format}"))

    # Sort the files based on the number in the filename
    image_files.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1)))

    return image_files


def extract_number(filename):
    match = re.search(r'0-(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else None


def check_and_clear_memory(
        device: str = None,
        iteration_number: int = None,
        every_n_iterations: int = 100,
        memory_percentage_threshold: float = 0.6
):
    """
    Check the memory usage of a GPU and clear the cache every n iterations or if it exceeds a certain threshold.
    :param device: The device to check the memory usage for.
    :param iteration_number: The current iteration number.
    :param every_n_iterations: Clear the cache every n iterations.
    :param memory_percentage_threshold: Percentage of memory usage that triggers a clearing.
    """

    if device and 'cuda' in device:
        logger = logging.getLogger(__name__)



        if (iteration_number % every_n_iterations == 0):
            # logger.info(f"Recurrent cuda cleanining")
            # logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            torch.cuda.memory_allocated(device)
            gc.collect()
            torch.cuda.empty_cache()

            if (iteration_number==0):
                logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
                logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")



        elif torch.cuda.memory_allocated(device) > memory_percentage_threshold * torch.cuda.get_device_properties(device).total_memory:
            # logger.info("Memory usage is high. Calling garbage collector and clearing cache.")
            # logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
            gc.collect()
            torch.cuda.empty_cache()

def large_tensor_nonzero(tensor, chunk_size=2**30):
    indices = []
    num_chunks = (tensor.numel() + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        chunk = tensor.flatten()[i * chunk_size:(i + 1) * chunk_size]
        chunk_indices = chunk.nonzero(as_tuple=True)[0] + i * chunk_size
        indices.append(chunk_indices)
    indices = torch.cat(indices)
    row_indices = indices // tensor.size(1)
    col_indices = indices % tensor.size(1)
    return torch.stack([row_indices, col_indices])

def get_equidistant_points(n_points=1024):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    r = np.sqrt(indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y = r * np.cos(theta), r * np.sin(theta)

    return x, y

def get_matrix_rank(matrix):
    return np.linalg.matrix_rank(matrix)


def map_matrix(neuron_list, neuron_names, matrix):

    map_list = np.zeros(len(neuron_list), dtype=int)
    for i, neuron_name in enumerate(neuron_list):
        if neuron_name in list(neuron_names):
            index = list(neuron_names).index(neuron_name)
            map_list[i] = index
        else:
            map_list[i] = 0

    mapped_matrix = matrix[np.ix_(map_list, map_list)]

    for i, neuron_name in enumerate(neuron_list):
        if neuron_name not in list(neuron_names):
            mapped_matrix[i, :] = 0
            mapped_matrix[:, i] = 0

    return mapped_matrix, map_list


def get_neuron_index(neuron_name, activity_neuron_list):
    """
    Returns the index of the neuron_name in activity_neuron_list.
    Raises ValueError if not found.
    """
    try:
        return activity_neuron_list.index(neuron_name)
    except ValueError:
        raise ValueError(f"Neuron '{neuron_name}' not found in activity_neuron_list.")
# Example usage
# matrix = np.random.rand(100, 100)
# rank = get_matrix_rank(matrix)
# print(f"The rank of the matrix is: {rank}")

def compute_spectral_density(matrix, bins=100):
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Create histogram
    density, edges = np.histogram(eigenvalues, bins=bins, density=True)

    # Compute bin centers
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, density


def get_2d_bounding_box(xx):

    x_min, y_min = torch.min(xx, dim=0).values
    x_max, y_max = torch.max(xx, dim=0).values

    bounding_box = {
        'x_min': x_min.item(),
        'x_max': x_max.item(),
        'y_min': y_min.item(),
        'y_max': y_max.item()
    }

    return bounding_box


def get_3d_bounding_box(xx):

    x_min, y_min, z_min = torch.min(xx, dim=0).values
    x_max, y_max, z_max = torch.max(xx, dim=0).values

    bounding_box = {
        'x_min': x_min.item(),
        'x_max': x_max.item(),
        'y_min': y_min.item(),
        'y_max': y_max.item(),
        'z_min': z_min.item(),
        'z_max': z_max.item()
    }

    return bounding_box


def get_top_fft_modes_per_pixel(im0, dt=1.0, top_n=3):
    """
    Compute the top N Fourier modes for each pixel and channel in a 4D time series image stack.

    Parameters:
        im0 (ndarray): shape (T, H, W, C)
        dt (float): time step between frames
        top_n (int): number of top frequency modes to return

    Returns:
        top_freqs (ndarray): shape (top_n, H, W, C), top frequencies per pixel/channel
        top_amps (ndarray): shape (top_n, H, W, C), corresponding amplitudes
    """
    T, H, W, C = im0.shape

    # Compute FFT frequencies
    freqs = np.fft.fftfreq(T, d=dt)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]

    # Compute FFT along time axis
    fft_vals = np.fft.fft(im0, axis=0)              # shape: (T, H, W, C)
    fft_mag = np.abs(fft_vals)[pos_mask, :, :, :]   # shape: (T_pos, H, W, C)

    # Get indices of top N frequencies per pixel/channel
    top_indices = np.argsort(fft_mag, axis=0)[-top_n:][::-1]  # shape: (top_n, H, W, C)

    # Gather top frequencies and amplitudes
    top_freqs = pos_freqs[top_indices]      # broadcast: top_n x H x W x C
    top_amps = np.take_along_axis(fft_mag, top_indices, axis=0)

    return top_freqs, top_amps


import torch.nn.functional as F


def total_variation_norm(im):
    # Compute the differences along the x-axis (horizontal direction)
    dx = im[:, 1:, :] - im[:, :-1, :]  # (batch, height-1, width, channels)

    # Compute the differences along the y-axis (vertical direction)
    dy = im[1:, :, :] - im[:-1, :, :]  # (batch, height, width-1, channels)

    # Sum squared differences and take the square root (L2 norm)
    tv_x = torch.sqrt(torch.sum(dx ** 2))  # Sum along channels
    tv_y = torch.sqrt(torch.sum(dy ** 2))  # Sum along channels

    # Total variation is the sum of x and y contributions
    return tv_x + tv_y


def check_file_exists(dataset_name):
    file_path = f'graphs_data/graphs_{dataset_name}/connection_matrix_list.pt'
    return os.path.isfile(file_path)


def find_suffix_pairs_with_index(neuron_list, suffix1, suffix2):
    pairs = []
    for i, neuron in enumerate(neuron_list):
        if neuron.endswith(suffix1):
            base_name = neuron[:-1]
            target_name = base_name + suffix2
            for j, other_neuron in enumerate(neuron_list):
                if other_neuron == target_name:
                    pairs.append(((i, neuron), (j, other_neuron)))
                    break  # Stop after finding the first match
    return pairs

def fit_polynomial(x, y, degree=3):
    """
    Fit a polynomial of given degree to data x, y.

    Parameters:
        x: numpy array or tensor of shape (N,) or (N, 1)
        y: numpy array or tensor of shape (N,) or (N, 1)
        degree: int, degree of polynomial to fit (default=3)

    Returns:
        poly: numpy.poly1d object representing the fitted polynomial
    """
    # Convert to 1D numpy arrays
    x_np = np.ravel(x)
    y_np = np.ravel(y)

    # Fit polynomial
    coeffs = np.polyfit(x_np, y_np, deg=degree)

    # Create polynomial object
    poly = np.poly1d(coeffs)

    return poly

def fit_polynomial_with_latex(x, y, degree=3):
    """
    Fit a polynomial of given degree to data x, y,
    and return a LaTeX formatted string of the polynomial.

    Parameters:
        x: numpy array or tensor of shape (N,) or (N, 1)
        y: numpy array or tensor of shape (N,) or (N, 1)
        degree: int, degree of polynomial to fit (default=3)

    Returns:
        poly: numpy.poly1d object representing the fitted polynomial
        latex_str: str, LaTeX formatted polynomial expression
    """
    x_np = np.ravel(x)
    y_np = np.ravel(y)

    coeffs = np.polyfit(x_np, y_np, deg=degree)
    poly = np.poly1d(coeffs)

    # Build LaTeX string
    terms = []
    for power, coeff in enumerate(coeffs[::-1]):
        # power = 0 means constant term
        # coeff for the current power
        if abs(coeff) < 1e-12:
            continue  # skip negligible coefficients

        coeff_str = f"{abs(coeff):.4g}"
        if power == 0:
            term = f"{coeff_str}"
        elif power == 1:
            term = f"{coeff_str} x"
        else:
            term = f"{coeff_str} x^{power}"

        # Add sign
        sign = "-" if coeff < 0 else "+"
        terms.append((sign, term))

    # First term sign handling
    if terms:
        first_sign, first_term = terms[0]
        if first_sign == "+":
            latex_expr = first_term
        else:
            latex_expr = first_sign + " " + first_term

        for sign, term in terms[1:]:
            latex_expr += f" {sign} {term}"
    else:
        latex_expr = "0"

    latex_str = f"${latex_expr}$"

    return poly, latex_str


import numpy as np
from collections import defaultdict


def reconstruct_time_series_from_xlist(x_list):
    """
    Reconstruct time series data from x_list with proper track ID mapping.

    Parameters:
    x_list: List of arrays, each containing frame data with columns:
            [track_ID, y_pos, x_pos, vel_y, vel_x, frame, fluo, fluo_unused1, fluo_unused2]

    Returns:
    time_series_dict: Dictionary {track_ID: [[frame, fluo_value], ...]}
    track_info_dict: Dictionary {track_ID: {'positions': [[y,x], ...], 'frames': [...]}}
    """

    time_series_dict = defaultdict(list)
    track_info_dict = defaultdict(lambda: {'positions': [], 'frames': []})

    for frame_idx, x_frame in enumerate(x_list):
        if x_frame.size == 0:  # Skip empty frames
            continue

        for i in range(x_frame.shape[0]):
            track_id = int(x_frame[i, 0])

            # Skip invalid track IDs
            if track_id < 0:
                continue

            frame_number = int(x_frame[i, 5])
            fluo_value = x_frame[i, 6]  # Only column 6 contains actual fluorescence data
            y_pos = x_frame[i, 1]
            x_pos = x_frame[i, 2]

            # Append to time series
            time_series_dict[track_id].append([frame_number, fluo_value])

            # Store position and frame info
            track_info_dict[track_id]['positions'].append([y_pos, x_pos])
            track_info_dict[track_id]['frames'].append(frame_number)

    time_series_dict = dict(time_series_dict)
    track_info_dict = dict(track_info_dict)

    for track_id in time_series_dict:
        # Sort by frame number
        time_series_dict[track_id] = sorted(time_series_dict[track_id], key=lambda x: x[0])

        # Convert to numpy arrays for easier handling
        time_series_dict[track_id] = np.array(time_series_dict[track_id])
        track_info_dict[track_id]['positions'] = np.array(track_info_dict[track_id]['positions'])
        track_info_dict[track_id]['frames'] = np.array(track_info_dict[track_id]['frames'])

    print(f'found {len(time_series_dict)} time_series')

    return time_series_dict, track_info_dict


def filter_tracks_by_length(time_series_dict, min_length=100):
    """
    Filter tracks to keep only those with sufficient length for Granger analysis.

    Parameters:
    time_series_dict: Dictionary from reconstruct_time_series_from_xlist
    min_length: Minimum number of time points required

    Returns:
    filtered_dict: Dictionary with only tracks meeting length requirement
    """
    filtered_dict = {}

    for track_id, time_series in time_series_dict.items():
        if len(time_series) >= min_length:
            filtered_dict[track_id] = time_series

    print(f"kept {len(filtered_dict)} tracks out of {len(time_series_dict)} total tracks")
    # print(f"Track lengths: {[len(ts) for ts in filtered_dict.values()]}")

    return filtered_dict

from sklearn.linear_model import LinearRegression
from scipy import stats

import warnings

warnings.filterwarnings('ignore')


def find_average_spatial_neighbors(track_info_dict, max_radius=50, min_radius=0, device='cpu'):
    """
    Find spatial nearest neighbors between tracks using average positions.

    Parameters:
    track_info_dict: Dictionary with track positions from reconstruct_time_series_from_xlist
    max_radius: Maximum distance for neighbors (pixels)
    min_radius: Minimum distance for neighbors (pixels)
    device: 'cpu' or 'cuda'

    Returns:
    neighbor_pairs: List of (track_id1, track_id2) tuples
    track_positions: Dictionary {track_id: [avg_y, avg_x]}
    """

    # Extract track IDs and compute average positions
    track_ids = list(track_info_dict.keys())
    positions = []

    for track_id in track_ids:
        pos_array = track_info_dict[track_id]['positions']
        avg_pos = np.mean(pos_array, axis=0)  # [avg_y, avg_x]
        positions.append(avg_pos)

    # Convert to torch tensors
    track_ids = np.array(track_ids)
    positions = torch.tensor(np.array(positions), dtype=torch.float32, device=device)

    print(f"computing distances for {len(track_ids)} tracks...")

    # Your distance calculation approach (adapted)
    dimension = 2  # y, x coordinates
    distance = torch.sum((positions[:, None, :] - positions[None, :, :]) ** 2, dim=2)

    # Create adjacency matrix
    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float()

    # Get edge indices
    edge_index = adj_t.nonzero().t().contiguous()

    # Convert back to track ID pairs
    neighbor_pairs = []
    for i in range(edge_index.shape[1]):
        idx1, idx2 = edge_index[0, i].item(), edge_index[1, i].item()
        track_id1, track_id2 = track_ids[idx1], track_ids[idx2]
        neighbor_pairs.append((track_id1, track_id2))

    # Create position lookup
    track_positions = {track_ids[i]: positions[i].cpu().numpy() for i in range(len(track_ids))}

    print(f"found {len(neighbor_pairs)} neighbor pairs within radius {max_radius}")

    return neighbor_pairs, track_positions


def fit_ar_model_with_bic(time_series, max_order=10):
    """
    Fit AR model with optimal order selection using BIC.

    Parameters:
    time_series: 1D array of time series values
    max_order: Maximum AR order to test

    Returns:
    best_order: Optimal AR order
    coefficients: AR coefficients for best model
    residuals: Residuals from best model
    bic_scores: BIC scores for all tested orders
    """

    if len(time_series) < max_order + 10:  # Need sufficient data
        return None, None, None, None

    bic_scores = []
    models = []

    # Test different AR orders
    for p in range(1, min(max_order + 1, len(time_series) // 3)):
        try:
            # Create lagged features
            X, y = create_ar_features(time_series, p)

            if X.shape[0] < p + 5:  # Need sufficient samples
                bic_scores.append(np.inf)
                models.append(None)
                continue

            # Fit linear regression (AR model)
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)

            # Calculate predictions and residuals
            y_pred = model.predict(X)
            residuals = y - y_pred

            # Calculate BIC
            n = len(y)
            mse = np.mean(residuals ** 2)
            log_likelihood = -0.5 * n * np.log(2 * np.pi * mse) - 0.5 * np.sum(residuals ** 2) / mse
            bic = -2 * log_likelihood + (p + 1) * np.log(n)  # +1 for intercept

            bic_scores.append(bic)
            models.append((model, residuals))

        except Exception as e:
            bic_scores.append(np.inf)
            models.append(None)

    if not any(score != np.inf for score in bic_scores):
        return None, None, None, None

    # Find best order
    best_idx = np.argmin(bic_scores)
    best_order = best_idx + 1
    best_model, best_residuals = models[best_idx]

    return best_order, best_model.coef_, best_residuals, bic_scores


def create_ar_features(time_series, order):
    """
    Create lagged features for AR model.

    Parameters:
    time_series: 1D array
    order: AR order (number of lags)

    Returns:
    X: Feature matrix [n_samples, order]
    y: Target vector [n_samples]
    """
    n = len(time_series)
    X = np.zeros((n - order, order))

    for i in range(order):
        X[:, i] = time_series[order - 1 - i:n - 1 - i]

    y = time_series[order:]

    return X, y


def fit_granger_models(ts1, ts2, max_order=10):
    """
    Fit both restricted and unrestricted models for Granger causality test.

    Parameters:
    ts1: Time series 1 (potential cause)
    ts2: Time series 2 (potential effect)
    max_order: Maximum AR order to test

    Returns:
    results: Dictionary with model results
    """

    # Ensure same length
    min_len = min(len(ts1), len(ts2))
    ts1 = ts1[:min_len]
    ts2 = ts2[:min_len]

    # Find optimal order for restricted model (ts2 only)
    best_order, _, _, _ = fit_ar_model_with_bic(ts2, max_order)

    if best_order is None:
        return None

    try:
        # Restricted model: ts2(t) = f(ts2(t-1), ts2(t-2), ..., ts2(t-p))
        X_restricted, y = create_ar_features(ts2, best_order)
        model_restricted = LinearRegression(fit_intercept=True)
        model_restricted.fit(X_restricted, y)
        residuals_restricted = y - model_restricted.predict(X_restricted)

        # Unrestricted model: ts2(t) = f(ts2(t-1), ..., ts2(t-p), ts1(t-1), ..., ts1(t-p))
        X_ts1, _ = create_ar_features(ts1, best_order)
        X_unrestricted = np.column_stack([X_restricted, X_ts1])

        model_unrestricted = LinearRegression(fit_intercept=True)
        model_unrestricted.fit(X_unrestricted, y)
        residuals_unrestricted = y - model_unrestricted.predict(X_unrestricted)

        return {
            'order': best_order,
            'residuals_restricted': residuals_restricted,
            'residuals_unrestricted': residuals_unrestricted,
            'n_samples': len(y)
        }

    except Exception as e:
        return None


def calculate_granger_causality(residuals_restricted, residuals_unrestricted):
    """
    Calculate Granger causality metric: GC = log(RSS_restricted / RSS_unrestricted)

    Parameters:
    residuals_restricted: Residuals from AR model (effect only)
    residuals_unrestricted: Residuals from VAR model (effect + cause)

    Returns:
    gc_value: Granger causality metric
    """
    rss_restricted = np.sum(residuals_restricted ** 2)
    rss_unrestricted = np.sum(residuals_unrestricted ** 2)

    if rss_unrestricted == 0:
        return 0

    gc_value = np.log(rss_restricted / rss_unrestricted)
    return gc_value


def compute_granger_difference(ts1, ts2, max_order=10):
    """
    Compute bidirectional Granger causality and difference.

    Parameters:
    ts1, ts2: Time series arrays (fluorescence values)
    max_order: Maximum AR order for BIC selection

    Returns:
    gc_12: Granger causality ts1 -> ts2
    gc_21: Granger causality ts2 -> ts1
    granger_diff: |gc_12 - gc_21|
    direction: 1 if ts1->ts2 stronger, 2 if ts2->ts1 stronger
    """

    # Test ts1 -> ts2
    result_12 = fit_granger_models(ts1, ts2, max_order)
    if result_12 is None:
        return None, None, None, None

    gc_12 = calculate_granger_causality(
        result_12['residuals_restricted'],
        result_12['residuals_unrestricted']
    )

    # Test ts2 -> ts1
    result_21 = fit_granger_models(ts2, ts1, max_order)
    if result_21 is None:
        return None, None, None, None

    gc_21 = calculate_granger_causality(
        result_21['residuals_restricted'],
        result_21['residuals_unrestricted']
    )

    # Compute difference and direction
    granger_diff = abs(gc_12 - gc_21)
    direction = 1 if gc_12 > gc_21 else 2

    return gc_12, gc_21, granger_diff, direction


def analyze_neighbor_pairs(neighbor_pairs, filtered_time_series, max_order=10):
    """
    Compute Granger causality for all neighbor pairs.

    Parameters:
    neighbor_pairs: List of (track_id1, track_id2) tuples
    filtered_time_series: Dictionary {track_id: time_series_array}
    max_order: Maximum AR order

    Returns:
    granger_results: Dictionary with results for each pair
    """

    granger_results = {}

    print(f"analyzing {len(neighbor_pairs)} neighbor pairs...")

    for i, (track1, track2) in enumerate(neighbor_pairs):
        if track1 not in filtered_time_series or track2 not in filtered_time_series:
            continue

        # Extract fluorescence values (column 1)
        ts1 = filtered_time_series[track1][:, 1]
        ts2 = filtered_time_series[track2][:, 1]

        # Compute Granger causality
        gc_12, gc_21, granger_diff, direction = compute_granger_difference(ts1, ts2, max_order)

        if granger_diff is not None:
            granger_results[(track1, track2)] = {
                'gc_12': gc_12,
                'gc_21': gc_21,
                'granger_diff': granger_diff,
                'direction': direction,
                'stronger_direction': track1 if direction == 1 else track2
            }

    print(f"successfully analyzed {len(granger_results)}")
    return granger_results


import numpy as np
from scipy.fft import fft, ifft
import networkx as nx


def iaaft_surrogate(time_series, max_iter=100, tolerance=1e-6):
    """IAAFT: preserves power spectrum + amplitude distribution, destroys correlations"""
    original = np.array(time_series)
    n = len(original)

    # Sort original amplitudes
    sorted_amplitudes = np.sort(original)

    # Initial random phase
    phases = np.random.uniform(0, 2 * np.pi, n)
    fft_original = fft(original)
    amplitudes = np.abs(fft_original)

    surrogate = original.copy()

    for _ in range(max_iter):
        # Step 1: match power spectrum
        fft_surrogate = amplitudes * np.exp(1j * phases)
        surrogate = np.real(ifft(fft_surrogate))

        # Step 2: match amplitude distribution
        sorted_indices = np.argsort(surrogate)
        surrogate[sorted_indices] = sorted_amplitudes

        # Update phases
        phases = np.angle(fft(surrogate))

    return surrogate


def statistical_testing(granger_results, filtered_time_series, n_surrogates=5000):
    """Generate surrogates and compute p-values"""
    significant_pairs = {}

    print(f"testing {len(granger_results)} pairs with {n_surrogates} surrogates...")

    for i, (pair, result) in enumerate(granger_results.items()):
        track1, track2 = pair
        original_diff = result['granger_diff']
        direction = result['direction']

        # Get cause and effect time series
        ts1 = filtered_time_series[track1][:, 1]
        ts2 = filtered_time_series[track2][:, 1]
        cause_ts = ts1 if direction == 1 else ts2
        effect_ts = ts2 if direction == 1 else ts1

        # Generate surrogates and test
        surrogate_diffs = []
        for _ in range(n_surrogates):
            cause_surrogate = iaaft_surrogate(cause_ts)

            if direction == 1:
                _, _, surr_diff, _ = compute_granger_difference(cause_surrogate, effect_ts)
            else:
                _, _, surr_diff, _ = compute_granger_difference(effect_ts, cause_surrogate)

            if surr_diff is not None:
                surrogate_diffs.append(surr_diff)

        # Calculate p-value
        if len(surrogate_diffs) > 0:
            p_value = np.mean(np.array(surrogate_diffs) > original_diff)

            if p_value < 0.05:
                significant_pairs[pair] = {
                    **result,
                    'p_value': p_value,
                    'n_surrogates': len(surrogate_diffs)
                }

        if (i + 1) % 10 == 0:
            print(f"tested {i + 1}/{len(granger_results)} pairs")

    print(f"found {len(significant_pairs)} significant pairs (p < 0.05)")
    return significant_pairs


def build_causality_network(significant_pairs, track_positions):
    """Build directed network from significant causality pairs"""
    G = nx.DiGraph()

    # Add nodes with positions
    for track_id, pos in track_positions.items():
        G.add_node(track_id, pos=pos)

    # Add edges with causality info
    for (track1, track2), result in significant_pairs.items():
        if result['direction'] == 1:
            G.add_edge(track1, track2, weight=result['granger_diff'], p_value=result['p_value'])
        else:
            G.add_edge(track2, track1, weight=result['granger_diff'], p_value=result['p_value'])

    return G


def compute_network_scores(G):
    """Calculate leader/follower and hub/authority scores"""
    # Leader/follower scores (out-degree vs in-degree)
    leader_scores = dict(G.out_degree())
    follower_scores = dict(G.in_degree())

    # Normalize
    max_leader = max(leader_scores.values()) if leader_scores.values() else 1
    max_follower = max(follower_scores.values()) if follower_scores.values() else 1

    leader_scores = {k: v / max_leader for k, v in leader_scores.items()}
    follower_scores = {k: v / max_follower for k, v in follower_scores.items()}

    # Hub/authority scores
    try:
        hub_scores, authority_scores = nx.hits(G, max_iter=1000)
    except:
        # Fallback if HITS fails
        hub_scores = {node: 0 for node in G.nodes()}
        authority_scores = {node: 0 for node in G.nodes()}

    return {
        'leader': leader_scores,
        'follower': follower_scores,
        'hub': hub_scores,
        'authority': authority_scores
    }


def visualize_network(G, network_scores, track_positions, save_path=None):
    """Create network visualization like Figure 1C & 1D"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Extract positions
    pos = {node: track_positions[node] for node in G.nodes()}

    # Plot 1: Leader/Follower
    leader_colors = [network_scores['leader'].get(node, 0) for node in G.nodes()]
    follower_colors = [network_scores['follower'].get(node, 0) for node in G.nodes()]

    ax1.scatter([pos[n][1] for n in G.nodes()], [pos[n][0] for n in G.nodes()],
                c=leader_colors, cmap='Reds', s=50, alpha=0.8)
    ax1.scatter([pos[n][1] for n in G.nodes()], [pos[n][0] for n in G.nodes()],
                c=follower_colors, cmap='Blues', s=30, alpha=0.6)

    # Draw edges
    for edge in G.edges():
        x_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        y_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        ax1.arrow(x_coords[0], y_coords[0],
                  x_coords[1] - x_coords[0], y_coords[1] - y_coords[0],
                  head_width=5, head_length=5, fc='gray', alpha=0.5)

    ax1.set_title('leader/follower scores')
    ax1.set_aspect('equal')

    # Plot 2: Hub/Authority
    hub_colors = [network_scores['hub'].get(node, 0) for node in G.nodes()]
    authority_colors = [network_scores['authority'].get(node, 0) for node in G.nodes()]

    ax2.scatter([pos[n][1] for n in G.nodes()], [pos[n][0] for n in G.nodes()],
                c=hub_colors, cmap='Reds', s=50, alpha=0.8)
    ax2.scatter([pos[n][1] for n in G.nodes()], [pos[n][0] for n in G.nodes()],
                c=authority_colors, cmap='Blues', s=30, alpha=0.6)

    for edge in G.edges():
        x_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        y_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        ax2.arrow(x_coords[0], y_coords[0],
                  x_coords[1] - x_coords[0], y_coords[1] - y_coords[0],
                  head_width=5, head_length=5, fc='gray', alpha=0.5)

    ax2.set_title('hub/authority scores')
    ax2.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# Full pipeline paper https://www.pnas.org/doi/epub/10.1073/pnas.2202204119

def run_granger_network_analysis(neighbor_pairs, filtered_time_series, track_positions):
    """Complete Granger causality network analysis"""

    # Step 1: Compute Granger causality
    granger_results = analyze_neighbor_pairs(neighbor_pairs, filtered_time_series)

    # Step 2: Statistical testing with IAAFT
    significant_pairs = statistical_testing(granger_results, filtered_time_series)

    # Step 3: Build network
    G = build_causality_network(significant_pairs, track_positions)

    # Step 4: Compute scores
    network_scores = compute_network_scores(G)

    # Step 5: Visualize
    visualize_network(G, network_scores, track_positions)

    return G, network_scores, significant_pairs

