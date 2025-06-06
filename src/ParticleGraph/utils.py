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
    if 'CElegans' in config_file_:
        config_file = os.path.join('CElegans', config_file_)
        pre_folder = 'CElegans/'
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
    """
    Maps the Varshney matrix to the given neuron list and sets rows/columns to zero for missing neurons.

    Parameters:
        neuron_list (list): List of all neuron names.
        neuron_names (list): List of neuron names in the Varshney dataset.
        matrix (torch.Tensor): Adjacency matrix from the Varshney dataset.

    Returns:
        torch.Tensor: Mapped matrix.
    """

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


