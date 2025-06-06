import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

def denoise_data(config, x_list, y_list, denoise_std, device):

    activity = torch.tensor(x_list[:, :, 6:7], dtype=torch.float32, device=device)
    activity = activity.squeeze()
    activity = activity.t()

    denoised_signals = []
    for i in trange(activity.shape[0]):
        signal = activity[i, :].clone().detach()
        match config.training.denoiser_type:
            case 'Gaussian_filter':
                denoised_signal = gaussian_filter(signal, torch.tensor([config.training.denoiser_param], dtype=torch.float32, device=device))

        denoised_signal = denoised_signal[0:signal.shape[0]]
        denoised_signals.append(to_numpy(denoised_signal))
        # visualize_signals(signal, denoised_signal)

    # Convert the list of denoised signals to a numpy array
    denoised_signals_array = np.stack(denoised_signals, axis=1)
    # Update the x_list array with the denoised values
    x_list[:, :, 6] = denoised_signals_array

    for k in range(1, x_list.shape[0] - 1):
        y_list[k] = (x_list[k + 1, :, 6:7] - x_list[k, :, 6:7]).squeeze() / config.simulation.delta_t

    return x_list, y_list



# def denoise_signal(config, signal, noise_std, device):
#     # noise_std = estimate_noise_std(signal)
#     denoised_signal = gaussian_filter(signal, torch.tensor([noise_std], dtype=torch.float32, device=device))
#     return denoised_signal


def estimate_noise_std(signal):
    # Estimate the noise standard deviation using the median absolute deviation
    valid_signal = signal[~torch.isnan(signal)]  # Filter out NaNs

    if valid_signal.numel() > 0:
        median = torch.median(valid_signal)
        mad = torch.median(torch.abs(valid_signal - median))
        noise_std = 1.4826 * mad
    else:
        noise_std = 1
    return noise_std


def gaussian_filter(signal, std):
    # Create a Gaussian kernel
    kernel_size = int(4 * std + 1)
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=signal.device)
    kernel = torch.exp(-0.5 * (x / std) ** 2)
    kernel = kernel / kernel.sum()

    # Apply the Gaussian filter
    signal_padded = F.pad(signal.unsqueeze(0).unsqueeze(0), (kernel_size // 2, kernel_size // 2), mode='reflect')
    signal_denoised = F.conv1d(signal_padded, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    return signal_denoised


def visualize_signals(original_signal, denoised_signal):
    plt.figure(figsize=(10, 6))
    plt.plot(to_numpy(original_signal), label='Original Signal', linewidth=2)
    plt.plot(to_numpy(denoised_signal), label='Denoised Signal', linewidth=2)
    plt.legend()
    plt.xlim([0,1000])
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Signal Denoising')
    plt.show()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

if __name__ == '__main__':

    x_list = np.load('/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/signal/signal_N2_a43_2/x_list_0.npy')

    activity = torch.tensor(x_list[:, :, 6:7])
    activity = activity.squeeze()
    activity = activity.t()

    plt.figure(figsize=(15, 10))
    n = np.random.permutation(x_list.shape[1])
    for i in range(25):
        plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
    plt.xlabel('time', fontsize=64)
    plt.ylabel('$x_{i}$', fontsize=64)
    # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.xlim([0, 1000])
    plt.tight_layout()
    plt.show()
    plt.savefig(f'tmp/activity_1000_raw.png', dpi=300)
    plt.close()

    # Denoise each time series signal
    denoised_signals = []
    for i in trange(activity.shape[0]):
        signal = activity[i, :].clone().detach()
        denoised_signal = denoise_signal(signal, None, signal.device)
        denoised_signal = denoised_signal[0:signal.shape[0]]
        denoised_signals.append(to_numpy(denoised_signal))
        # visualize_signals(signal, denoised_signal)

    # Convert the list of denoised signals to a numpy array
    denoised_signals_array = np.stack(denoised_signals, axis=1)

    # Update the x_list array with the denoised values
    x_list[:, :, 6] = denoised_signals_array


    activity = torch.tensor(x_list[:, :, 6:7])
    activity = activity.squeeze()
    activity = activity.t()

    plt.figure(figsize=(15, 10))
    for i in range(25):
        plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
    plt.xlabel('time', fontsize=64)
    plt.ylabel('$x_{i}$', fontsize=64)
    # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.xlim([0, 1000])
    plt.tight_layout()
    plt.show()







