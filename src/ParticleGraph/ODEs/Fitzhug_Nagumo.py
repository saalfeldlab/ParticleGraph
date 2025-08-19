import os

# Configure matplotlib to use headless backend before importing pyplot
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import torch
import torch.nn as nn

# from ParticleGraph.generators.utils import get_time_series
from matplotlib import pyplot as plt
from tqdm import trange
from ParticleGraph.utils import *
from ParticleGraph.config import ParticleGraphConfig
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torch.optim as optim
import time as Time

from config_Fitzhug_Nagumo import FitzhughNagumoConfig
from utils import setup_experiment_folders

# ----------------- SIREN Layer -----------------
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                    np.sqrt(6 / self.linear.in_features) / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output


class MLP(nn.Module):

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None,
                 initialisation=None):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)

        if initialisation == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

        self.layers.append(layer)

        if activation == 'tanh':
            self.activation = F.tanh
        else:
            self.activation = F.relu

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class model_duo(nn.Module):

    def __init__(self, device=None):
        super(model_duo, self).__init__()

        self.siren = Siren(in_features=1, out_features=1, hidden_features=128, hidden_layers=3,
                           outermost_linear=True).to(device)
        self.mlp0 = MLP(input_size=3, output_size=1, nlayers=5, hidden_size=128, device=device)
        self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

        # # Count parameters for each component
        # siren_params = sum(p.numel() for p in self.siren.parameters())
        # mlp0_params = sum(p.numel() for p in self.mlp0.parameters())
        # mlp1_params = sum(p.numel() for p in self.mlp1.parameters())
        # total_params = siren_params + mlp0_params + mlp1_params
        #
        # print(f"Model Parameter Count:")
        # print(f"  SIREN:     {siren_params:,} parameters")
        # print(f"  MLP0:      {mlp0_params:,} parameters")
        # print(f"  MLP1:      {mlp1_params:,} parameters")
        # print(f"  total:     {total_params:,} parameters")
        # print(f"  model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")



    def forward(self, x):
        return self.siren(x)


def initialize_siren_with_v(model, v_true, t_full, device, n_init_steps=1000):
    """
    Initialize SIREN network to fit v values

    Args:
        model: The model_duo instance
        v_true: Ground truth v values (tensor)
        t_full: Time points (tensor)
        device: Device for computation
        n_init_steps: Number of initialization steps
    """

    print("Initializing SIREN with v values...")

    # Create separate optimizer for SIREN initialization
    siren_optimizer = torch.optim.Adam(model.siren.parameters(), lr=1e-4)

    # Pre-train SIREN to fit v data
    for step in range(n_init_steps):
        # Sample random time points
        batch_size = len(v_true) // 4
        idx = torch.randint(0, len(v_true), (batch_size,))

        t_batch = t_full[idx]
        v_target = v_true[idx, None]

        # Forward pass
        v_pred = model.siren(t_batch)

        # Loss
        loss = F.mse_loss(v_pred, v_target)

        # Backward pass
        siren_optimizer.zero_grad()
        loss.backward()
        siren_optimizer.step()

        if step == 0:
            print(f"SIREN init step {step}, loss: {loss.item():.6f}")

    # Test initialization quality
    with torch.no_grad():
        v_pred_full = model.siren(t_full)
        init_mse = F.mse_loss(v_pred_full.squeeze(), v_true).item()
        print(f"SIREN initialization MSE: {init_mse:.6f}")

    return model


def plot_siren_v_init(model, v_true, t_full, save_path):
    """Plot SIREN vs v after initialization"""
    with torch.no_grad():
        v_siren = model.siren(t_full).squeeze()
        mse = F.mse_loss(v_siren, v_true).item()

    plt.figure(figsize=(10, 5))
    plt.plot(v_true.cpu(), 'white', linewidth=2, label='True v', alpha=0.8)
    plt.plot(v_siren.cpu(), 'cyan', linewidth=2, label='SIREN v', alpha=0.9)
    plt.title(f'SIREN vs V after initialization (MSE: {mse:.6f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    device = set_device('auto')
    print(f'device  {device}')

    config_root = "./config"
    config_file = "default"

    # config_file_list = ['noise_1', 'noise_2', 'noise_3', 'noise_4', 'noise_5']
    # config_file_list = ['lambda_2', 'lambda_3', 'lambda_4']
    config_file_list = ['recur_7_3', 'recur_7_4', 'recur_7_5']

    for config_file in config_file_list:


        config = FitzhughNagumoConfig.from_yaml(f"{config_root}/{config_file}.yaml")

        device = config.training.device
        a = config.system.a
        b = config.system.b
        epsilon = config.system.epsilon
        T = config.simulation.T
        dt = config.simulation.dt
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps)

        v_init = config.simulation.v_init
        w_init = config.simulation.w_init
        pulse_interval = config.simulation.pulse_interval
        pulse_duration = config.simulation.pulse_duration
        pulse_amplitude = config.simulation.pulse_amplitude
        noise_level = config.simulation.noise_level
        n_iter = config.training.n_iter
        batch_ratio = config.training.batch_ratio
        test_runs = config.training.test_runs
        learning_rate = config.training.learning_rate
        l1_lambda = config.training.l1_lambda
        weight_decay = config.training.weight_decay
        recursive_loop = config.training.recursive_loop
        lambda_jac = config.training.lambda_jac
        lambda_ratio = config.training.lambda_ratio
        lambda_amp = config.training.lambda_amp
        tau_vv = config.training.tau_vv
        tau_vw = config.training.tau_vw
        tau_wv = config.training.tau_wv
        tau_ww = config.training.tau_ww

        print(f"loaded config: {config.description}")
        print(f"system parameters: a={a}, b={b}, epsilon={epsilon}")
        print(f"simulation: T={T}, dt={dt}, n_steps={n_steps}")
        print(f"training: n_iter={n_iter}, lr={learning_rate}, test_runs={test_runs}")
        print(f"jacobian regularization: lambda_jac={lambda_jac}, lambda_ratio={lambda_ratio}")
        print(f"tau thresholds: vv={tau_vv}, vw={tau_vw}, wv={tau_wv}, ww={tau_ww}")

        # ===============================================================
        # SETUP EXPERIMENT FOLDERS AND METADATA
        # ===============================================================
        print("setting up experiment folders...")

        # Setup organized folder structure
        experiment_name = f"fitzhugh_nagumo_{config_file}"
        folders = setup_experiment_folders(experiment_name=experiment_name)

        # remove files from folders['training_plots']
        for file in os.listdir(folders['training_plots']):
            file_path = os.path.join(folders['training_plots'], file)
            if os.path.isfile(file_path):
                os.remove(file_path)


        # ===============================================================
        # GENERATE TRAINING DATA
        # ===============================================================
        print("generating fitzhugh-nagumo data...")

        # Initialize variables
        v = np.zeros(n_steps)
        w = np.zeros(n_steps)
        I_ext = np.zeros(n_steps)

        # Initial conditions
        v[0] = v_init
        w[0] = w_init

        # Pulse indexing
        n_pulses = int(np.ceil(T / pulse_interval))
        pulse_ids = (time // pulse_interval).astype(int)

        # phase_rnd = np.random.uniform(0, 2 * np.pi, n_pulses)
        # amplitude_rnd = pulse_amplitude * (1 + 0.25 * np.random.rand(n_pulses))

        phase_rnd = np.zeros(n_pulses)  # all pulses start at phase 0
        amplitude_rnd = np.full(n_pulses, pulse_amplitude)
        mask = (time % pulse_interval) < pulse_duration

        # Positive-centered sine pulses: baseline 1, oscillates above/below baseline slightly
        wave = 1.0 + 0.25 * np.sin(2 * np.pi * (time[mask] % pulse_interval) / pulse_interval + phase_rnd[pulse_ids[mask]])
        I_ext[mask] = amplitude_rnd[pulse_ids[mask]] * wave

        # for i, t in enumerate(time):
        #     if (t % pulse_interval) < pulse_duration:
        #         I_ext[i] = pulse_amplitude * (1 + 0.25 * np.sin(2 * np.pi * t / pulse_interval))

        # Rollout using Euler method
        for i in range(n_steps - 1):
            dv = v[i] - (v[i] ** 3) / 3 - w[i] + I_ext[i]
            dw = epsilon * (v[i] + a - b * w[i])
            v[i + 1] = v[i] + dt * dv + noise_level * np.random.randn()
            w[i + 1] = w[i] + dt * dw + noise_level * np.random.randn()

        plt.style.use('dark_background')

        v_true = torch.tensor(v, dtype=torch.float32, device=device)
        w_true = torch.tensor(w, dtype=torch.float32, device=device)
        I_ext = torch.tensor(I_ext, dtype=torch.float32, device=device)

        n_steps = len(w_true)
        t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)  # shape (10000, 1)
        w_true = torch.tensor(w_true, dtype=torch.float32, device=device)  # shape (10000,)

        batch_size = n_steps // 5

        # ===============================================================
        # TRAINING LOOP WITH DATA COLLECTION
        # ===============================================================
        print("starting training runs...")

        # Initialize data collection
        convergence_results = []
        loss_progression_data = {}

        log_every = 50  # adjust as needed
        log_data = {run + 1: {
            'iterations': [],
            'losses': [],
            'spectral_radius': [],
            'grad_dv_w_norm': [],
            'grad_dv_v_norm': [],
            'grad_dw_v_norm': [],
            'grad_dw_w_norm': [],
            'mlp0_weight_norm': [],
            'mlp1_weight_norm': [],
            'mlp0_grad_norm': [],
            'mlp1_grad_norm': [],
        } for run in range(test_runs)}

        for run in range(test_runs):
            # print(f"\ntraining run {run + 1}/{test_runs}")

            # Initialize loss tracking for this run
            loss_progression_data[run + 1] = {
                'iterations': [],
                'losses': []
            }

            model = model_duo(device=device)  # Siren(in_features=1, out_features=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Initialize SIREN with v values (optional)
            # model = initialize_siren_with_v(model, v_true, t_full, device, n_init_steps=1000)
            # siren_init_path = os.path.join(folders['plots'], f'siren_v_init_comparison_run_{run+1}.png')
            # plot_siren_v_init(model, v_true, t_full, siren_init_path)

            Time.sleep(1)
            loss_list = []
            grad_list = []

            lambda_jac_max = 1.0  # max weight for Jacobian penalty
            warmup_iters = int(0.15 * n_iter)  # 15% of total iterations for warm-up

            for iter in range(n_iter):
                idx = torch.randint(1, n_steps - 8, (batch_size,))
                idx = torch.unique(idx)
                t_batch = t_full[idx]
                idx = to_numpy(idx)

                # Initial states
                w = model.siren(t_batch)
                v0 = v_true[idx, None].clone().requires_grad_(True)
                w0 = model.siren(t_batch).clone().requires_grad_(True)
                v = v0
                optimizer.zero_grad()

                recursive_loop = 3
                accumulated_loss = 0.0

                # --- Annealing Jacobian regularizer ---
                if iter < warmup_iters:
                    lambda_jac = lambda_jac_max * (1 - iter / warmup_iters)
                else:
                    lambda_jac = 0.0

                for loop in range(recursive_loop):
                    dv_pred = model.mlp0(torch.cat((v, w, I_ext[idx, None].clone().detach()), dim=1))
                    dw_pred = model.mlp1(torch.cat((v, w), dim=1))

                    def jacobian_reg(output, wrt, tau, device):
                        g = torch.autograd.grad(
                            output.sum(), wrt,
                            create_graph=True, retain_graph=True, allow_unused=True
                        )[0]
                        g_norm = torch.tensor(0.0, device=device) if g is None else g.norm(p=2, dim=1).mean()
                        return torch.relu(tau - g_norm) ** 2

                    # Enforce Jacobian regularization for all four derivatives:
                    # ∂(dv/dt)/∂v → keep dv sensitive to v (self-excitation)
                    # ∂(dv/dt)/∂w → keep dv sensitive to w (recovery inhibition)
                    # ∂(dw/dt)/∂v → keep dw sensitive to v (voltage drives recovery)
                    # ∂(dw/dt)/∂w → optionally enforce weak sensitivity or suppression (w leak term)
                    if lambda_jac > 0:
                        R_jac_vv = jacobian_reg(dv_pred, v0, tau=tau_vv, device=v.device)  # ∂dv/∂v
                        R_jac_vw = jacobian_reg(dv_pred, w0, tau=tau_vw, device=v.device)  # ∂dv/∂w
                        R_jac_wv = jacobian_reg(dw_pred, v0, tau=tau_wv, device=v.device)  # ∂dw/∂v
                        R_jac_ww = jacobian_reg(dw_pred, w0, tau=tau_ww, device=v.device)  # ∂dw/∂w
                    else:
                        R_jac_vv = R_jac_vw = R_jac_wv = R_jac_ww = torch.tensor(0.0, device=v.device)

                    # Euler update
                    v = v + dt * dv_pred
                    w = w + dt * dw_pred

                    # Loss against ground truth
                    step_idx = idx + loop + 1
                    v_target = v_true[step_idx, None]
                    w_target_siren = model.siren(t_full[step_idx])

                    v_step_loss = (v - v_target).norm(2)
                    w_step_loss = (w - w_target_siren).norm(2)

                    step_weight = (loop + 1) / recursive_loop
                    step_loss = step_weight * (v_step_loss + w_step_loss)

                    # add both Jacobian penalties
                    accumulated_loss += step_loss + lambda_jac * (R_jac_vv + R_jac_vw + R_jac_wv + R_jac_ww)

                # Regularization
                l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.mlp1.parameters())
                l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())

                loss = accumulated_loss + l1_lambda * l1_penalty + weight_decay * l2_penalty
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

                if iter % log_every == 0:
                    log_data[run + 1]['iterations'].append(iter + 1)
                    log_data[run + 1]['losses'].append(loss.item())

                    # Enable grad for input states
                    w0 = model.siren(t_batch).detach().clone().requires_grad_(True)
                    v0 = v_true[idx, None].detach().clone().requires_grad_(True)

                    dv_pred = model.mlp0(torch.cat((v0, w0, I_ext[idx, None].detach()), dim=1))
                    dw_pred = model.mlp1(torch.cat((v0, w0), dim=1))

                    # Jacobian gradients
                    grad_dv_w = \
                    torch.autograd.grad(dv_pred.sum(), w0, create_graph=False, retain_graph=True, allow_unused=True)[0]
                    grad_dv_v = \
                    torch.autograd.grad(dv_pred.sum(), v0, create_graph=False, retain_graph=True, allow_unused=True)[0]
                    grad_dw_v = \
                    torch.autograd.grad(dw_pred.sum(), v0, create_graph=False, retain_graph=True, allow_unused=True)[0]
                    grad_dw_w = \
                    torch.autograd.grad(dw_pred.sum(), w0, create_graph=False, retain_graph=True, allow_unused=True)[0]

                    # Convert to scalar norms
                    log_data[run + 1]['grad_dv_w_norm'].append(
                        0.0 if grad_dv_w is None else grad_dv_w.norm(p=2, dim=1).mean().item())
                    log_data[run + 1]['grad_dv_v_norm'].append(
                        0.0 if grad_dv_v is None else grad_dv_v.norm(p=2, dim=1).mean().item())
                    log_data[run + 1]['grad_dw_v_norm'].append(
                        0.0 if grad_dw_v is None else grad_dw_v.norm(p=2, dim=1).mean().item())
                    log_data[run + 1]['grad_dw_w_norm'].append(
                        0.0 if grad_dw_w is None else grad_dw_w.norm(p=2, dim=1).mean().item())

                    # Jacobian spectral radius
                    J_v = torch.autograd.grad(dv_pred.sum(), v0, retain_graph=True, create_graph=False)[0]
                    J_w = torch.autograd.grad(dv_pred.sum(), w0, retain_graph=True, create_graph=False)[0]
                    K_v = torch.autograd.grad(dw_pred.sum(), v0, retain_graph=True, create_graph=False)[0]
                    K_w = torch.autograd.grad(dw_pred.sum(), w0, retain_graph=True, create_graph=False)[0]
                    if all(x is not None for x in [J_v, J_w, K_v, K_w]):
                        Jmat = torch.stack([
                            torch.stack([J_v.mean(), J_w.mean()]),
                            torch.stack([K_v.mean(), K_w.mean()])
                        ])
                        eigvals = torch.linalg.eigvals(Jmat)
                        spectral_radius = eigvals.abs().max().item()
                    else:
                        spectral_radius = 0.0  # Default value when gradients fail

                    log_data[run + 1]['spectral_radius'].append(spectral_radius)

                    # MLP weight and gradient norms
                    mlp0_weights = torch.cat([p.flatten() for p in model.mlp0.parameters()])
                    mlp1_weights = torch.cat([p.flatten() for p in model.mlp1.parameters()])
                    mlp0_grads = torch.cat([p.grad.flatten() for p in model.mlp0.parameters() if p.grad is not None])
                    mlp1_grads = torch.cat([p.grad.flatten() for p in model.mlp1.parameters() if p.grad is not None])

                    log_data[run + 1]['mlp0_weight_norm'].append(mlp0_weights.norm().item())
                    log_data[run + 1]['mlp1_weight_norm'].append(mlp1_weights.norm().item())
                    log_data[run + 1]['mlp0_grad_norm'].append(
                        mlp0_grads.norm().item() if mlp0_grads.numel() > 0 else 0.0)
                    log_data[run + 1]['mlp1_grad_norm'].append(
                        mlp1_grads.norm().item() if mlp1_grads.numel() > 0 else 0.0)

                    # grad_list.append((grad_dv_w_norm, grad_dv_v_norm, grad_dw_v_norm, grad_dw_w_norm, spectral_radius))

            # print(f"iteration {iter + 1}/{n_iter}, loss: {loss.item():.6f}, spectral radius ≈ {spectral_radius:.3f}")

            Time.sleep(1)
            # ===============================================================
            # ROLLOUT WITH SIREN BOOTSTRAP THEN SELF-CONSISTENT w
            # ===============================================================
            with torch.no_grad():
                t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)

                # Initialize states
                v = v_true[0:1].clone().detach().squeeze()
                w = w_true[0:1].clone().detach().squeeze()

                v_list = [v.clone()]
                w_list = [w.clone()]

                n_siren_bootstrap = 500  # number of steps to use SIREN for w

                for step in range(1, n_steps):
                    # --- Use SIREN for w only in the initial steps ---
                    if step < n_siren_bootstrap:
                        w = model.siren(t_full[step]).squeeze()

                    # Prepare inputs
                    v_input = v.unsqueeze(0).unsqueeze(1)  # shape [1,1]
                    w_input = w.unsqueeze(0).unsqueeze(1)  # shape [1,1]
                    I_input = I_ext[step].unsqueeze(0).unsqueeze(1)  # shape [1,1]

                    # MLP predictions
                    dv_pred = model.mlp0(torch.cat((v_input, w_input, I_input), dim=1))
                    dw_pred = model.mlp1(torch.cat((v_input, w_input), dim=1))

                    # Update states
                    v = v + dt * dv_pred.squeeze()
                    if step >= n_siren_bootstrap:
                        w = w + dt * dw_pred.squeeze()  # after bootstrap, w evolves via MLP

                    v_list.append(v.clone())
                    w_list.append(w.clone())

                v_list = torch.stack(v_list, dim=0)  # shape [n_steps, 1]
                w_list = torch.stack(w_list, dim=0)  # shape [n_steps, 1]

                v_mse = F.mse_loss(v_list[500:].squeeze(), v_true[500:]).item()
                w_mse = F.mse_loss(w_list[500:].squeeze(), w_true[500:]).item()
                total_mse = v_mse + w_mse

                # --- Compute affine-corrected W ---
                w_pred = w_list.cpu().numpy().squeeze()
                w_target = w_true.cpu().numpy().squeeze()
                if np.var(w_pred) < 1e-12:  # avoid divide by zero
                    a, b = 1.0, 0.0
                else:
                    a = np.cov(w_pred, w_target, bias=True)[0, 1] / np.var(w_pred)
                    b = np.mean(w_target) - a * np.mean(w_pred)

                w_corr = a * w_pred + b

                # MSE after affine correction
                w_mse = np.mean((w_corr[500:] - w_target[500:]) ** 2)
                total_mse = np.mean(
                    (v_list[500:].cpu().numpy().squeeze() - v_true[500:].cpu().numpy().squeeze()) ** 2) + w_mse

                convergence_results.append({
                    'run': run + 1,
                    'iteration': iter,
                    'loss': loss.item(),
                    'v_mse': v_mse,
                    'w_mse': w_mse,
                    'total_mse': total_mse
                })

                # print(f"rollout with bootstrap SIREN:  V MSE: {v_mse:.6f}, W corr MSE: {w_mse:.6f}, total MSE: {total_mse:.6f}")

                fig = plt.figure(figsize=(16, 8))
                plt.subplot(2, 1, 1)
                plt.plot(v_true.cpu().numpy(), label='true v', linewidth=3, alpha=0.7, c='white')
                plt.plot(v_list.cpu().numpy(), label='rollout v', linewidth=2, alpha=1, c='green')
                plt.plot(I_ext.cpu().numpy(), label='I_ext', linewidth=2, alpha=0.5, c='red')
                plt.xlim([0, n_steps // 2.5])
                plt.xlabel('time')
                plt.ylabel('v')
                plt.ylim([-2.5,2.5])
                plt.legend(loc='upper left', fontsize=12)
                plt.title(f'v (MSE: {v_mse:.4f})')
                plt.grid(True, alpha=0.3)

                # --- Plot corrected w ---
                plt.subplot(2, 1, 2)
                plt.plot(w_true.cpu().numpy(), label='true w', linewidth=3, alpha=0.7, c='white')
                plt.plot(w_list.cpu().numpy(), label='rollout w', linewidth=2, alpha=0.7, c='cyan')
                plt.plot(w_corr, label='corrected w (affine)', linewidth=2, alpha=0.8, c='orange')
                plt.xlim([0, n_steps // 2.5])
                plt.xlabel('time')
                plt.ylabel('w')
                plt.ylim([-1,2.5])
                plt.legend(loc='upper left', fontsize=12)
                plt.title(f'w latent variable (MSE: {w_mse:.4f}, w_corr = {a:.3f} * w_pred + {b:.3f})')
                plt.grid(True, alpha=0.3)


                plt.tight_layout()
                plt.show()
                training_plot_path = os.path.join(folders['training_plots'], f'nagumo_training_run_{run + 1}_iter_{iter + 1}.png')
                plt.savefig(training_plot_path, dpi=170, facecolor='black')
                plt.close()

                model_path = os.path.join(folders['models'], f'model_run_{run + 1}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'run': run + 1,
                    'iteration': iter,
                    'loss': loss.item(),
                    'v_mse': v_mse,
                    'w_mse': w_mse,
                    'total_mse': total_mse
                }, model_path)

        for run, data in log_data.items():
            fig, axs = plt.subplots(3, 2, figsize=(14, 12))
            iters = data['iterations']

            # Ensure all data arrays have the same length
            min_length = min(len(values) for values in data.values() if isinstance(values, list))

            # Truncate all arrays to the same length
            plot_data = {}
            for key, values in data.items():
                if isinstance(values, list):
                    plot_data[key] = values[:min_length]

            plot_iters = plot_data['iterations']

            # Loss - USE PROPER X-AXIS
            axs[0, 0].plot(plot_iters, plot_data['losses'], label='Loss', color='tab:blue')
            axs[0, 0].set_ylabel('loss')
            axs[0, 0].set_ylim([0, 6])

            # Spectral radius - USE PROPER X-AXIS
            axs[0, 1].plot(plot_iters, plot_data['spectral_radius'], label='Spectral radius', color='tab:orange')
            axs[0, 1].set_ylabel('spectral radius')

            # Gradient norms - USE PROPER X-AXIS
            axs[1, 0].plot(plot_iters, plot_data['grad_dv_w_norm'], label='∂dv/∂w', color='tab:blue')
            axs[1, 0].plot(plot_iters, plot_data['grad_dv_v_norm'], label='∂dv/∂v', color='tab:orange')
            axs[1, 0].set_ylabel('gradient norms')
            axs[1, 0].set_ylim([0, 6])
            axs[1, 0].legend()

            # Gradient norms - USE PROPER X-AXIS
            axs[1, 1].plot(plot_iters, plot_data['grad_dw_v_norm'], label='∂dw/∂v', color='tab:green')
            axs[1, 1].plot(plot_iters, plot_data['grad_dw_w_norm'], label='∂dw/∂w', color='tab:red')
            axs[1, 1].set_ylabel('gradient norms')
            axs[1, 1].set_ylim([0, 0.1])
            axs[1, 1].legend()

            # Weight norms - USE PROPER X-AXIS
            axs[2, 0].plot(plot_iters, plot_data['mlp0_weight_norm'], label='MLP0 weight', color='tab:blue')
            axs[2, 0].plot(plot_iters, [x * 10 for x in plot_data['mlp1_weight_norm']], label='MLP1 weight x10',
                           color='tab:orange')
            axs[2, 0].set_ylabel('weight norms')
            axs[2, 0].set_ylim([0, 25])
            axs[2, 0].legend()

            # Gradient norms of weights - USE PROPER X-AXIS
            axs[2, 1].plot(plot_iters, plot_data['mlp0_grad_norm'], label='MLP0 grad', color='tab:blue')
            axs[2, 1].plot(plot_iters, [x * 10 for x in plot_data['mlp1_grad_norm']], label='MLP1 grad x10',
                           color='tab:orange')
            axs[2, 1].set_ylabel('weight gradient norms')
            axs[2, 1].set_ylim([0, 200])
            axs[2, 1].legend()

            for ax in axs.flat:
                ax.set_xlabel('Iteration')

            plt.suptitle(f'Training Run {run} Trajectory Analysis')
            training_plot_path = os.path.join(
                folders['training_plots'],
                f'nagumo_trajectory_analysis_{run}.png'
            )
            plt.tight_layout()
            plt.savefig(training_plot_path, dpi=200, bbox_inches='tight', facecolor='black')
            plt.close(fig)

        print(f"\nCONVERGENCE SUMMARY")
        print(f"{'run':<4} {'Loss':<12} {'V MSE':<12} {'W MSE':<12} {'Total MSE':<12}")
        print(f"{'-' * 60}")
        for result in convergence_results:
            print(f"{result['run']:<4} {result['loss']:<12.6f} {result['v_mse']:<12.6f} {result['w_mse']:<12.6f} {result['total_mse']:<12.6f}")

        print("")
        print("")
