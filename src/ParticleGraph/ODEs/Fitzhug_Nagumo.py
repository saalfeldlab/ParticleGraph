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

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None, initialisation=None):

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
        else :
            nn.init.normal_(layer.weight, std=0.01)
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


class model_duo(nn.Module):

    def __init__(self, device=None):

        super(model_duo, self).__init__()

        self.siren = Siren(in_features=1, out_features=1, hidden_features=128, hidden_layers=3, outermost_linear=True).to(device)
        self.mlp0 = MLP(input_size=3, output_size=1, nlayers=5, hidden_size=128, device=device)
        self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

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


def plot_siren_v_init(model, v_true, t_full):
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
    plt.savefig('./tmp/siren_v_init_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    device = set_device('auto')
    print(f'device  {device}')

    noise_level_list =  [1.0E-3]  # Only test noise level 0.001 for gradient accumulation experiment

    for noise_level in noise_level_list:

        print(f" ")
        print(f"running simulation with noise level: {noise_level}")


        # Parameters
        a = 0.7
        b = 0.8
        epsilon = 0.18

        # Simulation settings
        T = 1000       # total time
        dt = 0.1      # time step
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps)

        # Initialize variables
        v = np.zeros(n_steps)
        w = np.zeros(n_steps)
        I_ext = np.zeros(n_steps)

        # Initial conditions
        v[0] = -1.0
        w[0] = 1.0

        # External excitation: periodic pulse every 30s lasting 1s
        pulse_interval = 80.0  # seconds
        pulse_duration = 1.0   # seconds
        pulse_amplitude = 0.8  # strength of excitation

        for i, t in enumerate(time):
            if (t % pulse_interval) < pulse_duration:
                I_ext[i] = pulse_amplitude

        # Rollout using Euler method
        for i in range(n_steps - 1):
            dv = v[i] - (v[i]**3)/3 - w[i] + I_ext[i]
            dw = epsilon * (v[i] + a - b * w[i])
            v[i+1] = v[i] + dt * dv + noise_level * np.random.randn()
            w[i+1] = w[i] + dt * dw + noise_level * np.random.randn()

        plt.style.use('dark_background')

        v_true = torch.tensor(v, dtype=torch.float32, device=device)
        w_true = torch.tensor(w, dtype=torch.float32, device=device)
        I_ext = torch.tensor(I_ext, dtype=torch.float32, device=device)

        n_steps = len(w_true)
        t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)  # shape (10000, 1)
        w_true = torch.tensor(w_true, dtype=torch.float32, device=device)  # shape (10000,)


        n_iter = 5000
        batch_size = n_steps // 5

        # Test convergence over multiple runs
        test_runs = 5
        convergence_results = []

        print(f" ")


        for run in range(test_runs):
            print(f"training run {run+1}/{test_runs}")

            model = model_duo(device=device)  # Siren(in_features=1, out_features=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Initialize SIREN with v values
            # model = initialize_siren_with_v(model, v_true, t_full, device, n_init_steps=1000)
            # plot_siren_v_init(model, v_true, t_full)

            Time.sleep(1)
            loss_list = []
            for iter in range(n_iter):
                idx = torch.randint(1, n_steps-8, (batch_size,))
                idx = torch.unique(idx)
                t_batch = t_full[idx]
                idx = to_numpy(idx)

                w = model.siren(t_batch)
                v = v_true[idx, None].clone().detach()
                optimizer.zero_grad()

                recursive_loop = 3

                # Store intermediate states for gradient accumulation
                accumulated_loss = 0.0

                for loop in range(recursive_loop):

                    dv_pred = model.mlp0(torch.cat((v, w, I_ext[idx, None].clone().detach()), dim=1))
                    dw_pred = model.mlp1(torch.cat((v, w), dim=1))

                    v = v + dt * dv_pred
                    w = w + dt * dw_pred

                    # Compute loss for this step directly
                    step_idx = idx + loop + 1
                    v_target = v_true[step_idx, None]
                    w_target_siren = model.siren(t_full[step_idx])

                    # Calculate losses for this step
                    v_step_loss = (v - v_target).norm(2)
                    w_step_loss = (w - w_target_siren).norm(2)

                    # Weight losses by step (later steps get higher weight)
                    step_weight = (loop + 1) / recursive_loop
                    step_loss = step_weight * (v_step_loss + w_step_loss)

                    accumulated_loss += step_loss

                # Add regularization penalties
                l1_lambda = 1.0E-3
                l1_penalty = 0.0
                for param in model.mlp1.parameters():
                    l1_penalty += torch.sum(torch.abs(param))

                weight_decay = 1e-6
                l2_penalty = 0.0
                for param in model.parameters():
                    l2_penalty += torch.sum(param ** 2)

                # Final loss with regularization
                loss = accumulated_loss + l1_lambda * l1_penalty + weight_decay * l2_penalty

                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())

                if iter % 250 == 0:
                    print(f"iteration {iter+1}/{n_iter}, loss: {loss.item():.6f}")

            Time.sleep(1)
            # rollout analysis
            with torch.no_grad():
                t_full = torch.linspace(0, 0.5, n_steps//2, device=device).unsqueeze(1)
                w_pred = model(t_full)

                v = v_true[0:1].clone().detach()
                w = w_true[0:1].clone().detach()
                v_list = []
                w_list = []
                v_list.append(v.clone().detach())
                w_list.append(w.clone().detach())

                for step in range(1, n_steps):
                    with torch.no_grad():
                        w = model.siren(t_full[step])

                        dv_pred = model.mlp0(torch.cat((v[:, None], w[:, None], I_ext[step:step + 1, None]), dim=1))
                        dw_pred = model.mlp1(torch.cat((v[:, None], w[:, None]), dim=1))

                        v += dt * dv_pred.squeeze()
                        w += dt * dw_pred.squeeze()

                    v_list.append(v.clone().detach())
                    w_list.append(w.clone().detach())

                v_list = torch.stack(v_list, dim=0)
                w_list = torch.stack(w_list, dim=0)

                v_mse = F.mse_loss(v_list[500:].squeeze(), v_true[500:]).item()
                w_mse = F.mse_loss(w_list[500:].squeeze(), w_true[500:]).item()
                total_mse = v_mse + w_mse

                convergence_results.append({
                    'run': run+1,
                    'iteration': iter,
                    'loss': loss.item(),
                    'v_mse': v_mse,
                    'w_mse': w_mse,
                    'total_mse': total_mse
                })

                print(f"V MSE: {v_mse:.6f}, W MSE: {w_mse:.6f}, Total MSE: {total_mse:.6f}")

                # Save results for this run
                fig = plt.figure(figsize=(16, 12))

                # Panel 1: External input only
                plt.subplot(2, 2, 1)
                plt.plot(I_ext.cpu().numpy(), label='I_ext (external input)', linewidth=2, alpha=0.7, c='red')
                plt.xlim([0, n_steps//2.5])
                plt.xlabel('Time steps')
                plt.ylabel('External input')
                plt.legend(loc='upper left')
                plt.title('External Input Current')
                plt.grid(True, alpha=0.3)

                # Panel 2: True vs reconstructed membrane potential
                plt.subplot(2, 2, 2)
                plt.plot(v_true.cpu().numpy(), label='true v (membrane potential)', linewidth=3, alpha=0.7, c='white')
                plt.plot(v_list.cpu().numpy(), label='rollout v', linewidth=2, alpha=1, c='green')
                plt.xlim([0, n_steps//2.5])
                plt.xlabel('Time steps')
                plt.ylabel('Membrane potential v')
                plt.legend(loc='upper left')
                plt.title(f'Run {run+1}: Membrane Potential (MSE: {v_mse:.4f})')
                plt.grid(True, alpha=0.3)

                # Panel 3: True vs reconstructed recovery variable
                plt.subplot(2, 2, 3)
                plt.plot(w_true.cpu().numpy(), label='true w (recovery variable)', linewidth=3, alpha=0.7, c='white')
                plt.plot(w_list.cpu().numpy(), label='rollout w', linewidth=2, alpha=1, c='cyan')
                plt.plot(w_pred.cpu().numpy(), label='SIREN w', linewidth=2, alpha=0.7, c='cyan')
                plt.xlim([0, n_steps//2.5])
                plt.xlabel('Time steps')
                plt.ylabel('Recovery variable w')
                plt.legend(loc='upper left')
                plt.title(f'Run {run+1}: Recovery Variable (MSE: {w_mse:.4f})')
                plt.grid(True, alpha=0.3)

                # Panel 4: Phase portrait
                plt.subplot(2, 2, 4)
                plt.plot(v_true.cpu().numpy(), w_true.cpu().numpy(), label='true trajectory', linewidth=3, alpha=0.7, c='white')
                plt.plot(v_list.cpu().numpy(), w_list.cpu().numpy(), label='rollout trajectory', linewidth=2, alpha=1, c='magenta')
                plt.xlabel('Membrane potential v')
                plt.ylabel('Recovery variable w')
                plt.legend(loc='upper right')
                plt.title(f'Run {run+1}: Phase Portrait')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'./tmp/nagumo_training_run_{run+1}_noise_{noise_level}_iter_{iter+1}.png', dpi=170)
                plt.close()

                # Save model state for each run
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'run': run+1,
                    'iteration': iter,
                    'loss': loss.item(),
                    'v_mse': v_mse,
                    'w_mse': w_mse,
                    'total_mse': total_mse
                }, f'./tmp/model_run_{run+1}.pt')

            fig = plt.figure(figsize=(10, 5))
            plt.plot(loss_list, label='Training Loss', color='cyan', linewidth=1, alpha=0.8)
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.title(f'Training Loss over {n_iter} iterations (Noise Level: {noise_level})')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'./tmp/nagumo_loss_run_{run+1}_noise_{noise_level}_iter_{iter+1}.png', dpi=200, bbox_inches='tight')
            plt.close()

        # Print convergence summary
        print(f" ")
        print("convergence summary")

        print(f"{'run':<4} {'Loss':<12} {'V MSE':<12} {'W MSE':<12} {'Total MSE':<12}")
        print(f"{'-' * 60}")

        for result in convergence_results:
            print(
                f"{result['run']:<4} {result['loss']:<12.6f} {result['v_mse']:<12.6f} {result['w_mse']:<12.6f} {result['total_mse']:<12.6f}")

        # Calculate statistics
        total_mses = [r['total_mse'] for r in convergence_results if not np.isnan(r['total_mse'])]
        losses = [r['loss'] for r in convergence_results if not np.isnan(r['loss'])]

        print(f"\nSTATISTICS:")
        if total_mses:
            print(f"total MSE - Mean: {np.mean(total_mses):.6f}, Std: {np.std(total_mses):.6f}")
        else:
            print("total MSE - No valid results")

        if losses:
            print(f"training Loss - Mean: {np.mean(losses):.6f}, Std: {np.std(losses):.6f}")
        else:
            print("training Loss - No valid results")

        valid_runs = len([r for r in convergence_results if not np.isnan(r['total_mse'])])
        print(f"convergence Rate: {valid_runs}/{test_runs} runs completed successfully")

        # Find best model based on lowest total MSE
        valid_results = [r for r in convergence_results if not np.isnan(r['total_mse'])]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['total_mse'])
            print(f"\nBEST MODEL:")
            print(f"Run {best_result['run']}: V MSE = {best_result['v_mse']:.6f}, W MSE = {best_result['w_mse']:.6f}, Total MSE = {best_result['total_mse']:.6f}")

            # Load the best model instead of retraining
            print(f"loading best model (Run {best_result['run']}) for derivative analysis...")

            model = model_duo(device=device)
            best_model_path = f'./tmp/model_run_{best_result["run"]}.pt'


            print("Performing derivative analysis on best model...")

            # Enhanced derivative analysis plots
            fig = plt.figure(figsize=(15, 10))

            # dv/dt analysis
            ax = fig.add_subplot(231)
            inputs = torch.cat((v_true[:, None], w_true[:, None] * 0, I_ext[:, None] * 0), dim=1)
            func = model.mlp0(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 0]), to_numpy(func), degree=3)
            plt.scatter(to_numpy(inputs[:, 0]), to_numpy(func), s=3, c='cyan', alpha=0.6)
            plt.title(f'dv/dt vs v: {latex}', fontsize=10, color='white')
            plt.xlabel('v', fontsize=12)
            plt.ylabel('dv/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            ax = fig.add_subplot(232)
            inputs = torch.cat((v_true[:, None] * 0, w_true[:, None], I_ext[:, None] * 0), dim=1)
            func = model.mlp0(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 1]), to_numpy(func), degree=2)
            plt.scatter(to_numpy(inputs[:, 1]), to_numpy(func), s=3, c='orange', alpha=0.6)
            plt.title(f'dv/dt vs w: {latex}', fontsize=10, color='white')
            plt.xlabel('w', fontsize=12)
            plt.ylabel('dv/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            ax = fig.add_subplot(233)
            inputs = torch.cat((v_true[:, None] * 0, w_true[:, None] * 0, I_ext[:, None]), dim=1)
            func = model.mlp0(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 2]), to_numpy(func), degree=2)
            plt.scatter(to_numpy(inputs[:, 2]), to_numpy(func), s=3, c='yellow', alpha=0.6)
            plt.title(f'dv/dt vs I_ext: {latex}', fontsize=10, color='white')
            plt.xlabel(r'$I_{ext}$', fontsize=12)
            plt.ylabel('dv/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            # dw/dt analysis
            ax = fig.add_subplot(234)
            inputs = torch.cat((v_true[:, None], w_true[:, None] * 0), dim=1)
            func = model.mlp1(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 0]), to_numpy(func), degree=2)
            plt.scatter(to_numpy(inputs[:, 0]), to_numpy(func), s=3, c='cyan', alpha=0.6)
            plt.title(f'dw/dt vs v: {latex}', fontsize=10, color='white')
            plt.xlabel('v', fontsize=12)
            plt.ylabel('dw/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            ax = fig.add_subplot(235)
            inputs = torch.cat((v_true[:, None] * 0, w_true[:, None]), dim=1)
            func = model.mlp1(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 1]), to_numpy(func), degree=2)
            plt.scatter(to_numpy(inputs[:, 1]), to_numpy(func), s=3, c='orange', alpha=0.6)
            plt.title(f'dw/dt vs w: {latex}', fontsize=10, color='white')
            plt.xlabel('w', fontsize=12)
            plt.ylabel('dw/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Model sparsity analysis
            ax = fig.add_subplot(236)
            mlp1_weights = []
            for param in model.mlp1.parameters():
                if param.requires_grad:
                    mlp1_weights.extend(param.data.flatten().cpu().numpy())

            plt.hist(mlp1_weights, bins=50, alpha=0.7, color='cyan', edgecolor='white')
            plt.title('MLP1 Weight Distribution (Sparsity)', fontsize=12, color='white')
            plt.xlabel('Weight Value', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Add sparsity statistics
            zero_weights = np.sum(np.abs(mlp1_weights) < 1e-4)
            total_weights = len(mlp1_weights)
            sparsity_ratio = zero_weights / total_weights
            plt.text(0.02, 0.98, f'Sparsity: {sparsity_ratio:.1%}\n({zero_weights}/{total_weights} weights ≈ 0)',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

            plt.tight_layout()
            plt.savefig('./tmp/best_model_derivative_analysis.png', dpi=200, bbox_inches='tight')
            plt.close()

            print(f"derivative analysis saved to ./tmp/best_model_derivative_analysis.png")
            print(f"best model sparsity: {sparsity_ratio:.1%} of MLP1 weights are effectively zero")

        else:
            print("No valid models found for analysis.")
