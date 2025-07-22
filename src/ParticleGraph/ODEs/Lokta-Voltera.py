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
        self.mlp0 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)
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

    print("initializing SIREN with v values...")

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

    noise_level = 0

    # Actual Bakarji paper predator-prey coefficients from source code
    args = [1.0, 0.1, 1.5, 0.75]  # [a, b, c, d_factor]
    a = args[0]  # 1.0 - prey growth rate
    b = args[1]  # 0.1 - predation rate
    c = args[2]  # 1.5 - predator death rate
    d = args[1] * args[3]  # 0.1 * 0.75 = 0.075 - conversion efficiency

    # Simulation settings
    T = 60  # total time to see oscillations
    dt = 0.01  # time step
    noise_level = 0.0  # set to 0 first to see clean oscillations
    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)

    # Initialize variables
    v = np.zeros(n_steps)  # x (prey/species 1)
    w = np.zeros(n_steps)  # y (predator/species 2)

    # Bakarji paper initial conditions from source code
    v[0] = 10.0  # z0_mean_sug[0] = 10
    w[0] = 5.0  # z0_mean_sug[1] = 5

    # Exact Bakarji paper equations from source code
    # see https://github.com/josephbakarji/deep-delay-autoencoder/tree/main
    for i in range(n_steps - 1):
        dv = a * v[i] - b * v[i] * w[i]  # args[0]*z[0] - args[1]*z[0]*z[1]
        dw = -c * w[i] + d * v[i] * w[i]  # -args[2]*z[1] + args[1]*args[3]*z[0]*z[1]

        v[i + 1] = v[i] + dt * dv + noise_level * np.random.randn()
        w[i + 1] = w[i] + dt * dw + noise_level * np.random.randn()

    # Plot
    plt.style.use('dark_background')
    plt.figure(figsize=(15, 4))

    # Panel 1: v vs t
    plt.subplot(1, 3, 1)
    plt.plot(time, v, 'cyan', linewidth=1, label='x (prey)')
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.title('x vs Time')

    # Panel 2: w vs t
    plt.subplot(1, 3, 2)
    plt.plot(time, w, 'orange', linewidth=1, label='y (predator)')
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.title('y vs Time')

    # Panel 3: v vs w (phase plot)
    plt.subplot(1, 3, 3)
    plt.plot(v, w, 'lime', linewidth=1, alpha=0.7)
    plt.xlabel('x (prey)')
    plt.ylabel('y (predator)')
    plt.title('Phase Plot (x vs y)')

    plt.tight_layout()
    plt.savefig(f'./tmp/voltera_bakarji_ode_simulation_noise_{noise_level}.png', dpi=200, bbox_inches='tight')
    plt.close()

    # print(f'mean min max v: {np.mean(v):.4f} {np.min(v):.4f} {np.max(v):.4f}')

    v_true = torch.tensor(v, dtype=torch.float32, device=device)
    w_true = torch.tensor(w, dtype=torch.float32, device=device)

    n_steps = len(w_true) # shape (10000, 1)
    w_true = torch.tensor(w_true, dtype=torch.float32, device=device)
    t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)

    n_iter = 8000
    batch_size = n_steps // 2

    # Test convergence over multiple runs
    test_runs = 5
    convergence_results = []

    print(f" ")

    for run in range(test_runs):
        print(f"training run {run+1}/{test_runs}")

        model = model_duo(device=device)  # Siren(in_features=1, out_features=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Initialize SIREN with v values
        # model = initialize_siren_with_v(model, v_true, t_full, device, n_init_steps=5000)
        # plot_siren_v_init(model, v_true, t_full)

        Time.sleep(1)
        loss_list = []
        for iter in range(n_iter):
            idx = torch.randint(1, n_steps-8, (batch_size,))
            idx = torch.sort(torch.unique(idx)).values
            t_batch = t_full[idx.clone().detach()]
            idx = to_numpy(idx)

            w = model.siren(t_batch)
            v = v_true[idx, None].clone().detach()
            optimizer.zero_grad()

            recursive_loop = 8
            l1_lambda = 1E-3

            # Store intermediate states for gradient accumulation
            accumulated_loss = 0.0

            for loop in range(recursive_loop):

                dv_pred = model.mlp0(torch.cat((v, w), dim=1))
                dw_pred = model.mlp1(torch.cat((v, w), dim=1))

                v = v + dt * dv_pred
                w = w + dt * dw_pred

                # Compute loss for this step directly
                step_idx = idx + loop + 1
                v_target = v_true[step_idx, None]
                w_target_siren = model.siren(t_full[step_idx])

                # Calculate losses for this step
                v_step_loss = (v - v_target).norm(2) * 100
                w_step_loss = (w - w_target_siren).norm(2)

                # Weight losses by step (later steps get higher weight)
                step_weight = (loop + 1) / recursive_loop
                step_loss = step_weight * (v_step_loss + w_step_loss)

                accumulated_loss += step_loss

            # Add regularization penalties


            l1_penalty = 0.0
            for param in model.mlp1.parameters():
                l1_penalty += torch.sum(torch.abs(param))

            # weight_decay = 1e-6
            # l2_penalty = 0.0
            # for param in model.parameters():
            #     l2_penalty += torch.sum(param ** 2)

            # Final loss with regularization
            loss = accumulated_loss + l1_lambda * l1_penalty # + weight_decay * l2_penalty

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if iter % 500 == 0:
                print(f"iteration {iter+1}/{n_iter}, loss: {loss.item():.6f}")

        Time.sleep(1)
        # rollout analysis
        with torch.no_grad():
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

                    dv_pred = model.mlp0(torch.cat((v[:, None], w[:, None]), dim=1))
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

            # Panel 2: True vs reconstructed membrane potential
            plt.subplot(2, 2, 1)
            plt.plot(v_true.cpu().numpy(), label='true v (membrane potential)', linewidth=3, alpha=0.7, c='white')
            plt.plot(v_list.cpu().numpy(), label='rollout v', linewidth=2, alpha=1, c='green')
            plt.xlim([0, n_steps//2])
            plt.xlabel('Time steps')
            plt.ylabel('Membrane potential v')
            plt.legend(loc='upper left')
            plt.title(f'Run {run+1}: Membrane Potential (MSE: {v_mse:.4f})')
            plt.grid(True, alpha=0.3)

            # Panel 3: True vs reconstructed recovery variable
            plt.subplot(2, 2, 2)
            plt.plot(w_true.cpu().numpy(), label='true w (recovery variable)', linewidth=3, alpha=0.7, c='white')
            plt.plot(w_list.cpu().numpy(), label='rollout w', linewidth=2, alpha=1, c='cyan')
            plt.plot(w_pred.cpu().numpy(), label='SIREN w', linewidth=2, alpha=0.7, c='cyan')
            plt.xlim([0, n_steps//2])
            plt.xlabel('Time steps')
            plt.ylabel('Recovery variable w')
            plt.legend(loc='upper left')
            plt.title(f'Run {run+1}: Recovery Variable (MSE: {w_mse:.4f})')
            plt.grid(True, alpha=0.3)

            # Panel 4: Phase portrait
            plt.subplot(2, 2, 3)
            plt.plot(v_true[0:n_steps//2].cpu().numpy(), w_true[0:n_steps//2].cpu().numpy(), label='true trajectory', linewidth=3, alpha=0.7, c='white')
            plt.plot(v_list[0:n_steps//2].cpu().numpy(), w_list[0:n_steps//2].cpu().numpy(), label='rollout trajectory', linewidth=2, alpha=1, c='magenta')
            plt.xlabel('Membrane potential v')
            plt.ylabel('Recovery variable w')
            plt.legend(loc='upper right')
            plt.title(f'Run {run+1}: Phase Portrait')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'./tmp/voltera_training_run_{run+1}_noise_{noise_level}_iter_{iter+1}.png', dpi=170)
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
        plt.savefig(f'./tmp/voltera_loss_run_{run+1}_noise_{noise_level}_iter_{iter+1}.png', dpi=200, bbox_inches='tight')
        plt.close()


        def extract_coeffs_regression(model, device, v_traj, w_traj):
            """Extract Volterra coefficients using polynomial regression"""

            # Use trajectory points directly (no meshgrid needed)
            inputs = torch.stack([v_traj, w_traj], dim=1)

            with torch.no_grad():
                # Get MLP outputs
                dv_dt = model.mlp0(inputs).squeeze()
                dw_dt = model.mlp1(inputs).squeeze()

                # Rest stays the same...
                v_flat = v_traj  # rename for consistency
                w_flat = w_traj

                # Build feature matrices for order 2 regression
                # For mlp0 (dv/dt): expect a*v - b*v*w
                X0 = torch.stack([v_flat, v_flat * w_flat], dim=1)  # [v, v*w]

                # For mlp1 (dw/dt): expect -c*w + d*v*w
                X1 = torch.stack([w_flat, v_flat * w_flat], dim=1)  # [w, v*w]

                # Solve least squares: X * coeffs = y
                coeffs0 = torch.linalg.lstsq(X0, dv_dt).solution
                coeffs1 = torch.linalg.lstsq(X1, dw_dt).solution

                # Extract coefficients
                a_reg = coeffs0[0].item()  # coeff of v
                b_reg = -coeffs0[1].item()  # coeff of -v*w
                c_reg = -coeffs1[0].item()  # coeff of -w
                d_reg = coeffs1[1].item()  # coeff of v*w

            return a_reg, b_reg, c_reg, d_reg


        # Extract coefficients using regression
        t_full_ = torch.linspace(0, 0.5, n_steps//2, device=device).unsqueeze(1)
        w_pred = model.siren(t_full_).squeeze()
        a_reg, b_reg, c_reg, d_reg = extract_coeffs_regression(model, device, v_true[0:n_steps//2], w_pred[0:n_steps//2])

        print("\nREGRESSION COEFFICIENT EXTRACTION:")
        print("ground Truth: a=1.0, b=0.1, c=1.5, d=0.075")
        print(f"regression:   a={a_reg:.3f}, b={b_reg:.3f}, c={c_reg:.3f}, d={d_reg:.3f}")

