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



# ----------------- SIREN Network -----------------
class Siren2(nn.Module):
    def __init__(self, in_features=1, hidden_features=128, hidden_layers=3, out_features=1,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        layers.append(nn.Linear(hidden_features, out_features))  # final linear layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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


class model_duo(nn.Module):

    def __init__(self, device=None):

        super(model_duo, self).__init__()

        self.siren = Siren(in_features=1, out_features=1, hidden_features=128, hidden_layers=3, outermost_linear=True).to(device)
        self.mlp0 = MLP(input_size=3, output_size=1, nlayers=5, hidden_size=128, device=device)
        self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

    def forward(self, x):

        return self.siren(x)


if __name__ == '__main__':

    device = set_device('auto')
    print(f'device  {device}')

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
        v[i+1] = v[i] + dt * dv
        w[i+1] = w[i] + dt * dw

    plt.style.use('dark_background')

    # Plotting
    fig = plt.figure(figsize=(12, 6))
    # Time series
    plt.subplot(2, 1, 1)
    plt.plot(time, v, label='v (membrane potential)')
    plt.plot(time, w, label='w (recovery variable)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('FitzHugh-Nagumo with Periodic Excitation')
    plt.legend()
    plt.grid(True)
    # Plot I_ext
    plt.subplot(2, 1, 2)
    plt.plot(time, I_ext, color='red', label='External input $I_{ext}$')
    plt.xlabel('Time')
    plt.ylabel('Input current')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./tmp/time_series.png', dpi=170)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(v, w)
    plt.xlabel('v')
    plt.ylabel('w')
    plt.title('Phase Portrait')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./tmp/phase_portrait.png', dpi=170)
    plt.close()

    v_true = torch.tensor(v, dtype=torch.float32, device=device)
    w_true = torch.tensor(w, dtype=torch.float32, device=device)
    I_ext = torch.tensor(I_ext, dtype=torch.float32, device=device)

    n_steps = len(w_true)
    t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)  # shape (10000, 1)
    w_true = torch.tensor(w_true, dtype=torch.float32, device=device)  # shape (10000,)

    model = model_duo(device=device)    #Siren(in_features=1, out_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    n_iter = 10001
    batch_size = 2000
    n_intervals = 10000

    # Test convergence over multiple runs
    test_runs = 10
    convergence_results = []

    for run in range(test_runs):
        print(f" ")
        print(f"training run {run+1}/{test_runs}")

        run_converged = False

        for iter in range(n_iter):
            idx = torch.randint(1, n_steps-8, (batch_size,))
            t_batch = t_full[idx]
            idx = to_numpy(idx)

            training ='recursive'

            match training:
                case '1D':
                    pred = model.siren(t_batch)
                    w_batch = w_true[idx].unsqueeze(1)
                    loss = F.mse_loss(pred, w_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if iter % 1000 == 0:
                        print(f"iter {iter}, Loss: {loss.item():.6f}")
                        with torch.no_grad():
                            pred_full = model.siren(t_full).squeeze().cpu().numpy()
                            plt.figure(figsize=(10, 4))
                            plt.plot(w_true.cpu().numpy(), label="Target")
                            plt.plot(pred_full, label="SIREN Output")
                            plt.title(f"iter {iter}")
                            plt.legend()
                            plt.show()

                case 'recursive':

                    w = model.siren(t_batch)
                    v = v_true[idx, None].clone().detach()
                    optimizer.zero_grad()

                    recursive_loop = 3

                    for loop in range(recursive_loop):

                        dv_pred = model.mlp0(torch.cat((v, w, I_ext[idx, None].clone().detach()), dim=1))
                        dw_pred = model.mlp1(torch.cat((v, w), dim=1))

                        v = v + dt * dv_pred
                        w = w + dt * dw_pred

                        idx = idx + 1

                    w_siren = model.siren(t_full[idx])
                    loss = (v-v_true[idx, None]).norm(2) + (w-w_siren).norm(2)

                    loss.backward()
                    optimizer.step()

                    if (iter>0) & (iter % n_intervals == 0):
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
                                    # w = model.siren(t_full[step])

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
                            plt.plot(v_list.cpu().numpy(), label='rollout v', linewidth=2, alpha=1, c='salmon')
                            plt.xlim([0, n_steps//2.5])
                            plt.xlabel('Time steps')
                            plt.ylabel('Membrane potential v')
                            plt.legend(loc='upper left')
                            plt.title(f'Run {run+1}: Membrane Potential (MSE: {v_mse:.4f})')
                            plt.grid(True, alpha=0.3)

                            # Panel 3: True vs reconstructed recovery variable
                            plt.subplot(2, 2, 3)
                            plt.plot(w_true.cpu().numpy(), label='true w (recovery variable)', linewidth=3, alpha=0.7, c='blue')
                            plt.plot(w_list.cpu().numpy(), label='rollout w', linewidth=2, alpha=1, c='cyan')
                            plt.plot(w_pred.cpu().numpy(), label='SIREN w', linewidth=2, alpha=0.7, c='green')
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
                            plt.savefig(f'./tmp/training_run_{run+1}_iter_{iter}.png', dpi=170)
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
        print(f"Total MSE - Mean: {np.mean(total_mses):.6f}, Std: {np.std(total_mses):.6f}")
    else:
        print("Total MSE - No valid results")

    if losses:
        print(f"Training Loss - Mean: {np.mean(losses):.6f}, Std: {np.std(losses):.6f}")
    else:
        print("Training Loss - No valid results")

    valid_runs = len([r for r in convergence_results if not np.isnan(r['total_mse'])])
    print(f"Convergence Rate: {valid_runs}/{test_runs} runs completed successfully")





