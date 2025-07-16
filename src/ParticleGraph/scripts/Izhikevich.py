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


class MLP(nn.Module):

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None, initialisation=None):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.05)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)

        if initialisation == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else :
            nn.init.normal_(layer.weight, std=0.05)
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

        self.siren = Siren(in_features=1, out_features=1, hidden_features=256, hidden_layers=5, first_omega_0=80., hidden_omega_0=80., outermost_linear=True).to(device)
        self.mlp_v = MLP(input_size=2, output_size=1, nlayers=5, hidden_size=128, device=device)  # dv/dt = f(v, u)
        self.mlp_u = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)   # du/dt = f(v, u)

    def forward(self, x):
        # Returns hidden variable u from SIREN
        return self.siren(x)


if __name__ == '__main__':

    device = set_device('auto')
    print(f'device  {device}')

    noise_level =  1.0E-3

    print(f" ")
    print(f"running simulation with noise level: {noise_level}")

    # Izhikevich model parameters - more excitable
    a = 0.02   # recovery time constant
    b = 0.2    # sensitivity of recovery variable u to subthreshold fluctuations
    c = -65    # after-spike reset value of v
    d = 6      # after-spike reset of u (reduced for more excitability)

    # Simulation settings
    T = 1000       # total time (ms)
    dt = 0.1       # time step (ms)
    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)

    # Initialize variables
    v = np.zeros(n_steps)
    u = np.zeros(n_steps)
    I_ext = np.zeros(n_steps)

    # Initial conditions
    v[0] = -70.0  # resting potential
    u[0] = b * v[0]  # steady state u

    # External current: much stronger stimulation
    pulse_interval = 100.0   # ms
    pulse_duration = 10.0    # ms (longer pulses)
    pulse_amplitude = 60.0   # even stronger current to ensure reliable spiking

    for i, t in enumerate(time):
        if (t % pulse_interval) < pulse_duration:
            I_ext[i] = pulse_amplitude
        else:
            I_ext[i] = 20.0  # even higher background current

    # Rollout using Euler method with spike reset
    for i in range(n_steps - 1):
        # Izhikevich equations
        dv = 0.04 * v[i]**2 + 5 * v[i] + 140 - u[i] + I_ext[i]
        du = a * (b * v[i] - u[i])

        v[i+1] = v[i] + dt * dv + noise_level * np.random.randn()
        u[i+1] = u[i] + dt * du + noise_level * np.random.randn()

        # Spike reset condition - slightly lower threshold
        if v[i+1] >= 28:  # reduced spike threshold to ensure triggering
            v[i+1] = c    # reset membrane potential
            u[i+1] = u[i+1] + d  # reset recovery variable

    plt.style.use('dark_background')

    v_true = torch.tensor(v, dtype=torch.float32, device=device)
    u_true = torch.tensor(u, dtype=torch.float32, device=device)
    I_ext = torch.tensor(I_ext, dtype=torch.float32, device=device)

    # Plot the simulation before training
    print("Plotting simulation dynamics...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Plot 1: External current
    axes[0].plot(time, I_ext.cpu().numpy(), 'r-', linewidth=2, alpha=0.8)
    axes[0].set_ylabel('External Current (pA)', fontsize=12)
    axes[0].set_title('Izhikevich Model Simulation - External Stimulation', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 500])  # Show first 500ms

    # Plot 2: Membrane potential (observable)
    axes[1].plot(time, v, 'g-', linewidth=2, alpha=0.8)
    axes[1].set_ylabel('Membrane Potential v (mV)', fontsize=12)
    axes[1].set_title('Membrane Potential - Observable Variable', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 500])
    axes[1].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Spike threshold')
    axes[1].axhline(y=-65, color='orange', linestyle='--', alpha=0.5, label='Reset value')
    axes[1].legend()

    # Plot 3: Recovery variable (hidden)
    axes[2].plot(time, u, 'c-', linewidth=2, alpha=0.8)
    axes[2].set_ylabel('Recovery Variable u', fontsize=12)
    axes[2].set_title('Recovery Variable - Hidden Variable', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 500])

    # Plot 4: Phase portrait
    axes[3].plot(v, u, 'm-', linewidth=2, alpha=0.8)
    axes[3].set_xlabel('Membrane Potential v (mV)', fontsize=12)
    axes[3].set_ylabel('Recovery Variable u', fontsize=12)
    axes[3].set_title('Phase Portrait (v-u space)', fontsize=14)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./tmp/izhikevich_simulation_before_training.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Print some statistics about the simulation
    print(f"Simulation statistics:")
    print(f"  Membrane potential range: {np.min(v):.2f} to {np.max(v):.2f} mV")
    print(f"  Recovery variable range: {np.min(u):.2f} to {np.max(u):.2f}")
    print(f"  Number of spikes (v > 25): {np.sum(v > 25)}")
    print(f"  External current range: {np.min(I_ext.cpu().numpy()):.2f} to {np.max(I_ext.cpu().numpy()):.2f} pA")
    print(f"  Saved simulation plot to: ./tmp/izhikevich_simulation_before_training.png")
    print("")

    n_steps = len(u_true)
    t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)  # shape (n_steps, 1)

    pretrain_runs = 20
    pretrain_iterations = 500
    pretraining_results = []
    batch_size = 2000

    for run in range(pretrain_runs):
        # Set different seed for each pre-training run
        seed = 42 + run * 1000  # Different seed for each run
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        model = model_duo(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        for iter in range(pretrain_iterations + 1):
            idx = torch.randint(1, n_steps-8, (batch_size,))
            t_batch = t_full[idx]
            idx = to_numpy(idx)

            training = 'recursive'

            match training:
                case 'recursive':
                    u = model.siren(t_batch)  # u is hidden variable from SIREN
                    v = v_true[idx, None].clone().detach()  # v is observable, start from true value
                    optimizer.zero_grad()

                    # Simpler curriculum for pre-training
                    if iter < 150:
                        recursive_loop = 2
                    elif iter < 300:
                        recursive_loop = 4
                    else:
                        recursive_loop = 6

                    # Store intermediate states for gradient accumulation
                    v_states = [v.clone()]
                    u_states = [u.clone()]
                    accumulated_loss = 0.0

                    for loop in range(recursive_loop):

                        # Predict derivatives
                        dv_pred = model.mlp_v(torch.cat((v, u), dim=1))
                        du_pred = model.mlp_u(torch.cat((v, u), dim=1))

                        # Euler step
                        v = v + dt * dv_pred
                        u = u + dt * du_pred

                        # Store states for gradient accumulation
                        v_states.append(v.clone())
                        u_states.append(u.clone())

                        idx = idx + 1

                    # Gradient accumulation across all recursive steps
                    for step in range(recursive_loop):
                        step_idx = idx - recursive_loop + step + 1

                        # Get target values for this step
                        v_target = v_true[step_idx, None]
                        u_target_siren = model.siren(t_full[step_idx])

                        # Calculate losses for this step
                        v_step_loss = (v_states[step + 1] - v_target).norm(2)
                        u_step_loss = (u_states[step + 1] - u_target_siren).norm(2)

                        # Weight losses by step (later steps get higher weight)
                        step_weight = (step + 1) / recursive_loop
                        step_loss = step_weight * (v_step_loss + u_step_loss)

                        accumulated_loss += step_loss

                    # Add regularization penalties
                    l1_lambda = 1.0E-3
                    l1_penalty = 0.0
                    for param in model.mlp_u.parameters():
                        l1_penalty += torch.sum(torch.abs(param))

                    weight_decay = 1e-6
                    l2_penalty = 0.0
                    for param in model.parameters():
                        l2_penalty += torch.sum(param ** 2)

                    # Final loss with regularization
                    loss = accumulated_loss + l1_lambda * l1_penalty + weight_decay * l2_penalty

                    loss.backward()
                    optimizer.step()

        # Evaluate pre-trained model
        with torch.no_grad():
            v = v_true[0:1].clone().detach()
            u = u_true[0:1].clone().detach()
            v_list = []
            u_list = []
            v_list.append(v.clone().detach())
            u_list.append(u.clone().detach())

            for step in range(1, n_steps):
                dv_pred = model.mlp_v(torch.cat((v[:, None], u[:, None]), dim=1))
                du_pred = model.mlp_u(torch.cat((v[:, None], u[:, None]), dim=1))

                v += dt * dv_pred.squeeze()
                u += dt * du_pred.squeeze()

                v_list.append(v.clone().detach())
                u_list.append(u.clone().detach())

            v_list = torch.stack(v_list, dim=0)
            u_list = torch.stack(u_list, dim=0)

            v_mse = F.mse_loss(v_list[500:].squeeze(), v_true[500:]).item()
            u_mse = F.mse_loss(u_list[500:].squeeze(), u_true[500:]).item()
            total_mse = v_mse + u_mse

            pretraining_results.append({
                'run': run+1,
                'total_mse': total_mse,
                'v_mse': v_mse,
                'u_mse': u_mse,
                'final_loss': loss.item(),
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            })

            print(f"pre-run {run+1}: V MSE = {v_mse:.6f}, U MSE = {u_mse:.6f}, Total MSE = {total_mse:.6f}")

    # Select best pre-trained model (filter out NaN and inf values)
    valid_pretraining_results = [r for r in pretraining_results if
                                not (np.isnan(r['u_mse']) or np.isinf(r['u_mse']))]

    if valid_pretraining_results:
        best_pretrain = min(valid_pretraining_results, key=lambda x: x['u_mse'])
        print(f"\nBest pre-trained model: Run {best_pretrain['run']}")
        print(f"Best pre-training MSE: {best_pretrain['u_mse']:.6f}")
        print(f"=" * 60)

        # MAIN TRAINING: Continue with best pre-trained model
        print(f"Starting main training with best pre-trained model (Run {best_pretrain['run']})")

        # Load the best pre-trained model
        model = model_duo(device=device)
        model.load_state_dict(best_pretrain['model_state'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        optimizer.load_state_dict(best_pretrain['optimizer_state'])

    convergence_results = []

    print(f" ")
    print(f"main training")  # Only one main training run

    n_iter = 50000
    n_intervals = 10000

    for iter in trange(n_iter+1):
        idx = torch.randint(1, n_steps-8, (batch_size,))
        t_batch = t_full[idx]
        idx = to_numpy(idx)

        training ='recursive'

        match training:
            case 'recursive':
                u = model.siren(t_batch)  # u is hidden variable from SIREN
                v = v_true[idx, None].clone().detach()  # v is observable, start from true value
                optimizer.zero_grad()

                if iter < 1000:
                    recursive_loop = 2
                elif iter < 3000:
                    recursive_loop = 4
                else:
                    recursive_loop = 6

                # Store intermediate states for gradient accumulation
                v_states = [v.clone()]
                u_states = [u.clone()]
                accumulated_loss = 0.0

                for loop in range(recursive_loop):

                    # Predict derivatives
                    dv_pred = model.mlp_v(torch.cat((v, u), dim=1))
                    du_pred = model.mlp_u(torch.cat((v, u), dim=1))

                    # Euler step
                    v = v + dt * dv_pred
                    u = u + dt * du_pred

                    # Store states for gradient accumulation
                    v_states.append(v.clone())
                    u_states.append(u.clone())

                    idx = idx + 1

                # Gradient accumulation across all recursive steps
                for step in range(recursive_loop):
                    step_idx = idx - recursive_loop + step + 1

                    # Get target values for this step
                    v_target = v_true[step_idx, None]
                    u_target_siren = model.siren(t_full[step_idx])

                    # Calculate losses for this step
                    v_step_loss = (v_states[step + 1] - v_target).norm(2)
                    u_step_loss = (u_states[step + 1] - u_target_siren).norm(2)

                    # Weight losses by step (later steps get higher weight)
                    step_weight = (step + 1) / recursive_loop
                    step_loss = step_weight * (v_step_loss + u_step_loss)

                    accumulated_loss += step_loss

                # Add regularization penalties
                l1_lambda = 1.0E-3
                l1_penalty = 0.0
                for param in model.mlp_u.parameters():
                    l1_penalty += torch.sum(torch.abs(param))

                weight_decay = 1e-6
                l2_penalty = 0.0
                for param in model.parameters():
                    l2_penalty += torch.sum(param ** 2)

                # Final loss with regularization
                loss = accumulated_loss + l1_lambda * l1_penalty + weight_decay * l2_penalty

                loss.backward()
                optimizer.step()

                if (iter>0) & (iter % n_intervals == 0):
                    with torch.no_grad():
                        u_pred = model(t_full)

                        v = v_true[0:1].clone().detach()
                        u = u_true[0:1].clone().detach()
                        v_list = []
                        u_list = []
                        v_list.append(v.clone().detach())
                        u_list.append(u.clone().detach())

                        for step in range(1, n_steps):
                            with torch.no_grad():
                                u = model.siren(t_full[step])

                                dv_pred = model.mlp_v(torch.cat((v[:, None], u[:, None]), dim=1))
                                du_pred = model.mlp_u(torch.cat((v[:, None], u[:, None]), dim=1))

                                v += dt * dv_pred.squeeze()
                                u += dt * du_pred.squeeze()

                            v_list.append(v.clone().detach())
                            u_list.append(u.clone().detach())

                        v_list = torch.stack(v_list, dim=0)
                        u_list = torch.stack(u_list, dim=0)

                        v_mse = F.mse_loss(v_list[500:].squeeze(), v_true[500:]).item()
                        u_mse = F.mse_loss(u_list[500:].squeeze(), u_true[500:]).item()
                        total_mse = v_mse + u_mse

                        convergence_results.append({
                            'run': 1,
                            'iteration': iter,
                            'loss': loss.item(),
                            'v_mse': v_mse,
                            'u_mse': u_mse,
                            'total_mse': total_mse
                        })

                        print(f"V MSE: {v_mse:.6f}, U MSE: {u_mse:.6f}, Total MSE: {total_mse:.6f}")

                        # Save results for this run
                        fig = plt.figure(figsize=(16, 12))

                        # Panel 1: External input only
                        plt.subplot(2, 2, 1)
                        plt.plot(I_ext.cpu().numpy(), label='I_ext (external current)', linewidth=2, alpha=0.7, c='red')
                        plt.xlim([0, n_steps//2.5])
                        plt.xlabel('Time steps')
                        plt.ylabel('External current (pA)')
                        plt.legend(loc='upper left')
                        plt.title('External Input Current')
                        plt.grid(True, alpha=0.3)

                        # Panel 2: True vs reconstructed membrane potential (observable)
                        plt.subplot(2, 2, 2)
                        plt.plot(v_true.cpu().numpy(), label='true v (membrane potential)', linewidth=3, alpha=0.7, c='white')
                        plt.plot(v_list.cpu().numpy(), label='rollout v', linewidth=2, alpha=1, c='green')
                        plt.xlim([0, n_steps//2.5])
                        plt.xlabel('Time steps')
                        plt.ylabel('Membrane potential (mV)')
                        plt.legend(loc='upper left')
                        plt.title(f'Run {1}: Membrane Potential - Observable (MSE: {v_mse:.4f})')
                        plt.grid(True, alpha=0.3)

                        # Panel 3: True vs reconstructed recovery variable (hidden)
                        plt.subplot(2, 2, 3)
                        plt.plot(u_true.cpu().numpy(), label='true u (recovery variable)', linewidth=3, alpha=0.7, c='white')
                        plt.plot(u_list.cpu().numpy(), label='rollout u', linewidth=2, alpha=1, c='cyan')
                        plt.plot(u_pred.cpu().numpy(), label='SIREN u', linewidth=2, alpha=0.7, c='cyan')
                        plt.xlim([0, n_steps//2.5])
                        plt.xlabel('Time steps')
                        plt.ylabel('Recovery variable u')
                        plt.legend(loc='upper left')
                        plt.title(f'Run {1}: Recovery Variable - Hidden (MSE: {u_mse:.4f})')
                        plt.grid(True, alpha=0.3)

                        # Panel 4: Phase portrait
                        plt.subplot(2, 2, 4)
                        plt.plot(v_true.cpu().numpy(), u_true.cpu().numpy(), label='true trajectory', linewidth=3, alpha=0.7, c='white')
                        plt.plot(v_list.cpu().numpy(), u_list.cpu().numpy(), label='rollout trajectory', linewidth=2, alpha=1, c='magenta')
                        plt.xlabel('Membrane potential v (mV)')
                        plt.ylabel('Recovery variable u')
                        plt.legend(loc='upper right')
                        plt.title(f'Run {1}: Phase Portrait (v-u space)')
                        plt.grid(True, alpha=0.3)

                        plt.tight_layout()
                        plt.savefig(f'./tmp/izhikevich_training_noise_{noise_level}_run_{1}_iter_{iter}.png', dpi=170)
                        plt.close()

                        # Save model state for each run
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'run': 1,
                            'iteration': iter,
                            'loss': loss.item(),
                            'v_mse': v_mse,
                            'u_mse': u_mse,
                            'total_mse': total_mse
                        }, f'./tmp/izhikevich_model_run_{1}.pt')

    # Print convergence summary
    print(f" ")
    print("convergence summary")

    print(f"{'run':<4} {'Loss':<12} {'V MSE':<12} {'U MSE':<12} {'Total MSE':<12}")
    print(f"{'-' * 60}")

    for result in convergence_results:
        print(
            f"{result['run']:<4} {result['loss']:<12.6f} {result['v_mse']:<12.6f} {result['u_mse']:<12.6f} {result['total_mse']:<12.6f}")

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
    test_runs = 1  # Since we only have one main training run
    print(f"Convergence Rate: {valid_runs}/{test_runs} runs completed successfully")

    # Find best model based on lowest total MSE
    valid_results = [r for r in convergence_results if not np.isnan(r['total_mse'])]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['total_mse'])
        print(f"\nBEST MODEL:")
        print(f"Run {best_result['run']}: V MSE = {best_result['v_mse']:.6f}, U MSE = {best_result['u_mse']:.6f}, Total MSE = {best_result['total_mse']:.6f}")

        # Load the best model instead of retraining
        print(f"Loading best model (Run {best_result['run']}) for derivative analysis...")

        model = model_duo(device=device)
        best_model_path = f'./tmp/izhikevich_model_run_{best_result["run"]}.pt'

        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model loaded successfully from {best_model_path}")
        except FileNotFoundError:
            print(f"Error: Could not find saved model at {best_model_path}")
            print("Skipping derivative analysis...")
        else:
            print("Performing derivative analysis on best model...")

            # Enhanced derivative analysis plots
            fig = plt.figure(figsize=(15, 10))

            # dv/dt analysis
            ax = fig.add_subplot(231)
            inputs = torch.cat((v_true[:, None], u_true[:, None] * 0), dim=1)
            func = model.mlp_v(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 0]), to_numpy(func), degree=3)
            plt.scatter(to_numpy(inputs[:, 0]), to_numpy(func), s=3, c='cyan', alpha=0.6)
            plt.title(f'dv/dt vs v: {latex}', fontsize=10, color='white')
            plt.xlabel('v (mV)', fontsize=12)
            plt.ylabel('dv/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            ax = fig.add_subplot(232)
            inputs = torch.cat((v_true[:, None] * 0, u_true[:, None]), dim=1)
            func = model.mlp_v(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 1]), to_numpy(func), degree=2)
            plt.scatter(to_numpy(inputs[:, 1]), to_numpy(func), s=3, c='orange', alpha=0.6)
            plt.title(f'dv/dt vs u: {latex}', fontsize=10, color='white')
            plt.xlabel('u', fontsize=12)
            plt.ylabel('dv/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            ax = fig.add_subplot(233)
            inputs = torch.cat((v_true[:, None], u_true[:, None]), dim=1)
            func = model.mlp_v(inputs)
            plt.scatter(to_numpy(v_true), to_numpy(func), s=3, c='red', alpha=0.6)
            plt.title('dv/dt vs v (full dynamics)', fontsize=10, color='white')
            plt.xlabel('v (mV)', fontsize=12)
            plt.ylabel('dv/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            # du/dt analysis
            ax = fig.add_subplot(234)
            inputs = torch.cat((v_true[:, None], u_true[:, None] * 0), dim=1)
            func = model.mlp_u(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 0]), to_numpy(func), degree=2)
            plt.scatter(to_numpy(inputs[:, 0]), to_numpy(func), s=3, c='cyan', alpha=0.6)
            plt.title(f'du/dt vs v: {latex}', fontsize=10, color='white')
            plt.xlabel('v (mV)', fontsize=12)
            plt.ylabel('du/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            ax = fig.add_subplot(235)
            inputs = torch.cat((v_true[:, None] * 0, u_true[:, None]), dim=1)
            func = model.mlp_u(inputs)
            poly, latex = fit_polynomial_with_latex(to_numpy(inputs[:, 1]), to_numpy(func), degree=2)
            plt.scatter(to_numpy(inputs[:, 1]), to_numpy(func), s=3, c='orange', alpha=0.6)
            plt.title(f'du/dt vs u: {latex}', fontsize=10, color='white')
            plt.xlabel('u', fontsize=12)
            plt.ylabel('du/dt', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Model sparsity analysis
            ax = fig.add_subplot(236)
            mlp_u_weights = []
            for param in model.mlp_u.parameters():
                if param.requires_grad:
                    mlp_u_weights.extend(param.data.flatten().cpu().numpy())

            plt.hist(mlp_u_weights, bins=50, alpha=0.7, color='cyan', edgecolor='white')
            plt.title('MLP_u Weight Distribution (Sparsity)', fontsize=12, color='white')
            plt.xlabel('Weight Value', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Add sparsity statistics
            zero_weights = np.sum(np.abs(mlp_u_weights) < 1e-4)
            total_weights = len(mlp_u_weights)
            sparsity_ratio = zero_weights / total_weights
            plt.text(0.02, 0.98, f'Sparsity: {sparsity_ratio:.1%}\n({zero_weights}/{total_weights} weights ≈ 0)',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

            plt.tight_layout()
            plt.savefig('./tmp/izhikevich_best_model_derivative_analysis.png', dpi=200, bbox_inches='tight')
            plt.close()

            print(f"Derivative analysis saved to ./tmp/izhikevich_best_model_derivative_analysis.png")
            print(f"Best model sparsity: {sparsity_ratio:.1%} of MLP_u weights are effectively zero")

            # Save best model with additional analysis
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_result': best_result,
                'sparsity_ratio': sparsity_ratio,
                'original_checkpoint': checkpoint
            }, './tmp/izhikevich_best_model_analyzed.pt')

            print("Best model with analysis saved to ./tmp/izhikevich_best_model_analyzed.pt")
    else:
        print("No valid models found for analysis.")
