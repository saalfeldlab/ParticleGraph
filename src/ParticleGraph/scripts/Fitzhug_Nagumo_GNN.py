import os

import numpy as np
import torch
import torch.nn as nn

# from ParticleGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt
from tifffile import imread, imsave
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
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)

        if initialisation == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else:
            nn.init.normal_(layer.weight, std=0.1)
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
    T = 1000  # total time
    dt = 0.1  # time step
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
    pulse_duration = 1.0  # seconds
    pulse_amplitude = 0.8  # strength of excitation

    for i, t in enumerate(time):
        if (t % pulse_interval) < pulse_duration:
            I_ext[i] = pulse_amplitude

    # Rollout using Euler method
    for i in range(n_steps - 1):
        dv = v[i] - (v[i] ** 3) / 3 - w[i] + I_ext[i]
        dw = epsilon * (v[i] + a - b * w[i])
        v[i + 1] = v[i] + dt * dv
        w[i + 1] = w[i] + dt * dw

    # Create tmp directory
    os.makedirs('tmp', exist_ok=True)

    plt.style.use('dark_background')

    # Plotting ground truth
    fig = plt.figure(figsize=(10, 10))
    # Time series
    plt.subplot(2, 2, 1)
    plt.plot(time, I_ext, color='red', linewidth=2)
    plt.xlabel('time', fontsize=16)
    plt.ylabel(r'$I_{ext}$', fontsize=16)
    plt.xlim([0, 300])
    plt.ylim([0, 1])
    plt.subplot(2, 2, 2)
    plt.plot(time, v, c='white', linewidth=2)
    plt.xlim([0, 300])
    plt.ylim([-2, 2])
    plt.xlabel('time', fontsize=16)
    plt.ylabel('v', fontsize=16)
    plt.subplot(2, 2, 4)
    plt.plot(time, w, c='green', linewidth=2)
    plt.xlim([0, 300])
    plt.ylim([-1.5, 1.5])
    plt.xlabel('time', fontsize=16)
    plt.ylabel('w', fontsize=16)
    plt.tight_layout()
    plt.savefig('./tmp/time_series_Nagumo.png', dpi=170)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(v, w)
    plt.xlabel('v', fontsize=16)
    plt.ylabel('w', fontsize=16)
    plt.title('phase Portrait')
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

    model = model_duo(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Try to load pre-trained model
    try:
        state_dict = torch.load(f'tmp/model_0.pt', map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print("Loaded pre-trained model")
    except:
        print("No pre-trained model found, starting from scratch")

    n_epochs = 200000
    batch_size = 2000

    # Training loss tracking
    train_losses = []
    eval_interval = 10000

    for epoch in trange(n_epochs):
        idx = torch.randint(1, n_steps - 8, (batch_size,))
        t_batch = t_full[idx]
        idx = to_numpy(idx)

        training = 'recursive'

        match training:
            case '1D':
                pred = model.siren(t_batch)
                w_batch = w_true[idx].unsqueeze(1)
                loss = F.mse_loss(pred, w_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

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
                loss = (v - v_true[idx, None]).norm(2) + (w - w_siren).norm(2)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                if (epoch >= 0) & (epoch % eval_interval == 0):
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

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
                                dv_pred = model.mlp0(
                                    torch.cat((v[:, None], w[:, None], I_ext[step:step + 1, None]), dim=1))
                                dw_pred = model.mlp1(torch.cat((v[:, None], w[:, None]), dim=1))

                                v += dt * dv_pred.squeeze()
                                w += dt * dw_pred.squeeze()

                            v_list.append(v.clone().detach())
                            w_list.append(w.clone().detach())

                        v_rollout = torch.stack(v_list, dim=0).squeeze().cpu().numpy()
                        w_rollout = torch.stack(w_list, dim=0).squeeze().cpu().numpy()

                        # Enhanced comprehensive plotting
                        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

                        # Plot 1: v comparison
                        axes[0, 0].plot(time, v_true.cpu().numpy(), label='True v',
                                        linewidth=2, alpha=0.7, color='white')
                        axes[0, 0].plot(time, v_rollout, label='SIREN+MLP rollout v',
                                        linewidth=2, color='cyan')
                        axes[0, 0].set_ylim([-2, 2])
                        axes[0, 0].set_xlabel('Time')
                        axes[0, 0].set_ylabel('Membrane Potential v')
                        axes[0, 0].legend()
                        axes[0, 0].set_title("Membrane Potential: SIREN+MLP vs True")
                        axes[0, 0].grid(True, alpha=0.3)

                        # Plot 2: w comparison and external current
                        ax2_twin = axes[0, 1].twinx()
                        line1 = axes[0, 1].plot(time, w_true.cpu().numpy(), label='True w',
                                                linewidth=2, color='orange', alpha=0.8)
                        axes[0, 1].plot(time, w_rollout, label='SIREN+MLP rollout w',
                                        linewidth=2, color='green', alpha=0.8)
                        line2 = ax2_twin.plot(time, I_ext.cpu().numpy(), label='External Current I_ext',
                                              linewidth=2, color='yellow', alpha=0.6)
                        axes[0, 1].set_xlabel('Time')
                        axes[0, 1].set_ylabel('Recovery Variable w', color='orange')
                        ax2_twin.set_ylabel('External Current I_ext', color='yellow')

                        # Combine legends for dual y-axis plot
                        lines1, labels1 = axes[0, 1].get_legend_handles_labels()
                        lines2, labels2 = ax2_twin.get_legend_handles_labels()
                        axes[0, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                        axes[0, 1].set_title("Recovery Variable w & External Stimulation")

                        # Plot 3: Phase portrait comparison
                        axes[1, 0].plot(v_true.cpu().numpy(), w_true.cpu().numpy(),
                                        label='True trajectory', linewidth=2, alpha=0.7, color='white')
                        axes[1, 0].plot(v_rollout, w_rollout,
                                        label='SIREN+MLP trajectory', linewidth=2, color='magenta')
                        axes[1, 0].set_xlabel('v (membrane potential)')
                        axes[1, 0].set_ylabel('w (recovery variable)')
                        axes[1, 0].legend()
                        axes[1, 0].set_title("Phase Portrait: v vs w")
                        axes[1, 0].grid(True, alpha=0.3)

                        # Plot 4: Training progress and SIREN w prediction
                        ax4_twin = axes[1, 1].twinx()
                        # Training loss (subsampled for clarity)
                        if len(train_losses) > 1000:
                            subsample = max(1, len(train_losses) // 1000)
                            epochs_sub = list(range(0, len(train_losses), subsample))
                            losses_sub = train_losses[::subsample]
                            line1 = ax4_twin.plot(epochs_sub, losses_sub, color='cyan', alpha=0.7,
                                                  label='Training Loss')
                            ax4_twin.set_ylabel('Training Loss', color='cyan')
                            ax4_twin.set_yscale('log')

                        # SIREN w prediction vs true w
                        w_siren_pred = model.siren(t_full).squeeze().cpu().numpy()
                        line2 = axes[1, 1].plot(time, w_true.cpu().numpy(), color='orange', linewidth=2,
                                                alpha=0.7, label='True w')
                        axes[1, 1].plot(time, w_siren_pred, color='red', linewidth=2,
                                        alpha=0.7, label='SIREN w')
                        axes[1, 1].set_xlabel('Time')
                        axes[1, 1].set_ylabel('Recovery Variable w', color='orange')
                        axes[1, 1].set_title("Training Progress & SIREN w Prediction")

                        # Combine legends
                        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
                        if len(train_losses) > 1000:
                            lines2, labels2 = ax4_twin.get_legend_handles_labels()
                            axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                        else:
                            axes[1, 1].legend()

                        plt.tight_layout()
                        plt.savefig('./tmp/comprehensive_siren_fitzhugh_nagumo.png', dpi=200, bbox_inches='tight')
                        plt.show()
                        plt.close()

                        # Calculate comprehensive metrics
                        mse_v = np.mean((v_rollout - v_true.cpu().numpy()) ** 2)
                        mae_v = np.mean(np.abs(v_rollout - v_true.cpu().numpy()))
                        mse_w = np.mean((w_rollout - w_true.cpu().numpy()) ** 2)
                        mae_w = np.mean(np.abs(w_rollout - w_true.cpu().numpy()))

                        # SIREN w prediction metrics
                        mse_w_siren = np.mean((w_siren_pred - w_true.cpu().numpy()) ** 2)
                        correlation_w = np.corrcoef(w_rollout, w_true.cpu().numpy())[0, 1]
                        correlation_w_siren = np.corrcoef(w_siren_pred, w_true.cpu().numpy())[0, 1]

                        # Phase portrait error
                        phase_error = np.mean(np.sqrt((v_rollout - v_true.cpu().numpy()) ** 2 +
                                                      (w_rollout - w_true.cpu().numpy()) ** 2))

                        print(f"\n{'=' * 60}")
                        print(f"COMPREHENSIVE EVALUATION METRICS - EPOCH {epoch}")
                        print(f"{'=' * 60}")
                        print(f"Training Loss:                 {loss.item():.6f}")
                        print(f"Rollout MSE (v):              {mse_v:.6f}")
                        print(f"Rollout MAE (v):              {mae_v:.6f}")
                        print(f"Rollout MSE (w):              {mse_w:.6f}")
                        print(f"Rollout MAE (w):              {mae_w:.6f}")
                        print(f"SIREN MSE (w):                {mse_w_siren:.6f}")
                        print(f"Rollout w Correlation:         {correlation_w:.4f}")
                        print(f"SIREN w Correlation:           {correlation_w_siren:.4f}")
                        print(f"Phase Portrait Error:          {phase_error:.6f}")
                        if len(train_losses) > 1:
                            print(f"Loss Reduction:                {train_losses[0] / train_losses[-1]:.2f}x")
                        print(f"{'=' * 60}")

                        # Save model and metrics
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'epoch': epoch,
                                    'loss': loss.item()},
                                   f'tmp/model.pt')

                        # Save comprehensive metrics
                        metrics = {
                            'epoch': epoch,
                            'training_loss': loss.item(),
                            'mse_v': mse_v,
                            'mae_v': mae_v,
                            'mse_w': mse_w,
                            'mae_w': mae_w,
                            'mse_w_siren': mse_w_siren,
                            'correlation_w_rollout': correlation_w,
                            'correlation_w_siren': correlation_w_siren,
                            'phase_error': phase_error,
                            'train_losses': train_losses
                        }
                        torch.save(metrics, f'tmp/siren_training_metrics.pt')

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

                        # Training loss subplot
                        ax = fig.add_subplot(236)
                        if len(train_losses) > 100:
                            subsample = max(1, len(train_losses) // 500)
                            epochs_sub = list(range(0, len(train_losses), subsample))
                            losses_sub = train_losses[::subsample]
                            plt.plot(epochs_sub, losses_sub, color='cyan', alpha=0.8)
                            plt.yscale('log')
                            plt.title('Training Loss Evolution', fontsize=12, color='white')
                            plt.xlabel('Training Steps', fontsize=12)
                            plt.ylabel('Loss', fontsize=12)
                            plt.grid(True, alpha=0.3)

                        plt.tight_layout()
                        plt.savefig('./tmp/enhanced_derivative_analysis.png', dpi=200, bbox_inches='tight')
                        plt.show()
                        plt.close()

    print("Training completed!")