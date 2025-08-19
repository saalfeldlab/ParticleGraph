import os
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from config_Fitzhug_Nagumo import FitzhughNagumoConfig
from utils import setup_experiment_folders
import torch.nn as nn


# Import model classes (assuming they're in the same directory)
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
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)


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
        self.activation = F.tanh if activation == 'tanh' else F.relu

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    config_root = "./config"
    # config_file_list = ['recur_6', 'recur_7', 'recur_8', 'recur_9', 'recur_10', 'recur_11', 'recur_12']
    # config_file_list = ['recur_6_sin', 'recur_7_sin', 'recur_8_sin', 'recur_9_sin', 'recur_10_sin', 'recur_11_sin', 'recur_12_sin']
    config_file_list = ['recur_7', 'recur_7_1', 'recur_7_2', 'recur_7_3', 'recur_7_4', 'recur_7_5']

    plt.style.use('dark_background')

    for config_file in config_file_list:
        print(f"\n=== Analyzing {config_file} ===")

        # Load config
        config = FitzhughNagumoConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        experiment_name = f"fitzhugh_nagumo_{config_file}"
        folders = setup_experiment_folders(experiment_name=experiment_name)

        # Simulation parameters
        T, dt = config.simulation.T, config.simulation.dt
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps)

        # Generate true data
        v_init, w_init = config.simulation.v_init, config.simulation.w_init
        pulse_interval = config.simulation.pulse_interval
        pulse_duration = config.simulation.pulse_duration
        pulse_amplitude = config.simulation.pulse_amplitude
        noise_level = config.simulation.noise_level
        a, b, epsilon = config.system.a, config.system.b, config.system.epsilon

        # Generate FitzHugh-Nagumo data
        v = np.zeros(n_steps)
        w = np.zeros(n_steps)
        I_ext = np.zeros(n_steps)
        v[0], w[0] = v_init, w_init

        n_pulses = int(np.ceil(T / pulse_interval))
        pulse_ids = (time // pulse_interval).astype(int)
        phase_rnd = np.zeros(n_pulses)
        amplitude_rnd = np.full(n_pulses, pulse_amplitude)
        mask = (time % pulse_interval) < pulse_duration
        wave = 1.0 + 0.25 * np.sin(
            2 * np.pi * (time[mask] % pulse_interval) / pulse_interval + phase_rnd[pulse_ids[mask]])
        I_ext[mask] = amplitude_rnd[pulse_ids[mask]] * wave

        for i in range(n_steps - 1):
            dv = v[i] - (v[i] ** 3) / 3 - w[i] + I_ext[i]
            dw = epsilon * (v[i] + a - b * w[i])
            v[i + 1] = v[i] + dt * dv + noise_level * np.random.randn()
            w[i + 1] = w[i] + dt * dw + noise_level * np.random.randn()

        v_true = torch.tensor(v, dtype=torch.float32, device=device)
        w_true = torch.tensor(w, dtype=torch.float32, device=device)
        I_ext = torch.tensor(I_ext, dtype=torch.float32, device=device)
        t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)

        # Load all models and run rollouts
        test_runs = config.training.test_runs
        v_rollouts, w_rollouts, w_corrected_list = [], [], []
        total_mses = []

        for run in range(test_runs):
            model_path = os.path.join(folders['models'], f'model_run_{run + 1}.pt')
            if not os.path.exists(model_path):
                print(f"Model {run + 1} not found, skipping...")
                continue

            # Load model
            model = model_duo(device=device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Rollout
            with torch.no_grad():
                v_roll = v_true[0:1].clone().detach().squeeze()
                w_roll = w_true[0:1].clone().detach().squeeze()
                v_list, w_list = [v_roll.clone()], [w_roll.clone()]

                n_siren_bootstrap = 500
                for step in range(1, n_steps):
                    if step < n_siren_bootstrap:
                        w_roll = model.siren(t_full[step]).squeeze()

                    v_input = v_roll.unsqueeze(0).unsqueeze(1)
                    w_input = w_roll.unsqueeze(0).unsqueeze(1)
                    I_input = I_ext[step].unsqueeze(0).unsqueeze(1)

                    dv_pred = model.mlp0(torch.cat((v_input, w_input, I_input), dim=1))
                    dw_pred = model.mlp1(torch.cat((v_input, w_input), dim=1))

                    v_roll = v_roll + dt * dv_pred.squeeze()
                    if step >= n_siren_bootstrap:
                        w_roll = w_roll + dt * dw_pred.squeeze()

                    v_list.append(v_roll.clone())
                    w_list.append(w_roll.clone())

                v_rollout = torch.stack(v_list, dim=0).cpu().numpy()
                w_rollout = torch.stack(w_list, dim=0).cpu().numpy()

                # Affine correction for W
                w_pred = w_rollout.squeeze()
                w_target = w_true.cpu().numpy().squeeze()
                if np.var(w_pred) < 1e-12:
                    a_coeff, b_coeff = 1.0, 0.0
                else:
                    a_coeff = np.cov(w_pred, w_target, bias=True)[0, 1] / np.var(w_pred)
                    b_coeff = np.mean(w_target) - a_coeff * np.mean(w_pred)
                w_corrected = a_coeff * w_pred + b_coeff

                v_rollouts.append(v_rollout.squeeze())
                w_rollouts.append(w_rollout.squeeze())
                w_corrected_list.append(w_corrected)

                # Calculate total MSE for outlier removal
                v_mse = np.mean((v_rollout[500:].squeeze() - v_true[500:].cpu().numpy()) ** 2)
                w_mse = np.mean((w_corrected[500:] - w_target[500:]) ** 2)
                total_mses.append(v_mse + w_mse)

        # Remove outliers (bottom 10%)
        n_runs = len(total_mses)
        n_outliers = max(1, int(0.1 * n_runs))
        outlier_indices = np.argsort(total_mses)[-n_outliers:]

        v_rollouts_clean = [v_rollouts[i] for i in range(n_runs) if i not in outlier_indices]
        w_corrected_clean = [w_corrected_list[i] for i in range(n_runs) if i not in outlier_indices]

        print(f"Removed {n_outliers} outliers from {n_runs} runs")

        # Calculate medians
        v_median = np.median(np.array(v_rollouts_clean), axis=0)
        w_median = np.median(np.array(w_corrected_clean), axis=0)

        # Calculate MSEs against median
        v_mses = [np.mean((v_roll[500:] - v_median[500:]) ** 2) for v_roll in v_rollouts_clean]
        w_mses = [np.mean((w_corr[500:] - w_median[500:]) ** 2) for w_corr in w_corrected_clean]

        print(f"V MSE vs median: {np.mean(v_mses):.6f} ± {np.std(v_mses):.6f}")
        print(f"W MSE vs median: {np.mean(w_mses):.6f} ± {np.std(w_mses):.6f}")

        # Calculate MSE between true W and median W
        w_true_vs_median_mse = np.mean((w_true.cpu().numpy()[500:] - w_median[500:]) ** 2)
        print(f"True W vs Median W MSE: {w_true_vs_median_mse:.6f}")

        # Plot
        fig = plt.figure(figsize=(16, 12))

        # V plot
        plt.subplot(3, 1, 1)
        plt.plot(v_true.cpu().numpy(), label='true v', linewidth=3, alpha=0.9, c='white')
        for i, v_roll in enumerate(v_rollouts_clean):
            plt.plot(v_roll, linewidth=1, alpha=0.6, c='green', label='rollout v' if i == 0 else '')
        plt.plot(I_ext.cpu().numpy(), label='I_ext', linewidth=2, alpha=0.5, c='red')
        plt.xlim([0, n_steps // 2.5])
        plt.xlabel('time')
        plt.ylabel('v')
        plt.ylim([-2.5, 2.5])
        plt.legend(loc='upper left', fontsize=12)
        plt.title(f'V dynamics - {config_file} (MSE vs median: {np.mean(v_mses):.4f})')
        plt.grid(True, alpha=0.3)

        # W plot
        plt.subplot(3, 1, 2)
        plt.plot(w_true.cpu().numpy(), label='true w', linewidth=3, alpha=0.9, c='white')
        for i, w_corr in enumerate(w_corrected_clean):
            plt.plot(w_corr, linewidth=1.5, alpha=0.7, c='orange', label='corrected w (affine)' if i == 0 else '')
        plt.xlim([0, n_steps // 2.5])
        plt.xlabel('time')
        plt.ylabel('w')
        plt.ylim([-1, 2.5])
        plt.legend(loc='upper left', fontsize=12)
        plt.title(f'W dynamics - {config_file} (MSE vs median: {np.mean(w_mses):.4f} ± {np.std(w_mses):.4f})')
        plt.grid(True, alpha=0.3)

        # True W vs Median W comparison
        plt.subplot(3, 1, 3)
        plt.plot(w_true.cpu().numpy(), label='true w', linewidth=6, alpha=0.5, c='white')
        plt.plot(w_median, label='median w', linewidth=2, alpha=1.0, c='orange')
        plt.xlim([0, n_steps // 2.5])
        plt.xlabel('time')
        plt.ylabel('w')
        plt.ylim([-1, 2.5])
        plt.legend(loc='upper left', fontsize=12)
        plt.title(f'True W vs Median W - {config_file} (MSE: {w_true_vs_median_mse:.4f})')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(folders['plots'], f'W_analysis_{config_file}.png')
        plt.savefig(output_path, dpi=170, facecolor='black', bbox_inches='tight')
        plt.show()
        print(f"Saved plot to {output_path}")