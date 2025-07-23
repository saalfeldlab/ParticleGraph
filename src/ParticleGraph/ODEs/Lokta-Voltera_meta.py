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
        self.mlp0 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)
        self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

    def forward(self, x):
        return self.siren(x)


def train_volterra_model(noise_level=0.0, verbose=True, random_seed=None):
    """
    Train Volterra predator-prey model with SIREN and MLPs

    This function exactly replicates the original training logic to ensure
    identical results.

    Args:
        noise_level: Noise level for data generation
        verbose: Whether to print progress
        random_seed: Random seed for reproducibility

    Returns:
        dict: Contains reconstructed coefficients and MSE values
    """

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

    device = set_device('auto')

    # Actual Bakarji paper predator-prey coefficients from source code
    args = [1.0, 0.1, 1.5, 0.75]  # [a, b, c, d_factor]
    a = args[0]  # 1.0 - prey growth rate
    b = args[1]  # 0.1 - predation rate
    c = args[2]  # 1.5 - predator death rate
    d = args[1] * args[3]  # 0.1 * 0.75 = 0.075 - conversion efficiency

    # Simulation settings
    T = 60  # total time to see oscillations
    dt = 0.01  # time step
    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)

    # Initialize variables
    v = np.zeros(n_steps)  # x (prey/species 1)
    w = np.zeros(n_steps)  # y (predator/species 2)

    # Bakarji paper initial conditions from source code
    v[0] = 10.0  # z0_mean_sug[0] = 10
    w[0] = 5.0  # z0_mean_sug[1] = 5

    # Exact Bakarji paper equations from source code
    for i in range(n_steps - 1):
        dv = a * v[i] - b * v[i] * w[i]  # args[0]*z[0] - args[1]*z[0]*z[1]
        dw = -c * w[i] + d * v[i] * w[i]  # -args[2]*z[1] + args[1]*args[3]*z[0]*z[1]

        v[i + 1] = v[i] + dt * dv + noise_level * np.random.randn()
        w[i + 1] = w[i] + dt * dw + noise_level * np.random.randn()

    # Convert to tensors (exactly as original)
    v_true = torch.tensor(v, dtype=torch.float32, device=device)
    w_true = torch.tensor(w, dtype=torch.float32, device=device)

    n_steps = len(w_true)  # shape (10000, 1)
    w_true = torch.tensor(w_true, dtype=torch.float32, device=device)
    t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)

    # Training parameters (exactly as original)
    n_iter = 8000
    batch_size = n_steps // 2

    # Initialize model and optimizer (exactly as original)
    model = model_duo(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop (EXACTLY as original)
    Time.sleep(1)
    loss_list = []
    for iter in range(n_iter):
        # Use exact same indexing as original
        idx = torch.randint(1, n_steps - 8, (batch_size,))
        idx = torch.sort(torch.unique(idx)).values
        t_batch = t_full[idx.clone().detach()]
        idx = to_numpy(idx)

        w = model.siren(t_batch)
        v = v_true[idx, None].clone().detach()
        optimizer.zero_grad()

        # Hardcoded values exactly as in original
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

        # Add regularization penalties (exactly as original)
        l1_penalty = 0.0
        for param in model.mlp1.parameters():
            l1_penalty += torch.sum(torch.abs(param))

        # Final loss with regularization
        loss = accumulated_loss + l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if iter % 500 == 0 and verbose:
            print(f"iteration {iter + 1}/{n_iter}, loss: {loss.item():.6f}")

    Time.sleep(1)

    # Rollout analysis (exactly as original)
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

    # Extract coefficients using regression (exactly as original)
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

    # Extract coefficients using regression (exactly as original)
    t_full_ = torch.linspace(0, 0.5, n_steps // 2, device=device).unsqueeze(1)
    w_pred = model.siren(t_full_).squeeze()
    a_reg, b_reg, c_reg, d_reg = extract_coeffs_regression(model, device, v_true[0:n_steps // 2],
                                                           w_pred[0:n_steps // 2])

    # Return results
    results = {
        'coefficients': {
            'a': a_reg,
            'b': b_reg,
            'c': c_reg,
            'd': d_reg,
            'true_a': a,
            'true_b': b,
            'true_c': c,
            'true_d': d
        },
        'mse': {
            'v_mse': v_mse,
            'w_mse': w_mse,
            'total_mse': total_mse
        },
        'training': {
            'loss_list': loss_list,
            'final_loss': loss.item(),
            'iteration': iter
        },
        'trajectories': {
            'v_pred': v_list,
            'w_pred': w_list,
            'v_true': v_true,
            'w_true': w_true
        },
        'model': model  # Return trained model
    }

    if verbose:
        print(f"V MSE: {v_mse:.6f}, W MSE: {w_mse:.6f}, Total MSE: {total_mse:.6f}")
        print("\nREGRESSION COEFFICIENT EXTRACTION:")
        print("Ground Truth: a=1.0, b=0.1, c=1.5, d=0.075")
        print(f"Reconstructed: a={a_reg:.3f}, b={b_reg:.3f}, c={c_reg:.3f}, d={d_reg:.3f}")

    return results


def run_parameter_sweep():
    """
    Explore parameter space one parameter at a time
    """

    # Parameter ranges to explore
    param_ranges = {
        'noise_level': [0.0, 0.01, 0.05, 0.1, 0.2]
    }

    best_overall = {'total_mse': float('inf'), 'params': None, 'coeffs': None}
    all_results = []

    print("=== VOLTERRA PARAMETER SWEEP ===\n")

    # Sweep each parameter independently
    for param_name, param_values in param_ranges.items():
        print(f"\n--- Sweeping {param_name.upper()} ---")

        for param_value in param_values:
            print(f"\nTesting {param_name}={param_value}")
            print("Runs: ", end="")

            # Run 3 times for statistics
            run_results = []
            valid_runs = 0

            for run in range(3):
                print(f"{run + 1}...", end="", flush=True)

                # Call with exact parameter name
                if param_name == 'noise_level':
                    result = train_volterra_model(noise_level=param_value, verbose=False, random_seed=42 + run)
                else:
                    result = train_volterra_model(verbose=False, random_seed=42 + run)

                # Check for NaN values
                mse = result['mse']['total_mse']
                coeffs = result['coefficients']

                # Check if any key values are NaN
                nan_detected = (
                        np.isnan(mse) or
                        np.isnan(coeffs['a']) or np.isnan(coeffs['b']) or
                        np.isnan(coeffs['c']) or np.isnan(coeffs['d']) or
                        np.isnan(result['mse']['v_mse']) or np.isnan(result['mse']['w_mse'])
                )

                if not nan_detected:
                    run_results.append(result)
                    valid_runs += 1
                else:
                    print("NaN!", end="", flush=True)

            # Skip if no valid runs
            if valid_runs == 0:
                print(f" SKIPPED - All runs contained NaN")
                continue

            print(f" ({valid_runs} valid)", end="")

            # Calculate statistics from valid runs only
            mse_values = [r['mse']['total_mse'] for r in run_results]
            coeffs = [r['coefficients'] for r in run_results]

            mean_mse = np.mean(mse_values)
            std_mse = np.std(mse_values)
            best_run_idx = np.argmin(mse_values)
            best_run = run_results[best_run_idx]

            # Store results
            result_summary = {
                'param_name': param_name,
                'param_value': param_value,
                'mean_mse': mean_mse,
                'std_mse': std_mse,
                'min_mse': min(mse_values),
                'best_coeffs': best_run['coefficients'],
                'best_v_mse': best_run['mse']['v_mse'],
                'best_w_mse': best_run['mse']['w_mse']
            }
            all_results.append(result_summary)

            # Update best overall
            if result_summary['min_mse'] < best_overall['total_mse']:
                best_overall = {
                    'total_mse': result_summary['min_mse'],
                    'params': {param_name: param_value},
                    'coeffs': result_summary['best_coeffs'],
                    'v_mse': result_summary['best_v_mse'],
                    'w_mse': result_summary['best_w_mse']
                }

            print(f" MSE: {mean_mse:.4f}±{std_mse:.4f} (best: {min(mse_values):.4f})")

            # Print best coefficients for this parameter set
            c = result_summary['best_coeffs']
            print(f"  Best coeffs: a={c['a']:.3f}, b={c['b']:.3f}, c={c['c']:.3f}, d={c['d']:.3f}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("BEST RESULTS OVERALL:")
    print(f"{'=' * 60}")
    print(f"Parameters: {best_overall['params']}")
    print(f"Total MSE: {best_overall['total_mse']:.6f}")
    print(f"V MSE: {best_overall['v_mse']:.6f}")
    print(f"W MSE: {best_overall['w_mse']:.6f}")

    c = best_overall['coeffs']
    print(f"Ground truth: a=1.0, b=0.1, c=1.5, d=0.075")
    print(f"Reconstructed: a={c['a']:.3f}, b={c['b']:.3f}, c={c['c']:.3f}, d={c['d']:.3f}")

    # Coefficient errors
    errors = {
        'a_error': abs(c['a'] - 1.0),
        'b_error': abs(c['b'] - 0.1),
        'c_error': abs(c['c'] - 1.5),
        'd_error': abs(c['d'] - 0.075)
    }
    print(
        f"Coeff errors: a={errors['a_error']:.3f}, b={errors['b_error']:.3f}, c={errors['c_error']:.3f}, d={errors['d_error']:.3f}")

    return best_overall, all_results


if __name__ == '__main__':
    # Test single run first with fixed seed
    print("Testing single run with fixed seed...")
    result = train_volterra_model(noise_level=0.0, verbose=True, random_seed=42)

    print("\nTesting single run with different seed...")
    result2 = train_volterra_model(noise_level=0.0, verbose=True, random_seed=123)

    # Then run parameter sweep
    print("\n" + "=" * 60)
    print("Starting parameter sweep...")
    best_result, all_results = run_parameter_sweep()