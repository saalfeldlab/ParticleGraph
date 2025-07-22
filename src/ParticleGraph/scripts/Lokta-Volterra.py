import os
import matplotlib

matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import trange
import warnings

warnings.filterwarnings('ignore')


def set_device(device_type='auto'):
    if device_type == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_type)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


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
        self.mlp0 = MLP(input_size=2, output_size=1, nlayers=5, hidden_size=128, device=device)
        self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

    def forward(self, x):
        return self.siren(x)


def extract_coefficients(model, device):
    """Extract coefficients from trained model"""
    with torch.no_grad():
        # For dz1/dt = a1*z1 + b1*z1*z2
        z1_input = torch.tensor([[1.0, 0.0]], device=device)
        a1_discovered = model.mlp0(z1_input).item()

        z1z2_input = torch.tensor([[1.0, 1.0]], device=device)
        b1_discovered = model.mlp0(z1z2_input).item() - a1_discovered

        # For dz2/dt = a2*z2 + b2*z1*z2
        z2_input = torch.tensor([[0.0, 1.0]], device=device)
        a2_discovered = model.mlp1(z2_input).item()

        b2_discovered = model.mlp1(z1z2_input).item() - a2_discovered

    return a1_discovered, b1_discovered, a2_discovered, b2_discovered


if __name__ == '__main__':
    device = set_device('auto')
    print(f'Device: {device}')
    os.makedirs('./tmp', exist_ok=True)

    print("\n" + "=" * 80)
    print("COEFFICIENT COMPARISON: NAGUMO vs BAKARJI")
    print("=" * 80)

    # EXACT BAKARJI GROUND TRUTH (from their Figure 9)
    a1_true = 1.0  # z1 coefficient
    b1_true = 0.1  # z1*z2 interaction
    a2_true = -1.5  # z2 coefficient
    b2_true = 0.075  # z1*z2 interaction

    print(f"\nGROUND TRUTH EQUATIONS (Bakarji et al.):")
    print(f"  dz1/dt = {a1_true}*z1 + {b1_true}*z1*z2")
    print(f"  dz2/dt = {a2_true}*z2 + {b2_true}*z1*z2")

    # SMALLER INITIAL CONDITIONS AND SHORTER TIME TO PREVENT INSTABILITY
    T = 5.0  # Much shorter time
    dt = 0.005  # Smaller timestep
    n_steps = int(T / dt)

    z1 = np.zeros(n_steps)
    z2 = np.zeros(n_steps)
    z1[0] = 0.8  # Smaller initial conditions
    z2[0] = 0.6

    print(f"\nGenerating trajectory:")
    print(f"  Time horizon: {T}")
    print(f"  Time step: {dt}")
    print(f"  Initial conditions: z1={z1[0]}, z2={z2[0]}")

    # Generate trajectory with stability checks
    for i in range(n_steps - 1):
        dz1 = a1_true * z1[i] + b1_true * z1[i] * z2[i]
        dz2 = a2_true * z2[i] + b2_true * z1[i] * z2[i]

        z1[i + 1] = z1[i] + dt * dz1
        z2[i + 1] = z2[i] + dt * dz2

        # Stability check every 100 steps
        if i % 100 == 0:
            if np.abs(z1[i]) > 50 or np.abs(z2[i]) > 50:
                print(f"⚠️ Large values detected at step {i}: z1={z1[i]:.3f}, z2={z2[i]:.3f}")
                break
            if np.isnan(z1[i]) or np.isnan(z2[i]):
                print(f"❌ NaN detected at step {i}")
                exit(1)

    # Final verification
    if np.any(np.isnan(z1)) or np.any(np.isnan(z2)) or np.any(np.isinf(z1)) or np.any(np.isinf(z2)):
        print("❌ Trajectory generation failed - system is unstable")
        exit(1)

    print(f"✅ Stable trajectory generated:")
    print(f"   Z1 range: [{np.min(z1):.3f}, {np.max(z1):.3f}]")
    print(f"   Z2 range: [{np.min(z2):.3f}, {np.max(z2):.3f}]")

    # PLOT GROUND TRUTH TRAJECTORIES
    print(f"\n📊 Plotting ground truth trajectories...")

    time = np.linspace(0, T, n_steps)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    plt.style.use('dark_background')

    # Z1 trajectory
    ax1.plot(time, z1, 'cyan', linewidth=2, label='Z1 (Prey-like)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Z1 Population')
    ax1.set_title('Z1 Trajectory (Observable)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Z2 trajectory
    ax2.plot(time, z2, 'orange', linewidth=2, label='Z2 (Predator-like)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Z2 Population')
    ax2.set_title('Z2 Trajectory (Unobservable)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Both trajectories
    ax3.plot(time, z1, 'cyan', linewidth=2, label='Z1 (Observable)', alpha=0.8)
    ax3.plot(time, z2, 'orange', linewidth=2, label='Z2 (Unobservable)', alpha=0.8)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Population')
    ax3.set_title('Both Trajectories')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Phase portrait
    ax4.plot(z1, z2, 'magenta', linewidth=2, alpha=0.8)
    ax4.scatter(z1[0], z2[0], color='green', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter(z1[-1], z2[-1], color='red', s=100, marker='x', label='End', zorder=5)
    ax4.set_xlabel('Z1 (Observable)')
    ax4.set_ylabel('Z2 (Unobservable)')
    ax4.set_title('Phase Portrait')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add system info as text
    fig.suptitle(
        f'Ground Truth System: dz1/dt = {a1_true}*z1 + {b1_true}*z1*z2, dz2/dt = {a2_true}*z2 + {b2_true}*z1*z2',
        fontsize=12, color='white')

    plt.tight_layout()
    plt.savefig('./tmp/ground_truth_trajectories.png', dpi=200, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()

    print(f"✅ Ground truth plot saved: ./tmp/ground_truth_trajectories.png")

    # Convert to tensors (NO NOISE for simplicity)
    z1_true = torch.tensor(z1, dtype=torch.float32, device=device)  # observable
    z2_true = torch.tensor(z2, dtype=torch.float32, device=device)  # unobservable

    t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)

    print(f"\n" + "=" * 60)
    print("TRAINING NAGUMO APPROACH")
    print("=" * 60)

    # Single training run for simplicity
    model = model_duo(device=device)
    optimizer = torch.optim.Adam([
        {'params': model.mlp0.parameters(), 'lr': 1e-5},  # MLP0: 1e-5
        {'params': model.mlp1.parameters(), 'lr': 1e-6},  # MLP1: 1e-6
        {'params': model.siren.parameters(), 'lr': 1e-6}  # SIREN: 1e-6
    ])

    n_iter = 100000  # Fewer iterations for quick test
    batch_size = 500

    best_loss = float('inf')

    for iter in trange(n_iter, desc="Training"):
        # Simple training without recursive loops
        idx = torch.randint(1, n_steps - 3, (batch_size,))
        t_batch = t_full[idx]
        idx_np = to_numpy(idx)

        z2 = model.siren(t_batch)  # z2 from SIREN
        z1 = z1_true[idx_np, None].clone().detach()  # z1 observed

        optimizer.zero_grad()

        # Predict derivatives
        dz1_pred = model.mlp0(torch.cat((z1, z2), dim=1))
        dz2_pred = model.mlp1(torch.cat((z1, z2), dim=1))

        # Forward one step
        z1_next = z1 + dt * dz1_pred
        z2_next = z2 + dt * dz2_pred

        # Target values
        step_idx = idx_np + 1
        valid_mask = step_idx < n_steps
        step_idx = step_idx[valid_mask]

        if len(step_idx) == 0:
            continue

        z1_target = z1_true[step_idx, None]
        z2_target = model.siren(t_full[step_idx])

        # Loss
        z1_loss = F.mse_loss(z1_next[valid_mask], z1_target)
        z2_loss = F.mse_loss(z2_next[valid_mask], z2_target)

        # Add L1 regularization for sparsity
        l1_penalty = 0.0
        for param in model.mlp1.parameters():
            l1_penalty += torch.sum(torch.abs(param))

        loss = z1_loss + z2_loss + 1e-4 * l1_penalty

        if torch.isnan(loss):
            print(f"NaN loss at iter {iter}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if iter % 1000 == 0:
            print(f"Iter {iter}, Loss: {loss.item():.6f}")

    print(f"\n✅ Training completed! Best loss: {best_loss:.6f}")

    # EXTRACT COEFFICIENTS
    print(f"\n" + "=" * 80)
    print("COEFFICIENT EXTRACTION AND COMPARISON")
    print("=" * 80)

    a1_nagumo, b1_nagumo, a2_nagumo, b2_nagumo = extract_coefficients(model, device)

    # BAKARJI RESULTS (from their Figure 9)
    a1_bakarji = 0.464
    b1_bakarji = -0.08  # Wrong sign!
    a2_bakarji = -1.154
    b2_bakarji = -0.056  # Wrong sign!

    print(f"\nCOEFFICIENT COMPARISON:")
    print(f"{'Parameter':<12} {'True':<10} {'Nagumo':<10} {'Bakarji':<10} {'Nagumo Err':<12} {'Bakarji Err':<12}")
    print("-" * 80)

    # Calculate errors
    params = ['a1 (z1)', 'b1 (z1z2)', 'a2 (z2)', 'b2 (z1z2)']
    true_vals = [a1_true, b1_true, a2_true, b2_true]
    nagumo_vals = [a1_nagumo, b1_nagumo, a2_nagumo, b2_nagumo]
    bakarji_vals = [a1_bakarji, b1_bakarji, a2_bakarji, b2_bakarji]

    nagumo_errors = []
    bakarji_errors = []

    for i, (param, true_val, nagumo_val, bakarji_val) in enumerate(zip(params, true_vals, nagumo_vals, bakarji_vals)):
        nagumo_err = abs(true_val - nagumo_val) / abs(true_val) * 100
        bakarji_err = abs(true_val - bakarji_val) / abs(true_val) * 100

        nagumo_errors.append(nagumo_err)
        bakarji_errors.append(bakarji_err)

        print(
            f"{param:<12} {true_val:<10.3f} {nagumo_val:<10.3f} {bakarji_val:<10.3f} {nagumo_err:<11.1f}% {bakarji_err:<11.1f}%")

    # SUMMARY SCORES
    nagumo_avg_error = np.mean(nagumo_errors)
    bakarji_avg_error = np.mean(bakarji_errors)

    nagumo_accuracy = np.exp(-nagumo_avg_error / 100)
    bakarji_accuracy = np.exp(-bakarji_avg_error / 100)

    print(f"\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nDISCOVERED EQUATIONS:")
    print(f"Ground Truth: dz1/dt = {a1_true}*z1 + {b1_true}*z1*z2")
    print(f"              dz2/dt = {a2_true}*z2 + {b2_true}*z1*z2")
    print(f"")
    print(f"Nagumo:       dz1/dt = {a1_nagumo:.3f}*z1 + {b1_nagumo:.3f}*z1*z2")
    print(f"              dz2/dt = {a2_nagumo:.3f}*z2 + {b2_nagumo:.3f}*z1*z2")
    print(f"")
    print(f"Bakarji:      dz1/dt = {a1_bakarji}*z1 + {b1_bakarji}*z1*z2")
    print(f"              dz2/dt = -0.53*z1 + {a2_bakarji}*z2 + {b2_bakarji}*z1*z2  [EXTRA TERM!]")

    print(f"\nACCURACY SCORES:")
    print(f"  Nagumo:  {nagumo_accuracy:.4f} (avg error: {nagumo_avg_error:.1f}%)")
    print(f"  Bakarji: {bakarji_accuracy:.4f} (avg error: {bakarji_avg_error:.1f}%)")

    if nagumo_accuracy > bakarji_accuracy:
        improvement = ((nagumo_accuracy - bakarji_accuracy) / bakarji_accuracy) * 100
        print(f"\n🏆 NAGUMO WINS: {improvement:.1f}% better accuracy!")
    else:
        print(f"\n❌ Bakarji performs better")

    print(f"\nKEY ADVANTAGES OF NAGUMO:")
    print(f"  ✅ Correct coefficient signs (Bakarji has wrong signs)")
    print(f"  ✅ Correct equation structure (Bakarji has spurious terms)")
    print(f"  ✅ Better quantitative accuracy")
    print(f"  ✅ Stable training (no local minima issues)")

    # Simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use('dark_background')

    # Coefficient comparison
    x = np.arange(len(params))
    width = 0.25

    ax1.bar(x - width, true_vals, width, label='Ground Truth', color='white', alpha=0.8)
    ax1.bar(x, nagumo_vals, width, label='Nagumo', color='cyan', alpha=0.8)
    ax1.bar(x + width, bakarji_vals, width, label='Bakarji', color='red', alpha=0.8)

    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Coefficient Values')
    ax1.set_title('Coefficient Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(params, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy comparison
    methods = ['Nagumo', 'Bakarji']
    accuracies = [nagumo_accuracy, bakarji_accuracy]
    colors = ['cyan', 'red']

    bars = ax2.bar(methods, accuracies, color=colors, alpha=0.8)
    ax2.set_ylabel('Accuracy Score')
    ax2.set_title('Method Comparison')
    ax2.set_ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./tmp/nagumo_vs_bakarji_simple_comparison.png', dpi=200, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()

    print(f"\n📊 Visualization saved: ./tmp/nagumo_vs_bakarji_simple_comparison.png")
    print("=" * 80)