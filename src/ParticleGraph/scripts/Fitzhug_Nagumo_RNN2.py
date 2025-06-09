import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# FitzHugh–Nagumo parameters
a = 0.7
b = 0.8
epsilon = 0.18

T = 1000
dt = 0.1
n_steps = int(T / dt)
time = np.linspace(0, T, n_steps)

# Initialize variables
v = np.zeros(n_steps)
w = np.zeros(n_steps)
I_ext = np.zeros(n_steps)

# Initial conditions
v[0] = -1.0
w[0] = 1.0

# External stimulation
pulse_interval = 80.0
pulse_duration = 1.0
pulse_amplitude = 0.8

for i, t in enumerate(time):
    if (t % pulse_interval) < pulse_duration:
        I_ext[i] = pulse_amplitude

# Euler integration
for i in range(n_steps - 1):
    dv = v[i] - (v[i]**3) / 3 - w[i] + I_ext[i]
    dw = epsilon * (v[i] + a - b * w[i])
    v[i + 1] = v[i] + dt * dv
    w[i + 1] = w[i] + dt * dw

# Convert to tensors
v_true = torch.tensor(v, dtype=torch.float32, device=device)
w_true = torch.tensor(w, dtype=torch.float32, device=device)
I_ext = torch.tensor(I_ext, dtype=torch.float32, device=device)

# Prepare training data
seq_len = 10
X_v, X_I, Y_v = [], [], []
for i in range(n_steps - seq_len - 1):
    X_v.append(v_true[i:i + seq_len])
    X_I.append(I_ext[i:i + seq_len])
    Y_v.append(v_true[i + 1:i + seq_len + 1])  # next v

X_v = torch.stack(X_v)[:, :, None]
X_I = torch.stack(X_I)[:, :, None]
Y_v = torch.stack(Y_v)[:, :, None]

# RNN model
class RNNModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.dv_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, v_seq, I_seq, h0=None):
        x = torch.cat([v_seq, I_seq], dim=2)
        h_seq, hn = self.rnn(x, h0)
        dv_input = torch.cat([h_seq, I_seq], dim=2)
        dv_dt = self.dv_mlp(dv_input)
        v_pred = v_seq + dt * dv_dt
        return v_pred, hn

# Initialize model
model = RNNModel(hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

# Create tmp directory
os.makedirs('tmp', exist_ok=True)

# Training loop with loss tracking
n_epochs = 30000
batch_size = 128
n_batches = X_v.shape[0]

train_losses = []
print("Starting training...")

for epoch in trange(n_epochs):
    idx = torch.randint(0, n_batches, (batch_size,))
    v_batch = X_v[idx].to(device)
    I_batch = X_I[idx].to(device)
    v_target = Y_v[idx].to(device)

    v_pred, _ = model(v_batch, I_batch)
    loss = loss_fn(v_pred, v_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if epoch % 500 == 0:
        print(f"[{epoch}] loss: {loss.item():.6f}")

# Save model
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_loss': loss.item()},
           f'tmp/model_RNN.pt')

print(f"Training completed. Final loss: {loss.item():.6f}")

# Load model for rollout (your original approach)
state_dict = torch.load(f'tmp/model_RNN.pt', map_location=device)
model.load_state_dict(state_dict['model_state_dict'])

# Rollout with model and extract hidden state
with torch.no_grad():
    v = v_true[0].view(1, 1, 1)
    h = None
    v_rollout = [v.item()]
    h_rollout = []

    for t in range(n_steps - 1):
        I_t = I_ext[t].view(1, 1, 1)
        v, h = model(v, I_t, h)
        v_rollout.append(v.item())
        h_rollout.append(h.squeeze().cpu().numpy())

    v_rollout = np.array(v_rollout)
    h_rollout = np.stack(h_rollout, axis=0)  # (T, hidden_dim)

# Enhanced plotting with multiple subplots
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: v comparison (your original but enhanced)
axes[0,0].plot(time, v_true.cpu().numpy(), label='True v',
               linewidth=2, alpha=0.7, color='white')
axes[0,0].plot(time, v_rollout, label='RNN rollout v',
               linewidth=2, color='cyan')
axes[0,0].set_ylim([-2, 2])
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('Membrane Potential v')
axes[0,0].legend()
axes[0,0].set_title("Membrane Potential: RNN vs True")
axes[0,0].grid(True, alpha=0.3)

# Plot 2: True hidden variable w and external current
ax2_twin = axes[0,1].twinx()
line1 = axes[0,1].plot(time, w_true.cpu().numpy(), label='True w (hidden)',
                       linewidth=2, color='orange', alpha=0.8)
line2 = ax2_twin.plot(time, I_ext.cpu().numpy(), label='External Current I_ext',
                      linewidth=2, color='yellow', alpha=0.6)
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel('Recovery Variable w', color='orange')
ax2_twin.set_ylabel('External Current I_ext', color='yellow')

# Combine legends for dual y-axis plot
lines1, labels1 = axes[0,1].get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
axes[0,1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
axes[0,1].set_title("True Hidden Variable w & External Stimulation")

# Plot 3: Phase portrait comparison
axes[1,0].plot(v_true.cpu().numpy(), w_true.cpu().numpy(),
               label='True trajectory', linewidth=2, alpha=0.7, color='white')
# Use first hidden dimension as w proxy for phase plot
w_proxy = h_rollout[:, 0]
# Fix dimension mismatch: v_rollout has n_steps, h_rollout has n_steps-1
axes[1,0].plot(v_rollout[1:], w_proxy,
               label='RNN trajectory (hidden state proxy)', linewidth=2, color='magenta')
axes[1,0].set_xlabel('v (membrane potential)')
axes[1,0].set_ylabel('w / hidden state')
axes[1,0].legend()
axes[1,0].set_title("Phase Portrait: v vs w")
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Training progress and hidden state analysis
ax4_twin = axes[1,1].twinx()
# Training loss (subsampled for clarity)
subsample = max(1, len(train_losses) // 1000)
epochs_sub = list(range(0, len(train_losses), subsample))
losses_sub = train_losses[::subsample]
line1 = ax4_twin.plot(epochs_sub, losses_sub, color='cyan', alpha=0.7, label='Training Loss')
ax4_twin.set_ylabel('Training Loss', color='cyan')
ax4_twin.set_yscale('log')

# Hidden state variance over time
hidden_variance = np.var(h_rollout, axis=1)
line2 = axes[1,1].plot(time[1:], hidden_variance, color='orange', linewidth=2, label='Hidden State Variance')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Hidden State Variance', color='orange')
axes[1,1].set_title("Training Progress & Hidden Dynamics")

# Combine legends
lines1, labels1 = axes[1,1].get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
axes[1,1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('./tmp/comprehensive_fitzhugh_nagumo_analysis.png', dpi=200, bbox_inches='tight')
plt.show()

# Additional detailed hidden state plot (your original but enhanced)
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
for i in range(min(5, h_rollout.shape[1])):
    plt.plot(time[1:], h_rollout[:, i], label=f'hidden dim {i}', alpha=0.8)
plt.title('RNN Hidden State Dimensions (Proxy for w dynamics)')
plt.xlabel('Time')
plt.ylabel('Hidden State Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
# Correlation analysis between hidden dimensions and true w
correlations = []
for i in range(h_rollout.shape[1]):
    corr = np.corrcoef(h_rollout[:, i], w_true[1:].cpu().numpy())[0,1]
    if np.isnan(corr):  # Handle NaN correlations
        corr = 0.0
    correlations.append(corr)

plt.bar(range(len(correlations)), correlations, alpha=0.7, color='orange')
plt.title('Correlation between Hidden Dimensions and True w')
plt.xlabel('Hidden Dimension Index')
plt.ylabel('Correlation with True w')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./tmp/detailed_hidden_state_analysis.png', dpi=200, bbox_inches='tight')
plt.show()

# Calculate comprehensive metrics
mse_v = np.mean((v_rollout[1:] - v_true[1:].cpu().numpy())**2)
mae_v = np.mean(np.abs(v_rollout[1:] - v_true[1:].cpu().numpy()))

# Find best hidden dimension correlation with w
best_corr_idx = np.argmax(np.abs(correlations))
best_correlation = correlations[best_corr_idx]

# Compute phase portrait error (distance between trajectories)
phase_error = np.mean(np.sqrt((v_rollout[1:] - v_true[1:].cpu().numpy())**2 +
                             (h_rollout[:, best_corr_idx] - w_true[1:].cpu().numpy())**2))

print(f"\n{'='*50}")
print(f"COMPREHENSIVE EVALUATION METRICS")
print(f"{'='*50}")
print(f"Final Training Loss:           {state_dict['final_loss']:.6f}")
print(f"Rollout MSE (v):              {mse_v:.6f}")
print(f"Rollout MAE (v):              {mae_v:.6f}")
print(f"Best Hidden-w Correlation:     {best_correlation:.4f} (dim {best_corr_idx})")
print(f"Phase Portrait Error:          {phase_error:.6f}")
print(f"Training Epochs Completed:     {n_epochs}")
print(f"Final Loss Reduction:          {train_losses[0]/train_losses[-1]:.2f}x")
print(f"{'='*50}")

# Save metrics
metrics = {
    'final_loss': state_dict['final_loss'],
    'mse_v': mse_v,
    'mae_v': mae_v,
    'best_correlation': best_correlation,
    'phase_error': phase_error,
    'train_losses': train_losses
}
torch.save(metrics, 'tmp/training_metrics.pt')
print("Metrics saved to tmp/training_metrics.pt")