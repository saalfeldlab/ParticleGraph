import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange

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

# Training loop
n_epochs = 30000
batch_size = 128
n_batches = X_v.shape[0]

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

    if epoch % 500 == 0:
        print(f"[{epoch}] loss: {loss.item():.6f}")

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


# Plot results
plt.style.use("dark_background")
plt.figure(figsize=(12, 5))
plt.plot(v_true.cpu().numpy(), label='True v', linewidth=2, alpha=0.5, color='white')
plt.plot(v_rollout, label='RNN rollout v', linewidth=2, color='cyan')
plt.xlabel('Time Step')
plt.ylabel('Membrane Potential v')
plt.legend()
plt.title("FitzHugh–Nagumo: RNN rollout vs. True")
plt.tight_layout()
plt.savefig('./tmp/rnn_fitzhug_rollout.png', dpi=170)
plt.show()


plt.figure(figsize=(12, 6))
for i in range(min(5, h_rollout.shape[1])):  # show first 5 dimensions
    plt.plot(h_rollout[:, i], label=f'hidden dim {i}')
plt.title('Learned hidden state dimensions (proxy for w)')
plt.xlabel('Time Step')
plt.ylabel('Hidden State Value')
plt.legend()
plt.tight_layout()
plt.savefig('./tmp/hidden_state_plot.png', dpi=170)
plt.show()





