import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

plt.style.use('dark_background')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dt = 0.1

# FitzHugh-Nagumo params (for generating synthetic data)
a, b, epsilon = 0.7, 0.8, 0.18
T = 1000
time = torch.linspace(0, T*dt, T).to(device)

T = 1000        # total time in seconds
dt = 0.1        # timestep
n_steps = int(T / dt)  # 10000

time = torch.linspace(0, T, n_steps).to(device)

# Generate synthetic data with true FitzHugh-Nagumo ODEs
def fhn_ode(t, y):
    v, w = y[..., 0], y[..., 1]
    I_ext = 0.8 * ((t % 60) < 1).float()
    dvdt = v - v**3 / 3 - w + I_ext
    dwdt = epsilon * (v + a - b * w)
    return torch.stack([dvdt, dwdt], dim=-1)

with torch.no_grad():
    y0 = torch.tensor([-1.0, 1.0]).to(device)
    true_y = odeint(fhn_ode, y0, time)

v_true = true_y[..., 0].unsqueeze(-1)  # shape [T,1]

# plot v_true

plt.figure(figsize=(12, 5))
plt.plot(v_true.cpu(), label='True v')
plt.show()


# Neural ODE function to learn f, g with NN
class FHNNeuralODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t, y):
        # y shape: [batch, 2]
        return self.net(y)

# Model wrapping the Neural ODE and initial latent state w0
class FHNNeuralODEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.odefunc = FHNNeuralODEFunc()
        # Learn initial latent w0, v0 fixed from data start point
        self.w0 = nn.Parameter(torch.tensor([0.0], device=device))

    def forward(self, t, v0):
        # v0 shape [1,1] -> [1]
        y0 = torch.cat([v0.view(-1), self.w0])  # shape [2]
        # pred_y = odeint(self.odefunc, y0, t)
        pred_y = odeint(self.odefunc, y0, t, method='rk4')
        return pred_y

# Prepare data for training
v0 = v_true[0].detach()  # initial v from data

model = FHNNeuralODEModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
n_epochs = 500
for epoch in trange(n_epochs):
    optimizer.zero_grad()
    pred_y = model(time, v0)  # shape [T, 2]
    v_pred = pred_y[..., 0].unsqueeze(-1)
    loss = loss_fn(v_pred, v_true)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        # Plot results
        pred_y = model(time, v0).detach().cpu()
        plt.figure(figsize=(12, 5))
        plt.plot(time.cpu(), v_true.cpu(), label='True v')
        plt.plot(time.cpu(), pred_y[..., 0], '--', label='Predicted v')
        plt.plot(time.cpu(), pred_y[..., 1], ':', label='Inferred w (latent)')
        plt.legend()
        plt.title("FitzHugh-Nagumo Neural ODE with latent w")
        plt.savefig('./tmp/fhn_neural_ode_results.png', dpi=170, bbox_inches='tight')
        plt.close()

