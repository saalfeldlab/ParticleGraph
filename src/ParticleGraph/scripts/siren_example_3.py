import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


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


# ----------------- SIREN Network -----------------
class Siren(nn.Module):
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


# ----------------- Training Setup -----------------
def train_siren_on_timeseries(w_true, n_epochs=10000, batch_size=1000, device='cuda'):
    n_steps = len(w_true)
    t_full = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)  # shape (10000, 1)
    w_true = torch.tensor(w_true, dtype=torch.float32, device=device)  # shape (10000,)

    model = Siren(in_features=1, out_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in trange(n_epochs):
        idx = torch.randint(0, n_steps, (batch_size,))
        t_batch = t_full[idx]  # (batch_size, 1)
        w_batch = w_true[idx].unsqueeze(1)  # (batch_size, 1)

        pred = model(t_batch)
        loss = F.mse_loss(pred, w_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            with torch.no_grad():
                pred_full = model(t_full).squeeze().cpu().numpy()
                plt.figure(figsize=(10, 4))
                plt.plot(w_true.cpu().numpy(), label="Target")
                plt.plot(pred_full, label="SIREN Output")
                plt.title(f"Epoch {epoch}")
                plt.legend()
                plt.show()

    return model


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    # Example: generate a synthetic time series
    n_points = 10000
    t = np.linspace(0, 1, n_points)
    w_true = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 13 * t)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_siren_on_timeseries(w_true, n_epochs=10000, batch_size=1000, device=device)
