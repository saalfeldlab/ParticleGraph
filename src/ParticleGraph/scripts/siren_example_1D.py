import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


# ----------- SIREN Layers ----------- #
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []

        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
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
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


# ----------- Dataset ----------- #
class TimeSeriesDataset(Dataset):
    def __init__(self, length=1024):
        self.length = length
        self.coords = torch.linspace(-1, 1, steps=length).unsqueeze(1)
        self.signal = self.generate_signal(self.coords)

    def generate_signal(self, coords):
        # Generate a synthetic signal: sum of sine waves + noise
        freq1, freq2 = 3.0, 9.0
        signal = torch.sin(freq1 * np.pi * coords) + 0.5 * torch.sin(freq2 * np.pi * coords)
        signal += 0.05 * torch.randn_like(signal)  # add noise
        return signal

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.signal


# ----------- Training Loop ----------- #
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = TimeSeriesDataset(length=1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Siren(in_features=1, out_features=1, hidden_features=128,
                  hidden_layers=3, outermost_linear=True,
                  first_omega_0=30, hidden_omega_0=30.).to(device)

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    total_steps = 1000
    steps_til_summary = 100


    for step in trange(total_steps):
        # Randomly sample 100 indices
        idx = torch.randperm(model_input.shape[1])[:100]

        x_sample = model_input[0, idx, :].to(device)  # shape: [100, 1]
        y_sample = ground_truth[0, idx, :].to(device)  # shape: [100, 1]

        model_output, coords = model(x_sample)

        loss = ((model_output - y_sample) ** 2).mean()

        if not step % steps_til_summary:
            print(f"Step {step}, Loss {loss.item():.6f}")

            with torch.no_grad():
                full_output, _ = model(model_input.squeeze(0).to(device))
                x = model_input.squeeze().cpu().numpy()
                y_true = ground_truth.squeeze().cpu().numpy()
                y_pred = full_output.squeeze().cpu().numpy()

                plt.plot(x, y_true, label='Target')
                plt.plot(x, y_pred, label='SIREN Output')
                plt.legend()
                plt.title(f'Step {step}')
                plt.show()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
