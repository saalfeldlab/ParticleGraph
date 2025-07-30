import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import ttest_ind

class NoisyMLP(nn.Module):
    def __init__(self, noise_std=0.0):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.noise_std = noise_std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = F.relu(x)
        x = self.fc2(x)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = F.relu(x)
        x = self.fc3(x)
        return x

def get_loaders(batch_size=128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size)

def train_and_evaluate(noise_std, device):
    model = NoisyMLP(noise_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_loaders()

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_levels = [0.0, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7]
all_results = {}

for noise in noise_levels:
    accuracies = [train_and_evaluate(noise, device) for _ in range(5)]
    all_results[noise] = accuracies

# T-test against noise=0
control = np.array(all_results[0.0])
significance_results = {}

for noise, accs in all_results.items():
    mean_acc = np.mean(accs)
    if noise == 0.0:
        p_val = 1.0
    else:
        _, p_val = ttest_ind(control, accs, equal_var=False)
    significance_results[noise] = (mean_acc, p_val)

print("Noise Level -> (Mean Accuracy, p-value vs noise=0):")
for noise, (mean_acc, p_val) in significance_results.items():
    print(f"{noise:8.1e} -> ({mean_acc:.4f}, p={p_val:.4f})")
