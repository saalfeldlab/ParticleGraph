import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib

matplotlib.use('Agg')  # Fix Qt display issues
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available, using simplified t-test")


class NoisyMLP(nn.Module):
    def __init__(self, noise_level=0.0):
        super().__init__()
        self.noise_level = noise_level
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.fc1(x)
        if self.noise_level > 0:
            x += torch.randn_like(x) * self.noise_level
        x = self.relu(x)

        x = self.fc2(x)
        if self.noise_level > 0:
            x += torch.randn_like(x) * self.noise_level
        x = self.relu(x)

        x = self.fc3(x)
        return x


def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Testing (explicitly disable noise for inference)
        model.eval()
        original_noise = model.noise_level
        model.noise_level = 0.0  # Ensure no noise during testing
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        model.noise_level = original_noise  # Restore original noise level

        test_acc = 100. * correct / len(test_loader.dataset)
        train_losses.append(train_loss / len(train_loader))
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Test Acc: {test_acc:.2f}%')

    return train_losses, test_accs


# Load MNIST data (automatically downloads if not present)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Test different noise levels with multiple runs
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
results = {}
n_runs = 5

print("Training models with different noise levels (5 runs each)...")
for noise in noise_levels:
    print(f"\n--- Noise Level: {noise} ---")
    all_train_losses = []
    all_test_accs = []
    final_accs = []

    for run in range(n_runs):
        print(f"  Run {run + 1}/5", end=" - ")
        model = NoisyMLP(noise_level=noise)
        train_losses, test_accs = train_model(model, train_loader, test_loader)
        all_train_losses.append(train_losses)
        all_test_accs.append(test_accs)
        final_accs.append(test_accs[-1])

    results[noise] = {
        'all_train_losses': all_train_losses,
        'all_test_accs': all_test_accs,
        'final_accs': final_accs,
        'mean_acc': sum(final_accs) / n_runs,
        'std_acc': (sum([(acc - sum(final_accs) / n_runs) ** 2 for acc in final_accs]) / (n_runs - 1)) ** 0.5
    }

# Plot results as bar charts (better for single epoch)
plt.figure(figsize=(14, 5))

# Extract means and stds for plotting
noise_labels = [str(n) for n in noise_levels]
mean_losses = [np.mean([losses[0] for losses in results[noise]['all_train_losses']]) for noise in noise_levels]
std_losses = [np.std([losses[0] for losses in results[noise]['all_train_losses']]) for noise in noise_levels]
mean_accs = [results[noise]['mean_acc'] for noise in noise_levels]
std_accs = [results[noise]['std_acc'] for noise in noise_levels]

plt.subplot(1, 2, 1)
bars1 = plt.bar(noise_labels, mean_losses, yerr=std_losses, capsize=5, alpha=0.7)
plt.xlabel('Noise Level')
plt.ylabel('Training Loss')
plt.title('Final Training Loss vs Noise Level')
plt.xticks(rotation=45)
# Color the best performer
best_loss_idx = np.argmin(mean_losses)
bars1[best_loss_idx].set_color('red')

plt.subplot(1, 2, 2)
bars2 = plt.bar(noise_labels, mean_accs, yerr=std_accs, capsize=5, alpha=0.7)
plt.xlabel('Noise Level')
plt.ylabel('Test Accuracy (%)')
plt.title('Final Test Accuracy vs Noise Level')
plt.xticks(rotation=45)
# Color the best performer
best_acc_idx = np.argmax(mean_accs)
bars2[best_acc_idx].set_color('green')

plt.tight_layout()
plt.savefig('mnist_noise_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'noise_MNIST_comparison.png'")

# Statistical significance test for all noise levels vs baseline
baseline_accs = results[0.0]['final_accs']  # No noise baseline
best_noise = max(noise_levels[1:], key=lambda x: results[x]['mean_acc'])  # Exclude 0.0

print(f"\n=== STATISTICAL ANALYSIS vs Baseline (Noise=0.0) ===")
print(f"Baseline: {results[0.0]['mean_acc']:.2f}% ± {results[0.0]['std_acc']:.2f}%")
print(f"Baseline runs: {[f'{acc:.1f}%' for acc in baseline_accs]}")
print("\nComparison with each noise level:")
print("-" * 70)

for noise in noise_levels[1:]:  # Skip 0.0 since it's the baseline
    test_accs = results[noise]['final_accs']

    # Statistical significance test
    if SCIPY_AVAILABLE:
        t_stat, p_value = stats.ttest_ind(test_accs, baseline_accs)
        p_str = f"{p_value:.4f}"
        if p_value < 0.001:
            sig_str = "*** (p < 0.001)"
        elif p_value < 0.01:
            sig_str = "**  (p < 0.01)"
        elif p_value < 0.05:
            sig_str = "*   (p < 0.05)"
        else:
            sig_str = "    (n.s.)"
    else:
        # Simple t-test calculation
        n1, n2 = len(test_accs), len(baseline_accs)
        mean1, mean2 = np.mean(test_accs), np.mean(baseline_accs)
        std1, std2 = np.std(test_accs, ddof=1), np.std(baseline_accs, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1 / n1 + 1 / n2))
        p_str = "N/A"
        sig_str = "(install scipy)"

    improvement = results[noise]['mean_acc'] - results[0.0]['mean_acc']
    direction = "↑" if improvement > 0 else "↓"

    print(f"Noise {noise:8}: {results[noise]['mean_acc']:.2f}% ± {results[noise]['std_acc']:.2f}% | "
          f"{direction}{abs(improvement):5.2f}pp | t={t_stat:6.3f} | p={p_str:8s} {sig_str}")

print(f"\nBest performing noise level: {best_noise}")
print("Legend: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant, pp = percentage points")

# Print final results with statistics
print("\n=== FINAL RESULTS (Mean ± Std over 5 runs) ===")
for noise in noise_levels:
    mean_acc = results[noise]['mean_acc']
    std_acc = results[noise]['std_acc']
    print(f"Noise {noise:8}: {mean_acc:.2f}% ± {std_acc:.2f}%")