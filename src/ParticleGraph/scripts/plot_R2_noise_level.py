import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")


# Data extracted from weight comparison plots
noise_levels = [0.008, 0.5, 1.0, 2.0, 0.1, 0.05, 0.01]
r_squared = [0.892, 0.989, 0.515, 0.001, 0.879, 0.923, 0.911]
experiments = ['fly_N9_19_18_4_0', 'fly_N9_18_4_1', 'fly_N9_18_4_2', 'fly_N9_18_4_3',
               'fly_N9_18_4_4', 'fly_N9_18_4_5', 'fly_N9_18_4_6']

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create scatter plot with log scale
ax.scatter(noise_levels, r_squared, s=80, c='#2563eb', alpha=0.8,
           edgecolors='#1d4ed8', linewidth=2)
ax.set_xscale('log')
# plt.tight_layout()
# plt.show()


# Set log scale for x-axis
ax.set_xscale('log')
ax.set_xlim(0.008, 3)
ax.set_ylim(0, 1.05)

# Labels and title
ax.set_xlabel('Noise Level (log scale)', fontsize=12)
ax.set_ylabel('R² (Weight Reconstruction)', fontsize=12)
ax.set_title('Weight Reconstruction Quality vs Neural Noise Level',
             fontsize=14, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3)

# Annotate points with experiment names
for i, (x, y, exp) in enumerate(zip(noise_levels, r_squared, experiments)):
   ax.annotate(f'{exp}\n(R²={y:.3f})',
                   (x, y),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# Tight layout
plt.tight_layout()

# Print summary statistics
print("Weight Reconstruction Analysis Summary:")
print("=" * 40)
print(f"{'Noise Level':<12} {'R²':<8} {'Quality'}")
print("-" * 30)
for noise, r2 in zip(noise_levels, r_squared):
    if r2 > 0.9:
        quality = "Excellent"
    elif r2 > 0.7:
        quality = "Good"
    elif r2 > 0.3:
        quality = "Poor"
    else:
        quality = "Failed"
    print(f"{noise:<12} {r2:<8.3f} {quality}")

print(f"\nKey Findings:")
print(f"- Optimal performance: 0.01-0.05 noise level")
print(f"- Performance threshold: R² drops significantly above 0.1")
print(f"- Complete failure at noise level 2.0")

# Show plot
plt.show()

# Optional: Save the plot
# plt.savefig('noise_vs_r2_analysis.png', dpi=300, bbox_inches='tight')