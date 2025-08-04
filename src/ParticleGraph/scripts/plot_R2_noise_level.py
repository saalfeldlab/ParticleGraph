import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

# Data from experiments
# Null edges experiments at optimal noise (structural corruption)
null_edges_optimal = [0, 1000, 10000, 100000, 1000000]
null_r2_optimal = [0.9889, 0.9975, 0.9978, 0.9995, 0.9827]
null_embedding_optimal = [72.9, 77.1, 83.7, 78.6, 76.6]
null_functional_optimal = [77.6, 75.9, 83.3, 80.4, 76.6]

# Signal noise experiments - CORRECTED DATA
# The original noise=0.0 point was actually noise=0.5, now corrected with true baseline from config 20_0
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
noise_r2 = [0.6466, 0.9112, 0.9234, 0.8792, 0.9889, 0.5154, 0.0003]  # First value corrected to true baseline
noise_embedding = [35.7, 37.3, 66.9, 68.9, 72.9, 51.2, 58.8]  # First value corrected to true baseline
noise_functional = [25.5, 32.6, 63.7, 62.3, 77.6, 47.6, 40.9]  # First value corrected to true baseline

# Null edges experiments at clean conditions (noise=0)
null_edges_clean = [0, 1000, 10000, 100000, 1000000, 2000000, 4000000]
null_r2_clean = [0.6466, 0.7790, 0.6450, 0.9137, 0.7046, 0.7417, 0.4642]
null_embedding_clean = [35.7, 39.0, 40.8, 26.0, 26.7, 29.8, 36.6]
null_functional_clean = [25.5, 35.5, 31.8, 18.8, 13.7, 14.8, 24.0]

# Create the plots - 3 rows, 3 columns
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Row 1: Null edges at optimal noise (original results)
axes[0,0].plot(null_edges_optimal, null_r2_optimal, 'o-', color='blue', linewidth=2, markersize=8)
axes[0,0].set_xlabel('Extra Null Edges')
axes[0,0].set_ylabel('R²')
axes[0,0].set_title('Null Edges + Optimal Noise: Weight Reconstruction')
axes[0,0].set_xscale('symlog')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_ylim(0, 1.05)

axes[0,1].plot(null_edges_optimal, null_embedding_optimal, 'o-', color='green', linewidth=2, markersize=8)
axes[0,1].set_xlabel('Extra Null Edges')
axes[0,1].set_ylabel('Embedding Accuracy (%)')
axes[0,1].set_title('Null Edges + Optimal Noise: Embedding')
axes[0,1].set_xscale('symlog')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_ylim(0, 100)

axes[0,2].plot(null_edges_optimal, null_functional_optimal, 'o-', color='purple', linewidth=2, markersize=8)
axes[0,2].set_xlabel('Extra Null Edges')
axes[0,2].set_ylabel('Functional Accuracy (%)')
axes[0,2].set_title('Null Edges + Optimal Noise: Functional')
axes[0,2].set_xscale('symlog')
axes[0,2].grid(True, alpha=0.3)
axes[0,2].set_ylim(0, 100)

# Row 2: Signal noise plots
axes[1,0].plot(noise_levels, noise_r2, 'o-', color='red', linewidth=2, markersize=8)
axes[1,0].set_xlabel('Noise Level')
axes[1,0].set_ylabel('R²')
axes[1,0].set_title('Signal Noise: Weight Reconstruction')
axes[1,0].set_xscale('symlog', linthresh=0.01)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_ylim(0, 1.05)

axes[1,1].plot(noise_levels, noise_embedding, 'o-', color='orange', linewidth=2, markersize=8)
axes[1,1].set_xlabel('Noise Level')
axes[1,1].set_ylabel('Embedding Accuracy (%)')
axes[1,1].set_title('Signal Noise: Embedding Clustering')
axes[1,1].set_xscale('symlog', linthresh=0.01)
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_ylim(0, 100)

axes[1,2].plot(noise_levels, noise_functional, 'o-', color='brown', linewidth=2, markersize=8)
axes[1,2].set_xlabel('Noise Level')
axes[1,2].set_ylabel('Functional Accuracy (%)')
axes[1,2].set_title('Signal Noise: Functional Clustering')
axes[1,2].set_xscale('symlog', linthresh=0.01)
axes[1,2].grid(True, alpha=0.3)
axes[1,2].set_ylim(0, 100)

# Row 3: Null edges at clean conditions (noise=0)
axes[2,0].plot(null_edges_clean, null_r2_clean, 'o-', color='darkblue', linewidth=2, markersize=8)
axes[2,0].set_xlabel('Extra Null Edges')
axes[2,0].set_ylabel('R²')
axes[2,0].set_title('Null Edges + Clean Data: Weight Reconstruction')
axes[2,0].set_xscale('symlog')
axes[2,0].grid(True, alpha=0.3)
axes[2,0].set_ylim(0, 1.05)

axes[2,1].plot(null_edges_clean, null_embedding_clean, 'o-', color='darkgreen', linewidth=2, markersize=8)
axes[2,1].set_xlabel('Extra Null Edges')
axes[2,1].set_ylabel('Embedding Accuracy (%)')
axes[2,1].set_title('Null Edges + Clean Data: Embedding')
axes[2,1].set_xscale('symlog')
axes[2,1].grid(True, alpha=0.3)
axes[2,1].set_ylim(0, 100)

axes[2,2].plot(null_edges_clean, null_functional_clean, 'o-', color='darkred', linewidth=2, markersize=8)
axes[2,2].set_xlabel('Extra Null Edges')
axes[2,2].set_ylabel('Functional Accuracy (%)')
axes[2,2].set_title('Null Edges + Clean Data: Functional')
axes[2,2].set_xscale('symlog')
axes[2,2].grid(True, alpha=0.3)
axes[2,2].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('comprehensive_robustness_analysis_3x3.png', dpi=300, bbox_inches='tight')
plt.close()

# Print comprehensive analysis
print("=== COMPREHENSIVE ROBUSTNESS ANALYSIS (CORRECTED) ===")

print("\n1. NULL EDGES + OPTIMAL NOISE (Row 1):")
print(f"Baseline: R²={null_r2_optimal[0]:.3f}, Embedding={null_embedding_optimal[0]:.1f}%, Functional={null_functional_optimal[0]:.1f}%")
optimal_null_opt_idx = np.argmax(null_embedding_optimal)
print(f"Optimal ({null_edges_optimal[optimal_null_opt_idx]} edges): R²={null_r2_optimal[optimal_null_opt_idx]:.3f}, Embedding={null_embedding_optimal[optimal_null_opt_idx]:.1f}%, Functional={null_functional_optimal[optimal_null_opt_idx]:.1f}%")

print("\n2. SIGNAL NOISE (Row 2) - CORRECTED:")
print(f"Clean (0.0): R²={noise_r2[0]:.3f}, Embedding={noise_embedding[0]:.1f}%, Functional={noise_functional[0]:.1f}%")
optimal_noise_idx = np.argmax(noise_embedding)
print(f"Optimal ({noise_levels[optimal_noise_idx]}): R²={noise_r2[optimal_noise_idx]:.3f}, Embedding={noise_embedding[optimal_noise_idx]:.1f}%, Functional={noise_functional[optimal_noise_idx]:.1f}%")

print("\n3. NULL EDGES + CLEAN DATA (Row 3):")
print(f"Baseline: R²={null_r2_clean[0]:.3f}, Embedding={null_embedding_clean[0]:.1f}%, Functional={null_functional_clean[0]:.1f}%")
optimal_null_clean_idx = np.argmax(null_embedding_clean)
print(f"Best ({null_edges_clean[optimal_null_clean_idx]} edges): R²={null_r2_clean[optimal_null_clean_idx]:.3f}, Embedding={null_embedding_clean[optimal_null_clean_idx]:.1f}%, Functional={null_functional_clean[optimal_null_clean_idx]:.1f}%")

print("\n=== CORRECTED KEY INSIGHTS ===")
true_baseline = noise_embedding[0]  # Now correctly using the true clean baseline
print(f"1. Signal noise dominance: Clean→Optimal noise gives {noise_embedding[optimal_noise_idx]/true_baseline:.1f}x embedding improvement")
print(f"2. Null edges alone (clean): {null_embedding_clean[optimal_null_clean_idx]/null_embedding_clean[0]:.1f}x embedding improvement (modest)")
print(f"3. Null edges + optimal noise: {null_embedding_optimal[optimal_null_opt_idx]/null_embedding_optimal[0]:.1f}x embedding improvement")
print(f"4. Combined synergy: {null_embedding_optimal[optimal_null_opt_idx]/true_baseline:.1f}x total improvement from TRUE baseline")

print(f"\n=== CORRECTED REGULARIZATION MECHANISM ANALYSIS ===")
print(f"Signal noise = PRIMARY regularization ({noise_embedding[optimal_noise_idx]/true_baseline:.1f}x effect)")
print(f"Null edges = SECONDARY regularization ({null_embedding_clean[optimal_null_clean_idx]/null_embedding_clean[0]:.1f}x effect)")
print(f"Combined = MULTIPLICATIVE effect ({null_embedding_optimal[optimal_null_opt_idx]/true_baseline:.1f}x total)")
print("\nIMPORTANT: Previous analysis underestimated improvements due to config error in 18_4_0!")
print("True clean baseline is much lower, making regularization effects even more dramatic.")







# Configure LaTeX font
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Best performance values from corrected experiments
conditions = ['baseline', 'extra edges', 'extra noise', 'extra noise\n+ edges']

# Best R² values (0-1 scale)
r2_values = [
    0.6466,  # Baseline (clean, no extras)
    0.9137,  # Extra edges alone (best at 100K edges)
    0.9889,  # Extra noise alone (best at noise=0.5)
    0.9995   # Extra noise + edges (best at 100K edges)
]

# Best embedding clustering accuracy (%)
embedding_values = [
    35.7,   # Baseline (clean, no extras)
    40.8,   # Extra edges alone (best at 10K edges)
    72.9,   # Extra noise alone (best at noise=0.5)
    83.7    # Extra noise + edges (best at 10K edges)
]

# Define colors for flat design
colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']

# Create side-by-side bar plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: R² Comparison
bars1 = ax1.bar(conditions, r2_values, color=colors)
ax1.set_ylabel(r'$R^2$', fontsize=16)
ax1.set_title('weight reconstruction', fontsize=16)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on R² bars
for bar, value in zip(bars1, r2_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}', ha='center', va='bottom')

# Plot 2: Embedding Clustering Comparison
bars2 = ax2.bar(conditions, embedding_values, color=colors)
ax2.set_ylabel(r'embedding accuracy (\%)', fontsize=16)
ax2.set_title('embedding clustering accuracy', fontsize=16)
ax2.set_ylim(0, 90)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on embedding bars
for bar, value in zip(bars2, embedding_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{value:.1f}\\%', ha='center', va='bottom')

# Adjust layout and rotate x-axis labels
plt.tight_layout()
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=45)

# Save the plot
plt.savefig('performance_comparison_bars.png', dpi=300, bbox_inches='tight')
plt.close()