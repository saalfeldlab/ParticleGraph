#!/usr/bin/env python3
"""Generate particle-particle interaction force plot."""

import numpy as np
import matplotlib.pyplot as plt

# Distance array
d = np.linspace(0.001, 0.15, 500)

# Parameters (from particle-particle.qmd)
sigma = 0.005
p1, p2 = 0.5, 1.0  # attraction
p3, p4 = 1.0, 1.0  # repulsion

# Force law: F = p1*exp(-d^(2p2)/(2σ²)) - p3*exp(-d^(2p4)/(2σ²))
F_attract = p1 * np.exp(-d**(2*p2) / (2*sigma**2))
F_repel = p3 * np.exp(-d**(2*p4) / (2*sigma**2))
F_total = F_attract - F_repel

# Simple repulsion model
alpha = 50
r_rep = 0.04
F_simple = np.where(d < r_rep, -alpha * np.exp(-5*d/r_rep), 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: General force law
ax1.plot(d, F_attract, 'g--', label='Attraction', linewidth=2)
ax1.plot(d, -F_repel, 'r--', label='Repulsion', linewidth=2)
ax1.plot(d, F_total, 'b-', label='Net force', linewidth=2.5)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax1.set_xlabel('Distance $d_{ij}$', fontsize=12)
ax1.set_ylabel('Force', fontsize=12)
ax1.set_title('General Attraction-Repulsion', fontsize=14)
ax1.legend()
ax1.set_xlim(0, 0.15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: Simple repulsion
ax2.fill_between(d[d < r_rep], 0, F_simple[d < r_rep], alpha=0.3, color='red')
ax2.plot(d, F_simple, 'r-', linewidth=2.5, label='Repulsion force')
ax2.axvline(x=r_rep, color='gray', linestyle='--', alpha=0.7, label=f'$r_{{rep}}$ = {r_rep}')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax2.set_xlabel('Distance $d_{ij}$', fontsize=12)
ax2.set_ylabel('Force', fontsize=12)
ax2.set_title('Simple Repulsion (Default)', fontsize=14)
ax2.legend()
ax2.set_xlim(0, 0.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('assets/pp_interaction.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: assets/pp_interaction.png")
plt.close()
