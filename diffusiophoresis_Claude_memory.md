# Working Memory: diffusiophoresis

## Knowledge Base

### Pattern Principles
- Brusselator with A=4.5, B=6.5 produces Turing labyrinthine/stripe patterns
- **CONFIRMED**: Opposite-sign mobilities (M1>0, M2<0) essential for particle-field coupling
- **CONFIRMED**: Stability range |M|=50-180; |M|≥200 crashes
- **CONFIRMED**: |M|=180 is optimal for clustering without instability
- **CONFIRMED**: D1=0.05 is optimal (lower D1=0.025 reduces clustering by 31%)
- **CONFIRMED**: D1/D2 ratio 0.0625 >> 0.125 for finer pattern wavelength
- clustering metric: 0.0156 (baseline) → 0.0395 (|M|=150) → 0.0795 (|M|=180) → 0.1092 (D1=0.05) → 0.0751 (D1=0.025)

### Failed Configurations
- Same-sign mobilities (M1=M2) → no particle clustering (forces cancel)
- |M|≥200 with opposite signs → numerical instability/crash
- D1=0.025 (too low) → weaker patterns than D1=0.05
- |M|=50 → too weak coupling, no visible organization

### Code Insights
(none yet - block 2 may explore code modifications)

---

## Previous Block Summary (Block 1)

**Achievement**: Progressed from uniform particles (4/10) to exceptional filamentary networks (9/10)

**Best config**: D1=0.05, M1=180, M2=-180, D2=0.8, Da_c=10, A=4.5, B=6.5
- clustering=0.1092, C1_std=2.0150

**Key findings**:
1. Opposite-sign mobilities create differential response → particle organization
2. |M|=180 is stability ceiling (200 crashes)
3. Lower D1 creates sharper gradients → stronger diffusiophoresis
4. Score progression: 4→4→2→4→7→8→3→9

---

## Current Block (Block 2)

### Block Info
- Iterations: 9-16
- Starting config: D1=0.05, M1=180, M2=-180, D2=0.8, Da_c=10, A=4.5, B=6.5
- n_frames=2000, n_particles=9600, n_nodes=10000
- **CODE MODIFICATIONS ALLOWED** (block boundary)

### Hypothesis for Block 2
Since we achieved 9/10, explore dimensions not yet tested:
1. **Parameter exploration**: Even lower D1 (0.025), or higher Da_c (faster reactions)
2. **Code modification**: Add particle feedback (consumption/production rates to create true coupling)
3. **Longer dynamics**: Increase n_frames to 4000 for more complex evolution
4. **Multi-scale**: Try higher mesh resolution (n_nodes=22500) for finer patterns

### Iterations This Block
(Starting block 2)

### Emerging Observations
- Block 1 established strong baseline with particle networks
- Next: push toward more complex/dynamic patterns (spirals, traveling waves)

