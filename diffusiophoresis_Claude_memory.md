# Working Memory: diffusiophoresis

## Regime Comparison

| Regime | mesh_model | particle_model | n_types | n_particles | Best Score | Key Insight |
| ------ | ---------- | -------------- | ------- | ----------- | ---------- | ----------- |
| Base   | Diffusiophoresis_Mesh | PDE_ParticleField_D | 1 | 9600 | 5/10 | Good Turing patterns, particle escape issue |

## Insights

| Category    | Finding                                              |
| ----------- | ---------------------------------------------------- |
| Patterns    | Brusselator produces clear Turing spots with current params |
| Performance | params_mesh A=4.5, B=6.5 creates visible spots |
| Failures    | [none yet] |

---

## Knowledge Base

### Established Principles

(Need 3+ iterations to establish)

### Open Questions

- Can reducing particle mobility improve retention while maintaining pattern coupling?
- Would longer simulation (more frames) allow plateau convergence?
- How will multi-type particles (n_particle_types=2,3) affect pattern formation?

### Failed Configurations

(None yet)

### Code Insights

(None yet)

### PDE Variants

| Variant | Model | Literature | Status | Best Score |
| ------- | ----- | ---------- | ------ | ---------- |
| Diffusiophoresis_Mesh | Brusselator | Prigogine (1968) | active | 5/10 |

### Particle Type Distribution

| n_particle_types | Count | Target |
| ---------------- | ----- | ------ |
| 1                | 1     | ~33%   |
| 2                | 0     | ~33%   |
| 3                | 0     | ~33%   |

**Action needed if imbalanced:** Consider n_particle_types=2 or 3 in upcoming iterations

---

## Previous Block Summary

(Block 1 in progress - no previous block)

---

## Current Block (Block 1)

### Block Info

mesh_model_name: Diffusiophoresis_Mesh (Brusselator)
particle_model_name: PDE_ParticleField_D
Iterations: 1-8

### Hypothesis

Block 1: Establish baseline and explore parameter space to find configurations with stable patterns and good particle confinement.

### Iterations This Block

**Iter 1 (baseline): 5/10**
- Config: n_particle_types=1, n_frames=2000, delta_t=0.0005
- Metrics: entropy=0.74, plateau=0.00, in_box=85.1%
- Visual: Clear Turing spots in fields, particles cluster around spots but 15% escape
- Key issue: plateau=0 (not converged), particle escape
- Next: Reduce mobility (M1, M2) to improve particle retention

### Emerging Observations

- Brusselator field dynamics work well - clear spot patterns emerge
- Current mobility values (-16, 16) may be too strong causing particle escape
- Need to balance particle-field coupling strength vs stability
- Spatial entropy at 0.74 is slightly high (ideal 0.4-0.7)

