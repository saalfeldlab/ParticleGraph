# Working Memory: diffusiophoresis

## Knowledge Base

### Pattern Principles
- D2/D1 ratio controls pattern wavelength; ratio ~10-16 favors spots, lower may favor stripes
- Turing condition: B > 1+A² for instability
- Particle organization correlates with field gradients - halos form at concentration boundaries

### Failed Configurations
(none yet)

### Code Insights
(none yet)

---

## Previous Block Summary
(First block - no previous)

---

## Current Block (Block 1)

### Block Info
- Parameters: A=4.5, B=6.5 (below Turing threshold 21.25), D1=0.05, D2=0.8
- Iterations: 1-8
- Focus: Find parameters that produce stripes or more complex patterns

### Hypothesis
Current A=4.5 makes Turing threshold very high (B > 21.25). Reducing A should lower threshold and allow exploring spot-to-stripe transition. Also try increasing B.

### Iterations This Block

**Iter 1 (5/10)**: Baseline. Classic spot pattern with particle halos. Static, uniform.
- Config: A=4.5, B=6.5, D1=0.05, D2=0.8, Da_c=15.0

**Iter 2 (7/10)**: LABYRINTHINE PATTERNS! Major improvement. Connected maze-like structures.
- Config: A=1.5, B=3.5, D1=0.05, D2=0.8, Da_c=15.0
- Key: Lowering A to 1.5 made B=3.5 > 1+A²=3.25 (just above Turing threshold)
- Particles form elongated clusters following field gradients
- Next: Try B=4.0-4.5 to push further into stripe regime, or reduce D2/D1 ratio

### Emerging Observations
- Near-threshold dynamics (B just above 1+A²) favor labyrinthine/connected patterns over spots
- Reducing A is more effective than increasing B for reaching Turing condition
- Particle-field coupling creates elongated clusters that follow field gradients

