# Pattern Exploration Log: diffusiophoresis

## Iter 1: 5/10
Node: id=1, parent=root
Mode/Strategy: explore (baseline)
Config: params_mesh[0]=[0.05, 15.0, 4.5, 6.5, 0.04, -16], params_mesh[1]=[0.8, 16], params_mesh[2]=[1.0, 180, -180], n_frames=2000, delta_t=5E-4
Score: 5/10
Visual: Classic Turing spot pattern in C1/C2 fields. Particles organize into ring-like halos around spot centers - pushed to boundaries between high/low concentration regions. Pattern is static and uniform.
Metrics: C1_mean=3.48, C1_std=1.56, C2_mean=1.54, C2_std=1.05, clustering=0.08
Literature: Turing condition B > 1+A² not satisfied (6.5 < 21.25), yet patterns form - possibly from initial perturbations/other dynamics
Observation: Baseline shows working simulation with spot patterns. Need to explore stripe-favoring parameters.
Mutation: None (baseline)
Parent rule: root (first iteration)
Next: Increase B toward Turing threshold and adjust D2/D1 ratio toward stripes

## Iter 2: 7/10
Node: id=2, parent=1
Mode/Strategy: exploit (UCB selection from node 1)
Config: params_mesh[0]=[0.05, 15.0, 1.5, 3.5, 0.04, -16], D1=0.05, D2=0.8, A=1.5, B=3.5, n_frames=2000, delta_t=5E-4
Score: 7/10
Visual: LABYRINTHINE PATTERN! Connected maze-like structures in C1/C2 fields instead of isolated spots. Particles form elongated clusters following the field pattern. Clear correlation between particle positions and field gradients. Dynamic evolution visible over 10 frames.
Metrics: C1_mean=0.38, C1_std=0.43, C2_mean=2.00, C2_std=0.84, pattern_growth=168, clustering=0.12
Literature: Turing condition B=3.5 > 1+A²=3.25 now satisfied (just above threshold). Near-threshold dynamics favor stripes/labyrinthine over spots.
Observation: Reducing A from 4.5 to 1.5 was key - lowered Turing threshold making B=3.5 effective. Connected topology is more interesting than isolated dots.
Mutation: A: 4.5 → 1.5, B: 6.5 → 3.5
Parent rule: Selected node 1 (UCB=1.21, highest)
Next: Explore slightly higher B (3.8-4.0) to see if stripes become more pronounced, or try lower D2/D1 ratio

