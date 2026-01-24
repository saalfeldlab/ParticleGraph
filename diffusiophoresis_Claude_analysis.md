# Pattern Exploration Log: diffusiophoresis

## Iter 1: 4/10
Node: id=1, parent=root
Mode/Strategy: initial baseline
Config: params_mesh=[[0.1, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [0.1, 250, 250, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 4/10
Visual: Classic Turing dot patterns in fields C1/C2. Particles remain uniformly distributed with very low clustering (0.0156). No particle-field correlation visible. Static final state.
Metrics: C1_std=0.27, C2_std=0.53, pattern_growth=105, clustering=0.016
Observation: Baseline produces expected Turing dots but particles don't self-organize. The coupling (Pe=0.1, M1=250, M2=250) may be too weak relative to particle diffusion/repulsion.
Next: parent=1, explore stronger particle-field coupling (increase Pe or mobilities)

## Iter 2: 4/10
Node: id=2, parent=1
Mode/Strategy: exploit - increased Pe coupling
Config: params_mesh=[[0.1, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 250, 250, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 4/10
Visual: Turing dot patterns in fields. Particles remain uniformly distributed - NO improvement in clustering. Pattern quality similar to baseline.
Metrics: C1_std=0.286, C2_std=0.559, pattern_growth=112, clustering=0.0086 (WORSE than iter 1)
Mutation: Pe: 0.1 -> 1.0 (10x increase)
Parent rule: From Node 1, increased particle-field coupling strength
Observation: Increasing Pe alone doesn't create clustering. Hypothesis: M1=M2=250 (same sign) causes canceling gradients since C1/C2 peaks are anti-correlated in Turing patterns. Need opposite-sign mobilities.
Next: parent=1, try opposite-sign mobilities (M1=250, M2=-250) to create net force toward pattern features

## Iter 3: 2/10
Node: id=3, parent=1
Mode/Strategy: exploit - opposite-sign mobilities
Config: params_mesh=[[0.1, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 250, -250, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 2/10
Visual: CRASH/INSTABILITY. Early frames (3-6) showed promising particle clustering forming network/filament structures correlating with Turing patterns! But simulation became numerically unstable - fields collapsed to uniform then blew up (white frames). All metrics returned nan.
Metrics: All nan (simulation crashed)
Mutation: M2: 250 -> -250 (opposite sign mobility)
Parent rule: From Node 1, tested opposite-sign mobilities hypothesis
Observation: The opposite mobility concept IS working - early frames showed particle-field coupling creating network structures. But |M|=250 is too strong causing runaway feedback. Need smaller magnitude mobilities for stability. Try M1=50, M2=-50.
Next: parent=1, try weaker opposite-sign mobilities (M1=50, M2=-50) for stable coupling

## Iter 4: 4/10
Node: id=4, parent=3
Mode/Strategy: exploit (from node 3 - reduce mobility magnitude)
Config: params_mesh=[[0.1, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 50, -50, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 4/10
Visual: Stable Turing dots in fields, particles uniformly distributed, no visible clustering
Metrics: C1_std=1.276 (higher than baseline), clustering=-0.0103 (no clustering)
Mutation: M1: 250->50, M2: -250->-50 (reduced magnitude by 5x, kept opposite signs)
Parent rule: Inherit opposite-sign mobility strategy from iter 3
Observation: Reduced mobility stabilized simulation (no crash) but particles still uniform. |M|=50 too weak to overcome particle diffusion. Need intermediate magnitude (M=100-150) or different coupling mechanism (consumption/production rates).
Next: parent=2, try intermediate mobility M1=150, M2=-150

## Iter 5: 7/10 ★ BREAKTHROUGH
Node: id=5, parent=4
Mode/Strategy: exploit - intermediate mobility magnitude
Config: params_mesh=[[0.1, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 150, -150, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 7/10
Visual: **STRIPE PATTERNS ACHIEVED!** C1/C2 fields develop from dots → connected stripe/labyrinthine structures. Particles form ring-like outlines around high-concentration regions, correlating with field patterns. Dynamic evolution visible across frames.
Metrics: C1_std=1.71 (strong patterns), clustering=0.0395 (4x baseline!), pattern_growth=218.93
Mutation: M1: 50->150, M2: -50->-150 (3x increase from iter 4)
Parent rule: Exploit mobility magnitude search from iter 4
Observation: |M|=150 is the sweet spot! Stable simulation with strong particle-field coupling. Opposite-sign mobilities create net force toward pattern boundaries. Particles outline the Turing structures. First stripe-like patterns instead of just dots.
Next: parent=5, explore further refinement or different parameter dimensions

## Iter 6: 8/10 ★★ IMPROVED
Node: id=6, parent=5
Mode/Strategy: exploit - push mobility higher
Config: params_mesh=[[0.1, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 8/10
Visual: **LABYRINTHINE PATTERNS WITH STRONG PARTICLE ORGANIZATION!** Fields show rich labyrinthine/stripe Turing structures. Particles form clear filamentary networks tracing pattern boundaries - much sharper definition than iter 5. Excellent particle-field correlation across all frames.
Metrics: C1_std=1.89 (stronger than iter 5), clustering=0.0795 (2x iter 5!), pattern_growth=224.23
Mutation: M1: 150->180, M2: -150->-180 (20% increase in mobility magnitude)
Parent rule: Exploit best node (UCB=2.28), increase mobility toward upper limit
Observation: |M|=180 further improves clustering without instability! Approaching upper stability limit (|M|=250 crashed). Clear labyrinthine patterns with excellent particle self-organization. Clustering doubled from iter 5.
Next: parent=6, try |M|=200 to find stability boundary OR explore different parameter dimension (D1/D2 ratio, consumption rate)

## Iter 7: 3/10 (UNSTABLE)
Node: id=7, parent=6
Mode/Strategy: exploit - test stability boundary
Config: params_mesh=[[0.1, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 200, -200, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 3/10
Visual: Before crash, excellent labyrinthine patterns with strong particle networks tracing field boundaries. However, simulation became unstable (all metrics nan). Early frames showed promising filamentary structures similar to iter 6.
Metrics: All nan (numerical instability in later frames)
Mutation: M1: 180->200, M2: -180->-200 (11% increase in mobility magnitude)
Parent rule: Exploit best node (UCB=2.532), test upper stability boundary
Observation: |M|=200 crosses the stability threshold! The feedback becomes too strong. **CONFIRMED: Stability boundary is between |M|=180 and |M|=200.** Must stay at |M|=180 or below. Next should explore different parameter dimensions (D1/D2 ratio, consumption rate, longer simulation).
Next: parent=6 (return to best stable config), explore different parameter dimension

## Iter 8: 9/10 ★★★ EXCEPTIONAL
Node: id=8, parent=6
Mode/Strategy: explore - different parameter dimension (D1 reduction)
Config: params_mesh=[[0.05, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 9/10
Visual: **EXCEPTIONAL PARTICLE NETWORKS!** Fields show beautiful labyrinthine/stripe Turing patterns with strong contrast. Particles form **cyan filamentary networks** tracing field boundaries. Clear evolution from random → organized → stable network structures. Late frames show complex, interconnected particle chains correlating precisely with field topology.
Metrics: clustering=0.1092 (37% improvement over iter 6!), C1_std=2.0150, pattern_growth=219.22
Mutation: D1: 0.1 -> 0.05 (50% reduction in C1 diffusion)
Parent rule: Explore orthogonal parameter dimension from best node (6)
Observation: **MAJOR FINDING** - Reducing D1 while keeping |M|=180 produces even stronger patterns! Lower D1 creates sharper gradients → stronger diffusiophoretic forces. D1/D2 ratio now 0.0625 (was 0.125) → finer pattern wavelength. This is the best result of block 1!
Next: Block 2 - consider code modifications or further D1 exploration

---

# BLOCK 1 SUMMARY

## Key Achievements
- **MILESTONE**: Progressed from uniform particles (4/10) to exceptional filamentary networks (9/10)
- Best configuration: D1=0.05, M1=180, M2=-180, D2=0.8, Da_c=10, A=4.5, B=6.5
- Best metrics: clustering=0.1092, C1_std=2.0150

## Established Principles
1. **Opposite-sign mobilities essential**: M1>0, M2<0 creates differential response to fields
2. **Stability boundary**: |M| ∈ [50, 180] stable; |M|≥200 crashes
3. **Optimal |M|=180**: Maximizes coupling without instability
4. **Lower D1 improves patterns**: D1=0.05 >> D1=0.1 (sharper gradients)
5. **D1/D2 ratio controls wavelength**: 0.0625 produces finer patterns than 0.125

## Score Progression
Iter 1: 4 → Iter 2: 4 → Iter 3: 2 → Iter 4: 4 → Iter 5: 7 → Iter 6: 8 → Iter 7: 3 → Iter 8: 9

---

# BLOCK 2

## Iter 9: 8/10
Node: id=9, parent=8
Mode/Strategy: exploit - push D1 even lower
Config: params_mesh=[[0.025, 10.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]], n_frames=2000, delta_t=5e-4, n_particles=9600, n_nodes=10000
Score: 8/10
Visual: Beautiful labyrinthine Turing patterns in fields. Particles form clear network structures correlating with field topology. Good evolution from random to organized. Stripes/chains visible throughout late frames.
Metrics: C1_std=1.8357, clustering=0.0751, pattern_growth=226.64
Mutation: D1: 0.05 -> 0.025 (50% reduction from best iter 8)
Parent rule: Exploit best node (8), continue D1 reduction direction
Observation: **TOO LOW** - D1=0.025 produces weaker patterns than D1=0.05. Clustering dropped from 0.1092 → 0.0751 (31% decrease). C1_std also dropped. There's an optimal D1 range around 0.05. Going lower creates overly sharp gradients that may destabilize or compress patterns. D1=0.05 remains optimal.
Next: parent=8, explore different dimension (Da_c, longer n_frames, or higher mesh resolution)

---

