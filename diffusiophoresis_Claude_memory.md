# Working Memory: diffusiophoresis

## Knowledge Base

### Pattern Principles
- D2/D1 ratio controls pattern wavelength; ratio 1-16 all produce labyrinthine in current regime
- Turing condition: B > 1+A² for instability (A=1.5 → B > 3.25)
- A=1.5, B=7.0 optimal for multi-scale nested patterns (7-8/10)
- Da_c=25 required for particle organization; lower Da_c causes anti-clustering
- Particle organization correlates with field gradients - halos/traces form at concentration boundaries
- Cross-diffusion χ breaks labyrinthine → cruciform, but cruciform is eigenmode-locked
- Higher B (deeper Turing) increases contrast and pattern complexity
- **n_frames=4000 critical** - longer simulation enables emergent particle asymmetry
- **Sweet spot**: M=±16, consumption=180 - higher values cause diffusive mixing that degrades organization
- **Particle density matters**: n_particles=9600 required for collective feedback; 6400 too sparse
- **EIGENMODE LOCK**: 4-fold cruciform is GEOMETRY-locked (square domain + periodic BC), NOT diffusion-locked
- **D1 CONTROLS EIGENMODE SELECTION**: D1=0.015-0.02 breaks cruciform lock by selecting higher-order modes with complex topology
- **D2=0.15 OPTIMAL**: D2/D1 ratio of 10 provides sharpest useful gradients; ratio 6.7 (D2=0.1) overshoots causing oscillatory dynamics
- **B=7.0 OPTIMAL**: B=6.5 weakens Turing instability, reducing gradient sharpness
- **LINEAR MOBILITY OPTIMAL**: v ∝ ∇C is correct; any nonlinearity (saturation, boost) disrupts boundary accumulation

### Failed Configurations
- Multi-type with extreme opposing mobilities (M=±20, consumption=±200) → NaN explosion
- High D2/D1 ratio (16) produces isolated spots, not interesting
- χ from -0.1 to 5.0 (50x range) all produce same cruciform topology
- Da_c=10 causes negative clustering (-0.24) - particles anti-aggregate
- delta_t=0.001 (double standard) → NaN explosion (CFL violation)
- cos(2πy) anisotropy at ANY strength (0.3-0.8) - wavelength mismatch with patterns
- Higher mobility (±24) destroys emergent asymmetry via diffusive mixing
- Higher consumption (240) destroys emergent asymmetry via diffusive mixing
- Lower particle count (6400) weakens collective feedback, inverts asymmetry
- **B=8.0 → NaN EXPLOSION** - Turing instability too strong, numerical blow-up (B=7.0 is safe limit)
- **Da_c=35 → NaN EXPLOSION** - reaction overwhelms diffusion with B=7.0
- ar_p1 increase (1.6→2.0) did NOT improve clustering - ar_params likely inactive with n_particle_types=1
- sigma increase (0.005→0.01) ALSO counterproductive - caused more spreading
- n_nodes=22500 (finer mesh) no improvement - eigenmode-locked pattern unchanged
- **TRUE TENSOR ANISOTROPY FAILED** - ratios 0.5, 0.25, 0.1 all produce cruciform (eigenmode from geometry, not diffusion)
- **Multi-type opposing mobilities → NaN** (full or partial, both unstable)
- **Asymmetric M1/M2 ratio** - no topology change, just mild redistribution
- **NOISE FAILED COMPREHENSIVELY** - amplitudes 0.01→1.0 all failed to break eigenmode, noise DEGRADES clustering
- **D1=0.01 too low** - wavelength too fine, particles spread across many gradients, clustering drops
- **D2=0.1 OVERSHOT** - D2/D1 ratio 6.7 too sharp, causes oscillatory dynamics not stable clustering
- **B=6.5 WORSE** - shallower Turing reduces gradients, clustering dropped 51%
- **Saturation mobility FAILED** - scale 2.0→0.0, all worse than linear
- **Boost mobility CATASTROPHIC** - exponent 0.5 caused 62% clustering collapse

### Code Insights
- Block 2: Added cross-diffusion term χ to Brusselator for advective coupling
- Cross-diffusion coefficient χ controls chemotaxis-like drift of C1 toward/away from C2
- Cruciform pattern is 4-fold eigenmode locked by square domain + periodic BC
- Block 3: cos(2πy) position-dependent aniso ineffective - wavelength mismatch
- Block 5: TRUE TENSOR ANISOTROPY (D1_x ≠ D1_y) also failed - eigenmode is geometry-locked
- Block 6: Stochastic noise term added but FAILED to break symmetry across full amplitude range
- **FUNDAMENTAL INSIGHT**: To break 4-fold symmetry, need to modify WAVELENGTH via D1, not noise/anisotropy
- **Block 8**: Nonlinear mobility modifications (saturation + boost) BOTH FAILED - LINEAR v∝∇C is OPTIMAL

### PDE Variants

| Variant | Model | Literature | Status | Best Score |
|---------|-------|------------|--------|------------|
| Diffusiophoresis_Mesh | Brusselator | Prigogine (1968) | **ACTIVE** | 7/10 (iter 51) |
| Diffusiophoresis_Mesh_GrayScott | Gray-Scott | Pearson (1993) | retired | 3/10 (incompatible gradients) |

---

## Previous Block Summaries

### Block 1-6 Summary
- Block 1: Achieved labyrinthine (7/10) by satisfying Turing condition (A=1.5)
- Block 2: Cross-diffusion χ broke labyrinthine → cruciform; B=5-6 without χ achieved 8/10
- Block 3: Position-dependent aniso FAILED; discovered EMERGENT ASYMMETRY at n_frames=4000 (iter 20: clustering=0.59)
- Block 4: Confirmed iter 20's asymmetry is STOCHASTIC; stability limits B≤7.0, Da_c≤30
- Block 5: Tensor anisotropy FAILED; multi-type particles FAILED (opposing mobilities → NaN)
- Block 6: Noise FAILED comprehensively; **D1 BREAKTHROUGH** - D1=0.015-0.02 breaks cruciform lock

### Block 7 (Iterations 49-56)
**Goal**: Enhance clustering while maintaining complex labyrinthine topology from D1 breakthrough.
**Key Results**: D2=0.15 optimal (clustering=0.485), parameter space exhausted
**Block Statistics**: Average 5.75/10, Best 7/10 (iter 51)

### Block 8 (Iterations 57-64)
**Goal**: Test nonlinear mobility modifications to break clustering plateau.
**Key Results**:
- Saturation approach (scale 2.0→0.0): ALL WORSE than linear baseline
- Boost approach (exponent=0.5): CATASTROPHIC collapse (clustering 0.1854, 62% below baseline)
- Linear restored (iters 63-64): Recovered to 0.41-0.43
- M=±18 test: Slight spreading, no improvement
**Conclusion**: LINEAR MOBILITY IS OPTIMAL - any nonlinearity disrupts boundary accumulation
**Block Statistics**: Average 5.375/10, Best 6/10

### Block 9 (Iterations 65-72)
**Goal**: Test Gray-Scott PDE variant for richer pattern dynamics.
**Key Results**: Created GrayScott variant, beautiful hexagonal patterns but V went negative causing collapse.
**Block Statistics**: Average 1.125/10, Best 2/10

### Block 10 (Iterations 73-80)
**Goal**: Fix Gray-Scott coupling via V-clamping and mobility optimization.
**Key Results**:
- V-clamping code mod helped stabilize but didn't solve fundamental issue
- Zero-coupling test (iter 75): GS patterns STABLE and BEAUTIFUL when uncoupled
- Mobility sweep: M=500→clustering 0.0818, M=1000→0.0277, M=2000→0.0084 (nonlinear, optimal near M=500)
- **ROOT CAUSE**: PDE_D divides gradients by domain_scale=32; GS fields ~1 vs Brusselator ~20 = 20-30x weaker effective gradients
- Architecture mismatch prevents proper particle coupling without extreme mobility (which causes oscillation)
**Conclusion**: Gray-Scott RETIRED - beautiful patterns but fundamentally incompatible with current PDE_D architecture. Return to Brusselator.
**Block Statistics**: Average 2.375/10, Best 3/10 (iters 73, 75, 79, 80)

---

## Current Block (Block 11)

### Block Info
- Starting iteration: 81
- Iterations: 81-88
- Focus: **Return to Brusselator - pursue STRIPE patterns via D1/D2 ratio tuning**

### Hypothesis
**Literature**: Stripes vs spots in Turing systems depend on D1/D2 ratio (Ouyang & Swinney 1991, Cross & Hohenberg 1993):
- Spots: D1/D2 << 1 (current ratio 0.1 with D1=0.015, D2=0.15)
- Stripes: D1/D2 ≈ 0.3-0.5 (higher diffusion ratio)

Previous best (iter 51): D1=0.015, D2=0.15, clustering=0.485, labyrinthine (but 4-fold symmetry)
Knowledge base: D1=0.015-0.02 breaks cruciform lock by selecting higher-order eigenmodes

**New approach**: Increase D1 to shift toward stripe regime while maintaining Turing instability.
- Try D1=0.05, D2=0.15 (ratio 0.33) - should favor stripes over spots
- Keep B=7.0, Da_c=25 (established optimal)
- Return to standard mobility M=±16, consumption=180

### Config for iter 81
- mesh_model_name: Diffusiophoresis_Mesh (Brusselator)
- D1=0.05, D2=0.15 (ratio 0.33 for stripe regime)
- A=1.5, B=7.0 (established optimal)
- Da_c=25, M1=-16, M2=+16, consumption=180, production=-180

### Iterations This Block
(Block 11 starts next iteration)

