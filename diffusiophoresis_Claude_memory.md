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
| Diffusiophoresis_Mesh | Brusselator | Prigogine (1968) | retired | 7/10 (iter 51) |
| Diffusiophoresis_Mesh_GrayScott | Gray-Scott | Pearson (1993) | **ACTIVE** | - |

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
**Key Results**:
- Created PDE_Diffusiophoresis_GrayScott.py (Pearson 1993)
- U field showed EXCELLENT spot/labyrinthine patterns (γ-worm regime)
- Particles formed beautiful web structures tracing boundaries (visible frames 3-5 in iters 69-71)
- **FATAL FLAW**: Particle consumption (even at 18) overwhelms Gray-Scott's slow UV² production
- V field went deeply negative in ALL runs, causing anti-clustering then NaN
- time_scale=5.0 helped (V improved 30%) but time_scale=10.0 caused NaN
**Conclusion**: Gray-Scott coupling requires EITHER V-clamping OR inverted coupling (consume U instead of V)
**Block Statistics**: Average 1.125/10, Best 2/10 (iters 65, 71)

---

## Current Block (Block 10)

### Block Info
- Starting iteration: 73
- Iterations: 73-80
- Focus: **Fix Gray-Scott V-field instability via code modification**

### Hypothesis
Block 9 showed Gray-Scott produces EXCELLENT patterns and particles DO trace boundaries correctly, but V goes negative causing collapse. Two possible fixes:

**Option A: V-clamping (code modification)**
- Add `V = torch.clamp(V, min=0)` or `V = torch.relu(V)` in forward()
- Physical justification: concentration cannot be negative
- Risk: May mask underlying instability

**Option B: Inverted coupling (config change)**
- Instead of particles consuming V, have them consume U (which is replenished by feed F)
- Swap: consumption affects U, production affects V
- Physical: more stable since F*(1-U) replenishes U continuously

**Decision**: Try Option A first (V-clamping) - quickest test to validate if positive V enables the promising particle webs to persist. If that fails or produces artifacts, try Option B.

### Code Modification Plan
**File**: `src/ParticleGraph/generators/PDE_Diffusiophoresis_GrayScott.py`
**Change**: Add V-clamping after reaction to prevent negative concentrations
```python
# After computing dV, clamp V to non-negative
V_new = V + dt * dV  # conceptually
V_new = torch.clamp(V_new, min=1e-6)  # prevent negative
```
Note: Actual implementation depends on how time integration is done in graph_data_generator

### Config for iter 73
- Keep γ-worm regime: Du=0.2, Dv=0.1, F=0.035, k=0.06
- time_scale=5.0 (safe, improved V in iter 71)
- M1=-16, M2=+16, consumption=18, production=-18
- V-clamping code modification applied

### Iterations This Block

**Iter 73: 3/10** - V-clamping code mod applied. Visual: excellent U-field spots, but C1_mean=-18.79 (U negative!), C2_mean=-1.26. clustering=0.0923 (very low). V-clamping in forward() insufficient - affects reactions but not stored values. Particle consumption still overwhelms field replenishment.

**Iter 74: 2/10** - Reduced consumption 18→5. Visual: Beautiful hexagonal spot pattern in U-field emerging from noise. V-field shows complementary pattern. BUT metrics WORSE: C1_mean=-18.91, C2_mean=-1.27, clustering=0.0652 (35% below iter 73). Fields still deeply negative despite V-clamping and reduced consumption. ROOT CAUSE: Any particle consumption overwhelms Gray-Scott feed rate F*(1-U). Next: zero consumption test to isolate pure GS dynamics.

**Iter 75: 3/10** - Zero coupling test (consumption=0, production=0). Visual: CRITICAL FINDING - Gray-Scott patterns are BEAUTIFUL and STABLE when uncoupled. Perfect hexagonal spots develop (frames 3-10). BUT particles show ZERO organization - uniformly scattered. Clustering=0.0558 (lowest). This proves: (1) GS model works, (2) coupling is the problem, (3) particles don't respond to gradients even with M1=-16, M2=+16. Next: investigate why particles don't follow gradients - may need higher mobility or gradients may be in wrong index.

**Iter 76: 2/10** - 3x mobility test (M1=-48, M2=+48). Visual: Perfect hexagonal GS spot pattern (same as iter 75). Particles STILL uniformly scattered with NO clustering response. Metrics: clustering=0.0501 (lowest yet, 10% below iter 75). C1_mean=-18.92, C2_mean=-1.27 still deeply negative. ROOT CAUSE ANALYSIS: Gray-Scott gradients are MUCH WEAKER than Brusselator (U in [0,1] vs C1 in [0,20]), plus PDE_D applies domain_scale=32 reduction. Combined with weak GS gradients, effective mobility ~30-60x too low. Next: extreme mobility test M=±500 to verify coupling mechanism actually works.

**Iter 77: 2/10** - EXTREME mobility test (M1=-500, M2=+500, 10x increase). Visual: Beautiful hexagonal GS spots (stable, consistent). Particles show SLIGHT improvement: clustering 0.0501→0.0818 (+63%). Still essentially scattered but confirms coupling IS working. CODE ANALYSIS: PDE_D line 143 divides all gradients by domain_scale=32. For GS (amplitude ~1) vs Brusselator (~20), effective gradient is 640x weaker. Need M~10000 OR remove domain_scale at block end. Next: try M=±1500 OR prepare code fix.

