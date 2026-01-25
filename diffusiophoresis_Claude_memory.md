# Working Memory: diffusiophoresis

## Knowledge Base

### Pattern Principles
- D2/D1 ratio controls pattern wavelength; ratio 1-16 all produce labyrinthine in current regime
- Turing condition: B > 1+A² for instability (A=1.5 → B > 3.25)
- A=1.5, B=5.0-7.0 optimal for multi-scale nested patterns (7-8/10)
- Da_c=25 required for particle organization; lower Da_c causes anti-clustering
- Particle organization correlates with field gradients - halos/traces form at concentration boundaries
- Cross-diffusion χ breaks labyrinthine → cruciform, but cruciform is eigenmode-locked
- Higher B (deeper Turing) increases contrast and pattern complexity
- **n_frames=4000 critical** - longer simulation enables emergent particle asymmetry
- **Sweet spot**: M=±16, consumption=180 - higher values cause diffusive mixing that degrades organization
- **Particle density matters**: n_particles=9600 required for collective feedback; 6400 too sparse
- **EIGENMODE LOCK**: 4-fold cruciform is GEOMETRY-locked (square domain + periodic BC), NOT diffusion-locked
- **NEW - D1 CONTROLS EIGENMODE SELECTION**: D1=0.015-0.02 breaks cruciform lock by selecting higher-order modes with complex topology

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

### Code Insights
- Block 2: Added cross-diffusion term χ to Brusselator for advective coupling
- Cross-diffusion coefficient χ controls chemotaxis-like drift of C1 toward/away from C2
- Cruciform pattern is 4-fold eigenmode locked by square domain + periodic BC
- Block 3: cos(2πy) position-dependent aniso ineffective - wavelength mismatch
- Block 5: TRUE TENSOR ANISOTROPY (D1_x ≠ D1_y) also failed - eigenmode is geometry-locked
- Block 6: Stochastic noise term added but FAILED to break symmetry across full amplitude range
- **FUNDAMENTAL INSIGHT**: To break 4-fold symmetry, need to modify WAVELENGTH via D1, not noise/anisotropy

---

## Previous Block Summaries

### Block 1 (Iterations 1-8)
- Started at 5/10 (baseline spots), achieved 7/10 (labyrinthine)
- Key discovery: A=1.5 unlocks labyrinthine by satisfying Turing condition

### Block 2 (Iterations 9-16)
- Added cross-diffusion χ term to Brusselator (code modification)
- χ broke labyrinthine → cruciform pattern (4-fold symmetric)
- **BEST RESULTS**: B=5.0-6.0 with χ=0 produced multi-scale nested patterns (8/10)

### Block 3 (Iterations 17-24)
- Position-dependent aniso diffusion FAILED - cos(2πy) wavelength mismatch
- B=7.0 adds internal complexity but cruciform 4-fold symmetry persists
- **KEY FINDING**: Longer simulation (n_frames=4000) enables EMERGENT PARTICLE ASYMMETRY!
- **BEST EVER**: Iter 20 achieved clustering=0.59 (RECORD), pos_std_y/x=1.34 (spontaneous symmetry breaking)

### Block 4 (Iterations 25-32)
- Tested ar_p1, sigma, B=8.0 (crash), Da_c=35 (crash), D1=0.04, n_nodes=22500
- **KEY FINDING**: Iter 20's asymmetry is STOCHASTIC, not reproducible via parameters
- **Stability limits confirmed**: B≤7.0, Da_c≤30 (25 optimal)

### Block 5 (Iterations 33-40)
- **CODE MOD**: Implemented true tensor anisotropy D1_x ≠ D1_y
- **FAILED**: Tensor aniso ratios 0.5, 0.25, 0.1 all produce cruciform - eigenmode is geometry-locked
- **FAILED**: Multi-type particles (same-sign mobilities → co-localize; opposing mobilities → NaN)
- **Block average**: 4.4/10 (2 NaN explosions)

### Block 6 (Iterations 41-48)
- **CODE MOD**: Added stochastic noise term dC1 += noise_amplitude * torch.randn_like(C1)
- **NOISE FAILED**: Amplitudes 0.01→0.05→0.3→1.0 ALL failed to break symmetry
- **CRITICAL**: Noise DEGRADES clustering (disabling noise improved clustering 0.33→0.42)
- **BREAKTHROUGH**: D1 reduction (0.05→0.02) BROKE cruciform lock!
- **Mechanism**: Smaller D1 → shorter wavelength → higher-order eigenmodes with complex topology
- **Optimal D1**: ~0.015-0.02 balances complexity vs clustering
- **Block average**: 5.4/10 (improvement from 4.4/10)

---

## Current Block (Block 7)

### Block Info
- Starting iteration: 49
- Iterations: 49-56
- Focus: Build on D1 breakthrough - enhance clustering while maintaining labyrinthine topology

### Hypothesis
With D1=0.015-0.02 breaking the cruciform eigenmode lock, we can now enhance particle clustering by:
1. Increasing particle-field coupling (M, consumption) - particles respond more strongly to field gradients
2. Fine-tuning D2/D1 ratio - balance between field dynamics and particle response time
3. Exploring intermediate Da_c values - faster reaction may enhance gradient sharpness

Literature: Diffusiophoretic accumulation scales with mobility × gradient strength (Prieve 2010). With more complex field topology, stronger mobility should create more coherent clustering at field boundaries.

### Code Modification Plan (Block 7)
**Option A: Particle-field coupling enhancement**
- Increase M from ±16 to ±20-24 with D1=0.015 baseline
- Risk: Higher M previously caused diffusive mixing (at D1=0.05)
- Hypothesis: With complex labyrinthine topology (D1=0.015), stronger M may create sharper clustering

**Option B: D2 adjustment for optimal D1/D2 ratio**
- Currently D2=0.2, D1=0.015 → ratio D2/D1=13.3
- Try D2=0.1 for ratio 6.7 → may sharpen C2 gradients

**Option C: Traveling wave regime**
- Modify Brusselator to add advection term for traveling waves
- dC1 += v · ∇C1 with constant velocity v

**Selected approach for Block 7**: Start with parameter exploration (M, D2, Da_c) using D1=0.015 baseline before attempting code changes.

### Iterations This Block

**Iter 49: 5/10** - CRUCIFORM RETURNED with increased M=±20, consumption=±220
- Config: D1=0.015, M=±20, consumption=±220
- Clustering: 0.206 (DROPPED from 0.39)
- Observation: Higher mobility DESTABILIZED complex labyrinthine → system relaxed to simpler 4-fold eigenmode
- Action: Revert M=±16, consumption=180 to test D1=0.015 alone

### Emerging Observations
- **HIGH MOBILITY IS HARMFUL** - M=±20 and consumption=220 caused pattern simplification
- D1=0.015 alone may still be effective; need to test with original M/consumption values
- The D1 breakthrough from block 6 requires LOWER mobility to maintain effect

