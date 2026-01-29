# Working Memory: diffusiophoresis

## Regime Comparison

| Regime | mesh_model | particle_model | n_types | n_particles | Best Score | Key Insight |
| ------ | ---------- | -------------- | ------- | ----------- | ---------- | ----------- |
| Base   | Diffusiophoresis_Mesh | PDE_ParticleField_D | 1 | 9600 | 9/10 | M1=-4, M2=4 gives ring->hexagonal patterns |
| Multi-2  | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 9/10 | Opposing mobilities -> phase separation + hexagonal mode |
| Multi-3  | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3 | 9600 | 9/10 | Tri-layer concentric rings (germ layer analog) |
| GrayScott-2 | PDE_Diffusiophoresis_GrayScott | PDE_ParticleField_D | 2 | 9600 | 7/10 | Concentric rings (no hexagonal breaking), high entropy |
| FHN-2 | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D | 2 | 9600 | 8/10 | Finer scale (~30 spots), network organization |
| FHN-3 | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D | 3 | 9600 | 9/10 | Hexagonal with multi-layer spots |
| FHN-Dv | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D | 1 | 9600 | 9/10 | **Dv=0.1-0.15 → SQUARE/GRID symmetry (novel!)** |
| Stripes | Diffusiophoresis_Mesh | PDE_ParticleField_D | 1 | 9600 | 9/10 | **A=3, B=10 (B/[1+A^2]=1) -> LABYRINTH** |
| Mixed | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 8/10 | **A=2.828, B=9 -> transitional mixed mode** |
| LogSens-FHN | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D_LogSensing | 1 | 9600 | 8/10 | **LogSensing+FHN Dv=0 → SQUARE (effective Dv>0)** |
| LogSens-FHN-Dv | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D_LogSensing | 2 | 9600 | 8/10 | **Dv=0.2+LogSensing → sharpest square + cell wall** |

## Insights

| Category    | Finding                                              |
| ----------- | ---------------------------------------------------- |
| Patterns    | Brusselator params (Da_c=15, A=4.5, B=6.5) -> hexagonal; (A=3, B=10) -> stripes |
| Performance | M1=-4, M2=4 achieves 100% particle retention with strong coupling |
| Symmetry    | **B/[1+A^2]=1 + B=10,A=3 -> stripes; B=13,A=3.46 -> spots (A matters!); B/[1+A^2]>1 -> hexagonal** |
| Multi-type  | Opposing mobilities (+/-4) create spontaneous phase separation |
| Gray-Scott  | F=0.035-0.04, k=0.06-0.065 produce concentric rings, NOT hexagonal |
| Failures    | M1=-16, M2=16 causes particle escape (0% in box)     |
| FHN vs Brus | Brusselator=tight clusters (0.45-0.53), FHN=dispersed (0.26-0.39) |
| Robustness  | **Hex survives ALL perturbations. Stripe=1-type-only (fails multi-type & reversed). Square=same-sign-only** |
| Transition  | **B=9 at Turing boundary produces MIXED mode (not pure hex or stripe)** |
| FHN-Dv     | **FHN Dv≥0.1 → SQUARE/GRID symmetry; Dv=0.05 → disordered transitional; Dv=0 → hexagonal** |
| Square req | **Square symmetry requires uniform mobility DIRECTION (not 1-type). Opposing mobilities destroy square→hexagonal** |
| Deep-Turing | **B=15 with B/[1+A^2]=0.71 → pixel-scale noise. B≤13 practical limit for 100×100 mesh** |
| Stripe req  | **Stripes need 1-type uniform particles + low A (≤3). Multi-type (even identical) destroys stripes** |
| LogSensing  | **LogSensing Brusselator-incompatible (3 failures), FHN-compatible (transforms hex→square)** |
| LogSens mech | **LogSensing acts as effective Dv>0 via concentration-dependent mobility. Additive with FHN Dv.** |
| Cell wall   | **2-type same-sign + diff consumption → boundary/interior type segregation (iter 64)** |

---

## Knowledge Base

### Established Principles
1. **Mobility sweet spot**: M1=+/-4, M2=+/-4 prevents escape while preserving field-particle coupling (confirmed 64 iterations)
2. **Opposing mobilities -> phase separation**: Type A (M1=-4, M2=4) vs Type B (M1=4, M2=-4) creates spontaneous spatial segregation (confirmed 12+ iterations)
3. **Extended simulation reveals mode selection**: Ring->hexagonal symmetry breaking occurs at 4000-6000 frames (confirmed 12+ iterations)
4. **100% particle retention**: All configs with M1=+/-4 or +/-8 achieve 100% particles_in_box (confirmed all iterations since iter 2)
5. **Mode selection is particle-type-independent**: 1-type, 2-type, AND 3-type all achieve identical patterns when field dynamics are fixed (confirmed 15+ iterations)
6. **Brusselator vs Gray-Scott dichotomy**: Brusselator achieves hexagonal symmetry breaking; Gray-Scott stays purely radial (confirmed iter 14)
7. **Brusselator vs FHN dichotomy**: Both achieve hexagonal, but Brusselator=tight clusters (0.45-0.53), FHN=dispersed/network (0.26-0.39) (confirmed iters 17-24)
8. **ar_p particle-particle params have minimal effect**: Doubling ar_p (1.6->3.0) did not improve clustering metric (confirmed iter 23)
9. **Turing boundary controls mode selection**: B/[1+A^2]=1.0 (at boundary) -> stripes/labyrinth; B/[1+A^2]>1 (deep unstable) -> hexagonal spots (confirmed iters 31-40)
10. **Sub-Turing = NO patterns**: B/[1+A^2]<1 produces disordered/noisy fields with no particle organization (confirmed iter 34)
11. **Stripe mode requires BOTH low A (≤3) AND 1-type uniform particles**: Multi-type (even identical) destroys stripes (confirmed iters 49, 54-55)
12. **Schnakenberg = radial symmetry only**: Like Gray-Scott (confirmed iters 41, 43, 44)
13. **FHN Dv is a symmetry selector**: Dv=0 → hexagonal spots, Dv=0.05 → disordered transitional, Dv≥0.1 → SQUARE/GRID symmetry. Square requires uniform mobility DIRECTION (confirmed iters 45-53)
14. **FHN square is intrinsically oscillatory**: All FHN Dv≥0.1 configs plateau=0.00 regardless of feedback (confirmed iters 51-53, 60-64)
15. **Deep-Turing is noise**: B=15 produces pixel-scale patterns too fine for particle organization. B≤13 practical limit for 100×100 mesh (confirmed iter 56)
16. **Robustness hierarchy**: Hexagonal (survives multi-type, reversed feedback, shuffling) > Square (survives same-sign multi-type, shuffling; fails opposing mobility) > Stripe (1-type uniform only)
17. **LogSensing is mesh-model selective**: Brusselator-incompatible (suppresses hex at M=4, noisy at M=8), FHN-compatible and transformative (produces square from hex regime) (confirmed iters 57-64)
18. **LogSensing acts as effective Dv**: Concentration-dependent mobility (∇C/C) creates implicit v-diffusion, shifting FHN Dv=0 hex→square. Additive with explicit FHN Dv (confirmed iters 60-64)
19. **Differential consumption → type-boundary segregation**: In square mode, lower-consumption type concentrates at boundaries, higher at interior — "cell wall" morphology (confirmed iter 64)

### Open Questions
- Can **active matter (self-propulsion)** in PDE_D create new collective dynamics (flocking, polar order)?
- Does the particle model change the set of ACCESSIBLE symmetry classes (beyond square)?
- What happens at very large FHN Dv (Dv=0.3, 0.5)? Does square persist or transition?
- Can velocity alignment (Vicsek model) create new pattern types not accessible via diffusiophoresis?
- Does LogSensing + Brusselator work at very low M (M=±2)?

### Answered Questions (Blocks 1-8)
- **Does hexagonal mode require 3 particle types?** NO — 1, 2, 3-type all work
- **Does Gray-Scott F,k regime change behavior?** F=0.035-0.04, k=0.06-0.065 all produce rings
- **Is the 9/10 result reproducible?** YES — robustness tests confirm
- **Does FHN achieve hexagonal?** YES — but with finer spatial scale
- **Do ar_p params affect clustering?** NO — minimal effect
- **Can chi, D1/D2, domain size break hexagonal?** NO
- **How to break hexagonal attractor?** A=3, B=10 (B/[1+A^2]=1) produces stripes!
- **Can Schnakenberg produce different patterns?** NO — radial only
- **Does FHN square + 3-type work?** YES — requires same-sign mobility
- **Does opposing mobility destroy square?** YES
- **Can uniform feedback stabilize FHN square (plateau>0)?** NO — intrinsically oscillatory
- **Can stripe mode work with multi-type?** NO — 1-type only
- **Does deep Turing (B=15) help?** NO — pixel-scale noise
- **Does LogSensing change pattern selection?** YES — hex→square with FHN, suppresses Brusselator
- **Are LogSensing and FHN Dv multiplicative?** NO — additive (same mechanism)
- **Does LogSensing square follow same robustness rules?** YES — identical to FHN Dv square

### Failed Configurations
- M1=-16, M2=16: particles escape rapidly
- Gray-Scott: cannot achieve hexagonal regardless of params
- Schnakenberg: cannot achieve hexagonal regardless of gamma
- ar_p, chi, D1/D2, domain size changes: do not affect mode selection
- FHN Dv=0.1 + opposing mobility: destroys square symmetry
- Deep-Turing (B=15): pixel-scale noise
- LogSensing + Brusselator (any M): no clean patterns (radial or noisy)

### Code Insights
- Base Brusselator (Diffusiophoresis_Mesh) works well for hex and stripe modes
- Gray-Scott and Schnakenberg produce only radial symmetry
- FHN variant with Dv parameter is a symmetry selector (hex/square)
- **PDE_D_LogSensing**: First particle model variant. Acts as effective Dv>0 with FHN. Brusselator-incompatible.
- Dynamic class loading: new PDE_D variants auto-discovered from filename (no registration needed)
- **Block 9 creates PDE_D_ActiveMatter**: Self-propelled particles with velocity alignment (Vicsek model)

### PDE Variants

| Variant | Model | Literature | Status | Best Score | Symmetry |
| ------- | ----- | ---------- | ------ | ---------- | -------- |
| Diffusiophoresis_Mesh | Brusselator | Prigogine (1968) | active | 9/10 | hexagonal OR stripes |
| PDE_Diffusiophoresis_GrayScott | Gray-Scott | Pearson (1993) Science 261 | tested | 7/10 | radial only |
| PDE_Diffusiophoresis_FHN | FitzHugh-Nagumo | FitzHugh (1961) | active | 9/10 | hexagonal OR square |
| PDE_Diffusiophoresis_Schnakenberg | Schnakenberg | Schnakenberg (1979) JTB 81:389 | tested | 7/10 | radial only |
| PDE_ParticleField_D | Linear diffusiophoresis | Base | active | 9/10 | (depends on mesh) |
| PDE_ParticleField_D_LogSensing | Log-sensing chemotaxis | Kalinin (2009) Biophys J | active | 8/10 | square (with FHN) |
| PDE_ParticleField_D_ActiveMatter | Active matter self-propulsion | Vicsek (1995) PRL | active | 7/10 | disordered spots |

---

## Previous Block Summaries

**Block 1 (Iters 1-8):** Established baseline, mobility sweet spot M=±4. Score avg: 7.6.

**Block 2 (Iters 9-16):** Gray-Scott PDE variant (radial only), particle-type independence. Score avg: 7.9.

**Block 3 (Iters 17-24):** FHN mesh model (finer hex), ar_p minimal effect. Score avg: 8.5.

**Block 4 (Iters 25-32):** BREAKTHROUGH: A=3, B=10 → stripes. Score avg: 8.1.

**Block 5 (Iters 33-40):** Turing bifurcation mapping. Score avg: 7.9.

**Block 6 (Iters 41-48):** Schnakenberg (radial), BREAKTHROUGH: FHN Dv=0.1 → square. Score avg: 7.4.

**Block 7 (Iters 49-56):** Turing boundary multi-dimensional. Stripe=1-type-only. Deep-Turing=noise. Score avg: 7.25.

**Block 8 (Iters 57-64):** First PDE_D modification (LogSensing). Brusselator-incompatible. FHN-compatible: LogSensing acts as effective Dv>0, producing square from hex regime. Additive with FHN Dv. "Cell wall" morphology with 2-type same-sign. Score avg: 7.125.

---

## Current Block (Block 9)

### Block Info
Parameters: Various mesh models + PDE_ParticleField_D_ActiveMatter (NEW) and continued exploration
Iterations: 65 to 72
Focus: Test second PDE_D variant (active matter / self-propulsion) + explore remaining open questions

### Hypothesis
Block 9 introduces the second particle model modification: **self-propelled active matter** (Vicsek model).

**Physics**: Real cells and bacteria are self-propelled, not just passively advected by gradients. Active matter adds:
1. **Self-propulsion**: Each particle has an intrinsic velocity (speed v0) in its heading direction
2. **Velocity alignment**: Nearby particles tend to align their headings (Vicsek interaction)
3. **Gradient-biased heading**: Field gradients bias heading rotation (chemotaxis meets self-propulsion)

This creates fundamentally different dynamics: particles can form **polar-ordered states** (flocking, bands, vortices) that are inaccessible with purely gradient-following diffusiophoresis. The key question is whether combining active matter self-propulsion with Turing reaction-diffusion fields produces new symmetry classes beyond hex/stripe/square.

**Plan**:
- Iters 65-66: ActiveMatter + Brusselator hex (1-type, then 2-type)
- Iters 67-68: ActiveMatter + Brusselator stripe (A=3, B=10)
- Iters 69-70: ActiveMatter + FHN hex/square
- Iters 71-72: Best ActiveMatter config with 3-type or parameter exploration

### Iterations This Block

**Iter 65** (7/10): ActiveMatter + Brusselator hex, 1-type. v0=0.5, alignment=0.3, gradient_bias=0.5, noise=0.1.
  Metrics: entropy=0.57, plateau=0.01, in_box=100%, clustering=0.52. Disordered irregular spots, never reaches steady state.
  Key: Self-propulsion keeps particles permanently moving (plateau=0.01). Spot morphology elongated/oblong vs standard circular.
  **Active matter signature: persistent motion + irregular morphology.** Lower entropy than standard hex.

**Iter 66** (7/10): ActiveMatter + Brusselator hex, 2-type (differential gradient_bias). Type 0: v0=0.3, gradient_bias=0.8 (gradient-driven). Type 1: v0=1.0, gradient_bias=0.1 (self-propelled).
  Metrics: entropy=0.75, plateau=0.00, in_box=100%, clustering=0.19. C1_std=1.62, pattern_growth=166.
  Key: **Type segregation!** Gradient-driven (orange) forms diffuse clusters, self-propelled (blue) forms tight MIPS-like clumps. No new symmetry class — still irregular spots. Stronger fields than 1-type (pattern_growth 166 vs 115). Lower clustering because orange dispersed.

**Iter 67** (6/10): ActiveMatter + Brusselator STRIPE mode (A=3, B=10), 1-type. v0=0.5, alignment=0.3, gradient_bias=0.7.
  Metrics: entropy=0.64, plateau=0.005, in_box=100%, clustering=0.54. C1_std=3.01, pattern_growth=431.7.
  Key: **Stripes completely destroyed by ActiveMatter** → ring/shell morphology with interior fragmentation. Dense outer boundary with internal holes. C1 field shows pixel-scale checkerboard noise (not clean Turing stripes). Confirms stripe mode is MOST fragile symmetry — fails with multi-type, reversed feedback, AND active matter.

### Emerging Observations
**CRITICAL: This section must ALWAYS be at the END of memory file. When adding new iterations, insert them BEFORE this section.**

- ActiveMatter always produces plateau≈0 (persistent motion). This is fundamental — self-propulsion prevents steady state.
- ActiveMatter disrupts BOTH hexagonal and stripe ordering. No new symmetry class produced.
- Iter 67: Stripe-regime (A=3,B=10) + ActiveMatter → ring/shell (NOT stripes). Very high pattern_growth but into pixel-noise.
- 2-type differential gradient_bias creates MIPS-like behavior: self-propelled particles form tight clumps (iter 66).
- **Emerging pattern**: ActiveMatter produces ring/shell morphology regardless of field regime. Self-propulsion creates circulation → boundary accumulation.
- Next: (1) ActiveMatter + FHN hex and square (iters 68-69), (2) Switch away from ActiveMatter if no new patterns emerge — explore remaining open questions with standard PDE_D.
- Key question evolving: ActiveMatter may only DISRUPT existing symmetries without creating new ones. Consider abandoning ActiveMatter exploration early if iter 68 confirms this.

