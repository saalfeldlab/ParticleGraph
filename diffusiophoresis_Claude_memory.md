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
| FHN-Dv | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D | 1 | 9600 | 9/10 | **Dv=0.1-0.15 -> SQUARE/GRID symmetry (novel!)** |
| Stripes | Diffusiophoresis_Mesh | PDE_ParticleField_D | 1 | 9600 | 9/10 | **A=3, B=10 (B/[1+A^2]=1) -> LABYRINTH** |
| Mixed | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 8/10 | **A=2.828, B=9 -> transitional mixed mode** |
| LogSens-FHN | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D_LogSensing | 1 | 9600 | 8/10 | **LogSensing+FHN Dv=0 -> SQUARE (effective Dv>0)** |
| LogSens-FHN-Dv | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D_LogSensing | 2 | 9600 | 8/10 | **Dv=0.2+LogSensing -> sharpest square + cell wall** |
| GM | PDE_Diffusiophoresis_GM | PDE_ParticleField_D | 1 | 9600 | 7/10 | Foam-like network, always unstable (plateau=0) |
| Da_c=5-2type | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 7/10 | **Da_c=5 -> COARSE NETWORK (Voronoi-like, novel!)** |
| Da_c=10-2type | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 8/10 | **Da_c=10 -> MIXED LABYRINTHINE (plateau=0.46)** |
| Anisotropic | Diffusiophoresis_Mesh | PDE_ParticleField_D_Anisotropic | 1 | 9600 | 8/10 | **alpha=0.25 -> ORIENTED STRIPES from hex regime (novel!)** |

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
| Robustness  | **Hex survives ALL perturbations. Stripe=1-type-only (fails multi-type & reversed). Square=1-type-only** |
| Transition  | **B=9 at Turing boundary produces MIXED mode (not pure hex or stripe)** |
| FHN-Dv     | **FHN Dv>=0.1 -> SQUARE/GRID symmetry; Dv=0.05 -> disordered transitional; Dv=0 -> hexagonal** |
| Square req | **Square requires SINGLE particle type (spatial homogeneity). Even identical-param multi-type fails** |
| Deep-Turing | **B=15 with B/[1+A^2]=0.71 -> pixel-scale noise. B<=13 practical limit for 100x100 mesh** |
| Stripe req  | **Stripes need 1-type uniform particles + low A (<=3). Multi-type (even identical) destroys stripes** |
| LogSensing  | **LogSensing Brusselator-incompatible (3 failures), FHN-compatible (transforms hex->square)** |
| LogSens mech | **LogSensing acts as effective Dv>0 via concentration-dependent mobility. Additive with FHN Dv.** |
| Cell wall   | **2-type same-sign + diff consumption -> boundary/interior type segregation (iter 64)** |
| ActiveMatter | **Self-propulsion disrupts ALL symmetry classes. No new patterns.** |
| GM model | **Foam-like network morphology (unique!), but fundamentally unstable (plateau=0).** |
| Da_c axis | **Da_c is a CONTINUOUS wavelength/mode selector: 5->coarse network, 10->mixed labyrinthine, 15->hex spots. Requires 2-type at low Da_c.** |

---

## Knowledge Base

### Established Principles
1. **Mobility sweet spot**: M1=+/-4, M2=+/-4 prevents escape while preserving field-particle coupling (confirmed 88 iterations)
2. **Opposing mobilities -> phase separation**: Type A (M1=-4, M2=4) vs Type B (M1=4, M2=-4) creates spontaneous spatial segregation (confirmed 12+ iterations)
3. **Extended simulation reveals mode selection**: Ring->hexagonal symmetry breaking occurs at 4000-6000 frames (confirmed 12+ iterations)
4. **100% particle retention**: All configs with M1=+/-4 or +/-8 achieve 100% particles_in_box (confirmed all iterations since iter 2)
5. **Mode selection is particle-type-independent for HEXAGONAL**: 1-type, 2-type, AND 3-type all achieve identical hex patterns when field dynamics are fixed (confirmed 15+ iterations)
6. **Brusselator vs Gray-Scott dichotomy**: Brusselator achieves hexagonal symmetry breaking; Gray-Scott stays purely radial (confirmed iter 14)
7. **Brusselator vs FHN dichotomy**: Both achieve hexagonal, but Brusselator=tight clusters (0.45-0.53), FHN=dispersed/network (0.26-0.39) (confirmed iters 17-24)
8. **ar_p particle-particle params have minimal effect**: Doubling ar_p (1.6->3.0) did not improve clustering metric (confirmed iter 23)
9. **Turing boundary controls mode selection**: B/[1+A^2]=1.0 (at boundary) -> stripes/labyrinth; B/[1+A^2]>1 (deep unstable) -> hexagonal spots (confirmed iters 31-40)
10. **Sub-Turing = NO patterns**: B/[1+A^2]<1 produces disordered/noisy fields with no particle organization (confirmed iter 34)
11. **Stripe mode requires BOTH low A (<=3) AND 1-type uniform particles**: Multi-type (even identical) destroys stripes (confirmed iters 49, 54-55)
12. **Schnakenberg = radial symmetry only**: Like Gray-Scott (confirmed iters 41, 43, 44)
13. **FHN Dv is a symmetry selector with bounded square regime**: Dv=0 -> hexagonal spots, Dv=0.05 -> disordered, Dv=0.1-0.2 -> SQUARE/GRID, Dv>=0.3 -> coarse boundary network. Square requires 1-type (confirmed iters 45-53, 70-72)
14. **FHN square is intrinsically oscillatory**: All FHN Dv>=0.1 configs plateau=0.00 regardless of feedback (confirmed iters 51-53, 60-64)
15. **Deep-Turing is noise**: B=15 produces pixel-scale patterns too fine for particle organization. B<=13 practical limit for 100x100 mesh (confirmed iter 56)
16. **Robustness hierarchy**: Hexagonal (survives multi-type, reversed feedback, shuffling) > Square (1-type only; fails opposing mobility, multi-type even identical) > Stripe (1-type uniform + low A only)
17. **LogSensing is mesh-model selective**: Brusselator-incompatible, FHN-compatible and transformative (confirmed iters 57-64)
18. **LogSensing acts as effective Dv**: Concentration-dependent mobility creates implicit v-diffusion (confirmed iters 60-64)
19. **Differential consumption -> type-boundary segregation**: "Cell wall" morphology (confirmed iter 64)
20. **ActiveMatter disrupts ALL patterns**: No new symmetry classes (confirmed iters 65-69)
21. **Square requires SPATIAL homogeneity**: Even identical-param multi-type fails (confirmed iter 72)
22. **GM model = unstable foam**: Unique morphology but fundamentally unstable (confirmed iters 73-75)
23. **Higher mesh resolution = no new patterns**: Not worth computational cost (confirmed iter 76)
24. **Consumption/production asymmetry is a dead end**: Moderate asymmetry destabilizes; extreme causes NaN (confirmed iters 79-80)
25. **Da_c is a continuous wavelength/mode selector**: Da_c=5->coarse network (2-type only), Da_c=10->mixed labyrinthine (plateau=0.46), Da_c=15->hex spots. Feature wavelength scales inversely with Da_c (confirmed iters 85-88)
26. **Low Da_c requires 2-type opposing mobility**: Da_c=5 + 1-type causes NaN crash (slow Turing can't fragment before runaway). 3-type neutral disrupts cooperative network (confirmed iters 86, 88)
27. **DensityDependent mobility = dead end**: Hill function density-dependent mobility locks ALL patterns into radial symmetry. Code path creates isotropic bias even when near-disabled (rho_0=200) (confirmed 4/4 iters 81-84)

### Open Questions
- Does FHN have its own Turing boundary analogous to Brusselator's B/[1+A^2]=1?
- Can **higher mobility M=8** with standard PDE_D produce different patterns from M=4?
- Do Brusselator stripes evolve further with extended simulation (8000+ frames)?
- Does FHN time_scale variation produce coarse/fine wavelength selection like Brusselator Da_c?
- Can **anisotropic particle mobility** break radial symmetry in a novel way?
- Does Da_c=7-8 produce a clean labyrinthine mode (between mixed and network)?

### Answered Questions (Blocks 1-11)
- **Does hexagonal mode require 3 particle types?** NO — 1, 2, 3-type all work
- **Does Gray-Scott F,k regime change behavior?** F=0.035-0.04, k=0.06-0.065 all produce rings
- **Is the 9/10 result reproducible?** YES — robustness tests confirm
- **Does FHN achieve hexagonal?** YES — but with finer spatial scale
- **Do ar_p params affect clustering?** NO — minimal effect
- **Can chi, D1/D2, domain size break hexagonal?** NO
- **How to break hexagonal attractor?** A=3, B=10 (B/[1+A^2]=1) produces stripes!
- **Can Schnakenberg produce different patterns?** NO — radial only
- **Does FHN square + 3-type work?** NO — even with identical params, 3-type destroys square
- **Does opposing mobility destroy square?** YES
- **Can uniform feedback stabilize FHN square (plateau>0)?** NO — intrinsically oscillatory
- **Can stripe mode work with multi-type?** NO — 1-type only
- **Does deep Turing (B=15) help?** NO — pixel-scale noise
- **Does LogSensing change pattern selection?** YES — hex->square with FHN, suppresses Brusselator
- **Are LogSensing and FHN Dv multiplicative?** NO — additive (same mechanism)
- **Can active matter create new symmetry classes?** NO — disrupts all patterns
- **Does GM produce different patterns?** YES (foam networks), but always unstable
- **Does higher mesh resolution unlock new patterns?** NO
- **Does FHN oscillatory regime (I=0.4) produce traveling waves?** NO — Turing dominates
- **Does asymmetric D1/D2 create new modes?** NO — causes particle collapse
- **Does extreme feedback asymmetry create new morphologies?** NO — NaN explosion
- **Does moderate consumption asymmetry enrich patterns?** NO — destabilizes
- **Does density-dependent mobility create new morphologies?** NO — always radial (4/4 iters)
- **Does lower Da_c change pattern selection?** YES — Da_c=5->coarse network, Da_c=10->labyrinthine
- **Does Da_c=5 network work with 1-type?** NO — NaN crash
- **Does Da_c=5 network work with 3-type?** NO — reverts to concentric radial

### Failed Configurations
- M1=-16, M2=16: particles escape rapidly
- Gray-Scott: cannot achieve hexagonal regardless of params
- Schnakenberg: cannot achieve hexagonal regardless of gamma
- ar_p, chi, D1/D2, domain size changes: do not affect mode selection
- FHN Dv=0.1 + opposing mobility: destroys square symmetry
- FHN Dv=0.1 + 3-type (any params): destroys square symmetry
- Deep-Turing (B=15): pixel-scale noise
- LogSensing + Brusselator (any M): no clean patterns
- ActiveMatter + any mesh model: disrupts all symmetry
- GM (Gierer-Meinhardt): always unstable (plateau=0)
- Extreme feedback asymmetry: NaN explosion
- Moderate consumption asymmetry: destabilizes
- Higher mesh resolution (150x150): no new patterns
- DensityDependent (all configs): 4/4 always radial
- Da_c=5 + 1-type: NaN crash (slow Turing + runaway collapse)
- Da_c=5 + 3-type neutral: reverts to radial layers

### Code Insights
- Base Brusselator (Diffusiophoresis_Mesh) works well for hex and stripe modes
- Gray-Scott and Schnakenberg produce only radial symmetry
- FHN variant with Dv parameter is a symmetry selector (hex/square)
- **PDE_D_LogSensing**: Acts as effective Dv>0 with FHN. Brusselator-incompatible.
- **PDE_D_ActiveMatter**: Self-propelled particles. Disrupts all pattern symmetry. Failed.
- **PDE_D_DensityDependent**: Density-dependent mobility via Hill function. Always radial. Failed.
- Dynamic class loading: new PDE_D variants auto-discovered from filename (no registration needed)

### PDE Variants

| Variant | Model | Literature | Status | Best Score | Symmetry |
| ------- | ----- | ---------- | ------ | ---------- | -------- |
| Diffusiophoresis_Mesh | Brusselator | Prigogine (1968) | active | 9/10 | hexagonal OR stripes |
| PDE_Diffusiophoresis_GrayScott | Gray-Scott | Pearson (1993) Science 261 | tested | 7/10 | radial only |
| PDE_Diffusiophoresis_FHN | FitzHugh-Nagumo | FitzHugh (1961) | active | 9/10 | hexagonal OR square |
| PDE_Diffusiophoresis_Schnakenberg | Schnakenberg | Schnakenberg (1979) JTB 81:389 | tested | 7/10 | radial only |
| PDE_ParticleField_D | Linear diffusiophoresis | Base | active | 9/10 | (depends on mesh) |
| PDE_ParticleField_D_LogSensing | Log-sensing chemotaxis | Kalinin (2009) Biophys J | active | 8/10 | square (with FHN) |
| PDE_ParticleField_D_ActiveMatter | Active matter self-propulsion | Vicsek (1995) PRL | failed | 7/10 | disordered only |
| PDE_Diffusiophoresis_GM | Gierer-Meinhardt | Gierer & Meinhardt (1972) Kybernetik | tested | 7/10 | foam/network (unstable) |
| PDE_ParticleField_D_DensityDependent | Density-dependent mobility | Cates & Tailleur (2015) ARCMP | failed | 6/10 | radial only |
| PDE_ParticleField_D_Anisotropic | Anisotropic diffusiophoresis | Tranquillo & Murray (1992) J Math Biol | active | 8/10 | oriented stripes |

---

## Previous Block Summaries

**Block 1 (Iters 1-8):** Established baseline, mobility sweet spot M=+-4. Score avg: 7.6.

**Block 2 (Iters 9-16):** Gray-Scott PDE variant (radial only), particle-type independence. Score avg: 7.9.

**Block 3 (Iters 17-24):** FHN mesh model (finer hex), ar_p minimal effect. Score avg: 8.5.

**Block 4 (Iters 25-32):** BREAKTHROUGH: A=3, B=10 -> stripes. Score avg: 8.1.

**Block 5 (Iters 33-40):** Turing bifurcation mapping. Score avg: 7.9.

**Block 6 (Iters 41-48):** Schnakenberg (radial), BREAKTHROUGH: FHN Dv=0.1 -> square. Score avg: 7.4.

**Block 7 (Iters 49-56):** Turing boundary multi-dimensional. Stripe=1-type-only. Deep-Turing=noise. Score avg: 7.25.

**Block 8 (Iters 57-64):** First PDE_D modification (LogSensing). Brusselator-incompatible. FHN-compatible: LogSensing acts as effective Dv>0. Score avg: 7.125.

**Block 9 (Iters 65-72):** ActiveMatter PDE_D variant (disrupts ALL symmetry). FHN Dv phase diagram completed. Score avg: 6.6.

**Block 10 (Iters 73-80):** GM mesh model (3 unstable), high-res (no new patterns), consumption asymmetry (destabilizes). Score avg: 6.0.

**Block 11 (Iters 81-88):** DensityDependent PDE_D variant (4/4 radial — failed). BREAKTHROUGH: Da_c is a continuous wavelength/mode selector (5->network, 10->labyrinthine, 15->hex). Da_c=5 requires 2-type. Score avg: 6.9.

---

## Current Block (Block 12)

### Block Info
Parameters: Anisotropic mobility PDE_D variant + FHN time_scale exploration + Da_c fine-mapping
Iterations: 89 to 96
Focus: New PDE_D_Anisotropic variant (directional mobility) + cross-model wavelength control

### Hypothesis
Block 12 introduces **PDE_D_Anisotropic**, a new particle model variant implementing **anisotropic diffusiophoretic mobility** — particles respond differently to gradients in x vs y direction:

**Physics**: v = [Mx * dC/dx, My * dC/dy], where Mx != My
- This breaks the isotropy of standard diffusiophoresis
- Anisotropy ratio alpha = My/Mx controls preferred direction
- Biologically motivated: cells on substrates have oriented motility due to cytoskeletal alignment, ECM fiber orientation
- Literature: Tranquillo & Murray (1992) J Math Biol 31:583-600 — contact guidance in wound healing

**Expected effects**:
1. Break radial symmetry without requiring Turing boundary (B/[1+A^2]=1)
2. Produce ELONGATED/ELLIPTICAL clusters instead of circular spots
3. Potentially select stripe orientation (aligned with high-mobility axis)
4. Create novel anisotropic morphologies not achievable by field modification alone

**Key questions**:
- Does anisotropic mobility break hexagonal symmetry into stripes?
- Can it produce novel elongated/oriented patterns?
- Is it compatible with both Brusselator and FHN?
- Does anisotropy + Da_c variation create new morphologies?

**Also explore**:
- FHN time_scale variation (FHN analog of Da_c)
- Da_c=7 fine-mapping of labyrinthine regime

**Plan**:
- Iter 89: Create PDE_D_Anisotropic + test with Brusselator hex, 1-type (alpha=0.25)
- Iter 90: Anisotropic + Brusselator hex, 2-type opposing
- Iter 91: FHN time_scale=10 (slow) + 2-type — does FHN have wavelength axis like Da_c?
- Iter 92: Anisotropic + FHN, 1-type
- Iter 93: Da_c=7, 2-type — fine-map labyrinthine regime
- Iter 94-96: Best performers, multi-type variants, cross-model tests

### Iterations This Block

**Iter 89** (1/8): PDE_D_Anisotropic alpha=0.25 + Brusselator hex, 1-type
- Config: Brusselator Da_c=15, A=4.5, B=6.5 (hex regime), n_particle_types=1, PDE_ParticleField_D_Anisotropic
- Metrics: entropy=0.61, plateau=0.00, in_box=100%, clustering=0.55
- pos_std_x=0.130, pos_std_y=0.261 (2:1 ratio = anisotropy confirmed)
- **NOVEL**: Oriented horizontal stripes/elongated spots from hex regime params!
- Anisotropy breaks hex symmetry into oriented linear structures
- Plateau=0 → still dynamic (oscillating/merging), not yet stable
- Score: 8/10 (novel symmetry-breaking mechanism, but transient)

**Iter 90** (2/8): PDE_D_Anisotropic alpha=0.25 + Brusselator hex, 2-type opposing
- Config: Same as iter 89 but n_particle_types=2, opposing mobility (M1=-4/+4)
- Metrics: entropy=0.83, plateau=0.00, in_box=100%, clustering=0.46
- pos_std_x=0.155, pos_std_y=0.307 (2:1 ratio preserved)
- **Anisotropic stripes SURVIVE 2-type** (unlike Turing stripes!)
- Type segregation: orange=boundary shells, blue=interior cores
- C1_std dropped 1.43→0.81 (opposing consumption cancels field gradients)
- Score: 8/10 (novel multi-type compatible anisotropic pattern)

**Iter 91** (3/8): FHN time_scale=10 (slow) + 2-type opposing → COARSE HEXAGONAL
- Config: FHN Du=0.05, a=0.5, b=1.0, eps=0.1, time_scale=10, Dv=0, n_particle_types=2, PDE_ParticleField_D
- Metrics: entropy=0.76, plateau=0.00, in_box=100%, clustering=0.17
- pos_std_x=0.239, pos_std_y=0.229 (isotropic — no anisotropy in standard PDE_D)
- ~15-20 coarse hex spots (vs ~30 at time_scale=50) — **time_scale IS a wavelength selector**
- Clustering=0.17 (very low) — coarse spots less effective at concentrating particles
- 2-type: orange clusters at C1 peaks, blue forms network chains between
- Score: 7/10 (confirms wavelength control but same symmetry class)

**Iter 92** (4/8): Anisotropic + FHN, 1-type → NETWORK WITH PERIPHERAL LOBES (anisotropy washed out)
- Config: FHN Du=0.05, a=0.5, b=1.0, eps=0.1, time_scale=50, Dv=0, PDE_ParticleField_D_Anisotropic alpha=0.25, n_particle_types=1
- Metrics: entropy=0.90, plateau=0.00, in_box=100%, clustering=0.25
- pos_std_x=0.217, pos_std_y=0.228 (ratio 1.05:1 — nearly isotropic! vs Brusselator 2:1)
- **Anisotropy DOES NOT transfer to FHN**: FHN's finer spatial scale averages out directional bias
- Pattern: central mass + network tendrils extending to peripheral lobes ("amoeboid" morphology)
- Score: 7/10 (informative negative — anisotropy is mesh-model-selective)

### Emerging Observations
**CRITICAL: This section must ALWAYS be at the END of memory file. When adding new iterations, insert them BEFORE this section.**

- **BREAKTHROUGH iter 89**: PDE_D_Anisotropic alpha=0.25 creates ORIENTED STRIPES from hex regime (new mechanism!)
- Anisotropic mobility selects orientation WITHOUT needing Turing boundary (B/[1+A^2]=1)
- Mechanism: weaker x-response → particles traverse vertical gradients quickly → accumulate at horizontal features
- **CONFIRMED iter 90**: Anisotropic stripes SURVIVE 2-type opposing mobility (Turing stripes don't!)
- Anisotropy = MORE ROBUST symmetry-breaking than Turing boundary: works with multi-type
- 2-type adds shell/core type segregation on top of orientation → "oriented tissue" morphology
- Anisotropy ratio (pos_std_y/x ≈ 2:1) is preserved across 1-type and 2-type configs
- **Iter 91**: FHN time_scale=10 → coarser hex (~15-20 vs ~30 spots). Confirms time_scale controls wavelength, but stays hexagonal (no symmetry transition like Brusselator Da_c). Clustering low (0.17).
- **Iter 92**: Anisotropy is MESH-MODEL-SELECTIVE — strong with Brusselator (2:1 ratio), nearly absent with FHN (1.05:1). Parallels LogSensing's selectivity. FHN's many small features average out directional bias.
- Next: Da_c=7 fine-mapping (iter 93) — between network (5) and labyrinthine (10)
