# Working Memory: diffusiophoresis (parallel)

## Regime Comparison

| Regime | mesh_model | particle_model | n_types | n_particles | Best Score | Key Insight |
| ------ | ---------- | -------------- | ------- | ----------- | ---------- | ----------- |
| 1-type weak | Diffusiophoresis_Mesh | PDE_ParticleField_D | 1 | 9600 | 7/10 | A=5.5,B=7.5 + M1=-8 + 150x150 mesh → sharp dispersed spot array (Iter 39, NEW BEST) |
| 2-type opposing | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 7/10 | A=5.5/B=7.5 + opposing + adhesion → hexagonal core-ring array (Iter 23) |
| 2-type same-sign | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 7/10 | same-sign moderate coupling → core-shell micro-clusters (Iter 12) |
| 2-type W-F bullseye | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 6/10 | Weber-Fechner K=0.3 → concentric bullseye (Iter 31) |
| 3-type opposing | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3 | 9600 | 8/10 | opposing + cross-type adhesion → flower/mandala tissue morphology (Iters 14 & 45, BEST) |
| 3-type same-sign | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3 | 9600 | 6/10 | same-sign → nested co-localization, less complex (Iters 16, 24) |
| GS any-type | Diffusiophoresis_Mesh_GrayScott | PDE_ParticleField_D | 2-3 | 9600 | 6/10 | GS + particles → radial-locked concentric rings at ANY coupling strength (Iters 33-34-38-40) |
| FHN 1-type | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D | 1 | 9600 | 6/10 | FHN 1-type → radial at 150x150 (Iter 50); network at 100x100 was low-res artifact (Iter 44) |
| FHN 3-type | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D | 3 | 9600 | 7/10 | FHN + 3-type opposing → concentric type-segregated rings (Iter 46, NEW) |
| GM 2-type | PDE_Diffusiophoresis_GM | PDE_ParticleField_D | 2 | 9600 | 5/10 | Stabilized (100% retention) with rho=0.05, mu=0.05, kappa=0.2; radial morphology (Iter 47) |
| Schnakenberg 2-type | PDE_Diffusiophoresis_Schnakenberg | PDE_ParticleField_D | 2 | 9600 | 5/10 | Schnakenberg gamma=60 + weak coupling → radial concentric (Iter 51); gamma=200 blew up (Iter 42) |

## Knowledge Base

### Established Principles

1. **Moderate coupling is a UNIVERSAL HARD stability limit**: |M1| <= 10 and consumption <= 100 for FHN; |M1| <= 12 and consumption <= 120 for Brusselator. FHN is MORE sensitive — |M1|=12 causes borderline blow-up (96% retention, C1_std=16). (UPGRADED: Iters 4,7,32 Brusselator; Iter 48 FHN confirmed model-specific tightening)
2. **D1 >= 0.05 required**: D1 < 0.05 causes numerical crash at 100x100 mesh with delta_t=5E-4. (Evidence: Iters 1, 2 crashed)
3. **Mobility sign determines pattern type, not stability**: Opposing-sign → spatial segregation. Same-sign → co-localized core-shell. Confirmed at 2-type AND 3-type, AND across Brusselator, GS, and FHN. (UPGRADED: Evidence now includes FHN Iter 46)
4. **Plateau=0 is universal**: All PDE models (Brusselator, GS, FHN, GM) under continuous injection drive non-equilibrium dynamics. (UPGRADED: 48 iterations, 5 PDE models)
5. **1-type sweet spot is |M1|=8, consumption=80, A=5.5/B=7.5**: Higher mesh resolution (150x150) enhances this further → 7/10 (Iter 39). (Evidence: Iters 11,15,19,39)
6. **Cross-type adhesion enhances OPPOSING-SIGN morphology specifically**: p[2,5]=0.3 sharpens inter-type boundaries in segregated configs. Negligible on same-sign (Iter 24: 6→6). Optimal ~0.3. (Evidence: Iters 10,14,17,23,24,45,46)
7. **Opposing-sign 3-type beats same-sign 3-type**: Opposing → 7-8/10; same-sign → 6/10. (Evidence: Iters 8/9/14/21/22/45/46 vs 16/24)
8. **Iter 14 is a robust local optimum with CONSUMER-DOMINANT asymmetry**: 14+ perturbations scored ≤7/10. Iter 45 (150x150) ties at 8/10 but doesn't surpass. Consumer must be strongest mover (|M_consumer|>|M_producer|). (Evidence: Iters 17,18,20,21,22,25,28,29,30,37,45)
9. **A=5.5/B=7.5 produces more/smaller Turing spots**: Key lever for 1-type and 2-type, not for 3-type. (Evidence: Iters 19, 23, 39 vs 15, 10)
10. **Chirality suppresses pattern elaboration at all tested values (0.3-0.5)**: Spiral drift overwhelms gradient-following. (Evidence: Iters 25, 27)
11. **Iter 23's hexagonal regime is 2-type specific**: 3-type in same params → hybrid but not >Iter 14/23. (Evidence: Iters 26, 29)
12. **Weber-Fechner affects symmetry selection, not just strength**: K=0.3 → bullseye, K=2.0 → kills patterns. (Evidence: Iters 18, 31)
13. **Michaelis-Menten is a secondary lever**: Km=0.2-0.5 produces near-Iter-14 quality. (Evidence: Iters 22, 30)
14. **Gray-Scott is fundamentally radial-locked with particles**: Any coupling strength and GS regime produces concentric rings. (UPGRADED: 4 iterations — Iters 33,34,38,40)
15. **Durotaxis p[1,3]=0.5 is neutral for 1-type**: Doesn't change dispersed spot array morphology. (Evidence: Iter 35)
16. **150x150 mesh (22500 nodes) is the OPTIMAL resolution for 9600 particles**: Finer mesh resolves more Turing modes but 200x200 (40000 nodes) DEGRADES all configs because particle density per Turing spot is too low. 1-type: 150x150=7/10 (Iter 39), 200x200=6/10 (Iter 49). 3-type: 150x150=8/10 (Iter 45), 200x200=6/10 (Iter 54). 2-type: 200x200=6/10 (Iter 55). (UPGRADED: Evidence: Iters 39,41,45,49,54,55)
17. **FHN is radial-locked at ALL particle counts and resolutions**: 1-type at 100x100 appeared to produce dispersed network (Iter 44, 6/10) but 1-type at 150x150 produces radial concentric rings (Iter 50, 6/10) — the network was a low-resolution artifact. 3-type → concentric rings (Iter 46, 7/10). FHN is fundamentally radial-locked like GS. (REVISED: Iter 50 contradicted 1-type network claim; Evidence: Iters 44, 46, 50)
18. **Non-Brusselator PDE models are radial-locked with particles**: ALL tested alternatives (GS, FHN, Schnakenberg, GM) produce radial/concentric morphology when combined with particles at all resolutions and particle counts. Only Brusselator achieves hexagonal symmetry-breaking with multi-spot arrays. This is because Brusselator's Turing instability creates MULTIPLE independent spots, while other models create single expanding wavefronts or weaker patterns. (UPGRADED: 5 non-Brusselator models tested; Evidence: GS Iters 33-40; FHN Iters 44,46,48,50; Schnakenberg Iter 51; GM Iter 47)

### Open Questions

- Would very low chirality (p[1,4]=0.1) on 3-type add subtle spiral features? (0.3+ too strong)
- Would very low Weber-Fechner (K=0.1-0.15) give a transitional regime between hexagonal and bullseye?
- Would durotaxis on MULTI-TYPE (2 or 3-type) create boundary-sensing effects? (1-type was neutral)
- Can GM at higher resolution (150x150) produce multi-spot breakup instead of single radial pattern?
- Would ASYMMETRIC diffusion (D1 anisotropic) create stripe selection in Brusselator?
- Would increasing n_particles to 14400+ at 150x150 increase particle density per spot and break 8/10?
- Would a new PDE_D variant with alignment/flocking (Vicsek/Boids) break radial lock for non-Brusselator models?
- Would concentration-dependent particle-particle interaction (attraction scales with local C1) create emergent sorting?
- ~~Can Schnakenberg work at gamma=50-80?~~ YES, gamma=60 stable but radial (Iter 51)
- ~~Would FHN 150x150 break radial lock?~~ NO, radial at all resolutions (Iter 50)
- ~~Would Brusselator 200x200 improve patterns?~~ NO, dead end for ALL types at 9600 particles (Iters 49,54,55)
- ~~Can n_frames=4000 break the 8/10 ceiling?~~ NO, ties at 8/10 with stronger fields but same morphology tier (Iter 53)

### Failed Configurations

- D1=0.03 + Da_c=20.0 (n_types=1): crash (Iter 1)
- D1=0.01 + Da_c=10.0 (n_types=1): crash (Iter 2)
- 3-type with M1=-24, consumption=250: all escape (Iter 4)
- 1-type with base params M1=-16, consumption=180: all escape (Iter 7)
- 3-type M1=-14, consumption=140: total blow-up, NaN (Iter 32)
- **Avoid D1 < 0.05** — crashes at current resolution
- **Avoid |M1| > 12 or consumption > 120 (Brusselator)** — HARD stability limit
- **Avoid |M1| > 10 or consumption > 100 (FHN)** — FHN is more sensitive than Brusselator (Iter 48: 96% retention, fields blowing up)
- **Avoid 1-type with |M1| >= 10** — likely unstable
- **Avoid Weber-Fechner K >= 2.0** — kills Turing breakup entirely (Iter 18)
- **Avoid chirality p[1,4] >= 0.3** — suppresses pattern elaboration (Iters 25, 27)
- **Gray-Scott + particles at ANY coupling strength** → radial-locked (Iters 33,34,38,40)
- **Schnakenberg gamma=200 + |M1|=10** → total blow-up (Iter 42). gamma=60 + |M1|=4 works (Iter 51, 5/10) but radial-locked
- **GM (Da=0.01, rho=0.1, mu_a=0.02, kappa=0)** → fields diverge (Iter 43). Need rho<=0.05, mu>=0.05, kappa>=0.2

### Code Insights

- PDE_D.py features: Weber-Fechner (p[2,4]), cross-type adhesion (p[2,5]), Michaelis-Menten (p[1,2]), durotaxis (p[1,3]), chirality (p[1,4]), **field-modulated pp adhesion (p[2,6], NEW Block 14)** — all backward-compatible
- PDE_Diffusiophoresis.py: parameterized damping via params_mesh[1][2]; chi cross-diffusion; noise amplitude
- Cross-type adhesion p[2,5]=0.3 is the ONLY code feature that meaningfully improved scores (7→8/10)
- All other PDE_D features (W-F, M-M, chirality, damping, durotaxis) either hurt or were neutral
- **Brusselator parameter space thoroughly exhausted** — 40+ iterations, Iter 14/45 tied at 8/10
- **Gray-Scott tested and FAILED** — fundamentally incompatible with particle coupling
- **FHN works but radial-locked with multi-type** — 1-type network is novel (6/10), 3-type is radial (7/10)
- **GM stabilized** — rho=0.05, mu=0.05, kappa=0.2 works; morphology is simple radial (5/10)
- **Schnakenberg stable at gamma=60** — but radial-locked like all non-Brusselator models (Iter 51, 5/10)
- **n_frames=4000 doesn't break ceiling** — Iter 53 ties at 8/10, higher C1_std (2.10) but same morphology tier
- **200x200 mesh is a DEAD END** — all types at 9600 particles degrade (Iters 49,54,55 all 6/10)
- **A=7/B=10 doesn't improve 3-type** — 7/10 vs 8/10 at A=5.5/B=7.5 (Iter 56). Marginally more escape.

### PDE Variants

| Variant | Model | Literature | Status | Best Symmetry | Best Particles |
| ------- | ----- | ---------- | ------ | ------------- | -------------- |
| Diffusiophoresis_Mesh | Brusselator | Prigogine (1968) | active (BEST) | flower/mandala | segregated 3-type (Iters 14,45: 8/10) |
| Diffusiophoresis_Mesh_GrayScott | Gray-Scott | Pearson (1993) | FAILED | radial (concentric rings) | radial-locked at all couplings |
| PDE_Diffusiophoresis_FHN | FHN | FitzHugh (1961) | active (radial-locked) | concentric rings (all types) | 7/10 radial (Iter 46), 6/10 radial (Iter 50) |
| PDE_Diffusiophoresis_Schnakenberg | Schnakenberg | Schnakenberg (1979) | STABLE (gamma=60) | radial (concentric rings) | 5/10 radial 2-type (Iter 51) |
| PDE_Diffusiophoresis_GM | Gierer-Meinhardt | Gierer & Meinhardt (1972) | STABILIZED | radial w/ folding | 5/10 radial (Iter 47) |

---

## Previous Block Summaries

### Block 1 (Iters 1-8)
- **Best: Iter 8 (3-type moderate coupling, 7/10)** — tissue-like stratification
- 1-type all failed, 2-type (5-6/10) reliable, 3-type best with moderate coupling
- Average: 4.4/10

### Block 2 (Iters 9-16)
- **Best: Iter 14 (3-type opposing + adhesion, 8/10)** — NEW OVERALL BEST, flower/mandala morphology
- Cross-type adhesion is the key enhancer (7→8/10 for 3-type)
- Average: 6.1/10

### Block 3 (Iters 17-24)
- **Iter 14 remains global best at 8/10** — four variants ≤7/10
- **NEW BEST 2-type: Iter 23 (7/10)** — hexagonal core-ring array
- Average: 6.4/10

### Block 4 (Iters 25-32)
- **Iter 14 still unbeatable** — 10+ perturbations all ≤7/10 or failed
- Chirality, W-F, M-M secondary levers or detrimental. Brusselator parameter space exhausted.
- Average: 5.5/10

### Block 5 (Iters 33-40)
- **Best: Iter 37, Iter 39 (7/10 each)** — Iter 39 is NEW BEST 1-type (150x150 mesh spots)
- Gray-Scott tested comprehensively (4 iters) and FAILED — radial-locked at all coupling strengths
- Higher mesh resolution (150x150) confirmed beneficial for 1-type patterns
- Average: 6.0/10

### Block 6 (Iters 41-48)
- **Best: Iter 45 (3-type Brusselator 150x150, 8/10)** — ties Iter 14. Iter 46 (FHN 3-type, 7/10) is second best.
- Tested 4 new PDE models: FHN (stable, novel network 1-type + radial 3-type), GM (stabilized with decay/saturation, radial 5/10), Schnakenberg (blew up at gamma=200). All non-Brusselator models are radial-locked with multi-type particles.
- FHN's 1-type network morphology (Iter 44, 6/10) is a genuinely new class. FHN 3-type (Iter 46, 7/10) competitive but radial.
- Principle #1 confirmed universal — FHN even more sensitive to coupling than Brusselator (Iter 48: 96% retention at |M1|=12).
- 150x150 mesh confirmed beneficial for all configs (principle #16 upgraded).
- Average: 6.0/10

Particle type distribution (cumulative): 1-type: 12, 2-type: 15, 3-type: 29. **Still 3-type heavy — Block 8 should include more 1-type and 2-type.**

### Block 7 (Iters 49-56)
- **Best: Iter 53 (3-type Brusselator 150x150 + n_frames=4000, 8/10)** — ties Iter 14/45 but doesn't break ceiling. C1_std=2.10 (highest ever) but same morphology tier.
- 200x200 mesh confirmed DEAD END for all types at 9600 particles (Iters 49,54,55 all 6/10).
- FHN 1-type network was low-res artifact (Iter 50). Schnakenberg stable at gamma=60 but radial (Iter 51).
- Consumer-dominant asymmetry strongly confirmed (Iter 52: 4/10). A=7/B=10 doesn't help 3-type (Iter 56: 7/10).
- Average: 5.75/10

---

## Current Block (Block 8)

### Block Info

Parameters: Focus on NEW interaction physics to break 8/10 ceiling. Code modifications allowed at block boundary.
mesh_model_name: Brusselator (only model that produces hexagonal patterns)
Iterations: 57-64 (parallel, 4 slots per batch)
Starting from: Iter 14/45/53 (3-type Brusselator 150x150, 8/10)

### Hypothesis

After 56 iterations, the 8/10 ceiling is EXTREMELY robust. All parameter-space explorations (resolution, simulation length, A/B values, coupling strengths, PDE models) have been exhausted without breaking it. The remaining levers are: (1) New particle interaction physics via code modifications — concentration-dependent pp forces, alignment/flocking, or density-dependent mobility, (2) Higher particle density (n_particles=14400 at 150x150), (3) Untested parameter combinations like very low chirality or durotaxis on multi-type. Block 8 strategy: Create a new PDE_D variant with concentration-dependent particle interactions (particles interact more strongly in high-C1 regions) + test higher particle counts + explore remaining parameter combos.

### Iterations This Block

(empty — new block)

### Emerging Observations

(empty — new block)
