# Working Memory: diffusiophoresis (parallel)

## Regime Comparison

| Regime | mesh_model | particle_model | n_types | n_particles | Best Score | Key Insight |
| ------ | ---------- | -------------- | ------- | ----------- | ---------- | ----------- |
| 1-type weak | Diffusiophoresis_Mesh | PDE_ParticleField_D | 1 | 9600 | 7/10 | A=5.5,B=7.5 + M1=-8 + 150x150 mesh → sharp dispersed spot array (Iter 39) |
| 1-type high-density | Diffusiophoresis_Mesh | PDE_ParticleField_D | 1 | 14400 | 7/10 | 14400 particles → denser spots but lower entropy, no quality gain (Iter 62) |
| 2-type opposing | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 7/10 | A=5.5/B=7.5 + opposing + adhesion → hexagonal core-ring array (Iter 23) |
| 2-type same-sign | Diffusiophoresis_Mesh | PDE_ParticleField_D | 2 | 9600 | 7/10 | same-sign moderate coupling → core-shell micro-clusters (Iter 12) |
| 3-type opposing | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3 | 9600 | 8/10 | opposing + cross-type adhesion → flower/mandala tissue morphology (Iters 14 & 45, BEST but TIME-LIMITED) |
| 3-type same-sign | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3 | 9600 | 6/10 | same-sign → nested co-localization, less complex (Iters 16, 24) |
| GS any-type | Diffusiophoresis_Mesh_GrayScott | PDE_ParticleField_D | 2-3 | 9600 | 6/10 | GS + particles → radial-locked at ANY coupling (Iters 33-40) |
| FHN 3-type | PDE_Diffusiophoresis_FHN | PDE_ParticleField_D | 3 | 9600 | 7/10 | FHN + 3-type opposing → concentric type-segregated rings (Iter 46) |
| GM 2-type | PDE_Diffusiophoresis_GM | PDE_ParticleField_D | 2 | 9600 | 5/10 | Stabilized; radial morphology (Iter 47) |
| Schnakenberg 2-type | PDE_Diffusiophoresis_Schnakenberg | PDE_ParticleField_D | 2 | 9600 | 5/10 | gamma=60 → radial concentric (Iter 51) |
| 1-type NLD labyrinth | Diffusiophoresis_Mesh (+NLD) | PDE_ParticleField_D | 1 | 9600 | 7/10 | A=3.0/B=5.5 + NLD delta=2.0 → LABYRINTHINE Turing (Iter 83, NOVEL) |
| 3-type NLD labyrinth | Diffusiophoresis_Mesh (+NLD) | PDE_ParticleField_D | 3 | 9600 | 7/10 | A=3.0/B=5.5 + NLD + 3-type → branching tissue on labyrinthine scaffold (Iter 85) |
| 1-type NLD vermiform | Diffusiophoresis_Mesh (+NLD) | PDE_ParticleField_D | 1 | 9600 | 7/10 | A=2.0/B=5.0 + NLD → VERMIFORM filamentary chains, strongest fields ever (Iter 87, NOVEL) |

## Knowledge Base

### Established Principles

1. **Moderate coupling is a UNIVERSAL HARD stability limit**: |M1| <= 10 and consumption <= 100 for FHN; |M1| <= 12 and consumption <= 120 for Brusselator hexagonal; **|M1| <= 8 and consumption <= 80 for Brusselator labyrinthine** (tighter limit). (Evidence: Iters 4,7,32,48,81,88)
2. **D1 >= 0.05 required**: D1 < 0.05 causes numerical crash. (Evidence: Iters 1, 2)
3. **Mobility sign determines pattern type, not stability**: Opposing-sign → segregation. Same-sign → co-localized. Across Brusselator, GS, FHN. (Evidence: Iters 3,8,10,14,23,46)
4. **Plateau=0 is universal**: All models under continuous injection drive non-equilibrium dynamics. (88 iterations, 5 PDE models).
5. **1-type sweet spot is |M1|=8, consumption=80, A=5.5/B=7.5 at 150x150**: Robust across sigma [0.005, 0.008] and particle counts [9600, 14400]. (Iters 11,15,19,39,62,76)
6. **Cross-type adhesion enhances OPPOSING-SIGN morphology specifically**: p[2,5]=0.3 sharpens boundaries. Negligible on same-sign. (Evidence: Iters 10,14,17,23,24,45,46,85)
7. **Opposing-sign 3-type beats same-sign 3-type**: Opposing → 7-8/10; same-sign → 6/10. (Evidence: Iters 8/9/14/21/22/45/46 vs 16/24)
8. **Iter 14 is a TIME-LIMITED local optimum with CONSUMER-DOMINANT asymmetry**: 30+ perturbations scored ≤7/10, and config BLOWS UP at 4000 frames (Iter 84). The chi=-16 regime is marginally unstable. (CONFIRMED DOWNGRADED: Iters 84, 81)
9. **A=5.5/B=7.5 produces more/smaller Turing spots**: Key for 1-type and 2-type, not 3-type. (Evidence: Iters 19, 23, 39, 56)
10. **Chirality has NO sweet spot**: 0.3+ suppresses patterns strongly. 0.1 is neutral. (Evidence: Iters 25, 27, 64, 75)
11. **Iter 23's hexagonal regime is 2-type specific**: 3-type in same params → not superior. (Evidence: Iters 26, 29)
12. **Weber-Fechner suppresses hexagonal at ANY K>0**: K=0.15 already forces radial/bullseye. NO useful regime. (Iters 18, 31, 80)
13. **Michaelis-Menten is a secondary lever**: Km=0.2-0.5 near-neutral. (Evidence: Iters 22, 30)
14. **Gray-Scott is fundamentally radial-locked with particles**. (Evidence: Iters 33,34,38,40)
15. **150x150 mesh (22500 nodes) is OPTIMAL for 9600 particles**: 200x200 degrades all configs. (Evidence: Iters 39,41,45,49,54,55,72)
16. **Non-Brusselator PDE models are radial-locked with particles**: ALL alternatives (GS, FHN, Schnakenberg, GM) produce radial morphology. Only Brusselator achieves hexagonal multi-spot arrays. (Evidence: 12+ iterations across 4 non-Brusselator models)
17. **NLD delta=2.0 + high B/A ratio → labyrinthine Turing patterns**: A=3.0/B=5.5 (B/A=1.83) + NLD delta=2.0 breaks hexagonal symmetry into labyrinthine/vermiform. A=2.0/B=5.0 (B/A=2.5) pushes further into fragmented vermiform. (NEW: Evidence: Iters 83, 85, 87)
18. **NLD delta=3.0 OVER-DAMPS hexagonal at A=5.5/B=7.5**: Higher NLD weakens field contrast. delta=2.0 is optimal for hexagonal regime. (NEW: Evidence: Iter 86)
19. **NLD is INCOMPATIBLE with strong coupling (chi=-16)**: NLD delta=1.0+ at chi=-16/consumption=100 → blowup. NLD requires moderate coupling (chi=-8). (NEW: Evidence: Iters 81, 84)
20. **ALL 8 PDE_D particle features exhausted**: W-F, M-M, chirality, durotaxis, pp_field_mod, DDM, alignment, sigma — only adhesion worked. (Evidence: 80 iterations, Blocks 1-10)

### Open Questions

- Would ASYMMETRIC diffusion (D1 anisotropic) create stripe selection in Brusselator?
- Would durotaxis on MULTI-TYPE (2 or 3-type) create boundary-sensing effects? (1-type was neutral)
- Would NLD at LOWER delta (0.5-1.0) at A=3.0/B=5.5 still produce labyrinthine, or is delta=2.0 required?
- Is the labyrinthine regime (A=3.0/B=5.5 + NLD) more TIME-STABLE than the hexagonal regime (A=4.5/B=6.5)?
- Can 2-type opposing particles on the labyrinthine regime (A=3.0/B=5.5 + NLD) create core-shell vermiform filaments?
- Can 3-type on vermiform regime (A=2.0/B=5.0 + NLD) create novel tissue morphology?
- Would A=2.5/B=5.5 (B/A=2.2, intermediate) bridge labyrinthine and vermiform regimes?
- Would CROSS-DIFFUSION (chi parameter in mesh model, not mobility) + NLD create coupled multi-scale patterns?
- Can we find a TRULY TIME-STABLE 8/10 pattern using the labyrinthine regime at moderate coupling?

### Failed Configurations

- D1=0.03/0.01 + high Da_c → crash (Iters 1, 2)
- 3-type with M1=-24, consumption=250 → all escape (Iter 4)
- 1-type with |M1|=16, consumption=180 → all escape (Iter 7)
- 3-type M1=-14, consumption=140 → NaN blow-up (Iter 32)
- **Avoid D1 < 0.05, |M1| > 12 (Brusselator hex), |M1| > 8 (Brusselator labyrinthine), |M1| > 10 (FHN)**
- **Avoid Weber-Fechner K > 0, chirality >= 0.3**
- **Gray-Scott, Schnakenberg gamma=200, GM without stabilization** → all fail or radial-locked
- **200x200 mesh at 9600 particles** → dead end (Iters 49,54,55)
- **pp_field_mod** → neutral (Iters 61, 63)
- **DDM** → neutral on multi-type, harmful on 1-type (Iters 65-72)
- **Velocity alignment** → cosmetic (Iters 73-79). HURTS 2-type.
- **NLD delta=1.0 + chi=-16** → blowup (Iter 81). NLD incompatible with strong coupling.
- **Iter 14 at 4000 frames** → blowup (Iter 84). chi=-16 regime is marginally unstable.
- **NLD delta=3.0 at A=5.5/B=7.5** → over-damps fields (Iter 86, 6/10)
- **chi=-10 + consumption=100 on labyrinthine** → degrades coherence (Iter 88, 5/10)

### Code Insights

- PDE_D.py features: W-F (p[2,4]), adhesion (p[2,5]), M-M (p[1,2]), durotaxis (p[1,3]), chirality (p[1,4]), pp_field_mod (p[2,6]), DDM (p[1,5]), alignment (p[2,7]) — all backward-compatible
- Cross-type adhesion p[2,5]=0.3 is the ONLY PDE_D feature that improved scores (7→8/10)
- **ALL 8 PDE_D features tested**: only adhesion worked
- **Block 11 code change: NLD in Brusselator** — D1(C1) = D1_base*(1+delta*(C1-A)^2/A^2). Enabled labyrinthine and vermiform patterns.
- Brusselator is the ONLY mesh model producing non-radial patterns; NLD extends it to labyrinthine/vermiform
- **Next code lever for Block 12**: Consider CUBIC autocatalysis modification, substrate inhibition, or time-delayed feedback in Brusselator to find new pattern regimes

### PDE Variants

| Variant | Model | Literature | Status | Best Score |
| ------- | ----- | ---------- | ------ | ---------- |
| Diffusiophoresis_Mesh | Brusselator | Prigogine (1968) | active (BEST) | 8/10 (Iters 14,45,53) but TIME-LIMITED |
| Diffusiophoresis_Mesh (+ NLD) | Brusselator + nonlinear diffusion | Gambino et al. (2013) | active (PROMISING) | 7/10 (Iters 83, 85, 87 — labyrinthine+vermiform) |
| Diffusiophoresis_Mesh_GrayScott | Gray-Scott | Pearson (1993) | FAILED | 6/10 radial-locked |
| PDE_Diffusiophoresis_FHN | FHN | FitzHugh (1961) | radial-locked | 7/10 (Iter 46) |
| PDE_Diffusiophoresis_Schnakenberg | Schnakenberg | Schnakenberg (1979) | radial-locked | 5/10 (Iter 51) |
| PDE_Diffusiophoresis_GM | Gierer-Meinhardt | Gierer & Meinhardt (1972) | radial-locked | 5/10 (Iter 47) |

---

## Previous Block Summaries

### Blocks 1-7 (Iters 1-56)
- **Global best: Iter 14/45/53 (3-type opposing Brusselator 150x150, 8/10)** — flower/mandala tissue morphology
- **Best 2-type: Iter 23 (7/10)** — hexagonal core-ring array
- **Best 1-type: Iter 39 (7/10)** — dispersed spot array at 150x150
- 56 iterations across 5 PDE mesh models, 6 PDE_D code features, resolutions 100x100 to 200x200
- Brusselator is the only model producing hexagonal patterns; all others radial-locked
- Cross-type adhesion p[2,5]=0.3 is the only code feature that improved scores

### Block 8 (Iters 57-64)
- No 8/10 ceiling broken. Best: 7/10 (Iters 61, 62, 64).
- pp_field_mod=0.5 NEUTRAL. 14400 particles same 7/10. Chirality=0.1 neutral.

### Block 9 (Iters 65-72) — DDM FAILED
- **No 8/10 ceiling broken.** Best: 7/10 (Iters 70, 72). Average: 6.25/10.
- DDM tested across full range [0.15-1.0]: NEUTRAL on multi-type, HARMFUL on 1-type.

### Block 10 (Iters 73-80) — ALIGNMENT COSMETIC + W-F DEAD END
- **No 8/10 ceiling broken.** Scores: 7,6,6,7,7,7,7,5. Average: 6.5/10.
- All 8 PDE_D features exhaustively tested. Only adhesion helped. Strategy shift to mesh model.

### Block 11 (Iters 81-88) — NLD: LABYRINTHINE BREAKTHROUGH
- Scores: 1, 7, 7, 1, 7, 6, 7, 5 → Average: 5.1/10, Best: 7/10 (Iters 82, 83, 85, 87)
- **NLD + high B/A ratio → LABYRINTHINE Turing (Iter 83, NOVEL)**: First non-hexagonal Brusselator pattern in 83 iters
- **B/A=2.5 + NLD → VERMIFORM filaments (Iter 87, NOVEL)**: Strongest fields ever (pattern_growth=294)
- 3-type on labyrinthine produces branching tissue (Iter 85, 7/10) — promising hybrid
- NLD delta=3.0 over-damps hexagonal (Iter 86, 6/10). delta=2.0 optimal.
- Labyrinthine has tighter coupling limit: |chi|<=8 (vs hexagonal |chi|<=12)
- Iter 14 confirmed TIME-LIMITED (blows up at 4000 frames). No 8/10 ceiling broken but 2 novel pattern types.

---

## Current Block (Block 12)

### Block Info

Parameters: Brusselator + NLD, exploring labyrinthine+vermiform regime in depth
mesh_model_name: Diffusiophoresis_Mesh (with NLD toggle via params_mesh[1][3])
Iterations: 89-96 (parallel, 4 slots per batch)
Starting from: Iter 85 (3-type labyrinthine 7/10), Iter 87 (1-type vermiform 7/10), Iter 83 (1-type labyrinthine 7/10)
Code modification: Consider substrate inhibition or saturation kinetics in Brusselator reaction terms

### Hypothesis

The labyrinthine (Iter 83/85, A=3.0/B=5.5) and vermiform (Iter 87, A=2.0/B=5.0) regimes are the most significant discoveries since Iter 14. They produce qualitatively NEW pattern types that no parameter tuning on standard Brusselator achieved. Key priorities:
1. Can multi-type particles on labyrinthine/vermiform fields break the 8/10 ceiling?
2. What B/A ratio and NLD delta give the richest patterns?
3. Is a code modification (substrate inhibition/saturation) needed to get beyond 7/10 in these regimes?
4. Test intermediate A values (A=2.5) to bridge labyrinthine and vermiform.

### Iterations This Block

(Block 12 starts at Iter 89)

### Emerging Observations

(To be filled as Block 12 progresses)
