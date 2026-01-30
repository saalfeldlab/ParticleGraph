# Pattern Exploration Log: diffusiophoresis

## Iter 54: 7/10 - Brusselator stripe mode + 2-type reversed feedback → DISORDERED (stripes destroyed)
Node: id=54, parent=49
Mode/Strategy: explore — test stripe mode (A=3, B=10) with 2-type same-sign mobility + reversed consumption/production
Config: mesh_model=Diffusiophoresis_Mesh, particle_model=PDE_ParticleField_D, n_particle_types=2, shuffle=true
  params_mesh: D1=1.0, Da_c=15.0, A=3.0, B=10.0, mu=1.0; D2=8.0
  params: Type 0 [-4,4,45,-45,1.6,1.0,1.6,1.5], Type 1 [-4,4,-45,45,1.6,1.0,1.6,1.5]
  Pe=1.0, coup=[180,-180], n_frames=6000, delta_t=0.0005, n_particles=9600, n_nodes=10000
n_particle_types: 2
Metrics: entropy=[0.85], plateau=[0.00], in_box=[100.0]%, clustering=[0.36]
  C1_std=2.84, C2_std=1.55, pattern_growth=310x
Assessment:
  - Symmetry: none (disordered)
  - Particles: clustered (moderate — type-separated but irregular)
  - Stability: unstable (plateau=0.00, ongoing dynamics)
  - Novelty: variant (reversed feedback disrupts known stripe regime)
Visual: Fields show disordered/noisy labyrinthine pattern without clean stripes or hexagonal periodicity. C1 has irregular amplitude variations. Particles show type separation: blue (type 0) concentrates centrally while orange (type 1) forms scattered clusters throughout. No stripe formation achieved.
Mutation: n_particle_types: 1->2, consumption/production reversed for type 1, A: 4.5->3.0, B: 6.5->10.0
Observation: **CRITICAL FINDING: Stripe mode is MORE FRAGILE than hexagonal.** Hexagonal survives reversed feedback (iter 50 showed hex with type-segregation). But stripe mode at Turing boundary (B/[1+A^2]=1.0) is destroyed by competing consumption/production. The exact balance at the Turing boundary is disrupted when one type reinforces and another counteracts the pattern. This asymmetry between hex and stripe robustness is a novel finding.
Next: parent=31

## Iter 53: 7/10 - FHN square + 3-type ALL IDENTICAL params → square persists but plateau=0.00 STILL
Node: id=53, parent=51
Mode/Strategy: exploit — test whether uniform feedback (identical params for all 3 types) solves convergence failure
Config: mesh_model=PDE_Diffusiophoresis_FHN, particle_model=PDE_ParticleField_D, n_particle_types=3, shuffle=false
  params_mesh: Du=0.5, a=0.7, b=0.8, eps=0.08, I=0, time_scale=20; Dv=0.1
  params: ALL 3 types identical [-4,4,45,-45,1.6,1.0,1.6,1.5]
  n_frames=6000, delta_t=0.0005, n_particles=9600, n_nodes=10000
n_particle_types: 3
Metrics: entropy=0.81, plateau=0.00, in_box=100.0%, clustering=-0.07
  C1_std=1.07, C2_std=0.61, pattern_growth=122.2x
Assessment:
  - Symmetry: other (square/grid)
  - Particles: segregated (concentric type layers following square contours)
  - Stability: transient (plateau=0.00, vel=0.25 at frame 120)
  - Novelty: variant (same square regime as iters 45-52)
Visual: Clear square symmetry in both C1 and C2 fields with concentric square rings. 3-type particles form ordered concentric layers: blue center, orange boundary filament, green outer edge — all following square contours. By late frames (100-120), asymmetries emerge at corners. Pattern growth=122x, particles still actively moving.
Mutation: params: conflicting feedback types → ALL IDENTICAL (all M1=-4, M2=4, consumption=45, production=-45)
Observation: **CRITICAL**: Uniform feedback does NOT solve convergence. plateau=0.00 persists even with all 3 types identical. Non-convergence is INTRINSIC to FHN Dv=0.1 square regime — the square is fundamentally a transient/oscillatory mode, not caused by conflicting feedback types. This falsifies the hypothesis from iters 51-52.
Next: parent=49

## Iter 52: 8/10 - FHN square + 3-type shuffled → SQUARE SURVIVES + self-organized type sorting
Node: id=52, parent=51
Mode/Strategy: shuffle — same config as iter 51 with shuffle_particle_types=true (even iteration 4/8)
Config: mesh_model=PDE_Diffusiophoresis_FHN, particle_model=PDE_ParticleField_D, n_particle_types=3, shuffle=true
  params_mesh: Du=0.5, a=0.7, b=0.8, eps=0.08, I=0, time_scale=20; Dv=0.1
  params: Type 0 [-4,4,45,-45,...], Type 1 [-4,4,-45,45,...], Type 2 [-4,4,22,-22,...]
  n_frames=6000, delta_t=0.0005, n_particles=9600, n_nodes=10000
n_particle_types: 3
Metrics: entropy=0.80, plateau=0.00, in_box=100.0%, clustering=0.00
  C1_std=1.09, C2_std=0.63, pattern_growth=126.5x
Assessment:
  - Symmetry: square
  - Particles: network (type-sorted filaments at square boundaries)
  - Stability: transient (plateau=0.00, still evolving)
  - Novelty: variant (confirms iter 51 with shuffled conditions)
Visual: Square pattern clearly visible in both C1 and C2 fields. Despite shuffled initial conditions, particles SELF-ORGANIZE into spatial niches: blue (type 0) fills square interior, orange (type 1) forms boundary ring/filaments, green (type 2) at outer edges. Circular→square transition visible in evolution. Novel emergent type sorting from random initial mixing.
Mutation: shuffle_particle_types: false → true (iter 51 → 52)
Observation: Square symmetry is ROBUST to shuffled initial conditions. More importantly, shuffled 3-type particles self-sort into spatial niches within the square structure — type 0 (standard feedback) occupies center, type 1 (reversed feedback) forms boundary, type 2 (weak) at edges. This demonstrates genuine SELF-ORGANIZATION, not just inheritance from ordered initial conditions. Plateau=0.00 persists — conflicting feedback types prevent convergence.
Next: parent=TBD

## Iter 51: 7/10 - FHN square + 3-type same-sign mobility → square survives but unstable
Node: id=51, parent=45
Mode/Strategy: explore — FHN Dv=0.1 square + 3-type same-sign mobilities (all M1=-4, M2=4)
Config: mesh_model=PDE_Diffusiophoresis_FHN, params_mesh Du=0.5, a=0.7, b=0.8, eps=0.08, Dv=0.1, n_frames=6000
n_particle_types: 3 (shuffle=false)
  Type 0: M1=-4, M2=4, cons=45, prod=-45 (standard)
  Type 1: M1=-4, M2=4, cons=-45, prod=45 (reversed feedback)
  Type 2: M1=-4, M2=4, cons=22, prod=-22 (half-strength)
Metrics: entropy=0.81, plateau=0.00, in_box=100%, clustering=-0.08
Assessment:
  - Symmetry: square (concentric square ring structure preserved)
  - Particles: segregated (type-ordered concentric rings with square geometry)
  - Stability: transient (plateau=0.00, system not converged)
  - Novelty: variant (first successful 3-type square result)
Visual: C1/C2 fields show clear square ring pattern. Particles form concentric square bands: blue(type 0)=center cloud, orange(type 1)=sharp square ring boundary, green(type 2)=outer edge. Pattern evolves from circular→square over simulation. Subsidiary spots emerging at corners by final frame.
Mutation: n_particle_types: 1->3 with same-sign mobilities and varied feedback strengths
Observation: **ANSWERS OPEN QUESTION**: Square symmetry survives multi-type when all types have SAME mobility direction. Opposing mobilities (iter 46) destroyed square→hex. The symmetry-breaking mechanism is mobility-DIRECTION-dependent, not multi-type per se. However, conflicting feedback (type 1 reversed) prevents convergence (plateau=0.00). Clustering=-0.08 indicates near-uniform particle distribution (no tight clusters). Pattern is still evolving at end — may need longer simulation or reduced feedback conflict to stabilize.
Next: parent=45

## Iter 50: 8/10 - Same-sign mobility 2-type → type-segregated hexagonal
Node: id=50, parent=root
Mode/Strategy: explore — 2-type with same-sign mobilities (M1=-4, M2=4 both) but reversed consumption/production
Config: params_mesh=[0.05, 15.0, 4.5, 6.5, 0.04, 0.0], n_frames=6000, delta_t=5E-4, n_particle_types=2, shuffle=true
  Type 0: M1=-4, M2=4, consumption=45, production=-45 (standard)
  Type 1: M1=-4, M2=4, consumption=-45, production=45 (reversed feedback)
n_particle_types: 2
Metrics: entropy=0.60, plateau=0.00, in_box=100%, clustering=0.41
  C1_std=0.80, C2_std=0.56, pattern_growth=112.5x
Assessment:
  - Symmetry: hexagonal
  - Particles: segregated
  - Stability: transient (plateau=0.00, still evolving at 6000 frames)
  - Novelty: variant
Visual: Hexagonal ~25-30 spots fully developed by frame 80. Both particle types attracted to same field peaks (same-sign mobility), but TYPE-SEGREGATION emerges: orange (type 1) forms tight clusters at C2 peaks, blue (type 0) forms diffuse cloud in inter-spot regions. Reversed consumption/production creates feedback asymmetry — type 0 reinforces pattern (positive feedback), type 1 counteracts (negative feedback) — yet hexagonal mode persists. Pattern not converged (plateau=0.00).
Mutation: params consumption/production: [45,-45] for both → [45,-45] vs [-45,45] (reversed type 1); n_particle_types: 1→2; shuffle: true
Observation: Same-sign mobility with reversed feedback creates novel type-segregation: both types migrate to same hexagonal spots but occupy different micro-niches (tight cluster vs diffuse halo). Hexagonal attractor is robust even against one type providing negative feedback. Plateau=0.00 suggests competition between positive and negative feedback types prevents convergence — dynamic equilibrium rather than steady state.
Next: parent=root

## Iter 48: 8/10 - FHN Dv=0.15 → SQUARE SYMMETRY CONFIRMED (robust regime)
Node: id=48, parent=45
Mode/Strategy: explore (boundary-probe) — test Dv=0.15 to probe upper boundary of square regime
Config: mesh_model=PDE_Diffusiophoresis_FHN, Du=0.5, Dv=0.15, a=0.7, b=0.8, eps=0.08, I=0, time_scale=20
n_particle_types: 1, shuffle=true (no effect on 1-type)
params: M1=-4, M2=4, consumption=45, production=-45
n_frames: 6000, delta_t=0.0005, n_particles=9600, n_nodes=10000
Metrics: entropy=0.7686, plateau=0.0000, in_box=100.0%, clustering=0.0858
C1_std=1.1348, C2_std=0.6004, pattern_growth=120.08
Assessment:
  - Symmetry: **square** (four-fold, same as Dv=0.1 iter 45)
  - Particles: network (square grid walls with central void)
  - Stability: transient (plateau=0, ongoing dynamics)
  - Novelty: variant (confirms Dv=0.1 square is robust, not narrow island)
Visual: C1 field shows CLEAR square central feature with rectangular boundary layers — even MORE pronounced square geometry than Dv=0.1. Particles form walls along square field boundaries. Evolution: rings → rectangular deformation → square grid. Late frames show additional sub-structures within square grid (more complexity than Dv=0.1). C1_std=1.135 comparable to Dv=0.1's 1.06 — pattern amplitudes similar.
Mutation: Dv: 0.1 -> 0.15 (parent iter 45)
Observation: **Dv=0.15 STILL produces square symmetry.** The hex→square bifurcation is sharp and located in [0.05, 0.1). Once past the bifurcation, square mode is ROBUST. The symmetry phase diagram is: Dv=0 → hexagonal, Dv=0.05 → disordered/transitional, Dv≥0.1 → square. Clustering=0.09 (positive, less negative than Dv=0.1's -0.08 or Dv=0.05's -0.15) suggests slightly thicker walls at higher Dv.

### Block 6 Summary (Iters 41-48)
**Focus**: Tested Schnakenberg PDE variant + FHN Dv symmetry selector exploration
**Key discoveries**:
1. **Schnakenberg = radial only** (like Gray-Scott). gamma=200→single radial mode (plateau=0.62!), gamma=500→concentric rings, gamma=1000→NaN. Added to Established Principles.
2. **FHN Dv is a symmetry selector**: Dv=0→hexagonal, Dv=0.05→disordered transitional, Dv≥0.1→SQUARE/GRID (novel symmetry!). Sharp bifurcation in [0.05, 0.1).
3. **Square symmetry requires 1-type**: 2-type opposing mobilities destroy square→hexagonal (iter 46).
4. **Square symmetry is robust**: Confirmed at Dv=0.1 AND Dv=0.15.
5. **Schnakenberg achieves genuine steady state** (plateau=0.62 at gamma=200) — unique among all PDE variants tested.
Score progression: 7-5-7-7-9-8-8-8

Next: parent=45
---

## Iter 47: 8/10 - FHN Dv=0.05 → Disordered transitional (hex↔square boundary)
Node: id=47, parent=45
Mode/Strategy: explore (boundary-probe) — map Dv symmetry selector between hex (Dv=0) and square (Dv=0.1)
Config: FHN Du=0.5, Dv=0.05, a=0.7, b=0.8, eps=0.08, 1-type, no shuffle
n_particle_types: 1
Metrics: entropy=0.8101, plateau=0.0000, in_box=100.0%, clustering=-0.1502
Assessment:
  - Symmetry: other (disordered/transitional)
  - Particles: network (irregular filamentary walls)
  - Stability: transient (still evolving at frame 120)
  - Novelty: novel (first disordered transitional state between symmetry classes)
Visual: Early frames show concentric rings with rectangular corner distortions from boundary coupling. Mid-simulation develops pentagonal/spiral asymmetric distortions. Final frames show highly irregular broken pattern — C1 field has disordered lobes with no consistent symmetry. Particles form irregular filamentary network with thin walls and scattered internal clusters. Not hexagonal, not square — caught between two attractors. C1_std=1.018 (strong pattern), clustering=-0.15 (most negative ever = thinnest network walls).
Mutation: Dv: 0.1 -> 0.05 (halved inhibitor diffusion from square symmetry config)
Observation: Dv=0.05 is in a **disordered transitional zone** between hex (Dv=0) and square (Dv=0.1). The symmetry transition is NOT gradual — it's a sharp bifurcation. Dv=0.05 cannot stabilize either attractor, producing chaotic/dynamic pattern evolution. The hex→square transition likely occurs over a narrow Dv window ~[0.07-0.09]. This pattern has biological interest as a "frustrated" system analogous to quasicrystalline or glassy states. clustering=-0.15 is the most negative ever recorded, indicating particles concentrate into the thinnest possible walls.
Next: parent=45 (square symmetry) for iter 48 — test Dv=0.15 to probe upper boundary of square regime

## Iter 46: 8/10 - FHN Dv=0.1 + 2-type shuffle → Hexagonal (NOT square)
Node: id=46, parent=45
Mode/Strategy: exploit + multi-type (test square symmetry robustness with opposing mobilities)
Config: FHN Du=0.5, Dv=0.1, a=0.7, b=0.8, eps=0.08, params=[[-4,4,45,-45,1.6,1.0,1.6,1.5],[4,-4,-45,45,1.8,1.0,1.1,1.9]], n_particle_types=2, shuffle=true, n_frames=6000
n_particle_types: 2
Metrics: entropy=0.8123, plateau=0.0000, in_box=100.0%, clustering=0.2595
Assessment:
  - Symmetry: hexagonal
  - Particles: segregated (blue=clustered at C1 peaks, orange=network between)
  - Stability: transient (plateau=0, pattern still growing at 6000 frames)
  - Novelty: variant (FHN hexagonal with type segregation, NOT square)
Visual: Early radial phase separation (blue disk center, orange ring) → hexagonal spot formation (~6 spots mid-sim) → proliferating spots filling domain (~20+ final). C1 bright spots on dark background, C2 dark spots on bright background. Blue particles cluster at C1 peaks, orange particles form filamentary network connecting spots. Strong type segregation. C1_std=1.277 is strongest FHN field pattern ever recorded.
Mutation: n_particle_types: 1→2, shuffle: false→true (from iter 45's square config)
Observation: **Square symmetry from iter 45 is NOT robust to multi-type particles.** Adding opposing mobilities (M1=±4) disrupts the boundary-coupling mechanism that produced four-fold symmetry. Instead, standard FHN hexagonal spots emerge with type-segregated particles. This suggests square symmetry requires uniform particle response — opposing mobilities create competing gradients that break the boundary coupling. clustering=0.26 is intermediate between iter 45's -0.08 (network) and Brusselator's 0.45-0.53 (tight clusters), reflecting mixed cluster+network morphology.
Next: parent=45

## Iter 43: 8/10 - Schnakenberg gamma=500 → Concentric rings with peripheral spots
Node: id=43, parent=41
Mode/Strategy: exploit (moderate gamma increase after gamma=1000 crash)
Config: mesh_model=PDE_Diffusiophoresis_Schnakenberg, Du=0.05, Dv=1.0, gamma=500, a=0.1, b=0.9, M1=-4, M2=4
n_particle_types: 1
Metrics: entropy=0.7801, plateau=0.0000, in_box=100.0%, clustering=0.5341
Assessment:
  - Symmetry: radial (concentric rings with peripheral spot hints)
  - Particles: clustered (ring-following with central concentration)
  - Stability: transient (plateau=0, still evolving at 6000 frames)
  - Novelty: variant (Schnakenberg radial mode, distinct from Brusselator hex)
Visual: C1/C2 fields develop prominent concentric ring structure - dark central void surrounded by bright ring, with 3-4 spot-like features at periphery. Particles show density modulation following ring pattern: central accumulation, ring-shaped depletion, outer concentration band. Much stronger field variation than gamma=200 (C1_std=0.27 vs 0.01). Pattern_growth=70x indicates strong Turing instability.
Mutation: gamma: 200 -> 500
Observation: gamma=500 successfully produces multi-mode Schnakenberg patterns (vs 1-mode at gamma=200) without NaN divergence (gamma=1000). However, plateau dropped from 0.62 to 0.00 - higher gamma pushes Schnakenberg out of steady-state into oscillatory/evolving regime. The radial symmetry persists (no hexagonal breaking). Schnakenberg at gamma=500 resembles Gray-Scott more than Brusselator - both produce concentric rings rather than hexagonal spots. Strong field variation (C1_std=0.27) suggests Turing instability is active but mode selection favors rings.
Next: parent=43

## Iter 42: 2/10 - Schnakenberg gamma=1000 CATASTROPHIC FAILURE (NaN divergence)
Node: id=42, parent=41
Mode/Strategy: exploit (increase gamma for more Turing modes)
Config: mesh_model=PDE_Diffusiophoresis_Schnakenberg, Du=0.05, Dv=1.0, gamma=1000, a=0.1, b=0.9, Pe=1.0, cons=180, prod=-180
n_particle_types: 1, shuffle=false
Metrics: entropy=0.0000, plateau=0.0000, in_box=0.0%, clustering=NaN
Assessment:
  - Symmetry: none
  - Particles: collapsed (escaped/NaN)
  - Stability: unstable (NaN divergence by frame 2)
  - Novelty: repeat (numerical instability, similar to iter 1)
Visual: Frame 1 shows initial noisy field and particle disk. By frame 2 ALL fields are NaN (blank white), ALL particles escaped (empty domain). Complete numerical blow-up. Gamma=1000 amplifies Schnakenberg reaction terms 5x beyond gamma=200, exceeding CFL stability limit for delta_t=0.0005.
Mutation: gamma: 200 -> 1000 (5x increase to shorten Turing wavelength)
Observation: **gamma=1000 is numerically unstable** with delta_t=0.0005. The Schnakenberg reaction terms R_u = gamma*(a - u + u²v) diverge immediately. Need more conservative increase: gamma=500 (2.5x) or smaller delta_t. Schnakenberg may have narrower stability window than Brusselator due to quadratic autocatalysis.
Next: parent=41, try gamma=500 (moderate increase) with same delta_t

---

## Iter 41: 7/10 - Schnakenberg SINGLE RADIAL MODE (too few modes)
Node: id=41, parent=root
Mode/Strategy: explore (first test of Schnakenberg PDE variant)
Config: mesh_model=PDE_Diffusiophoresis_Schnakenberg, Du=0.05, Dv=1.0, gamma=200, a=0.1, b=0.9, Pe=1.0, cons=180, prod=-180
n_particle_types: 1, shuffle=false
Metrics: entropy=0.7784, plateau=0.6207, in_box=100.0%, clustering=0.5351
Assessment:
  - Symmetry: radial
  - Particles: clustered (single large disk)
  - Stability: stable (plateau=0.62 - first time >0.5!)
  - Novelty: variant (similar to Gray-Scott radial, but more stable)
Visual: Single large-scale Turing mode. C1 shows blue center/yellow surround, C2 complementary. Particles form one compact circular cluster. Very weak field variation (C1_std=0.012, C2_std=0.016). Pattern forms early and remains static (stable plateau). No hexagonal breaking even at 6000 frames.
Mutation: mesh_model: Diffusiophoresis_Mesh → PDE_Diffusiophoresis_Schnakenberg; gamma=200, a=0.1, b=0.9 (new model)
Observation: Schnakenberg produces Turing pattern but at too large a wavelength - only 1 mode fits in domain. Need higher gamma or adjusted diffusion to get shorter wavelength (multiple spots). Notable: plateau=0.62 is first >0.5 value - Schnakenberg is genuinely more stable than Brusselator/FHN (which always show plateau~0). Field variation is 5-10x weaker than Brusselator.
Next: parent=41, increase gamma to 1000 for more modes

---

## Iter 40: 8/10 - B=9 AT TURING BOUNDARY → TRANSITIONAL MIXED MODE (NOVEL!)
Node: id=40, parent=38 (highest UCB=2.671)
Mode/Strategy: exploit (testing B=9 at Turing boundary for stripe/hexagonal transition)
Config: params_mesh=[D1=1.0, Da_c=15, A=2.828, B=9.0, mu=0, chi=0], D2=0.25, n_particle_types=2, shuffle=true
n_particle_types: 2
Metrics: entropy=[0.735], plateau=[0.000], in_box=[100.0]%, clustering=[0.177]
C1_std=2.170, C2_std=2.832, pattern_growth=566.4
Assessment:
  - Symmetry: other (mixed/transitional - NOT pure stripes, NOT pure hexagonal)
  - Particles: network (filamentary along field boundaries, weak phase separation)
  - Stability: transient (pattern_growth=566 still evolving)
  - Novelty: novel (transitional pattern between hex and stripe regimes)
Visual: C1 field shows MIXED morphology - elongated labyrinthine patches coexisting with some isolated spots. Neither clean hexagonal array nor clean labyrinthine stripes. Particles form filaments along field boundaries with orange dominant over blue. Phase separation is weak (clustering=0.177, lowest of stripe-like configs). Transition point between B=8 (hex) and B=10 (stripes).
Mutation: A: 3.0->2.828, B: 10->9.0 (maintaining B/[1+A^2]=1.0, testing intermediate B)
Observation: B=9 IS the transition point! Pattern is neither purely hexagonal nor purely labyrinthine - it is a MIXED MODE with both morphologies coexisting. This confirms the stripe/hex transition occurs between B=8 and B=10, with B=9 producing a co-existence regime. Clustering drops to 0.177 (vs 0.35-0.41 for B=10 stripes, 0.35 for B=8 hex) suggesting the mixed mode prevents strong particle organization.
Next: Block 5 complete. Block 6 planning.

## Block 5 Summary (Iters 33-40)

Focus: Exploit the stripe discovery from Block 4. Tuned A/B boundary, explored sub-Turing regime, tested multi-type stripe dynamics.

Key Findings:
1. Turing bifurcation structure FULLY MAPPED: B/[1+A^2]<1 = no patterns (iter 34); =1.0+LOW B = hex (iter 35); =1.0+MED B(8) = hex (iter 39); =1.0+B=9 = MIXED (iter 40); =1.0+HIGH B(10) = stripes (iters 36-38); >1 = hex spots (iter 33)
2. Stripe mode is ROBUST to particle types: 1-type (iter 31), 2-type shuffled (iter 36), 3-type ordered (iter 37), 3-type shuffled (iter 38) ALL produce stripes at A=3, B=10
3. Neutral particle ring mechanism: 3-type with M=0 neutral type forms coherent ring at B=10 (iter 37), partially disrupted by shuffle (iter 38)
4. Stripe/hex transition discovered at B=9: Mixed coexistence regime with neither pure hexagonal nor pure labyrinthine morphology (iter 40)
5. Sub-Turing regime = NO patterns: B/[1+A^2]<1 produces disordered noise, opposing mobilities cannot compensate (iter 34)

Score progression: 8-6-8-9-9-8-8-8
Best: 9/10 at iters 36 (stripes+phase separation), 37 (stripes+neutral ring)

---

## Iter 39: 8/10 - INTERMEDIATE B=8 AT TURING BOUNDARY → HEXAGONAL (NOT STRIPES)
Node: id=39, parent=37 (highest UCB=2.055)
Mode/Strategy: exploit (testing intermediate B value to find stripe/hexagonal transition)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=1, shuffle_particle_types=false (odd iter), boundary=periodic
params_mesh: [[1.0, 15, 2.646, 8.0, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]]
  A=2.646, B=8.0 → B/[1+A²]=8.0/(1+7.0013)=0.9998≈1.0 (Turing boundary)
n_particle_types: 1
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000
Metrics: entropy=0.557, plateau=0.042, in_box=100.0%, clustering=0.351, pattern_growth=342.22
Score: 8/10
Assessment:
  - Symmetry: **hexagonal** (clear hexagonal spot array, ~20-25 spots)
  - Particles: **clustered** (tight crescent-shaped clusters around field peaks)
  - Stability: stable (100% particles retained, low plateau=0.042)
  - Novelty: **variant** (confirms B=8 is NOT high enough for stripes)
Visual: Clear hexagonal spot pattern. C1 shows ~20-25 dark spots in roughly hexagonal arrangement with bright green/yellow rims. C2 shows complementary bright spots (orange/yellow) against purple background. Particles form crescent-shaped clusters around the high-C2 peaks with some dispersed particles between clusters. Evolution: uniform → ring → hexagonal breakup follows standard Brusselator pathway. No stripe/labyrinth character observed at any stage.
Mutation: A: 3.0→2.646, B: 10→8.0 (maintaining B/[1+A²]≈1.0 but reducing absolute reaction rate)
Observation: **B=8 AT TURING BOUNDARY → HEXAGONAL, NOT STRIPES.** This narrows the stripe/hexagonal transition: B=5→hexagonal (iter 35), B=8→hexagonal (iter 39), B=10→stripes (iters 31,36-38). The stripe mode requires B≥9-10, not just B/[1+A²]≈1. Higher absolute B drives stronger nonlinear reaction dynamics that favor stripe/labyrinth over spot patterns. pattern_growth=342 (intermediate between B=5's 194 and B=10's 700-800) suggests gradual strengthening of dynamics with B. Clustering=0.351 is lower than typical hexagonal (0.45-0.53) suggesting weaker particle concentration at the boundary.
Next: parent=38

---

## Iter 38: 8/10 - SHUFFLED 3-TYPE STRIPES: NEUTRAL RING PARTIALLY PRESERVED
Node: id=38, parent=37
Mode/Strategy: exploit + shuffle (mandatory even iteration: shuffle=true for direct comparison with iter 37)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=3, shuffle_particle_types=true (even iter), boundary=periodic
params_mesh: [[1.0, 15, 3.0, 10.0, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [0, 0, 0, 0, 1.6, 1.0, 1.6, 1.7], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]]
  Type 0 (blue): M1=-4, M2=4 (attracted to C1 peaks)
  Type 1 (orange): M1=0, M2=0 (NEUTRAL - passive tracer)
  Type 2 (green): M1=4, M2=-4 (repelled from C1 peaks)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600 (3200 each), n_nodes: 10000
Metrics: entropy=0.808, plateau=0.00, in_box=100.0%, clustering=0.410, pattern_growth=799.62
Score: 8/10
Assessment:
  - Symmetry: **stripes/labyrinth** (meandering stripe pattern confirmed)
  - Particles: **segregated/network** (types self-organize by mobility even from shuffled initial condition)
  - Stability: stable (100% particles retained)
  - Novelty: **variant** (shuffled version of iter 37's novel neutral ring finding)
Visual: Stripe/labyrinth field pattern robust. C1 shows dark central void with bright meandering channels; C2 complementary. Particles: Blue (Type 0) concentrates in filamentary networks near center tracking C1 gradients. Orange (Type 1, neutral) forms intermediate ring-like distribution but MORE DISPERSED than iter 37 — the neutral ring is less coherent when starting from shuffled positions. Green (Type 2) forms outer ring/halo. Despite shuffled initial conditions, the 3 types self-organize into distinct radial zones driven by their mobility differences. The labyrinthine stripe structure is fully preserved.
Mutation: parent=37 (3 types, shuffle=false → neutral ring) → shuffle_particle_types=true (same params otherwise)
Observation: **SHUFFLE PARTIALLY DISRUPTS NEUTRAL RING BUT PRESERVES MOBILITY-DRIVEN SEGREGATION.** Compared to iter 37 (shuffle=false), the neutral ring (orange) is less coherent — entropy increased (0.808 vs 0.786) and clustering increased (0.410 vs 0.372) suggesting tighter local clusters but less global coherence. The key finding is that neutral particles starting from randomized positions do NOT re-form a perfect ring but still concentrate at an intermediate radius. Active types (blue/green) still sort by mobility direction. The stripe field pattern itself is completely insensitive to initial particle arrangement — confirming it's driven purely by Brusselator dynamics. pattern_growth slightly higher (799 vs 794), consistent with iter 37.
Next: parent=37

---

## Iter 37: ★ 9/10 ★ - STRIPES + 3 TYPES = NEUTRAL RING + ACTIVE FILAMENTS (NOVEL!)
Node: id=37, parent=36
Mode/Strategy: exploit (testing stripe config A=3, B=10 with 3 types: attract/neutral/repel)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=3, shuffle_particle_types=false (odd iter), boundary=periodic
params_mesh: [[1.0, 15, 3.0, 10.0, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [0, 0, 0, 0, 1.6, 1.0, 1.6, 1.7], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]]
  Type 0 (blue): M1=-4, M2=4 (attracted to C1 peaks)
  Type 1 (orange): M1=0, M2=0 (NEUTRAL - passive tracer)
  Type 2 (green): M1=4, M2=-4 (repelled from C1 peaks)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600 (3200 each), n_nodes: 10000
Metrics: entropy=0.786, plateau=0.00, in_box=100.0%, clustering=0.372, pattern_growth=793.99
Score: 9/10
Assessment:
  - Symmetry: **stripes/labyrinth** (meandering bands confirmed)
  - Particles: **NOVEL ORGANIZATION** - neutral ring + active filaments
  - Stability: stable (100% particles retained)
  - Novelty: **NOVEL** (first 3-type stripe with neutral tracer → emergent ring structure)
Visual: ★ STRIPES + 3 TYPES CREATES NOVEL ORGANIZATION ★ The stripe configuration robustly produces labyrinthine patterns with 3 types. CRITICAL FINDING: The neutral (Type 1, orange) particles form a COHERENT RING/TORUS structure while active types (blue=Type 0, green=Type 2) disperse along stripe boundaries as filaments. C1 shows dark central void with meandering bright channels; C2 shows complementary pattern. Without shuffle, initial concentric type arrangement (blue center, orange ring, green outer) partially persists but is modulated by field-driven dynamics.
Mutation: parent=36 (2 types, shuffled) → 3 types with neutral middle type, shuffle=false
Observation: **NEUTRAL TRACER CREATES EMERGENT RING**. The zero-mobility Type 1 particles remain more coherent (forming ring) because they only respond to particle-particle repulsion, not field gradients. Active Types 0 and 2 with opposing mobilities disperse along stripe boundaries tracking field gradients. This creates a novel tri-layer organization: active filaments (blue/green) + neutral ring (orange). The entropy=0.786 is high (more spread than iter 36's 0.657) due to the ring structure taking up space. clustering=0.372 similar to iter 36. The stripe mode is now confirmed robust across 1-type (iter 31), 2-type shuffled (iter 36), and 3-type ordered (iter 37).
Next: parent=37

---

## Iter 36: 9/10 - ★ STRIPES + MULTI-TYPE = STRIPE-ALIGNED PHASE SEPARATION ★
Node: id=36, parent=35
Mode/Strategy: exploit (testing stripe config A=3, B=10 with 2 opposing mobility types + shuffle)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 3.0, 10.0, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types opposing)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000
Metrics: entropy=0.657, plateau=0.00, in_box=100.0%, clustering=0.362, pattern_growth=692.01
Score: 9/10
Assessment:
  - Symmetry: **stripes/labyrinth** (meandering bands, NOT hexagonal spots)
  - Particles: **network/segregated** (particles align along stripe boundaries, types separate)
  - Stability: stable (100% particles retained, plateau=0 but pattern maintains)
  - Novelty: **NOVEL** (first demonstration of stripe + multi-type phase separation)
Visual: ★ STRIPES CONFIRMED with 2 types! ★ The stripe configuration (A=3, B=10, B/[1+A²]=1.0) robustly produces labyrinthine stripes even with 2-type opposing mobilities. Critically, particles organize along stripe boundaries creating network/filamentary structure. Phase separation occurs ALONG the stripes - blue and orange types segregate to different stripe flanks. This is qualitatively different from hexagonal phase separation where types occupy different spot clusters. Initial shuffled distribution (frame 1) evolves to stripe-aligned segregation (frames 8-10).
Mutation: parent=35 (A=2, B=5→hexagonal) → (A=3, B=10→stripes) + n_particle_types=2 + shuffle=true
Observation: **STRIPE MODE ROBUSTNESS CONFIRMED**. A=3, B=10 (B/[1+A²]=1.0) produces stripes regardless of particle type count (iter 31: 1 type, iter 36: 2 types). This confirms the mode selection is driven by FIELD DYNAMICS (absolute A,B values at Turing boundary), not particle interactions. The 2-type opposing mobility system creates a new organizational pattern: stripe-flanking segregation, where types occupy opposite sides of labyrinthine bands. This is distinct from the hexagonal cluster-occupation phase separation. clustering=0.362 is moderate (between hexagonal ~0.4-0.5 and network ~0.26).

---

## Iter 35: 8/10 - Stripe Robustness Test REFUTED → A,B Absolute Values Matter
Node: id=35, parent=31
Mode/Strategy: exploit (testing if stripe emerges at different A,B with same Turing ratio)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=1, shuffle_particle_types=false, boundary=periodic
params_mesh: [[1.0, 15, 2.0, 5.0, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]] (1 type)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000
Metrics: entropy=0.607, plateau=0.00, in_box=100.0%, clustering=0.266, pattern_growth=193.56
Score: 8/10
Assessment:
  - Symmetry: **hexagonal** (NOT stripes - surprising!)
  - Particles: clustered (moderate, ~15-20 clusters)
  - Stability: stable (100% particles retained, lower pattern_growth)
  - Novelty: variant (reveals A,B absolute values matter, not just ratio)
Visual: At A=2, B=5 → B/[1+A²]=5/5=1.0 (EXACT same ratio as iter 31's stripes). However, pattern is HEXAGONAL, not stripes! Clear spot pattern develops with ~15-20 well-defined clusters. C1 and C2 fields show complementary hexagonal spots. Particles track field peaks faithfully.
Mutation: A,B: 3.0, 10 → 2.0, 5.0 (same B/[1+A²]=1.0 ratio but different absolute values)
Observation: **STRIPES REQUIRE MORE THAN JUST B/[1+A²]=1.0**. The ratio is necessary but NOT sufficient. With A=3, B=10 (iter 31) we got stripes. With A=2, B=5 (same ratio=1.0) we get hexagonal. KEY INSIGHT: The absolute reaction rates (proportional to B and A) affect mode selection. Higher B=10 may push system into stripe regime through stronger nonlinearity. Lower B=5 allows hexagonal mode to dominate. pattern_growth=193.6 vs 711 (iter 31) suggests lower B = more stable dynamics. clustering=0.266 is lower than typical hexagonal (0.4-0.5), indicating weaker clustering at lower reaction rates.

---

## Iter 34: 6/10 - Sub-Turing Regime NOVEL → Disordered/Uniform
Node: id=34, parent=33
Mode/Strategy: explore (testing below Turing instability threshold)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 3.2, 10, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types opposing)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000
Metrics: entropy=0.790, plateau=0.00, in_box=100.0%, clustering=0.040, pattern_growth=508.41
Score: 6/10
Assessment:
  - Symmetry: **none** (disordered/noisy field pattern)
  - Particles: **uniform** (no clustering, no phase separation)
  - Stability: transient (plateau=0, high pattern_growth but no coherent structures)
  - Novelty: **NOVEL** (first time testing sub-Turing regime)
Visual: At A=3.2, B=10 → B/[1+A²]=10/11.24=0.89 (BELOW Turing boundary). Fields show speckled/mottled noise rather than coherent spots or stripes. Particles (both types) remain completely uniform and intermixed - NO phase separation despite opposing mobilities. This is the first time we've seen the system fail to produce organized patterns.
Mutation: A: 2.9 → 3.2 (B/[1+A²]: 1.063 → 0.89, going BELOW Turing boundary)
Observation: **SUB-TURING = NO PATTERN FORMATION**. Below B/[1+A²]=1, the Brusselator cannot sustain Turing instability. Fields become noisy/disordered and particles remain uniform. Key insight: Pattern formation (stripes OR hexagonal) REQUIRES B/[1+A²]≥1. The boundary (=1) gives stripes, above (>1) gives hexagonal, below (<1) gives NOTHING. This completes our understanding of the Turing bifurcation structure. clustering=0.04 is the LOWEST we've ever recorded (vs typical 0.3-0.5 for hexagonal).

---

## Iter 33: 8/10 - Stripe Stabilization Test REFUTED → Hexagonal Returns
Node: id=33, parent=31
Mode/Strategy: exploit (attempting to stabilize stripes by moving slightly above Turing boundary)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=1, shuffle_particle_types=false, boundary=periodic
params_mesh: [[1.0, 15, 2.9, 10, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]] (1 type)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000
Metrics: entropy=0.648, plateau=0.00, in_box=100.0%, clustering=0.477, pattern_growth=722.83
Score: 8/10
Assessment:
  - Symmetry: **hexagonal** (NOT stripes - boundary hypothesis refined)
  - Particles: clustered (tight ~15-20 clusters at C1 peaks)
  - Stability: stable (100% particles retained)
  - Novelty: variant (confirms stripe threshold is EXACTLY B/[1+A²]=1.0)
Visual: At A=2.9, B=10 → B/[1+A²]=10/9.41=1.063, we are ABOVE the Turing boundary. Pattern reverts to hexagonal spots (~15-20 visible). Particles form tight clusters at spot centers with some ring-like structures around spots at late times. C1 field shows characteristic spot pattern with dark (low C1) background. This is the SAME hexagonal mode as previous iterations (4.5, 6.5) just with different A,B values.
Mutation: A: 3.0 → 2.9 (B/[1+A²]: 1.0 → 1.063, moving above boundary)
Observation: **STRIPE MODE REQUIRES EXACT TURING BOUNDARY**. Hypothesis that "slightly above boundary would stabilize stripes" was REFUTED. Even B/[1+A²]=1.063 (6% above boundary) reverts to hexagonal spots. This establishes that stripes/labyrinth require B/[1+A²]≈1.0 precisely. The hexagonal attractor dominates everywhere B/[1+A²]>1. pattern_growth remains high (722.8), suggesting high-B systems are inherently dynamic regardless of mode.

---

## Iter 32: 8/10 - Stripe Pattern Robustness with 3 Types
Node: id=32, parent=31
Mode/Strategy: robustness-test (verify stripe pattern with 3 particle types)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=3, shuffle_particle_types=false, boundary=periodic
params_mesh: [[1.0, 15, 3.0, 10, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9], [0, 0, 0, 0, 1.8, 1.0, 2.0, 1.0]] (3 types)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000
Metrics: entropy=0.769, plateau=0.00, in_box=100.0%, clustering=0.279, pattern_growth=686.17
Score: 8/10
Assessment:
  - Symmetry: **STRIPES/LABYRINTH** (confirmed - interconnected filaments)
  - Particles: network (all 3 types follow stripe boundaries)
  - Stability: stable (100% particles retained)
  - Novelty: variant (stripe pattern confirmed but metrics differ from 1-type)
Visual: Concentric 3-type initialization → develops into LABYRINTHINE structure similar to iter 31. C1/C2 fields show interconnected stripe-like domains. All three particle types (blue=attracted to C1, orange=repelled, green=neutral) organize along stripe boundaries. Pattern is maintained but slightly more dispersed than 2-type shuffled version.
Mutation: n_particle_types: 2 → 3, shuffle: true → false (testing stripe robustness with 3 types)
Observation: **STRIPE PATTERN IS PARTICLE-TYPE-INDEPENDENT** - A=3, B=10 (B/[1+A²]=1.0) produces stripes/labyrinth with 1, 2, AND 3 particle types. Comparing: iter 31 (2-type, shuffled): clustering=0.392, entropy=0.698; iter 32 (3-type, ordered): clustering=0.279, entropy=0.769. The concentric initialization spreads particles more (lower clustering) but stripe mode selection is driven by field dynamics, not particle organization. pattern_growth=686 remains very high (consistent with instability boundary dynamics).

---

## Block 4 Summary (Iters 25-32)

**Hypothesis tested**: Can we break the hexagonal attractor in Brusselator field dynamics?

**Exhaustive exploration (ALL produced hexagonal)**:
- chi (cross-diffusion): 0.0 → 0.05 → 0.2 (iters 25-26)
- D1/D2 ratio: 4 → 2 → 1 (iters 27-28)
- n_nodes (domain size): 10000 → 22500 (iter 29)
- B (Turing depth): 6.5 → 10 with A=4.5 (iter 30)

**★ BREAKTHROUGH (iter 31)**: A=3.0, B=10 → B/[1+A²]=1.0 (exactly at Turing boundary) → **STRIPES/LABYRINTH!**

**Key discovery**: **TURING BOUNDARY POSITIONING CONTROLS MODE SELECTION**
- B/[1+A²] = 1.0 (at boundary): stripes/labyrinth
- B/[1+A²] > 1 (deep unstable): hexagonal spots

**Signature metrics for stripes vs spots**:
| Pattern | Entropy | Clustering | pattern_growth |
|---------|---------|------------|----------------|
| Hexagonal (typical) | 0.81-0.88 | 0.30-0.34 | 30-60 |
| Stripes/Labyrinth | 0.70-0.77 | 0.28-0.39 | 680-710 |

**Score progression**: 8→8→8→8→8→8→★9→8

---

## Iter 31: 9/10 ★ BREAKTHROUGH - STRIPES/LABYRINTH!
Node: id=31, parent=30
Mode/Strategy: explore (A parameter modification - at Turing instability BOUNDARY B=1+A²)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 3.0, 10, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types with opposing mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000
Metrics: entropy=0.698, plateau=0.00, in_box=100.0%, clustering=0.392, pattern_growth=711.94
Score: 9/10
Assessment:
  - Symmetry: **STRIPES/LABYRINTH** (interconnected filaments, NOT hexagonal spots!)
  - Particles: **NETWORK** (filamentary organization following stripe boundaries)
  - Stability: stable (100% particles retained)
  - Novelty: **NOVEL** (FIRST non-hexagonal Brusselator pattern in 31 iterations!)
Visual: **DRAMATICALLY DIFFERENT PATTERN!** Initial shuffled 2-type → develops into INTERCONNECTED LABYRINTHINE structure. C1/C2 fields show elongated stripe-like domains instead of isolated spots. Particles organize into FILAMENTARY NETWORKS tracing stripe boundaries. Late frames (columns 7-10) show clear stripe-like features with connected particle chains. This is the FIRST successful symmetry breaking away from hexagonal attractor!
Mutation: A: 4.5 → 3.0 (at Turing boundary: B=10, 1+A²=10, so B/[1+A²]=1.0)
Observation: **★ BREAKTHROUGH - A=3, B=10 BREAKS HEXAGONAL ATTRACTOR! ★** By placing the system EXACTLY at Turing instability boundary (B=1+A²=10), the mode selection SHIFTS from spots to stripes/labyrinth! Key evidence: (1) Visual inspection shows interconnected filaments NOT isolated spots; (2) entropy=0.698 LOWER than hexagonal (0.81-0.88) indicating more structured/connected pattern; (3) clustering=0.392 HIGHER than hexagonal (0.30-0.34) indicating tighter particle aggregation within filaments; (4) pattern_growth=711.94 EXTREMELY HIGH (12x typical!) indicating active dynamics at the instability boundary. **NEW PRINCIPLE**: B/[1+A²]=1 (at boundary) selects stripes; B/[1+A²]>1 (deep in unstable) selects spots!
Next: parent=31 (confirm with n_particle_types=3 and shuffle=false per alternation rule)

## Iter 30: 8/10
Node: id=30, parent=29
Mode/Strategy: explore (testing B=10 - deeper into Turing unstable regime, B>>1+A²)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 10, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types with opposing mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 10000 (reverted to 100x100)
Metrics: entropy=0.878, plateau=0.00, in_box=100.0%, clustering=0.301, pattern_growth=58.59
Score: 8/10
Assessment:
  - Symmetry: hexagonal (6-7 spot: center spot + peripheral ring)
  - Particles: clustered (tight clusters at spots, phase separated by type)
  - Stability: stable (100% particles retained)
  - Novelty: repeat (same hexagonal as all Block 4 tests)
Visual: Shuffled 2-type to concentric rings to hexagonal spot pattern. C1/C2 fields show clear complementary hexagonal structure. Blue particles cluster at spot locations, orange particles at outer ring and interstitial regions. Pattern develops faster due to higher B but settles to same hexagonal mode.
Mutation: B: 6.5 -> 10 (deeper Turing instability, now B=10 >> 1+A^2=21.25, so B/[1+A^2]=0.47)
Observation: **B=10 STILL produces hexagonal!** The higher B increases pattern_growth (58.59 vs ~30-38 for B=6.5) indicating faster dynamics, but does NOT change the final mode selection. Clustering=0.301 slightly lower than iter 29 (0.337), entropy=0.878 slightly higher - both indicating slightly more dispersed organization. **B EXPLORATION EXHAUSTED** - sixth consecutive hexagonal result in Block 4 despite chi (0-0.2), D1/D2 (1-4), n_nodes (10000-22500), B (6.5-10). Need A parameter or code modification.
Next: parent=30 (try A parameter modification A: 4.5 -> 3.0 to change activator production)

## Iter 29: 8/10
Node: id=29, parent=28
Mode/Strategy: explore (domain size - testing smaller mesh 150×150 to constrain wavelengths)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types with opposing mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600, n_nodes: 22500 (150×150)
Metrics: entropy=0.812, plateau=0.00, in_box=100.0%, clustering=0.337, pattern_growth=37.82
Score: 8/10
Assessment:
  - Symmetry: hexagonal (7-spot: 1 center + 6 peripheral)
  - Particles: clustered (tight clusters at spots)
  - Stability: stable (100% particles retained)
  - Novelty: repeat (same hexagonal as all Block 4 tests)
Visual: Smaller domain (150×150) produces IDENTICAL pattern to standard domain (100×100). Shuffled 2-type → concentric rings → hexagonal 7-spot. Blue particles at spots, orange at periphery. Fields show complementary C1/C2 hexagonal structure.
Mutation: n_nodes: 10000 → 22500 (domain size 100×100 → 150×150)
Observation: **SMALLER DOMAIN DOES NOT BREAK HEXAGONAL!** The mesh resolution change has NO effect on mode selection - same 7-spot hexagonal attractor. Clustering=0.337 is slightly higher than iters 25-28 (~0.31-0.32). Entropy=0.812 slightly lower. The Brusselator hexagonal attractor is EXTREMELY robust across: chi=0-0.2, D1/D2=1-4, n_nodes=10000-22500. **EXHAUSTED PARAMETER EXPLORATION** - need A/B parameter modification or code change to break this attractor.
Next: parent=29 (try B parameter increase - deeper Turing instability B>1+A²)

## Iter 28: 8/10
Node: id=28, parent=27
Mode/Strategy: explore (testing D1/D2=1 - equal diffusion boundary between spots/stripes per Turing theory)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0], [1.0, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types with opposing mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.854, plateau=0.00, in_box=100.0%, clustering=0.313, pattern_growth=29.78
Score: 8/10
Assessment:
  - Symmetry: hexagonal (7-spot: 1 center + 6 peripheral)
  - Particles: clustered (phase separated - blue at spots, orange at periphery)
  - Stability: stable (100% particles retained)
  - Novelty: repeat (same hexagonal as D1/D2=2,4)
Visual: Shuffled 2-type → concentric rings → hexagonal 7-spot pattern. Blue particles form tight clusters at spot locations, orange particles form outer ring and interstitial regions. Fields show complementary C1/C2 structure with 7 clear peaks/troughs.
Mutation: D1/D2 ratio: 2 → 1 (D2: 0.5 → 1.0)
Observation: **D1/D2=1 STILL produces hexagonal!** The hexagonal attractor is EXTREMELY robust - spans D1/D2=1-4 range without changing mode. Metrics nearly identical across all D1/D2 tests (entropy~0.85, clustering~0.31-0.32). **D1/D2 EXPLORATION EXHAUSTED** - the Brusselator with these A/B parameters is strongly committed to hexagonal mode regardless of diffusion ratio. Need different approach: A/B parameter modification or domain size constraint.
Next: parent=28 (try smaller domain n_nodes=22500 to constrain wavelengths)

## Iter 27: 8/10
Node: id=27, parent=25
Mode/Strategy: explore (testing D1/D2 ratio for stripe selection - Turing theory predicts D1/D2~1 → stripes)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0], [0.5, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types with opposing mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.850, plateau=0.00, in_box=100.0%, clustering=0.321, pattern_growth=29.55
Score: 8/10
Assessment:
  - Symmetry: hexagonal (clear 6-7 spot pattern)
  - Particles: clustered (tight spots at C1 peaks)
  - Stability: stable (100% particles retained)
  - Novelty: repeat (same hexagonal as chi experiments)
Visual: Shuffled 2-type → concentric rings → clear hexagonal 6-7 spot pattern. Final frames show 1 central spot + 6 peripheral in classic arrangement. Blue particles form tight clusters at spots, orange dispersed in intermediate regions. Fields show complementary C1 (peaks at spots) / C2 (troughs at spots) structure.
Mutation: D1/D2 ratio: 4 → 2 (D2: 0.25 → 0.5)
Observation: **D1/D2=2 does NOT produce stripes** - hexagonal pattern persists! The Brusselator in this configuration is STRONGLY attracted to hexagonal mode. Metrics nearly identical to chi experiments (entropy~0.85, clustering~0.32). This suggests the attractor basin for hexagonal mode extends across D1/D2=2-4 range. Need even more extreme ratio (D1/D2~1 or less) OR different approach entirely.
Next: parent=25 (try D1/D2=1 or explore different mechanism for stripe selection)

## Iter 26: 8/10
Node: id=26, parent=25
Mode/Strategy: exploit (continue chi exploration - testing 4x stronger cross-diffusion)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0.2], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types with opposing mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.857, plateau=0.00, in_box=100.0%, clustering=0.318, pattern_growth=31.06
Score: 8/10
Assessment:
  - Symmetry: hexagonal (clear 7-spot pattern: 1 center + 6 peripheral)
  - Particles: clustered (particles form blue regions at C1 peaks)
  - Stability: stable (100% particles retained)
  - Novelty: repeat (identical pattern to chi=0.05)
Visual: Shuffled 2-type init → concentric ring formation → clear hexagonal 7-spot pattern. Central spot surrounded by 6 peripheral spots in classic hexagonal arrangement. Both particle types co-locate at spots. Fields show complementary C1 (high at spots) / C2 (low at spots) structure.
Mutation: chi: 0.05 → 0.2 (4x stronger cross-diffusion)
Observation: **Cross-diffusion chi=0.2 produces IDENTICAL results to chi=0.05**: same hexagonal 7-spot, same entropy (~0.86), same clustering (~0.32). The cross-diffusion parameter (at least up to chi=0.2) does NOT affect Turing mode selection - the Brusselator's intrinsic activator-inhibitor dynamics completely dominate. Cross-diffusion may only become relevant at MUCH higher values (>1.0?) or may not couple to the mode selection mechanism at all. **CHI EXPLORATION EXHAUSTED** - need new direction.
Next: parent=25 (try D1/D2 ratio modification - D1/D2 closer to 1 for stripe selection per Turing theory)

## Iter 25: 8/10
Node: id=25, parent=root
Mode/Strategy: explore (Block 4 starts - testing cross-diffusion chi parameter to break hexagonal attractor)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0.05], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.6, 1.0, 1.6, 1.9]] (2 types with opposing mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.856, plateau=0.00, in_box=100.0%, clustering=0.317, pattern_growth=39.78
Score: 8/10
Assessment:
  - Symmetry: hexagonal (clear 7-spot pattern: 1 center + 6 peripheral)
  - Particles: clustered/network (particles form blue regions at C1 peaks)
  - Stability: stable (100% particles retained)
  - Novelty: variant (first test with chi≠0)
Visual: Shuffled 2-type init → concentric ring formation → clear hexagonal 7-spot pattern by final frames. Central bright spot surrounded by 6 peripheral spots. Particles (blue/orange) cluster in correspondence with C1 peaks. Fields show clear complementary C1/C2 hexagonal structure.
Mutation: chi: 0 → 0.05 (added cross-diffusion term coupling C1 diffusion to C2 gradient)
Observation: **Cross-diffusion chi=0.05 does NOT break hexagonal attractor.** The pattern is indistinguishable from chi=0 Brusselator - same 6-7 spot hexagonal symmetry emerges. Entropy=0.856 is high (higher than Brusselator without chi), clustering=0.317 is moderate. This suggests chi=0.05 is too weak to affect pattern selection, or cross-diffusion in this formulation doesn't couple to mode selection. Need to try stronger chi or explore D1/D2 ratio instead.
Next: parent=25 (try stronger chi=0.2 or modify D1/D2 ratio)

## Iter 24: 9/10
Node: id=24, parent=21
Mode/Strategy: exploit (highest UCB node=21 at 2.771, completing model comparison matrix: 1-type Brusselator)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=1, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]] (1 type with standard mobility M1=-4, M2=4)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.610, plateau=0.00, in_box=100.0%, clustering=0.447, pattern_growth=108.25
Score: 9/10
Assessment:
  - Symmetry: hexagonal (clear 6-7 spots in final frames)
  - Particles: clustered (tight clusters at field maxima)
  - Stability: stable (100% particles retained)
  - Novelty: variant (completes model comparison matrix)
Visual: Beautiful progression: random uniform → concentric rings → hexagonal 6-7 spots with tight particle clusters. Single-type creates coherent aggregation without phase separation. Fields show clear complementary hexagonal C1/C2 patterns.
Mutation: n_particle_types: 3 → 1 (completing Brusselator × n_types comparison)
Observation: **1-TYPE BRUSSELATOR ACHIEVES NEAR-IDENTICAL CLUSTERING TO 3-TYPE**: clustering=0.447 (vs 3-type's 0.530). This is DIFFERENT from FHN where 1-type had clustering=-0.206! The Brusselator field dynamics are the PRIMARY driver of clustering - particle types mainly add visual multi-layer structure but don't dramatically change clustering metric. Entropy is LOWER (0.61 vs 0.71 for 3-type) suggesting even tighter localization. Pattern_growth is higher (108 vs 38) suggesting more dynamic fields with 1-type.
Next: Block 3 complete - prepare Block 4

## BLOCK 3 SUMMARY (Iters 17-24)
**Focus**: FHN mesh model exploration + Brusselator comparison + particle-particle interactions
**Score progression**: 8→9→8→8→9→9→8→9 (Best: 9/10 at iters 18, 21, 22, 24)
**Key discoveries**:
1. **FHN produces finer-scale patterns**: ~30+ spots vs Brusselator's 6-7, with filamentary/network particle organization
2. **FHN hexagonal is particle-type-independent**: 1, 2, 3 types all achieve hexagonal (like Brusselator)
3. **Brusselator vs FHN trade-off QUANTIFIED**: Brusselator = tight clusters (clustering~0.5), low entropy (0.61-0.71); FHN = dispersed (clustering~0.27), high entropy (0.8-0.84)
4. **FHN epsilon effect**: Lower epsilon (0.04 vs 0.08) increases clustering but also pattern_growth (more dynamic)
5. **ar_p particle-particle params have MINIMAL effect**: Doubling ar_p (1.6→3.0) did not improve clustering
6. **1-type Brusselator works well**: clustering=0.447 nearly matches 3-type's 0.530
**Particle type distribution**: 1-type: 2 (iters 19, 24), 2-type: 3 (iters 17, 20, 23), 3-type: 3 (iters 18, 21, 22) - BALANCED

## Iter 23: 8/10
Node: id=23, parent=22
Mode/Strategy: exploit (highest UCB nodes=21,22 tied at 2.632, chose 22 to explore particle-particle interactions)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 3.0, 1.0, 3.0, 1.5], [4, -4, -45, 45, 3.0, 1.0, 3.0, 1.9]] (2 types with INCREASED ar_p1,ar_p3=3.0)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.838, plateau=0.00, in_box=100.0%, clustering=0.275, pattern_growth=29.91
Score: 8/10
Assessment:
  - Symmetry: hexagonal (clear 6 spots by final frame)
  - Particles: clustered (type-mixed within hexagonal spots)
  - Stability: stable (100% particles retained, low pattern_growth)
  - Novelty: variant (testing pp interaction parameters)
Visual: Shuffled 2-type init → concentric ring formation → hexagonal 6-fold symmetry. Final frames show 6 clear hexagonal spots with blue/orange particles mixed within each spot rather than segregated. Fields show complementary C1/C2 hexagonal patterns.
Mutation: ar_p1,ar_p3: 1.6 → 3.0 (increased particle-particle attraction/repulsion strength)
Observation: **Particle-particle interaction strength (ar_p) has MINIMAL effect on clustering**. Despite doubling ar_p values (1.6→3.0), clustering=0.275 is LOWER than iter 22's 0.530 (same Brusselator, 3-types, standard ar_p). The key difference is n_particle_types: 3-types creates tighter measured clustering than 2-types regardless of ar_p values. The ar_p parameters may affect local dynamics but not the macroscopic pattern metrics. Score 8/10 (still good hexagonal, but less tight than 3-type version).
Next: parent=21 or 22 (explore 1-type Brusselator or continue parameter exploration)

## Iter 22: 9/10
Node: id=22, parent=21
Mode/Strategy: exploit (highest UCB node=21, returning to Brusselator for direct comparison with FHN)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=3, shuffle_particle_types=true, boundary=periodic
params_mesh: [[1.0, 15, 4.5, 6.5, 0, 0], [0.25, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]] (3 types with opposing mobilities)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.706, plateau=0.00, in_box=100.0%, clustering=0.530, pattern_growth=38.11
Score: 9/10
Assessment:
  - Symmetry: hexagonal (clear 6-7 spots in final frames)
  - Particles: clustered (tight multi-layer hexagonal organization)
  - Stability: stable (100% particles retained, low pattern_growth)
  - Novelty: variant (direct comparison Brusselator vs FHN)
Visual: **BRUSSELATOR TIGHTER CLUSTERS THAN FHN!** Beautiful progression: shuffled random → concentric 3-type rings (blue/orange/green) → hexagonal 6-7 spots with tight particle clusters. Each spot shows multi-layer concentric particle organization. Fields show clear complementary hexagonal C1/C2 patterns.
Mutation: mesh_model_name: PDE_Diffusiophoresis_FHN → Diffusiophoresis_Mesh (returning to Brusselator for comparison)
Observation: **BRUSSELATOR vs FHN QUANTIFIED**: Both achieve hexagonal symmetry with 3-types, but:
- Clustering: Brusselator=0.53 >> FHN=0.27 (Brusselator 2x tighter)
- Entropy: Brusselator=0.71 < FHN=0.84 (Brusselator more localized)
- Pattern_growth: Brusselator=38 << FHN=463 (Brusselator 12x more converged)
Brusselator produces DENSER, MORE CONVERGED clusters while FHN produces more DISPERSED, DYNAMIC patterns. Both valid hexagonal modes but different "character". Brusselator best for tight aggregation, FHN for network/filamentary organization.
Next: parent=22 or 21 (explore particle-particle interactions or continue model comparison)

## Iter 21: 9/10
Node: id=21, parent=20
Mode/Strategy: exploit (highest UCB nodes=19/20 tied at 2.214, chose 20 to test 3-types with low epsilon)
Config: mesh_model=PDE_Diffusiophoresis_FHN, n_particle_types=3, shuffle_particle_types=true, boundary=periodic
params_mesh: [[0.5, 0.7, 0.8, 0.04, 0.0, 20.0], [0.0...], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]] (3 types with opposing mobilities)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.837, plateau=0.00, in_box=100.0%, clustering=0.265, pattern_growth=463.12
Score: 9/10
Assessment:
  - Symmetry: hexagonal (clear 6-fold symmetry in final frames)
  - Particles: clustered (multi-layer concentric rings within hexagonal spots)
  - Stability: stable (100% particles retained, clear pattern convergence)
  - Novelty: variant (combining 3-types with low epsilon from iter 20)
Visual: **MULTI-LAYER HEXAGONAL SPOTS!** Beautiful trajectory: shuffled random init → concentric 3-layer rings (blue/orange/green type separation) → hexagonal 6-fold symmetry breaking with ~6-7 spots. Each spot contains multi-layer concentric particle rings. Fields show clear hexagonal patterns with complementary C1/C2 distributions.
Mutation: n_particle_types: 2 → 3 + epsilon: 0.04 (combining 3-type richness with low epsilon clustering)
Observation: **Epsilon effect differs by particle type count**: 3-types with epsilon=0.04 (iter 21) gives clustering=0.265, while 2-types with epsilon=0.04 (iter 20) gave clustering=0.391. This is OPPOSITE to expectation - more types distributes particles into layers, reducing apparent "clustering" metric while creating richer structure. Entropy increased (0.84 vs 0.71) supporting this interpretation: more spatial complexity with 3-types. Score 9/10 for aesthetic quality of multi-layer hexagonal organization.
Next: parent=18 or 20 (explore other epsilon values or return to Brusselator comparison)

## Iter 20: 8/10
Node: id=20, parent=19
Mode/Strategy: exploit (highest UCB node=19, tuning FHN epsilon parameter)
Config: mesh_model=PDE_Diffusiophoresis_FHN, n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[0.5, 0.7, 0.8, **0.04**, 0.0, 20.0], [0.0...], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...]] (2 types with opposing mobilities)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.709, plateau=0.00, in_box=100.0%, clustering=0.391, pattern_growth=1084.17
Score: 8/10
Visual: **EPSILON TUNING IMPROVES CLUSTERING!** Lower epsilon (0.04 vs 0.08) produces: (1) strong hexagonal 6-fold symmetry in fields, (2) improved particle clustering (0.391 vs 0.259-0.315 in iter 17-19), (3) very high pattern_growth (1084) indicating dynamic patterns. Particles form tighter clusters than previous FHN runs but still looser than Brusselator's 0.6+.
Mutation: epsilon: 0.08 → 0.04 (slower recovery variable for more persistent activation)
Observation: **Epsilon is key FHN tuning parameter**: Lower epsilon increases pattern_growth dramatically (1084 vs ~130-450 before) and improves clustering (0.39 vs 0.26-0.31). However, entropy dropped (0.71 vs 0.80-0.83) suggesting less spatial complexity. Trade-off: tighter clusters but less intricate patterns. Next: try even lower epsilon or tune other params (a, b).
Next: parent=19 (continue exploit of FHN parameter space)

## Iter 19: 8/10
Node: id=19, parent=18
Mode/Strategy: multi-type balance (1-type under-represented, testing FHN with n_particle_types=1)
Config: mesh_model=PDE_Diffusiophoresis_FHN, n_particle_types=1, shuffle_particle_types=true, boundary=periodic
params_mesh: [[0.5, 0.7, 0.8, 0.08, 0.0, 20.0], [0.0...], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]] (1 type with standard mobility)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.803, plateau=0.00, in_box=100.0%, clustering=-0.206, pattern_growth=129.59
Score: 8/10
Visual: **FHN + 1-TYPE ALSO ACHIEVES HEXAGONAL!** Progression: random init → concentric rings with clear radial symmetry → **hexagonal 6-fold symmetry breaking** in final frames. Final state shows ~6-7 hexagonal spots in C1/C2 fields with filamentary particle network connecting vertices. Less tight clustering than 3-type (clustering -0.206 vs +0.259).
Mutation: n_particle_types: 3 → 1 (testing if FHN hexagonal requires multiple types)
Observation: **KEY REVISION**: FHN hexagonal mode is NOT particle-type-dependent! 1-type (iter 19) achieves same hexagonal structure as 3-type (iter 18). Pattern_growth (129.59) even lower than 3-type (183.38), suggesting 1-type may be most stable. The difference between iters 17 (2-type, finer network) and 18/19 (3-type/1-type, hexagonal) may be due to other config differences or stochastic variation rather than particle type count.
Next: parent=18 (exploit highest UCB, explore FHN parameter space)

## Iter 18: 9/10
Node: id=18, parent=17
Mode/Strategy: exploit (highest UCB node=17, continue FHN exploration with 3-types)
Config: mesh_model=PDE_Diffusiophoresis_FHN, n_particle_types=3, shuffle_particle_types=true, boundary=periodic
params_mesh: [[0.5, 0.7, 0.8, 0.08, 0.0, 20.0], [0.0...], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]] (3 types with opposing mobilities)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.816, plateau=0.00, in_box=100.0%, clustering=0.259, pattern_growth=183.38
Score: 9/10
Visual: **FHN + 3-TYPE ACHIEVES HEXAGONAL MODE!** Remarkable trajectory: shuffled random init → concentric multi-layer rings (3 particle types clearly separated) → HEXAGONAL 6-fold symmetry breaking in final frames. Final frames show ~6-7 distinct hexagonal spots in fields with corresponding particle clustering. Clear multi-layer organization within each spot.
Mutation: n_particle_types: 2 → 3 (testing particle count effect on FHN)
Observation: **KEY DISCOVERY**: FHN hexagonal mode is ALSO particle-type-dependent! 2-types (iter 17) → finer network (~30 spots, 8/10), 3-types (iter 18) → hexagonal mode (~6-7 spots, 9/10). This mirrors Brusselator behavior but via different PDE mechanism. Pattern_growth (183) lower than iter 17 (451) suggests more stable configuration.
Next: parent=18

## Iter 17: 8/10
Node: id=17, parent=root
Mode/Strategy: explore (new PDE variant - FitzHugh-Nagumo)
Config: mesh_model=PDE_Diffusiophoresis_FHN, n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[0.5, 0.7, 0.8, 0.08, 0.0, 20.0], [0.0...], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...]] (2 types with opposing mobilities)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.826, plateau=0.00, in_box=100.0%, clustering=0.315, pattern_growth=451.99
Score: 8/10
Visual: FHN achieves hexagonal with FINER spatial scale (~30+ spots vs Brusselator's ~6). Particles form network/filamentary structure, not tight clusters.
Mutation: mesh_model_name: Diffusiophoresis_Mesh → PDE_Diffusiophoresis_FHN
Observation: FHN qualitatively different - faster dynamics (pattern_growth 25x higher), finer wavelength, network particle organization
Next: parent=17

## Iter 16: 9/10 >>> BLOCK 2 FINAL <<<
Node: id=16, parent=12
Mode/Strategy: robustness-test (verify reproducibility of best result)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=3, shuffle_particle_types=true
params_mesh: [[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]] (3 types with opposing mobilities)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.654, plateau=0.00, in_box=100.0%, clustering=0.621, pattern_growth=17.74
Score: 9/10
Visual: **ROBUSTNESS CONFIRMED!** Successfully replicates iter 12's hexagonal mode. From shuffled random init → concentric tri-layer rings (blue center, orange middle, green outer) → beautiful HEXAGONAL 6-fold symmetry breaking in final frames. Fields show 6-spot arrangement, particles show 6-armed star with clear type segregation. Clustering (0.621) matches iter 12 (0.615) precisely!
Mutation: [robustness test] - same config as iter 12
Observation: **KEY VALIDATION**: The 9/10 hexagonal mode result is REPRODUCIBLE, not a lucky random seed! Clustering values nearly identical (0.621 vs 0.615), pattern trajectory identical. This confirms the full random→rings→hexagonal trajectory is a robust attractor of the Brusselator + diffusiophoresis system.

=== BLOCK 2 SUMMARY ===
Iterations: 9-16 (8 total)
Focus: Gray-Scott PDE variant exploration + particle type independence testing
Score progression: 7→7→8→9→8→7→8→9
Best score: 9/10 (iters 12, 16)

Key Findings:
1. **Gray-Scott produces different patterns**: Concentric rings only, no hexagonal breaking even with 6000 frames (iter 14)
2. **Hexagonal mode is particle-type-independent**: 1, 2, AND 3-type systems all achieve ring→hexagonal with Brusselator (iters 13, 15, 16)
3. **Concentric pattern is true attractor**: shuffle_particle_types=true → same final pattern as ordered init (iter 11)
4. **6000 frames required for mode selection**: 4000 frames insufficient for hexagonal breaking
5. **Reproducibility confirmed**: iter 16 robustness test replicates iter 12's metrics precisely

Questions answered this block:
- Does hexagonal mode require 3 particle types? NO
- Is concentric pattern an initial condition artifact? NO (true attractor)
- Does Gray-Scott break radial symmetry? NO (fundamental difference from Brusselator)

Open questions for Block 3:
- Can particle-particle interactions break radial symmetry faster?
- What particle dynamics modifications would create non-radial patterns with Gray-Scott?
- Can asymmetric mobility magnitudes create directed transport?

Next: parent=12

## Iter 15: 8/10
Node: id=15, parent=12
Mode/Strategy: multi-type balance (1-type under-represented: 2/14 = 14%)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=1
params_mesh: [[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]]
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.616, plateau=0.00, in_box=100.0%, clustering=0.498, pattern_growth=72.31
Score: 8/10
Visual: **HEXAGONAL MODE WITH 1 TYPE!** Clear trajectory: random noise → concentric ring pattern → beautiful hexagonal 6-fold Turing spots in fields AND particles. Final frames show ~20+ hexagonal spots in C1/C2 fields with particle cluster at each spot. Highest pattern_growth this block (72.31).
Mutation: n_particle_types: 3→1 (single type test), retained 6000 frames
Observation: **KEY FINDING**: Hexagonal mode selection does NOT depend on particle type count at all! 1-type system achieves same ring→hexagonal trajectory as 2-type and 3-type. This confirms that the mode selection is driven entirely by Brusselator field dynamics, with particles passively following field gradients. Clustering (0.50) intermediate between 3-type (0.62) and 2-type (0.42).
Next: parent=12 or 13, final block iteration - consider robustness test or boundary exploration

## Iter 1: 2/10
Node: id=1, parent=root
Mode/Strategy: explore (initial baseline)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 1
params: [[-16, 16, 180, -180, 1.6, 1.0, 1.6, 1.5], [-8, 8, 90, -90, 1.8, 1.8, 1.1, 1.9], [-4, 4, 45, -45, 1.7, 1.8, 1.1, 1.9]]
n_frames: 2000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.00, plateau=0.00, in_box=0.0%
Score: 2/10
Visual: Fields show excellent Turing spot patterns (C1/C2 with clear activator-inhibitor spots). Particles initially cluster at field minima but rapidly escape simulation box by frame ~15. Later frames show empty domain.
Mutation: Initial baseline configuration
Observation: Mobility parameters M1=-16, M2=16 are too strong - particles escape domain. Field dynamics (Da_c=15, A=4.5, B=6.5) produce good Turing patterns. Need to reduce mobility while preserving field patterns.
Next: parent=1, reduce mobility parameters (M1, M2) by 4x to prevent particle escape

## Iter 2: 7/10
Node: id=2, parent=1
Mode/Strategy: exploit (mobility reduction to fix particle escape)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 1
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [-2, 2, 22, -22, 1.8, 1.8, 1.1, 1.9], [-1, 1, 11, -11, 1.7, 1.8, 1.1, 1.9]]
n_frames: 2000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.594, plateau=0.00, in_box=100.0%
Score: 7/10
Visual: Dramatic improvement! Fields develop beautiful ring/target patterns that evolve into Turing spots. Particles stay in box (100%) and form concentric ring structures that track field gradients, eventually condensing into spot-like clusters matching the field pattern. Clear ring→spot transition visible over simulation.
Mutation: M1: -16→-4, M2: 16→4, consumption/production: 180→45
Observation: The 4x mobility reduction was critical - particles now stay in box while still responding to field gradients. Pattern evolution shows ring formation → ring fragmentation → spot array. High clustering (0.58) and good spatial entropy (0.59) indicate organized structure. Plateau=0 suggests still evolving; may benefit from longer simulation.
Next: parent=2, explore extending n_frames OR try n_particle_types=2 for richer dynamics

## Iter 3: 8/10
Node: id=3, parent=2
Mode/Strategy: multi-type (exploring n_particle_types=2 for phase separation)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 2
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.8, 1.0, 1.6, 1.5]]
n_frames: 2000, delta_t: 5E-4, n_particles: 9600 (4800 each type)
Metrics: entropy=0.809, plateau=0.00, in_box=100.0%
Score: 8/10
Visual: Excellent phase separation! Two particle types self-organize into distinct spatial domains: Type 0 (M1=-4, orange) forms outer ring attracted to C1 gradients, Type 1 (M1=+4, blue) concentrates in center repelled from C1 peaks. Field develops complex rotating 3-fold Turing patterns. Beautiful co-evolution of field spirals and particle segregation.
Mutation: n_particle_types: 1→2, Type 1 params flipped to opposing mobilities (M1: -4→+4, M2: 4→-4)
Observation: Opposing mobilities create spontaneous phase separation - biologically analogous to cell sorting or tissue boundary formation. Spatial entropy higher (0.809) because particles spread within their respective domains rather than tight clustering. Pattern growth strong (10.81). Ring-core structure stable throughout simulation.
Next: parent=3, exploit success - try n_particle_types=3 for even richer dynamics, or tune mobility magnitude

## Iter 4: 9/10
Node: id=4, parent=3
Mode/Strategy: multi-type (n_particle_types=3 for tri-layer organization)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 3
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]]
n_frames: 2000, delta_t: 5E-4, n_particles: 9600 (3200 each type)
Metrics: entropy=0.684, plateau=0.00, in_box=100.0%
Score: 9/10
Visual: Stunning tri-layer concentric ring organization! Type 2 (blue, M1=0, M2=8) forms tight central cluster (follows C2 gradient only). Type 0 (orange, M1=-4, M2=4) forms middle ring. Type 1 (green, M1=4, M2=-4) forms outermost ring. Field shows stable circular disc pattern with C1 ring/C2 center structure. Pattern remarkably stable throughout simulation with some oscillation in final frames.
Mutation: n_particle_types: 2→3, Type 2 added with unique mobility signature (M1=0, M2=8)
Observation: Three-type system achieves clear radial stratification - morphogenetically analogous to germ layer formation (ectoderm/mesoderm/endoderm). Spatial entropy (0.684) slightly reduced from 2-type (0.809) due to tighter organization within each layer. Clustering very high (0.608). The Type 2 configuration (M1=0, responds only to C2) creates distinctive behavior - pure C2 following without C1 influence.
Next: parent=4, consider exploit with mobility fine-tuning OR explore alternative dynamics (boundary=periodic, or increased n_frames)

## Iter 5: 9/10
Node: id=5, parent=4
Mode/Strategy: robustness-test (verify reproducibility of high-scoring config)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 3
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]]
n_frames: 4000, delta_t: 5E-4, n_particles: 9600 (3200 each type)
Metrics: entropy=0.679, plateau=0.00, in_box=100.0%
Score: 9/10
Visual: Reproducible tri-layer concentric ring organization. Blue (Type 2) tight central cluster, orange (Type 0) middle ring, green (Type 1) outer ring. Field patterns show stable disc with some late-stage complexity - note ring instabilities appearing in final frames that fragment the particle layers slightly. Overall structure remains intact throughout extended 4000-frame simulation.
Mutation: n_frames: 2000→4000 (extended simulation to test stability)
Observation: Pattern confirmed reproducible with 2x longer simulation. Metrics nearly identical to Iter 4 (entropy: 0.684→0.679, clustering: 0.608→0.631). Plateau=0 persists even at 4000 frames, indicating ongoing dynamics rather than convergence - this is likely oscillatory Brusselator behavior. Late frames show some ring fragmentation suggesting field instability may eventually disrupt particle organization.
Next: parent=5, extend n_frames further (6000) to test if oscillatory dynamics are sustainable or if pattern degrades

## Iter 6: 9/10
Node: id=6, parent=5
Mode/Strategy: exploit (extended simulation for long-term dynamics)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 3
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]]
n_frames: 6000, delta_t: 5E-4, n_particles: 9600 (3200 each type)
Metrics: entropy=0.644, plateau=0.00, in_box=100.0%
Score: 9/10
Visual: Remarkable symmetry transition discovered! Early frames show concentric tri-layer rings (as before), but extended simulation reveals transition to **hexagonal 6-fold symmetry** in both fields and particles. Field develops 6-spot Turing pattern (C1 hot spots in hexagonal arrangement). Particles reorganize from rings → 6-armed star pattern with central cluster maintained. Final frames show clear hexagonal mode selection by Brusselator.
Mutation: n_frames: 4000→6000 (further extended to observe long-term evolution)
Observation: Extended simulation reveals pattern maturation - radial symmetry breaking into hexagonal Turing mode. This is classic reaction-diffusion behavior - finite wavelength instability selects 6-fold mode. Clustering highest yet (0.621), spatial entropy moderate (0.644). Pattern growth strong (18.14). Plateau=0 confirms ongoing dynamics. The ring→hexagon transition is biologically significant (honeycomb packing, hexagonal cell arrays).
Next: parent=6 (highest UCB), explore boundary effects OR vary Brusselator params to test different mode selections

## Iter 7: 9/10
Node: id=7, parent=6
Mode/Strategy: explore (boundary='periodic' with 2 particle types)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 2
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...]]
n_frames: 4000, delta_t: 5E-4, n_particles: 9600 (4800 each type)
boundary: periodic
Metrics: entropy=0.766, plateau=0.00, in_box=100.0%, clustering=0.45
Score: 9/10
Visual: Beautiful evolution with periodic boundary! Field C1/C2 show rotating 3-lobe pattern → hexagonal 7-spot arrangement (1 central + 6 peripheral). Particles: Type 0 (orange) outer ring/matrix, Type 1 (blue) central clusters. Late frames show exquisite particle-field coupling - blue particles form flower-like 6-7 cluster pattern tracking C1 spots, orange particles fill interstitial space. Clear 6-fold symmetry emergence.
Mutation: boundary: none→'periodic', n_particle_types: 3→2 (simplify to test boundary effect)
Observation: Periodic boundary confirms 100% particle retention as expected. Key finding: 2-type system with periodic boundary produces similar hexagonal Turing mode selection as 3-type system. Spatial entropy higher (0.766 vs 0.644) because particles spread more with periodic BC. The 7-spot final pattern is classic Turing wavelength selection fitting periodic domain. Reduced complexity (2 types vs 3) yields comparable quality, suggesting periodicity + opposing mobilities are key drivers.
Next: parent=7 (exploit success), final iteration of block - consider varying Da_c or testing n_particle_types=1 with periodic boundary to balance type distribution

## Iter 8: 8/10
Node: id=8, parent=7
Mode/Strategy: explore (balance particle type distribution - testing n_particle_types=1 with periodic BC)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 1
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]]
n_frames: 4000, delta_t: 5E-4, n_particles: 9600
boundary: periodic
Metrics: entropy=0.574, plateau=0.00, in_box=100.0%, clustering=0.55
Score: 8/10
Visual: Good pattern development with single particle type and periodic BC. Fields evolve from uniform → concentric rings → hexagonal spot array (classic Brusselator mode selection). Single-type particles (blue disc) develop internal structure tracking field gradients - final frames show clustered ring/hexagonal organization matching field spots. Less dramatic than multi-type but clean Turing-particle coupling demonstrated.
Mutation: n_particle_types: 2→1 (balance type distribution, test single-type + periodic)
Observation: Single-type configuration with periodic boundary confirms good particle-field coupling but loses the phase separation richness of multi-type systems. Entropy lower (0.574 vs 0.766) due to less spatial diversity. Clustering moderate (0.55). Pattern quality good but not exceptional - multi-type systems (2 or 3 types) with opposing mobilities remain superior for biologically interesting patterns.
Next: Block complete - proceed to Block 2

---

## Block 1 Summary

**Key Achievements:**
1. **Mobility calibration**: Found M1=±4, M2=±4 sweet spot (4x reduction from initial params) that prevents particle escape while preserving strong field-particle coupling
2. **Multi-type phase separation**: Opposing mobilities (Type 0: M1=-4, M2=4 vs Type 1: M1=4, M2=-4) create spontaneous spatial segregation
3. **Tri-layer stratification (9/10)**: 3-type system with distinct mobility signatures creates germ-layer-like radial organization
4. **Hexagonal mode selection**: Extended simulations reveal ring→hexagonal symmetry breaking (classic Turing wavelength selection)
5. **Periodic boundary**: Enables comparable pattern quality with simpler configurations

**Best Configurations:**
- 3-type tri-layer: Score 9/10, entropy 0.64-0.68, clustering 0.6+ (Iters 4-6)
- 2-type + periodic: Score 9/10, entropy 0.77, clustering 0.45 (Iter 7)

**Score Progression:** 2 → 7 → 8 → 9 → 9 → 9 → 9 → 8 (stable plateau at 9/10)

**Particle Type Distribution:** 1-type: 3, 2-type: 2, 3-type: 3 (well balanced)

**Open Questions for Block 2:**
- Can Gray-Scott or FitzHugh-Nagumo PDE variants produce different pattern modes?
- Does higher Da_c (faster Brusselator) affect mode selection?
- Can asymmetric mobility magnitudes create directed transport?

---

## Block 2: Gray-Scott Exploration

## Iter 9: 7/10
Node: id=9, parent=8
Mode/Strategy: code-change (Gray-Scott PDE variant, first successful run)
Config: params_mesh=[[0.2, 0.035, 0.06, 50.0, 0, 0], [0.1, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 2
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.8, 1.0, 1.6, 1.5]]
n_frames: 4000, delta_t: 5E-4, n_particles: 9600 (4800 each type)
mesh_model_name: PDE_Diffusiophoresis_GrayScott
Metrics: entropy=0.868, plateau=0.00, in_box=100.0%
clustering=0.346, pattern_growth=26.79, C1_std=0.530, C2_std=0.134
Score: 7/10
Visual: Gray-Scott variant produces concentric target ring patterns in both C1 and C2 fields - distinct from Brusselator hexagonal mode selection. Rings evolve from uniform state, progressively developing sharper concentric structure. Particles maintain clean 2-type phase separation: blue (Type 1) concentrated at center, orange (Type 0) forms outer shell with ring banding. Late frames show blue particles developing internal spiral/swirl structure while orange particles show ring-like radial banding. Field patterns remain radially symmetric without breaking to hexagonal - different qualitative behavior from Brusselator.
Mutation: mesh_model_name: Diffusiophoresis_Mesh → PDE_Diffusiophoresis_GrayScott, params_mesh: Brusselator→Gray-Scott (Du=0.2, F=0.035, k=0.06, scale=50, Dv=0.1)
Observation: Gray-Scott variant successfully runs and produces qualitatively different patterns from Brusselator. Key differences: (1) radial ring/target patterns instead of hexagonal spots, (2) much stronger pattern growth (26.8x vs ~18x), (3) higher spatial entropy (0.87 vs ~0.65-0.77) suggesting more diffuse particle distribution. Particle phase separation maintained. Plateau=0 continues. The F=0.035, k=0.06 regime (gamma region of Gray-Scott phase space) produces concentric rings rather than the expected worms/stripes - possibly because radial initialization biases toward ring modes. Need to test different F,k regimes to access self-replicating spots or maze patterns.
Next: Explore different Gray-Scott F,k regime (alpha/beta region for spots) or increase simulation length

## Iter 10: 7/10
Node: id=10, parent=9
Mode/Strategy: exploit (F,k regime shift + multi-type balance)
Config: params_mesh=[[0.2, 0.04, 0.065, 50.0, 0, 0], [0.1, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 3
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]]
n_frames: 4000, delta_t: 5E-4, n_particles: 9600 (3200 each type)
mesh_model_name: PDE_Diffusiophoresis_GrayScott
Metrics: entropy=0.845, plateau=0.00, in_box=100.0%
clustering=0.455, pattern_growth=22.09, C1_std=0.515, C2_std=0.110
Score: 7/10
Visual: Beautiful tri-layer concentric ring structure with Gray-Scott fields. Field C1/C2 show multiple concentric rings with complex internal structure (3-5 ring bands visible). Particles: Type 0 (blue) forms compact central disk, Type 1 (orange) forms intermediate ring, Type 2 (green) forms outermost shell. Tri-layered "germ layer" organization achieved with Gray-Scott variant. Pattern remains stable throughout simulation without hexagonal symmetry breaking.
Mutation: F: 0.035→0.04, k: 0.06→0.065 (alpha/spot regime shift), n_particle_types: 2→3
Observation: F,k regime shift did NOT produce self-replicating spots as hoped - patterns remain radially symmetric (target rings). However, tri-type particle organization successfully reproduces germ-layer stratification seen with Brusselator. Clustering improved (0.455 vs 0.346) suggesting tighter particle organization with 3 types. Entropy slightly lower (0.845 vs 0.868). Gray-Scott appears fundamentally different from Brusselator - consistently produces radial modes without hexagonal breaking. The radial initialization may bias system toward ring modes regardless of F,k regime.
Next: parent=10, try lowering F or using extreme k to access different Gray-Scott regimes (mu/theta for maze patterns, or soliton spots)

## Iter 11: 8/10
Node: id=11, parent=10
Mode/Strategy: explore (Brusselator comparison with shuffle_particle_types=true)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 3
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]]
n_frames: 4000, delta_t: 5E-4, n_particles: 9600 (3200 each type)
mesh_model_name: Diffusiophoresis_Mesh (Brusselator)
shuffle_particle_types: true
Metrics: entropy=0.682, plateau=0.00, in_box=100.0%
clustering=0.631, pattern_growth=12.12, C1_std=0.173, C2_std=0.061
Score: 8/10
Visual: Key test - shuffled initial conditions (random particle type mixing) with Brusselator. Despite starting from randomly mixed particle types, system self-organizes into SAME concentric tri-layer structure: blue core, orange middle ring, green outer shell. Initial frames show chaotic particle mixing, then rapid self-organization by frame 3. Field develops concentric ring Turing pattern. Final frames show ring breathing/oscillation - dynamic equilibrium with expanding and contracting ring structure.
Mutation: mesh_model_name: PDE_Diffusiophoresis_GrayScott → Diffusiophoresis_Mesh, shuffle_particle_types: false → true
Observation: **Critical finding** - The tri-layer concentric pattern is a TRUE ATTRACTOR, not an artifact of ordered initial conditions. Despite random type mixing at start, particles self-organize into identical concentric pattern seen with radial initialization. This confirms diffusiophoretic sorting is robust. Clustering (0.631) is HIGHEST seen yet - stronger type segregation from shuffled start than ordered start. Brusselator entropy (0.682) lower than Gray-Scott (0.845) indicating more structured patterns. The breathing/oscillation in late frames shows dynamic equilibrium. Still no hexagonal symmetry breaking at 4000 frames.
Next: parent=11, extend simulation to 6000 frames to test hexagonal mode selection, or try asymmetric mobilities

## Iter 12: 9/10
Node: id=12, parent=11
Mode/Strategy: exploit (extended simulation for hexagonal mode selection)
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 3
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]]
n_frames: 6000, delta_t: 5E-4, n_particles: 9600 (3200 each type)
mesh_model_name: Diffusiophoresis_Mesh (Brusselator)
shuffle_particle_types: true
Metrics: entropy=0.644, plateau=0.00, in_box=100.0%
clustering=0.615, pattern_growth=18.46, C1_std=0.225, C2_std=0.092
Score: 9/10
Visual: **Hexagonal symmetry breaking achieved from shuffled init!** Early frames show shuffled particles (random type mixing) rapidly self-organizing into concentric rings. Mid-simulation: stable tri-layer rings (blue core, orange ring, green outer). Late frames: CLEAR hexagonal 6-fold symmetry emerges - both fields and particles break from radial to hexagonal mode. Final frames show 6-lobe flower pattern in fields and 6-armed star pattern in particles. Most dramatic symmetry-breaking transition observed in simulation.
Mutation: n_frames: 4000→6000 (extended for mode selection)
Observation: **Major milestone** - Extended simulation (6000 frames) with shuffled init demonstrates full pattern trajectory: random → concentric rings → hexagonal 6-fold mode. This proves: (1) concentric pattern is true attractor, (2) hexagonal mode selection occurs given sufficient time, (3) shuffled init does NOT prevent symmetry breaking. Metrics: entropy 0.644 (structured), clustering 0.615 (high), 100% retention. The ring→hexagonal transition occurs ~frame 70-100 (late simulation). Pattern growth 18.46 confirms ongoing development. Brusselator mode selection is robust to initial conditions.
Next: parent=12, explore asymmetric mobilities or try 2-type with shuffle to test type number dependence of hexagonal mode

## Iter 13: 8/10
Node: id=13, parent=12
Mode/Strategy: explore (test if hexagonal mode requires 3 types or occurs with 2 types)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), n_particle_types=2, shuffle_particle_types=true
params_mesh: [[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], [4, -4, -45, 45, 1.8, 1.0, 1.6, 1.5]]
n_frames: 6000 (extended)
Metrics: entropy=0.797, plateau=0.00, in_box=100.0%, clustering=0.422, pattern_growth=21.20
Score: 8/10
Visual: **HEXAGONAL MODE WITH 2 TYPES CONFIRMED!** From shuffled random init: particles → concentric 2-layer rings (blue center, orange outer) → hexagonal 6-fold pattern. Fields develop 6-spot hexagonal arrangement. Particles show clear blue core splitting into ~6 clusters with orange ring having 6-fold modulation.
Mutation: n_particle_types: 3→2 (testing if hexagonal mode depends on type count)
Observation: **ANSWERS OPEN QUESTION** - Hexagonal symmetry breaking does NOT require 3 particle types. 2-type system achieves same qualitative trajectory: random → rings → hexagonal. However, 3-type has tighter clustering (0.62 vs 0.42) and lower entropy (0.64 vs 0.80), suggesting 3-type produces more organized patterns. 2-type hexagonal is "looser" but still achieves mode selection.
Next: parent=12 (continue from 3-type which gave best clustering)

## Iter 14: 7/10
Node: id=14, parent=12
Mode/Strategy: explore (Gray-Scott extended simulation comparison)
Config: mesh_model=PDE_Diffusiophoresis_GrayScott, n_particle_types=3, shuffle_particle_types=true
params_mesh: [[0.2, 0.04, 0.065, 50.0, 0, 0], [0.1, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...], [0, 8, 0, -90, ...]]
n_frames: 6000 (extended), delta_t: 5E-4, n_particles: 9600 (3200 each type)
Metrics: entropy=0.861, plateau=0.00, in_box=100.0%, clustering=0.413, pattern_growth=24.20
Score: 7/10
Visual: Gray-Scott with extended 6000-frame simulation. From shuffled init → rapid organization into concentric tri-layer rings (blue center, orange middle, green outer). Field develops 5-6 concentric target rings. **CRITICAL: Pattern remains PURELY RADIAL throughout entire simulation - NO hexagonal symmetry breaking despite 6000 frames.** Confirms qualitative difference between Gray-Scott and Brusselator dynamics.
Mutation: [comparison run] - same extended frames/shuffled init as iter 12 but with Gray-Scott instead of Brusselator
Observation: **ANSWERS OPEN QUESTION** - Gray-Scott does NOT break radial symmetry even with extended simulation and shuffled init. This confirms Brusselator vs Gray-Scott produce fundamentally different pattern dynamics: Brusselator achieves hexagonal mode selection, Gray-Scott remains radially symmetric. The difference lies in the reaction kinetics, not the initial conditions or simulation length.
Next: parent=12, return to Brusselator and explore parameter variations or 1-type configuration

---

## Block 3: FitzHugh-Nagumo and Particle-Particle Exploration

## Iter 17: 8/10
Node: id=17, parent=16
Mode/Strategy: code-change (FitzHugh-Nagumo PDE variant, first test)
Config: mesh_model=PDE_Diffusiophoresis_FHN, n_particle_types=2, shuffle_particle_types=true, boundary=periodic
params_mesh: [[0.5, 0.7, 0.8, 0.08, 0.0, 20.0], [0.0, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
params: [[-4, 4, 45, -45, ...], [4, -4, -45, 45, ...]]
n_frames: 6000, delta_t: 5E-4, n_particles: 9600 (4800 each type)
Metrics: entropy=0.826, plateau=0.00, in_box=100.0%, clustering=0.315, pattern_growth=451.99
Score: 8/10
Visual: **FHN ACHIEVES HEXAGONAL MODE WITH FINER SPATIAL SCALE!** Dramatic evolution: random noise → concentric rings (frames 2-3) → rapid breakup into multi-spot array (frames 4-5) → rich field with 30+ hexagonal spots. Fields show high-contrast spot array with much finer wavelength than Brusselator (~6 spots). Particles: initial phase separation into blue core/orange ring, then complex breakup into networked/filamentary structure. Final frames show blue particle clusters connected by delicate strands, orange particles in interstitial matrix. Qualitatively different from both Brusselator and Gray-Scott.
Mutation: mesh_model_name: Diffusiophoresis_Mesh → PDE_Diffusiophoresis_FHN (FitzHugh-Nagumo excitable regime)
Observation: **KEY FINDING - FHN produces qualitatively different patterns!** (1) Hexagonal symmetry breaking achieved (like Brusselator, unlike Gray-Scott); (2) FINER spatial scale - ~30+ spots vs Brusselator's ~6; (3) Pattern_growth 451.99 is 25x higher than Brusselator (~18) indicating much stronger field dynamics; (4) Lower clustering (0.315 vs 0.6+) because particles spread into network structure rather than tight clusters; (5) High entropy (0.826) reflects diffuse network organization. FHN excitable dynamics create fast, fine-scale patterns. The a=0.7, b=0.8, epsilon=0.08 parameters put system in excitable regime where perturbations trigger propagating waves that organize into spot array.
Next: parent=17, exploit FHN success - try n_particle_types=3 or tune FHN parameters for different wavelengths

## Iter 44: 7/10 - Schnakenberg gamma=500 + 2-type shuffle → Same radial pattern
Node: id=44, parent=43
Mode/Strategy: exploit + multi-type (test if opposing mobilities break radial symmetry)
Config: Schnakenberg Du=0.05, Dv=1.0, gamma=500, a=0.1, b=0.9, n_particle_types=2, shuffle=true, n_frames=6000, delta_t=0.0005
n_particle_types: 2
Metrics: entropy=0.8684, plateau=0.0000, in_box=100.0%, clustering=0.4278
C1_std=0.2812, C2_std=0.3561, pattern_growth=71.23
Assessment:
  - Symmetry: radial
  - Particles: segregated (blue center, orange halo)
  - Stability: stable (100% retention, but plateau=0 → oscillatory)
  - Novelty: repeat (same radial mode as iter 43)
Visual: Concentric ring pattern with large central void in C1. Fields show spiral/labyrinth sub-structure in outer rings but dominant radial organization. Despite shuffle=true, particles self-organized into clean phase separation: type 0 (blue) compact central disk, type 1 (orange) outer halo. Identical organization to non-shuffled case.
Mutation: n_particle_types: 1->2, shuffle_particle_types: false->true, added opposing mobilities
Observation: Shuffle=true has NO effect on Schnakenberg radial outcome — particles always self-organize into concentric type bands. Opposing mobilities produce clear phase separation but cannot break radial symmetry. This mirrors the Brusselator finding (principle #5) but for a radial (not hexagonal) attractor. Schnakenberg gamma=500 is fundamentally radial like Gray-Scott. Score 7/10 — functional but repetitive; radial symmetry is the Schnakenberg attractor.
Next: parent=43 (try Schnakenberg Turing boundary, OR switch to FHN boundary exploration)

## Iter 45: 9/10 - FHN Dv=0.1 → NOVEL SQUARE/GRID symmetry breaking!
Node: id=45, parent=43
Mode/Strategy: explore (switch from Schnakenberg to FHN boundary exploration with nonzero Dv)
Config: mesh_model=PDE_Diffusiophoresis_FHN, Du=0.5, Dv=0.1, a=0.7, b=0.8, epsilon=0.08, I=0, time_scale=20
n_particle_types: 1, shuffle=false (N/A for 1-type)
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]]
params_mesh: [[0.5, 0.7, 0.8, 0.08, 0.0, 20.0], [0.1, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.8095, plateau=0.0000, in_box=100.0%, clustering=-0.0806
C1_std=1.0579, C2_std=0.6102, pattern_growth=122.05
Assessment:
  - Symmetry: **other (SQUARE/GRID)** — four-fold symmetry, unprecedented!
  - Particles: **network** — thin filamentary walls forming square grid structure
  - Stability: transient (plateau=0, still evolving at frame 120)
  - Novelty: **NOVEL** — first square symmetry in 45 iterations!
Visual: **BREAKTHROUGH — SQUARE SYMMETRY BREAKING!** Evolution: (1) Random noise → concentric rings (frames 1-3). (2) Rings develop rectangular deformation (frames 4-6) — the circular symmetry breaks toward four-fold. (3) C1 field develops SQUARE central feature with rounded corners + peripheral spots at corners (frames 7-8). (4) Final state: particles form remarkable square/rectangular NETWORK — thin filamentary walls creating grid-like structure with central square void and four corner concentrations. C1 shows strong square pattern (std=1.06, highest ever for FHN). The four-fold symmetry likely arises from interaction between FHN wave dynamics and the periodic square boundary condition — the Dv=0.1 inhibitor diffusion creates long-range inhibition that couples to the boundary geometry.
Mutation: mesh_model_name: PDE_Diffusiophoresis_Schnakenberg → PDE_Diffusiophoresis_FHN; Dv: 0.0 → 0.1 (key change enabling long-range inhibition)
Observation: **KEY FINDING — Nonzero Dv in FHN enables a completely new symmetry class!** (1) Previous FHN (Dv=0) produced hexagonal spots (iter 17-24); (2) With Dv=0.1, inhibitor diffuses long-range, coupling to the square periodic boundary → four-fold symmetry; (3) clustering=-0.08 (most negative ever) = extremely thin particle walls/filaments, not clusters; (4) C1_std=1.06 is the highest FHN field variation, indicating very strong pattern amplitude; (5) Pattern_growth=122x indicates continued dynamics — system may not have fully equilibrated; (6) This is the FIRST non-radial, non-hexagonal, non-stripe symmetry discovered. Opens entirely new pattern regime. (7) The square symmetry may be a boundary effect (periodic box is square) but this is biologically relevant — real tissues have boundary constraints.
Next: parent=45, exploit this novel square symmetry — test with 2-type particles (even iteration → shuffle=true)

## Iter 49: 8/10 - High-B Brusselator gives SPOTS not stripes despite Turing boundary
Node: id=49, parent=root
Mode/Strategy: explore — Brusselator high-B stripe regime (B=13, A=3.464)
Config: mesh_model=Diffusiophoresis_Mesh (Brusselator), D1=0.05, Da_c=15, A=3.464, B=13, mu=0.04, chi=0
n_particle_types: 1, shuffle=false (odd iteration 1/8)
params: [[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5]]
params_mesh: [[0.05, 15.0, 3.464, 13.0, 0.04, 0.0], [0.5, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=0.5646, plateau=0.1389, in_box=100.0%, clustering=0.4207
C1_std=2.3282, C2_std=4.4014, pattern_growth=880.29
Assessment:
  - Symmetry: hexagonal
  - Particles: clustered
  - Stability: transient (plateau=0.14, still evolving)
  - Novelty: variant (hexagonal at Turing boundary, not expected stripes)
Visual: Evolution: (1) Initial radial ring (frame 10) with large central concentration depression in C1 / peak in C2. (2) Ring breaks into multiple spots by frame 50 (~20-30 clusters). (3) By frame 90, spot pattern persists with similar count and spatial distribution — hexagonal-like arrangement of clustered spots. C1 shows dark spots (low concentration) with higher surrounding field. C2 shows bright spots. Pattern is spot-dominated, NOT labyrinthine/stripe as predicted.
Mutation: A: 4.5 → 3.464; B: 6.5 → 13.0 (B/[1+A^2] = 13/13 = 1.0, exact Turing boundary)
Observation: **IMPORTANT — B/[1+A^2]=1.0 alone does NOT guarantee stripe mode!** At B=10, A=3 (iter 31), stripes formed. At B=13, A=3.464, SPOTS formed despite identical B/[1+A^2] ratio. Key differences: (1) Higher absolute B (13 vs 10) → stronger nonlinear reaction → spots nucleate before stripes can form; (2) Higher A (3.464 vs 3.0) → different homogeneous steady state (A, B/A) shifts pattern selection; (3) Pattern_growth=880x is extremely high vs typical ~100x, indicating vigorous ongoing dynamics; (4) plateau=0.14 is lowest seen for a patterned simulation — system hasn't equilibrated. (5) This refines our stripe criterion: it's not just B/[1+A^2]=1.0, but specifically requires **low A, high B** to get stripes. The A value matters independently of the ratio.
Next: parent=49 (test 2-type with same config, even iteration → shuffle=true)

## Iter 55: 7/10 - Brusselator stripe regime + 3-type ALL IDENTICAL → labyrinthine/disordered (not clean stripes)
Node: id=55, parent=31
Mode/Strategy: explore — test stripe mode (A=3, B=10) with 3-type ALL IDENTICAL params (uniform feedback)
Config: params_mesh=[1.0, 15.0, 3.0, 10.0, 1.0, 0], D2=8.0, Pe=1.0, coup_in=180, coup_out=-180
n_particle_types: 3 (all identical: M1=-4, M2=4, consumption=45, production=-45)
shuffle_particle_types: false (odd iteration 7/8)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=[0.85], plateau=[0.00], in_box=[100.0]%, clustering=[0.37]
C1_std=2.82, C2_std=1.53, pattern_growth=306.2x
Assessment:
  - Symmetry: other (labyrinthine/disordered — not clean stripes, not hexagonal)
  - Particles: clustered (moderate, 0.37 — scattered micro-clusters without periodic organization)
  - Stability: unstable (plateau=0.00, no convergence)
  - Novelty: variant (labyrinthine with 3-type is new, but messy)
Visual: (1) Frame 1: standard 3-layer concentric rings (blue center, orange ring, green outer). (2) Frame 60: particles have mixed and spread. C1 shows emerging labyrinthine structure with dark channels and bright spots. C2 shows complementary pattern with isolated yellow blobs near edges. Particles mostly green-dominant in center with orange/blue scattered. Clustering=0.46 mid-sim. (3) Frame 120: C1 labyrinthine persists, C2 has large yellow blobs at edges. Particles dispersed with moderate micro-clustering (0.37). No clean periodic structure — neither stripe nor hexagonal.
Mutation: n_particle_types: 1 → 3 (all identical params); parent=31 (original 1-type stripe A=3,B=10)
Observation: **3-type ALL IDENTICAL feedback at Turing boundary DOES NOT reproduce clean stripes.** Iter 31 (1-type, same A=3, B=10) produced clean stripes/labyrinth. Iter 54 (2-type reversed feedback) destroyed stripes completely. Now iter 55 (3-type identical) produces messy labyrinthine — better than iter 54 but not clean. This means stripe mode is sensitive to particle density per type: 9600 particles of one type creating uniform feedback is different from 3x3200 particles even with identical params. The spatial distribution of 3 types (concentric rings initially) creates localized heterogeneity in feedback, disrupting the delicate Turing boundary balance. **Stripe mode requires BOTH uniform feedback AND uniform particle distribution (1-type).** Entropy=0.85 is higher than typical stripe (0.56-0.70 range) indicating less spatial structure.
Next: parent=31 (stripe success)

## Iter 56: 6/10 - Deep-Turing Brusselator (B=15) + 2-type same-sign → pixel-scale noise, near-uniform particles
Node: id=56, parent=5
Mode/Strategy: explore — test deep-Turing regime (B/[1+A^2]=0.71) with high B to check pattern wavelength change
Config: params_mesh=[1.0, 15.0, 4.5, 15.0, 1.0, 0], D2=8.0, Pe=1.0, coup_in=180, coup_out=-180
n_particle_types: 2 (both M1=-4, M2=4; Type 1: consumption=90, production=-20)
shuffle_particle_types: true (even iteration 8/8)
n_frames: 6000, delta_t: 5E-4, n_particles: 9600
Metrics: entropy=[0.97], plateau=[0.00], in_box=[100.0]%, clustering=[0.16]
C1_std=3.64, C2_std=1.75, pattern_growth=350.0x
Assessment:
  - Symmetry: none (pixel-scale noise, no coherent periodicity)
  - Particles: uniform (entropy=0.97, clustering=0.16 — nearly homogeneous distribution)
  - Stability: unstable (plateau=0.00)
  - Novelty: repeat (poor result — deep Turing produces overly fine patterns)
Visual: Fields show fine-grained pixel-scale mosaic/checkerboard patterns (C1) with scattered bright blobs at edges (C2). No coherent hexagonal or stripe periodicity — wavelength is at mesh resolution limit. Particles start clustered (frame 10: blue center, orange ring from initial shuffled distribution) but disperse to nearly uniform by frame 120. C2 develops a few large bright blobs at corners/edges but most of the field is noisy. Pattern is effectively unstructured at the particle interaction scale.
Mutation: B: 6.5 → 15.0; A: 4.5 (unchanged); B/[1+A^2] = 0.71 (from 0.31); Type 1 consumption: 45→90, production: -45→-20
Observation: **Deep-Turing regime (B/[1+A^2]=0.71, B=15) produces pixel-scale patterns too fine for particle organization.** The dominant Turing mode wavelength decreases with increasing B/[1+A^2] ratio (or with increasing B at fixed ratio). At B=6.5 (B/[1+A^2]=0.31), well-defined hexagonal spots form. At B=15, the pattern wavelength approaches the mesh resolution, creating noise-like fields that cannot drive coherent particle clustering. Additionally, heterogeneous consumption (90 vs 45) with different production (-20 vs -45) further disrupts pattern formation. The high entropy (0.97) confirms near-uniform particle distribution — the fine-grained field gradients average out at the particle scale. **Practical limit: B≤13 for meaningful particle organization on 100×100 mesh.**

## Block 7 Summary (Iters 49-56)

Block 7 explored Brusselator extreme regimes and multi-type interactions.

**Key findings:**
1. **Turing boundary is multi-dimensional**: B/[1+A^2]=1.0 is necessary but NOT sufficient for stripes. Higher absolute B (13 vs 10) at same ratio favors spots. LOW A (≤3) specifically required for stripes.
2. **Stripe mode is 1-type only**: Reversed feedback destroys stripes (iter 54). Even identical 3-type destroys stripes (iter 55). Only 1-type uniform particle distribution achieves clean labyrinthine stripes.
3. **FHN square is intrinsically oscillatory**: All FHN Dv=0.1 configs (iters 51-53) plateau=0.00 regardless of particle feedback. Non-convergence is intrinsic to the FHN square regime.
4. **FHN square = uniform mobility, not 1-type**: 3-type with same-sign mobility preserves square (iter 51-52). Square survives shuffled IC (iter 52) with emergent type sorting. Only opposing mobilities destroy square.
5. **Deep-Turing is too fine**: B=15 with B/[1+A^2]=0.71 produces pixel-scale noise. Practical B limit ≤13 for 100×100 mesh.
6. **Same-sign mobility + reversed feedback**: Creates type-segregation within hexagonal (iter 50) — mobility dominates feedback.
7. **Robustness hierarchy confirmed**: Hexagonal > Square > Stripe.

**Scores**: 8, 8, 7, 8, 7, 7, 7, 6. Average: 7.25/10. Lower than Block 6 (7.38) — exploring boundaries reveals limitations.

**Type distribution this block**: 1-type: 2 (iters 49, 53-proxy); 2-type: 3 (iters 50, 54, 56); 3-type: 3 (iters 51, 52, 55). Well balanced.

## Iter 57: 6/10 - LogSensing + Brusselator hex params → RADIAL (symmetry breaking suppressed)
Node: id=57, parent=5
Mode/Strategy: baseline — first test of PDE_D_LogSensing particle model with proven Brusselator hex config
Config: params=[-4, 4, 45, -45, 1.6, 1.0, 1.6, 1.5], params_mesh=[D1=1.0, Da_c=15, A=4.5, B=6.5, D2=8.0], n_frames=6000, delta_t=5E-4, n_nodes=10000
n_particle_types: 1
particle_model: PDE_ParticleField_D_LogSensing
mesh_model: Diffusiophoresis_Mesh (Brusselator)
Metrics: entropy=[0.73], plateau=[0.00], in_box=[100.0]%, clustering=[0.55], C1_std=[0.18], C2_std=[0.06], pattern_growth=[12.36]
Assessment:
  - Symmetry: radial
  - Particles: clustered (radial rings within disc, not hexagonal)
  - Stability: transient (plateau=0.00, still evolving at frame 6000)
  - Novelty: variant (same params that give hex with base PDE_D now give radial with LogSensing)
Visual: Fields show 3-4 concentric rings centered in domain (C1 dark center with alternating rings, C2 complementary). Particles form a large disc (~60% of domain) with subtle internal radial density modulation matching field ring positions. Small voids/gaps appear at ring boundaries in later frames. No hexagonal spots or symmetry breaking at any point during 6000 frames. Temporal evolution: random→contracting disc→concentric rings with increasing ring count.
Mutation: particle_model: PDE_ParticleField_D → PDE_ParticleField_D_LogSensing (Weber-Fechner log-sensing)
Observation: **LogSensing suppresses hexagonal symmetry breaking.** With identical Brusselator params (A=4.5, B=6.5) that reliably produce hexagonal patterns under base PDE_D, LogSensing maintains purely radial symmetry. The 1/(C+C0) denominator in the sensing equation weakens particle response at high concentration regions, reducing the positive feedback loop needed for azimuthal instability. Clustering metric (0.55) is comparable to base hex configs (0.45-0.53) but the clustering is radial, not hexagonal. The zero plateau suggests the system may still be evolving — possible that hex could emerge with longer simulation or stronger coupling. Field variation (C1_std=0.18) is weaker than typical hex configs (~0.3-0.5), suggesting LogSensing also dampens field pattern amplitude via reduced particle feedback.
Next: parent=57

## Iter 58: 7/10 - LogSensing + doubled mobility (M=±8) → PARTIAL HEXAGONAL (ring→spot transition)
Node: id=58, parent=57
Mode/Strategy: exploit — increase M1/M2 from ±4 to ±8 to compensate for log-sensing dampening
Config: particle_model=PDE_ParticleField_D_LogSensing, mesh_model=Diffusiophoresis_Mesh
  params: [[-8, 8, 45, -45, 1.6, 1.0, 1.6, 1.5]]
  params_mesh: [[1.0, 15.0, 4.5, 6.5, 1.0, 0], [8.0, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
  n_particle_types=1, n_frames=6000, delta_t=5E-4, n_nodes=10000, shuffle=false
n_particle_types: 1
Metrics: entropy=[0.65], plateau=[0.00], in_box=[100.0]%, clustering=[0.54]
  C1_std=0.66, C2_std=0.32, pattern_growth=64.58
Assessment:
  - Symmetry: other (partial hexagonal — ring→spot transition, not fully resolved)
  - Particles: clustered (large aggregates with scattered particles)
  - Stability: unstable (plateau=0.00, max_vel=490 at end, still evolving rapidly)
  - Novelty: variant (intermediate state between radial and hexagonal)
Visual: Dramatic evolution visible in montage: frames 1-4 show concentric rings (like iter 57), frames 5-8 show rings breaking into discrete spots, final frames show ~15-20 field spots with particles concentrated in ~5-8 large clusters. Fields show strong Turing patterns (C1_std=0.66, C2_std=0.32 — much stronger than iter 57's 0.18). Particle clusters are irregular, not yet organized into hexagonal lattice. Checkerboard artifacts in field spots at final frame suggest numerical instability from high mobility.
Mutation: params M1,M2: [-4,4] -> [-8,8] (doubled mobility to compensate for log-sensing dampening)
Observation: **Doubling mobility partially restores symmetry breaking with LogSensing.** At M=±4 (iter 57), LogSensing stayed purely radial. At M=±8, radial symmetry breaks into spots, but the system does not converge — plateau=0.00 and velocities remain extremely high (max=490). The stronger mobility overcomes LogSensing's dampening at high C but also creates dynamical instability. C1_std jumped from 0.18 (iter 57) to 0.66, confirming the stronger particle-field feedback amplifies field patterns. However, the "self-limiting aggregation" benefit of LogSensing may be lost when M is too high — particles still form dense clusters (clustering=0.54, similar to base hex). The 1/(C+C0) factor ≈ 0.18 at C=4.4 means effective mobility is ~1.4 (M*factor = 8*0.18), comparable to the M=4 linear regime but with nonlinear spatial variation.
Next: parent=57

## Iter 59: 5/10 - LogSensing + Brusselator stripe mode → DISORDERED NOISE (no stripes)
Node: id=59, parent=57
Mode/Strategy: explore — test LogSensing with stripe-mode Brusselator (A=3, B=10, B/[1+A^2]=1.0)
Config: params_mesh=[[1.0, 15.0, 3.0, 10.0, 1.0, 0], [8.0, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]], params=[[-8, 8, 45, -45, 1.6, 1.0, 1.6, 1.5]], n_frames=6000, delta_t=5E-4, n_particles=9600, n_nodes=10000
n_particle_types: 1
particle_model: PDE_ParticleField_D_LogSensing
mesh_model: Diffusiophoresis_Mesh
Metrics: entropy=[0.83], plateau=[0.00], in_box=[100.0]%, clustering=[0.44]
C1_std=2.86, C2_std=1.67, pattern_growth=333.8
Assessment:
  - Symmetry: none
  - Particles: clustered (weak, central concentration)
  - Stability: unstable (plateau=0.00, still evolving)
  - Novelty: repeat (disordered like other LogSensing failures)
Visual: Fields show pixel-scale noisy patterning rather than clean stripes/labyrinth. C1 has fine-grained checkerboard with some larger dark patches, NOT the coherent stripe domains seen with base PDE_D at same params. C2 shows yellow blobs on purple background with similar pixel-scale noise. Particles form weak central concentration with radial falloff — no stripe-aligned organization, no hexagonal. Very high C1_std=2.86 indicates wildly varying concentrations but without coherent spatial structure. Compare to base PDE_D stripe mode (iter 31) which produced clean parallel bands.
Mutation: mesh params A=3, B=10 (stripe mode), particle_model=LogSensing with M=±8
Observation: **LogSensing destroys Turing-boundary stripe mode selection.** The stripe mode at B/[1+A^2]=1.0 requires delicate balance of field dynamics — the concentration-dependent mobility (1/(C+C0)) from LogSensing creates spatially inhomogeneous feedback that disrupts the coherent stripe formation. At M=±8 the effective mobility varies from ~1 (at high C) to ~8 (at low C), creating extreme spatial heterogeneity in particle-field coupling. The result is pixel-scale noise rather than the clean stripes base PDE_D achieves. Combined with iter 57-58 findings: LogSensing suppresses hex at M=±4, achieves partial hex at M=±8, and destroys stripes at M=±8. The nonlinear mobility is fundamentally incompatible with the delicate Turing mode selection.
Next: parent=57

## Iter 60: 8/10 - LogSensing + FHN hex (Dv=0) → NOVEL SQUARE-NETWORK morphology
Node: id=60, parent=57
Mode/Strategy: explore — switch mesh model to FHN (different field dynamics) with LogSensing particle model
Config: particle_model=PDE_ParticleField_D_LogSensing, mesh_model=PDE_Diffusiophoresis_FHN
params=[-8, 8, 45, -45, 1.6, 1.0, 1.6, 1.5] (M=±8, 1-type)
params_mesh: FHN [Du=0.5, a=0.7, b=0.8, eps=0.08, I_ext=0, time_scale=20], Dv=0.0, Pe=1.0, coup=±180
n_particle_types: 1
n_frames: 6000, delta_t=0.0005, n_nodes=10000
Metrics: entropy=0.66, plateau=0.00, in_box=100%, clustering=0.46
C1_std=1.16, C2_std=0.64, pattern_growth=128.7
Assessment:
  - Symmetry: other (square-network, ~4-fold)
  - Particles: network (skeletal structure with square symmetry)
  - Stability: unstable (plateau=0.00, still dynamically evolving)
  - Novelty: novel (first LogSensing non-radial non-hexagonal pattern)
Visual: Dramatic pattern evolution over 6000 frames. Early: noisy → radial ring (frames 1-4). Mid-late: ring breaks into spotty structure THEN develops rectangular/square central region with ~4-6 spots and corner extensions (frames 5-8). Final frame (120): particles form intricate skeletal network with clear 4-fold symmetry — central square ring with arms extending to corners, embedded spot clusters. C1 field shows rectangular central bright region with spot features inside, dark moat, and corner extensions. C2 field shows complementary inverse. Strong pattern_growth=128.7x indicates robust Turing pattern development. Despite plateau=0.00, the structure is visually coherent and complex.
Mutation: mesh_model Brusselator→FHN (with LogSensing particle model retained), Dv=0 (hex mode in base PDE_D)
Observation: **FHN + LogSensing produces NOVEL square-network morphology!** This is remarkable because: (1) FHN with Dv=0 + base PDE_D produces HEXAGONAL; (2) FHN with Dv≥0.1 + base PDE_D produces SQUARE; but (3) FHN Dv=0 + LogSensing produces square-like symmetry. The log-sensing nonlinearity (∇C/C) apparently shifts the effective diffusivity ratio, mimicking what Dv>0 does in the base model. The concentration-dependent mobility creates an implicit v-diffusion effect because particles differentially respond in high-C vs low-C regions, breaking hexagonal symmetry toward 4-fold. This is a genuinely novel interaction between particle model and mesh model — LogSensing acts as an effective Dv>0 via spatially heterogeneous mobility. Pattern_growth=128.7 confirms robust Turing instability (vs 333.8 for iter 59 noisy disorder — here patterns are much more organized). The square-network structure resembles biological vasculature more than Turing spots.
Next: parent=60

## Iter 61: 7/10 - LogSensing + FHN Dv=0 + 2-type opposing → HEXAGONAL (square destroyed)
Node: id=61, parent=60
Mode/Strategy: robustness-test — test if iter 60's novel square-network survives 2-type opposing mobility
Config: particle_model=PDE_ParticleField_D_LogSensing, mesh_model=PDE_Diffusiophoresis_FHN
params: Type 0 [-8, 8, 45, -45, 1.6, 1.0, 1.6, 1.5], Type 1 [8, -8, -45, 45, 1.8, 1.0, 1.1, 1.9]
params_mesh: FHN [Du=0.5, a=0.7, b=0.8, eps=0.08, I_ext=0, time_scale=20], Dv=0.0, Pe=1.0, coup=±180
n_particle_types: 2
n_frames: 6000, delta_t=0.0005, n_nodes=10000
Metrics: entropy=0.77, plateau=0.00, in_box=100%, clustering=0.32
C1_std=12.70, C2_std=2.85, pattern_growth=570.35
Assessment:
  - Symmetry: hexagonal (roughly ~15-20 spots in hex-like arrangement)
  - Particles: segregated (Type 0 clusters on C1 peaks, Type 1 filamentary network between spots)
  - Stability: unstable (plateau=0.00, max_vel=222, still evolving)
  - Novelty: variant (phase separation with dual morphology, but hex symmetry not novel)
Visual: Fields develop ~15-20 spots in roughly hexagonal arrangement by mid-simulation. C1 shows bright spots (mean=5.77, std=12.7 — very strong amplification). C2 shows complementary dark spots. Particles show CLEAR PHASE SEPARATION: orange (Type 0, M1=-8) clusters tightly on C1 peaks forming dense spots; blue (Type 1, M1=+8) is repelled from C1 peaks and forms extended filamentary networks connecting the interstices between spots. The dual morphology (clustered Type 0 + network Type 1) is visually striking but the overall symmetry is hexagonal, NOT square. C1_std=12.7 is 10x higher than iter 60's 1.16, indicating much stronger field amplification with 2-type feedback. Very high pattern_growth=570.35 (vs 128.7 for iter 60).
Mutation: n_particle_types: 1→2, opposing mobility (M1=-8/+8, M2=+8/-8)
Observation: **Opposing mobility destroys LogSensing square-network → hexagonal**, exactly as it destroyed FHN Dv=0.1 square in iter 51. This confirms robustness principle #17: square symmetry requires uniform mobility DIRECTION. The phase-separated dual morphology (cluster + network) is interesting but follows established 2-type hexagonal behavior. Key insight: LogSensing square-network (iter 60) has the SAME fragility as FHN Dv=0.1 square — both require same-sign mobility. The 10x stronger field amplification with 2-type (C1_std=12.7 vs 1.16) suggests the opposing consumption/production feedback amplifies Turing instability. Higher entropy (0.77 vs 0.66) and lower clustering (0.32 vs 0.46) reflect the more distributed, network-like particle organization.
Next: parent=60

## Iter 62: 8/10 - LogSensing + FHN Dv=0.1 (dual square mechanisms) → ENHANCED SQUARE
Node: id=62, parent=60
Mode/Strategy: explore — combine both square-promoting mechanisms: LogSensing (effective Dv>0) + intrinsic FHN Dv=0.1
Config: particle_model=PDE_ParticleField_D_LogSensing, mesh_model=PDE_Diffusiophoresis_FHN
params=[[-8, 8, 45, -45, 1.6, 1.0, 1.6, 1.5]]
params_mesh: FHN [Du=0.5, a=0.7, b=0.8, eps=0.08, I_ext=0, time_scale=20], Dv=0.1, Pe=1.0, coup=±180
n_particle_types: 1, n_particles: 9600, n_frames: 6000, delta_t: 5E-4
Metrics: entropy=0.63, plateau=0.00, in_box=100%, clustering=0.48,
C1_std=1.15, C2_std=0.64, pattern_growth=127.03
Assessment:
  - Symmetry: square (clear 4-fold symmetry with concentric square bands)
  - Particles: network (skeletal square-network filaments with nodes at grid intersections)
  - Stability: transient (plateau=0.00, intrinsically oscillatory as established)
  - Novelty: variant (enhanced square vs iter 60, but same symmetry class)
Visual: Montage shows clear progression from random→rings→square. Early frames (1-2) show noise, frames 3-4 develop concentric rings, frames 5-6 break into 4-fold square geometry. Late frames (7-10) show well-defined square grid with rounded corners — particles trace square-network filaments. C1 shows prominent square-shaped patches with clear 4-fold symmetry. C2 complementary. The square geometry is MORE DEFINED than iter 60 (LogSensing+Dv=0) — edges are straighter, corners more angular, pattern more regular. Metrics nearly identical to iter 60 (entropy 0.63 vs 0.66, clustering 0.48 vs 0.46, C1_std 1.15 vs 1.16, pattern_growth 127 vs 129).
Mutation: params_mesh Dv: 0.0→0.1 (added intrinsic square mechanism on top of LogSensing)
Observation: **Dual square mechanisms reinforce but don't create qualitatively new behavior.** Both FHN Dv=0.1 and LogSensing act through the same underlying principle — effective v-diffusion. Dv=0.1 adds explicit diffusion; LogSensing creates spatially-varying effective mobility (∇C/C) that acts like implicit v-diffusion. Combining them produces a more geometrically regular square than either alone, but metrics are virtually identical. This confirms that BOTH mechanisms access the SAME symmetry class through the same physical principle. The square pattern is the most well-defined seen in any iteration — straighter edges, more angular corners. Key insight: LogSensing and FHN Dv are additive for square promotion, not multiplicative.
Next: parent=60

## Iter 63: 8/10 - LogSensing + FHN Dv=0, 3-type same-sign → SQUARE-NETWORK (multi-type confirmed)
Node: id=63, parent=60
Mode/Strategy: multi-type — test if LogSensing-induced square-network survives 3-type same-sign mobility
Config: particle_model=PDE_ParticleField_D_LogSensing, mesh_model=PDE_Diffusiophoresis_FHN
params=[[-8, 8, 45, -45, 1.6, 1.0, 1.6, 1.5], [-8, 8, 30, -30, 1.8, 1.0, 1.1, 1.9], [-8, 8, 60, -60, 2.0, 1.0, 2.0, 1.0]]
params_mesh: FHN [Du=0.5, a=0.7, b=0.8, eps=0.08, I_ext=0, time_scale=20], Dv=0.0, Pe=1.0, coup=±180
n_particle_types: 3, n_particles: 9600 (3200 each), n_frames: 6000, delta_t: 5E-4
Metrics: entropy=0.65, plateau=0.00, in_box=100%, clustering=0.36,
C1_std=1.23, C2_std=0.67, pattern_growth=134.14
Assessment:
  - Symmetry: square (clear 4-fold symmetry)
  - Particles: network (square-network with 3-type sub-structure)
  - Stability: transient (plateau=0.00, intrinsically oscillatory)
  - Novelty: variant (LogSensing square-network + multi-type, confirms robustness)
Visual: Montage shows progression from 3-type concentric initialization → ring development → SQUARE symmetry breaking by frame 5-6 → well-defined square-network by late frames. Row 1 (C1): clear transition from random→rings→square patches with 4-fold symmetry. The square geometry is well-defined with spots at grid nodes and connecting structures. Row 2 (particles): 3-type particles initially in concentric bands separate into distinct sub-populations within the square lattice. Different types appear to occupy different positions relative to the square grid — some at nodes, others along edges, creating a rich multi-layered square architecture. Row 3 (fields): Square symmetry in both C1 and C2 with complementary patterns. Row 4 (particle detail): Clear 3-type phase separation with each type tracing different parts of the square-network structure.
Mutation: n_particle_types: 1→3, all same-sign mobility (M1=-8, M2=8), varying consumption/production
Observation: **LogSensing-induced square-network survives 3-type same-sign mobility**, matching the behavior of FHN Dv=0.1 square (iter 52). Clustering dropped (0.36 vs 0.48 for 1-type iter 60) as expected — 3 types distribute particles more broadly across the lattice. The 3-type configuration adds visual richness (type-specific sub-structures) without changing the underlying 4-fold symmetry. This further confirms the established principle: square symmetry requires uniform mobility DIRECTION but is agnostic to particle type count. LogSensing and FHN Dv=0.1 both follow the same robustness rules. The varying consumption/production strengths (45/30/60) create differentiated feedback per type, enriching the morphology without breaking the symmetry.
Next: parent=60

## Iter 64: 8/10 - LogSensing + FHN Dv=0.2, 2-type same-sign → STRONGEST SQUARE with type-segregated boundary
Node: id=64, parent=60
Mode/Strategy: explore — higher Dv (0.2) + LogSensing dual mechanisms, 2-type same-sign mobility for type balance
Config: particle_model=PDE_ParticleField_D_LogSensing, mesh_model=PDE_Diffusiophoresis_FHN
params: Type 0 [-8, 8, 45, -45, 1.6, 1.0, 1.6, 1.5], Type 1 [-8, 8, 30, -30, 1.8, 1.0, 1.1, 1.9]
params_mesh: FHN [Du=0.5, a=0.7, b=0.8, eps=0.08, I_ext=0, time_scale=20], Dv=0.2, Pe=1.0, coup=±180
n_particle_types: 2, n_particles: 9600, n_frames: 6000, delta_t: 5E-4
Metrics: entropy=0.648, plateau=0.00, in_box=100%, clustering=0.467
C1_std=1.121, C2_std=0.618, pattern_growth=123.55
Assessment:
  - Symmetry: square (strong 4-fold symmetry, sharp nested square contours)
  - Particles: segregated (Type 1/orange at square boundary, Type 0/blue fills interior)
  - Stability: transient (plateau=0.00, intrinsically oscillatory — confirmed across all FHN square configs)
  - Novelty: variant (enhanced square with type-boundary segregation)
Visual: Clear evolution from random→concentric rings→SQUARE. Frame 30: early stage with central C1 peak and emerging rectangular deformation, 2-type particles show orange ring + blue disc. Frame 80: fully developed square geometry in C1/C2 — nested square regions with clear 4-fold symmetry, particles widely distributed with weak clustering (0.028). Frame 120: contracted, well-defined nested concentric squares. Orange (type 1) particles form sharp square boundary contour/"cell wall". Blue (type 0) particles sparse inside, distributed. The SHARPEST square edges seen in any iteration — Dv=0.2 + LogSensing creates the strongest 4-fold selection. Corner asymmetries emerging in late frames suggest ongoing dynamics.
Mutation: Dv: 0.1→0.2 (increased intrinsic square mechanism), n_particle_types: 1→2 (same-sign)
Observation: **Dv=0.2 + LogSensing produces the SHARPEST square geometry yet.** The square edges are straighter and corners more angular than Dv=0.1 or Dv=0+LogSensing. Higher Dv amplifies the intrinsic 4-fold mechanism, and LogSensing adds the effective-Dv boost. 2-type same-sign preserves square as expected. Novel finding: TYPE SEGREGATION WITHIN SQUARE — orange particles (Type 1, lower consumption 30 vs 45) concentrate at square BOUNDARY, blue (Type 0, higher consumption) fills interior. This "cell wall" morphology — outer barrier of one type enclosing interior of another — is biologically interesting (resembles epithelial boundary organization). Metrics nearly identical to Dv=0.1 configs (C1_std ~1.1, pattern_growth ~124), confirming additive but not multiplicative enhancement. All FHN square configs plateau=0.00 regardless of Dv or particle model.
Next: parent=60

## Block 8 Summary (Iters 57-64)

Block 8 introduced the first particle model modification: **PDE_D_LogSensing** (logarithmic gradient sensing, Weber-Fechner law).

**Key findings:**
1. **LogSensing is Brusselator-incompatible**: 3 Brusselator tests failed — M=±4 gives radial (iter 57), M=±8 gives partial hex (iter 58) or noise (iter 59). LogSensing's spatially varying effective mobility (∇C/C) disrupts Brusselator's delicate Turing mode selection.
2. **LogSensing is FHN-compatible and transformative**: LogSensing + FHN Dv=0 produces SQUARE-NETWORK (iter 60), which FHN Dv=0 + base PDE_D gives hexagonal. LogSensing acts as effective Dv>0 via concentration-dependent mobility.
3. **LogSensing and FHN Dv are additive**: Both promote square through same underlying mechanism (effective v-diffusion). Combining them gives sharper geometry (iters 62, 64) but same metrics.
4. **LogSensing square follows same robustness rules**: Same-sign mobility preserves square (iters 60, 62-64), opposing mobility destroys it (iter 61). Identical to FHN Dv=0.1 square rules.
5. **Dv=0.2 is optimal for square sharpness**: Highest tested Dv produces sharpest edges with LogSensing (iter 64).
6. **Type segregation creates "cell wall" morphology**: 2-type same-sign with different consumption rates produces boundary-interior segregation (iter 64) — biologically relevant.

**Scores**: 6, 7, 5, 8, 7, 8, 8, 8. Average: 7.125/10.

**Type distribution this block**: 1-type: 5 (iters 57-60, 62); 2-type: 2 (iters 61, 64); 3-type: 1 (iter 63). Heavily 1-type skewed — Block 9 needs more multi-type.

### Variant: PDE_D_ActiveMatter
Literature: Vicsek et al. (1995) PRL 75:1226; Cates & Tailleur (2015) ARCMP 6:219
Rationale: All previous particle models are purely gradient-following (passive). Active matter introduces self-propulsion (intrinsic motility) and Vicsek-style velocity alignment, enabling fundamentally new collective states: flocking, polar bands, vortices, motility-induced phase separation (MIPS). These dynamics are inaccessible with diffusiophoresis or log-sensing alone. Real cells and bacteria are self-propelled — this brings the model closer to biological reality.
Config: particle_model_name: PDE_ParticleField_D_ActiveMatter
Params reinterpretation: [v0, alignment, gradient_bias, noise_amp, ar_p1, ar_p2, ar_p3, ar_p4]
Key physics: v = v0*heading + gradient_bias*(M1*∇C1 + M2*∇C2) + alignment*(neighbor_vel - vel) + noise

## Iter 65: 7/10 - ActiveMatter + Brusselator hex (1-type) → CLUSTERED + PERSISTENT MOTION
Node: id=65, parent=root
Mode/Strategy: baseline — First test of PDE_D_ActiveMatter with Brusselator hex mode (1-type)
Config: particle_model=PDE_ParticleField_D_ActiveMatter, mesh_model=Diffusiophoresis_Mesh
  params=[0.5, 0.3, 0.5, 0.1, 1.6, 1.0, 1.6, 1.5] (v0=0.5, alignment=0.3, gradient_bias=0.5, noise=0.1)
  params_mesh: D1=1.0, Da_c=15.0, A=4.5, B=6.5; D2=8.0; Pe=1.0, coup_in=180, coup_out=-180
  n_particle_types=1, n_particles=9600, n_frames=6000, delta_t=5E-4
Metrics: entropy=0.5714, plateau=0.0114, in_box=100.0%, clustering=0.5188
  C1_std=1.49, C2_std=0.58, pattern_growth=115.1
Assessment:
  - Symmetry: none (disordered spots, not hexagonal)
  - Particles: clustered (0.52 — strong clustering at spots)
  - Stability: transient (plateau=0.01, persistent motion — hallmark of active matter)
  - Novelty: variant (same clustering as standard PDE_D, but with persistent dynamics)
Visual: 8-12 large irregular spots form. Spots are elongated/oblong rather than circular. Particle clusters show internal substructure. Spots continue to coarsen, merge, and drift throughout simulation — never reaching steady state. Some spots appear to have "hollow" interiors where particles circulate. The elongation and persistent motion are active matter signatures not seen with standard diffusiophoresis.
Mutation: particle_model: PDE_ParticleField_D → PDE_ParticleField_D_ActiveMatter (v0=0.5, alignment=0.3, gradient_bias=0.5, noise=0.1)
Observation: ActiveMatter produces similar clustering strength (0.52) to standard PDE_D but with two key differences: (1) NO steady state — plateau=0.01 vs typical 0.5+ for standard hex. Self-propulsion keeps particles permanently in motion. (2) Spot morphology is more irregular/elongated, suggesting self-propulsion disrupts the circular spot shape. The gradient_bias=0.5 means gradients contribute only 50% — may need to increase for cleaner field-driven organization. Lower entropy (0.57) than standard hex (0.7+) suggests less organized spatial distribution.
Next: parent=65

## Iter 66: 7/10 - ActiveMatter + Brusselator hex (2-type, differential gradient_bias) → TYPE-SEGREGATED CLUSTERS
Node: id=66, parent=65
Mode/Strategy: exploit — 2-type ActiveMatter with differential gradient_bias (type 0: gradient-driven, type 1: self-propelled)
Config: particle_model=PDE_ParticleField_D_ActiveMatter, mesh_model=Diffusiophoresis_Mesh
  params type 0=[0.3, 0.3, 0.8, 0.05, 1.6, 1.0, 1.6, 1.5] (v0=0.3, alignment=0.3, gradient_bias=0.8, noise=0.05)
  params type 1=[1.0, 0.5, 0.1, 0.2, 1.6, 1.0, 1.6, 1.5] (v0=1.0, alignment=0.5, gradient_bias=0.1, noise=0.2)
  params_mesh: D1=1.0, Da_c=15.0, A=4.5, B=6.5; D2=8.0; Pe=1.0, coup_in=180, coup_out=-180
  n_particle_types=2, n_particles=9600, n_frames=6000, delta_t=5E-4
n_particle_types: 2
Metrics: entropy=0.7503, plateau=0.0000, in_box=100.0%, clustering=0.1914
  C1_std=1.62, C2_std=0.83, pattern_growth=166.3
Assessment:
  - Symmetry: none (irregular spot arrangement, no clear hex/stripe/square order)
  - Particles: clustered (weak — 0.19, dispersed compared to 1-type)
  - Stability: transient (plateau=0.00, never reaches steady state)
  - Novelty: variant (type segregation effect is new — tight blue clusters vs diffuse orange)
Visual: Clear type segregation emerges. Type 0 (orange, gradient-driven, v0=0.3, gradient_bias=0.8) forms broad diffuse clusters tracking field maxima. Type 1 (blue, self-propelled, v0=1.0, gradient_bias=0.1) concentrates into tight compact clumps at select locations — characteristic of motility-induced phase separation (MIPS). Spots are irregular, various sizes, some elongated. Field patterns are stronger than iter 65 (C1_std 1.62 vs 1.49, pattern_growth 166 vs 115). Higher entropy (0.75 vs 0.57) reflects more complex spatial organization from 2-type interaction.
Mutation: n_particle_types: 1 → 2, params type differentiation (v0=0.3/1.0, gradient_bias=0.8/0.1, noise=0.05/0.2)
Observation: Differential gradient_bias creates clear functional differentiation: gradient-followers (type 0) vs self-propelled (type 1). Self-propelled type forms tight MIPS-like clusters while gradient-followers spread more diffusely. Lower clustering metric (0.19 vs 0.52) because orange particles are broadly dispersed rather than concentrated. Stronger field patterns (higher C1_std, pattern_growth) suggest 2-type feedback differently reinforces Turing instability. Still no new symmetry class — active matter disrupts hexagonal ordering into irregular spots. Zero plateau confirms active matter permanently prevents steady state.
Next: parent=65

## Iter 67: 6/10 - ActiveMatter + Brusselator STRIPE mode → RING/SHELL (no stripes)
Node: id=67, parent=65
Mode/Strategy: exploit — ActiveMatter + Brusselator stripe mode (A=3, B=10, B/[1+A^2]=1.0), 1-type
Config: params=[0.5, 0.3, 0.7, 0.1, 1.6, 1.0, 1.6, 1.5], params_mesh=[[1.0, 15.0, 3.0, 10.0, 1.0, 0], [8.0, 4, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 1
particle_model: PDE_ParticleField_D_ActiveMatter, mesh_model: Diffusiophoresis_Mesh
Metrics: entropy=0.64, plateau=0.005, in_box=100%, clustering=0.54
  C1_std=3.01, C2_std=2.16, pattern_growth=431.7
Assessment:
  - Symmetry: radial (ring/shell morphology)
  - Particles: network (dense outer ring with interior fragmentation)
  - Stability: transient (plateau=0.005, never reaches steady state)
  - Novelty: variant (ring/shell structure is recognizable but distinct from standard radial)
Visual: Particles form a **ring/shell structure** — dense outer boundary with internal sub-structure (holes, fragmented clumps). NOT stripes. The C1 field shows pixel-scale checkerboard noise outside the particle region with smoother structure inside. C2 field shows bright concentric region with ring features. Very high pattern_growth (431.7) but fields are noisy/checkerboard rather than clean Turing patterns. The ring shape slowly drifts and deforms over time. Interior fragments reorganize frame-to-frame. High clustering (0.54) from dense boundary ring.
Mutation: mesh params A=4.5→3.0, B=6.5→10.0 (stripe regime), gradient_bias=0.5→0.7
Observation: **ActiveMatter completely destroys stripe mode.** Self-propulsion prevents particles from passively following field stripe gradients. Instead, particles aggregate into a ring/shell structure — likely driven by self-propulsion + gradient-biased heading creating a circulation that concentrates particles at a boundary. The checkerboard noise in C1 suggests the Turing boundary condition (B/[1+A^2]=1.0) combined with particle feedback produces pixel-scale instability rather than clean stripes. This confirms that stripes are the MOST fragile symmetry mode — they fail with multi-type, reversed feedback, AND active matter. Higher pattern_growth (431.7 vs 115 in iter 65) suggests the stripe-regime field dynamics amplify more strongly but into noise rather than organized patterns.
Next: parent=65

## Iter 68: 7/10 - ActiveMatter + FHN hex (Dv=0) → BOUNDARY NETWORK (novel organization)
Node: id=68, parent=65
Mode/Strategy: exploit — ActiveMatter + FHN hex mode (Dv=0), 1-type. Direct comparison with iter 65 (ActiveMatter + Brusselator hex).
Config: params=[0.5, 0.3, 0.5, 0.1, 1.6, 1.0, 1.6, 1.5], params_mesh=[[0.5, 0.7, 0.8, 0.08, 0.0, 20], [0.0, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 1
particle_model: PDE_ParticleField_D_ActiveMatter, mesh_model: PDE_Diffusiophoresis_FHN
Metrics: entropy=0.79, plateau=0.036, in_box=100%, clustering=-0.13
  C1_std=10.15, C2_std=1.81, pattern_growth=361.3
Assessment:
  - Symmetry: none (irregular large patches)
  - Particles: network (filamentary boundary-tracing)
  - Stability: transient (plateau=0.036, higher than prior ActiveMatter runs)
  - Novelty: variant (boundary-network distinct from Brusselator ActiveMatter results)
Visual: Particles form **thin filamentary networks tracing boundaries between large C1 field domains**. Unlike Brusselator+ActiveMatter (tight spot clusters), FHN+ActiveMatter produces connected edge networks. C1 field develops large irregular patches (coarser than standard FHN fine spots). C2 field anti-correlated as expected. The boundary-network morphology resembles cell-wall patterns (cf. iter 64 with LogSensing 2-type) but achieved here via active matter with 1-type. Temporal evolution: uniform → radial → develops into large irregular domains with particles outlining the interfaces. Negative clustering (-0.13) indicates anti-clustering — particles are spread into thin filaments rather than concentrated into clumps.
Mutation: mesh_model: Diffusiophoresis_Mesh → PDE_Diffusiophoresis_FHN, params_mesh changed from Brusselator to FHN
Observation: **FHN + ActiveMatter produces qualitatively different particle organization than Brusselator + ActiveMatter.** Brusselator gives disordered spots (clustering=0.52); FHN gives boundary networks (clustering=-0.13). This is consistent with the established FHN-vs-Brusselator dichotomy (FHN=dispersed, Brusselator=clustered) but amplified by active matter. The boundary-network organization is biologically interesting — resembles cell membranes or tissue boundaries. Higher plateau (0.036 vs 0.005-0.01) suggests FHN partially tames active matter persistent motion. Very high C1_std (10.15 vs 1.49 for iter 65) indicates FHN generates much stronger field patterns with ActiveMatter than Brusselator does. Still no new SYMMETRY class but the network ORGANIZATION is distinct.
Next: parent=65

## Iter 69: 7/10 - ActiveMatter + FHN SQUARE (Dv=0.1) → boundary network (square destroyed)
Node: id=69, parent=65
Mode/Strategy: exploit — ActiveMatter + FHN SQUARE mode (Dv=0.1), 1-type. Tests whether active matter disrupts square symmetry.
Config: params=[0.5, 0.3, 0.5, 0.1, 1.6, 1.0, 1.6, 1.5], params_mesh=[[0.5, 0.7, 0.8, 0.08, 0.0, 20], [0.1, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 1
particle_model: PDE_ParticleField_D_ActiveMatter, mesh_model: PDE_Diffusiophoresis_FHN
Metrics: entropy=0.77, plateau=0.033, in_box=100%, clustering=-0.089
  C1_std=10.14, C2_std=1.65, pattern_growth=329.5
Assessment:
  - Symmetry: none (irregular large domains, no square lattice)
  - Particles: network (thin boundary-tracing filaments)
  - Stability: transient (plateau=0.033, persistently moving)
  - Novelty: repeat (same boundary-network as iter 68 FHN+ActiveMatter Dv=0)
Visual: Despite using FHN Dv=0.1 (the established square-symmetry regime), ActiveMatter completely destroys the regular square lattice. C1 field shows ~3-5 large irregular domains instead of the expected ~20-30 regular square cells. C2 field similarly coarse and irregular. Particles form **thin boundary networks** tracing domain walls — identical morphology to iter 68 (FHN Dv=0 + ActiveMatter). At frame 40, fragmentary boundary network with interior features. By frame 80, more coherent but rectangular boundaries aligned to box edges (not intrinsic symmetry). By frame 120, boundaries continue reshaping without convergence. Negative clustering (-0.089) confirms anti-clustering/network organization. The key observation: Dv=0 and Dv=0.1 produce indistinguishable results with ActiveMatter — the self-propulsion overwhelms the Dv-dependent symmetry selection mechanism.
Mutation: params_mesh row 1: [0.0, 0, 0, 0, 0, 0] → [0.1, 0, 0, 0, 0, 0] (Dv=0→0.1, hex→square regime)
Observation: **ActiveMatter erases FHN Dv symmetry selection.** Without ActiveMatter, FHN Dv=0 → hexagonal, Dv=0.1 → square. With ActiveMatter, BOTH give identical boundary-network morphology. The self-propulsion velocity (v0=0.5) overwhelms the gradient-driven motion that encodes symmetry information. This confirms the ActiveMatter verdict: it disrupts ALL symmetry classes (hex, stripe, square) by breaking the gradient-following mechanism essential for lattice formation. Metrics nearly identical to iter 68 (entropy 0.77 vs 0.79, clustering -0.089 vs -0.13, plateau 0.033 vs 0.036). ActiveMatter exploration complete — pivoting to remaining open questions (large FHN Dv).
Next: parent=45

## Iter 70: 7/10 - FHN Dv=0.3 (large Dv) + standard diffusiophoresis → COARSE BOUNDARY NETWORK
Node: id=70, parent=51
Mode/Strategy: explore — FHN Dv=0.3 (beyond established square regime) with standard PDE_ParticleField_D, 1-type. Probing whether square symmetry persists or transitions at very large Dv.
Config: params=[[-4, 4, 180, -180, 1.6, 1.0, 1.6, 1.5]], params_mesh=[[0.5, 0.7, 0.8, 0.08, 0.0, 20], [0.3, 0, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]]
n_particle_types: 1
particle_model: PDE_ParticleField_D, mesh_model: PDE_Diffusiophoresis_FHN
Metrics: entropy=0.68, plateau=0.00, in_box=100%, clustering=0.13
  C1_std=8.00, C2_std=1.56, pattern_growth=312.1
Assessment:
  - Symmetry: none (large irregular domains, no lattice)
  - Particles: network (thin boundary-tracing filaments, sparse)
  - Stability: unstable (plateau=0.00, permanently rearranging)
  - Novelty: variant (coarse-domain boundary network, distinct from square and hex)
Visual: Dv=0.3 dramatically increases Turing wavelength compared to Dv=0.1 square. C1 field develops only 3-5 very large irregular domains (vs ~20-30 square cells at Dv=0.1). C2 field anti-correlated, similarly coarse. Particles form **thin filamentary networks** tracing domain boundaries. At frame 20, concentric rings still visible (radial phase). By frame 40, ring structure breaks into large-scale boundary network with interior features — ring+boundary morphology with a central rosette shape. By frame 80, particles compressed into rectangular boundary shell aligned to box edges with few interior filaments. By frame 100-120, open irregular boundary filaments with scattered particles, continuously evolving. Clustering=0.13 (low but positive — some local accumulation along boundaries). plateau=0.00 — never reaches steady state; domains continuously merge and reshape. C1_std=8.0 (slightly lower than Dv=0.1 at ~10), indicating less sharp field contrast with higher diffusion.
Mutation: params_mesh row 1: [0.1, 0, 0, 0, 0, 0] → [0.3, 0, 0, 0, 0, 0] (Dv=0.1→0.3, tripled v-diffusion)
Observation: **FHN Dv=0.3 EXITS the square regime into coarse-domain boundary network.** The Turing wavelength scales with sqrt(Dv), so tripling Dv from 0.1→0.3 increases wavelength by ~1.7×, meaning only 3-5 domains fit in the unit box (vs ~20-30 at Dv=0.1). This completely destroys the lattice structure — too few domains for any regular symmetry. Particles track domain walls as thin filaments but the pattern never stabilizes (plateau=0.00 vs 0.00 for Dv=0.1 square, both oscillatory). This establishes a clear phase diagram: FHN Dv=0 → hexagonal (many small spots), Dv=0.05 → disordered transitional, Dv=0.1-0.15 → square (intermediate domains), Dv=0.3 → coarse boundary network (few large domains). The square regime is bounded: Dv ∈ [0.1, ~0.2]. Beyond Dv≈0.2, too few Turing cells remain for lattice order. Lower entropy (0.68 vs 0.77) confirms less structured spatial distribution. **Dv controls a continuous transition from fine hex → square → coarse network as wavelength increases.**
Next: parent=51

## Iter 71: 6/10 - FHN Dv=0.1 square + 3-type same-sign → BOUNDARY NETWORK (square destroyed)
Node: id=71, parent=51
Mode/Strategy: multi-type — 3-type same-sign with differential consumption on FHN Dv=0.1 (expected square)
Config: FHN params [Du=0.5, a=0.7, b=0.8, eps=0.08, I=0, ts=20], Dv=0.1, M1=-4/M2=4 all types, consumption=220/180/140, n_particle_types=3, n_frames=6000
n_particle_types: 3
Metrics: entropy=0.64, plateau=0.00, in_box=100%, clustering=-0.017, C1_std=20.15, pattern_growth=498.2
Assessment:
  - Symmetry: none (expected square, got coarse boundary network)
  - Particles: network (thin filaments along domain walls)
  - Stability: unstable (plateau=0.00, never settles)
  - Novelty: variant (boundary network seen before at Dv=0.3; but here caused by different mechanism)
Visual: Initial 3-type concentric rings (blue core, orange mid, green outer) → radial breaking → particles migrate to field domain boundaries forming thin filaments. Only 3-5 large irregular field domains (NOT the ~20 regular square cells expected). Green (type 2, lowest consumption=140) dominates filaments; orange (type 0, highest=220) and blue (type 1, middle=180) are sparse scattered dots. C1 field shows large blobs, not square lattice. C2 field shows complementary large-scale structure. Final state: boundary network with type-segregated particle distribution where low-consumption type dominates.
Mutation: n_particle_types: 1→3 + differential consumption (220/180/140) on FHN Dv=0.1 square config
Observation: **Differential consumption with 3-type DESTROYS FHN square symmetry.** The consumption heterogeneity across concentric initial type bands creates asymmetric field depletion, breaking the radial→square transition and instead producing radial→coarse boundary network. Total consumption is the same as uniform 180×9600, so it's the SPATIAL HETEROGENEITY not total amount that disrupts the Turing instability. Green (lowest consumption=140) survives at boundaries because it depletes fields least, maintaining gradient-driven motion. High-consumption types rapidly deplete local fields and lose gradient drive. **3-type square likely requires IDENTICAL params** (no differential consumption), unlike 2-type where moderate differential consumption preserved square (iter 52/64). Key insight: Square symmetry is more fragile to consumption heterogeneity than hexagonal symmetry (which survives multi-type with any params). C1_std=20.15 is very high (2× the Dv=0.1 square value of ~10), indicating field destabilization by heterogeneous consumption.
Next: parent=51

## Iter 72: 6/10 - FHN Dv=0.1 square + 3-type IDENTICAL params → COARSE BOUNDARY NETWORK (square destroyed)
Node: id=72, parent=51
Mode/Strategy: multi-type — test whether square survives 3 particle types when ALL params are identical (no consumption heterogeneity)
Config: FHN Du=0.5, a=0.7, b=0.8, eps=0.08, I=0.0, time_scale=20, Dv=0.1. 3 particle types ALL with M1=-4, M2=4, consumption=180, production=-180, identical ar_p. Pe=1.0, n_particles=9600, n_frames=6000, delta_t=5e-4, n_nodes=10000.
n_particle_types: 3 (identical params)
Metrics: entropy=[0.70], plateau=[0.00], in_box=[100.0]%, clustering=[0.014]
Assessment:
  - Symmetry: none (expected square, got coarse boundary network with 3-5 large irregular domains)
  - Particles: network (thin filaments tracing domain walls, all 3 types mixed)
  - Stability: unstable (plateau=0.00, oscillatory as expected for FHN)
  - Novelty: variant (same boundary network as Dv=0.3/iter70, but different mechanism)
Visual: Initial 3-type concentric rings → radial breaking → particles migrate to field domain boundaries forming thin mixed-type filaments. Only 3-5 large irregular field domains instead of ~20 regular square cells. All three particle types are visible on the filaments (green, orange, blue all present), confirming no type segregation with identical params. C1 field shows 3-5 large irregular blobs, C2 shows complementary structure. By frame 80, particles form clear boundary outline with interior void. By frame 120, boundary network becomes sparser with particles concentrated on field domain walls.
Mutation: n_particle_types: 1→3 with ALL params IDENTICAL (consumption=180, production=-180 for all) on FHN Dv=0.1 square config
Observation: **3-type IDENTICAL params STILL destroys FHN square symmetry.** Even without any consumption heterogeneity, the mere SPATIAL heterogeneity from concentric initial type bands disrupts the uniform Turing instability needed for square. The initial radial type distribution (blue core, orange mid, green outer) creates radial density variation during the radial→pattern transition, biasing toward large irregular domains. Clustering=0.014 (near zero) confirms dispersed boundary network, not clustered spots. C1_std=9.38 matches Dv=0.3 coarse network (8.0) rather than Dv=0.1 square (~10). Key insight: **Square symmetry REQUIRES single particle type.** It is NOT sufficient to make multi-type params identical — the initial spatial distribution of types (concentric bands) breaks the translational symmetry needed for square lattice formation. This is in contrast to hexagonal symmetry which is robust to multi-type at any configuration. **Square fragility is about SPATIAL HOMOGENEITY, not parameter homogeneity.**
Next: parent=root (block end)

---

## Block 9 Summary (Iters 65-72)

**Theme**: Active matter self-propulsion (Vicsek model) + continued FHN/Brusselator exploration.

**Key findings**:
1. **ActiveMatter verdict (5 iters, 65-69)**: Self-propelled particles DISRUPT all symmetry classes (hex, stripe, square). No new symmetry produced. FHN-Brusselator dichotomy persists (Brusselator=tight clusters, FHN=dispersed network). Persistent motion (plateau≈0) + irregular morphology = signature.
2. **FHN Dv phase diagram completed (iter 70)**: Dv=0→hex, Dv=0.05→disordered, Dv=0.1-0.15→square, Dv≥0.3→coarse boundary network. Square bounded: Dv ∈ [0.1, ~0.2].
3. **Square requires SINGLE particle type (iters 71-72)**: Neither differential consumption (iter 71) NOR identical-params 3-type (iter 72) preserves square. The spatial heterogeneity from concentric initial bands is enough to break square. This is fundamentally different from hexagonal, which survives ALL multi-type configs.

**Score average**: (7+7+6+7+7+7+6+6)/8 = 6.6/10 (lowest block average — ActiveMatter mostly failed to improve patterns).

**Implications for next block**: ActiveMatter explored and closed. Need new direction.

## Iter 73
Node: id=73, parent=root
Mode/Strategy: explore — NEW Gierer-Meinhardt mesh model, 1-type baseline
Config: params_mesh=[[0.01, 0.1, 0.02, 0.01, 0.1, 20], [0.5, 0.02, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]], n_frames=6000, delta_t=5E-4, n_particles=9600, n_particle_types=1
mesh_model_name: PDE_Diffusiophoresis_GM, particle_model_name: PDE_ParticleField_D
n_particle_types: 1
Metrics: entropy=0.643, plateau=0.00, in_box=100.0%, clustering=0.063
Assessment:
  - Symmetry: radial (with emerging hexagonal spots)
  - Particles: network (loose chains/clusters connecting field spikes)
  - Stability: unstable (plateau=0.00, high particle velocities vel=14.8 at end)
  - Novelty: variant (GM spots are sharper/spike-like vs Brusselator smooth spots, but overall organization similar to early Brusselator before hex breaking)
Visual: C1 shows sharp spike-like spots with strong radial organization from initial seed. Ring of ~20 spots around center plus outer spots. C2 shows dominant ring structure with bright spots. Particles form loose networks connecting spike locations — NOT tight clusters (clustering=0.06). Pattern grows explosively (3461x). C1 goes NEGATIVE (mean=-22.5) suggesting possible numerical issues. High particle velocity (vel=14.8) indicates ongoing dynamics at frame 99.
Mutation: mesh_model: Diffusiophoresis_Mesh -> PDE_Diffusiophoresis_GM (new model); params_mesh redesigned for GM
Observation: GM produces sharp spike-like field patterns (expected from a^2/h nonlinearity) but particles don't cluster tightly around them — they form loose networks instead. The negative C1 values and zero plateau suggest the GM dynamics are too vigorous or parameters need tuning. The Da/Dh ratio (0.01/0.5 = 0.02) gives strong Turing instability. The time_scale=20 may be too aggressive. The radial bias from initialization hasn't fully broken — compare with Brusselator which fully breaks to hex by 6000 frames. Key finding: GM spots are morphologically different from Brusselator spots (sharper, spike-like), but particle organization is WEAKER (clustering 0.06 vs Brusselator 0.45-0.53).
Next: parent=73

## Iter 74
Node: id=74, parent=73
Mode/Strategy: exploit — GM with reduced time_scale (20→5) + 2-type opposing mobilities
Config: params_mesh=[[0.01, 0.1, 0.02, 0.01, 0.2, 5], [0.5, 0.02, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]], n_frames=6000, delta_t=5E-4, n_particles=9600, n_particle_types=2
mesh_model_name: PDE_Diffusiophoresis_GM, particle_model_name: PDE_ParticleField_D
n_particle_types: 2
Metrics: entropy=0.851, plateau=0.00, in_box=100.0%, clustering=0.335
Assessment:
  - Symmetry: radial (with emerging hexagonal spots interior)
  - Particles: segregated (2-type radial phase separation: orange=shell, blue=interior network)
  - Stability: unstable (plateau=0.00, growing velocities, boundary artifacts in late frames)
  - Novelty: variant (type-segregation similar to Brusselator 2-type but with GM's sharper spots)
Visual: Two changes from iter 73 — reduced time_scale (20→5) and 2-type opposing mobilities. C1 shows emerging hexagonal-like dark spots within a central domain, still with radial bias. C2 complementary pattern. Particles show clear 2-type phase separation: orange (type 0, M1=-4,M2=4) forms outer ring/shell, blue (type 1, M1=4,M2=-4) concentrates in interior forming network connecting spot locations. By frame 60, clustering=0.433 (good), but by frame 120 the system destabilizes with particles spreading outward and field boundary artifacts (yellow edge bleeding). C1 still goes negative (mean=-17.7). Pattern growth explosive (2758x).
Mutation: time_scale: 20 -> 5 (slower GM dynamics); kappa: 0.1 -> 0.2 (stronger saturation); n_particle_types: 1 -> 2 (opposing mobilities)
Observation: Reduced time_scale significantly improved particle-field coupling (clustering 0.063→0.335). The 2-type opposing mobility creates clear radial type segregation as expected. However, the system remains fundamentally unstable — plateau=0.00, velocities increasing over time, and boundary artifacts appear in late frames. Negative C1 values persist despite kappa increase. The GM model's self-regulating a²/h mechanism doesn't fully prevent runaway dynamics at these parameter values. Two key issues: (1) negative concentrations (unphysical), (2) growing domain with boundary artifacts. The pattern is interesting (type-segregated network within ring) but morphologically similar to Brusselator 2-type results. GM needs further parameter tuning for stability, or the model may need different Da/Dh ratios.
Next: parent=73

## Iter 75
Node: id=75, parent=73
Mode/Strategy: exploit — GM stability fix: broader patterns (Da 0.01→0.05), stronger decay (mu_a 0.02→0.05), max saturation (kappa 0.2→0.5), slower dynamics (time_scale 5→2), weaker coupling (consumption 180→90). 1-type to isolate stability question.
Config: params_mesh=[[0.05, 0.1, 0.05, 0.01, 0.5, 2], [0.5, 0.02, 0, 0, 0, 0], [1.0, 90, -90, 0.05, 0, 0]], n_frames=6000, delta_t=5E-4, n_particles=9600, n_particle_types=1
mesh_model_name: PDE_Diffusiophoresis_GM, particle_model_name: PDE_ParticleField_D
n_particle_types: 1
Metrics: entropy=0.708, plateau=0.00, in_box=100.0%, clustering=0.452
Assessment:
  - Symmetry: other (irregular foam/network, not hexagonal or radial)
  - Particles: network (foam-like walls around voids, degrading over time)
  - Stability: unstable (plateau=0.00, accelerating velocities 0.05→1.35, pattern degrading in late frames)
  - Novelty: variant (foam-like network morphology differs from Brusselator hex clusters, but unstable)
Visual: GM with stabilization parameters produces a distinct "foam-like" network morphology. C1 develops large irregular amoeboid spots (not sharp spikes like iter 73, due to higher Da=0.05). C2 anti-correlated with clear deep-blue minima. Particles form a beautiful foam/honeycomb network by frame 60 (clustering=0.540) — walls surrounding void spaces corresponding to field spot locations. However, the network progressively degrades: by frame 80, particles form thin filaments with large voids; by frame 100-120, the filaments break and particles disperse outward. C1 mean goes -1.1→-2.7→-4.0→-5.95 (still negative, worsening). Pattern growth=1672x. Velocity accelerates throughout (0.05→0.33→0.76→1.35).
Mutation: Da: 0.01 -> 0.05 (smoother patterns); mu_a: 0.02 -> 0.05 (faster decay); kappa: 0.2 -> 0.5 (max saturation); time_scale: 5 -> 2 (slower); consumption: 180 -> 90 (weaker coupling); n_particle_types: 2 -> 1 (isolate stability)
Observation: The five-parameter stabilization improved clustering (0.335→0.452) and produced a visually interesting foam-like network morphology that's qualitatively different from Brusselator hexagonal clusters. The foam structure (particles = walls, voids = field spots) is a network type not seen with other models. However, GM REMAINS fundamentally unstable — plateau=0.00 in all 3 iterations (73-75) regardless of parameter adjustments. Negative C1 persists and worsens over time. The foam network degrades into thin filaments then disperses. Key insight: GM's a²/h nonlinearity produces inherently different morphology (foam networks vs tight clusters) but the lack of stability may be fundamental to the GM formulation in this code. Three consecutive unstable results suggest either (a) a code-level issue with GM implementation or (b) GM needs radically different parameters. Given 3 consecutive unstable results, consider pivoting away from GM for iter 76.
Next: parent=root

## Iter 76: 7/10 - High-res Brusselator (150×150) + 3-type ecosystem → finer hex spots but slow convergence
Node: id=76, parent=root
Mode/Strategy: explore — Pivot from GM (3 consecutive unstable). Test higher mesh resolution (22500 nodes = 150×150) with proven Brusselator 3-type config. Addresses open question about resolution AND balances particle type distribution.
Config: Brusselator mesh (D1=0.05, Da_c=15, A=4.5, B=6.5), D2=1.0, PDE_D standard, 3-type ecosystem (Type 0: M1=-4/M2=4/cons=180/prod=-180; Type 1: M1=4/M2=-4/cons=-90/prod=90; Type 2: M1=0/M2=0/cons=0/prod=0), n_nodes=22500 (150×150), 6000 frames, delta_t=5E-4
n_particle_types: 3
Metrics: entropy=0.741, plateau=0.035, in_box=100.0%, clustering=0.485
Assessment:
  - Symmetry: radial (emerging hexagonal spots but not fully broken radial symmetry)
  - Particles: clustered (Type 0 orange forms network walls, Type 2 green stays as passive ring, Type 1 blue at center)
  - Stability: transient (plateau=0.035, pattern still evolving/rotating, not converged in 6000 frames)
  - Novelty: repeat (same Brusselator hexagonal morphology as 100×100, just finer resolution)
Visual: 150×150 mesh resolves ~7-8 C1 dark spots in quasi-hexagonal arrangement within larger circular structure, vs ~5-6 at 100×100. C2 shows complementary bright spots. Particles show strong 3-type radial segregation: orange (Type 0, M1=-4) dominates outer region and forms walls/networks around field spot voids; green ring (Type 2, neutral M=0) acts as passive barrier at fixed radius; blue core (Type 1, M1=4) stays at center. C1_mean=3.87±0.81, C2_mean=1.69±0.47. Pattern growth=94x. Mid-simulation (frame 30) shows emerging hex with 8 spots; late frames (90-120) show pattern rotation and drift with ~6-7 spots, suggesting ongoing dynamics not converging. The neutral Type 2 (zero mobility, zero consumption) adds no interesting dynamics — just sits as inert barrier.
Mutation: n_nodes: 10000 -> 22500 (100×100 -> 150×150 mesh); n_particle_types: 1 -> 3 (with neutral Type 2)
Observation: Higher resolution (150×150) does NOT unlock qualitatively new patterns. It provides finer spatial detail (more spots fit in domain) but at the cost of much slower convergence — plateau drops from typical 0.3-0.5 at 100×100 to just 0.035. This suggests the system needs proportionally longer simulations at higher resolution. The 3-type config with a neutral type (M=0, cons=0) is uninteresting: the neutral type acts as an inert barrier, not participating in pattern formation. Effective dynamics are 2-type (attracted + repelled) with a passive ring. For multi-type to be interesting, ALL types need nonzero coupling. Key result: **Higher resolution = same physics, finer details, slower convergence. Not worth the cost unless targeting specific wavelength questions.**
Next: parent=root

## Iter 77: 7/10 - FHN oscillatory regime (I=0.4) + 2-type opposing → hex spots, NO traveling waves, unstable
Node: id=77, parent=root
Mode/Strategy: explore — FHN oscillatory regime (I=0.4 external current pushes past Hopf bifurcation). Tests whether oscillatory dynamics produce traveling waves or spiral waves. 2-type opposing mobilities for particle type balance. Dv=0 (hex baseline) to isolate effect of I alone.
Config: FHN mesh (Du=0.05, a=0.7, b=0.8, epsilon=0.08, I=0.4, time_scale=50), Dv=0.0, PDE_D standard, 2-type opposing (Type 0: M1=-4/M2=4/cons=180/prod=-180; Type 1: M1=4/M2=-4/cons=-90/prod=90), n_nodes=10000 (100×100), 6000 frames, delta_t=5E-4
n_particle_types: 2
Metrics: entropy=0.690, plateau=0.000, in_box=100.0%, clustering=0.267
Assessment:
  - Symmetry: hexagonal (standard FHN hex spots, ~15-20 spots)
  - Particles: segregated (orange=spot clusters, blue=filamentary chains between spots)
  - Stability: unstable (plateau=0.00, velocities remain high ~60, no convergence)
  - Novelty: variant (filamentary blue type chains are morphologically interesting)
Visual: C1 field develops strong hexagonal Turing spots (~15-20 bright spots) by frame 60+. C2 shows complementary dark spots with merging/amoeboid boundaries. NO traveling waves or spirals despite I=0.4 oscillatory regime. Particle dynamics: Type 0 (orange, M1=-4) concentrates at C1 peaks forming small clusters at spot locations. Type 1 (blue, M1=4) forms elongated filamentary chains/networks connecting between spots — resembling a cytoskeletal network. This blue filament morphology is distinct from prior 2-type hex results where both types form compact clusters. C1_mean grows explosively (2→113, 735x growth). Velocities never plateau (vel=59 at final frame). System oscillates continuously — I=0.4 prevents settling.
Mutation: mesh_model: Brusselator → FHN; I: 0 → 0.4 (oscillatory regime); n_particle_types: varies → 2
Observation: **FHN oscillatory regime (I=0.4) does NOT produce traveling waves.** Instead, it creates standard hexagonal Turing spots but with permanent instability (plateau=0.00). The external current I=0.4 prevents the system from reaching steady state — particles continue oscillating indefinitely. The blue type forming filamentary chains between spots is a visually interesting variant of phase separation, but the lack of convergence limits pattern quality. Key insight: **Turing patterns dominate over oscillatory dynamics in the coupled particle-field system** — particle coupling pins the pattern spatially, preventing wave propagation even in the oscillatory regime. This answers the open question about FHN oscillatory regime.
Next: parent=root

## Iter 78: 6/10 - Brusselator ASYMMETRIC DIFFUSION (D1=0.2, 4× standard) → elongated blobs, collapsed particles
Node: id=78, parent=root
Mode/Strategy: explore — test asymmetric diffusion (D1=0.2 vs standard 0.05) to change Turing wavelength
Config: Brusselator (D1=0.2, Da_c=15, A=4.5, B=6.5), PDE_D standard, M1=-4/M2=4, 1-type, 6000 frames, n_nodes=10000
n_particle_types: 1
Metrics: entropy=[0.37], plateau=[0.00], in_box=[100.0]%, clustering=[0.80]
Assessment:
  - Symmetry: other (elongated/anisotropic blobs — mixed spot-stripe character)
  - Particles: collapsed (single dense central cluster, sparse scatter elsewhere)
  - Stability: unstable (plateau=0.00, velocities growing, max=263)
  - Novelty: variant (field morphology is novel — elongated worm-like spots — but particle organization degraded)
Visual: D1=0.2 (4× standard) produces LARGER Turing features as expected from wavelength scaling λ∝√(D1/D2). Fields show ~15-20 elongated blob-like spots that are noticeably bigger, more irregular, and more anisotropic than standard D1=0.05 hexagonal spots. Some are worm-shaped (proto-stripe character), others round. Pattern breaks radial symmetry by frame 40 into an irregular arrangement. However, particles COLLAPSE: by frame 80, most particles are in a single dense central cluster with only sparse scatter at other spots. Clustering=0.804 (highest this block) but this reflects collapse, not distributed clustering. Spatial entropy=0.374 is LOW (concentrated). C1_std=1.51, C2_std=1.12 — strong field variation. C1_mean=3.56 (healthy, no negative concentrations). Pattern growth=224x.
Mutation: D1: 0.05 -> 0.2 (4× increase in activator diffusion)
Observation: **High D1 changes Turing wavelength but causes particle collapse.** The 4× increase in D1 produces larger, more elongated features (as predicted by λ∝√D1), creating worm-like blobs with mixed spot-stripe character. This is morphologically interesting for fields. BUT particles collapse to center rather than distributing across spots. Hypothesis: larger pattern wavelength = fewer spots = stronger competition for particles = winner-take-all collapse. The standard D1=0.05 gives ~5-6 closely spaced spots that particles distribute among; D1=0.2 gives wider spacing and particles funnel into the nearest/strongest spot. **Asymmetric diffusion answers open question: it DOES change morphology (elongated features) but DEGRADES particle organization (collapse).** Intermediate D1 (e.g., 0.1) might preserve elongated character while preventing collapse.
Next: parent=root

## Iter 79: 2/10 - Extreme feedback asymmetry → NaN EXPLOSION (total system collapse)
Node: id=79, parent=root
Mode/Strategy: explore — 3-type extreme feedback asymmetry with proven Brusselator hex
Config: Brusselator (D1=0.05, Da_c=15, A=4.5, B=6.5), PDE_D standard, 3-type (Type 0: M1=-4/M2=4/cons=360/prod=-180; Type 1: M1=4/M2=-4/cons=90/prod=-360; Type 2: M1=-4/M2=4/cons=180/prod=-90), n_nodes=10000, 6000 frames
n_particle_types: 3
Metrics: entropy=[0.00], plateau=[0.00], in_box=[0.0]%, clustering=[NaN]
Assessment:
  - Symmetry: none (NaN collapse)
  - Particles: collapsed (0% in box — escaped/NaN)
  - Stability: unstable (NaN explosion ~frame 15-20)
  - Novelty: repeat (failure mode)
Visual: Early frames (1-10) show normal 3-type concentric rings with emerging Brusselator spots + satellite blobs. By frame 10, max velocity=59.5 (dangerously high). By frame 20, complete NaN explosion — all fields blank, all particles gone. Consumption=360 (2x standard) created runaway positive feedback destroying the system.
Mutation: consumption: [180,180,180] -> [360,90,180] (extreme asymmetry); production: [-180,-180,-180] -> [-180,-360,-90] (extreme asymmetry)
Observation: Extreme feedback asymmetry (consumption=360, production=-360) causes NaN explosion. Net chemical depletion/accumulation from asymmetric rates steepens gradients beyond stability. The +/-180 standard values are near the upper limit. Answers open question: extreme feedback asymmetry does NOT create new morphologies — it destroys stability.
Next: parent=root (proven hex config for block end)

## Iter 80: 6/10 - 2-type moderate differential consumption → unstable dispersed hex
Node: id=80, parent=root
Mode/Strategy: explore — 2-type moderate differential consumption within proven Brusselator hex
Config: Brusselator (D1=0.05, Da_c=15, A=4.5, B=6.5), PDE_D standard, 2-type (Type 0: M1=-4/M2=4/cons=240/prod=-120; Type 1: M1=4/M2=-4/cons=120/prod=-240), n_nodes=10000, 6000 frames
n_particle_types: 2
Metrics: entropy=[0.89], plateau=[0.00], in_box=[100.0]%, clustering=[0.08]
Assessment:
  - Symmetry: hexagonal (7-8 spots visible in fields)
  - Particles: segregated (blue=spot centers, orange=shell/matrix) but very loose
  - Stability: unstable (plateau=0.00, max velocity increasing to 118)
  - Novelty: variant (weaker than standard opposing mobility)
Visual: C1 field shows clean hexagonal spots (7-8 dark regions) that develop by frame 20 and persist through frame 120. C2 field shows complementary bright rings at spot boundaries. Particles show 2-type phase separation: blue (Type 1, repelled from C1) concentrates at spot centers forming small ring clusters, orange (Type 0, attracted to C1) forms a broad shell/matrix. However, particle organization is MUCH weaker than standard 2-type configs — clustering drops from 0.422 (frame 20) to 0.082 (final). Velocity keeps increasing (max=118 at end), indicating ongoing instability. The differential consumption (240 vs 120) weakens the field pattern through asymmetric depletion, which degrades particle organization.
Mutation: consumption: [180,180] -> [240,120]; production: [-180,-180] -> [-120,-240]
Observation: Moderate differential consumption weakens particle organization rather than enhancing it. Higher per-type consumption rates (even within supposed stability limits) push plateau to 0 and dramatically reduce clustering (0.08 vs 0.4+ for standard). The asymmetry creates a net chemical imbalance that destabilizes the field-particle feedback loop. Standard symmetric consumption (180 each) is superior. Confirms: consumption/production asymmetry is a dead end for pattern enrichment — symmetric rates are optimal for stability.
Next: parent=root (block end — code modification)

---

## Block 10 Summary (Iters 73-80)

**Focus**: New Gierer-Meinhardt mesh model, higher mesh resolution, FHN oscillatory regime, Brusselator parameter space edges.

**Key Findings**:
1. **Gierer-Meinhardt (GM) model failed**: 3 consecutive unstable results (iters 73-75). Foam-like network morphology is unique but fundamentally unstable (plateau=0 always). Negative C1 concentrations persist despite all stabilization attempts (time_scale reduction, increased damping/diffusion).
2. **Higher mesh resolution (150×150) answered**: Same physics, finer detail, much slower convergence. Not worth the computational cost.
3. **FHN oscillatory regime (I=0.4) answered**: Does NOT produce traveling waves. Turing patterns dominate over oscillatory dynamics; particle coupling pins patterns spatially. Results in permanently unstable hex spots.
4. **Asymmetric diffusion (D1=0.2) answered**: Creates elongated worm-like Turing features (larger wavelength) but particles collapse to a single dense central cluster due to winner-take-all competition.
5. **Extreme feedback asymmetry answered**: consumption=360/production=-360 causes NaN explosion. ±180 is the practical stability ceiling.
6. **Moderate differential consumption answered**: Weaker than symmetric consumption. Asymmetric depletion destabilizes the feedback loop.

**Score Summary**: 73=7/10, 74=7/10, 75=7/10, 76=7/10, 77=7/10, 78=6/10, 79=1/10, 80=6/10. Average=6.0/10.

**New Established Principles**:
- GM model produces foam-like networks but is fundamentally unstable (3/3 iters)
- Higher mesh resolution (150×150) does not unlock new patterns; same physics, slower convergence
- FHN oscillatory regime = destabilized hex, not traveling waves
- Asymmetric D1/D2 changes feature size, not mode selection; degrades particle organization
- Consumption/production asymmetry (per-type) is a dead end; symmetric rates are optimal

**Direction for Block 11**: All mesh models and parameter explorations have been exhausted for new symmetry classes. The remaining frontier is **new PDE_D particle dynamics**. Consider a variant with **density-dependent mobility** — particles that sense and respond to local particle density, not just field gradients. This could create new self-organization through particle-particle feedback mediated by fields.

## Iter 81
Node: id=81, parent=root
Mode/Strategy: code-modification — NEW PDE_D_DensityDependent variant baseline test
Config: params_mesh=[[0.05,15,4.5,6.5,0.02,0], [1.0,0,0,0,0,0], [1.0,180,-180,0.05,0,0]], n_frames=6000, delta_t=5E-4
n_particle_types: 1
particle_model: PDE_ParticleField_D_DensityDependent
mesh_model: Diffusiophoresis_Mesh (Brusselator)
Metrics: entropy=[0.79], plateau=[0.19], in_box=[100.0]%, clustering=[0.53]
Assessment:
  - Symmetry: radial
  - Particles: collapsed (single large disk)
  - Stability: transient (plateau=0.19, still moving)
  - Novelty: variant (density-dependent suppresses hex → radial disk)
Visual: C1 field shows smooth radial gradient (low center, high periphery). C2 complementary. Particles form a single dense circular disk filling ~40% of domain. Disk edge shows lattice-like shell packing. Interior uniformly filled. No hexagonal symmetry breaking occurs — density-dependent mobility traps particles in aggregate where Hill function suppresses motility. Disk slowly expands over simulation but never fragments.
Mutation: particle_model: PDE_ParticleField_D -> PDE_ParticleField_D_DensityDependent (defaults: rho_0=15, hill_n=2)
Observation: **Density-dependent mobility SUPPRESSES Turing-driven particle pattern formation.** The Hill function creates a motility trap: once particles aggregate, local density exceeds rho_0=15, mobility drops, particles can't redistribute to follow field gradients. The normally-reliable Brusselator hex config (A=4.5, B=6.5, M=±4) produces only a single radial blob. Clustering=0.53 (high) but reflects one big aggregate, not multiple clusters. Entropy=0.79 is moderate (disk covers significant area). Key question: is rho_0=15 too low? Higher rho_0 might allow hex formation while still limiting extreme aggregation.
Next: parent=81

## Iter 82
Node: id=82, parent=81
Mode/Strategy: exploit — DensityDependent + Brusselator hex, 2-type opposing mobilities
Config: params=[[-4,4,180,-180,1.6,1.0,1.6,1.5],[4,-4,180,-180,1.6,1.0,1.6,1.5]], params_mesh=[[0.05,15,4.5,6.5,0.02,0],[1.0,0,0,0,0,0],[1.0,180,-180,0.05,0,0]], n_frames=6000, delta_t=5E-4
n_particle_types: 2
particle_model: PDE_ParticleField_D_DensityDependent
mesh_model: Diffusiophoresis_Mesh (Brusselator)
DensityDependent params: rho_0=15 (default), hill_n=2 (default)
Metrics: entropy=[0.82], plateau=[0.00], in_box=[100.0]%, clustering=[0.50]
Assessment:
  - Symmetry: radial (with angular modulation at boundary)
  - Particles: clustered (phase-separated concentric + scalloped boundary)
  - Stability: unstable (plateau=0.00, still evolving)
  - Novelty: variant (scalloped ring = partial symmetry breaking)
Visual: 2-type particles form classic phase separation: Type 0 (blue) dense inner disk, Type 1 (orange) outer ring. Key difference from standard 2-type: the boundary between types develops SCALLOPED modulation. ~20 periodic bumps/protrusions emerge along the ring boundary in late frames (100+). Fields show corresponding spot-like perturbations arranged in a ring at the gradient boundary. This is PARTIAL Turing symmetry breaking — angular modulation at intermediate density (near rho_0) while dense core stays frozen. Interior disk remains smooth and circular. Pattern still evolving at frame 120 (plateau=0.00).
Mutation: n_particle_types: 1 -> 2 (opposing mobilities +/-4), maintaining DensityDependent with rho_0=15
Observation: **2-type opposing mobility partially rescues symmetry breaking under DensityDependent.** The phase separation creates a gradient boundary zone where density is intermediate (~rho_0), allowing Turing perturbations to manifest as angular modulation. Dense regions (core and outer ring) remain trapped. This produces a novel "scalloped ring" morphology — not seen with standard PDE_D (which produces full hex). Plateau=0.00 indicates ongoing evolution; longer simulation might show more breaking. The key insight: density-dependent mobility creates SPATIAL HETEROGENEITY in pattern-forming capacity — patterns emerge only where density is near rho_0.
Next: parent=82

## Iter 83: 6/10 - DensityDependent + FHN hex, 1-type, rho_0=50 → CONCENTRIC RINGS (radial)
Node: id=83, parent=82
Mode/Strategy: explore — DensityDependent + FHN mesh, 1-type. rho_0 raised 15→50.
Config: params=[-4,4,180,-180,1.6,1.0,1.6,1.5], params_mesh FHN [Du=0.5,a=0.7,b=0.8,eps=0.08,I=0,ts=20] Dv=0, n_frames=6000, delta_t=5e-4, n_particles=9600, n_particle_types=1
particle_model: PDE_ParticleField_D_DensityDependent
mesh_model: PDE_Diffusiophoresis_FHN
DensityDependent params: rho_0=50, hill_n=2
n_particle_types: 1
Metrics: entropy=[0.81], plateau=[0.00], in_box=[100.0]%, clustering=[0.21]
Assessment:
  - Symmetry: radial (concentric rings)
  - Particles: clustered (concentric multi-ring + dense core)
  - Stability: unstable (plateau=0.00, still evolving)
  - Novelty: repeat (radial disk, same qualitative outcome as iter 81)
Visual: Particles form concentric rings with a dense central disk. FHN fields develop strong concentric Turing rings (C1: bright center + ring modulation, C2: complementary dark center + rings). By late frames (70-99), C2 field shows angular modulation at outer rings (dark lobes at corners suggesting early 4-fold breaking), but particles remain locked in radial ring mode. Dense core becomes slightly more diffuse internally at late times while outer rings sharpen. Some particles redistribute to box edges. Clustering=0.21 (low, FHN-typical dispersed). The ring structure is more "open" than iter 81's single blob — rho_0=50 allows more spatial spread — but fundamentally still radial, not hexagonal.
Mutation: mesh_model: Brusselator→FHN (Dv=0 hex regime), rho_0: 15→50
Observation: **FHN + DensityDependent (rho_0=50) still produces radial rings, NOT hexagonal.** Raising rho_0 from 15→50 weakened the motility trap enough to allow multi-ring formation (vs single blob at rho_0=15), but the density-dependent mobility still prevents the azimuthal instability needed for hexagonal symmetry breaking. FHN Dv=0 normally produces hex with standard PDE_D, so DensityDependent is the suppressor. Key comparison: iter 81 (Brusselator+DD rho_0=15)=single disk, iter 82 (Brusselator+DD 2-type)=scalloped ring, iter 83 (FHN+DD rho_0=50 1-type)=concentric rings. Pattern: DensityDependent ALWAYS produces radial symmetry regardless of mesh model or rho_0. The Hill function motility suppression creates an isotropic density feedback that locks radial symmetry. Need much higher rho_0 (>>50) or different approach.
Next: parent=82

## Iter 84: 6/10 - DensityDependent + Brusselator hex, 3-type, rho_0=200 (near-disabled) → RADIAL with angular modulation
Node: id=84, parent=82
Mode/Strategy: explore — test rho_0=200 (near-disabled density-dependence) with 3-type complex ecosystem
Config: params=[[-4,4,180,-180,1.6,1.0,1.6,1.5],[4,-4,-180,180,1.8,1.8,1.1,1.9],[0,0,0,0,2.0,1.0,2.0,1.0]], params_mesh=[[0.5,15,4.5,6.5,1,0],[1,0,0,0,0,0],[1,180,-180,0.05,200,2]], n_frames=6000, delta_t=5e-4, particle_model=DensityDependent, mesh_model=Brusselator
n_particle_types: 3
Metrics: entropy=[0.74], plateau=[0.00], in_box=[100.0]%, clustering=[0.52]
Assessment:
  - Symmetry: radial (dominant, with emerging angular modulation)
  - Particles: clustered (concentric 3-type rings with late angular perturbation)
  - Stability: transient (plateau=0.00, still evolving)
  - Novelty: variant (angular modulation within radial is partially new)
Visual: Three particle types form concentric radial bands: blue (type-0) compact core, orange (type-1) middle ring, green (type-2) outer ring. Fields (C1, C2) develop smooth concentric rings early. By frame 60, angular perturbations emerge in ring structure. By frame 90, distinct blob-like features appear embedded in C1 ring valleys (dark spots) and C2 (bright spots). By frame 120, hybrid pattern: concentric rings with embedded azimuthal spots, outer ring boundary shows scalloped/beaded decoration. Orange particles develop angular modulation — some scatter outward along preferred directions forming star-like/rosette pattern. Blue core compactifies. Green outer ring remains relatively smooth. The type-2 (neutral: M1=0, M2=0) particles are field-inert, forming a passive outer boundary.
Mutation: n_particle_types: 1→3 (complex ecosystem template), rho_0: 50→200
Observation: **rho_0=200 (near-disabled) + 3-type still produces radial.** Even with very high rho_0 that effectively disables the Hill function (few particles have 200 neighbors within interaction radius), the DensityDependent code path still anchors radial symmetry. Late angular modulation in fields begins propagating to particles but cannot overcome radial bias. This confirms the radial lock is NOT just from rho_0 being too low — the DensityDependent code itself (or the interaction between density-counting and gradient calculation) introduces isotropic bias. 4/4 DensityDependent iterations are radial. However, the 3-type config here shows slightly more angular structure than 1-type (cf. iter 81), possibly because opposing-mobility type-1 particles create local field perturbations that partially break radial symmetry. The type-2 neutral particles (M=0) are inert bystanders. Conclusion: DensityDependent is a confirmed radial lock regardless of rho_0 or particle type count. Time to pivot away.
Next: parent=root (pivot strategy)

## Iter 85: 8/10 - Brusselator Da_c=5 (slow dynamics) + 2-type opposing → COARSE NETWORK (novel!)
Node: id=85, parent=root
Mode/Strategy: explore — PIVOT from DensityDependent (4/4 dead end). Test lower Da_c=5 (was 15) with standard PDE_D. First time exploring slow Turing reaction kinetics.
Config: params=[[-4,4,180,-180,1.6,1.0,1.6,1.5],[4,-4,-180,180,1.8,1.8,1.1,1.9]], params_mesh=[[0.5,5,4.5,6.5,1,0],[1,0,0,0,0,0],[1,180,-180,0.05,0,0]], n_frames=6000, delta_t=5e-4, particle_model=PDE_ParticleField_D, mesh_model=Diffusiophoresis_Mesh
n_particle_types: 2
Metrics: entropy=[0.85], plateau=[0.00], in_box=[100.0]%, clustering=[0.05]
Assessment:
  - Symmetry: none (irregular network, no periodic lattice)
  - Particles: network (type-0 blue filaments, type-1 orange interstices)
  - Stability: transient (plateau=0.00, still evolving at frame 120)
  - Novelty: novel (coarse irregular network distinct from hex/stripe/square/radial)
Visual: Blue (type-0) particles form an interconnected NETWORK of filaments/boundaries resembling a Voronoi-like cellular skeleton. Orange (type-1) particles fill the interstices as scattered clusters. Fields: C1 shows large irregular blobs (much coarser than Da_c=15 hex spots, ~5-8 features instead of 20-30). C2 shows complementary dark spots. Pattern wavelength ~3-4× larger than standard hex. Temporal: Frame 10 shows initial concentric rings. Frame 40 shows irregular blob breakup and network emergence. Frame 60-90 network is established but continues coarsening. Frame 120 network persists but boundaries shift — still dynamically evolving.
Mutation: Da_c: 15→5 (3× slower Turing reaction), particle_model: DensityDependent→PDE_ParticleField_D (standard)
Observation: **Lower Da_c=5 produces a NOVEL coarse network morphology.** The 3× slower reaction dynamics fundamentally change pattern selection. Instead of ~25 hexagonal spots (Da_c=15), only ~5-8 large irregular blobs form, with particles organizing into network filaments rather than spot clusters. Blue type-0 (negative M1=-4, attracted to C1 minima) traces out a connected skeleton structure reminiscent of foam/Voronoi cells. This is visually distinct from ALL previous patterns (hex, stripe, square, radial, foam). The coarser wavelength comes from slower instability growth: Da_c=5 means the Turing instability develops 3× more slowly, favoring fewer, larger features. Key difference from GM foam (iter 73-75): this network is composed of TYPE-SEGREGATED filaments (blue=skeleton, orange=fill), while GM foam showed all particles in network strands. The clustering=0.049 (very low, near-uniform) confirms network organization. Plateau=0.00 suggests continued evolution — may need more frames or different Da_c to stabilize. This opens a new parameter axis for exploration: Da_c as a wavelength/mode selector. Next: test intermediate Da_c values (Da_c=8-10) or extend frames to see if network stabilizes.
Next: parent=85

## Iter 86: 2/10 - Da_c=5 + 1-type robustness test → NaN CRASH (particle collapse)
Node: id=86, parent=85
Mode/Strategy: robustness-test — test whether Da_c=5 network morphology is intrinsic to slow Turing or requires 2-type opposing mobility
Config: params=[[-4,4,180,-180,1.6,1.0,1.6,1.5]], params_mesh=[[0.5,5,4.5,6.5,1,0],[1,0,0,0,0,0],[1,180,-180,0.05,0,0]], n_frames=6000, delta_t=5e-4, particle_model=PDE_ParticleField_D, mesh_model=Diffusiophoresis_Mesh
n_particle_types: 1
Metrics: entropy=[0.00], plateau=[0.00], in_box=[0.0]%, clustering=[NaN]
Assessment:
  - Symmetry: none (simulation crashed)
  - Particles: collapsed (runaway aggregation → NaN explosion)
  - Stability: unstable (NaN by frame ~25)
  - Novelty: repeat (collapse failure mode seen before)
Visual: Frame 1: initial particle disk, noisy fields. Frame 3: fields developing large C1 depression, particles still in disk (clustering=0.571). Frame 5: field pattern growing (76× pattern), particles contracting. Frame 10: strong radial field pattern, particles collapsed into smaller dense cluster (clustering=0.625). Frame 20: severe collapse — particles squeezed into tiny dense clump with scattered escapees. Max particle velocity=28.66 (runaway). Diffusion=31.09. Frame 30: complete NaN explosion — fields blank, all particles escaped/NaN, all metrics=0/NaN.
Mutation: n_particle_types: 2→1 (removed opposing mobility)
Observation: **Da_c=5 network REQUIRES 2-type opposing mobility.** With 1-type, all 9600 particles have same-sign mobility (M1=-4, M2=4), creating unidirectional convergence toward C1 minima. At standard Da_c=15, field patterns develop fast enough to distribute particles into multiple hex clusters. At Da_c=5, the 3× slower Turing instability cannot fragment the initial aggregate before positive feedback (more particles → more consumption → deeper C1 well → stronger gradient → more particles) drives runaway collapse. By frame 20, particle velocities exceed 28 and the simulation diverges. This explains WHY iter 85 worked: opposing mobilities (type-0 attracted to C1 minima, type-1 repelled) create mutual competition that prevents any single location from accumulating all particles. Conclusion: Da_c=5 is below the 1-type stability threshold — the reaction-diffusion field cannot develop complex patterns before particle aggregation runaway. The "slow Turing network" is a 2-type cooperative phenomenon.
Next: parent=85

## Iter 87: 8/10 - Brusselator Da_c=10 + 2-type opposing → MIXED LABYRINTHINE + SPOTS (intermediate morphology)
Node: id=87, parent=85
Mode/Strategy: exploit — map Da_c transition from coarse network (Da_c=5) to hexagonal (Da_c=15). Test Da_c=10 as intermediate.
Config: params_mesh=[D1=0.5, Da_c=10, A=4.5, B=6.5, mu=1], D2=1.0, Pe=1.0, consumption=180, production=-180. PDE_ParticleField_D + Diffusiophoresis_Mesh. n_particle_types=2 (opposing: M1=-4/+4, M2=+4/-4). n_frames=6000, delta_t=5e-4, n_nodes=10000.
n_particle_types: 2
Metrics: entropy=[0.69], plateau=[0.46], in_box=[100.0]%, clustering=[0.12]
Assessment:
  - Symmetry: other (mixed labyrinthine bands + peripheral spots)
  - Particles: network (elongated curvilinear bands with type segregation)
  - Stability: stable (plateau=0.46 — best in block)
  - Novelty: variant (intermediate between Da_c=5 network and Da_c=15 hex)
Visual: Frame 20: Spiral/concentric ring stage — blue type-0 forms inner spiral arms, orange type-1 traces outer rings. Fields show strong concentric Turing pattern with spiral modulation. Frame 60: Transition to irregular blob morphology — C1 field shows large dark patches (low concentration) with emerging spots on periphery. Blue particles concentrate in elongated curvilinear bands at C1 peaks. Orange dispersed more uniformly. Frame 90: Evolved into mixed mode — elongated blobs + scattered peripheral spots. Blue in thick curvilinear bands (neither rings nor hex). Orange at boundaries. Distinctly BETWEEN Da_c=5 network and Da_c=15 hex. Frame 120: Final state ~5-8 large irregular elongated features (approaching labyrinthine/stripe) with additional peripheral spots. Blue type-0 in thick irregular bands, orange type-1 at boundaries. Fields show coarse mixed spot+stripe morphology. Feature wavelength ~2× standard hex, ~0.5× Da_c=5 network.
Mutation: params_mesh Da_c: 5→10 (intermediate Turing speed, 2× faster than iter 85, 2× slower than standard)
Observation: **Da_c=10 IS a genuine intermediate morphology** — confirms Da_c is a CONTINUOUS wavelength/mode selector, not a binary switch. Key findings: (1) Features scale ~inversely with Da_c: Da_c=5→very coarse network, Da_c=10→intermediate bands+spots, Da_c=15→fine hex spots. (2) **Plateau=0.46** is the highest stability in this block (vs 0.00 for Da_c=5 and all DensityDependent). Intermediate Turing speed promotes convergence — fast enough to fragment particles before runaway, slow enough to reach steady state. (3) Morphology is "mixed labyrinthine" — elongated irregular bands reminiscent of stripe mode but without the ordering. Could represent a Turing mixed mode at this Da_c. (4) Type segregation pattern preserved: blue=band interior, orange=boundary. Same cooperative 2-type architecture as Da_c=5. (5) C1_std=1.24 and C2_std=0.74 confirm robust field pattern development. Da_c transition map: Da_c=5 (coarse network, plateau=0), Da_c=10 (mixed labyrinthine, plateau=0.46), Da_c=15 (hexagonal spots, plateau varies). This maps a new dimension of pattern control.
Next: parent=85

## Iter 88: 6/10 - Brusselator Da_c=5 + 3-type ecosystem → CONCENTRIC SEGREGATED RINGS (type-layered radial)
Node: id=88, parent=85
Mode/Strategy: multi-type — test Da_c=5 coarse network with 3-type ecosystem (addresses type balance).
Config: params_mesh=[D1=0.5, Da_c=5, A=4.5, B=6.5, mu=1], D2=1.0, Pe=1.0, consumption=180, production=-180. PDE_ParticleField_D + Diffusiophoresis_Mesh. n_particle_types=3 (Type 0: M1=-4/M2=4, Type 1: M1=4/M2=-4, Type 2: M1=0/M2=0 neutral). n_frames=6000, delta_t=5e-4, n_nodes=10000.
n_particle_types: 3
Metrics: entropy=[0.79], plateau=[0.00], in_box=[100.0]%, clustering=[0.37]
Assessment:
  - Symmetry: radial (concentric type-segregated layers with angular modulation)
  - Particles: segregated (3 distinct concentric layers: blue core, green ring, orange periphery)
  - Stability: transient (plateau=0.00, still evolving at frame 120)
  - Novelty: variant (combines 3-type layering with Da_c=5 coarse features)
Visual: Frame 30: Three-layer concentric structure forming — blue (type 0, attracted to C1 peaks) clusters centrally, green (type 2, neutral M=0) forms a prominent thick ring at its initialization radius, orange (type 1, repelled from C1 peaks) spreads to periphery with scattered clusters. C1/C2 fields show coarse (~5 spots) central depletion with emerging surrounding pattern. Frame 60: Blue particles develop internal sub-structure (irregular network within central region). Green ring stable and prominent — acts as passive barrier between type 0 and type 1. Orange scattered with some clustering at field spots. C1 shows distinct coarse spots (Da_c=5 wavelength) in exterior region. Frame 90: Blue central network becomes more pronounced — elongated connections visible within green shell. Orange has diffused further into periphery with ~6-8 clusters aligned with C1 spots. Green ring slightly thinner, still prominent. Frame 120: Blue forms messy internal network within green-bounded domain. Orange particles at boundary and periphery with field-guided clustering. Green ring clearly the dominant structural feature. Field patterns (5-8 coarse spots) visible but particles mostly follow type-driven layering rather than field-guided organization.
Mutation: n_particle_types: 2->3, params: added Type 2 (M1=0, M2=0, neutral). Da_c=5 preserved from iter 85.
Observation: **3-type at Da_c=5 produces type-layered radial, NOT the 2-type coarse network.** Key findings: (1) Neutral type 2 (M1=0, M2=0) acts as a passive spatial barrier — particles initialized in the middle ring stay there, creating a "cell wall" that constrains type 0 to interior and type 1 to exterior. (2) This prevents the type-cooperative network formation that made Da_c=5 + 2-type interesting (iter 85). (3) Clustering=0.37 is intermediate — not strong clusters, not a network. Reflects mixed organization. (4) Plateau=0.00 confirms Da_c=5 still doesn't stabilize (consistent with iter 85). (5) The neutral type breaks the opposing-mobility cooperation that made Da_c=5 network novel. (6) Entropy=0.79 is good (dispersed) but patterns are mostly type-driven rather than field-driven. (7) Confirmed: Da_c=5 coarse network is a 2-TYPE COOPERATIVE phenomenon — 3-type disrupts it (consistent with established principle that multi-type disrupts non-hexagonal modes).
Next: parent=85 (block end)

### Block 11 Summary (Iters 81-88)
**Focus**: PDE_D_DensityDependent particle model + Da_c exploration
**Score range**: 6-8/10 (avg 6.9)

**Key findings**:
1. **DensityDependent = DEAD END** (iters 81-84): Hill-function density-dependent mobility locks ALL patterns into radial symmetry regardless of mesh model, rho_0 value, or particle type count. Even near-disabled (rho_0=200) still radial. The density-counting code path introduces isotropic bias. 4/4 iterations confirm failure. Added to Failed Configurations.
2. **BREAKTHROUGH: Da_c is a continuous wavelength/mode selector** (iters 85-88): Reducing Da_c from 15->5 produces fundamentally different morphologies. Da_c=5->coarse network (novel, 5-8 large blobs, Voronoi-like), Da_c=10->mixed labyrinthine bands+spots, Da_c=15->standard hexagonal. This is a genuinely new parameter axis not previously explored.
3. **Da_c=5 requires 2-type opposing mobility** (iter 86): 1-type at Da_c=5 causes NaN crash (slow Turing can't fragment particles before runaway). 3-type breaks cooperative network into radial layers (iter 88).
4. **Da_c=10 has best stability (plateau=0.46)** (iter 87): Intermediate Turing speed provides optimal convergence. Both too slow (Da_c=5, plateau=0) and too fast (Da_c=15) are less stable in 2-type configs.
5. **DensityDependent code variant added to Failed list**: No useful regime found in any combination with Brusselator, FHN, 1/2/3 types.

**Particle type balance this block**: 1-type: 3 iters (81, 83, 86); 2-type: 3 iters (82, 85, 87); 3-type: 2 iters (84, 88). Slightly 3-type under-represented.

---

## Block 12: Anisotropic mobility + cross-model wavelength control (Iters 89-96)

## Iter 89: 8/10 - PDE_D_Anisotropic alpha=0.25 + Brusselator hex → ORIENTED STRIPES (NOVEL!)
Node: id=89, parent=root
Mode/Strategy: code-modification — test new PDE_D_Anisotropic with Brusselator hex, 1-type
Config: params_mesh=[[0.5,15,4.5,6.5,1,0],[1,0,0,0,0,0],[1,180,-180,0.05,0.25,0]], n_frames=6000, delta_t=5e-4, n_particles=9600, n_nodes=10000
n_particle_types: 1
particle_model_name: PDE_ParticleField_D_Anisotropic
mesh_model_name: Diffusiophoresis_Mesh
Metrics: entropy=[0.61], plateau=[0.00], in_box=[100.0]%, clustering=[0.55]
C1_std=1.43, C2_std=0.96, pattern_growth=191.35, pos_std_x=0.130, pos_std_y=0.261
Assessment:
  - Symmetry: stripes (horizontally oriented, mixed with elongated spots)
  - Particles: clustered (along horizontal linear structures)
  - Stability: transient (plateau=0.00, still dynamic)
  - Novelty: novel — ANISOTROPY-INDUCED ORIENTED STRIPES from hex regime!
Visual: Brusselator field (Da_c=15, A=4.5, B=6.5 — normally hex) produces horizontally-oriented elongated features with alpha=0.25. Particles align along horizontal linear structures. pos_std_y/pos_std_x = 2.0, confirming directional bias. Field features are a mix of short horizontal bars and elongated spots rather than standard circular hex spots.
Mutation: code: PDE_D (isotropic) -> PDE_D_Anisotropic (alpha=0.25, weaker x-response)
Observation: **BREAKTHROUGH** — Anisotropic mobility selects stripe-like orientation from WITHIN the hexagonal regime (B/[1+A^2]=0.34, deep in hex territory). This is mechanistically different from Turing-boundary stripes (A=3, B=10). Particles with weaker x-response preferentially move in y, but this creates horizontal trapping: particles quickly traverse vertical gradients but accumulate at horizontal features where the remaining x-gradient they can't efficiently follow holds them. The 2:1 pos_std_y/pos_std_x ratio quantifies the anisotropy. Plateau=0 suggests dynamic oscillation (spots merging/splitting horizontally). New symmetry-breaking mechanism discovered!
Next: parent=89, test alpha=0.25 + 2-type opposing to see if anisotropic stripes survive multi-type (unlike Turing stripes which fail multi-type)

## Iter 90: 8/10 - PDE_D_Anisotropic alpha=0.25 + Brusselator hex, 2-type opposing → ORIENTED ELONGATED CLUSTERS WITH TYPE SEGREGATION
Node: id=90, parent=89
Mode/Strategy: exploit — test anisotropic stripes with 2-type opposing mobility
Config: params=[[-4,4,180,-180,1.6,1.0,1.6,1.5],[4,-4,-180,180,1.6,1.0,1.6,1.5]], params_mesh=[[0.5,15,4.5,6.5,1,0],[1,0,0,0,0,0],[1,180,-180,0.05,0.25,0]], n_frames=6000, delta_t=5e-4, n_particles=9600, n_nodes=10000
n_particle_types: 2
particle_model_name: PDE_ParticleField_D_Anisotropic
mesh_model_name: Diffusiophoresis_Mesh
Metrics: entropy=[0.83], plateau=[0.00], in_box=[100.0]%, clustering=[0.46]
C1_std=0.81, C2_std=0.46, pattern_growth=92.70, pos_std_x=0.155, pos_std_y=0.307
Assessment:
  - Symmetry: stripes (horizontally oriented elongated clusters)
  - Particles: segregated (type 0=orange boundary shells, type 1=blue interior cores)
  - Stability: transient (plateau=0.00, still dynamic)
  - Novelty: variant — anisotropic orientation preserved with 2-type, plus type-boundary segregation
Visual: Brusselator fields show horizontally elongated dark spots/bars in C1, bright elongated features in C2. Particles show clear 2-type phase separation: orange (type 0, M1=-4, attracted to C1 low-concentration zones) forms boundary shells around elongated clusters, while blue (type 1, M1=4, attracted to C1 high-concentration valleys) fills cluster interiors. ~8-10 elongated clusters arranged irregularly with clear horizontal preferred orientation. Resembles biological tissue with cell-type boundary/interior organization. pos_std_y/pos_std_x = 1.98, confirming 2:1 anisotropy ratio preserved from iter 89.
Mutation: n_particle_types: 1 -> 2 (opposing mobility), same alpha=0.25
Observation: **KEY FINDING** — Anisotropic stripes SURVIVE 2-type opposing mobility! This contrasts sharply with Turing stripes (A=3, B=10) which are destroyed by multi-type (principle #11). The anisotropy mechanism is fundamentally different from Turing-boundary stripe selection: it works by preferential trapping rather than bifurcation parameter tuning. The 2-type version adds type-boundary segregation (shell/core morphology) on top of the anisotropic orientation. Entropy increased (0.61→0.83) as 2-type creates richer spatial distribution. Clustering slightly decreased (0.55→0.46) as type segregation spreads particles more evenly around clusters. C1_std dropped (1.43→0.81) — opposing consumption/production partially cancel field gradients. Pattern growth decreased (191→93) for same reason. The pos_std anisotropy ratio is maintained at ~2:1. This establishes that anisotropic mobility is a MORE ROBUST symmetry-breaking mechanism than Turing-boundary stripes.
Next: parent=89, test FHN time_scale=10 with 2-type (explore FHN wavelength control, switch to new axis per block plan)

## Iter 91: 7/10 - FHN time_scale=10 (slow) + 2-type opposing → COARSE HEXAGONAL (wavelength control confirmed)
Node: id=91, parent=root
Mode/Strategy: explore — test FHN time_scale as wavelength selector (analog of Brusselator Da_c)
Config: params=[[-4,4,180,-180,1.6,1.0,1.6,1.5],[4,-4,-180,180,1.6,1.0,1.6,1.5]], params_mesh=[[0.05,0.5,1.0,0.1,0.0,10],[0,0,0,0,0,0],[1,180,-180,0.05,0,0]], n_frames=6000, delta_t=5e-4, n_particles=9600, n_nodes=10000
n_particle_types: 2
particle_model_name: PDE_ParticleField_D
mesh_model_name: PDE_Diffusiophoresis_FHN
Metrics: entropy=[0.76], plateau=[0.00], in_box=[100.0]%, clustering=[0.17]
C1_std=147.56, C2_std=10.88, pattern_growth=2176.77, pos_std_x=0.239, pos_std_y=0.229
Assessment:
  - Symmetry: hexagonal (coarse, ~15-20 spots vs ~30 at time_scale=50)
  - Particles: network (blue chains) + clustered (orange at C1 peaks) — dual organization
  - Stability: transient (plateau=0.00, FHN intrinsically oscillatory)
  - Novelty: variant — coarser FHN hex confirms time_scale wavelength control
Visual: FHN fields show ~15-20 well-separated bright C1 spots with complementary C2 dark spots. Spots are clearly LARGER and FEWER than standard FHN (time_scale=50 gives ~30 finer spots). 2-type particles: orange (type 0, M1=-4) clusters at C1 peaks, blue (type 1, M1=4) forms thin network/chains in interstitial regions between spots. Field amplitudes are much higher than Brusselator (C1_mean=89.7, C1_std=147.6 vs Brusselator C1_std~1-2) — FHN operates in a different amplitude regime. Late frames show continued particle motion (vel=39.3) despite quasi-stable field pattern.
Mutation: mesh_model: Brusselator -> FHN, time_scale: 50 -> 10, particle_model: Anisotropic -> standard PDE_D
Observation: **FHN time_scale IS a wavelength selector** — time_scale=10 (5x slower than default 50) produces ~2x coarser hexagonal spots. Feature wavelength scales with sqrt(time_scale) approximately. However, unlike Brusselator Da_c which transitions between DIFFERENT symmetry classes (network→labyrinthine→hex), FHN time_scale so far only changes spot SIZE while staying hexagonal. Clustering=0.17 is very low (vs Brusselator hex ~0.5, standard FHN hex ~0.3), suggesting the coarse spots are less effective at concentrating particles. This answers the open question about FHN wavelength control: YES, time_scale controls wavelength, but appears to stay within hex symmetry class. Need to test more extreme values (time_scale=3-5) to see if symmetry transition occurs.
Next: parent=89, test Anisotropic + FHN, 1-type (per block plan iter 92)

## Iter 92: 7/10 - Anisotropic + FHN, 1-type → NETWORK WITH PERIPHERAL LOBES (anisotropy washed out)
Node: id=92, parent=89
Mode/Strategy: exploit — test Anisotropic PDE_D with FHN mesh model (does anisotropy break FHN hex like it broke Brusselator hex?)
Config: FHN Du=0.05, a=0.5, b=1.0, eps=0.1, time_scale=50, Dv=0, PDE_ParticleField_D_Anisotropic alpha=0.25, n_particle_types=1, n_frames=6000
n_particle_types: 1
Metrics: entropy=[0.90], plateau=[0.00], in_box=[100.0]%, clustering=[0.25]
Assessment:
  - Symmetry: other (quasi-radial with distorted peripheral lobes)
  - Particles: network (central mass + network tendrils extending to field peaks)
  - Stability: transient (plateau=0.00, actively evolving peripheral structures)
  - Novelty: variant (network-lobe morphology is somewhat different but not a new symmetry class)
Visual: Early frames show clear elliptical particle disc (taller than wide, consistent with alpha=0.25). Mid-simulation (frame 50) — elliptical core with clustering=0.51. Late simulation (frames 80-99) — FHN field develops large-scale peripheral concentration lobes, particles extend network tendrils from central mass toward these lobes. Final pattern is a central rounded mass with 4-6 irregular peripheral extensions — "amoeboid" or "brain coral" morphology. C1 field shows large central teal region (low C1) surrounded by irregular yellow/green/purple peripheral features. pos_std ratio is only 1.05:1 (nearly isotropic!) compared to Brusselator's 2:1. The FHN field dynamics dominate over the anisotropic mobility bias.
Mutation: mesh_model: Brusselator -> FHN (keeping Anisotropic PDE_D alpha=0.25)
Observation: **Anisotropy is mesh-model-selective**: alpha=0.25 creates strong 2:1 orientation ratio with Brusselator (large features susceptible to directional bias) but nearly isotropic 1.05:1 with FHN (many small features average out directional bias). This parallels LogSensing's mesh-model selectivity — particle dynamics modifications interact differently with different field models. FHN's finer spatial scale (~30 small spots vs Brusselator's ~10 large spots) means individual particles experience gradient directions that average out, nullifying the anisotropy at the population level. The resulting morphology is a hybrid: FHN-typical network organization but with larger-scale "lobe" structures at the periphery that may reflect the anisotropy distorting the radial initial condition breakup. Clustering=0.25 is typical FHN (lower than Brusselator). Entropy=0.90 is very high, indicating broad distribution. Score 7/10 — informative negative result (anisotropy doesn't transfer to FHN), plus moderate network-lobe morphology.
Next: parent=87, Da_c=7 fine-mapping of labyrinthine regime (per block plan iter 93)
