# Diffusiophoresis Pattern Exploration

**Reference**: See for current context understanding: https://www.sciencedirect.com/science/article/abs/pii/S2590238525005569

**Goal**: Explore the simulation/code space to discover parameter and code configurations that produce **biologically interesting patterns** in diffusiophoresis simulations. In this exploration be stimulated by discovering very complex even UNKOWN and UNSEEN patterns. Next try to understand the conditions in the simulated dynamics that lead to different topology.

**Current status**: To date, we only observe **dot patterns**. A first milestone would be to achieve **stripe patterns** as a more complex Turing instability mode.

## What is "Biologically Interesting"?

The diffusiophoresis simulation models:

- **Brusselator reaction-diffusion** on a mesh (produces Turing patterns)
- **Particles** that respond to concentration gradients (diffusiophoresis)
- **Particle-field coupling** where particles affect and are affected by the chemical fields

Interesting patterns include:

- Turing patterns (spots, stripes, labyrinthine)
- Traveling waves, spirals
- Particle clustering and self-organization
- Dynamic pattern formation over time
- Multi-scale structures

Boring patterns:

- Uniform/homogeneous fields
- Static, unchanging states
- Chaotic noise without structure
- Immediate collapse to equilibrium

---

## State of the Art Reasoning

**Ground your reasoning in scientific literature.** Include a brief "Literature:" line in log entries.

### Example Log Entry

```
Literature: Gray-Scott mitosis regime (Pearson 1993)
Hypothesis: Increase B toward stripe-spot transition
```

---

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default: 8) exploring one configuration space.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

### Code Modification Rules

| When                                  | Allowed Changes                                                     |
| ------------------------------------- | ------------------------------------------------------------------- |
| Within block (iterations 1-8)         | Config parameters ONLY                                              |
| At block boundary (>>> BLOCK END <<<) | Config parameters OR code modifications OR **PDE variant creation** |

**IMPORTANT**: Code modifications and PDE variant creation are ONLY allowed at the end of a block when you see `>>> BLOCK END <<<` in the prompt. During regular iterations within a block, you can only modify config parameters.

**PDE Variant Creation**: At block boundaries, you can create:

- **Field-field variants** (e.g., `PDE_Diffusiophoresis_GrayScott.py`) - See [Step 5.3](#step-53-create-pde-variant-block-end-only)
- **Particle dynamics variants** (e.g., `PDE_D_Boids.py`) - See [Step 5.4](#step-54-create-pde_d-variant-block-end-only)

---

## File Structure

### 1. Full Log (append-only record)

**File**: `diffusiophoresis_Claude_analysis.md`

- Append every iteration's full log entry
- Append block summaries
- **Never read this file** - it's for human record only

### 2. Working Memory

**File**: `diffusiophoresis_Claude_memory.md`

- **READ at start of each iteration**
- **UPDATE at end of each iteration**
- Contains: established principles + previous blocks summary + current block iterations
- Fixed size (~500 lines max)

---

## Iteration Workflow (Steps 1-5, every iteration)

### Step 1: Read Working Memory

Read `diffusiophoresis_Claude_memory.md` to recall:

- Established principles about what produces interesting patterns
- Previous block findings
- Current block progress

### Step 2: Analyze Current Results

**Visual Analysis (CRITICAL):**

Examine the montage image showing 10 frames from the simulation.

- **Montage location**: Provided in the prompt as `montage_path` (e.g., `graphs_data/{dataset_name}/montage_iter_XXX.png`)
- **Individual frames**: `graphs_data/{dataset_name}/Fig/Fig_0_XXXXXX.png`
- **Exploration archive**: `log/Claude_exploration/instruction_diffusiophoresis/activity/` contains montages from all iterations

The montage shows 10 evenly-spaced frames (2 rows × 5 columns) from early to late simulation time.

**2x2 Figure Layout (each frame):**

- **Top row**: Field concentrations C1 (left) and C2 (right) - Turing patterns
- **Bottom left**: Particle spatial organization - **PRIMARY FOCUS**
- **Bottom right**: Velocity arrows - not important, can ignore

**What to look for (in priority order):**

1. **Particle spatial organization (bottom left)** - PRIMARY
   - Clustering, aggregation, self-organization
   - Correlation with field patterns
   - Dynamic reorganization over time
   - Multi-scale structures

2. **Field patterns C1/C2 (top row)** - SECONDARY
   - Turing patterns (spots, stripes, labyrinthine)
   - Pattern wavelength and regularity
   - Traveling waves, spirals
   - Dynamic vs static behavior

**Score the pattern 0-10:**

| Score | Description                                                                              |
| ----- | ---------------------------------------------------------------------------------------- |
| 0-1   | Uniform/collapsed - no patterns, boring                                                  |
| 2-3   | Minimal structure - weak gradients, little organization                                  |
| 4-5   | Basic patterns - simple spots or stripes, predictable                                    |
| 6-7   | Complex patterns - multiple scales, dynamic behavior                                     |
| 8-9   | Rich dynamics - spirals, traveling waves, emergent clustering                            |
| 10    | Exceptional - novel self-organization, multi-scale structure, scientifically interesting |

**Metrics from `analysis.log`:**

- Frame count, simulation parameters
- Any computed metrics (field gradients, particle distributions)

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file with running notes

**Log Format:**

```
## Iter N: [score]/10
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary/code-modification/multi-type]
Config: params_mesh=[...], n_frames=X, delta_t=Y, ...
n_particle_types: [1/2/3]
Score: [N]/10
Visual: [description of patterns observed]
Mutation: [param or code]: [old] -> [new]
Parent rule: [one line]
Observation: [what did this change reveal?]
Next: parent=P
```

### Step 4: Parent Selection (UCB)

Read `ucb_scores.txt`:

- If empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**Strategies:**

| Condition                           | Strategy            | Action                                     |
| ----------------------------------- | ------------------- | ------------------------------------------ |
| Default                             | **exploit**         | Highest UCB node, try mutation             |
| 3+ consecutive score >= 7           | **failure-probe**   | Extreme parameter to find boundary         |
| 4+ consecutive improving            | **explore**         | Branch to different parameter dimension    |
| Low scores across block             | **code-change**     | Consider modifying PDE equations (iter 5+) |
| Score = 10 found                    | **robustness-test** | Re-run same config to verify               |
| n_particle_types=1 over-represented | **multi-type**      | Switch to n_particle_types=2 or 3          |

**IMPORTANT - Particle Type Diversity:**

Maintain roughly equal exploration of different particle type counts:

- ~33% of iterations should use `n_particle_types: 1`
- ~33% of iterations should use `n_particle_types: 2`
- ~33% of iterations should use `n_particle_types: 3`

Multi-type configurations enable richer dynamics:

- Different types can have opposing mobilities (one attracted, one repelled by gradients)
- Cross-type attraction/repulsion creates phase separation or mixing
- Multiple types can create predator-prey or symbiotic dynamics

### Step 5: Edit Config or Code

#### Step 5.1: Edit Config (default)

**Simulation Parameters (can change within block):**

```yaml
simulation:
  params_mesh:
    - [D1, Da_c, A, B, mu, ...] # Brusselator: diffusion, Damköhler, A, B
    - [D2, M2, ...] # C2 field parameters
    - [Pe, consumption, production, influence_radius, ...] # Particle-field coupling

  n_frames: 4000 # simulation length (1000-10000)
  delta_t: 5.0E-4 # time step (1E-5 to 1E-3)
  n_particles: 9600 # particle count
  n_nodes: 10000 # mesh resolution - MUST BE PERFECT SQUARE
```

**IMPORTANT: n_nodes must be a perfect square** (the mesh is n×n grid).
Use only these values: `10000` (100×100), `22500` (150×150), `40000` (200×200), `62500` (250×250).
Do NOT use values like 25000, 30000, etc. - simulation will crash.

**Key Brusselator parameters (row 0 of params_mesh):**

- `D1`: Diffusion coefficient for C1 (0.01-1.0)
- `Da_c`: Damköhler number - reaction rate (1-100)
- `A`, `B`: Brusselator parameters - control pattern type
  - A affects equilibrium concentration
  - B > 1 + A² triggers Turing instability

**Particle-field coupling (row 2 of params_mesh):**

- `Pe`: Péclet number - advection vs diffusion (0.01-10)
- `consumption_rate`: how particles consume field
- `production_rate`: how particles produce field

#### Step 5.2: Modify Code (BLOCK END only)

**Files you can modify:**

| File                                                   | What to change                                                                                                       |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `src/ParticleGraph/generators/PDE_Diffusiophoresis.py` | Brusselator reaction equations (R1, R2), diffusion terms, damping, pattern formation dynamics                        |
| `src/ParticleGraph/generators/PDE_D.py`                | Diffusiophoretic velocities (M1, M2 mobility), particle-particle repulsion, field gradients, particle→field feedback |
| `src/ParticleGraph/generators/graph_data_generator.py` | `data_generate_particle_field()`: simulation loop, time stepping, boundary conditions, initialization                |

**Example code modifications:**

1. **Change reaction kinetics** (PDE_Diffusiophoresis.py):

```python
# Original Brusselator:
R1 = self.Da_c * (self.A - (self.B+1)*C1 + C1*C1*C2)

# Try Gray-Scott instead:
R1 = -C1*C2*C2 + self.A*(1-C1)
```

2. **Change particle-field coupling** (PDE_D.py):

```python
# Original: linear mobility
velocities = (self.M1 * grad_C1 + self.M2 * grad_C2) * dir_norm

# Try nonlinear response:
velocities = torch.tanh(self.M1 * grad_C1) * dir_norm
```

3. **Add damping or noise** (PDE_Diffusiophoresis.py):

```python
# Add stochastic term:
dC1 = diff_C1 + R1 + 0.01 * torch.randn_like(C1)
```

**Reference models for particle motion:**

The ParticleGraph repo contains other motion models that can inspire code modifications:

- `src/ParticleGraph/generators/PDE_A.py` - Arbitrary attraction/repulsion between particles (distance-dependent forces)
- `src/ParticleGraph/generators/PDE_B.py` - Boids model with different particle types (alignment, cohesion, separation)

These can be used as reference for adding particle-particle interactions beyond simple repulsion.

### Multi-Type Particle Support in PDE_D

`PDE_D.py` now supports multiple particle types with per-type parameters for diffusiophoresis and PDE_A-style attraction-repulsion.

**Per-type params layout (8 parameters per type):**

```yaml
simulation:
  # [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
  params:
    - [-16, 16, 180, -180, 1.6, 1.0, 1.6, 1.5] # Type 0
    - [-8, 8, 90, -90, 1.8, 1.8, 1.1, 1.9] # Type 1
    - [-4, 4, 45, -45, 1.7, 1.8, 1.1, 1.9] # Type 2
  n_particle_types: 3
  sigma: 0.005 # Used for attraction-repulsion kernel
```

**Parameter meanings:**
| Parameter | Description |
|-----------|-------------|
| M1 | Mobility coefficient for C1 gradient (diffusiophoresis) |
| M2 | Mobility coefficient for C2 gradient (diffusiophoresis) |
| consumption | Rate at which particles consume C1 field |
| production | Rate at which particles produce C2 field |
| ar_p1 | Attraction strength (PDE_A formula) |
| ar_p2 | Attraction exponent (PDE_A formula) |
| ar_p3 | Repulsion strength (PDE_A formula) |
| ar_p4 | Repulsion exponent (PDE_A formula) |

**Attraction-repulsion formula (from PDE_A):**

```
f = ar_p1 * exp(-d^(2*ar_p2) / (2σ²)) - ar_p3 * exp(-d^(2*ar_p4) / (2σ²))
```

- First term: attraction (positive ar_p1 pulls particles together)
- Second term: repulsion (ar_p3 pushes particles apart at short range)
- Different types can have different interaction strengths

**To enable multi-type:**

1. Set `n_particle_types: N` in config
2. Check N entries to `params:` list (one per type)
3. Set `sigma:` for the interaction kernel width
4. **Keep total particle count constant**: When changing `n_particle_types`, particles are distributed equally among types. The total `n_particles` should remain ~9600 to maintain simulation density. Example: 1 type = 9600 particles, 2 types = 9600 total (4800 each), 3 types = 9600 total (3200 each).

**⚠️ DIVERSITY REQUIREMENT:**

**You MUST test n_particle_types=2 and n_particle_types=3 as frequently as n_particle_types=1.**

Track your particle type distribution and actively correct imbalances. If you notice most recent iterations used 1 type, your NEXT iteration should use 2 or 3 types.

**Quick-start templates for multi-type configs:**

**2 particle types (opposing responses):**

```yaml
simulation:
  params:
    - [-16, 16, 180, -180, 1.6, 1.0, 1.6, 1.5] # Type 0: attracted to C1 peaks
    - [16, -16, -180, 180, 1.8, 1.0, 1.1, 1.9] # Type 1: repelled from C1 peaks
  n_particle_types: 2
  n_particles: 9600 # 4800 each type
  sigma: 0.005
```

**3 particle types (complex ecosystem):**

```yaml
simulation:
  params:
    - [-16, 16, 180, -180, 1.6, 1.0, 1.6, 1.5] # Type 0: consumer
    - [8, -8, -90, 90, 1.8, 1.8, 1.1, 1.9] # Type 1: producer
    - [0, 0, 0, 0, 2.0, 1.0, 2.0, 1.0] # Type 2: neutral/interactor
  n_particle_types: 3
  n_particles: 9600 # 3200 each type
  sigma: 0.005
```

**Safety rules:**

- Make ONE change at a time
- Document hypothesis for the change
- Compare directly to parent (same config, code-only diff)
- Never modify GNN_LLM.py

#### Step 5.3: Create PDE Variant (BLOCK END only)

**Before creating a variant:** Check the PDE Variants table in Working Memory and existing files in `src/ParticleGraph/generators/` to avoid duplicating work. Only create a new variant if the desired physics isn't already implemented.

**When to create a variant:** Create a new PDE file to test fundamentally different reaction-diffusion models while preserving the base Brusselator.

**⚠️ REQUIREMENTS:**

1. **ONLY at block boundaries** - Never during regular iterations
2. **MUST cite scientific literature** - Every variant must reference source model
3. **MUST include `PARAMS_DOC`** - Self-documenting parameter structure
4. **MUST add compatibility attributes** - Add `self.A` and `self.B` in `__init__` (required by base class)

**Naming convention:**

| File Name                           | Config `mesh_model_name`          |
| ----------------------------------- | --------------------------------- |
| `PDE_Diffusiophoresis_GrayScott.py` | `Diffusiophoresis_Mesh_GrayScott` |
| `PDE_Diffusiophoresis_FHN.py`       | `Diffusiophoresis_Mesh_FHN`       |

**Creating a variant (5 steps):**

1. **Copy base file** and rename class to match filename
2. **Add docstring with literature citation** (author, year, journal)
3. **Add PARAMS_DOC** with model equations and parameter descriptions
4. **Add compatibility attributes in `__init__`:**
   ```python
   # Required for compatibility with base class expectations
   self.A = torch.tensor(1.0, device=p.device)  # Initial U value
   self.B = torch.tensor(0.0, device=p.device)  # Initial V value
   ```
5. **Implement reaction equations** in `forward()`, update config

**Common errors and fixes:**

| Error                              | Cause                 | Fix                                      |
| ---------------------------------- | --------------------- | ---------------------------------------- |
| `NameError: mesh_model_name`       | Using bare variable   | Use `config.graph_model.mesh_model_name` |
| `KeyError` in PyG                  | Class not registered  | Ensure class name matches file suffix    |
| `AttributeError: no attribute 'A'` | Missing compatibility | Add `self.A`, `self.B` in `__init__`     |

**Established models:**

| Model               | Key Params | Literature                 |
| ------------------- | ---------- | -------------------------- |
| **Gray-Scott**      | F, k       | Pearson (1993) Science 261 |
| **FitzHugh-Nagumo** | a, b, ε    | FitzHugh (1961) Biophys J  |
| **Schnakenberg**    | a, b, γ    | Schnakenberg (1979) JTB    |

**Log format:**

```
### Variant: PDE_Diffusiophoresis_GrayScott
Literature: Pearson (1993) Science 261:189-192
Rationale: [why this model]
Config: mesh_model_name: Diffusiophoresis_Mesh_GrayScott
```

**Note:** New variants are auto-committed by GNN_LLM after creation.

#### Step 5.4: Create PDE_D Variant (BLOCK END only)

**Before creating a variant:** Check the PDE Variants table in Working Memory and existing `PDE_D_*.py` files in `src/ParticleGraph/generators/` to avoid duplicating work. Only create a new variant if the desired particle dynamics isn't already implemented.

**When to create a variant:** Create a new PDE_D file to test fundamentally different particle dynamics (diffusiophoresis, boids, chemotaxis, etc.) while preserving the base PDE_D.

**⚠️ REQUIREMENTS:**

1. **ONLY at block boundaries** - Never during regular iterations
2. **MUST cite scientific literature** - Every variant must reference source model
3. **MUST include `PARAMS_DOC`** - Self-documenting parameter structure for `params` slots
4. **MUST maintain same interface** - Same `__init__` signature and `forward(data, direction)` modes

**Naming convention:**

| File Name             | Config `particle_model_name`     |
| --------------------- | -------------------------------- |
| `PDE_D_Boids.py`      | `PDE_ParticleField_D_Boids`      |
| `PDE_D_Chemotaxis.py` | `PDE_ParticleField_D_Chemotaxis` |

**Creating a variant (5 steps):**

1. **Copy base PDE_D.py** and rename class to match filename (e.g., `PDE_D_Boids`)
2. **Add docstring with literature citation** (author, year, journal)
3. **Add PARAMS_DOC** documenting how `params` slots are interpreted:
   ```python
   PARAMS_DOC = {
       "model_name": "Boids",
       "literature": "Reynolds (1987) SIGGRAPH 'Flocks, Herds, and Schools'",
       "params": [
           {"index": 0, "name": "alignment", "description": "Alignment strength", "typical_range": [0.1, 2.0]},
           {"index": 1, "name": "cohesion", "description": "Cohesion strength", "typical_range": [0.1, 2.0]},
           # ... etc
       ]
   }
   ```
4. **Implement particle dynamics** in `message()` for modes: `'fp'`, `'pf'`, `'pp'`
5. **Update config** with new `particle_model_name` and appropriate `params` values

**Parameter reinterpretation:**

Each PDE_D variant reinterprets the same `params` tensor slots according to its own physics:

| Variant          | params[type][0:4] interpretation               |
| ---------------- | ---------------------------------------------- |
| PDE_D (base)     | M1, M2, consumption, production                |
| PDE_D_Boids      | alignment, cohesion, separation, vision_radius |
| PDE_D_Chemotaxis | sensitivity, adaptation_rate, threshold, ...   |

**Established models for particle dynamics:**

| Model             | Key Behavior                               | Literature               |
| ----------------- | ------------------------------------------ | ------------------------ |
| **Boids**         | Flocking (alignment, cohesion, separation) | Reynolds (1987) SIGGRAPH |
| **Chemotaxis**    | Gradient sensing with adaptation           | Keller-Segel (1971) JTB  |
| **Active Matter** | Self-propelled particles                   | Vicsek (1995) PRL        |

**Log format:**

```
### Variant: PDE_D_Boids
Literature: Reynolds (1987) SIGGRAPH 'Flocks, Herds, and Schools'
Rationale: [why this model for particle dynamics]
Config: particle_model_name: PDE_ParticleField_D_Boids
```

**Note:** New PDE_D variants are auto-committed by GNN_LLM after creation.

---

## Block Workflow (End of Block)

### Step 1: Edit Instructions

Add/modify rules based on block experience:

- If branching rate < 20% → add exploration rule
- If stuck at low scores → add code-change trigger

### Step 2: Update Memory

- Summarize block findings (2-3 lines)
- Update Knowledge Base with confirmed principles
- Clear "Iterations This Block" section

---

## Working Memory Structure

```markdown
## Regime Comparison

| Regime | mesh_model | particle_model | n_types | n_particles | Best R² | Key Insight |
| ------ | ---------- | -------------- | ------- | ----------- | ------- | ----------- |
| Base   | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3 | 2000 | - | baseline |

## Insights

| Category    | Finding                                              |
| ----------- | ---------------------------------------------------- |
| Patterns    | [key pattern observations]                           |
| Performance | [what configs work well]                             |
| Failures    | [what to avoid]                                      |

---

## Knowledge Base

### Established Principles

[Confirmed findings across 3+ iterations - see Knowledge Base Guidelines]

### Open Questions

[Tentative patterns needing more testing, contradictions to resolve]

### Failed Configurations

[What to avoid]

### Code Insights

[What code changes helped/hurt]

### PDE Variants

| Variant                         | Model       | Literature       | Status  | Best Score |
| ------------------------------- | ----------- | ---------------- | ------- | ---------- |
| Diffusiophoresis_Mesh           | Brusselator | Prigogine (1968) | active  | 5/10       |
| Diffusiophoresis_Mesh_GrayScott | Gray-Scott  | Pearson (1993)   | testing | -          |

### Particle Type Distribution (TRACK THIS!)

| n_particle_types | Count | Target |
| ---------------- | ----- | ------ |
| 1                | X     | ~33%   |
| 2                | Y     | ~33%   |
| 3                | Z     | ~33%   |

**Action needed if imbalanced:** If one type is under-represented, use it in next iteration!

---

## Previous Block Summary

[Short summary of last block]

---

## Current Block

### Block Info

Parameters: ...
mesh_model_name: [current variant]
Iterations: M to M+8

### Hypothesis

[What are we exploring this block?]

### Iterations This Block

[Current block iterations only]

### Emerging Observations

[What's working/failing]
**CRITICAL: This section must ALWAYS be at the END of memory file. When adding new iterations, insert them BEFORE this section.**
```

---

## Knowledge Base Guidelines

### What to Add to Established Principles

Examples:

- ✓ "High diffusion_u with low production_v creates traveling waves" (causal, generalizable)
- ✓ "n_particle_types=3 produces richer clustering than n_types=1" (experimental finding)
- ✓ "consumption > 0.5 destabilizes patterns" (boundary condition)
- ✗ "params_mesh=[1.0, 0.5, 0.1] worked in Block 4" (too specific)
- ✗ "Block 3 got score 8" (not a principle)

### Evidence Hierarchy

| Level            | Criterion                              | Action                 |
| ---------------- | -------------------------------------- | ---------------------- |
| **Established**  | Consistent across 3+ iterations/blocks | Add to Principles      |
| **Tentative**    | Observed 1-2 times                     | Add to Open Questions  |
| **Contradicted** | Conflicting evidence                   | Note in Open Questions |

### What to Add to Open Questions

- Patterns needing more testing
- Contradictions between blocks
- Theoretical predictions not yet verified

---

## Background: Diffusiophoresis Physics

### Brusselator Model (PDE_Diffusiophoresis.py)

```
dC1/dt = D1 * ∇²C1 + Da_c * (A - (B+1)*C1 + C1²*C2)
dC2/dt = D2 * ∇²C2 + Da_c * (B*C1 - C1²*C2)
```

- **Turing instability**: B > 1 + A² produces patterns
- **Pattern wavelength**: depends on D1/D2 ratio
- Higher Da_c → faster dynamics

### Diffusiophoresis (PDE_D.py)

Particles move in response to concentration gradients:

```
v_particle = M1 * ∇C1 + M2 * ∇C2
```

- Positive mobility → move up gradient (toward higher concentration)
- Negative mobility → move down gradient (away from concentration)

### Coupling

- Particles can consume/produce chemicals
- This creates feedback: particles affect fields, fields move particles
- Feedback can stabilize or destabilize patterns
