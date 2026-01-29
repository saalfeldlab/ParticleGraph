# Diffusiophoresis Pattern Exploration

**Reference**: See for current context understanding: https://www.sciencedirect.com/science/article/abs/pii/S2590238525005569

**Goal**: Explore the simulation/code space to discover parameter and code configurations that produce **biologically interesting patterns** in particle-field simulations.
Understanding morphogenesis requires more than generating patterns; it requires understanding how interacting processes converge toward stable, functional forms. This work presents a **closed-loop experimental framework** in which experiments, reasoning, and long-term memory are tightly coupled, with a large language model (LLM) operating as an **active scientific agent**. Note, the LLM does not only explore parameter space; it is allowed, at controlled points, to **modify and replace the governing partial differential equations (PDEs)** that define the system dynamics. In this framework, PDEs are treated as _hypotheses_, not fixed truths.
Rather than treating particles or fields as primary objects, the framework treats **interactions as fundamental**, with structure arising from their mutual constraint. The LLM evaluates simulation outcomes, formulates mechanistic hypotheses, and directs subsequent interventions through structured exploration and persistent memory, enabling cumulative understanding across regimes rather than isolated optimization.

Four Coupled Interactions, four PDEs:

Field–Field Reaction-diffusion PDE Turing patterns via activator-inhibitor dynamics \
Field–Particle Diffusiophoresis Field gradients drive particle motion \
Particle–Field Consumption/production Particles locally modify concentrations \
Particle–Particle Attraction-repulsion Short-range forces between particles

---

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default: 8) exploring one configuration space.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

## File Structure (CRITICAL)

You maintain TWO files:

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

#### 2.1 **Visual Analysis (CRITICAL):**

Examine the **Individual frames**: `graphs_data/{dataset_name}/Fig/Fig_0_XXXXXX.png`

**2x2 Figure Layout (each frame):**

- **Top row**: Field concentrations C1 (left) and C2 (right)
- **Bottom left**: Particle spatial organization

**What to look for (in priority order):**

1. **Particle patterns (bottom left)** - PRIMARY
2. **Field patterns C1/C2 (top row)** - SECONDARY

**Assess the pattern using these dimensions:**

| Dimension | Categories |
|-----------|------------|
| **Symmetry** | `none` \| `radial` \| `hexagonal` \| `stripes` \| `other` |
| **Particle organization** | `collapsed` \| `uniform` \| `clustered` \| `network` \| `segregated` |
| **Stability** | `unstable` (NaN/escape) \| `transient` \| `stable` |
| **Novelty** | `repeat` \| `variant` \| `novel` |

#### 2.2 **Metrics from `analysis.log`:**

**Primary metrics (use these for scoring):**

- `spatial_entropy`: **COMPLEXITY** - normalized entropy of particle spatial distribution (0-1). Values ~0.3-0.7 indicate structured patterns. Near 1 = uniform/boring, near 0 = collapsed to single point.
- `plateau`: **STABILITY** - convergence score (0-1). Compares mean particle velocity in final 20% vs early 20% of simulation. Near 1 = velocity dropped (steady state), near 0 = still moving fast.
- `particles_in_box_pct`: **CRITICAL** - percentage of particles remaining in [0,1] box. Low values indicate particle escape/instability.

**Secondary metrics:**

- `clustering`: particle clustering metric (0.289 - pos_std) / 0.289. Values near 0 = uniform, higher = clustered.
- `pattern_growth`: change in field std over time. Positive = patterns developing.
- `C1_std`, `C2_std`: field concentration variation. Higher = stronger Turing patterns.

**Metric interpretation for assessment:**

| Metric | Stability=stable | Particles=clustered | Particles=network |
|--------|------------------|---------------------|-------------------|
| plateau | > 0.5 | any | any |
| clustering | any | > 0.4 | -0.2 to 0.3 |
| particles_in_box | > 90% | > 90% | > 90% |
| spatial_entropy | 0.3-0.8 | 0.5-0.8 | 0.7-0.9 |

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file with running notes

**Log Format {config}\_analysis.md:**
This format is compulsory

```
## Iter N
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary/code-modification/multi-type]
Config: params_mesh=[...], n_frames=X, delta_t=Y, ...
n_particle_types: [1/2/3]
Metrics: entropy=[X.XX], plateau=[X.XX], in_box=[XX.X]%, clustering=[X.XX]
Assessment:
  - Symmetry: [none/radial/hexagonal/stripes/other]
  - Particles: [collapsed/uniform/clustered/network/segregated]
  - Stability: [unstable/transient/stable]
  - Novelty: [repeat/variant/novel]
Visual: [description of patterns observed]
Mutation: [param or code]: [old] -> [new]
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
| 3+ consecutive `stable` + `hexagonal` | **boundary-probe** | Extreme parameter to find limits           |
| 3+ consecutive same symmetry        | **explore**         | Branch to different parameter dimension    |
| 3+ consecutive `unstable`           | **code-change**     | Consider modifying PDE equations (iter 5+) |
| `novel` pattern found               | **robustness-test** | Re-run same config to verify               |
| n_particle_types=1 over-represented | **multi-type**      | Switch to n_particle_types=2 or 3          |
| n_particle_types uniform over-represented | **multi-type** | Switch to n_particle_types=2 or 3          |

**Note:** `shuffle_particle_types` is not working — all runs show concentric radial type bands at initialization regardless of this setting. Always keep `shuffle_particle_types: false`.

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

**⚠️ Code modifications (Steps 5.2 and 5.3) are ONLY available when the prompt explicitly lists "Code files you can modify".**
If no code file paths are provided in the prompt, you MUST only modify config parameters (Step 5.1). Do NOT attempt to create, edit, or write any `.py` files.

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
  n_particle_types: 1 # int  1, 2, or 3
  shuffle_particle_types: false # not working — always keep false
```

**shuffle_particle_types**: Not working. All runs produce concentric radial type bands regardless of this setting. Always keep `false`.

**IMPORTANT: n_nodes must be a perfect square** (the mesh is n×n grid).
Use only these values: `10000` (100×100), `22500` (150×150), `40000` (200×200), `62500` (250×250).
Do NOT use values like 25000, 30000, etc. - simulation will crash.

**To enable multi-type:**

1. Set `n_particle_types: N` in config
2. **`simulation.params` MUST have exactly N rows** (one per particle type). Mismatched row count causes a runtime crash in PDE_D.
3. **Keep total particle count constant**: When changing `n_particle_types`, particles are distributed equally among types. The total `n_particles` should remain ~9600 to maintain simulation density. Example: 1 type = 9600 particles, 2 types = 9600 total (4800 each), 3 types = 9600 total (3200 each).

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

- Make ONE change at a time to get causal understanding right

#### Step 5.2: Modify Code, create PDE_Diffusiophoresis.py variant

This code implement the **mesh_model** that governs the field-field interaction
**IMPORTANT**: PDE variant creation are ONLY allowed at the end of a block when you see `>>> BLOCK END <<<` in the prompt. During regular iterations within a block, you can only modify config parameters.
**Before creating a variant:** Check the PDE Variants table in Working Memory and existing files in `src/ParticleGraph/generators/` to avoid duplicating work. Only create a new variant if the desired physics isn't already implemented.

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

**⚠️ CRITICAL: Update `utils.py` init_mesh function**

When creating a new mesh model variant, you MUST also update `src/ParticleGraph/generators/utils.py` to add a case for your new `mesh_model_name` in the `init_mesh()` function's match statement (~line 980).

Example - if creating `PDE_Diffusiophoresis_GrayScott.py` with `mesh_model_name: PDE_Diffusiophoresis_GrayScott`:
```python
# In utils.py init_mesh() function, add a new case:
case 'PDE_Diffusiophoresis_GrayScott':
    # Gray-Scott initialization
    node_value = torch.zeros((n_nodes, 2), device=device)
    node_value[:, 0] = 1.0  # U = 1
    node_value[:, 1] = 0.0  # V = 0
    n_seeds = max(1, n_nodes // 100)
    seed_indices = torch.randperm(n_nodes)[:n_seeds]
    node_value[seed_indices, 0] = 0.5
    node_value[seed_indices, 1] = 0.25
```

**Common errors and fixes:**

| Error                                          | Cause                              | Fix                                            |
| ---------------------------------------------- | ---------------------------------- | ---------------------------------------------- |
| `UnboundLocalError: 'node_value'`              | Missing case in `init_mesh()`      | Add case for new mesh_model_name in `utils.py` |
| `NameError: mesh_model_name`                   | Using bare variable                | Use `config.graph_model.mesh_model_name`       |
| `KeyError` in PyG                              | Class not registered               | Ensure class name matches file suffix          |
| `AttributeError: no attribute 'A'`             | Missing compatibility              | Add `self.A`, `self.B` in `__init__`           |

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

#### Step 5.3: Modify code, create PDE_D.py Variant

This code implement the **particle_model** that governs altogether the field-particle, particle-particle and the particle-field interactions
**IMPORTANT**: PDE variant creation are ONLY allowed at the end of a block when you see `>>> BLOCK END <<<` in the prompt. During regular iterations within a block, you can only modify config parameters.
**Before creating a variant:** Check the PDE Variants table in Working Memory and existing files in `src/ParticleGraph/generators/` to avoid duplicating work. Only create a new variant if the desired physics isn't already implemented.

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
## Per-block Regime Comparison

| Regime | mesh_model            | particle_model      | n_types | Symmetry | Particles | Key Insight |
| ------ | --------------------- | ------------------- | ------- | -------- | --------- | ----------- |
| Base   | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3       | radial   | clustered | baseline    |

## Insights

| Category    | Finding                    |
| ----------- | -------------------------- |
| Patterns    | [key pattern observations] |
| Performance | [what configs work well]   |
| Failures    | [what to avoid]            |

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

| Variant                         | Model       | Literature       | Status  | Best Symmetry | Best Particles |
| ------------------------------- | ----------- | ---------------- | ------- | ------------- | -------------- |
| Diffusiophoresis_Mesh           | Brusselator | Prigogine (1968) | active  | hexagonal     | clustered      |
| Diffusiophoresis_Mesh_GrayScott | Gray-Scott  | Pearson (1993)   | tested  | radial        | network        |
| PDE_Diffusiophoresis_FHN        | FHN         | FitzHugh (1961)  | active  | hexagonal     | network        |


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
- ✓ "Brusselator achieves hexagonal symmetry, Gray-Scott stays radial" (model comparison)
- ✗ "params_mesh=[1.0, 0.5, 0.1] worked in Block 4" (too specific)
- ✗ "Iter 12 was good" (not a principle)

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
