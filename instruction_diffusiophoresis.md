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
  n_particle_types: 1 # int  1, 2, or 3
```

**IMPORTANT: n_nodes must be a perfect square** (the mesh is n×n grid).
Use only these values: `10000` (100×100), `22500` (150×150), `40000` (200×200), `62500` (250×250).
Do NOT use values like 25000, 30000, etc. - simulation will crash.

**To enable multi-type:**

1. Set `n_particle_types: N` in config
2. **Keep total particle count constant**: When changing `n_particle_types`, particles are distributed equally among types. The total `n_particles` should remain ~9600 to maintain simulation density. Example: 1 type = 9600 particles, 2 types = 9600 total (4800 each), 3 types = 9600 total (3200 each).

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
## Regime Comparison

| Regime | mesh_model            | particle_model      | n_types | n_particles | Best R² | Key Insight |
| ------ | --------------------- | ------------------- | ------- | ----------- | ------- | ----------- |
| Base   | Diffusiophoresis_Mesh | PDE_ParticleField_D | 3       | 2000        | -       | baseline    |

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
