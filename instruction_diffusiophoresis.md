# Diffusiophoresis Pattern Exploration

**Reference**: See for current context understanding: https://www.sciencedirect.com/science/article/abs/pii/S2590238525005569

**Goal**: Explore the simulation/code space to discover parameter and code configurations that produce **biologically interesting patterns** in diffusiophoresis simulations.

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

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default: 8) exploring one configuration space.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

### Code Modification Rules

| When | Allowed Changes |
|------|-----------------|
| Within block (iterations 1-8) | Config parameters ONLY |
| At block boundary (>>> BLOCK END <<<) | Config parameters OR code modifications |

**IMPORTANT**: Code modifications are ONLY allowed at the end of a block when you see `>>> BLOCK END <<<` in the prompt. During regular iterations within a block, you can only modify config parameters.

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

| Score | Description |
|-------|-------------|
| 0-1   | Uniform/collapsed - no patterns, boring |
| 2-3   | Minimal structure - weak gradients, little organization |
| 4-5   | Basic patterns - simple spots or stripes, predictable |
| 6-7   | Complex patterns - multiple scales, dynamic behavior |
| 8-9   | Rich dynamics - spirals, traveling waves, emergent clustering |
| 10    | Exceptional - novel self-organization, multi-scale structure, scientifically interesting |

**Metrics from `analysis.log`:**
- Frame count, simulation parameters
- Any computed metrics (field gradients, particle distributions)

### Step 3: Write Outputs

Append to Full Log and Working Memory:

**Log Format:**
```
## Iter N: [score]/10
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary/code-modification]
Config: params_mesh=[...], n_frames=X, delta_t=Y, ...
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

| Condition | Strategy | Action |
|-----------|----------|--------|
| Default | **exploit** | Highest UCB node, try mutation |
| 3+ consecutive score >= 7 | **failure-probe** | Extreme parameter to find boundary |
| 4+ consecutive improving | **explore** | Branch to different parameter dimension |
| Low scores across block | **code-change** | Consider modifying PDE equations (iter 5+) |
| Score = 10 found | **robustness-test** | Re-run same config to verify |

### Step 5: Edit Config or Code

#### Step 5.1: Edit Config (default)

**Simulation Parameters (can change within block):**

```yaml
simulation:
  params_mesh:
    - [D1, Da_c, A, B, mu, ...]    # Brusselator: diffusion, Damköhler, A, B
    - [D2, M2, ...]                 # C2 field parameters
    - [Pe, consumption, production, influence_radius, ...] # Particle-field coupling

  n_frames: 4000          # simulation length (1000-10000)
  delta_t: 5.0E-4         # time step (1E-5 to 1E-3)
  n_particles: 9600       # particle count
  n_nodes: 10000          # mesh resolution - MUST BE PERFECT SQUARE
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

| File | What to change |
|------|----------------|
| `src/ParticleGraph/generators/PDE_Diffusiophoresis.py` | Brusselator reaction equations (R1, R2), diffusion terms, damping, pattern formation dynamics |
| `src/ParticleGraph/generators/PDE_D.py` | Diffusiophoretic velocities (M1, M2 mobility), particle-particle repulsion, field gradients, particle→field feedback |
| `src/ParticleGraph/generators/graph_data_generator.py` | `data_generate_particle_field()`: simulation loop, time stepping, boundary conditions, initialization |

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

**Configuring multiple particle types:**

**Example 1: Arbitrary attraction/repulsion** (from `config/arbitrary/arbitrary_3.yaml`):
```yaml
description: 'attraction-repulsion with 3 types particles'
dataset: 'arbitrary_3'

simulation:
  params: [[1.6233, 1.0413, 1.6012, 1.5615], [1.7667, 1.8308, 1.0855, 1.9055], [1.7226, 1.7850, 1.0584, 1.8579]]
  func_params: [['arbitrary', 0, 0], ['arbitrary', 1, 1], ['arbitrary', 2, 2]]
  min_radius: 0
  max_radius: 0.075
  n_particles: 4800
  n_particle_types: 3
  n_frames: 250
  delta_t: 0.1
  boundary: 'periodic'

graph_model:
  particle_model_name: 'PDE_A'
  mesh_model_name: ''
```

**Example 2: Boids with multiple types** (from `config/boids/boids_16_256.yaml`):
```yaml
description: 'Boids 16 different types'
dataset: 'boids_16_256'

simulation:
  # Each type has [alignment, cohesion, separation] parameters
  params: [[27.6, 92.5, 48.2], [32.0, 51.8, 29.8], [23.6, 35.0, 13.5], [3.3, 76.4, 13.0]]
  min_radius: 0.001
  max_radius: 0.04
  n_particles: 1792
  n_particle_types: 4  # Can be up to 16
  n_frames: 8000
  delta_t: 0.5
  boundary: 'periodic'

graph_model:
  particle_model_name: 'PDE_B'
  mesh_model_name: ''
  prediction: '2nd_derivative'  # Boids uses acceleration
```

Key differences:
- **PDE_A** (arbitrary): distance-dependent attraction/repulsion forces
- **PDE_B** (boids): alignment, cohesion, separation behaviors per type

**Note**: The current `PDE_D.py` (diffusiophoresis) does NOT support multiple particle types - it uses single values (M1, M2, consumption_rate) for all particles. To add type-specific behavior, you would need to modify at a BLOCK END:

1. **`PDE_D.py`**:
   - Read particle type from `x[:, 5]` (type index)
   - Use type-specific parameters from `params` array
   - Apply different mobility/consumption/production per type

2. **`src/ParticleGraph/models/utils.py`** (function around line 1643):
   - Add `PDE_ParticleField_D` to the model initialization match statement
   - See how `PDE_A` and `PDE_B` handle multiple types via `Interaction_Particle`

**Safety rules:**
- Make ONE change at a time
- Document hypothesis for the change
- Compare directly to parent (same config, code-only diff)
- Never modify GNN_LLM.py

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
## Knowledge Base

### Pattern Principles
[What parameter ranges produce interesting patterns]

### Failed Configurations
[What to avoid]

### Code Insights
[What code changes helped/hurt]

---

## Previous Block Summary
[Short summary of last block]

---

## Current Block

### Block Info
Parameters: ...
Iterations: M to M+8

### Hypothesis
[What are we exploring this block?]

### Iterations This Block
[Current block iterations only]

### Emerging Observations
[What's working/failing]
```

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
