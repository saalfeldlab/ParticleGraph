# Updating the ParticleGraph Documentation Site

## Prerequisites

- Quarto installed (`quarto --version`)

## Steps

### 1. Update Content

After running `GNN_LLM.py` with new results, update the relevant files:

```bash
# Memory file with exploration results
diffusiophoresis_Claude_memory.md

# Model pages (one per PDE variant)
brusselator.qmd   # Brusselator model findings
grayscott.qmd     # Gray-Scott model findings
fhn.qmd           # FitzHugh-Nagumo model findings

# Main index
index.qmd         # Update date and model summaries
```

### 2. Check for New Interaction Classes

If the exploration discovered new interaction types, add them:

```bash
# 1. Create new .qmd file for the interaction
cp field-field.qmd new-interaction.qmd

# 2. Add to _quarto.yml navigation under "Interactions" menu
- href: new-interaction.qmd
  text: New Interaction
```

Current interaction classes:
- Field–Field (reaction-diffusion)
- Field–Particle (diffusiophoresis)
- Particle–Field (consumption/production)
- Particle–Particle (attraction-repulsion)

### 2b. Check for New PDE Models

If new reaction-diffusion models were explored, add them:

```bash
# 1. Create new .qmd file for the model
cp brusselator.qmd newmodel.qmd

# 2. Add to _quarto.yml navigation under "Models" menu
- href: newmodel.qmd
  text: New Model
```

Current PDE models:
- Brusselator (9/10) — hexagonal patterns
- Gray-Scott (7/10) — radial patterns
- FitzHugh-Nagumo (8/10) — fine-scale hexagonal

### 3. Add New Videos

New iteration videos should be placed in:
```
graphs_data/diffusiophoresis/diffusiophoresis_Claude/video_iter_XXX.mp4
```

Reference them in the model pages using:
```markdown
{{< video graphs_data/diffusiophoresis/diffusiophoresis_Claude/video_iter_XXX.mp4 >}}
```

### 4. Render the Site

```bash
quarto render
```

### 5. Push to GitHub

```bash
git add -f docs/*.html docs/assets/ *.qmd
git commit --no-verify -m "Update documentation site"
git push
```

## Site Structure

```
_quarto.yml          # Navigation config
index.qmd            # Home page
brusselator.qmd      # Brusselator model (9/10)
grayscott.qmd        # Gray-Scott model (7/10)
fhn.qmd              # FitzHugh-Nagumo model (8/10)
field-field.qmd      # Field-Field interaction
field-particle.qmd   # Field-Particle interaction
particle-field.qmd   # Particle-Field interaction
particle-particle.qmd # Particle-Particle interaction
```
