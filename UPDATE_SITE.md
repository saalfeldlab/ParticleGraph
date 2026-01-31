# Updating the ParticleGraph Documentation Site

## Prerequisites

Before updating site content, ensure the exploration has produced new data:
- Montages, videos, and UCB trees in the media directory (see below)
- Updated `diffusiophoresis_Claude_analysis.md` and `diffusiophoresis_Claude_memory.md`

## Quick Render

```bash
cd /workspace/ParticleGraph
quarto render
```

Output goes to `_site/`.

## Clean Rebuild (stale images)

If the site still shows old montages (e.g. bottom-right panel printing
metrics text instead of the flow field), delete both `_site/` and the
Quarto cache:

```bash
rm -rf _site .quarto
quarto render
```

## Image / Video Source

All media must come from one directory only:

```
log/Claude_exploration/instruction_diffusiophoresis_parallel/
  montage/   montage_iter_NNN.png   (iters 003-088)
  figure/    figure_iter_NNN.png
  video/     video_iter_NNN.mp4
  tree/      ucb_tree_iter_NNN.png  (every 8 iters)
  config/    iter_NNN_slot_NN.yaml
```

The old path `graphs_data/diffusiophoresis/diffusiophoresis_Claude/` is
**deprecated** and must not be used — those images have a metrics text
panel instead of the flow field in the bottom-right.

### Montage layout (4 rows x 10 time-steps)

| Row | Content |
|-----|---------|
| 1   | Chemical field C1 |
| 2   | Particles with velocity arrows |
| 3   | C1/C2 fields with particle overlays |
| 4   | **Particle flow field** (replaced old metrics panel) |

## Quarto Source Files

| File | Content | Last Update |
|------|---------|-------------|
| `_quarto.yml` | Site config, navbar, theme | |
| `index.qmd` | Landing page | Block 11 |
| `field-field.qmd` | Field-Field (Turing patterns) | Block 11 |
| `field-particle.qmd` | Diffusiophoresis | |
| `particle-field.qmd` | Particle-Field | |
| `particle-particle.qmd` | Particle-Particle | |
| `brusselator.qmd` | Brusselator model (hex + stripe + NLD labyrinthine/vermiform) | Block 11 |
| `grayscott.qmd` | Gray-Scott model | |
| `fhn.qmd` | FitzHugh-Nagumo model | |
| `schnakenberg.qmd` | Schnakenberg model | |
| `llm-exploration.qmd` | Exploration gallery (montages, videos, UCB trees) | Block 11 |
| `mechanochemical.qmd` | Mechanochemical / MPM extension | |

## Update Procedure

### Step 1: Update Quarto Content

Update these .qmd files based on new discoveries in `diffusiophoresis_Claude_memory.md`:

1. **llm-exploration.qmd** — Add new block gallery, update regime comparison, principles, score table, UCB trees
2. **brusselator.qmd** — Add new pattern modes if discovered (e.g. NLD labyrinthine/vermiform from Block 11)
3. **index.qmd** — Update date, iteration count, discovery summary
4. **field-field.qmd** — Update model descriptions if new capabilities discovered
5. Other model pages as needed

### Step 2: Render the Site

```bash
cd /workspace/ParticleGraph
rm -rf _site .quarto
quarto render
```

### Step 3: Push to GitHub

```bash
cd /workspace/ParticleGraph
git add -A
git commit -m "Update site with Block N results"
git push
```

## Adding New Iterations

Add to `llm-exploration.qmd`:

```markdown
![Caption](log/Claude_exploration/instruction_diffusiophoresis_parallel/montage/montage_iter_NNN.png){.lightbox group="blockN"}

{{< video log/Claude_exploration/instruction_diffusiophoresis_parallel/video/video_iter_NNN.mp4 >}}
```

## NFS Mount

`/workspace` is mounted from `prfs.hhmi.org:/groups/saalfeld/home/allierc/Graph`,
so both paths point to the same files.

## Current State (Block 11, 88 iterations)

- **Best scores**: 8/10 (Iters 14, 45, 53 — 3-type Brusselator flower/mandala)
- **Novel patterns**: Labyrinthine Turing (Iter 83), vermiform filaments (Iter 87)
- **20 established principles** across 11 blocks
- **5 PDE models tested**, only Brusselator produces non-radial morphologies
- **NLD code modification** (Block 11): nonlinear diffusion enables labyrinthine/vermiform modes
