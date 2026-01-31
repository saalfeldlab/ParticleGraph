# Parallel Mode Addendum

This addendum applies when running in **parallel mode** (GNN_LLM_parallel.py). Follow all rules from the base instruction file, with these modifications.

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot has its own config file, metrics log, and montage image
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field is pre-set to route data to separate directories — **DO NOT change the `dataset` field**
- Only modify `simulation:` parameters (and `claude:` where allowed)

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices:

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB node, or same parent different param |
| 2 | **explore** | Under-visited node, or new parameter dimension |
| 3 | **principle-test** | Randomly pick one Established Principle from `memory.md` and design an experiment that tests or challenges it (see below) |

You may deviate from this split based on context (e.g., all exploit if early in block, all boundary-probe if everything converges).

### Slot 3: Principle Testing

At each batch, slot 3 should be used to **validate or challenge** one of the Established Principles listed in the working memory (`{config}_memory.md`):

1. Read the "Established Principles" section in memory.md
2. **Randomly select one principle** (rotate through them across batches — do not repeat the same one consecutively)
3. Design a config that specifically tests this principle:
   - If the principle says "X works when Y", test it under a different condition
   - If the principle says "Z always fails", try to make Z succeed
   - If the principle gives a range, test at the boundary
4. In the log entry, write: `Mode/Strategy: principle-test`
5. In the Mutation line, include: `Testing principle: "[quoted principle text]"`
6. After results, update the principle's evidence level in memory.md:
   - Confirmed → keep in Established Principles
   - Contradicted → move to Open Questions with note

If there are no Established Principles yet (early in the experiment), use slot 3 as a **boundary-probe** instead.

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the simulation regime
- Create 4 diverse initial simulation parameter variations
- Suggested spread: vary key PDE parameters (e.g., D_C1, D_C2, Da_c, chi, consumption/production rates)
- All 4 slots can have different simulation parameters
- Write the planned initial variations to the working memory file

## Logging Format

Same as base instructions, but you write 4 entries per batch:

```
## Iter N: [score]/10
Node: id=N, parent=P
Mode/Strategy: [strategy]
Score: [score]/10
Config: [key simulation parameters]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change. For principle-test slots, append the principle being tested (e.g., `Mutation: chi: -16 -> -32. Testing principle: "chi magnitude > 20 causes instability"`).

Write all 4 entries before editing the 4 config files for the next batch.

## Block Boundaries

- At block boundaries, follow the base instruction file's block-end workflow for all 4 slots
- Block end is detected when any iteration in the batch hits `n_iter_block`

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- Do not draw conclusions from a single failure (may be stochastic)

## Code Modifications

Code modifications (PDE changes) follow the same block-end rules as the base instruction file. When modifying code at block end, all 4 slots share the same code changes for the next block.
