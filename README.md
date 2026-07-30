# ARC_AGI

UBP/GLM experiments on ARC, now consolidated into one operational entry point.

*E. R. A. Craig — updated 30 July 2026*

## Current status

**Best verified score on `data/training`: 9/50 (18%)**

This repository now has a single practical entry point for the current best stack:

- **`v064_ubp_glm_operational.py`** — consolidated operational solver
- learns only from train pairs
- applies the learned rule to the held-out test input
- benchmarks the current 50-task dev/training subset
- writes markdown + JSON reports explaining both successes and failures

This is the honest state of the system: it solves a small but real slice of ARC tasks when the transformation collapses to a single interpretable rule family. It still fails when the task needs chained conditional reasoning, object ranking, structural selection, or size-changing extraction.

## What it currently solves

| Task | Solver | Family |
|---|---|---|
| `00dbd492` | `multi_interior_fill` | enclosed-region fill learned by region size |
| `1e0a9b12` | `gravity_down` | column-wise gravity / compaction |
| `396d80d7` | `minkowski_distance` | distance-selected fill |
| `45737921` | `local_swap` | two-colour component swap |
| `54d82841` | `colour_center_fill` | object-centre projection to bottom row |
| `575b1a71` | `column_rank_fill` | global column-rank fill |
| `a85d4709` | `marker_fill_85` | row-marker driven fill |
| `ae58858e` | `cond_recolour` | component-size conditional recolour |
| `e48d4e1a` | `cross_shift_by_markers` | marker-count-driven cross translation |

## Why it solves these

The working tasks all share one property: after looking at the train pairs, the transformation can be captured by **one stable rule family**.

Examples:
- **single geometric law**: gravity, cross translation, centre projection
- **single global ordering law**: column-rank fill
- **single object predicate**: component size threshold
- **single metric law**: distance/adjacency-guided fill
- **single region-fill law**: enclosed-region size to fill colour

UBP/GLM helps here because the system can describe the task as a structural perturbation and then route it into a compact solver family.

## Why it still fails on the rest

The current bottleneck is **solver expressiveness**, not bookkeeping.

Most remaining tasks require one or more of these missing capabilities:

1. **Conditional recolouring beyond one predicate**  
   The same colour is preserved in one place, erased in another, and transformed elsewhere.

2. **Relational object selection**  
   The system must choose *which* object, block, frame, or region matters before transforming it.

3. **Multi-step composition**  
   Many tasks need sequences like: select object → erase support pattern → derive new colour → rebuild geometry.

4. **Size-changing transforms**  
   Crop, extraction, block selection, downsampling, and region compression are still weak.

5. **Template transfer / structural rewriting**  
   Some tasks do not merely recolour or fill; they reassign ownership of space between object classes.

In short: the current solver set is good at **one-rule tasks** and weak at **decision chains**.

## Operational entry point

### Benchmark the repository subset

```bash
python v064_ubp_glm_operational.py --batch data/training
```

Outputs:
- console summary
- `REPORTS/v064_operational_report.md`
- `REPORTS/v064_operational_report.json`
- `glm_state/ubp_glm_operational_state.json`

### Explain one task

```bash
python v064_ubp_glm_operational.py --task data/training/e48d4e1a.json
```

This prints:
- whether the task is solved
- which solver family handled it
- the predicted grid
- the diagnosis for why it is or is not covered

## Architecture

### 1. Physics-guided task signature

The system still uses the unified-physics framing:
- Hamming / activation change
- interference
- force
- cascade steps
- coarse task category: `expand`, `enrich`, `preserve`, `simplify`, `compress`

These are useful for grouping and explaining tasks, but they do **not** by themselves solve ARC.

### 2. Verified solver registry

`v064` consolidates the verified families into one registry:
- `multi_interior_fill`
- `gravity_down`
- `minkowski_distance`
- `local_swap`
- `colour_center_fill`
- `column_rank_fill`
- `marker_fill_85`
- `cond_recolour`
- `cross_shift_by_markers`

Each solver must reproduce **all train pairs exactly** before its test prediction is accepted.

### 3. Honest diagnosis for misses

For every unsolved task, `v064` records structural blockers such as:
- no consistent global colour mapping
- needs conditional recolouring
- needs relational object selection
- needs size-changing transform
- needs multi-step erase + synthesis
- introduces derived fill colour from structure

## Key files

```text
README.md                         ← this file
v064_ubp_glm_operational.py      ← consolidated operational system
v062_unified_learning.py         ← unified-state / physics-guided predecessor
v063_simplify_compress.py        ← simplify/compress experiments
v049_consolidated.py             ← earlier consolidated solver
v032_distance_rule.py            ← distance-rule discovery
REPORTS/v064_operational_report.md
REPORTS/v064_operational_report.json
glm_state/ubp_glm_operational_state.json
```

## Recommended next engineering steps

If you want the next real jump in score, the highest-leverage additions are:

1. **Object-selection + crop/extract solver family**  
   Especially for `compress` tasks like `662c240a`.

2. **Two-step and three-step composition search**  
   Not brute force over everything — guided composition over a small typed operator set.

3. **Predicate induction over objects, not cells**  
   Example: “select the solid rectangle”, “select the block with minority corner pattern”, “select the component touched by marker X”.

4. **Template transfer after selection**  
   Needed for tasks where one object's footprint is rewritten using another object's logic.

5. **Typed solver routing**  
   First classify: same-size vs size-change, object-selection vs fill, recolour vs rewrite — then search only within that branch.

## Bottom line

This repo now has a **single operational UBP/GLM system** that is easy to run, benchmark, and inspect. It is not claiming general ARC competence. It is claiming something narrower and true:

- the current UBP/GLM substrate can already solve a handful of real ARC tasks,
- the solver families are interpretable,
- the failure modes are now explicit,
- and the next bottleneck is the addition of compositional, object-selective, size-changing solvers.

## License

MIT.
