# ARC_AGI — Rewiring & Realignment Brief

*Companion to `METHODS_TRIED.md` / `EXPERIMENT_TRACKER.md` — read after those, not instead of them. This is the "what to actually change" note.*

## 1. Where the wall really is

Not "need more sensors." You have 45+ modules, and the ledger already proved most sensory depth (Cayley-Menger, Lucas-Lehmer, totient kinetics) is diagnostic-only. The wall is structural, in three specific places.

### Disconnection 1 — Candidate generation never conditions on objects
- `v039_compositional_search.py` beam-searches compositions of ~20 DSL ops, but blind — nothing in it conditions an op on a per-object property. That's why its own docstring calls it blocked by "the conditional gap."
- `generative/geometric_language.py`'s `read_sentence()` builds **one flat `GeometricSentence`** per task — a single modal rotation/direction/colour-mapping aggregated across every cell in every train pair. It won a real solve (`45737921`, free k-arm) because that task's rule happened to be global. It structurally cannot express "rank-2 objects get colour A, rank-3 get colour B," because nothing partitions cells by object before the aggregate is built.

### Disconnection 2 — The object/relation vocabulary exists but only *matches*, never *induces*
- `object_extractor.py`'s `GridObject` (colour, bbox, centroid, cell_count) and `object_crg_full.py`'s 30 `SpatialRelation` + 38 `TransformType` members are real and working.
- But that vocabulary powers B5–B7 (analogy/chain/group): find one similar object in train, copy its known transform. Nothing searches "which property + threshold predicts the outcome across *all* train pairs" — the actual shape of your two live disruption-lens wins, which are currently hand-written, one function per discovered pattern, in `v045_disruption_fisher.py`.

### Disconnection 3 — "Learning as it goes" is real code, wired to the wrong thing (or nothing)
- `vendor/GLM24_continuous_learner.py` is a genuine continuous-learning loop — co-occurrence, vector refinement, persistence — but it reads *chat vocabulary*. It has never seen an ARC grid.
- `generative/crg_persistence.py` (`save_crg`/`load_crg` → `data/crg_state.json`) is built for ARC specifically ("accumulates knowledge... across tasks"). Confirmed by grep: called nowhere outside its own module — not in `run_pipeline.py`, `v029`, `v044`, or `v045`.
- Even the leftover `data/crg_state.json` (161 nodes, 140 edges) stores **literal instances** — e.g. `input_colour: 0, output_colour: 3, transform_type: appear, position_delta: [4.0, 10.0], size_ratio: 48.0`. Colours and pixel deltas are task-specific; nothing here generalizes across a different palette or grid size. This is the concrete mechanism behind your own E8 finding ("no cross-task transfer observed").

## 2. The one fix that addresses all three

Automate the induction step you currently do by hand, and make *its output* — not raw edges — the thing that persists.

**Build `v046_predicate_induction.py`** (extends the `verify_and_predict` pattern already in v045):
1. Decompose train input/output grids into objects (`GridObject` — already built).
2. Align input objects to output objects (overlap / nearest centroid).
3. Compute a property vector per object: `cell_count`, `colour`, bbox dims, centroid, **size-rank among siblings**, touches-border, neighbour-count, plus relations from `SpatialRelation` (ADJACENT, CONTAINS, ALIGNED_H/V).
4. Generate candidate predicates per property: `==`, `>=`, `<=`, `rank==k`, `is_max`, `is_min`.
5. Keep predicates whose truth value maps to the *same* outcome for every object in every train pair — not just one.
6. Hard-gate the reconstructed grid against every train pair (D1 stays non-negotiable).
7. Tiebreak survivors with the existing MDL/Occam priority (C5) — no new ranker needed.

**Then build the accumulator (this is the "learns as it goes" part):**
- When step 5 finds a surviving predicate, store the **template** — property + operator + outcome-shape, with colours/thresholds left as parameters — not the literal instance.
- Each new task tries the growing template library first (cheap, ranked by past success count) before falling back to fresh search.
- This is what `v045`'s fixed 12-pattern menu already is, built by hand once. Automating "notice a new pattern → add it to the menu" is the whole ask.

## 3. Build order

1. `v046_predicate_induction.py` — run against `data/training`, get the real solve-rate delta, don't estimate it.
2. A template library (`data/rule_templates.json`) replacing the unused `crg_state.json` mechanism — same idea, right grain this time.
3. Feed the library into `v045`'s fisher as additional candidate generators, ranked by prior success.
4. Only then revisit `geometric_language.py`: partition by object class before building the sentence, so it can express conditional rules, not just global ones.

## 4. What not to repeat (already proven, per METHODS_TRIED.md)

- Don't soften the hard gate (D2/D3 — tried, backfired every time).
- Don't add more diagnostic-only sensors — zero solves each, already logged.
- Don't extend blind DSL composition (v039) — it's the disconnection, not the fix.
- Don't expect literal edge accumulation to transfer across tasks (E8) — now you know the exact mechanism why not.

## 5. To study alongside this

- **ARGA** (Xu et al., 2022) / **GPAR** (Lei et al., 2024) — object graphs + constrained relational search; closest peer lineage to §2.
- **Rocha, Dutra, Santos Costa & Reis (2025)** — casts ARC as sequential Inductive Logic Programming over object-centric abstractions; near-identical shape to the predicate induction step above.
- **DreamCoder-style library learning** — the general pattern behind §2's accumulator: compress solved traces into new named, parameterized primitives instead of logging raw instances.
