# Gate 3 — Ranker Validation Report (v0.3 update)

**Date:** 26 July 2026
**Status:** REOPENED — v0.3 PatternLearner provides an alternative path that may make NRCI discrimination moot for many tasks

## v0.3 Development

The v0.3 PatternLearner was added as a parallel approach to the v0.2 symbolic search. The learner:
- Encodes each train pair's input+output as 24-bit hex colours (using GLM18's `vector_to_colour`)
- Records the transformation as an edge in a CRG (Concept Relation Graph)
- Detects common patterns across train pairs: identity, recolour, gravity, geometric, composite
- For composite tasks, detects positional patterns (swap, consistent recolour at positions)

## v0.3 Test Results

**8/8 v0.3 self-tests pass.** All 9 synthetic tasks (4 v0.1 + 5 v0.3) solved correctly:

| Task | Learner Pattern | Source | Correct |
|------|-----------------|--------|---------|
| rotate_90 | geometric | learner | ✓ |
| recolour | recolour | learner | ✓ |
| gravity | gravity | learner | ✓ |
| count_fill | count | symbolic | ✓ |
| identity | identity | learner | ✓ |
| recolour_consistent | recolour | learner | ✓ |
| gravity_up | gravity | learner | ✓ |
| flip_h | geometric | learner | ✓ |
| composite_swap | composite | learner | ✓ |

## Real ARC Task Results (v0.3)

On the same 10 real ARC tasks:
- **Learner-only: 0/10 solved**
- **v0.3 (learner + symbolic): 1/10 solved** (same task as v0.2: `1e0a9b12`)

The learner correctly identifies pattern types on real tasks (recolour, composite) but its predictions don't match the expected outputs. This is because real ARC tasks have more complex transformations than the learner's pattern detector can handle — the composite fallback (positional replay) doesn't generalise when the test input has different values than the train inputs.

## Critical Finding: NRCI Discrimination

The v0.3 results **confirm the v0.2 finding**: when only 1 candidate passes the train filter, NRCI has nothing to discriminate. The learner doesn't solve this — it provides an ALTERNATIVE path (learn the pattern directly) that bypasses the ranker entirely for tasks where the pattern is detectable.

For the 10 real tasks:
- 6 tasks classified as "composite" (no detectable pattern → fallback replay → wrong)
- 3 tasks classified as "recolour" (detected recolour pattern → but the mapping is wrong)
- 1 task solved by symbolic search (1e0a9b12)

## What's Needed to Close Gate 3

The v0.3 learner needs **better pattern detection** for real ARC tasks:
1. **Detect partial recolour patterns** — the learner currently requires 60% consistency across pairs; real tasks may have conditional recolouring (e.g., "recolour cells that are adjacent to colour X")
2. **Detect multi-step transformations** — the learner detects single ops; real tasks often require compositions (e.g., "rotate then recolour")
3. **Use the CRG for relational reasoning** — the learner builds a CRG but doesn't yet use it for prediction (the `find_nearest_transform` method exists but isn't called in `predict`)

## Conclusion

**Gate 3 remains NOT CLOSED.** The v0.3 PatternLearner is a promising alternative approach (8/8 synthetic tasks solved), but it doesn't yet generalise to real ARC tasks. The NRCI ranker's discrimination problem persists because the train filter remains too strict.

**Recommended next steps:**
1. Wire the CRG's `find_nearest_transform` into the learner's `predict` method — when no pattern is detected, use the CRG to find the nearest learned transformation and apply it
2. Add conditional pattern detection (e.g., "recolour cells based on neighbour properties")
3. Add multi-step pattern detection (e.g., "rotate then recolour" as a 2-step learned program)
4. Run the null-model comparison on 100 tasks with the v0.3 pipeline to see if the learner changes the NRCI advantage picture

---

*Next: Gate 4 (DSL Integration) and Gate 5 (Submission Readiness) reports.*
