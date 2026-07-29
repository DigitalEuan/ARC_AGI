# ARC_AGI
The UBP method of working with the ARC AGI benchmark - geometry can speak.

* E R A Craig, New Zealand July 2026

The Golay-Leech geometry has been promoted from a mathematical curiosity to a candidate substrate for general cognition — a 24-dimensional error-correcting code that also happens to support reasoning.

## The ARC-AGI-3 Challenge
The Abstraction and Reasoning Corpus, now in its third iteration, is François Chollet's standing challenge to the field of artificial intelligence. The benchmark presents a small number of input-output grid pairs (typically two to five) drawn from a held-out vocabulary of visual transformations, and asks the solver to apply the same transformation to a held-out test input. Each task is essentially unique — there is no training corpus, no fine-tuning phase, and the 2026 iteration deliberately introduces task families that did not appear in any prior year.

## ARC-AGI-3 and the GLM share a deep structural kinship 
— both treat information as discrete, topological, and coherence-ranked — and that the gap between them is closable by extending the GLM's grammar in three modest directions.

## Current capability

**Solve rate: 5/50 (10%)** on 50 real ARC-AGI-2 training tasks.

| # | Task | Solver | Module |
|---|------|--------|--------|
| 1 | `1e0a9b12` | dsl_GRAVITY_DOWN | v029 (DSL ops) |
| 2 | `45737921` | free_k_arm (soft neighbourhood) | v029 (senses) |
| 3 | `396d80d7` | Minkowski p=1.5 distance rule | v032/v033 (distance) |
| 4 | `575b1a71` | Column-rank fill | v044 (disruption lens) |
| 5 | `ae58858e` | Component size ≥ 4 → recolour | v044 (disruption lens) |

## Architecture

### The Disruption Lens (new — July 2026)

The ARC problem is reframed as a **perturbation response**:
- **P** = the substrate (background grid)
- **A** = the input (perturbation)
- **B** = the output (equilibrium after perturbation propagates)

The disruption lens asks: "How does A perturb P, and what does the perturbed state look like?" This reveals **global patterns** that local approaches (distance, neighbourhood, DSL) miss.

Key discovery: `575b1a71` uses a **column-rank fill** rule — fill colour = rank of the cell's column among all columns that contain zeros. This depends on the *entire grid structure*, not just local context.

### Sensor Suite

| Sensor | Module | What it measures |
|--------|--------|-----------------|
| Minkowski distance | `v033_minkowski_sweep.py` | L₁, L₁.₅, L₂, L∞ distance fields |
| Totient kinetics | `spatial_totient_kinetics.py` | R(N), C(N), tension, φ(N) |
| Cayley-Menger | `v036_cayley_menger.py` | Object identity, containment |
| Lucas-Lehmer | `v037_lucas_lehmer.py` | Trajectory fingerprint (RSC) |
| Per-colour Minkowski | `v038_per_colour_minkowski.py` | Distance to each colour |
| MOG addresses | `v040_conditional_recolour.py` | 24-bit local context fingerprint |
| Neighbourhood bitmask | `v041_neighbourhood_bitmask.py` | 8-bit Moore neighbourhood |
| Object segmentation | `v042_object_level.py` | Connected components, frames |
| Disruption analysis | `v044_disruption.py` | Perturbation patterns |

### The 24-bit Leech address

Every ARC cell is encoded into a 24-bit vector via the MOG_CATEGORIES partition:
- **Mirrors** (bits 0-5): colour fingerprint
- **Information** (bits 6-11): position topology
- **Activation** (bits 12-17): spatial context
- **Potential** (bits 18-23): relational fingerprint

The `mog_leech.py` module provides the exact coordinate system: 4 rows × 6 columns, row-major, with named layers (Mirrors, Information, Activation, Potential).

### The GLM Pipeline (v029)

```
Task arrives
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: SENSES (6 sensory modules)                            │
│   Touch, Sight, Proprioception, Audition, Smell, Taste         │
│   Each sense can GENERATE candidates (not just observe).       │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: CORTEX (rule derivation + reasoning)                  │
│   Cortex v2, Meta-rule, Displacement extrapolation,            │
│   Thoughts layer, Coherence thought                            │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: VERIFICATION (hard gate)                              │
│   Every candidate must reproduce every train pair EXACTLY.     │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: TIEBREAK (Occam's razor + sensory alignment)          │
│   Source priority (MDL) + smell similarity + rhythm match       │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: DIAGNOSIS (full sensory readout)                      │
│   NRCI, LDP, HDRB, MOG, auditory rhythm, per-cell coherence    │
└─────────────────────────────────────────────────────────────────┘
```

## Experiment Tracker

See `EXPERIMENT_TRACKER.md` for the full log of all 16 modules built and tested.

### What worked (5 solves)
1. **DSL gravity** — simple spatial operation
2. **k-arm similarity** — neighbourhood matching
3. **Minkowski p=1.5** — automated distance rule discovery
4. **Column-rank fill** — global disruption pattern
5. **Component-size recolour** — object-level rule

### What didn't work (and why)
- Totient kinetics, Cayley-Menger, Lucas-Lehmer — feature enrichment only
- Per-colour Minkowski — no colour-specific distance rules
- Compositional search — blocked by conditional gap
- Pattern matching (fill/spread/grow) — ARC fills aren't simple
- Rule learning (global/conditional/neighbour) — conditions too complex

### The Fundamental Finding
**Every ARC task needs conditional recolouring.** No task has a consistent global colour mapping. The conditions are more complex than single-cell neighbourhood rules.

## UBP Laws Applied

| Law | Application |
|-----|-------------|
| LAW_PATTERN_001 | "Visual puzzles are coherence maps" — disruption lens |
| LAW_OPTICAL_TOGGLE_001 | "Neighbour-dependent toggle" — propagation rules |
| LAW_TGIC_369_GENESIS | "3-Axis, 6-Face, 9-Neighbour" — spatial constraints |
| LAW_TOPOLOGICAL_ERASURE_001 | "Geometric stability over magnitude" — erasure patterns |

## File structure

```
ARC_AGI/
├── README.md                          ← this file
├── EXPERIMENT_TRACKER.md              ← full experiment log
├── EXPERIMENT_LOG.md                  ← detailed findings
├── DISRUPTION_ANALYSIS.md             ← disruption lens results
├── METHODS_TRIED.md                   ← ledger of methods tried
├── WORKLOG.md                         ← development worklog
│
├── v029_pipeline.py                   ← main GLM pipeline (5-stage)
├── v032_distance_rule.py              ← Minkowski distance rules (SOLVES 396d80d7)
├── v033_minkowski_sweep.py            ← vectorized p-norm sweep
├── v034_totient_kinetics.py           ← geometric number theory
├── v035_combined_pipeline.py          ← unified pipeline
├── v036_cayley_menger.py              ← object identity
├── v037_lucas_lehmer.py               ← trajectory fingerprint
├── v038_per_colour_minkowski.py       ← per-colour distance layers
├── v039_compositional_search.py       ← multi-step DSL composition
├── v040_conditional_recolour.py       ← MOG-encoded conditional recolour
├── v041_neighbourhood_bitmask.py      ← neighbourhood-masked recolour
├── v042_object_level.py               ← object segmentation
├── v043_composition_grammar.py        ← Occam's razor grammar
├── v044_disruption.py                 ← disruption lens (SOLVES 575b1a71)
├── v045_disruption_fisher.py          ← systematic pattern search
│
├── mog_leech.py                       ← MOG/Leech-24 coordinate system
├── spatial_totient_kinetics.py        ← Totient Reaction Kinetics engine
│
├── arc_loader/                        ← ARC task loading
├── encoder/                           ← 24-bit grid encoder
├── dsl/                               ← 162 DSL operators
├── generative/                        ← hex learner, geometric language, CRG
├── grammar/                           ← Φ-grammar, smart candidates
├── vendor/                            ← UBP backbone + senses + cortex
├── lingo/                             ← geometric translator, lingo chat
├── data/training/                     ← 50 real ARC-AGI-2 training tasks
├── REPORTS/                           ← gate validation reports
└── tests/                             ← test suites
```

## Quick start

```bash
# Run the main pipeline
python v029_pipeline.py --batch data/training --verbose

# Run the disruption fisher
python v045_disruption_fisher.py --batch data/training --verbose

# Run the Minkowski sweep
python v033_minkowski_sweep.py --batch data/training --verbose

# Run the disruption analysis
python v044_disruption.py --batch data/training --verbose

# Run MOG/Leech self-tests
python mog_leech.py --self-test

# Run Totient Kinetics
python spatial_totient_kinetics.py
```

## Version history

| Version | Solve rate | Key addition |
|---------|-----------|-------------|
| v0.20 | 1/50 (2%) | 162 DSL ops + hard gate |
| v0.21 | 1/50 (2%) | HDRB + hex-colour learning |
| v0.22 | 1/50 (2%) | Geometric language + free k-arm |
| v0.23 | 2/50 (4%) | Soft neighbourhood matching |
| v0.24 | 2/50 (4%) | 6 senses + geometric grammar |
| v0.29 | 2/50 (4%) | Top-down coherence thought |
| v0.32 | 3/50 (6%) | **Minkowski distance rules** |
| v0.33 | 3/50 (6%) | Automated Minkowski sweep (p=1.5) |
| v0.34 | 3/50 (6%) | Totient kinetics |
| v0.36 | 3/50 (6%) | Cayley-Menger object identity |
| v0.37 | 3/50 (6%) | Lucas-Lehmer trajectory fingerprint |
| v0.38 | 3/50 (6%) | Per-colour Minkowski layers |
| v0.39 | 3/50 (6%) | Compositional DSL search |
| v0.40 | 3/50 (6%) | MOG-encoded conditional recolour |
| v0.41 | 3/50 (6%) | Neighbourhood bitmask rules |
| v0.42 | 3/50 (6%) | Object segmentation |
| v0.43 | 3/50 (6%) | Composition grammar (Occam's razor) |
| **v0.44** | **5/50 (10%)** | **Disruption lens** |
| v0.45 | 5/50 (10%) | Systematic disruption fisher |

## Key insights

1. **The disruption lens works** — it found 2 new solves that 14 previous modules missed
2. **Minkowski p=1.5** expresses the composite Manhattan+Chebyshev rule as a single fractional norm
3. **ARC tasks are perturbation responses** — the input disrupts the substrate, the output is the equilibrium
4. **Global rules exist** — column-rank fill, component-size recolour depend on entire grid structure
5. **The tools work** — 10/10 synthetic tests pass; the gap is in composition, not capability
6. **The UBP/GLM is a substrate for capability** — it can encode any spatial relationship, but doesn't know which to use

## What's next

1. **Per-colour distance layers** — `dist_to_colour_X` for each non-bg colour
2. **Arrangement topology classifier** — scattered/frame/dense mode detection
3. **Compositional search with disruption classification** — use disruption type to select tools
4. **Weighted Minkowski** — row/column bias for anisotropic patterns
5. **Multi-sensor fusion** — combine all feature layers

## License

MIT — same as the parent UBP_Repo.

## Acknowledgements

Built on the UBP framework by E. R. A. Craig (DigitalEuan), using the
Golay [24,12,8] code, Leech lattice, and the Y = π/(π²+2) observer
constant.
