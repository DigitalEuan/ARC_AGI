# ARC_AGI
The UBP method of working with the ARC AGI benchmark - geometry can speak.

* E R A Craig, New Zealand July 2026

The Golay-Leech geometry has been promoted from a mathematical curiosity to a candidate substrate for general cognition — a 24-dimensional error-correcting code that also happens to support reasoning.

## The ARC-AGI-3 Challenge
The Abstraction and Reasoning Corpus, now in its third iteration, is François Chollet's standing challenge to the field of artificial intelligence. The benchmark presents a small number of input-output grid pairs (typically two to five) drawn from a held-out vocabulary of visual transformations, and asks the solver to apply the same transformation to a held-out test input. Each task is essentially unique — there is no training corpus, no fine-tuning phase, and the 2026 iteration deliberately introduces task families that did not appear in any prior year.

## ARC-AGI-3 and the GLM share a deep structural kinship 
— both treat information as discrete, topological, and coherence-ranked — and that the gap between them is closable by extending the GLM's grammar in three modest directions.

## Perception Layer: The 24-bit Grid Encoder on MOG_CATEGORIES
The encoder is the only entirely new component in the architecture. Its job is to convert a 2-D coloured grid into a 24-bit vector that preserves the four orthogonal descriptors needed for downstream reasoning: which colours are present, how many objects each colour has, where the dominant object sits, and how the objects relate topologically. The bit budget in Figure 4.2 reuses the existing GLM01_substrate MOG_CATEGORIES partition verbatim — Mirrors bits 0–5, Information bits 6–11, Activation bits 12–17, Potential bits 18–23 — but assigns each quadrant a new ARC-specific role that maps onto its existing categories.

# GLM × ARC-AGI-3 — Geometry Language Machine for ARC

**Version:** v0.29 (Cortex + Thoughts Layer + Top-Down Coherence)
**Author:** E. R. A. Craig (DigitalEuan), Auckland, NZ
**Date:** 29 July 2026

## What this is

A system that adapts the **Geometry Language Machine (GLM)** — built on the
**Universal Binary Principle (UBP)**, Golay codes, Leech lattice, and the
Y observer constant — to solve **ARC-AGI-3** abstraction-and-reasoning tasks.

The GLM treats every ARC cell as a **physical object** with a 24-bit Leech
lattice address (which IS a hex colour). Transformations are addressed as
**displacements in 24-bit space**, not symbolic lookups. The system has
**6 senses** (touch, sight, proprioception, audition, smell, taste), a
**cortex** that derives relational rules, and a **thoughts layer** that
writes structured reasoning as text.

## Current capability

**Solve rate: 2/50 (4%)** on 50 real ARC-AGI-2 training tasks.

The 2 solved tasks:
- `1e0a9b12` — gravity (downward collapse), solved by DSL `GRAVITY_DOWN`
- `45737921` — contextual recolour, solved by free k-arm with soft
  neighbourhood matching

On close-but-wrong tasks (70-95% cell accuracy):
- `396d80d7` — 95.31% accuracy via cortex relational rule
  ("7 with trigger in diagonal NOT cardinal → target")
- `7acdf6d3` — 94.67% via hex k-NN
- `ae58858e` — 91.67% via hex k-NN
- `e509e548` — 86.16% via hex k-NN

## Architecture

```
Task arrives
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: SENSES (6 sensory modules)                            │
│                                                                 │
│  Touch        k-arm + soft neighbourhood matching               │
│  (geometric_language)                                           │
│                                                                 │
│  Sight        colour bridge: 24-bit → RGB332 (256 colours)      │
│  (colour_space_bridge)                                          │
│                                                                 │
│  Proprioception  MOG bit-addressed meaning (576 dims/cell)    │
│  (mog_meaning_encoder)                                          │
│                                                                 │
│  Audition     periodicity/rhythm detection + generation         │
│  (auditory_sense)                                               │
│                                                                 │
│  Smell        long-range Gestalt (4×4 downsampled icon)         │
│  (smell_taste_sense)                                            │
│                                                                 │
│  Taste        local composition (histogram + texture)           │
│  (smell_taste_sense + taste_generative)                         │
│                                                                 │
│  Each sense can GENERATE candidates (not just observe).         │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: CORTEX (rule derivation + reasoning)                  │
│                                                                 │
│  Cortex v2 (cortex_v2.py)                                       │
│    - Y as EXTERNAL observer (not internal point)                │
│    - Y wobble: ~3 bits Hamming (from Y's continued fraction)    │
│    - Jaccard viewpoint comparison (orthographic vs perspective) │
│    - Relational rules: "A has T in diagonal NOT cardinal → C"  │
│                                                                 │
│  Meta-rule (meta_rule.py)                                       │
│    - Relational condition + dynamic contextual lookup           │
│    - Extrapolates to unseen trigger colours                     │
│                                                                 │
│  Displacement extrapolation (displacement_extrapolation.py)     │
│    - Uses NoiseCellV3's elastic_limit from UBP substrate        │
│    - Predicts target by applying train displacement to unseen   │
│                                                                 │
│  Thoughts layer (thoughts_layer.py)                             │
│    - Writes actual text + numbers as structured Thoughts        │
│    - 5 thought generators:                                      │
│      1. Global recolour                                         │
│      2. Relational trigger                                      │
│      3. Meta-rule (relational + extrapolation)                  │
│      4. Arithmetic pattern (add/sub/mul/complement)             │
│      5. Top-down coherence (output-driven refinement)           │
│    - Each thought: observation, pattern, hypothesis, prediction,│
│      confidence, evidence, references                           │
│    - Selects best thought that passes hard gate                 │
│                                                                 │
│  Coherence thought (coherence_thought.py)                       │
│    - "What target colour makes the output most coherent?"       │
│    - Scores candidates via: smell, taste, rhythm, NRCI,         │
│      perfect_distance (from PERFECT_V1 substrate)               │
│    - Top-down: starts from desired output, works backward       │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: VERIFICATION (hard gate)                              │
│                                                                 │
│  Every candidate must reproduce every train pair EXACTLY.       │
│  No soft thresholds, no fuzzy matching.                         │
│  This is the only reliable filter.                              │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: TIEBREAK (Occam's razor + sensory alignment)          │
│                                                                 │
│  Source priority (MDL):                                         │
│    identity < gravity/rotate/flip < shift/crop < recolour      │
│    < train_map < compose < hex_uniform < hex_colour_map         │
│    < hex_nearest < taste < analogy/chain/group                  │
│                                                                 │
│  Secondary: smell similarity, rhythm match, HDRB signature      │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: DIAGNOSIS (full sensory readout)                      │
│                                                                 │
│  NRCI, LDP, eml, HDRB signature, colour bridge identity,       │
│  MOG meaning decoder, auditory rhythm, per-cell coherence       │
└─────────────────────────────────────────────────────────────────┘
```

## Key concepts

### The 24-bit Leech address

Every ARC cell (row, col, colour, grid_h, grid_w) is encoded into a
24-bit vector via the UBP `ontological_position_to_vector` pipeline.
The 24 bits are partitioned into 4 MOG quadrants:
- **Mirrors** (bits 0-5): colour fingerprint
- **Information** (bits 6-11): position topology
- **Activation** (bits 12-17): grid dimensions
- **Potential** (bits 18-23): coherence pattern

The 24-bit vector IS a hex colour (#RRGGBB). This is the "data IS
address" principle.

### The Y constant

Y = π/(π²+2) ≈ 0.2647 — the observer constant. It is EXTERNAL to the
system: the "read" position between us experiencing the results and the
mechanisms making the results. It is NOT a spatial point inside the
24-bit address space.

The Y wobble (~3 bits Hamming) comes from Y's continued fraction
convergent 248/937 (the 7th convergent, accurate to 6 decimal places).
The wobble is the source of indeterminism — different Y positions give
slightly different perspective views.

### The 6 senses

| Sense | Module | What it perceives |
|-------|--------|-------------------|
| Touch | `geometric_language` | Cell context, 8-neighbour signature, k-arm reach |
| Sight | `colour_space_bridge` | Hex colour, RGB332, complement, harmony |
| Proprioception | `mog_meaning_encoder` | Bit-level meaning, 576-dim identity per cell |
| Audition | `auditory_sense` | Periodicity, rhythm, tiling structure |
| Smell | `smell_taste_sense` | Long-range Gestalt (4×4 downsampled icon) |
| Taste | `smell_taste_sense` | Local composition (histogram + texture) |

### The cortex

The cortex derives rules from train pairs and applies them to test. It
has 6 rule types (tried in order):
1. **Trigger-mapping**: "A with T in direction D → C"
2. **Dynamic contextual**: "A next to B → mapping[B]"
3. **Pattern**: "IF property P THEN transform"
4. **Orthographic**: global colour mapping (Y's outside view)
5. **Perspective**: focal vs peripheral (Y's inside view, with wobble)
6. **Relational**: "A has T in diagonal AND NOT in cardinal → C"

### The thoughts layer

The cortex writes its reasoning as structured `Thought` objects:
```
THOUGHT #2
  Observation: Colour 7 changes to 2 in 2 cells
  Pattern: Trigger: colour 6 in direction SE
  Hypothesis: If cell is 7 and has 6 in SE, set to 2
  Confidence: 1.00
  Passes train: True
```

5 thought generators produce competing thoughts; the best one that
passes the hard gate is selected.

### The top-down coherence thought

Asks: "What target colour makes the output grid most coherent?"

Scores each candidate output using:
- Smell similarity to train outputs (25%)
- Taste similarity to train outputs (25%)
- Rhythm match (20%)
- NRCI (15%) — peak at 0.75 (manifested range)
- Perfect distance (15%) — Hamming distance to PERFECT_V1 substrate

Refines bottom-up predictions by trying all 10 colours per uncertain
cell and picking the most coherent.

## File structure

```
arc_agi3/
├── README.md                      ← this file
├── METHODS_TRIED.md               ← ledger of every method tried
├── v029_pipeline.py               ← main pipeline (5-stage)
├── sharpened_pipeline.py          ← v0.20 baseline (DSL + hard gate)
│
├── arc_loader/                    ← ARC task loading
│   └── loader.py                  (Grid, ARCTask, TrainPair, TestInput)
│
├── encoder/
│   └── arc_to_24bit.py            (grid → 24-bit vector)
│
├── dsl/
│   └── arc_dsl_full.py            (162 DSL operators)
│
├── generative/
│   ├── hex_learner.py             ← hex-colour address learning (k-NN + voting)
│   ├── geometric_language.py      ← direction/Time/rotation primitives + free k-arm
│   ├── geometric_grammar.py       ← noun/verb/object/action/duration/gate
│   ├── object_crg_full.py         (ObjectCRG: 30 SpatialRelations, 38 TransformTypes)
│   ├── generative_transformer_full.py (CRG + Φ-grammar)
│   ├── prediction_paths.py        (analogy, chain, group prediction)
│   ├── object_extractor.py        (grid → objects)
│   ├── per_object_24d.py          (per-object 24D Leech addresses)
│   ├── srcc.py                    (Self-Referential Computational Cycle)
│   ├── crg_persistence.py         (save/load CRG)
│   └── ubp_action_engine.py       (regime-directed generation)
│
├── grammar/
│   ├── phi_grammar_arc_full.py    (full Φ-grammar with conditionals)
│   ├── smart_candidates.py        (candidate generator using all 162 ops)
│   └── conditional_candidates.py  (position-dependent detectors)
│
├── vendor/                        ← UBP backbone + senses + cortex
│   ├── ubp_unified_v5.py          ← UBP core (Golay, Leech, MOG, NoiseCellV3)
│   ├── spatial_arithmetic.py      (R(n), eml, RationalGeometry)
│   ├── spatial_arithmetic_compat.py (compat layer)
│   ├── hdrb.py                    ← Hodge-De Rham Bridge (4 pillars)
│   ├── colour_space_bridge.py     ← 2×4×8×32=2048 Golay-Leech ↔ colour bridge
│   ├── mog_meaning_encoder.py     ← per-bit 24D meaning (576 dims/cell)
│   ├── auditory_sense.py          ← periodicity/rhythm (generative)
│   ├── smell_taste_sense.py       ← smell (Gestalt) + taste (composition)
│   ├── taste_generative.py        ← taste-driven candidate generation
│   ├── per_cell_coherence.py      ← per-cell NRCI + coherence delta map
│   ├── cortex.py                  ← cortex v1 (Y as internal point)
│   ├── cortex_v2.py               ← cortex v2 (Y external + wobble + relational)
│   ├── meta_rule.py               ← meta-rule (relational + dynamic lookup)
│   ├── displacement_extrapolation.py ← NoiseCellV3 elastic_limit extrapolation
│   ├── thoughts_layer.py          ← structured thoughts (5 generators)
│   ├── coherence_thought.py       ← top-down coherence (output-driven)
│   ├── ldp.py                     (Literal Data Physics)
│   ├── ldp_codec.py               (LDP codec)
│   ├── tgic_v3.py                 (TGIC engine)
│   ├── GLM18_hex_colour.py        (24-bit → #RRGGBB)
│   └── ... (other UBP modules)
│
├── lingo/
│   ├── geometric_translator.py    (Totient Reaction Kinetics translator)
│   ├── lingo_translator.py        (human ↔ UBP-Lingo)
│   ├── lingo_chat.py              (GLM chat reasoning)
│   └── ubp_integration.py         (TopologicalALU, ObserverDynamics)
│
├── triadic_verifier.py            (Oracle + Swarm + NoiseCore)
├── ldp_grid_metrics.py            (LDP mass/tension/zone for grids)
│
├── data/
│   ├── training/                  (50 real ARC-AGI-2 training tasks)
│   └── crg_state.json             (persisted CRG)
│
├── tests/
│   ├── test_arc_agi3.py
│   ├── test_arc_agi3_v03.py
│   ├── test_arc_agi3_v05.py
│   └── test_arc_agi3_v02.py
│
├── REPORTS/
│   ├── gate1_encoder_validation.md
│   ├── gate2_grammar_extension.md
│   └── gate3_ranker_validation.md
│
└── scripts/                       (diagnostic scripts, in /home/z/my-project/scripts/)
```

## Quick start

```bash
cd arc_agi3

# Run the v0.29 pipeline on all 50 tasks
python v029_pipeline.py --batch data/training --verbose

# Run on a subset
python v029_pipeline.py --batch data/training --max-tasks 10

# Run individual modules (each has a self-test)
python vendor/cortex_v2.py
python vendor/thoughts_layer.py
python vendor/coherence_thought.py
python vendor/displacement_extrapolation.py
python vendor/meta_rule.py
python vendor/auditory_sense.py
python vendor/smell_taste_sense.py
python vendor/colour_space_bridge.py
python vendor/mog_meaning_encoder.py
python vendor/per_cell_coherence.py
python vendor/hdrb.py
python generative/geometric_language.py
python generative/geometric_grammar.py
python generative/hex_learner.py

# See the methods tried and their effects
cat METHODS_TRIED.md
```

## Version history

| Version | Solve rate | Key addition |
|---------|-----------|-------------|
| v0.20 | 1/50 (2%) | 162 DSL ops + hard gate (patched for hanging ops) |
| v0.21 | 1/50 (2%) | HDRB (4 pillars) + hex-colour learning |
| v0.22 | 1/50 (2%) | Geometric language (direction/Time/rotation) + free k-arm |
| v0.23 | 2/50 (4%) | Soft neighbourhood matching + NRCI 0.7 coherence gate |
| v0.24 | 2/50 (4%) | 6 senses + geometric grammar + per-cell coherence |
| v0.25 | 2/50 (4%) | Cortex v1 (Y as internal observer point) |
| v0.26 | 2/50 (4%) | Cortex v2 (Y external + wobble + relational rules) |
| v0.27 | 2/50 (4%) | Meta-rule (relational + dynamic lookup + extrapolation) |
| v0.28 | 2/50 (4%) | Displacement extrapolation + thoughts layer |
| v0.29 | 2/50 (4%) | Top-down coherence thought (output-driven refinement) |

## What works

1. **Hard gate** — exact train-pair reproduction is the only reliable filter
2. **Per-cell 24-bit Leech address** — the substrate (data IS address)
3. **Hex-colour k-NN with soft voting** — the "wobble", 70-95% cell accuracy on close tasks
4. **~20 DSL ops** — gravity, rotate, flip, recolour, shift, crop, tile
5. **Soft neighbourhood matching** — converted 45737921 from 95.83% to exact
6. **Relational rules** — "diagonal NOT cardinal" correctly identifies 396d80d7's pattern
7. **Thoughts layer** — structured, readable reasoning with multiple competing thoughts
8. **Top-down coherence** — output-driven refinement using all 6 senses
9. **Occam's razor tiebreak** — source priority by description length

## What doesn't work (yet)

1. **Semantic extrapolation** — predicting the target for an unseen trigger colour
   when the train mapping doesn't follow an arithmetic or address-proximity pattern
   (the 1→9 transformation on 396d80d7)
2. **Y as spatial position** — the perspective view doesn't discriminate because
   cells are roughly equidistant from Y in 24-bit space. The wobble (3 bits) is
   too small to create meaningful perspective differences on small grids.
3. **NRCI as ranker** — coherence ≠ correctness. NRCI measures structural
   coherence, not whether the transformation is right.
4. **Most of the 50+ GLM modules** — SRCC, TGIC, Bell partitions, triadic verifier,
   CRG persistence, lingo chat, geometric translator — contribute zero to solve rate.
   They add observability without generative power.

## The remaining bottleneck

The system can now:
- Derive the correct relational rule ("diagonal NOT cardinal") ✓
- Identify the correct trigger→target mapping from train ✓
- Write its reasoning as structured text ✓
- Consider multiple competing thoughts ✓
- Attempt extrapolation via the UBP substrate ✓
- Refine predictions top-down using all 6 senses ✓

What it can't do yet:
- **Semantic extrapolation**: predict 1→9 when train has {6→2, 4→1}.
  The actual relationship (1→9 = "colour complement" or "10-1") isn't
  captured by Hamming distance, arithmetic patterns, or displacement curves.
- **Multi-step thought chains**: thoughts can reference each other (via
  the `references` field) but don't yet build chains where thought N+1
  uses thought N's output as input.

## The Y constant's role

Y = π/(π²+2) ≈ 0.2647 is wired in as the external observer with wobble.
Its mathematical properties (transcendental, low Kolmogorov complexity,
closest simple function of π to the closed-loop zero) suggest it should
be a **universal scaling factor**, not a spatial position.

Potential roles not yet explored:
- **Confidence threshold**: accept extrapolation if confidence > 1-Y ≈ 0.735
- **Rule prior weight**: weight rules by Y^depth (deeper rules decay)
- **Perspective foreshortening**: use Y^distance instead of 1/(1+distance)

The Y's "time to shine" likely requires using it as a **decision boundary**
or **scaling factor**, not as a spatial eye position. This is open for
further research.

## License

MIT — same as the parent UBP_Repo.

## Acknowledgements

Built on the UBP framework by E. R. A. Craig (DigitalEuan), using the
Golay [24,12,8] code, Leech lattice, and the Y = π/(π²+2) observer
constant. The dimension projection audit (DIMENSION_PROJECTION_REVIEW.md)
provided essential corrections to the algebraic foundations.
