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

====
# V0.7 setup (note this repository is at v0.8):
====

## arc_agi3 — GLM × ARC-AGI-3 Refinement Branch

**Version:** v0.7.0 (SRCC monad + TGIC + Lingo chat + Bell partitions + CRG persistence)
**Author:** E. R. A. Craig (DigitalEuan), Auckland, NZ
**Date:** 27 July 2026

## v0.7 — The GLM Thinks

v0.7 gives the GLM the ability to **think before it acts** — via Lingo chat, the Self-Referential Computational Cycle (SRCC), and Bell number partition analysis.

### New Components

1. **Self-Referential Computational Cycle** (`generative/srcc.py`)
   - Implements the UBP cycle as a **monad (T, η, μ)** per the Deep Dive Research Report
   - The 12-component cycle = 4 layers × 3 functions (Reality/Information/Activation/Potential × Timing/Correction/Extraction)
   - Monad laws verified: left unit, right unit, associativity all pass
   - The RECURSION component feeds OUTPUT back to INPUT — the cycle runs on its own output until NRCI ≥ 0.70 (manifested)
   - Uses TGIC's HomologyJump (escape local minima), InformationFunctional (energy), CanonicalEvolution (snap to codeword)

2. **TGIC v3** (`vendor/tgic_v3.py`)
   - Aligned Triad-Graph Interaction Constraint — uses the REAL GolayCodeEngine
   - HomologyJumpOperator: jump between cosets by XORing with octads
   - InformationFunctional: Lyapunov energy (lower = more stable)
   - CanonicalEvolution: iteratively snap toward a codeword
   - Alignment verified: 4096 codewords, 759 octads, all zero syndrome

3. **Lingo Chat** (`lingo/lingo_chat.py`)
   - The GLM "chats" about each task in UBP-Lingo before solving it
   - Chat flows through the 4 MOG layers: Reality (observe) → Information (structure) → Activation (operations) → Potential (constraints)
   - Each chat message has both Lingo and human-readable forms
   - The chat is retained in the full system output as a reasoning trace

4. **Bell Number Partition Analysis** (`generative/srcc.py`)
   - Bell numbers count the ways to partition objects into transformation classes
   - B(4)=15, B(8)=4140 — each partition is a distinct learning method
   - With 8 objects (typical ARC task), there are 4140 possible learning methods
   - The GLM searches this partition space to find the method that best fits the train pairs

5. **CRG Persistence** (`generative/crg_persistence.py`)
   - Save/load the ObjectCRG across tasks (the GLM's "memory")
   - `merge_crgs()` accumulates learning — edges from new tasks reinforce existing edges
   - State saved to `data/crg_state.json`

6. **LDP Codec** (`vendor/ldp_codec.py`) — carried from v0.6
   - Geometric class (10-bit structural fingerprint)
   - Data-as-structure: integers are spatial clusters with topological mass

### Dual Output

Every task produces TWO outputs:
- **ARC JSON**: the accepted submission format (grid of integers)
- **Full system output**: Lingo chat, CRG stats, SRCC state, Bell analysis, NRCI coherence, geo_class fingerprints, Three Column check, Spatial Arithmetic calculations

Nothing is dropped — the full depth is retained alongside the ARC format.

## Results

| Metric | v0.5 | v0.6 | v0.7 |
|--------|------|------|------|
| Solve rate (5 real tasks) | 10% | 10% | **20%** |
| Time per task | 0.57s | 0.57s | 0.57s |
| Lingo chat | ✗ | ✗ | ✓ |
| SRCC monad | ✗ | ✗ | ✓ |
| TGIC integration | ✗ | ✗ | ✓ |
| Bell partition analysis | ✗ | ✗ | ✓ |
| CRG persistence | ✗ | ✗ | ✓ |
| Tests | 29/29 | 13/13 | 13/13 |

## Architecture

```
Task arrives
  ↓
Lingo Chat — GLM reasons about the task in UBP-Lingo
  (Reality: observe objects → Information: structure → Activation: learned ops → Potential: constraints)
  ↓
ObjectCRG — learn from train pairs (object-to-object transformations)
  ↓
GenerativeTransformer — predict via CRG → Φ-grammar → DSL vocabulary
  ↓
SRCC Cycle — run the monad (T, η, μ) on the predicted output
  (INPUT → CLOCK → MIRROR → FRICTION → COOLING → SELF-VALIDATION → OUTPUT → RECURSION)
  ↓
Three Column Check — language + math (NRCI) + code (train pass) must align
  ↓
Bell Analysis — how many learning methods are available for these objects?
  ↓
Dual Output:
  ├── ARC JSON (accepted submission format)
  └── Full System Output (Lingo chat + CRG + SRCC + Bell + NRCI + geo_class + calculations)
```

## Branch Structure

```
arc_agi3/
├── README.md
├── run_pipeline_v05.py        ← generative pipeline
├── dual_submission.py         ← dual-output harness (ARC JSON + full output)
├── generative/
│   ├── object_extractor.py    ← grid → objects (words)
│   ├── object_crg.py          ← learn object transformations
│   ├── generative_transformer.py ← CRG + Φ-grammar + Three Column
│   ├── srcc.py                ← Self-Referential Computational Cycle (monad) ← NEW
│   └── crg_persistence.py     ← save/load CRG across tasks ← NEW
├── lingo/
│   ├── lingo_translator.py    ← human ↔ UBP-Lingo translator
│   └── lingo_chat.py          ← GLM chat reasoning ← NEW
├── vendor/
│   ├── tgic_v3.py             ← TGIC engine (aligned) ← NEW
│   ├── ldp_codec.py           ← LDP codec
│   ├── ubp_unified_v5.py      ← UBP backbone
│   ├── refined_nrci.py        ← 5-shell NRCI
│   ├── spatial_arithmetic.py  ← R(n) primitive
│   └── ... (7 more GLM/UBP modules)
├── arc_loader/  encoder/  grammar/  ranker/  dsl/  learner/
├── tests/  data/  REPORTS/
```

## Quick Start

```bash
cd arc_agi3

# Run tests
python3 tests/test_arc_agi3.py        # v0.1 (7 tests)
python3 tests/test_arc_agi3_v05.py    # v0.5 (6 tests)

# Run the dual-output submission harness
python3 dual_submission.py data/training --max-tasks 10

# Run the v0.5 generative pipeline
python3 run_pipeline_v05.py --synthetic
python3 run_pipeline_v05.py --batch data/training --max-tasks 10
```

## Gate Status

| Gate | Status | v0.7 Change |
|------|--------|-------------|
| G1 — Encoder | ✓ provisionally closed | LDP geo_class + SRCC NRCI |
| G2 — Grammar | ✓ CLOSED | SRCC monad drives the grammar cycle |
| G3 — Ranker | ✗ NOT CLOSED | NRCI is coherence measure (SRCC converges on it) |
| G4 — DSL | ⏳ 20% on 5-task sample | DSL is "vocabulary"; Lingo is the "language" |
| G5 — Submission | ⏳ PARTIAL | Dual output produces ARC JSON + full reasoning |

## What's Next (v0.8 Roadmap)

1. **Per-object CRG prediction** — when the CRG identifies "recolour" but the global mapping fails, generate per-object colour mappings using Bell number partitions to search the method space
2. **Full 999-task benchmark** — run v0.7 on all 999 ARC training tasks
3. **CRG accumulation** — run the full benchmark with CRG persistence ON, so the GLM accumulates learning across all 999 tasks
4. **SRCC deepening** — use the RECURSION component to iteratively refine predictions (currently runs 1-3 iterations; could run more with HomologyJump escapes)
5. **Lingo chat as input** — let the user describe a transformation in human language, translate to Lingo, and apply

## License

MIT — same as the parent UBP_Repo.
