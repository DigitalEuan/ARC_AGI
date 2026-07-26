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
