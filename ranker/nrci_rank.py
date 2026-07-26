"""
nrci_rank.py — NRCI-based hypothesis ranker
=============================================

Ranks candidate Programs by the refined NRCI of their output encoding.
Combines two filters:
  1. Empirical adequacy: program must reproduce every train pair exactly
  2. Structural coherence: output's 24-bit encoding must have high refined NRCI

This is the ARC-specific analogue of the primality_NRCI pipeline's
dual-criterion (Miller-Rabin arithmetic primality + structural NRCI).

The ranker produces a four-way verdict per candidate:
  TRAIN-PASS/NRCI-HIGH   → submit (top-k)
  TRAIN-PASS/NRCI-LOW    → marginal
  TRAIN-FAIL/NRCI-HIGH   → structural curiosity (do not submit)
  TRAIN-FAIL/NRCI-LOW    → discard

Usage:
    from nrci_rank import Ranker, RankResult
    from grammar import generate_candidates
    from arc_loader import load_task

    task = load_task("task.json")
    candidates = generate_candidates(task)
    ranker = Ranker()
    results = ranker.rank(task, candidates)
    for r in results[:5]:
        print(f"{r.verdict}: nrci={r.nrci_refined:.4f}  {r.program}")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys, os

# Make vendored UBP backbone importable
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from ubp_unified_v5 import GOLAY_ENGINE, LEECH_ENGINE
from refined_nrci import RefinedNRCI

# Make arc_loader, encoder, dsl importable
_PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from arc_loader import Grid, ARCTask
from dsl import Program
from encoder import encode_grid


# ══════════════════════════════════════════════════════════════════════════════
# RANK RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RankResult:
    """Per-candidate ranking result."""
    program: Program
    train_pass: bool                  # True iff program reproduces all train outputs
    nrci_basic: float                 # basic NRCI (Leech) of test output
    nrci_refined: float               # 5-shell refined NRCI of test output
    manifested: bool                  # True iff refined NRCI ≥ 0.70
    test_output: Optional[Grid] = None  # the predicted output grid (None if program crashed)
    error: Optional[str] = None       # exception message if program crashed

    @property
    def verdict(self) -> str:
        """Four-way verdict per the v2 study §4.4."""
        if self.error:
            return "ERROR"
        if self.train_pass and self.manifested:
            return "SUBMIT"
        if self.train_pass and not self.manifested:
            return "MARGINAL"
        if not self.train_pass and self.manifested:
            return "CURIOSITY"
        return "DISCARD"

    def __repr__(self):
        if self.error:
            return f"RankResult({self.verdict}, error={self.error})"
        return (f"RankResult({self.verdict}, "
                f"nrci={self.nrci_refined:.4f}, {self.program})")


# ══════════════════════════════════════════════════════════════════════════════
# RANKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Ranker:
    """NRCI-based hypothesis ranker.

    Parameters
    ----------
    manifestation_threshold : float
        Refined NRCI ≥ this value counts as 'manifested'. Default 0.70,
        matching the UBP substrate's existing threshold.
    refine_engine : RefinedNRCI or None
        Pre-built RefinedNRCI instance. If None, builds one with the
        GOLAY_ENGINE on first use.
    """
    manifestation_threshold: float = 0.70
    refine_engine: Optional[RefinedNRCI] = None

    def __post_init__(self):
        if self.refine_engine is None:
            self.refine_engine = RefinedNRCI(golay_engine=GOLAY_ENGINE)

    def _score_grid(self, grid: Grid) -> Tuple[float, float, bool]:
        """Encode a grid, snap, and score with both basic and refined NRCI.
        Returns (nrci_basic, nrci_refined, manifested)."""
        v, report = encode_grid(grid)
        return (report.nrci_basic, report.nrci_refined, report.manifested)

    def _evaluate_program(self, program: Program, task: ARCTask) -> RankResult:
        """Apply program to test input, score output. Returns RankResult."""
        try:
            # Empirical adequacy check
            train_pass = program.matches_train(task)

            # Produce test output
            test_input = task.test[0].input
            test_output = program.apply(test_input)

            # Score output
            nrci_b, nrci_r, manifested = self._score_grid(test_output)

            return RankResult(
                program=program,
                train_pass=train_pass,
                nrci_basic=nrci_b,
                nrci_refined=nrci_r,
                manifested=manifested,
                test_output=test_output,
            )
        except Exception as e:
            return RankResult(
                program=program,
                train_pass=False,
                nrci_basic=0.0,
                nrci_refined=0.0,
                manifested=False,
                test_output=None,
                error=f"{type(e).__name__}: {e}",
            )

    def rank(self, task: ARCTask, candidates: List[Program]) -> List[RankResult]:
        """Rank all candidates. Returns sorted list (highest refined NRCI first).

        Sorting key (descending):
          1. train_pass (must reproduce all train pairs) — hard filter
          2. no error
          3. shorter program length (Occam's razor — among train-pass programs,
             prefer simpler ones; this prevents spurious 2-op compositions from
             beating the correct 1-op program when both pass train)
          4. refined NRCI (structural coherence tiebreak)
        """
        results = [self._evaluate_program(p, task) for p in candidates]
        results.sort(key=lambda r: (
            r.train_pass and r.error is None,
            r.error is None,
            -len(r.program),   # shorter programs first (negate for descending sort)
            r.nrci_refined,
        ), reverse=True)
        return results

    def top_k(self, task: ARCTask, candidates: List[Program], k: int = 3) -> List[RankResult]:
        """Return the top-k SUBMIT-or-MARGINAL candidates."""
        results = self.rank(task, candidates)
        return [r for r in results if r.verdict in ("SUBMIT", "MARGINAL")][:k]

    def best(self, task: ARCTask, candidates: List[Program]) -> Optional[RankResult]:
        """Return the single best candidate, or None if no candidates pass even the train filter."""
        results = self.rank(task, candidates)
        for r in results:
            if r.train_pass and r.error is None:
                return r
        return None


# ══════════════════════════════════════════════════════════════════════════════
# NULL-MODEL RANKER (for falsification Test 2)
# ══════════════════════════════════════════════════════════════════════════════

import random

@dataclass
class RandomRanker:
    """Uniform-random ranker — the Test 2 null model.

    Among programs that pass the empirical-adequacy filter, picks one uniformly
    at random. Used by the falsification protocol in Chapter 7 to test whether
    NRCI is doing real work.
    """
    seed: int = 42

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def rank(self, task: ARCTask, candidates: List[Program]) -> List[RankResult]:
        results = []
        for p in candidates:
            try:
                train_pass = p.matches_train(task)
                test_output = p.apply(task.test[0].input)
                # Uniform random score in [0.5, 1.0] — same range as NRCI but random
                rand_score = self._rng.uniform(0.5, 1.0)
                results.append(RankResult(
                    program=p, train_pass=train_pass,
                    nrci_basic=rand_score, nrci_refined=rand_score,
                    manifested=rand_score >= 0.70,
                    test_output=test_output,
                ))
            except Exception as e:
                results.append(RankResult(
                    program=p, train_pass=False,
                    nrci_basic=0.0, nrci_refined=0.0, manifested=False,
                    error=f"{type(e).__name__}: {e}",
                ))
        # Sort by random score, with train_pass tiebreak
        results.sort(key=lambda r: (r.train_pass, r.nrci_refined), reverse=True)
        return results

    def best(self, task: ARCTask, candidates: List[Program]) -> Optional[RankResult]:
        results = self.rank(task, candidates)
        for r in results:
            if r.train_pass and r.error is None:
                return r
        return None
