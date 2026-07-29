"""
geometric_grammar.py — the language of geometry, Part 2
========================================================

The user's insight: "For language we can use the layers to define
noun/verb, object/action, duration, gates and then probably more on
the per-bit layers".

This module maps the MOG layers to linguistic categories:

  NOUN     = a cell with stable colour + position
             (Mirrors + Information quadrants stable)
  VERB     = a transformation between two grids
             (the delta between input and output)
  OBJECT   = a connected region of nouns
             (cells that move/transform together)
  ACTION   = a verb applied to an object
             (the actual transformation event)
  DURATION = the number of Time steps
             (how many train pairs = how many Time steps)
  GATE     = a thoughtful stop
             (NRCI threshold, hard-gate verification)

Plus per-bit linguistic categories (from MOG meaning encoder):

  BIT_NOUN = a bit with stable value across train
  BIT_VERB = a bit that flips between input and output
  BIT_TONE = the bit's MOG quadrant (Mirrors/Info/Act/Pot)

The grammar lets the GLM "speak" about a task in structured language:

  "OBJECT_1 (a 3x3 region of colour 2 at position (1,1))
   underwent ACTION (recolour 2→5)
   over DURATION 1 (one Time step)
   passing GATE_0.7 (NRCI 0.83 ≥ 0.7)
   and GATE_TRAIN (hard gate passed)."

This is what "language machine, not script pipeline" means.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
from fractions import Fraction
import sys, os, math

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from arc_loader import Grid, ARCTask
from generative.hex_learner import address_cell, address_grid, HexCell
from vendor.mog_meaning_encoder import (
    decode_meaning, CellMeaning, meaning_distance, meaning_similarity,
    BitAddress, MOG_QUADRANTS,
)


# ══════════════════════════════════════════════════════════════════════════════
# NOUN — a cell with stable identity
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Noun:
    """A noun — a cell with stable colour and position.

    A noun is identified by:
      - position (row, col)
      - colour
      - MOG meaning signature (576-dim identity)
      - coherence (NRCI)
    """
    row: int
    col: int
    colour: int
    meaning: CellMeaning
    nrci: float = 0.0

    def name(self) -> str:
        """A linguistic name for this noun."""
        return f"cell_{self.colour}_at_{self.row}_{self.col}"

    def __repr__(self):
        return f"Noun({self.row},{self.col},c={self.colour},nrci={self.nrci:.2f})"


def extract_nouns(grid: Grid) -> List[List[Noun]]:
    """Extract all nouns from a grid."""
    from vendor.per_cell_coherence import cell_nrci
    addrs = address_grid(grid)
    nouns = []
    for r in range(grid.height):
        row = []
        for c in range(grid.width):
            cell = addrs[r][c]
            meaning = decode_meaning(cell.vector, r, c, cell.colour)
            nrci = cell_nrci(cell).nrci_float
            row.append(Noun(row=r, col=c, colour=cell.colour,
                            meaning=meaning, nrci=nrci))
        nouns.append(row)
    return nouns


# ══════════════════════════════════════════════════════════════════════════════
# VERB — a transformation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Verb:
    """A verb — a transformation between two grids.

    A verb is described by:
      - rotation: the D4 rotation applied (or "identity")
      - colour_mapping: {old_colour: new_colour}
      - per_cell_deltas: list of (input_address, output_address) per cell
      - dominant_delta: the most common cell delta
      - delta_confidence: fraction of cells with the dominant delta
    """
    rotation: str = "identity"
    colour_mapping: Dict[int, int] = field(default_factory=dict)
    dominant_delta: int = 0
    delta_confidence: float = 0.0
    n_cells_changed: int = 0
    n_cells_unchanged: int = 0

    def name(self) -> str:
        """A linguistic name for this verb."""
        parts = []
        if self.rotation != "identity":
            parts.append(self.rotation)
        if self.colour_mapping:
            cm = ", ".join(f"{k}→{v}" for k, v in sorted(self.colour_mapping.items()))
            parts.append(f"recolour({cm})")
        if not parts:
            return "identity"
        return " + ".join(parts)

    def __repr__(self):
        return f"Verb({self.name()}, conf={self.delta_confidence:.2f})"


def extract_verb(grid_in: Grid, grid_out: Grid) -> Optional[Verb]:
    """Extract the verb (transformation) from an input → output pair."""
    if grid_in.shape != grid_out.shape:
        return None

    # Rotation
    from generative.geometric_language import infer_rotation
    rot = infer_rotation(grid_in, grid_out) or "identity"

    # Colour mapping
    colour_targets: Dict[int, List[int]] = defaultdict(list)
    for r in range(grid_in.height):
        for c in range(grid_in.width):
            old = grid_in.cells[r][c]
            new = grid_out.cells[r][c]
            if old != new:
                colour_targets[old].append(new)
    colour_mapping = {}
    for old, targets in colour_targets.items():
        colour_mapping[old] = Counter(targets).most_common(1)[0][0]

    # Per-cell deltas
    in_addrs = address_grid(grid_in)
    out_addrs = address_grid(grid_out)
    deltas = []
    n_changed = 0
    n_unchanged = 0
    for r in range(grid_in.height):
        for c in range(grid_in.width):
            delta = in_addrs[r][c].address_int ^ out_addrs[r][c].address_int
            deltas.append(delta)
            if grid_in.cells[r][c] != grid_out.cells[r][c]:
                n_changed += 1
            else:
                n_unchanged += 1

    # Dominant delta
    if deltas:
        delta_counts = Counter(deltas)
        dominant_delta, top_count = delta_counts.most_common(1)[0]
        confidence = top_count / len(deltas)
    else:
        dominant_delta = 0
        confidence = 0.0

    return Verb(
        rotation=rot, colour_mapping=colour_mapping,
        dominant_delta=dominant_delta, delta_confidence=confidence,
        n_cells_changed=n_changed, n_cells_unchanged=n_unchanged,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT — a connected region of nouns
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GObject:
    """An object — a connected region of nouns that transform together.

    Identified by:
      - cells: list of (row, col) positions
      - colour: the dominant colour
      - bbox: bounding box (r_min, c_min, r_max, c_max)
      - area: number of cells
      - centre: (cr, cc) centre of mass
    """
    cells: List[Tuple[int, int]] = field(default_factory=list)
    colour: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    area: int = 0
    centre: Tuple[float, float] = (0.0, 0.0)

    def name(self) -> str:
        return f"object_{self.colour}_area_{self.area}"

    def __repr__(self):
        return f"GObject(c={self.colour}, area={self.area}, bbox={self.bbox})"


def extract_objects(grid: Grid) -> List[GObject]:
    """Extract all objects (4-connected regions of same non-zero colour)."""
    h, w = grid.shape
    visited = [[False] * w for _ in range(h)]
    objects = []

    for r in range(h):
        for c in range(w):
            if visited[r][c] or grid.cells[r][c] == 0:
                continue
            # BFS for connected region of same colour
            colour = grid.cells[r][c]
            cells = []
            stack = [(r, c)]
            r_min = r_max = r
            c_min = c_max = c
            while stack:
                cr, cc = stack.pop()
                if (cr < 0 or cr >= h or cc < 0 or cc >= w
                        or visited[cr][cc] or grid.cells[cr][cc] != colour):
                    continue
                visited[cr][cc] = True
                cells.append((cr, cc))
                r_min = min(r_min, cr)
                r_max = max(r_max, cr)
                c_min = min(c_min, cc)
                c_max = max(c_max, cc)
                stack.extend([(cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)])

            if cells:
                cr = sum(r for r, _ in cells) / len(cells)
                cc = sum(c for _, c in cells) / len(cells)
                objects.append(GObject(
                    cells=cells, colour=colour,
                    bbox=(r_min, c_min, r_max, c_max),
                    area=len(cells), centre=(cr, cc),
                ))
    return objects


# ══════════════════════════════════════════════════════════════════════════════
# ACTION — a verb applied to an object
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Action:
    """An action — a verb applied to a specific object.

    Describes: "OBJECT underwent VERB"
    """
    object_in: GObject
    object_out: Optional[GObject]
    verb: Verb

    def name(self) -> str:
        return f"{self.object_in.name()} → {self.verb.name()}"

    def __repr__(self):
        return f"Action({self.name()})"


def extract_actions(grid_in: Grid, grid_out: Grid) -> List[Action]:
    """Extract all actions (object-level transformations) from a pair."""
    if grid_in.shape != grid_out.shape:
        return []
    verb = extract_verb(grid_in, grid_out)
    if verb is None:
        return []
    objects_in = extract_objects(grid_in)
    objects_out = extract_objects(grid_out)
    actions = []
    # Match objects by colour and approximate position
    for o_in in objects_in:
        # Find the closest object in output
        best_match = None
        best_dist = float("inf")
        for o_out in objects_out:
            if o_out.colour != o_in.colour and not verb.colour_mapping:
                continue
            # Check if colours match (after mapping)
            mapped_colour = verb.colour_mapping.get(o_in.colour, o_in.colour)
            if o_out.colour != mapped_colour:
                continue
            dist = ((o_in.centre[0] - o_out.centre[0]) ** 2 +
                    (o_in.centre[1] - o_out.centre[1]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_match = o_out
        actions.append(Action(object_in=o_in, object_out=best_match, verb=verb))
    return actions


# ══════════════════════════════════════════════════════════════════════════════
# DURATION — number of Time steps
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Duration:
    """Duration — the number of Time steps in a task."""
    n_train_pairs: int
    n_time_steps: int  # = n_train_pairs (each pair is one step)

    def name(self) -> str:
        return f"duration_{self.n_time_steps}"


def extract_duration(task: ARCTask) -> Duration:
    return Duration(n_train_pairs=len(task.train), n_time_steps=len(task.train))


# ══════════════════════════════════════════════════════════════════════════════
# GATE — a thoughtful stop
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Gate:
    """A gate — a thoughtful stop in the pipeline.

    Types of gates:
      - COHERENCE: NRCI >= threshold (default 0.7)
      - TRAIN_PASS: exact train-pair reproduction
      - RHYTHM_MATCH: train/test rhythm alignment >= threshold
      - SMELL_MATCH: train/test smell similarity >= threshold
    """
    name: str
    threshold: float
    value: float
    passed: bool

    def name_str(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name}({self.value:.2f} {'>=' if self.passed else '<'} {self.threshold}) [{status}]"


def check_coherence_gate(nrci: float, threshold: float = 0.7) -> Gate:
    passed = nrci >= threshold
    return Gate(name="coherence", threshold=threshold, value=nrci, passed=passed)


def check_train_gate(passes: bool) -> Gate:
    return Gate(name="train_pass", threshold=1.0, value=1.0 if passes else 0.0, passed=passes)


def check_rhythm_gate(match_score: float, threshold: float = 0.5) -> Gate:
    passed = match_score >= threshold
    return Gate(name="rhythm_match", threshold=threshold, value=match_score, passed=passed)


def check_smell_gate(similarity: float, threshold: float = 0.7) -> Gate:
    passed = similarity >= threshold
    return Gate(name="smell_match", threshold=threshold, value=similarity, passed=passed)


# ══════════════════════════════════════════════════════════════════════════════
# Per-bit linguistic categories
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BitLinguistics:
    """Per-bit linguistic categories.

    For each of the 24 bits of a cell's address:
      - BIT_NOUN: bit has stable value across all train pairs
      - BIT_VERB: bit flips between input and output
      - BIT_TONE: the bit's MOG quadrant (Mirrors/Info/Act/Pot)
    """
    bit_nouns: List[bool] = field(default_factory=list)  # 24 bools
    bit_verbs: List[bool] = field(default_factory=list)  # 24 bools
    bit_tones: List[str] = field(default_factory=list)   # 24 quadrant names


def extract_bit_linguistics(grid_in: Grid, grid_out: Grid,
                              other_train_pairs: List[Tuple[Grid, Grid]] = None
                              ) -> BitLinguistics:
    """Extract per-bit linguistic categories for a cell.

    For each bit position 0-23:
      - BIT_NOUN: the bit value is the same in input and output (stable)
      - BIT_VERB: the bit value differs between input and output (flipped)
      - BIT_TONE: the MOG quadrant of the bit
    """
    if grid_in.shape != grid_out.shape:
        return BitLinguistics()

    # Use the first cell as representative
    in_addr = address_cell(0, 0, grid_in.cells[0][0], grid_in.height, grid_in.width)
    out_addr = address_cell(0, 0, grid_out.cells[0][0], grid_out.height, grid_out.width)

    bit_nouns = []
    bit_verbs = []
    bit_tones = []
    for i in range(24):
        in_bit = in_addr.vector[i]
        out_bit = out_addr.vector[i]
        is_noun = (in_bit == out_bit)
        is_verb = (in_bit != out_bit)
        quadrant = MOG_QUADRANTS[i // 6]["name"] if i < 24 else "?"
        # Adjust for actual quadrant (bits 0-5, 6-11, 12-17, 18-23)
        q_idx = i // 6
        if q_idx < 4:
            tone = MOG_QUADRANTS[q_idx]["name"]
        else:
            tone = "?"
        bit_nouns.append(is_noun)
        bit_verbs.append(is_verb)
        bit_tones.append(tone)

    return BitLinguistics(bit_nouns=bit_nouns, bit_verbs=bit_verbs, bit_tones=bit_tones)


# ══════════════════════════════════════════════════════════════════════════════
# Full linguistic description of a task
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskDescription:
    """A full linguistic description of an ARC task.

    Combines nouns, verbs, objects, actions, duration, and gates.
    This is what the GLM "says" about the task.
    """
    nouns_in: List[Noun] = field(default_factory=list)
    nouns_out: List[Noun] = field(default_factory=list)
    verb: Optional[Verb] = None
    objects_in: List[GObject] = field(default_factory=list)
    objects_out: List[GObject] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    duration: Optional[Duration] = None
    gates: List[Gate] = field(default_factory=list)
    bit_linguistics: Optional[BitLinguistics] = None

    def describe(self) -> str:
        """Generate a natural-language description of the task."""
        lines = ["Task Description:"]
        if self.duration:
            lines.append(f"  DURATION: {self.duration.name()}")

        if self.verb:
            lines.append(f"  VERB: {self.verb.name()}")
            lines.append(f"    confidence: {self.verb.delta_confidence:.2f}")
            lines.append(f"    cells changed: {self.verb.n_cells_changed}")
            lines.append(f"    cells unchanged: {self.verb.n_cells_unchanged}")

        if self.objects_in:
            lines.append(f"  OBJECTS_IN: {len(self.objects_in)}")
            for o in self.objects_in[:3]:
                lines.append(f"    {o}")

        if self.actions:
            lines.append(f"  ACTIONS: {len(self.actions)}")
            for a in self.actions[:3]:
                lines.append(f"    {a.name()}")

        if self.gates:
            lines.append(f"  GATES:")
            for g in self.gates:
                lines.append(f"    {g.name_str()}")

        if self.bit_linguistics:
            n_nouns = sum(self.bit_linguistics.bit_nouns)
            n_verbs = sum(self.bit_linguistics.bit_verbs)
            lines.append(f"  BIT LINGUISTICS: {n_nouns} nouns, {n_verbs} verbs")

        return "\n".join(lines)


def describe_task(task: ARCTask) -> TaskDescription:
    """Build a full linguistic description of an ARC task."""
    desc = TaskDescription()

    # Duration
    desc.duration = extract_duration(task)

    # Use the first train pair for verb/actions/bit_linguistics
    if task.train:
        pair = task.train[0]
        desc.verb = extract_verb(pair.input, pair.output)
        desc.objects_in = extract_objects(pair.input)
        desc.objects_out = extract_objects(pair.output)
        desc.actions = extract_actions(pair.input, pair.output)
        desc.bit_linguistics = extract_bit_linguistics(pair.input, pair.output)

    return desc


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

def _self_test():
    print("Geometric Grammar self-test")
    print("=" * 60)

    from arc_loader import TrainPair, TestInput

    # A simple recolour task
    inp = Grid([[1, 2, 0], [1, 2, 0], [1, 2, 0]])
    out = Grid([[2, 3, 0], [2, 3, 0], [2, 3, 0]])
    test = Grid([[1, 2, 0], [1, 2, 0]])
    task = ARCTask(name="recolour",
                   train=[TrainPair(input=inp, output=out)],
                   test=[TestInput(input=test, expected_output=Grid([[2, 3, 0], [2, 3, 0]]))])

    desc = describe_task(task)
    print(desc.describe())

    # Test gates
    print("\n[Gates]")
    g1 = check_coherence_gate(0.85, threshold=0.7)
    g2 = check_coherence_gate(0.45, threshold=0.7)
    g3 = check_train_gate(True)
    print(f"  {g1.name_str()}")
    print(f"  {g2.name_str()}")
    print(f"  {g3.name_str()}")

    print("\n" + "=" * 60)
    print("Self-test complete.")


if __name__ == "__main__":
    _self_test()
