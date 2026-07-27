"""
per_object_24d.py — per-object 24D Leech addresses and hex colour codes
=========================================================================

The core reframe: information IS physical geometry. Each ARC cell gets:
  - A 24D Leech lattice address (via the Golay encoder)
  - A hex colour code (#RRGGBB — the 24-bit vector IS a colour)
  - A geometric signature (R(n), C(N), tension, reaction regime)

Instead of treating the grid as a flat 2D array of integers, we treat
each cell as a PHYSICAL OBJECT with a real position in 24D space. The
transformation between input and output is then a DISPLACEMENT in 24D
space, not a symbolic lookup.

The four conditions for structure:
  1. Reality: each cell has a spatial N-gon footprint (R(n))
  2. Information: each cell has a discrete bit-position topology
  3. Activation: operations are physical dynamics (cluster merging, etc.)
  4. Constraints: Golay snap corrects drift; NRCI measures coherence

NRCI is NOT a gate here — it's a COHERENCE MEASURE. We use it to
UNDERSTAND the transformation (how coherent is the result?), not to
FILTER candidates (accept/reject). Every candidate is kept; NRCI tells
us how much structure each one has.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from fractions import Fraction
from collections import defaultdict
import sys, os, math

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
_PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from arc_loader import Grid, ARCTask
from ubp_unified_v5 import (
    GOLAY_ENGINE, LEECH_ENGINE,
    ontological_position_to_vector, MOG_CATEGORIES,
)
from lingo.geometric_translator import (
    phi, sub_cycles, R_n, geometric_tension, analyze_reaction,
    compute_signature, GeometricSignature,
)
from lingo.ubp_integration import (
    nrci_fraction, ObserverDynamics, GENESIS_SEEDS, get_genesis_seed,
    R_n_fraction,
)


# ══════════════════════════════════════════════════════════════════════════════
# CELL ADDRESS — each ARC cell gets a 24D Leech address + hex colour
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CellAddress:
    """The 24D physical address of a single ARC cell.

    Each cell is not just an integer — it's a PHYSICAL OBJECT with:
      - position: (row, col) in the 2D grid
      - colour: the ARC palette value (0-9)
      - vector_24d: 24-bit Leech lattice address
      - hex_colour: #RRGGBB — the 24-bit vector IS a colour
      - nrci: coherence measure (NOT a gate — just a measurement)
      - genesis_seed: which UBP primitive this cell's shape matches
      - geometric_signature: R(n), C(N), tension, etc.
    """
    row: int
    col: int
    colour: int
    vector_24d: List[int] = field(default_factory=list)
    hex_colour: str = "#000000"
    nrci: Fraction = Fraction(0)
    nrci_float: float = 0.0
    coherence_label: str = "SUBLIMINAL"
    genesis_seed: Optional[str] = None
    geo_sig: Optional[GeometricSignature] = None

    def __repr__(self):
        return (f"CellAddress(r={self.row}, c={self.col}, colour={self.colour}, "
                f"hex={self.hex_colour}, nrci={self.nrci_float:.4f} [{self.coherence_label}])")


def assign_cell_address(row: int, col: int, colour: int,
                         grid_shape: Tuple[int, int]) -> CellAddress:
    """Assign a 24D Leech address to a single ARC cell.

    The address is computed from the cell's physical properties:
      - Bits 0-5 (Reality/Mirrors): colour fingerprint (M_Charge)
      - Bits 6-11 (Information): position in the grid (I_Topology)
      - Bits 12-17 (Activation): grid dimensions (A_Force)
      - Bits 18-23 (Potential): colour × position hash (P_Coherence)

    This grounds each cell in 24D space — it's not an abstract number
    but a physical object with a real lattice address.
    """
    h, w = grid_shape

    # Quadrant 1 (Mirrors, bits 0-5): colour fingerprint
    # Map colour (0-9) to a 6-bit Gray code
    colour_code = colour ^ (colour >> 1)  # Gray code
    palette_code = colour_code & 0x3F

    # Quadrant 2 (Information, bits 6-11): position topology
    # Encode (row, col) as a 6-bit position hash
    pos_code = ((row * 7 + col * 13) ^ (row * 3 + col * 5)) & 0x3F
    pos_code = pos_code ^ (pos_code >> 1)  # Gray code

    # Quadrant 3 (Activation, bits 12-17): grid dimensions
    # Encode (h, w) as a 6-bit dimension code
    dim_code = ((h & 0x07) << 3) | (w & 0x07)
    dim_code = dim_code ^ (dim_code >> 1)  # Gray code

    # Quadrant 4 (Potential, bits 18-23): colour × position coherence
    # A hash that captures the relationship between colour and position
    coh_code = ((colour * 31 + row * 17 + col * 23) % 64) & 0x3F
    coh_code = coh_code ^ (coh_code >> 1)  # Gray code

    # Pack into 24-bit integer
    position_24bit = (
        (palette_code << 18) |
        (pos_code << 12) |
        (dim_code << 6) |
        coh_code
    )

    # Convert to 24-bit vector via the UBP pipeline
    vector = ontological_position_to_vector(position_24bit)

    # Snap to Golay codeword
    snapped, _ = GOLAY_ENGINE.snap_to_codeword(vector)

    # Compute NRCI as a Fraction (coherence measure, NOT a gate)
    nrci = nrci_fraction(snapped)
    nrci_f = float(nrci)

    # Classify coherence (measurement, not filter)
    coherence = ObserverDynamics.classify(nrci)

    # Hex colour (the 24-bit vector IS a colour)
    try:
        from GLM18_hex_colour import vector_to_colour
        hex_c = vector_to_colour(vector)
    except ImportError:
        n = 0
        for i, bit in enumerate(vector):
            if bit:
                n |= (1 << (23 - i))
        hex_c = f"#{n:06x}"

    # Genesis seed (which UBP primitive does this cell match?)
    seed = get_genesis_seed(colour) if colour > 0 else None

    # Geometric signature
    geo_sig = compute_signature(colour) if colour > 0 else None

    return CellAddress(
        row=row, col=col, colour=colour,
        vector_24d=vector,
        hex_colour=hex_c,
        nrci=nrci,
        nrci_float=nrci_f,
        coherence_label=coherence,
        genesis_seed=seed["seed"] if seed else None,
        geo_sig=geo_sig,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ADDRESSED GRID — a grid where every cell has a 24D address
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AddressedGrid:
    """A grid where every cell has a 24D Leech address.

    This is the physical-geometry view of an ARC grid: not a 2D array
    of integers, but a 2D arrangement of physical objects in 24D space.
    """
    grid: Grid
    addresses: List[List[CellAddress]] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def cell_at(self, r: int, c: int) -> CellAddress:
        return self.addresses[r][c]

    def all_addresses(self) -> List[CellAddress]:
        return [addr for row in self.addresses for addr in row]

    def colour_addresses(self, colour: int) -> List[CellAddress]:
        """Get all cells of a specific colour."""
        return [addr for row in self.addresses for addr in row if addr.colour == colour]

    def mean_nrci(self) -> float:
        """Mean NRCI across all cells — a coherence measure of the whole grid."""
        addrs = self.all_addresses()
        if not addrs:
            return 0.0
        return sum(a.nrci_float for a in addrs) / len(addrs)

    def coherence_distribution(self) -> Dict[str, int]:
        """Distribution of coherence labels across all cells."""
        dist = defaultdict(int)
        for addr in self.all_addresses():
            dist[addr.coherence_label] += 1
        return dict(dist)

    def hex_palette(self) -> Set[str]:
        """The set of unique hex colours in the grid."""
        return {addr.hex_colour for addr in self.all_addresses() if addr.colour > 0}


def address_grid(grid: Grid) -> AddressedGrid:
    """Assign 24D Leech addresses to every cell in a grid."""
    h, w = grid.shape
    addresses = []
    for r in range(h):
        row_addrs = []
        for c in range(w):
            addr = assign_cell_address(r, c, grid.cells[r][c], grid.shape)
            row_addrs.append(addr)
        addresses.append(row_addrs)
    return AddressedGrid(grid=grid, addresses=addresses)


# ══════════════════════════════════════════════════════════════════════════════
# PER-OBJECT TRANSFORMER — generate transformations per object
# ══════════════════════════════════════════════════════════════════════════════

class PerObjectTransformer:
    """Transforms each object independently based on its 24D address.

    The key reframe: instead of applying one transformation to the whole
    grid, we look at each object's 24D address and determine what
    transformation IT needs, based on:
      1. Its hex colour (which colours map to which)
      2. Its geometric signature (R(n), C(N), tension)
      3. Its coherence (NRCI — as a measurement, not a gate)
      4. The learned CRG edge for its colour

    This is the "two ways to solve" insight: the same task can be solved
    by treating it as a grid-level operation OR as a per-object operation.
    The per-object view often reveals structure the grid-level view misses.
    """

    def __init__(self):
        from generative.object_crg import ObjectCRG
        self.crg = ObjectCRG()

    def learn_from_task(self, task: ARCTask) -> None:
        """Learn per-object transformations from train pairs."""
        self.crg.learn_from_task(task)

        # Also learn per-COLOUR transformations by comparing
        # input and output cell addresses
        self.colour_transforms: Dict[int, Dict[str, Any]] = {}
        for pair in task.train:
            in_addrs = address_grid(pair.input)
            out_addrs = address_grid(pair.output)
            if in_addrs.shape != out_addrs.shape:
                continue  # shape changed — can't do cell-level comparison
            for r in range(in_addrs.shape[0]):
                for c in range(in_addrs.shape[1]):
                    in_addr = in_addrs.cell_at(r, c)
                    out_addr = out_addrs.cell_at(r, c)
                    if in_addr.colour != out_addr.colour:
                        # This cell's colour changed
                        old = in_addr.colour
                        new = out_addr.colour
                        if old not in self.colour_transforms:
                            self.colour_transforms[old] = {
                                "target_colours": defaultdict(int),
                                "hex_changes": [],
                                "nrci_changes": [],
                            }
                        self.colour_transforms[old]["target_colours"][new] += 1
                        self.colour_transforms[old]["hex_changes"].append(
                            (in_addr.hex_colour, out_addr.hex_colour)
                        )
                        self.colour_transforms[old]["nrci_changes"].append(
                            (in_addr.nrci_float, out_addr.nrci_float)
                        )

        # Resolve each colour's most common target
        self.resolved_mapping: Dict[int, int] = {}
        for old, info in self.colour_transforms.items():
            if info["target_colours"]:
                most_common = max(info["target_colours"],
                                  key=info["target_colours"].get)
                self.resolved_mapping[old] = most_common

    def predict(self, task: ARCTask) -> Grid:
        """Predict the test output by applying per-object transformations.

        For each cell in the test input:
          1. Look up its colour in the resolved mapping
          2. If a mapping exists, apply it
          3. If not, keep the original colour
        """
        test_input = task.test[0].input
        h, w = test_input.shape

        # Apply the resolved colour mapping
        out_cells = []
        for r in range(h):
            row = []
            for c in range(w):
                old = test_input.cells[r][c]
                if old in self.resolved_mapping:
                    row.append(self.resolved_mapping[old])
                else:
                    row.append(old)
            out_cells.append(row)

        return Grid(out_cells)

    def predict_with_addresses(self, task: ARCTask) -> Tuple[Grid, AddressedGrid]:
        """Predict AND return the addressed version of the prediction.

        This lets us inspect the 24D addresses of the predicted output,
        measuring its coherence (NRCI) as a quality indicator — NOT a gate.
        """
        predicted = self.predict(task)
        addressed = address_grid(predicted)
        return predicted, addressed

    def summary(self) -> str:
        """Summarise what was learned."""
        lines = [
            f"PerObjectTransformer summary:",
            f"  CRG edges: {len(self.crg.all_edges)}",
            f"  CRG dominant: {self.crg.dominant_transform_type()}",
            f"  Colour transforms: {len(self.colour_transforms)} colours tracked",
            f"  Resolved mapping: {self.resolved_mapping}",
        ]
        if self.colour_transforms:
            for old, info in sorted(self.colour_transforms.items()):
                targets = dict(info["target_colours"])
                lines.append(f"    colour {old} → {targets}")
                if info["nrci_changes"]:
                    avg_in = sum(n[0] for n in info["nrci_changes"]) / len(info["nrci_changes"])
                    avg_out = sum(n[1] for n in info["nrci_changes"]) / len(info["nrci_changes"])
                    lines.append(f"      NRCI: {avg_in:.4f} → {avg_out:.4f} (Δ={avg_out-avg_in:+.4f})")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# DUAL SOLVER — solve a task both ways and compare
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DualSolution:
    """The result of solving a task two different ways."""
    task_id: str
    # Grid-level solution (the UBPActionEngine approach)
    grid_level_pred: Optional[Grid] = None
    grid_level_nrci: float = 0.0
    # Per-object solution (the PerObjectTransformer approach)
    per_object_pred: Optional[Grid] = None
    per_object_nrci: float = 0.0
    # Addressed grid info
    input_coherence: float = 0.0
    output_coherence_per_object: float = 0.0
    output_coherence_grid_level: float = 0.0
    # Which approach was correct?
    grid_level_correct: Optional[bool] = None
    per_object_correct: Optional[bool] = None
    # Which approach was chosen?
    chosen: str = ""  # "grid_level" or "per_object"
    chosen_correct: Optional[bool] = None

    def summary(self) -> str:
        lines = [
            f"DualSolution for {self.task_id}:",
            f"  Grid-level:  NRCI={self.grid_level_nrci:.4f}  correct={self.grid_level_correct}",
            f"  Per-object:  NRCI={self.per_object_nrci:.4f}  correct={self.per_object_correct}",
            f"  Input coherence: {self.input_coherence:.4f}",
            f"  Chosen: {self.chosen}  correct={self.chosen_correct}",
        ]
        if self.grid_level_correct and self.per_object_correct:
            lines.append("  ⚡ BOTH methods solve it — structural insight here!")
        elif self.grid_level_correct and not self.per_object_correct:
            lines.append("  → Grid-level wins (per-object misses structure)")
        elif not self.grid_level_correct and self.per_object_correct:
            lines.append("  → Per-object wins (grid-level misses per-object detail)")
        else:
            lines.append("  ✗ Neither method solves it")
        return "\n".join(lines)


def solve_dual(task: ARCTask) -> DualSolution:
    """Solve a task using both grid-level and per-object approaches.

    The 'two ways to solve' insight: comparing the two approaches
    reveals WHERE the structure lives — at the grid level or the
    per-object level. This is a clue for improving the system.
    """
    result = DualSolution(task_id=task.name)

    # Address the test input (measure its coherence)
    input_addrs = address_grid(task.test[0].input)
    result.input_coherence = input_addrs.mean_nrci()

    # Method 1: Grid-level (UBPActionEngine)
    from generative.ubp_action_engine import UBPActionEngine
    grid_engine = UBPActionEngine()
    result.grid_level_pred = grid_engine.solve(task)
    if result.grid_level_pred:
        _, enc = encode_grid(result.grid_level_pred)
        result.grid_level_nrci = enc.nrci_refined

    # Method 2: Per-object (PerObjectTransformer)
    obj_engine = PerObjectTransformer()
    obj_engine.learn_from_task(task)
    result.per_object_pred, out_addrs = obj_engine.predict_with_addresses(task)
    result.per_object_nrci = out_addrs.mean_nrci()
    result.output_coherence_per_object = out_addrs.mean_nrci()

    # Check correctness
    expected = task.test[0].expected_output
    if expected is not None:
        result.grid_level_correct = (result.grid_level_pred == expected)
        result.per_object_correct = (result.per_object_pred == expected)

    # Choose: prefer the per-object solution if it reproduces train pairs
    # (per-object is more precise; grid-level is more general)
    # Verify per-object prediction against train pairs
    per_obj_passes = True
    for pair in task.train:
        if pair.input.shape == pair.output.shape:
            test_mapping = obj_engine.resolved_mapping
            reconstructed = Grid([
                [test_mapping.get(v, v) for v in row]
                for row in pair.input.cells
            ])
            if reconstructed != pair.output:
                per_obj_passes = False
                break

    if per_obj_passes and result.per_object_pred:
        result.chosen = "per_object"
        result.chosen_correct = result.per_object_correct
    elif result.grid_level_pred:
        result.chosen = "grid_level"
        result.chosen_correct = result.grid_level_correct

    return result


# Need encode_grid import
from encoder import encode_grid
