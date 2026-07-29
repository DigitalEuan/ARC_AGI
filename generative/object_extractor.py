"""
object_extractor.py — decompose grids into objects (the GLM's "words")
=========================================================================

The key reframe: a grid is a SENTENCE, objects are WORDS.

Each connected component (same-colour, 8-neighbour) is an "object" — a word
in the GLM's vocabulary. Each object gets a 24-bit vector (its "word vector")
via the existing encoder, but applied per-object instead of per-grid.

This is the foundation of the generative approach:
  - The CRG learns object-to-object transformations (not grid-to-grid)
  - The Φ-grammar generates transformations that apply to objects
  - NRCI ranks the coherence of the reassembled output

The existing DSL ops become "known words" the GLM can reference — they're
the lingo it's learned from the v0.1-v0.4 work. Failures from those versions
are boundary discoveries that teach the CRG what doesn't work.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
import sys, os

# Make packages importable
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
_PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from arc_loader import Grid
from encoder.arc_to_24bit import (
    _palette_to_6bit, _cardinality_to_6bit, _spatial_anchor_to_6bit,
    _relational_to_6bit, _count_objects, _dominant_object_bbox,
    _dominant_object_centroid, _totient_subcycles,
)
from ubp_unified_v5 import (
    GOLAY_ENGINE, LEECH_ENGINE,
    ontological_position_to_vector, MOG_CATEGORIES,
)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT — a single connected component, encoded as a 24-bit "word vector"
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GridObject:
    """A single connected component within a grid.

    This is the GLM's "word" — the atomic unit of meaning. Each object has:
      - cells: the (row, col) positions it occupies
      - colour: the integer colour (1-9)
      - bbox: bounding box (rmin, rmax, cmin, cmax)
      - centroid: (row, col) centre of mass
      - vector: 24-bit UBP vector (its "word vector")
      - colour_hex: the #RRGGBB hex colour of its vector
      - nrci: refined NRCI score (coherence of this object's vector)
    """
    cells: List[Tuple[int, int]]
    colour: int
    grid_shape: Tuple[int, int]           # the parent grid's dimensions
    bbox: Tuple[int, int, int, int]       # rmin, rmax, cmin, cmax
    centroid: Tuple[float, float]
    vector: List[int] = field(default_factory=list)
    colour_hex: str = "#000000"
    nrci_basic: float = 0.0
    nrci_refined: float = 0.0

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    # Compatibility properties for the full GenerativeTransformer
    # which uses r1/c1/r2/c2 instead of bbox=(rmin, rmax, cmin, cmax)
    @property
    def r1(self) -> int:
        return self.bbox[0]  # rmin
    @property
    def r2(self) -> int:
        return self.bbox[1]  # rmax
    @property
    def c1(self) -> int:
        return self.bbox[2]  # cmin
    @property
    def c2(self) -> int:
        return self.bbox[3]  # cmax

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[2] + 1

    @property
    def height(self) -> int:
        return self.bbox[1] - self.bbox[0] + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def fill_ratio(self) -> float:
        """Fraction of bbox that's filled (cell_count / area)."""
        return self.cell_count / max(self.area, 1)

    def __repr__(self):
        return (f"GridObject(colour={self.colour}, cells={self.cell_count}, "
                f"bbox={self.bbox}, nrci={self.nrci_refined:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT EXTRACTION — decompose a grid into connected components
# ══════════════════════════════════════════════════════════════════════════════

def extract_objects(grid: Grid, min_size: int = 1) -> List[GridObject]:
    """Decompose a grid into connected components (8-neighbour adjacency).

    Each connected component of the same colour becomes a GridObject.
    Background (colour 0) is NOT extracted as objects.

    Parameters
    ----------
    grid : Grid
        The grid to decompose.
    min_size : int
        Minimum cell count for an object (1 = single cells count).

    Returns
    -------
    List[GridObject]
        All objects in the grid, ordered by (colour, size descending).
    """
    h, w = grid.shape
    seen = [[False] * w for _ in range(h)]
    objects: List[GridObject] = []

    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid.cells[r][c] == 0:
                continue
            # BFS flood fill for this colour
            colour = grid.cells[r][c]
            cells: List[Tuple[int, int]] = []
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and not seen[nr][nc]
                                and grid.cells[nr][nc] == colour):
                            seen[nr][nc] = True
                            stack.append((nr, nc))

            if len(cells) < min_size:
                continue

            # Compute bbox and centroid
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            bbox = (min(rs), max(rs), min(cs), max(cs))
            centroid = (sum(rs) / len(rs), sum(cs) / len(cs))

            obj = GridObject(
                cells=cells,
                colour=colour,
                grid_shape=grid.shape,
                bbox=bbox,
                centroid=centroid,
            )
            # Encode this object as a 24-bit vector
            _encode_object(obj, grid)
            objects.append(obj)

    # Sort by colour, then by size (largest first)
    objects.sort(key=lambda o: (o.colour, -o.cell_count))
    return objects


def _encode_object(obj: GridObject, parent_grid: Grid) -> None:
    """Encode a single object as a 24-bit vector.

    Uses the same MOG_CATEGORIES bit-budget as the grid encoder, but
    computes features per-object instead of per-grid:
      bits 0-5:   colour (the object's colour, Gray-coded)
      bits 6-11:  size bucket (log-scaled cell count)
      bits 12-17: spatial anchor (centroid + bbox, relative to parent grid)
      bits 18-23: shape signature (fill ratio + aspect + topology)
    """
    # Quadrant 1 (Mirrors, bits 0-5): colour fingerprint
    palette = frozenset({obj.colour})
    palette_code = _palette_to_6bit(palette)

    # Quadrant 2 (Information, bits 6-11): size bucket
    cardinality_code = _cardinality_to_6bit(obj.cell_count)

    # Quadrant 3 (Activation, bits 12-17): spatial anchor
    # Use the object's centroid relative to the parent grid centre
    h, w = obj.grid_shape
    grid_cr, grid_cc = (h - 1) / 2, (w - 1) / 2
    cr, cc = obj.centroid
    quadrant = (2 if cr > grid_cr else 0) + (1 if cc > grid_cc else 0)

    # Spatial radius via R(n) — n = cell count
    from spatial_arithmetic_compat import value_to_radius, radius_to_value
    n_cells = obj.cell_count
    if n_cells >= 3:
        R_n = value_to_radius(n_cells)
        k_scalar = radius_to_value(R_n)
        radius_bucket = k_scalar % 4
    else:
        radius_bucket = min(3, n_cells)

    # Aspect
    bh = obj.height
    bw = obj.width
    ratio = bw / max(bh, 1)
    if 0.67 <= ratio <= 1.5:
        aspect = 0
    elif ratio > 1.5 and ratio <= 3.0:
        aspect = 1
    elif ratio < 0.67 and ratio >= 0.33:
        aspect = 2
    else:
        aspect = 3

    anchor_code = ((quadrant << 4) | (radius_bucket << 2) | aspect)
    anchor_code = (anchor_code ^ (anchor_code >> 1)) & 0x3F  # Gray code

    # Quadrant 4 (Potential, bits 18-23): shape signature
    fill = obj.fill_ratio
    if fill < 0.3:
        fill_bucket = 0  # sparse
    elif fill < 0.6:
        fill_bucket = 1  # partial
    elif fill < 0.9:
        fill_bucket = 2  # mostly full
    else:
        fill_bucket = 3  # solid

    # Topology: totient sub-cycles of the cell count
    subcycles = _totient_subcycles(n_cells)
    topo_bucket = subcycles & 0x07  # 3 bits

    relational_code = ((fill_bucket & 0x07) << 3) | topo_bucket
    relational_code = (relational_code ^ (relational_code >> 1)) & 0x3F

    # Pack into 24-bit integer
    position_24bit = (
        (palette_code      << 18) |
        (cardinality_code  << 12) |
        (anchor_code         << 6) |
        relational_code
    )

    # Use the existing UBP pipeline
    vector = ontological_position_to_vector(position_24bit)
    obj.vector = vector

    # Golay snap + NRCI
    snapped, _ = GOLAY_ENGINE.snap_to_codeword(vector)
    obj.nrci_basic = float(LEECH_ENGINE.calculate_nrci(snapped))

    # Refined NRCI
    try:
        from refined_nrci import RefinedNRCI
        rnrci = RefinedNRCI(golay_engine=GOLAY_ENGINE)
        obj.nrci_refined = float(rnrci.compute([float(x) for x in snapped]))
    except ImportError:
        obj.nrci_refined = obj.nrci_basic

    # Hex colour
    try:
        from GLM18_hex_colour import vector_to_colour
        obj.colour_hex = vector_to_colour(vector)
    except ImportError:
        n = 0
        for i, bit in enumerate(vector):
            if bit:
                n |= (1 << (23 - i))
        obj.colour_hex = f"#{n:06x}"


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT PAIRING — match objects between input and output grids
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectPair:
    """A matched pair of objects (input → output) from a train pair.

    The CRG stores these as learned transformations.
    """
    input_obj: GridObject
    output_obj: Optional[GridObject]  # None if the object disappeared
    transform_type: str               # "recolour", "move", "resize", "appear", "disappear", "unchanged"
    colour_changed: bool
    position_changed: bool
    size_changed: bool

    def __repr__(self):
        return (f"ObjectPair({self.transform_type}: "
                f"colour {self.input_obj.colour}→"
                f"{self.output_obj.colour if self.output_obj else 'gone'})")


def pair_objects(input_objects: List[GridObject],
                 output_objects: List[GridObject]) -> List[ObjectPair]:
    """Match objects between input and output grids.

    Uses a greedy nearest-centroid matching: for each input object, find
    the output object whose centroid is closest. If no output object is
    close enough, the input object "disappeared".

    This is the GLM's analogue of word-alignment in translation.
    """
    pairs: List[ObjectPair] = []
    used_output: Set[int] = set()  # indices of output objects already matched

    for in_obj in input_objects:
        best_idx = -1
        best_dist = float('inf')
        for i, out_obj in enumerate(output_objects):
            if i in used_output:
                continue
            # Distance between centroids
            dr = in_obj.centroid[0] - out_obj.centroid[0]
            dc = in_obj.centroid[1] - out_obj.centroid[1]
            dist = (dr * dr + dc * dc) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0 and best_dist < 5.0:  # reasonable threshold
            out_obj = output_objects[best_idx]
            used_output.add(best_idx)
            colour_changed = in_obj.colour != out_obj.colour
            position_changed = (
                abs(in_obj.centroid[0] - out_obj.centroid[0]) > 0.5
                or abs(in_obj.centroid[1] - out_obj.centroid[1]) > 0.5
            )
            size_changed = in_obj.cell_count != out_obj.cell_count

            if not colour_changed and not position_changed and not size_changed:
                t_type = "unchanged"
            elif colour_changed and not position_changed and not size_changed:
                t_type = "recolour"
            elif not colour_changed and position_changed and not size_changed:
                t_type = "move"
            elif not colour_changed and not position_changed and size_changed:
                t_type = "resize"
            else:
                t_type = "composite"

            pairs.append(ObjectPair(
                input_obj=in_obj,
                output_obj=out_obj,
                transform_type=t_type,
                colour_changed=colour_changed,
                position_changed=position_changed,
                size_changed=size_changed,
            ))
        else:
            # Object disappeared
            pairs.append(ObjectPair(
                input_obj=in_obj,
                output_obj=None,
                transform_type="disappear",
                colour_changed=True,
                position_changed=True,
                size_changed=True,
            ))

    # Any unmatched output objects = "appeared"
    for i, out_obj in enumerate(output_objects):
        if i not in used_output:
            pairs.append(ObjectPair(
                input_obj=GridObject(  # dummy input
                    cells=[], colour=0, grid_shape=out_obj.grid_shape,
                    bbox=(0, 0, 0, 0), centroid=(0, 0),
                ),
                output_obj=out_obj,
                transform_type="appear",
                colour_changed=True, position_changed=True, size_changed=True,
            ))

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# GRID SUMMARY — a "sentence" of object "words"
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GridSentence:
    """A grid decomposed into objects — the GLM's "sentence" representation.

    The grid is a sentence; each object is a word. The sentence has:
      - objects: the words (ordered by colour, then size)
      - palette: the set of colours present
      - dominant_colour: the most common non-zero colour
      - shape: the grid dimensions
      - object_count: how many words in the sentence
    """
    grid: Grid
    objects: List[GridObject]
    palette: frozenset
    dominant_colour: int
    shape: Tuple[int, int]

    @property
    def object_count(self) -> int:
        return len(self.objects)

    def __repr__(self):
        return (f"GridSentence({self.shape}, {self.object_count} objects, "
                f"palette={sorted(self.palette)})")


def grid_to_sentence(grid: Grid) -> GridSentence:
    """Decompose a grid into a GridSentence (objects = words)."""
    objects = extract_objects(grid)
    return GridSentence(
        grid=grid,
        objects=objects,
        palette=grid.palette(),
        dominant_colour=grid.dominant_colour(),
        shape=grid.shape,
    )
