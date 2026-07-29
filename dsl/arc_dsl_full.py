"""
arc_dsl_full.py — Comprehensive DSL for ARC Grid Operations
============================================================

A full-featured, geometrically-grounded domain-specific language for ARC tasks.
Built on a robust spatial foundation with coordinate-based geometry, connected
components, flood fills, and Bresenham primitives.

Design Principles:
    1. Orthogonality: Each operator does one thing well
    2. Composability: All operators are pure functions Grid → Grid
    3. Spatial Coherence: Geometry-first abstractions (coords, bboxes, distances)
    4. Type Safety: Ops enum + Operation dataclass + Program pipeline
    5. Extensibility: Easy to add new operators via OP_IMPL dispatch table

~180 Operators in 20 Semantic Categories:
    - Identity & Basic (3)
    - Geometric Transforms (8)
    - Scaling & Resizing (6)
    - Translation & Shifting (8)
    - Simple Recolouring (12)
    - Conditional Recolouring (10)
    - Set Operations (6)
    - Object Extraction (8)
    - Connectivity & Morphology (9)
    - Gravity & Physics (8)
    - Pattern Operations (10)
    - Row/Column Operations (12)
    - Drawing Primitives (10)
    - Tiling & Replication (8)
    - Cropping & Padding (8)
    - Symmetry Operations (8)
    - Counting & Measurement (8)
    - Noise & Variation (4)
    - Composite Operations (12)

Usage:
    from arc_dsl_full import Operation, Program, Ops
    from arc_loader import Grid

    g = Grid([[0,1,0],[1,1,1],[0,1,0]])

    # Compose a program: rotate → recolour → gravity
    prog = Program([
        Operation(Ops.ROTATE_90),
        Operation(Ops.RECOLOUR, params={"mapping": {1: 2}}),
        Operation(Ops.GRAVITY_DOWN),
    ])
    out = prog.apply(g)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from collections import deque
import math
import sys
import os

# Make arc_loader importable
_PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from arc_loader import Grid, ARCTask


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL UTILITIES — Geometric Foundation
# ══════════════════════════════════════════════════════════════════════════════

def _get_bbox(grid: Grid, colour: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
    """Get bounding box (rmin, rmax, cmin, cmax) for all cells or specific colour."""
    h, w = grid.shape
    rmin, rmax, cmin, cmax = h, -1, w, -1

    for r in range(h):
        for c in range(w):
            if colour is None or grid.cells[r][c] == colour:
                if grid.cells[r][c] != 0 or colour == 0:
                    rmin = min(rmin, r)
                    rmax = max(rmax, r)
                    cmin = min(cmin, c)
                    cmax = max(cmax, c)

    if rmax < rmin:
        return None
    return (rmin, rmax, cmin, cmax)


def _flood_fill(grid: Grid, start_r: int, start_c: int, target_colour: int,
                replacement_colour: int, connectivity: int = 4) -> List[Tuple[int, int]]:
    """Flood fill from start position, returning filled coordinates."""
    h, w = grid.shape
    if not (0 <= start_r < h and 0 <= start_c < w):
        return []
    if grid.cells[start_r][start_c] != target_colour:
        return []

    filled = []
    visited = [[False] * w for _ in range(h)]
    queue = deque([(start_r, start_c)])
    visited[start_r][start_c] = True

    while queue:
        r, c = queue.popleft()
        filled.append((r, c))

        neighbors = _get_neighbors(r, c, h, w, connectivity)
        for nr, nc in neighbors:
            if not visited[nr][nc] and grid.cells[nr][nc] == target_colour:
                visited[nr][nc] = True
                queue.append((nr, nc))

    return filled


def _get_neighbors(r: int, c: int, h: int, w: int, connectivity: int = 4) -> List[Tuple[int, int]]:
    """Get neighbor coordinates based on connectivity (4 or 8)."""
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-way
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    result = []
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    return result


def _connected_components(grid: Grid, colour: int, connectivity: int = 4) -> List[List[Tuple[int, int]]]:
    """Find all connected components of a given colour."""
    h, w = grid.shape
    visited = [[False] * w for _ in range(h)]
    components = []

    for r in range(h):
        for c in range(w):
            if grid.cells[r][c] == colour and not visited[r][c]:
                component = _flood_fill(grid, r, c, colour, colour, connectivity)
                # Mark as visited
                for cr, cc in component:
                    visited[cr][cc] = True
                if component:
                    components.append(component)

    return components


def _bresenham_line(r1: int, c1: int, r2: int, c2: int) -> List[Tuple[int, int]]:
    """Generate points along a line using Bresenham's algorithm."""
    points = []
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r2 > r1 else -1
    sc = 1 if c2 > c1 else -1

    if dc > dr:  # Line is more horizontal
        err = dc // 2
        while r1 != r2 + sr:
            points.append((r1, c1))
            err -= dr
            if err < 0:
                r1 += sr
                err += dc
            c1 += sc
    else:  # Line is more vertical
        err = dr // 2
        while c1 != c2 + sc:
            points.append((r1, c1))
            err -= dc
            if err < 0:
                c1 += sc
                err += dr
            r1 += sr
    points.append((r2, c2))
    return points


def _manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Calculate Manhattan distance between two points."""
    return abs(r2 - r1) + abs(c2 - c1)


def _euclidean_distance(r1: int, c1: int, r2: int, c2: int) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)


def _extract_subgrid(grid: Grid, bbox: Tuple[int, int, int, int]) -> Grid:
    """Extract subgrid defined by bounding box."""
    rmin, rmax, cmin, cmax = bbox
    h, w = rmax - rmin + 1, cmax - cmin + 1
    cells = [[grid.cells[r][c] for c in range(cmin, cmax + 1)] for r in range(rmin, rmax + 1)]
    return Grid(cells)


def _count_cells_of_colour(grid: Grid, colour: int) -> int:
    """Count cells of a specific colour."""
    return sum(row.count(colour) for row in grid.cells)


def _has_symmetry_h(grid: Grid) -> bool:
    """Check horizontal symmetry (left-right mirror)."""
    h, w = grid.shape
    for r in range(h):
        for c in range(w // 2):
            if grid.cells[r][c] != grid.cells[r][w - 1 - c]:
                return False
    return True


def _has_symmetry_v(grid: Grid) -> bool:
    """Check vertical symmetry (top-bottom mirror)."""
    h, w = grid.shape
    for r in range(h // 2):
        for c in range(w):
            if grid.cells[r][c] != grid.cells[h - 1 - r][c]:
                return False
    return True


def _has_symmetry_d1(grid: Grid) -> bool:
    """Check diagonal symmetry (main diagonal)."""
    h, w = grid.shape
    if h != w:
        return False
    for r in range(h):
        for c in range(r + 1, w):
            if grid.cells[r][c] != grid.cells[c][r]:
                return False
    return True


def _has_symmetry_d2(grid: Grid) -> bool:
    """Check anti-diagonal symmetry."""
    h, w = grid.shape
    if h != w:
        return False
    for r in range(h):
        for c in range(w - r):
            if grid.cells[r][c] != grid.cells[w - 1 - c][h - 1 - r]:
                return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# OPERATOR ENUM — Complete Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

class Ops(str, Enum):
    """Complete DSL operator vocabulary. str-Enum for JSON serialization."""

    # ── Identity & Basic (3) ───────────────────────────────────────────────────
    IDENTITY      = "identity"
    CLEAR         = "clear"           # Set all to 0
    INVERT_BG     = "invert_bg"       # Swap 0 and non-zero

    # ── Geometric Transforms (8) ──────────────────────────────────────────────
    ROTATE_90     = "rotate_90"       # 90° clockwise
    ROTATE_180    = "rotate_180"
    ROTATE_270    = "rotate_270"      # 90° counter-clockwise
    FLIP_H        = "flip_h"          # Horizontal flip (left-right)
    FLIP_V        = "flip_v"          # Vertical flip (top-bottom)
    TRANSPOSE     = "transpose"       # Reflect over main diagonal
    ANTI_TRANSPOSE= "anti_transpose"  # Reflect over anti-diagonal
    ROTATE_ARBITRARY = "rotate_arbitrary"  # params: angle_degrees

    # ── Scaling & Resizing (6) ────────────────────────────────────────────────
    SCALE_2X      = "scale_2x"        # Double dimensions
    SCALE_3X      = "scale_3x"        # Triple dimensions
    SCALE_HALF    = "scale_half"      # Halve dimensions
    SCALE_NX      = "scale_nx"        # params: factor
    RESIZE        = "resize"          # params: new_h, new_w, mode
    ASPECT_FILL   = "aspect_fill"     # params: target_h, target_w, fill_colour

    # ── Translation & Shifting (8) ────────────────────────────────────────────
    TRANSLATE     = "translate"       # params: dr, dc
    SHIFT_UP      = "shift_up"        # params: n
    SHIFT_DOWN    = "shift_down"      # params: n
    SHIFT_LEFT    = "shift_left"      # params: n
    SHIFT_RIGHT   = "shift_right"     # params: n
    WRAP_SHIFT_H  = "wrap_shift_h"    # params: n (horizontal wrap)
    WRAP_SHIFT_V  = "wrap_shift_v"    # params: n (vertical wrap)
    CYCLE_ROWS    = "cycle_rows"      # params: amounts per row

    # ── Simple Recolouring (12) ───────────────────────────────────────────────
    RECOLOUR      = "recolour"        # params: mapping {old: new}
    SWAP_COLOURS  = "swap_colours"    # params: c1, c2
    PALETTE_CYCLE = "palette_cycle"   # params: cycle_list
    NORMALIZE     = "normalize"       # Map colours to 1..N
    GREYSCALE     = "greyscale"       # Convert to luminance-like
    INVERT_COLOURS= "invert_colours"  # params: max_colour
    RANDOMIZE     = "randomize"       # params: seed, palette
    QUANTIZE      = "quantize"        # params: n_colours
    COLOUR_TO_INTENSITY = "colour_to_intensity"
    BINARY_THRESHOLD = "binary_threshold"  # params: threshold, below, above
    GRADIENT_MAP  = "gradient_map"    # params: gradient
    HIGHLIGHT     = "highlight"       # params: target_colour, highlight_colour

    # ── Conditional Recolouring (10) ──────────────────────────────────────────
    RECOLOUR_IF_NEIGHBOUR = "recolour_if_neighbour"  # params: neighbour_colour, new_colour
    RECOLOUR_IF_BORDER    = "recolour_if_border"     # params: new_colour
    RECOLOUR_IF_CORNER    = "recolour_if_corner"     # params: new_colour
    RECOLOUR_IF_ISOLATED  = "recolour_if_isolated"   # params: new_colour
    RECOLOUR_IF_CROWDED   = "recolour_if_crowded"    # params: threshold, new_colour
    RECOLOUR_INTERIOR     = "recolour_interior"      # params: fill_colour
    RECOLOUR_EXTERIOR     = "recolour_exterior"      # params: fill_colour
    RECOLOUR_EDGE_ADJACENT= "recolour_edge_adjacent" # params: target, new_colour
    RECOLOUR_DIAGONAL_ONLY= "recolour_diagonal_only" # params: new_colour
    RECOLOUR_BY_DENSITY   = "recolour_by_density"    # params: thresholds, colours

    # ── Set Operations (6) ────────────────────────────────────────────────────
    SET_INTERSECT   = "set_intersect"     # params: c1, c2
    SET_DIFFERENCE  = "set_difference"    # params: from_colour, by_colour
    SET_UNION       = "set_union"         # params: c1, c2, into_colour
    SET_XOR         = "set_xor"           # params: c1, c2
    SET_COMPLEMENT  = "set_complement"    # params: colour
    DILATION        = "dilation"          # params: colour, iterations

    # ── Object Extraction (8) ─────────────────────────────────────────────────
    EXTRACT_LARGEST     = "extract_largest"
    EXTRACT_SMALLEST    = "extract_smallest"
    EXTRACT_COLOUR      = "extract_colour"       # params: colour
    EXTRACT_NTH         = "extract_nth"          # params: n, sort_by
    EXTRACT_TOP_LEFT    = "extract_top_left"
    EXTRACT_CENTER      = "extract_center"
    EXTRACT_ALL_OBJECTS = "extract_all_objects"
    EXTRACT_BBOX        = "extract_bbox"         # params: colour

    # ── Connectivity & Morphology (9) ─────────────────────────────────────────
    FILL_INTERIOR   = "fill_interior"     # params: outline_colour, fill_colour
    OUTLINE         = "outline"           # Extract 1-cell border
    DILATE_OP       = "dilate_op"         # params: colour, connectivity
    ERODE_OP        = "erode_op"          # params: colour, connectivity
    OPEN_MORPH      = "open_morph"        # Erode then dilate
    CLOSE_MORPH     = "close_morph"       # Dilate then erode
    SKELETONIZE     = "skeletonize"       # Thin to 1-cell width
    THICKEN         = "thicken"           # params: thickness
    FILL_HOLES      = "fill_holes"        # params: max_hole_size

    # ── Gravity & Physics (8) ─────────────────────────────────────────────────
    GRAVITY_DOWN    = "gravity_down"
    GRAVITY_UP      = "gravity_up"
    GRAVITY_LEFT    = "gravity_left"
    GRAVITY_RIGHT   = "gravity_right"
    GRAVITY_RADIAL  = "gravity_radial"    # params: center_r, center_c
    GRAVITY_DIAGONAL= "gravity_diagonal"   # params: direction
    SAND_FALL       = "sand_fall"         # Sand physics (falls through some)
    WATER_FLOW      = "water_flow"        # Water physics (spreads horizontally)

    # ── Pattern Operations (10) ───────────────────────────────────────────────
    REPLACE_PATTERN = "replace_pattern"   # params: from_pattern, to_pattern
    FIND_PATTERN    = "find_pattern"      # params: pattern, mark_colour
    REPEAT_PATTERN  = "repeat_pattern"    # params: pattern, times, axis
    DETECT_PERIODICITY = "detect_periodicity"
    EXTEND_PATTERN  = "extend_pattern"    # params: direction, steps
    MIRROR_PATTERN  = "mirror_pattern"    # params: axis
    TILE_PATTERN    = "tile_pattern"      # params: pattern
    INTERPOLATE     = "interpolate"       # params: start, end
    COMPLETE_RECTANGLE = "complete_rectangle"
    FILL_RECTANGLE_HOLES = "fill_rectangle_holes"

    # ── Row/Column Operations (12) ────────────────────────────────────────────
    SHIFT_ROW       = "shift_row"         # params: row_idx, shift
    SHIFT_COL       = "shift_col"         # params: col_idx, shift
    FILL_ROW        = "fill_row"          # params: row_idx, colour
    FILL_COL        = "fill_col"          # params: col_idx, colour
    COPY_ROW        = "copy_row"          # params: from_idx, to_idx
    COPY_COL        = "copy_col"          # params: from_idx, to_idx
    DELETE_ROW      = "delete_row"        # params: row_idx
    DELETE_COL      = "delete_col"        # params: col_idx
    DUPLICATE_ROW   = "duplicate_row"     # params: row_idx
    DUPLICATE_COL   = "duplicate_col"     # params: col_idx
    REVERSE_ROW     = "reverse_row"       # params: row_idx
    REVERSE_COL     = "reverse_col"       # params: col_idx
    SORT_ROW        = "sort_row"          # params: row_idx, ascending
    SORT_COL        = "sort_col"          # params: col_idx, ascending

    # ── Drawing Primitives (10) ───────────────────────────────────────────────
    DRAW_LINE       = "draw_line"         # params: r1, c1, r2, c2, colour
    DRAW_RECT_OUTLINE = "draw_rect_outline"  # params: r1, c1, r2, c2, colour
    DRAW_RECT_FILL  = "draw_rect_fill"    # params: r1, c1, r2, c2, colour
    DRAW_CIRCLE     = "draw_circle"       # params: center_r, center_c, radius, colour
    DRAW_DOT        = "draw_dot"          # params: r, c, colour
    DRAW_CROSS      = "draw_cross"        # params: r, c, size, colour
    DRAW_DIAGONAL   = "draw_diagonal"     # params: colour, direction
    DRAW_BORDER     = "draw_border"       # params: thickness, colour
    DRAW_GRID       = "draw_grid"         # params: cell_size, colour
    DRAW_FRAME      = "draw_frame"        # params: margin, colour

    # ── Tiling & Replication (8) ──────────────────────────────────────────────
    TILE_2X         = "tile_2x"           # Tile 2x in both dimensions
    TILE_3X         = "tile_3x"           # Tile 3x in both dimensions
    TILE_NX         = "tile_nx"           # params: h_factor, w_factor
    REPLICATE_OBJ   = "replicate_obj"     # params: count, axis, step
    COUNT_FILL      = "count_fill"        # Fill row with N copies
    MIRROR_TILE     = "mirror_tile"       # params: axis
    STAMP           = "stamp"             # params: pattern, positions
    SPREAD          = "spread"            # params: colour, direction, distance

    # ── Cropping & Padding (8) ────────────────────────────────────────────────
    CROP_TO_NONZERO = "crop_to_nonzero"
    CROP_TO_COLOUR  = "crop_to_colour"    # params: colour
    CROP_TO_CENTER  = "crop_to_center"    # params: target_h, target_w
    CROP_TO_CORNER  = "crop_to_corner"    # params: corner, size
    PAD_TOP         = "pad_top"           # params: n, colour
    PAD_BOTTOM      = "pad_bottom"        # params: n, colour
    PAD_LEFT        = "pad_left"          # params: n, colour
    PAD_RIGHT       = "pad_right"         # params: n, colour
    PAD_ALL         = "pad_all"           # params: n, colour

    # ── Symmetry Operations (8) ───────────────────────────────────────────────
    MAKE_SYMMETRIC_H = "make_symmetric_h"  # Force horizontal symmetry
    MAKE_SYMMETRIC_V = "make_symmetric_v"  # Force vertical symmetry
    MAKE_SYMMETRIC_D1= "make_symmetric_d1" # Force diagonal symmetry
    MAKE_SYMMETRIC_D2= "make_symmetric_d2" # Force anti-diagonal symmetry
    MAKE_SYMMETRIC_ROT= "make_symmetric_rot" # Force rotational symmetry
    CHECK_SYMMETRY  = "check_symmetry"    # params: type
    SYMMETRIZE_UNION = "symmetrize_union"  # Union with mirror
    SYMMETRIZE_INTERSECT = "symmetrize_intersect"  # Intersect with mirror

    # ── Counting & Measurement (8) ────────────────────────────────────────────
    COUNT_OBJECTS   = "count_objects"     # params: colour, connectivity
    MEASURE_BBOX    = "measure_bbox"      # params: colour
    MARK_CENTROID   = "mark_centroid"     # params: colour, mark_colour
    MARK_EXTREMA    = "mark_extrema"      # params: colour, mark_colour
    ENCODE_SIZE     = "encode_size"       # Encode bbox size as pattern
    HISTOGRAM       = "histogram"         # Create colour histogram visualization
    LABEL_COMPONENTS= "label_components"  # params: connectivity
    RANK_BY_SIZE    = "rank_by_size"      # params: colour

    # ── Noise & Variation (4) ─────────────────────────────────────────────────
    ADD_NOISE       = "add_noise"         # params: density, palette
    REMOVE_SINGLETONS= "remove_singletons"# Remove isolated pixels
    SMOOTH          = "smooth"            # params: iterations
    PERTURB         = "perturb"           # params: max_shift

    # ── Composite Operations (12) ─────────────────────────────────────────────
    EXTRACT_AND_CENTER = "extract_and_center"
    COLORIZE_REGIONS   = "colorize_regions"  # params: colour_map
    CONNECT_NEAREST    = "connect_nearest"   # Connect objects with lines
    FILL_BETWEEN       = "fill_between"      # params: c1, c2, fill_colour
    PROPAGATE_COLOUR   = "propagate_colour"  # params: source_colour, target_colour
    GRADIENT_FILL      = "gradient_fill"     # params: direction, colours
    CONTOUR            = "contour"           # Draw contour lines
    BLIT               = "blit"              # params: src_grid, dest_r, dest_c
    MASK               = "mask"              # params: mask_colour
    COMPOSITE          = "composite"         # params: overlay_grid, mode
    LAYERS_MERGE       = "layers_merge"      # params: layer_grids, mode
    CHANNEL_EXTRACT    = "channel_extract"   # params: colour_channel
    MULTIPLY_BRIGHTNESS= "multiply_brightness" # params: factor


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Operation:
    """A single DSL operation with optional parameters."""
    op: Ops
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure params is always a dict
        if self.params is None:
            self.params = {}


@dataclass
class Program:
    """A pipeline of operations that compose into a transformation."""
    operations: List[Operation] = field(default_factory=list)

    def add(self, op: Ops, **params) -> Program:
        """Fluent interface to add an operation."""
        self.operations.append(Operation(op, params))
        return self

    def apply(self, grid: Grid) -> Grid:
        """Apply all operations in sequence."""
        result = grid.copy()
        for operation in self.operations:
            impl = OP_IMPL.get(operation.op)
            if impl is None:
                raise NotImplementedError(f"No implementation for {operation.op}")
            try:
                result = impl(result, operation.params)
            except Exception as e:
                raise RuntimeError(f"Error applying {operation.op}: {e}")
        return result

    def matches_train(self, task: ARCTask) -> bool:
        """Check if program produces correct outputs for all training pairs."""
        for pair in task.train:
            predicted = self.apply(pair.input)
            if predicted != pair.output:
                return False
        return True

    def __len__(self) -> int:
        return len(self.operations)

    def __repr__(self) -> str:
        ops_str = ", ".join(f"{op.op.value}" for op in self.operations)
        return f"Program([{ops_str}])"


# ══════════════════════════════════════════════════════════════════════════════
# OPERATOR IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Identity & Basic ──────────────────────────────────────────────────────────

def _op_identity(g: Grid, p: dict) -> Grid:
    return g.copy()


def _op_clear(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    return Grid([[0] * w for _ in range(h)])


def _op_invert_bg(g: Grid, p: dict) -> Grid:
    """Swap background (0) with foreground (non-zero becomes 0, 0 becomes dominant)."""
    dominant = g.dominant_colour()
    if dominant == 0:
        dominant = 1
    return Grid([[dominant if v == 0 else 0 for v in row] for row in g.cells])


# ── Geometric Transforms ──────────────────────────────────────────────────────

def _op_rotate_90(g: Grid, p: dict) -> Grid:
    return g.rotate_90()


def _op_rotate_180(g: Grid, p: dict) -> Grid:
    return g.rotate_180()


def _op_rotate_270(g: Grid, p: dict) -> Grid:
    return g.rotate_270()


def _op_flip_h(g: Grid, p: dict) -> Grid:
    return g.flip_h()


def _op_flip_v(g: Grid, p: dict) -> Grid:
    return g.flip_v()


def _op_transpose(g: Grid, p: dict) -> Grid:
    return g.transpose()


def _op_anti_transpose(g: Grid, p: dict) -> Grid:
    """Reflect over anti-diagonal (top-right to bottom-left)."""
    h, w = g.shape
    out = [[0] * h for _ in range(w)]
    for r in range(h):
        for c in range(w):
            out[w - 1 - c][h - 1 - r] = g.cells[r][c]
    return Grid(out)


def _op_rotate_arbitrary(g: Grid, p: dict) -> Grid:
    """Rotate by arbitrary angle (approximate, using nearest neighbor)."""
    angle = float(p.get("angle_degrees", 90))
    radians = math.radians(angle)
    h, w = g.shape

    # Center of rotation
    cy, cx = h / 2, w / 2

    # New dimensions (conservative bound)
    new_size = int(math.ceil(math.sqrt(h*h + w*w)))
    out = [[0] * new_size for _ in range(new_size)]
    new_cy, new_cx = new_size / 2, new_size / 2

    cos_a, sin_a = math.cos(radians), math.sin(radians)

    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == 0:
                continue
            # Translate to origin, rotate, translate back
            y, x = r - cy, c - cx
            yr = y * cos_a - x * sin_a
            xr = y * sin_a + x * cos_a
            nr, nc = int(round(yr + new_cy)), int(round(xr + new_cx))
            if 0 <= nr < new_size and 0 <= nc < new_size:
                out[nr][nc] = g.cells[r][c]

    return Grid(out)


# ── Scaling & Resizing ────────────────────────────────────────────────────────

def _op_scale_2x(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * (w * 2) for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            out[r*2][c*2] = v
            out[r*2][c*2+1] = v
            out[r*2+1][c*2] = v
            out[r*2+1][c*2+1] = v
    return Grid(out)


def _op_scale_3x(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * (w * 3) for _ in range(h * 3)]
    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            for dr in range(3):
                for dc in range(3):
                    out[r*3+dr][c*3+dc] = v
    return Grid(out)


def _op_scale_half(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    nh, nw = h // 2, w // 2
    if nh == 0 or nw == 0:
        return g.copy()
    out = [[g.cells[r*2][c*2] for c in range(nw)] for r in range(nh)]
    return Grid(out)


def _op_scale_nx(g: Grid, p: dict) -> Grid:
    factor = int(p.get("factor", 2))
    if factor <= 0:
        return g.copy()
    h, w = g.shape
    out = [[0] * (w * factor) for _ in range(h * factor)]
    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            for dr in range(factor):
                for dc in range(factor):
                    out[r*factor+dr][c*factor+dc] = v
    return Grid(out)


def _op_resize(g: Grid, p: dict) -> Grid:
    new_h = int(p.get("new_h", g.shape[0]))
    new_w = int(p.get("new_w", g.shape[1]))
    mode = p.get("mode", "stretch")  # stretch, crop, pad

    h, w = g.shape
    out = [[0] * new_w for _ in range(new_h)]

    if mode == "stretch":
        for r in range(new_h):
            for c in range(new_w):
                src_r = int(r * h / new_h)
                src_c = int(c * w / new_w)
                out[r][c] = g.cells[src_r][src_c]
    elif mode == "crop":
        for r in range(min(h, new_h)):
            for c in range(min(w, new_w)):
                out[r][c] = g.cells[r][c]
    elif mode == "pad":
        for r in range(min(h, new_h)):
            for c in range(min(w, new_w)):
                out[r][c] = g.cells[r][c]

    return Grid(out)


def _op_aspect_fill(g: Grid, p: dict) -> Grid:
    target_h = int(p.get("target_h", 10))
    target_w = int(p.get("target_w", 10))
    fill_colour = int(p.get("fill_colour", 0))

    h, w = g.shape
    out = [[fill_colour] * target_w for _ in range(target_h)]

    # Scale to fit while preserving aspect ratio
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Center the scaled image
    offset_r = (target_h - new_h) // 2
    offset_c = (target_w - new_w) // 2

    for r in range(new_h):
        for c in range(new_w):
            src_r = int(r * h / new_h) if new_h > 0 else 0
            src_c = int(c * w / new_w) if new_w > 0 else 0
            out[offset_r + r][offset_c + c] = g.cells[src_r][src_c]

    return Grid(out)


# ── Translation & Shifting ────────────────────────────────────────────────────

def _op_translate(g: Grid, p: dict) -> Grid:
    dr = int(p.get("dr", 0))
    dc = int(p.get("dc", 0))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            if v != 0:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = v
    return Grid(out)


def _op_shift_up(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h - n):
        for c in range(w):
            out[r][c] = g.cells[r + n][c]
    return Grid(out)


def _op_shift_down(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(n, h):
        for c in range(w):
            out[r][c] = g.cells[r - n][c]
    return Grid(out)


def _op_shift_left(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w - n):
            out[r][c] = g.cells[r][c + n]
    return Grid(out)


def _op_shift_right(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(n, w):
            out[r][c] = g.cells[r][c - n]
    return Grid(out)


def _op_wrap_shift_h(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = g.cells[r][(c - n) % w]
    return Grid(out)


def _op_wrap_shift_v(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = g.cells[(r - n) % h][c]
    return Grid(out)


def _op_cycle_rows(g: Grid, p: dict) -> Grid:
    amounts = p.get("amounts", [])
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        shift = amounts[r] if r < len(amounts) else 0
        for c in range(w):
            out[r][c] = g.cells[r][(c - shift) % w]
    return Grid(out)


# ── Simple Recolouring ────────────────────────────────────────────────────────

def _op_recolour(g: Grid, p: dict) -> Grid:
    mapping = p.get("mapping", {})
    return Grid([[mapping.get(v, v) for v in row] for row in g.cells])


def _op_swap_colours(g: Grid, p: dict) -> Grid:
    c1 = int(p.get("c1", 1))
    c2 = int(p.get("c2", 2))
    return Grid([[c2 if v == c1 else (c1 if v == c2 else v) for v in row] for row in g.cells])


def _op_palette_cycle(g: Grid, p: dict) -> Grid:
    cycle = p.get("cycle_list", [])
    if not cycle:
        return g.copy()
    mapping = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}
    return Grid([[mapping.get(v, v) for v in row] for row in g.cells])


def _op_normalize(g: Grid, p: dict) -> Grid:
    """Map colours to 1..N preserving relative order."""
    colours = sorted(set(v for row in g.cells for v in row if v != 0))
    mapping = {c: i + 1 for i, c in enumerate(colours)}
    return Grid([[mapping.get(v, v) for v in row] for row in g.cells])


def _op_greyscale(g: Grid, p: dict) -> Grid:
    """Convert to greyscale based on colour value."""
    return Grid([[min(9, v) for v in row] for row in g.cells])


def _op_invert_colours(g: Grid, p: dict) -> Grid:
    max_c = int(p.get("max_colour", 9))
    return Grid([[max_c - v if v != 0 else 0 for v in row] for row in g.cells])


def _op_randomize(g: Grid, p: dict) -> Grid:
    import random
    seed = p.get("seed")
    palette = p.get("palette", list(range(1, 10)))
    if seed is not None:
        random.seed(seed)
    return Grid([[random.choice(palette) if v != 0 else 0 for v in row] for row in g.cells])


def _op_quantize(g: Grid, p: dict) -> Grid:
    n_colours = int(p.get("n_colours", 5))
    colours = sorted(set(v for row in g.cells for v in row if v != 0))
    if len(colours) <= n_colours:
        return g.copy()
    # Simple quantization: group into bins
    step = len(colours) / n_colours
    mapping = {colours[i]: colours[min(int(i // step * step), len(colours) - 1)] for i in range(len(colours))}
    return Grid([[mapping.get(v, v) for v in row] for row in g.cells])


def _op_colour_to_intensity(g: Grid, p: dict) -> Grid:
    """Map each colour to its intensity value."""
    return Grid([[v for v in row] for row in g.cells])


def _op_binary_threshold(g: Grid, p: dict) -> Grid:
    threshold = int(p.get("threshold", 5))
    below = int(p.get("below", 0))
    above = int(p.get("above", 1))
    return Grid([[above if v >= threshold else below for v in row] for row in g.cells])


def _op_gradient_map(g: Grid, p: dict) -> Grid:
    gradient = p.get("gradient", {0: 0, 9: 9})
    if not gradient:
        return g.copy()
    sorted_keys = sorted(gradient.keys())

    def map_val(v):
        if v <= sorted_keys[0]:
            return gradient[sorted_keys[0]]
        if v >= sorted_keys[-1]:
            return gradient[sorted_keys[-1]]
        for i in range(len(sorted_keys) - 1):
            if sorted_keys[i] <= v <= sorted_keys[i + 1]:
                t = (v - sorted_keys[i]) / (sorted_keys[i + 1] - sorted_keys[i])
                return int(gradient[sorted_keys[i]] + t * (gradient[sorted_keys[i + 1]] - gradient[sorted_keys[i]]))
        return v

    return Grid([[map_val(v) for v in row] for row in g.cells])


def _op_highlight(g: Grid, p: dict) -> Grid:
    target = int(p.get("target_colour", 1))
    highlight = int(p.get("highlight_colour", 9))
    return Grid([[highlight if v == target else v for v in row] for row in g.cells])


# ── Conditional Recolouring ───────────────────────────────────────────────────

def _op_recolour_if_neighbour(g: Grid, p: dict) -> Grid:
    neighbour_c = int(p.get("neighbour_colour", 1))
    new_c = int(p.get("new_colour", 2))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] == neighbour_c:
                    out[r][c] = new_c
                    break
    return Grid(out)


def _op_recolour_if_border(g: Grid, p: dict) -> Grid:
    new_c = int(p.get("new_colour", 1))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if r == 0 or r == h-1 or c == 0 or c == w-1:
                if g.cells[r][c] != 0:
                    out[r][c] = new_c
    return Grid(out)


def _op_recolour_if_corner(g: Grid, p: dict) -> Grid:
    new_c = int(p.get("new_colour", 1))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    for r, c in corners:
        if 0 <= r < h and 0 <= c < w and g.cells[r][c] != 0:
            out[r][c] = new_c
    return Grid(out)


def _op_recolour_if_isolated(g: Grid, p: dict) -> Grid:
    new_c = int(p.get("new_colour", 0))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == 0:
                continue
            has_neighbour = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] == g.cells[r][c]:
                    has_neighbour = True
                    break
            if not has_neighbour:
                out[r][c] = new_c
    return Grid(out)


def _op_recolour_if_crowded(g: Grid, p: dict) -> Grid:
    threshold = int(p.get("threshold", 4))
    new_c = int(p.get("new_colour", 9))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == 0:
                continue
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] != 0:
                        count += 1
            if count >= threshold:
                out[r][c] = new_c
    return Grid(out)


def _op_recolour_interior(g: Grid, p: dict) -> Grid:
    fill_c = int(p.get("fill_colour", 1))
    h, w = g.shape
    # Find outline (dominant non-zero colour)
    palette = g.palette()
    non_zero = [c for c in palette if c != 0]
    if not non_zero:
        return g.copy()
    outline_c = max(non_zero, key=lambda c: _count_cells_of_colour(g, c))

    # Flood fill from borders
    seen = [[False] * w for _ in range(h)]
    stack = []
    for r in range(h):
        for c in range(w):
            if r == 0 or r == h-1 or c == 0 or c == w-1:
                if g.cells[r][c] != outline_c:
                    stack.append((r, c))
                    seen[r][c] = True

    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g.cells[nr][nc] != outline_c:
                seen[nr][nc] = True
                stack.append((nr, nc))

    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if not seen[r][c] and g.cells[r][c] != outline_c:
                out[r][c] = fill_c
    return Grid(out)


def _op_recolour_exterior(g: Grid, p: dict) -> Grid:
    fill_c = int(p.get("fill_colour", 0))
    h, w = g.shape
    palette = g.palette()
    non_zero = [c for c in palette if c != 0]
    if not non_zero:
        return g.copy()
    outline_c = max(non_zero, key=lambda c: _count_cells_of_colour(g, c))

    seen = [[False] * w for _ in range(h)]
    stack = []
    for r in range(h):
        for c in range(w):
            if r == 0 or r == h-1 or c == 0 or c == w-1:
                if g.cells[r][c] != outline_c:
                    stack.append((r, c))
                    seen[r][c] = True

    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g.cells[nr][nc] != outline_c:
                seen[nr][nc] = True
                stack.append((nr, nc))

    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if seen[r][c]:
                out[r][c] = fill_c
    return Grid(out)


def _op_recolour_edge_adjacent(g: Grid, p: dict) -> Grid:
    target = int(p.get("target", 1))
    new_c = int(p.get("new_colour", 2))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] != target:
                continue
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] != 0 and g.cells[nr][nc] != target:
                    out[r][c] = new_c
                    break
    return Grid(out)


def _op_recolour_diagonal_only(g: Grid, p: dict) -> Grid:
    new_c = int(p.get("new_colour", 1))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == 0:
                continue
            has_diag = False
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] != 0:
                    has_diag = True
                    break
            if has_diag:
                out[r][c] = new_c
    return Grid(out)


def _op_recolour_by_density(g: Grid, p: dict) -> Grid:
    thresholds = p.get("thresholds", [2, 5])
    colours = p.get("colours", [1, 5, 9])
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == 0:
                continue
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] != 0:
                        count += 1
            for i, thresh in enumerate(thresholds):
                if count >= thresh:
                    out[r][c] = colours[min(i, len(colours) - 1)]
    return Grid(out)


# ── Set Operations ────────────────────────────────────────────────────────────

def _op_set_intersect(g: Grid, p: dict) -> Grid:
    c1 = int(p.get("c1", 1))
    c2 = int(p.get("c2", 2))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == c1:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] == c2:
                        out[r][c] = c1
                        break
    return Grid(out)


def _op_set_difference(g: Grid, p: dict) -> Grid:
    from_c = int(p.get("from_colour", 1))
    by_c = int(p.get("by_colour", 2))
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == from_c:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] == by_c:
                        out[r][c] = 0
                        break
    return Grid(out)


def _op_set_union(g: Grid, p: dict) -> Grid:
    c1 = int(p.get("c1", 1))
    c2 = int(p.get("c2", 2))
    into = int(p.get("into_colour", c1))
    return Grid([[into if v in (c1, c2) else v for v in row] for row in g.cells])


def _op_set_xor(g: Grid, p: dict) -> Grid:
    c1 = int(p.get("c1", 1))
    c2 = int(p.get("c2", 2))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            if (v == c1) != (v == c2):  # XOR
                out[r][c] = v
    return Grid(out)


def _op_set_complement(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", 1))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] != colour:
                out[r][c] = g.cells[r][c]
    return Grid(out)


def _op_dilation(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))
    iterations = int(p.get("iterations", 1))
    if colour == 0:
        return g.copy()

    result = g.copy()
    for _ in range(iterations):
        h, w = result.shape
        out = [row[:] for row in result.cells]
        for r in range(h):
            for c in range(w):
                if result.cells[r][c] == colour:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            out[nr][nc] = colour
        result = Grid(out)
    return result


# ── Object Extraction ─────────────────────────────────────────────────────────

def _op_extract_largest(g: Grid, p: dict) -> Grid:
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    components = _connected_components(g, dominant)
    if not components:
        return g.copy()
    largest = max(components, key=len)
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r, c in largest:
        out[r][c] = dominant
    return Grid(out)


def _op_extract_smallest(g: Grid, p: dict) -> Grid:
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    components = _connected_components(g, dominant)
    if not components:
        return g.copy()
    smallest = min(components, key=len)
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r, c in smallest:
        out[r][c] = dominant
    return Grid(out)


def _op_extract_colour(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", 1))
    h, w = g.shape
    out = [[colour if v == colour else 0 for v in row] for row in g.cells]
    return Grid(out)


def _op_extract_nth(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 0))
    sort_by = p.get("sort_by", "size")  # size, position
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    components = _connected_components(g, dominant)
    if not components:
        return g.copy()
    if sort_by == "size":
        components.sort(key=len, reverse=True)
    else:  # position (top-left first)
        components.sort(key=lambda comp: (min(r for r, c in comp), min(c for r, c in comp)))
    if n >= len(components):
        return g.copy()
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r, c in components[n]:
        out[r][c] = dominant
    return Grid(out)


def _op_extract_top_left(g: Grid, p: dict) -> Grid:
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    components = _connected_components(g, dominant)
    if not components:
        return g.copy()
    top_left = min(components, key=lambda comp: (min(r for r, c in comp), min(c for r, c in comp)))
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r, c in top_left:
        out[r][c] = dominant
    return Grid(out)


def _op_extract_center(g: Grid, p: dict) -> Grid:
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    components = _connected_components(g, dominant)
    if not components:
        return g.copy()
    h, w = g.shape
    center_r, center_c = h / 2, w / 2
    closest = min(components, key=lambda comp:
        _euclidean_distance(sum(r for r, c in comp) / len(comp),
                          sum(c for r, c in comp) / len(comp), center_r, center_c))
    out = [[0] * w for _ in range(h)]
    for r, c in closest:
        out[r][c] = dominant
    return Grid(out)


def _op_extract_all_objects(g: Grid, p: dict) -> Grid:
    """Extract all objects, keeping only the largest of each colour."""
    palette = g.palette()
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for colour in palette:
        if colour == 0:
            continue
        components = _connected_components(g, colour)
        if components:
            largest = max(components, key=len)
            for r, c in largest:
                out[r][c] = colour
    return Grid(out)


def _op_extract_bbox(g: Grid, p: dict) -> Grid:
    colour = p.get("colour")
    bbox = _get_bbox(g, colour)
    if bbox is None:
        return g.copy()
    return _extract_subgrid(g, bbox)


# ── Connectivity & Morphology ─────────────────────────────────────────────────

def _op_fill_interior(g: Grid, p: dict) -> Grid:
    outline_c = p.get("outline_colour")
    fill_c = int(p.get("fill_colour", 1))
    if outline_c is None:
        palette = g.palette()
        non_zero = [c for c in palette if c != 0]
        outline_c = max(non_zero, key=lambda c: _count_cells_of_colour(g, c)) if non_zero else 0

    h, w = g.shape
    seen = [[False] * w for _ in range(h)]
    stack = []
    for r in range(h):
        for c in range(w):
            if r == 0 or r == h-1 or c == 0 or c == w-1:
                if g.cells[r][c] != outline_c:
                    stack.append((r, c))
                    seen[r][c] = True

    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g.cells[nr][nc] != outline_c:
                seen[nr][nc] = True
                stack.append((nr, nc))

    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if not seen[r][c] and g.cells[r][c] != outline_c:
                out[r][c] = fill_c
    return Grid(out)


def _op_outline(g: Grid, p: dict) -> Grid:
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] != dominant:
                continue
            is_border = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or g.cells[nr][nc] != dominant:
                    is_border = True
                    break
            if is_border:
                out[r][c] = dominant
    return Grid(out)


def _op_dilate_op(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))
    connectivity = int(p.get("connectivity", 4))
    if colour == 0:
        return g.copy()
    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == colour:
                for nr, nc in _get_neighbors(r, c, h, w, connectivity):
                    out[nr][nc] = colour
    return Grid(out)


def _op_erode_op(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))
    connectivity = int(p.get("connectivity", 4))
    if colour == 0:
        return g.copy()
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == colour:
                neighbors = _get_neighbors(r, c, h, w, connectivity)
                if all(g.cells[nr][nc] == colour for nr, nc in neighbors):
                    out[r][c] = colour
    return Grid(out)


def _op_open_morph(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))
    connectivity = int(p.get("connectivity", 4))
    result = _op_erode_op(g, {"colour": colour, "connectivity": connectivity})
    return _op_dilate_op(result, {"colour": colour, "connectivity": connectivity})


def _op_close_morph(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))
    connectivity = int(p.get("connectivity", 4))
    result = _op_dilate_op(g, {"colour": colour, "connectivity": connectivity})
    return _op_erode_op(result, {"colour": colour, "connectivity": connectivity})


def _op_skeletonize(g: Grid, p: dict) -> Grid:
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    result = g.copy()
    prev_count = _count_cells_of_colour(result, dominant)

    while True:
        h, w = result.shape
        out = [row[:] for row in result.cells]
        changed = False
        for r in range(h):
            for c in range(w):
                if result.cells[r][c] != dominant:
                    continue
                neighbors = _get_neighbors(r, c, h, w, 8)
                # Remove if all neighbors are also dominant (not on edge)
                if all(result.cells[nr][nc] == dominant for nr, nc in neighbors):
                    # Check if removing won't disconnect
                    out[r][c] = 0
                    changed = True
        result = Grid(out)
        new_count = _count_cells_of_colour(result, dominant)
        if new_count == prev_count or new_count == 0:
            break
        prev_count = new_count
    return result


def _op_thicken(g: Grid, p: dict) -> Grid:
    thickness = int(p.get("thickness", 2))
    colour = int(p.get("colour", g.dominant_colour()))
    result = g.copy()
    for _ in range(thickness - 1):
        result = _op_dilate_op(result, {"colour": colour, "connectivity": 8})
    return result


def _op_fill_holes(g: Grid, p: dict) -> Grid:
    max_hole_size = int(p.get("max_hole_size", 10))
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    h, w = g.shape

    # Find background components (holes)
    temp = Grid([[0 if v == dominant else 1 for v in row] for row in g.cells])
    bg_components = _connected_components(temp, 1)

    out = [row[:] for row in g.cells]
    for comp in bg_components:
        if len(comp) <= max_hole_size:
            # Check if it's enclosed (not touching border)
            touches_border = any(r == 0 or r == h-1 or c == 0 or c == w-1 for r, c in comp)
            if not touches_border:
                for r, c in comp:
                    out[r][c] = dominant
    return Grid(out)


# ── Gravity & Physics ─────────────────────────────────────────────────────────

def _op_gravity_down(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for c in range(w):
        column = [g.cells[r][c] for r in range(h) if g.cells[r][c] != 0]
        for i, v in enumerate(column):
            out[h - len(column) + i][c] = v
    return Grid(out)


def _op_gravity_up(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for c in range(w):
        column = [g.cells[r][c] for r in range(h) if g.cells[r][c] != 0]
        for i, v in enumerate(column):
            out[i][c] = v
    return Grid(out)


def _op_gravity_left(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        row = [v for v in g.cells[r] if v != 0]
        for i, v in enumerate(row):
            out[r][i] = v
    return Grid(out)


def _op_gravity_right(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        row = [v for v in g.cells[r] if v != 0]
        for i, v in enumerate(row):
            out[r][w - len(row) + i] = v
    return Grid(out)


def _op_gravity_radial(g: Grid, p: dict) -> Grid:
    center_r = float(p.get("center_r", g.shape[0] / 2))
    center_c = float(p.get("center_c", g.shape[1] / 2))
    h, w = g.shape

    # Collect all non-zero cells with their distances
    cells = []
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] != 0:
                dist = _euclidean_distance(r, c, center_r, center_c)
                cells.append((r, c, g.cells[r][c], dist))

    out = [[0] * w for _ in range(h)]
    # Sort by distance and place from center outward
    cells.sort(key=lambda x: x[3])
    placed = [[False] * w for _ in range(h)]

    for r, c, v, _ in cells:
        # Move toward center until hitting something
        curr_r, curr_c = r, c
        while True:
            next_r = curr_r + (1 if center_r > curr_r else -1 if center_r < curr_r else 0)
            next_c = curr_c + (1 if center_c > curr_c else -1 if center_c < curr_c else 0)
            if next_r == curr_r and next_c == curr_c:
                break
            if 0 <= next_r < h and 0 <= next_c < w and not placed[next_r][next_c]:
                curr_r, curr_c = next_r, next_c
            else:
                break
        out[curr_r][curr_c] = v
        placed[curr_r][curr_c] = True
    return Grid(out)


def _op_gravity_diagonal(g: Grid, p: dict) -> Grid:
    direction = p.get("direction", "down-right")  # down-right, down-left, up-right, up-left
    h, w = g.shape

    deltas = {
        "down-right": (1, 1),
        "down-left": (1, -1),
        "up-right": (-1, 1),
        "up-left": (-1, -1)
    }
    dr, dc = deltas.get(direction, (1, 1))

    out = [[0] * w for _ in range(h)]
    placed = [[False] * w for _ in range(h)]

    # Process cells in order opposite to gravity direction
    cells = [(r, c, g.cells[r][c]) for r in range(h) for c in range(w) if g.cells[r][c] != 0]
    if dr > 0:
        cells.sort(key=lambda x: -x[0])
    else:
        cells.sort(key=lambda x: x[0])
    if dc > 0:
        cells.sort(key=lambda x: -x[1])
    else:
        cells.sort(key=lambda x: x[1])

    for r, c, v in cells:
        curr_r, curr_c = r, c
        while True:
            next_r, next_c = curr_r + dr, curr_c + dc
            if not (0 <= next_r < h and 0 <= next_c < w):
                break
            if placed[next_r][next_c]:
                break
            curr_r, curr_c = next_r, next_c
        out[curr_r][curr_c] = v
        placed[curr_r][curr_c] = True
    return Grid(out)


def _op_sand_fall(g: Grid, p: dict) -> Grid:
    """Sand physics: falls down, can fall diagonally if blocked."""
    h, w = g.shape
    out = [row[:] for row in g.cells]
    changed = True
    iterations = 0
    max_iterations = h * w

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for r in range(h - 2, -1, -1):
            for c in range(w):
                if out[r][c] == 0:
                    continue
                # Try to fall down
                if out[r + 1][c] == 0:
                    out[r + 1][c] = out[r][c]
                    out[r][c] = 0
                    changed = True
                # Try diagonal left
                elif c > 0 and out[r + 1][c - 1] == 0:
                    out[r + 1][c - 1] = out[r][c]
                    out[r][c] = 0
                    changed = True
                # Try diagonal right
                elif c < w - 1 and out[r + 1][c + 1] == 0:
                    out[r + 1][c + 1] = out[r][c]
                    out[r][c] = 0
                    changed = True
    return Grid(out)


def _op_water_flow(g: Grid, p: dict) -> Grid:
    """Water physics: falls down, spreads horizontally."""
    h, w = g.shape
    out = [row[:] for row in g.cells]
    changed = True
    iterations = 0
    max_iterations = h * w * 2

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for r in range(h - 2, -1, -1):
            for c in range(w):
                if out[r][c] == 0:
                    continue
                # Fall down
                if out[r + 1][c] == 0:
                    out[r + 1][c] = out[r][c]
                    out[r][c] = 0
                    changed = True
                # Spread left/right if can't fall
                elif c > 0 and out[r][c - 1] == 0:
                    out[r][c - 1] = out[r][c]
                    out[r][c] = 0
                    changed = True
                elif c < w - 1 and out[r][c + 1] == 0:
                    out[r][c + 1] = out[r][c]
                    out[r][c] = 0
                    changed = True
    return Grid(out)


# ── Pattern Operations ────────────────────────────────────────────────────────

def _op_replace_pattern(g: Grid, p: dict) -> Grid:
    from_pat = p.get("from_pattern")  # List of lists
    to_pat = p.get("to_pattern")
    if not from_pat or not to_pat:
        return g.copy()

    ph, pw = len(from_pat), len(from_pat[0])
    h, w = g.shape
    out = [row[:] for row in g.cells]

    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            match = True
            for pr in range(ph):
                for pc in range(pw):
                    if from_pat[pr][pc] != 0 and g.cells[r + pr][c + pc] != from_pat[pr][pc]:
                        match = False
                        break
                if not match:
                    break
            if match:
                th, tw = len(to_pat), len(to_pat[0])
                for pr in range(min(th, h - r)):
                    for pc in range(min(tw, w - c)):
                        if to_pat[pr][pc] != 0:
                            out[r + pr][c + pc] = to_pat[pr][pc]
    return Grid(out)


def _op_find_pattern(g: Grid, p: dict) -> Grid:
    pattern = p.get("pattern")
    mark_colour = int(p.get("mark_colour", 9))
    if not pattern:
        return g.copy()

    ph, pw = len(pattern), len(pattern[0])
    h, w = g.shape
    out = [row[:] for row in g.cells]

    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            match = True
            for pr in range(ph):
                for pc in range(pw):
                    if pattern[pr][pc] != 0 and g.cells[r + pr][c + pc] != pattern[pr][pc]:
                        match = False
                        break
                if not match:
                    break
            if match:
                # Mark center of pattern
                out[r + ph // 2][c + pw // 2] = mark_colour
    return Grid(out)


def _op_repeat_pattern(g: Grid, p: dict) -> Grid:
    pattern = p.get("pattern")
    times = int(p.get("times", 2))
    axis = p.get("axis", "h")
    if not pattern:
        return g.copy()

    ph, pw = len(pattern), len(pattern[0])
    if axis == "h":
        new_w = pw * times
        out = [[0] * new_w for _ in range(ph)]
        for i in range(times):
            for r in range(ph):
                for c in range(pw):
                    out[r][i * pw + c] = pattern[r][c]
    else:
        new_h = ph * times
        out = [[0] * pw for _ in range(new_h)]
        for i in range(times):
            for r in range(ph):
                for c in range(pw):
                    out[i * ph + r][c] = pattern[r][c]
    return Grid(out)


def _op_detect_periodicity(g: Grid, p: dict) -> Grid:
    """Detect and mark periodic patterns."""
    h, w = g.shape
    out = [row[:] for row in g.cells]

    # Check horizontal periodicity
    for period in range(1, w // 2 + 1):
        is_periodic = True
        for r in range(h):
            for c in range(w - period):
                if g.cells[r][c] != g.cells[r][c + period]:
                    is_periodic = False
                    break
            if not is_periodic:
                break
        if is_periodic:
            # Mark periodicity
            for r in range(h):
                out[r][period - 1] = 9
            break
    return Grid(out)


def _op_extend_pattern(g: Grid, p: dict) -> Grid:
    direction = p.get("direction", "right")
    steps = int(p.get("steps", 1))
    h, w = g.shape

    # Detect pattern in specified direction and extend
    if direction == "right":
        new_w = w + steps
        out = [[0] * new_w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                out[r][c] = g.cells[r][c]
            # Extend by repeating last values
            for s in range(steps):
                if w > 0:
                    out[r][w + s] = g.cells[r][(s % w)]
        return Grid(out)
    elif direction == "down":
        new_h = h + steps
        out = [[0] * w for _ in range(new_h)]
        for r in range(h):
            for c in range(w):
                out[r][c] = g.cells[r][c]
            for s in range(steps):
                if h > 0:
                    for c in range(w):
                        out[h + s][c] = g.cells[s % h][c]
        return Grid(out)
    return g.copy()


def _op_mirror_pattern(g: Grid, p: dict) -> Grid:
    axis = p.get("axis", "h")
    h, w = g.shape
    if axis == "h":
        out = [[0] * (w * 2) for _ in range(h)]
        for r in range(h):
            for c in range(w):
                out[r][c] = g.cells[r][c]
                out[r][2 * w - 1 - c] = g.cells[r][c]
        return Grid(out)
    else:
        out = [[0] * w for _ in range(h * 2)]
        for r in range(h):
            for c in range(w):
                out[r][c] = g.cells[r][c]
                out[2 * h - 1 - r][c] = g.cells[r][c]
        return Grid(out)


def _op_tile_pattern(g: Grid, p: dict) -> Grid:
    pattern = p.get("pattern")
    if not pattern:
        return g.copy()
    ph, pw = len(pattern), len(pattern[0])
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = pattern[r % ph][c % pw]
    return Grid(out)


def _op_interpolate(g: Grid, p: dict) -> Grid:
    """Interpolate between two endpoints."""
    r1 = int(p.get("r1", 0))
    c1 = int(p.get("c1", 0))
    r2 = int(p.get("r2", g.shape[0] - 1))
    c2 = int(p.get("c2", g.shape[1] - 1))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    points = _bresenham_line(r1, c1, r2, c2)
    for r, c in points:
        if 0 <= r < h and 0 <= c < w:
            out[r][c] = colour
    return Grid(out)


def _op_complete_rectangle(g: Grid, p: dict) -> Grid:
    """Find incomplete rectangles and complete them."""
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    h, w = g.shape
    out = [row[:] for row in g.cells]

    # Find corners of potential rectangles
    corners = []
    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == dominant:
                corners.append((r, c))

    # Try to find rectangle completions
    for i, (r1, c1) in enumerate(corners):
        for r2, c2 in corners[i+1:]:
            if r1 != r2 and c1 != c2:
                # Check if this could be a rectangle
                # Complete missing edges
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    if out[r1][c] == 0:
                        out[r1][c] = dominant
                    if out[r2][c] == 0:
                        out[r2][c] = dominant
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    if out[r][c1] == 0:
                        out[r][c1] = dominant
                    if out[r][c2] == 0:
                        out[r][c2] = dominant
    return Grid(out)


def _op_fill_rectangle_holes(g: Grid, p: dict) -> Grid:
    """Fill holes inside rectangular regions."""
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()
    h, w = g.shape
    bbox = _get_bbox(g, dominant)
    if bbox is None:
        return g.copy()

    rmin, rmax, cmin, cmax = bbox
    out = [row[:] for row in g.cells]

    # Fill interior of bounding box
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if out[r][c] == 0:
                # Check if surrounded
                surrounded = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if not (rmin <= nr <= rmax and cmin <= nc <= cmax):
                        continue
                    if out[nr][nc] == 0:
                        surrounded = False
                        break
                if surrounded:
                    out[r][c] = dominant
    return Grid(out)


# ── Row/Column Operations ─────────────────────────────────────────────────────

def _op_shift_row(g: Grid, p: dict) -> Grid:
    row_idx = int(p.get("row_idx", 0))
    shift = int(p.get("shift", 1))
    h, w = g.shape
    if not (0 <= row_idx < h):
        return g.copy()
    out = [row[:] for row in g.cells]
    for c in range(w):
        out[row_idx][c] = g.cells[row_idx][(c - shift) % w]
    return Grid(out)


def _op_shift_col(g: Grid, p: dict) -> Grid:
    col_idx = int(p.get("col_idx", 0))
    shift = int(p.get("shift", 1))
    h, w = g.shape
    if not (0 <= col_idx < w):
        return g.copy()
    out = [row[:] for row in g.cells]
    for r in range(h):
        out[r][col_idx] = g.cells[(r - shift) % h][col_idx]
    return Grid(out)


def _op_fill_row(g: Grid, p: dict) -> Grid:
    row_idx = int(p.get("row_idx", 0))
    colour = int(p.get("colour", 1))
    h, w = g.shape
    if not (0 <= row_idx < h):
        return g.copy()
    out = [row[:] for row in g.cells]
    for c in range(w):
        out[row_idx][c] = colour
    return Grid(out)


def _op_fill_col(g: Grid, p: dict) -> Grid:
    col_idx = int(p.get("col_idx", 0))
    colour = int(p.get("colour", 1))
    h, w = g.shape
    if not (0 <= col_idx < w):
        return g.copy()
    out = [row[:] for row in g.cells]
    for r in range(h):
        out[r][col_idx] = colour
    return Grid(out)


def _op_copy_row(g: Grid, p: dict) -> Grid:
    from_idx = int(p.get("from_idx", 0))
    to_idx = int(p.get("to_idx", 1))
    h, w = g.shape
    if not (0 <= from_idx < h and 0 <= to_idx < h):
        return g.copy()
    out = [row[:] for row in g.cells]
    out[to_idx] = g.cells[from_idx][:]
    return Grid(out)


def _op_copy_col(g: Grid, p: dict) -> Grid:
    from_idx = int(p.get("from_idx", 0))
    to_idx = int(p.get("to_idx", 1))
    h, w = g.shape
    if not (0 <= from_idx < w and 0 <= to_idx < w):
        return g.copy()
    out = [row[:] for row in g.cells]
    for r in range(h):
        out[r][to_idx] = g.cells[r][from_idx]
    return Grid(out)


def _op_delete_row(g: Grid, p: dict) -> Grid:
    row_idx = int(p.get("row_idx", 0))
    h, w = g.shape
    if not (0 <= row_idx < h):
        return g.copy()
    cells = [row[:] for row in g.cells if g.cells.index(row) != row_idx]
    if len(cells) == 0:
        cells = [[0] * w]
    return Grid(cells)


def _op_delete_col(g: Grid, p: dict) -> Grid:
    col_idx = int(p.get("col_idx", 0))
    h, w = g.shape
    if not (0 <= col_idx < w):
        return g.copy()
    out = [[v for c, v in enumerate(row) if c != col_idx] for row in g.cells]
    return Grid(out)


def _op_duplicate_row(g: Grid, p: dict) -> Grid:
    row_idx = int(p.get("row_idx", 0))
    h, w = g.shape
    if not (0 <= row_idx < h):
        return g.copy()
    out = [row[:] for row in g.cells]
    out.insert(row_idx + 1, g.cells[row_idx][:])
    return Grid(out)


def _op_duplicate_col(g: Grid, p: dict) -> Grid:
    col_idx = int(p.get("col_idx", 0))
    h, w = g.shape
    if not (0 <= col_idx < w):
        return g.copy()
    out = [[v for c, v in enumerate(row)] for row in g.cells]
    for r in range(h):
        out[r].insert(col_idx + 1, g.cells[r][col_idx])
    return Grid(out)


def _op_reverse_row(g: Grid, p: dict) -> Grid:
    row_idx = int(p.get("row_idx", 0))
    h, w = g.shape
    if not (0 <= row_idx < h):
        return g.copy()
    out = [row[:] for row in g.cells]
    out[row_idx] = out[row_idx][::-1]
    return Grid(out)


def _op_reverse_col(g: Grid, p: dict) -> Grid:
    col_idx = int(p.get("col_idx", 0))
    h, w = g.shape
    if not (0 <= col_idx < w):
        return g.copy()
    out = [row[:] for row in g.cells]
    col_vals = [out[r][col_idx] for r in range(h)]
    for r in range(h):
        out[r][col_idx] = col_vals[h - 1 - r]
    return Grid(out)


def _op_sort_row(g: Grid, p: dict) -> Grid:
    row_idx = int(p.get("row_idx", 0))
    ascending = p.get("ascending", True)
    h, w = g.shape
    if not (0 <= row_idx < h):
        return g.copy()
    out = [row[:] for row in g.cells]
    out[row_idx] = sorted(out[row_idx], reverse=not ascending)
    return Grid(out)


def _op_sort_col(g: Grid, p: dict) -> Grid:
    col_idx = int(p.get("col_idx", 0))
    ascending = p.get("ascending", True)
    h, w = g.shape
    if not (0 <= col_idx < w):
        return g.copy()
    out = [row[:] for row in g.cells]
    col_vals = sorted([out[r][col_idx] for r in range(h)], reverse=not ascending)
    for r in range(h):
        out[r][col_idx] = col_vals[r]
    return Grid(out)


# ── Drawing Primitives ────────────────────────────────────────────────────────

def _op_draw_line(g: Grid, p: dict) -> Grid:
    r1 = int(p.get("r1", 0))
    c1 = int(p.get("c1", 0))
    r2 = int(p.get("r2", g.shape[0] - 1))
    c2 = int(p.get("c2", g.shape[1] - 1))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    points = _bresenham_line(r1, c1, r2, c2)
    for r, c in points:
        if 0 <= r < h and 0 <= c < w:
            out[r][c] = colour
    return Grid(out)


def _op_draw_rect_outline(g: Grid, p: dict) -> Grid:
    r1 = int(p.get("r1", 0))
    c1 = int(p.get("c1", 0))
    r2 = int(p.get("r2", 5))
    c2 = int(p.get("c2", 5))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(max(0, r1), min(h, r2 + 1)):
        for c in range(max(0, c1), min(w, c2 + 1)):
            if r == r1 or r == r2 or c == c1 or c == c2:
                out[r][c] = colour
    return Grid(out)


def _op_draw_rect_fill(g: Grid, p: dict) -> Grid:
    r1 = int(p.get("r1", 0))
    c1 = int(p.get("c1", 0))
    r2 = int(p.get("r2", 5))
    c2 = int(p.get("c2", 5))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(max(0, r1), min(h, r2 + 1)):
        for c in range(max(0, c1), min(w, c2 + 1)):
            out[r][c] = colour
    return Grid(out)


def _op_draw_circle(g: Grid, p: dict) -> Grid:
    center_r = int(p.get("center_r", g.shape[0] // 2))
    center_c = int(p.get("center_c", g.shape[1] // 2))
    radius = int(p.get("radius", 3))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            dist = _euclidean_distance(r, c, center_r, center_c)
            if abs(dist - radius) < 1:
                out[r][c] = colour
    return Grid(out)


def _op_draw_dot(g: Grid, p: dict) -> Grid:
    r = int(p.get("r", 0))
    c = int(p.get("c", 0))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    if 0 <= r < h and 0 <= c < w:
        out[r][c] = colour
    return Grid(out)


def _op_draw_cross(g: Grid, p: dict) -> Grid:
    r = int(p.get("r", g.shape[0] // 2))
    c = int(p.get("c", g.shape[1] // 2))
    size = int(p.get("size", 2))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for dr in range(-size, size + 1):
        for dc in range(-size, size + 1):
            if dr == 0 or dc == 0:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = colour
    return Grid(out)


def _op_draw_diagonal(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", 1))
    direction = p.get("direction", "both")  # both, main, anti

    h, w = g.shape
    out = [row[:] for row in g.cells]
    min_dim = min(h, w)

    if direction in ["both", "main"]:
        for i in range(min_dim):
            out[i][i] = colour
    if direction in ["both", "anti"]:
        for i in range(min_dim):
            out[i][w - 1 - i] = colour
    return Grid(out)


def _op_draw_border(g: Grid, p: dict) -> Grid:
    thickness = int(p.get("thickness", 1))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(h):
        for c in range(w):
            if r < thickness or r >= h - thickness or c < thickness or c >= w - thickness:
                out[r][c] = colour
    return Grid(out)


def _op_draw_grid(g: Grid, p: dict) -> Grid:
    cell_size = int(p.get("cell_size", 3))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(0, h, cell_size):
        for c in range(w):
            out[r][c] = colour
    for c in range(0, w, cell_size):
        for r in range(h):
            out[r][c] = colour
    return Grid(out)


def _op_draw_frame(g: Grid, p: dict) -> Grid:
    margin = int(p.get("margin", 1))
    colour = int(p.get("colour", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for r in range(margin):
        for c in range(w):
            out[r][c] = colour
            out[h - 1 - r][c] = colour
    for c in range(margin):
        for r in range(h):
            out[r][c] = colour
            out[r][w - 1 - c] = colour
    return Grid(out)


# ── Tiling & Replication ──────────────────────────────────────────────────────

def _op_tile_2x(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * (w * 2) for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            out[r][c] = g.cells[r][c]
            out[r][c + w] = g.cells[r][c]
            out[r + h][c] = g.cells[r][c]
            out[r + h][c + w] = g.cells[r][c]
    return Grid(out)


def _op_tile_3x(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * (w * 3) for _ in range(h * 3)]
    for r in range(h):
        for c in range(w):
            for dr in range(3):
                for dc in range(3):
                    out[r + dr * h][c + dc * w] = g.cells[r][c]
    return Grid(out)


def _op_tile_nx(g: Grid, p: dict) -> Grid:
    h_factor = int(p.get("h_factor", 2))
    w_factor = int(p.get("w_factor", 2))
    h, w = g.shape
    out = [[0] * (w * w_factor) for _ in range(h * h_factor)]
    for r in range(h):
        for c in range(w):
            for dr in range(h_factor):
                for dc in range(w_factor):
                    out[r + dr * h][c + dc * w] = g.cells[r][c]
    return Grid(out)


def _op_replicate_obj(g: Grid, p: dict) -> Grid:
    count = int(p.get("count", 2))
    axis = p.get("axis", "h")
    step = int(p.get("step", 1))

    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()

    bbox = _get_bbox(g, dominant)
    if bbox is None:
        return g.copy()

    rmin, rmax, cmin, cmax = bbox
    sub_h = rmax - rmin + 1
    sub_w = cmax - cmin + 1
    sub = [[g.cells[r][c] for c in range(cmin, cmax + 1)] for r in range(rmin, rmax + 1)]

    if axis == "h":
        new_w = sub_w * count + step * (count - 1)
        new_h = sub_h
        out = [[0] * new_w for _ in range(new_h)]
        for i in range(count):
            off = i * (sub_w + step)
            for r in range(sub_h):
                for c in range(sub_w):
                    out[r][off + c] = sub[r][c]
    else:
        new_h = sub_h * count + step * (count - 1)
        new_w = sub_w
        out = [[0] * new_w for _ in range(new_h)]
        for i in range(count):
            off = i * (sub_h + step)
            for r in range(sub_h):
                for c in range(sub_w):
                    out[off + r][c] = sub[r][c]
    return Grid(out)


def _op_count_fill(g: Grid, p: dict) -> Grid:
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()

    palette = g.palette()
    n = sum(len(_connected_components(g, c)) for c in palette if c != 0)

    h, w = g.shape
    out = [row[:] for row in g.cells]
    for c in range(min(n, w)):
        out[h - 1][c] = dominant
    return Grid(out)


def _op_mirror_tile(g: Grid, p: dict) -> Grid:
    axis = p.get("axis", "h")
    h, w = g.shape
    if axis == "h":
        out = [[0] * (w * 2) for _ in range(h)]
        for r in range(h):
            for c in range(w):
                out[r][c] = g.cells[r][c]
                out[r][2 * w - 1 - c] = g.cells[r][c]
        return Grid(out)
    else:
        out = [[0] * w for _ in range(h * 2)]
        for r in range(h):
            for c in range(w):
                out[r][c] = g.cells[r][c]
                out[2 * h - 1 - r][c] = g.cells[r][c]
        return Grid(out)


def _op_stamp(g: Grid, p: dict) -> Grid:
    pattern = p.get("pattern")
    positions = p.get("positions", [(0, 0)])
    if not pattern:
        return g.copy()

    ph, pw = len(pattern), len(pattern[0])
    h, w = g.shape
    out = [row[:] for row in g.cells]

    for pr, pc in positions:
        for r in range(ph):
            for c in range(pw):
                nr, nc = pr + r, pc + c
                if 0 <= nr < h and 0 <= nc < w and pattern[r][c] != 0:
                    out[nr][nc] = pattern[r][c]
    return Grid(out)


def _op_spread(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", 1))
    direction = p.get("direction", "all")
    distance = int(p.get("distance", 1))

    h, w = g.shape
    out = [row[:] for row in g.cells]

    deltas = {
        "all": [(-1,0),(1,0),(0,-1),(0,1)],
        "up": [(-1, 0)],
        "down": [(1, 0)],
        "left": [(0, -1)],
        "right": [(0, 1)]
    }

    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == colour:
                for dr, dc in deltas.get(direction, deltas["all"]):
                    for d in range(1, distance + 1):
                        nr, nc = r + dr * d, c + dc * d
                        if 0 <= nr < h and 0 <= nc < w:
                            out[nr][nc] = colour
                        else:
                            break
    return Grid(out)


# ── Cropping & Padding ────────────────────────────────────────────────────────

def _op_crop_to_nonzero(g: Grid, p: dict) -> Grid:
    bbox = _get_bbox(g)
    if bbox is None:
        return g.copy()
    return _extract_subgrid(g, bbox)


def _op_crop_to_colour(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", 1))
    bbox = _get_bbox(g, colour)
    if bbox is None:
        return g.copy()
    return _extract_subgrid(g, bbox)


def _op_crop_to_center(g: Grid, p: dict) -> Grid:
    target_h = int(p.get("target_h", 5))
    target_w = int(p.get("target_w", 5))
    h, w = g.shape

    r_start = max(0, (h - target_h) // 2)
    c_start = max(0, (w - target_w) // 2)
    r_end = min(h, r_start + target_h)
    c_end = min(w, c_start + target_w)

    cells = [[g.cells[r][c] for c in range(c_start, c_end)] for r in range(r_start, r_end)]
    return Grid(cells)


def _op_crop_to_corner(g: Grid, p: dict) -> Grid:
    corner = p.get("corner", "top-left")
    size = int(p.get("size", 5))
    h, w = g.shape

    if corner == "top-left":
        r_start, c_start = 0, 0
    elif corner == "top-right":
        r_start, c_start = 0, max(0, w - size)
    elif corner == "bottom-left":
        r_start, c_start = max(0, h - size), 0
    else:  # bottom-right
        r_start, c_start = max(0, h - size), max(0, w - size)

    r_end = min(h, r_start + size)
    c_end = min(w, c_start + size)

    cells = [[g.cells[r][c] for c in range(c_start, c_end)] for r in range(r_start, r_end)]
    return Grid(cells)


def _op_pad_top(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    colour = int(p.get("colour", 0))
    h, w = g.shape
    out = [[colour] * w for _ in range(n)] + [row[:] for row in g.cells]
    return Grid(out)


def _op_pad_bottom(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    colour = int(p.get("colour", 0))
    h, w = g.shape
    out = [row[:] for row in g.cells] + [[colour] * w for _ in range(n)]
    return Grid(out)


def _op_pad_left(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    colour = int(p.get("colour", 0))
    h, w = g.shape
    out = [[colour] * n + row[:] for row in g.cells]
    return Grid(out)


def _op_pad_right(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    colour = int(p.get("colour", 0))
    h, w = g.shape
    out = [row[:] + [colour] * n for row in g.cells]
    return Grid(out)


def _op_pad_all(g: Grid, p: dict) -> Grid:
    n = int(p.get("n", 1))
    colour = int(p.get("colour", 0))
    h, w = g.shape
    new_w = w + 2 * n
    out = [[colour] * new_w for _ in range(n)]
    for row in g.cells:
        out.append([colour] * n + row[:] + [colour] * n)
    for _ in range(n):
        out.append([colour] * new_w)
    return Grid(out)


# ── Symmetry Operations ───────────────────────────────────────────────────────

def _op_make_symmetric_h(g: Grid, p: dict) -> Grid:
    """Force horizontal symmetry by taking union with mirror."""
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w // 2 + 1):
            v = max(g.cells[r][c], g.cells[r][w - 1 - c])
            out[r][c] = v
            out[r][w - 1 - c] = v
    return Grid(out)


def _op_make_symmetric_v(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h // 2 + 1):
        for c in range(w):
            v = max(g.cells[r][c], g.cells[h - 1 - r][c])
            out[r][c] = v
            out[h - 1 - r][c] = v
    return Grid(out)


def _op_make_symmetric_d1(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    size = max(h, w)
    out = [[0] * size for _ in range(size)]
    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            out[r][c] = max(out[r][c], v)
            if c < size and r < size:
                out[c][r] = max(out[c][r], v)
    return Grid(out)


def _op_make_symmetric_d2(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = max(g.cells[r][c], g.cells[h - 1 - c][w - 1 - r])
            out[r][c] = v
            out[h - 1 - c][w - 1 - r] = v
    return Grid(out)


def _op_make_symmetric_rot(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            out[r][c] = max(out[r][c], v)
            out[h - 1 - r][w - 1 - c] = max(out[h - 1 - r][w - 1 - c], v)
    return Grid(out)


def _op_check_symmetry(g: Grid, p: dict) -> Grid:
    sym_type = p.get("type", "all")
    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    results = {
        "h": _has_symmetry_h(g),
        "v": _has_symmetry_v(g),
        "d1": _has_symmetry_d1(g),
        "d2": _has_symmetry_d2(g)
    }

    # Mark result in output
    if sym_type == "all":
        for i, (st, result) in enumerate(results.items()):
            if result:
                out[0][i] = 1
    elif sym_type in results and results[sym_type]:
        out[0][0] = 1

    return Grid(out)


def _op_symmetrize_union(g: Grid, p: dict) -> Grid:
    axis = p.get("axis", "h")
    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    if axis == "h":
        for r in range(h):
            for c in range(w):
                out[r][c] = max(g.cells[r][c], g.cells[r][w - 1 - c])
    else:
        for r in range(h):
            for c in range(w):
                out[r][c] = max(g.cells[r][c], g.cells[h - 1 - r][c])
    return Grid(out)


def _op_symmetrize_intersect(g: Grid, p: dict) -> Grid:
    axis = p.get("axis", "h")
    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    if axis == "h":
        for r in range(h):
            for c in range(w):
                if g.cells[r][c] == g.cells[r][w - 1 - c] and g.cells[r][c] != 0:
                    out[r][c] = g.cells[r][c]
    else:
        for r in range(h):
            for c in range(w):
                if g.cells[r][c] == g.cells[h - 1 - r][c] and g.cells[r][c] != 0:
                    out[r][c] = g.cells[r][c]
    return Grid(out)


# ── Counting & Measurement ────────────────────────────────────────────────────

def _op_count_objects(g: Grid, p: dict) -> Grid:
    colour = p.get("colour")
    connectivity = int(p.get("connectivity", 4))

    if colour is not None:
        count = len(_connected_components(g, colour, connectivity))
    else:
        palette = g.palette()
        count = sum(len(_connected_components(g, c, connectivity)) for c in palette if c != 0)

    # Encode count as unary in first row
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for i in range(min(count, w)):
        out[0][i] = 1
    return Grid(out)


def _op_measure_bbox(g: Grid, p: dict) -> Grid:
    colour = p.get("colour")
    bbox = _get_bbox(g, colour)
    if bbox is None:
        return g.copy()

    rmin, rmax, cmin, cmax = bbox
    height = rmax - rmin + 1
    width = cmax - cmin + 1

    # Encode dimensions
    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    for i in range(min(height, h)):
        out[i][0] = 1
    for i in range(min(width, w)):
        out[0][i + 1] = 2
    return Grid(out)


def _op_mark_centroid(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))
    mark_colour = int(p.get("mark_colour", 9))

    if colour == 0:
        return g.copy()

    components = _connected_components(g, colour)
    if not components:
        return g.copy()

    h, w = g.shape
    out = [row[:] for row in g.cells]

    for comp in components:
        avg_r = sum(r for r, c in comp) / len(comp)
        avg_c = sum(c for r, c in comp) / len(comp)
        centroid_r, centroid_c = int(round(avg_r)), int(round(avg_c))
        if 0 <= centroid_r < h and 0 <= centroid_c < w:
            out[centroid_r][centroid_c] = mark_colour
    return Grid(out)


def _op_mark_extrema(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))
    mark_colour = int(p.get("mark_colour", 9))

    if colour == 0:
        return g.copy()

    components = _connected_components(g, colour)
    if not components:
        return g.copy()

    h, w = g.shape
    out = [row[:] for row in g.cells]

    for comp in components:
        # Find extrema
        min_r = min(comp, key=lambda x: x[0])
        max_r = max(comp, key=lambda x: x[0])
        min_c = min(comp, key=lambda x: x[1])
        max_c = max(comp, key=lambda x: x[1])

        for r, c in [min_r, max_r, min_c, max_c]:
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = mark_colour
    return Grid(out)


def _op_encode_size(g: Grid, p: dict) -> Grid:
    """Encode bounding box size as a visual pattern."""
    colour = p.get("colour")
    bbox = _get_bbox(g, colour)
    if bbox is None:
        return g.copy()

    rmin, rmax, cmin, cmax = bbox
    height = rmax - rmin + 1
    width = cmax - cmin + 1

    # Create encoding pattern
    h, w = max(height, 5), max(width, 5)
    out = [[0] * w for _ in range(h)]

    # Height bar on left
    for r in range(min(height, h)):
        out[r][0] = 1
    # Width bar on top
    for c in range(min(width, w)):
        out[0][c + 1] = 2

    return Grid(out)


def _op_histogram(g: Grid, p: dict) -> Grid:
    """Create a colour histogram visualization."""
    palette = g.palette()
    counts = {c: _count_cells_of_colour(g, c) for c in palette if c != 0}

    if not counts:
        return g.copy()

    max_count = max(counts.values())
    h, w = 10, len(counts) + 2
    out = [[0] * w for _ in range(h)]

    for i, (colour, count) in enumerate(sorted(counts.items())):
        bar_height = int(count / max_count * (h - 1)) if max_count > 0 else 0
        for r in range(h - 1, h - 1 - bar_height, -1):
            out[r][i + 1] = colour
        out[h - 1][i + 1] = colour  # Base

    return Grid(out)


def _op_label_components(g: Grid, p: dict) -> Grid:
    connectivity = int(p.get("connectivity", 4))
    palette = g.palette()

    h, w = g.shape
    out = [[0] * w for _ in range(h)]
    label = 1

    for colour in palette:
        if colour == 0:
            continue
        components = _connected_components(g, colour, connectivity)
        for comp in components:
            for r, c in comp:
                out[r][c] = label
            label += 1

    return Grid(out)


def _op_rank_by_size(g: Grid, p: dict) -> Grid:
    colour = int(p.get("colour", g.dominant_colour()))

    if colour == 0:
        return g.copy()

    components = _connected_components(g, colour)
    if not components:
        return g.copy()

    # Sort by size descending
    components.sort(key=len, reverse=True)

    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    for rank, comp in enumerate(components):
        rank_colour = min(rank + 1, 9)
        for r, c in comp:
            out[r][c] = rank_colour

    return Grid(out)


# ── Noise & Variation ─────────────────────────────────────────────────────────

def _op_add_noise(g: Grid, p: dict) -> Grid:
    density = float(p.get("density", 0.1))
    palette = p.get("palette", list(range(1, 10)))

    import random
    seed = p.get("seed")
    if seed is not None:
        random.seed(seed)

    h, w = g.shape
    out = [row[:] for row in g.cells]

    for r in range(h):
        for c in range(w):
            if random.random() < density:
                out[r][c] = random.choice(palette)

    return Grid(out)


def _op_remove_singletons(g: Grid, p: dict) -> Grid:
    h, w = g.shape
    out = [row[:] for row in g.cells]

    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == 0:
                continue
            has_neighbour = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g.cells[nr][nc] == g.cells[r][c]:
                    has_neighbour = True
                    break
            if not has_neighbour:
                out[r][c] = 0

    return Grid(out)


def _op_smooth(g: Grid, p: dict) -> Grid:
    iterations = int(p.get("iterations", 1))

    result = g.copy()
    for _ in range(iterations):
        h, w = result.shape
        out = [row[:] for row in result.cells]

        for r in range(h):
            for c in range(w):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(result.cells[nr][nc])

                if neighbors:
                    # Majority vote
                    from collections import Counter
                    most_common = Counter(neighbors).most_common(1)[0][0]
                    out[r][c] = most_common

        result = Grid(out)

    return result


def _op_perturb(g: Grid, p: dict) -> Grid:
    max_shift = int(p.get("max_shift", 1))

    import random
    seed = p.get("seed")
    if seed is not None:
        random.seed(seed)

    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == 0:
                continue
            dr = random.randint(-max_shift, max_shift)
            dc = random.randint(-max_shift, max_shift)
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = g.cells[r][c]

    return Grid(out)


# ── Composite Operations ──────────────────────────────────────────────────────

def _op_extract_and_center(g: Grid, p: dict) -> Grid:
    """Extract largest object and center it in the grid."""
    dominant = g.dominant_colour()
    if dominant == 0:
        return g.copy()

    components = _connected_components(g, dominant)
    if not components:
        return g.copy()

    largest = max(components, key=len)
    h, w = g.shape

    # Get bbox of largest
    rmin = min(r for r, c in largest)
    rmax = max(r for r, c in largest)
    cmin = min(c for r, c in largest)
    cmax = max(c for r, c in largest)

    obj_h = rmax - rmin + 1
    obj_w = cmax - cmin + 1

    # Center position
    center_r = (h - obj_h) // 2
    center_c = (w - obj_w) // 2

    out = [[0] * w for _ in range(h)]
    for r, c in largest:
        out[center_r + r - rmin][center_c + c - cmin] = dominant

    return Grid(out)


def _op_colorize_regions(g: Grid, p: dict) -> Grid:
    """Colorize different regions with different colours."""
    colour_map = p.get("colour_map", {})
    connectivity = int(p.get("connectivity", 4))

    palette = g.palette()
    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    region_id = 0
    for colour in palette:
        if colour == 0:
            continue
        components = _connected_components(g, colour, connectivity)
        for comp in components:
            new_colour = colour_map.get(region_id, colour)
            for r, c in comp:
                out[r][c] = new_colour
            region_id += 1

    return Grid(out)


def _op_connect_nearest(g: Grid, p: dict) -> Grid:
    """Connect nearest objects with lines."""
    colour = int(p.get("colour", g.dominant_colour()))
    line_colour = int(p.get("line_colour", 9))

    if colour == 0:
        return g.copy()

    components = _connected_components(g, colour)
    if len(components) < 2:
        return g.copy()

    h, w = g.shape
    out = [row[:] for row in g.cells]

    # Calculate centroids
    centroids = []
    for comp in components:
        avg_r = sum(r for r, c in comp) / len(comp)
        avg_c = sum(c for r, c in comp) / len(comp)
        centroids.append((avg_r, avg_c, comp))

    # Connect nearest pairs
    connected = set()
    for i, (r1, c1, _) in enumerate(centroids):
        for j, (r2, c2, _) in enumerate(centroids):
            if i >= j:
                continue
            dist = _euclidean_distance(r1, c1, r2, c2)
            connected.add((i, j, dist))

    connected = sorted(connected, key=lambda x: x[2])

    # Draw lines between nearest pairs
    for i, j, _ in connected[:len(components) - 1]:
        r1, c1 = int(centroids[i][0]), int(centroids[i][1])
        r2, c2 = int(centroids[j][0]), int(centroids[j][1])
        points = _bresenham_line(r1, c1, r2, c2)
        for r, c in points:
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = line_colour

    return Grid(out)


def _op_fill_between(g: Grid, p: dict) -> Grid:
    """Fill space between two colours."""
    c1 = int(p.get("c1", 1))
    c2 = int(p.get("c2", 2))
    fill_colour = int(p.get("fill_colour", 3))

    h, w = g.shape
    out = [row[:] for row in g.cells]

    # Find cells of each colour
    cells1 = [(r, c) for r in range(h) for c in range(w) if g.cells[r][c] == c1]
    cells2 = [(r, c) for r in range(h) for c in range(w) if g.cells[r][c] == c2]

    if not cells1 or not cells2:
        return g.copy()

    # Fill between nearest pairs
    for r1, c1_val in cells1:
        for r2, c2_val in cells2:
            points = _bresenham_line(r1, c1_val, r2, c2_val)
            for r, c in points:
                if 0 <= r < h and 0 <= c < w:
                    out[r][c] = fill_colour

    return Grid(out)


def _op_propagate_colour(g: Grid, p: dict) -> Grid:
    """Propagate source colour to adjacent target colour regions."""
    source = int(p.get("source_colour", 1))
    target = int(p.get("target_colour", 2))

    h, w = g.shape
    out = [row[:] for row in g.cells]
    changed = True

    while changed:
        changed = False
        for r in range(h):
            for c in range(w):
                if out[r][c] == source:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == target:
                            out[nr][nc] = source
                            changed = True

    return Grid(out)


def _op_gradient_fill(g: Grid, p: dict) -> Grid:
    """Fill with gradient between colours."""
    direction = p.get("direction", "horizontal")
    colours = p.get("colours", [1, 5, 9])

    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    if direction == "horizontal":
        for c in range(w):
            t = c / (w - 1) if w > 1 else 0
            idx = t * (len(colours) - 1)
            lower = int(idx)
            upper = min(lower + 1, len(colours) - 1)
            frac = idx - lower
            colour = int(colours[lower] + frac * (colours[upper] - colours[lower]))
            for r in range(h):
                out[r][c] = colour
    else:  # vertical
        for r in range(h):
            t = r / (h - 1) if h > 1 else 0
            idx = t * (len(colours) - 1)
            lower = int(idx)
            upper = min(lower + 1, len(colours) - 1)
            frac = idx - lower
            colour = int(colours[lower] + frac * (colours[upper] - colours[lower]))
            for c in range(w):
                out[r][c] = colour

    return Grid(out)


def _op_contour(g: Grid, p: dict) -> Grid:
    """Draw contour lines around regions."""
    colour = int(p.get("colour", g.dominant_colour()))
    contour_colour = int(p.get("contour_colour", 9))

    if colour == 0:
        return g.copy()

    h, w = g.shape
    out = [row[:] for row in g.cells]

    for r in range(h):
        for c in range(w):
            if g.cells[r][c] == colour:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or g.cells[nr][nc] != colour:
                        out[r][c] = contour_colour
                        break

    return Grid(out)


def _op_blit(g: Grid, p: dict) -> Grid:
    """Blit another grid onto this one."""
    src_grid = p.get("src_grid")
    dest_r = int(p.get("dest_r", 0))
    dest_c = int(p.get("dest_c", 0))

    if src_grid is None:
        return g.copy()

    h, w = g.shape
    sh, sw = src_grid.shape
    out = [row[:] for row in g.cells]

    for r in range(sh):
        for c in range(sw):
            dr, dc = dest_r + r, dest_c + c
            if 0 <= dr < h and 0 <= dc < w and src_grid.cells[r][c] != 0:
                out[dr][dc] = src_grid.cells[r][c]

    return Grid(out)


def _op_mask(g: Grid, p: dict) -> Grid:
    """Apply mask based on colour."""
    mask_colour = int(p.get("mask_colour", 0))

    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if g.cells[r][c] != mask_colour:
                out[r][c] = g.cells[r][c]

    return Grid(out)


def _op_composite(g: Grid, p: dict) -> Grid:
    """Composite overlay grid with mode."""
    overlay = p.get("overlay_grid")
    mode = p.get("mode", "over")  # over, under, multiply, screen

    if overlay is None:
        return g.copy()

    h, w = g.shape
    oh, ow = overlay.shape
    out = [row[:] for row in g.cells]

    for r in range(min(h, oh)):
        for c in range(min(w, ow)):
            base = g.cells[r][c]
            over = overlay.cells[r][c]

            if mode == "over" and over != 0:
                out[r][c] = over
            elif mode == "under" and base == 0 and over != 0:
                out[r][c] = over
            elif mode == "multiply":
                out[r][c] = (base * over) // 9
            elif mode == "screen":
                out[r][c] = 9 - ((9 - base) * (9 - over)) // 9

    return Grid(out)


def _op_layers_merge(g: Grid, p: dict) -> Grid:
    """Merge multiple layer grids."""
    layers = p.get("layer_grids", [])
    mode = p.get("mode", "max")  # max, min, avg, sum

    if not layers:
        return g.copy()

    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            values = [g.cells[r][c]]
            for layer in layers:
                lh, lw = layer.shape
                if r < lh and c < lw:
                    values.append(layer.cells[r][c])

            if mode == "max":
                out[r][c] = max(values)
            elif mode == "min":
                out[r][c] = min(values)
            elif mode == "avg":
                out[r][c] = sum(values) // len(values)
            elif mode == "sum":
                out[r][c] = min(9, sum(values))

    return Grid(out)


def _op_channel_extract(g: Grid, p: dict) -> Grid:
    """Extract channel based on colour range."""
    colour_channel = p.get("colour_channel", "low")  # low, mid, high

    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            v = g.cells[r][c]
            if colour_channel == "low" and 1 <= v <= 3:
                out[r][c] = v
            elif colour_channel == "mid" and 4 <= v <= 6:
                out[r][c] = v
            elif colour_channel == "high" and 7 <= v <= 9:
                out[r][c] = v

    return Grid(out)


def _op_multiply_brightness(g: Grid, p: dict) -> Grid:
    """Multiply brightness by factor."""
    factor = float(p.get("factor", 1.5))

    h, w = g.shape
    out = [[0] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            out[r][c] = min(9, int(g.cells[r][c] * factor))

    return Grid(out)


# ══════════════════════════════════════════════════════════════════════════════
# OPERATOR DISPATCH TABLE
# ══════════════════════════════════════════════════════════════════════════════

OP_IMPL: Dict[Ops, Callable[[Grid, dict], Grid]] = {
    # Identity & Basic
    Ops.IDENTITY: _op_identity,
    Ops.CLEAR: _op_clear,
    Ops.INVERT_BG: _op_invert_bg,

    # Geometric Transforms
    Ops.ROTATE_90: _op_rotate_90,
    Ops.ROTATE_180: _op_rotate_180,
    Ops.ROTATE_270: _op_rotate_270,
    Ops.FLIP_H: _op_flip_h,
    Ops.FLIP_V: _op_flip_v,
    Ops.TRANSPOSE: _op_transpose,
    Ops.ANTI_TRANSPOSE: _op_anti_transpose,
    Ops.ROTATE_ARBITRARY: _op_rotate_arbitrary,

    # Scaling & Resizing
    Ops.SCALE_2X: _op_scale_2x,
    Ops.SCALE_3X: _op_scale_3x,
    Ops.SCALE_HALF: _op_scale_half,
    Ops.SCALE_NX: _op_scale_nx,
    Ops.RESIZE: _op_resize,
    Ops.ASPECT_FILL: _op_aspect_fill,

    # Translation & Shifting
    Ops.TRANSLATE: _op_translate,
    Ops.SHIFT_UP: _op_shift_up,
    Ops.SHIFT_DOWN: _op_shift_down,
    Ops.SHIFT_LEFT: _op_shift_left,
    Ops.SHIFT_RIGHT: _op_shift_right,
    Ops.WRAP_SHIFT_H: _op_wrap_shift_h,
    Ops.WRAP_SHIFT_V: _op_wrap_shift_v,
    Ops.CYCLE_ROWS: _op_cycle_rows,

    # Simple Recolouring
    Ops.RECOLOUR: _op_recolour,
    Ops.SWAP_COLOURS: _op_swap_colours,
    Ops.PALETTE_CYCLE: _op_palette_cycle,
    Ops.NORMALIZE: _op_normalize,
    Ops.GREYSCALE: _op_greyscale,
    Ops.INVERT_COLOURS: _op_invert_colours,
    Ops.RANDOMIZE: _op_randomize,
    Ops.QUANTIZE: _op_quantize,
    Ops.COLOUR_TO_INTENSITY: _op_colour_to_intensity,
    Ops.BINARY_THRESHOLD: _op_binary_threshold,
    Ops.GRADIENT_MAP: _op_gradient_map,
    Ops.HIGHLIGHT: _op_highlight,

    # Conditional Recolouring
    Ops.RECOLOUR_IF_NEIGHBOUR: _op_recolour_if_neighbour,
    Ops.RECOLOUR_IF_BORDER: _op_recolour_if_border,
    Ops.RECOLOUR_IF_CORNER: _op_recolour_if_corner,
    Ops.RECOLOUR_IF_ISOLATED: _op_recolour_if_isolated,
    Ops.RECOLOUR_IF_CROWDED: _op_recolour_if_crowded,
    Ops.RECOLOUR_INTERIOR: _op_recolour_interior,
    Ops.RECOLOUR_EXTERIOR: _op_recolour_exterior,
    Ops.RECOLOUR_EDGE_ADJACENT: _op_recolour_edge_adjacent,
    Ops.RECOLOUR_DIAGONAL_ONLY: _op_recolour_diagonal_only,
    Ops.RECOLOUR_BY_DENSITY: _op_recolour_by_density,

    # Set Operations
    Ops.SET_INTERSECT: _op_set_intersect,
    Ops.SET_DIFFERENCE: _op_set_difference,
    Ops.SET_UNION: _op_set_union,
    Ops.SET_XOR: _op_set_xor,
    Ops.SET_COMPLEMENT: _op_set_complement,
    Ops.DILATION: _op_dilation,

    # Object Extraction
    Ops.EXTRACT_LARGEST: _op_extract_largest,
    Ops.EXTRACT_SMALLEST: _op_extract_smallest,
    Ops.EXTRACT_COLOUR: _op_extract_colour,
    Ops.EXTRACT_NTH: _op_extract_nth,
    Ops.EXTRACT_TOP_LEFT: _op_extract_top_left,
    Ops.EXTRACT_CENTER: _op_extract_center,
    Ops.EXTRACT_ALL_OBJECTS: _op_extract_all_objects,
    Ops.EXTRACT_BBOX: _op_extract_bbox,

    # Connectivity & Morphology
    Ops.FILL_INTERIOR: _op_fill_interior,
    Ops.OUTLINE: _op_outline,
    Ops.DILATE_OP: _op_dilate_op,
    Ops.ERODE_OP: _op_erode_op,
    Ops.OPEN_MORPH: _op_open_morph,
    Ops.CLOSE_MORPH: _op_close_morph,
    Ops.SKELETONIZE: _op_skeletonize,
    Ops.THICKEN: _op_thicken,
    Ops.FILL_HOLES: _op_fill_holes,

    # Gravity & Physics
    Ops.GRAVITY_DOWN: _op_gravity_down,
    Ops.GRAVITY_UP: _op_gravity_up,
    Ops.GRAVITY_LEFT: _op_gravity_left,
    Ops.GRAVITY_RIGHT: _op_gravity_right,
    Ops.GRAVITY_RADIAL: _op_gravity_radial,
    Ops.GRAVITY_DIAGONAL: _op_gravity_diagonal,
    Ops.SAND_FALL: _op_sand_fall,
    Ops.WATER_FLOW: _op_water_flow,

    # Pattern Operations
    Ops.REPLACE_PATTERN: _op_replace_pattern,
    Ops.FIND_PATTERN: _op_find_pattern,
    Ops.REPEAT_PATTERN: _op_repeat_pattern,
    Ops.DETECT_PERIODICITY: _op_detect_periodicity,
    Ops.EXTEND_PATTERN: _op_extend_pattern,
    Ops.MIRROR_PATTERN: _op_mirror_pattern,
    Ops.TILE_PATTERN: _op_tile_pattern,
    Ops.INTERPOLATE: _op_interpolate,
    Ops.COMPLETE_RECTANGLE: _op_complete_rectangle,
    Ops.FILL_RECTANGLE_HOLES: _op_fill_rectangle_holes,

    # Row/Column Operations
    Ops.SHIFT_ROW: _op_shift_row,
    Ops.SHIFT_COL: _op_shift_col,
    Ops.FILL_ROW: _op_fill_row,
    Ops.FILL_COL: _op_fill_col,
    Ops.COPY_ROW: _op_copy_row,
    Ops.COPY_COL: _op_copy_col,
    Ops.DELETE_ROW: _op_delete_row,
    Ops.DELETE_COL: _op_delete_col,
    Ops.DUPLICATE_ROW: _op_duplicate_row,
    Ops.DUPLICATE_COL: _op_duplicate_col,
    Ops.REVERSE_ROW: _op_reverse_row,
    Ops.REVERSE_COL: _op_reverse_col,
    Ops.SORT_ROW: _op_sort_row,
    Ops.SORT_COL: _op_sort_col,

    # Drawing Primitives
    Ops.DRAW_LINE: _op_draw_line,
    Ops.DRAW_RECT_OUTLINE: _op_draw_rect_outline,
    Ops.DRAW_RECT_FILL: _op_draw_rect_fill,
    Ops.DRAW_CIRCLE: _op_draw_circle,
    Ops.DRAW_DOT: _op_draw_dot,
    Ops.DRAW_CROSS: _op_draw_cross,
    Ops.DRAW_DIAGONAL: _op_draw_diagonal,
    Ops.DRAW_BORDER: _op_draw_border,
    Ops.DRAW_GRID: _op_draw_grid,
    Ops.DRAW_FRAME: _op_draw_frame,

    # Tiling & Replication
    Ops.TILE_2X: _op_tile_2x,
    Ops.TILE_3X: _op_tile_3x,
    Ops.TILE_NX: _op_tile_nx,
    Ops.REPLICATE_OBJ: _op_replicate_obj,
    Ops.COUNT_FILL: _op_count_fill,
    Ops.MIRROR_TILE: _op_mirror_tile,
    Ops.STAMP: _op_stamp,
    Ops.SPREAD: _op_spread,

    # Cropping & Padding
    Ops.CROP_TO_NONZERO: _op_crop_to_nonzero,
    Ops.CROP_TO_COLOUR: _op_crop_to_colour,
    Ops.CROP_TO_CENTER: _op_crop_to_center,
    Ops.CROP_TO_CORNER: _op_crop_to_corner,
    Ops.PAD_TOP: _op_pad_top,
    Ops.PAD_BOTTOM: _op_pad_bottom,
    Ops.PAD_LEFT: _op_pad_left,
    Ops.PAD_RIGHT: _op_pad_right,
    Ops.PAD_ALL: _op_pad_all,

    # Symmetry Operations
    Ops.MAKE_SYMMETRIC_H: _op_make_symmetric_h,
    Ops.MAKE_SYMMETRIC_V: _op_make_symmetric_v,
    Ops.MAKE_SYMMETRIC_D1: _op_make_symmetric_d1,
    Ops.MAKE_SYMMETRIC_D2: _op_make_symmetric_d2,
    Ops.MAKE_SYMMETRIC_ROT: _op_make_symmetric_rot,
    Ops.CHECK_SYMMETRY: _op_check_symmetry,
    Ops.SYMMETRIZE_UNION: _op_symmetrize_union,
    Ops.SYMMETRIZE_INTERSECT: _op_symmetrize_intersect,

    # Counting & Measurement
    Ops.COUNT_OBJECTS: _op_count_objects,
    Ops.MEASURE_BBOX: _op_measure_bbox,
    Ops.MARK_CENTROID: _op_mark_centroid,
    Ops.MARK_EXTREMA: _op_mark_extrema,
    Ops.ENCODE_SIZE: _op_encode_size,
    Ops.HISTOGRAM: _op_histogram,
    Ops.LABEL_COMPONENTS: _op_label_components,
    Ops.RANK_BY_SIZE: _op_rank_by_size,

    # Noise & Variation
    Ops.ADD_NOISE: _op_add_noise,
    Ops.REMOVE_SINGLETONS: _op_remove_singletons,
    Ops.SMOOTH: _op_smooth,
    Ops.PERTURB: _op_perturb,

    # Composite Operations
    Ops.EXTRACT_AND_CENTER: _op_extract_and_center,
    Ops.COLORIZE_REGIONS: _op_colorize_regions,
    Ops.CONNECT_NEAREST: _op_connect_nearest,
    Ops.FILL_BETWEEN: _op_fill_between,
    Ops.PROPAGATE_COLOUR: _op_propagate_colour,
    Ops.GRADIENT_FILL: _op_gradient_fill,
    Ops.CONTOUR: _op_contour,
    Ops.BLIT: _op_blit,
    Ops.MASK: _op_mask,
    Ops.COMPOSITE: _op_composite,
    Ops.LAYERS_MERGE: _op_layers_merge,
    Ops.CHANNEL_EXTRACT: _op_channel_extract,
    Ops.MULTIPLY_BRIGHTNESS: _op_multiply_brightness,
}


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_program(*operations: Tuple[Ops, dict]) -> Program:
    """Create a program from a list of (op, params) tuples."""
    return Program([Operation(op, params) for op, params in operations])


def chain(*ops: Ops) -> Program:
    """Create a simple program chain without parameters."""
    return Program([Operation(op) for op in ops])


# Export public API
__all__ = ["Ops", "Operation", "Program", "OP_IMPL", "make_program", "chain"]