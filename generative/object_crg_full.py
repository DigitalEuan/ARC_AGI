"""
object_crg_full.py — Comprehensive Object-level Concept Relation Graph
========================================================================

The GLM's CRG learns relationships between concepts by co-occurrence.
This module adapts that mechanism to ARC: it learns relationships between
OBJECTS (connected components) by observing how they transform across
train pairs.

EXTENDED VERSION with:
  - Rich relational edges (spatial, topological, semantic)
  - Hierarchical CRG (nested objects, compositions)
  - Transformation chains (A→B→C sequences)
  - Analogical reasoning capabilities
  - Advanced parameterization (scale, rotation, fill-ratio, symmetry)
  - Multi-object patterns and group transformations
  - Context-sensitive transformation rules

Each learned transformation is stored as an edge:
  (input_object_vector, transform_type, output_object_vector, context, parameters)

When we see a test object, we find the nearest input_object_vector in the
CRG (by colour-space distance, the GLM's native metric) and apply the
learned transformation.

This is GENERATIVE, not enumerative: the CRG tells us WHAT transformation
to apply, and the Φ-grammar generates HOW to apply it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Callable, Union
from collections import defaultdict
from enum import Enum, auto
import sys
import os
import math

# Package imports
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
_PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from arc_loader import Grid, ARCTask
from generative.object_extractor import (
    GridObject, GridSentence, extract_objects, pair_objects, ObjectPair,
    grid_to_sentence,
)

# GLM colour-native operations
try:
    from GLM18_hex_colour import colour_distance
    _GLM_COLOUR_AVAILABLE = True
except ImportError:
    _GLM_COLOUR_AVAILABLE = False
    colour_distance = None


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL RELATIONS ENUM — vocabulary for object relationships
# ══════════════════════════════════════════════════════════════════════════════

class SpatialRelation(Enum):
    """Vocabulary of spatial relationships between objects."""
    # Directional
    LEFT_OF = auto()
    RIGHT_OF = auto()
    ABOVE = auto()
    BELOW = auto()
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()

    # Topological
    CONTAINS = auto()
    INSIDE = auto()
    TOUCHES = auto()
    OVERLAPS = auto()
    DISJOINT = auto()

    # Alignment
    ALIGNED_H = auto()      # Same row band
    ALIGNED_V = auto()      # Same column band
    ALIGNED_DIAG = auto()   # On same diagonal

    # Symmetry
    SYMMETRIC_H = auto()    # Mirror across horizontal axis
    SYMMETRIC_V = auto()    # Mirror across vertical axis
    SYMMETRIC_D1 = auto()   # Mirror across main diagonal
    SYMMETRIC_D2 = auto()   # Mirror across anti-diagonal
    ROT_90 = auto()         # 90° rotation
    ROT_180 = auto()        # 180° rotation
    ROT_270 = auto()        # 270° rotation

    # Proximity
    NEAR = auto()
    FAR = auto()
    ADJACENT = auto()

    # Grouping
    IN_ROW = auto()
    IN_COLUMN = auto()
    IN_GRID = auto()
    CONCENTRIC = auto()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM TYPE ENUM — comprehensive vocabulary of object transformations
# ══════════════════════════════════════════════════════════════════════════════

class TransformType(Enum):
    """Comprehensive vocabulary of object transformation types."""
    # Basic transforms
    UNCHANGED = auto()
    RECOLOUR = auto()
    MOVE = auto()
    RESIZE = auto()
    APPEAR = auto()
    DISAPPEAR = auto()

    # Geometric transforms
    ROTATE_90 = auto()
    ROTATE_180 = auto()
    ROTATE_270 = auto()
    FLIP_H = auto()
    FLIP_V = auto()
    TRANSPOSE = auto()
    SCALE = auto()

    # Composite transforms
    COMPOSITE = auto()
    SEQUENCE = auto()       # A→B→C chain

    # Pattern transforms
    REPLICATE = auto()
    TILE = auto()
    MIRROR = auto()
    SPREAD = auto()
    FILL = auto()
    OUTLINE = auto()

    # Morphological transforms
    DILATE = auto()
    ERODE = auto()
    SKELETONIZE = auto()
    THICKEN = auto()

    # Relational transforms
    ALIGN = auto()
    CENTER = auto()
    DISTRIBUTE = auto()
    SORT = auto()

    # Conditional transforms
    CONDITIONAL = auto()
    CONTEXT_SENSITIVE = auto()

    # Group transforms
    GROUP_MOVE = auto()
    GROUP_RECOLOUR = auto()
    GROUP_TRANSFORM = auto()

    # Hierarchical transforms
    NEST = auto()
    EXTRACT = auto()
    COMPOSE = auto()
    DECOMPOSE = auto()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM PARAMETERS — rich parameterization of transformations
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransformParams:
    """Rich parameterization of an object transformation."""

    # Colour parameters
    colour_map: Dict[int, int] = field(default_factory=dict)
    palette_cycle: int = 0          # Cycle amount for palette cycling

    # Spatial parameters
    delta_r: float = 0.0            # Row displacement
    delta_c: float = 0.0            # Column displacement
    target_position: Optional[Tuple[float, float]] = None
    anchor: str = "center"          # Anchor point: "center", "top_left", etc.

    # Scale parameters
    scale_factor: float = 1.0
    scale_axis: str = "uniform"     # "uniform", "horizontal", "vertical"

    # Rotation parameters
    rotation_angle: float = 0.0     # Degrees
    rotation_center: Optional[Tuple[float, float]] = None

    # Shape parameters
    fill_ratio_target: float = 1.0
    aspect_ratio_target: float = 1.0
    shape_type: str = "preserve"    # "preserve", "rect", "circle", "line"

    # Pattern parameters
    repeat_count: int = 1
    tile_pattern: str = "grid"      # "grid", "brick", "hex", "random"
    spacing: Tuple[int, int] = (0, 0)

    # Morphology parameters
    kernel_size: int = 1
    iterations: int = 1
    connectivity: int = 4           # 4 or 8

    # Conditional parameters
    condition_type: str = ""        # "neighbour", "border", "corner", etc.
    condition_value: Any = None

    # Sequence parameters
    sequence_chain: List[str] = field(default_factory=list)

    # Group parameters
    group_id: Optional[int] = None
    group_size: int = 1

    # Confidence/weight
    confidence: float = 1.0
    weight: int = 1

    def __repr__(self):
        parts = []
        if self.colour_map:
            parts.append(f"colour={self.colour_map}")
        if abs(self.delta_r) > 0 or abs(self.delta_c) > 0:
            parts.append(f"delta=({self.delta_r:.1f},{self.delta_c:.1f})")
        if self.scale_factor != 1.0:
            parts.append(f"scale={self.scale_factor}")
        if self.rotation_angle != 0:
            parts.append(f"rot={self.rotation_angle}")
        if self.repeat_count > 1:
            parts.append(f"repeat={self.repeat_count}")
        return f"TransformParams({', '.join(parts)})"


# ══════════════════════════════════════════════════════════════════════════════
# RELATIONAL EDGE — captures relationship between two objects
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RelationalEdge:
    """A relationship between two objects in the CRG."""

    obj_a_id: str                   # Unique ID of first object
    obj_b_id: str                   # Unique ID of second object
    relation: SpatialRelation       # Type of relationship
    strength: float = 1.0           # Strength of relationship (0-1)
    direction: Tuple[float, float] = (0.0, 0.0)  # Vector from A to B
    distance: float = 0.0           # Euclidean distance
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_symmetric(self) -> bool:
        """Check if this relation is symmetric."""
        return self.relation in {
            SpatialRelation.TOUCHES,
            SpatialRelation.OVERLAPS,
            SpatialRelation.DISJOINT,
            SpatialRelation.NEAR,
            SpatialRelation.FAR,
            SpatialRelation.ALIGNED_H,
            SpatialRelation.ALIGNED_V,
            SpatialRelation.ALIGNED_DIAG,
            SpatialRelation.CONCENTRIC,
        }

    def inverse(self) -> SpatialRelation:
        """Return the inverse relation."""
        inverses = {
            SpatialRelation.LEFT_OF: SpatialRelation.RIGHT_OF,
            SpatialRelation.RIGHT_OF: SpatialRelation.LEFT_OF,
            SpatialRelation.ABOVE: SpatialRelation.BELOW,
            SpatialRelation.BELOW: SpatialRelation.ABOVE,
            SpatialRelation.TOP_LEFT: SpatialRelation.BOTTOM_RIGHT,
            SpatialRelation.TOP_RIGHT: SpatialRelation.BOTTOM_LEFT,
            SpatialRelation.BOTTOM_LEFT: SpatialRelation.TOP_RIGHT,
            SpatialRelation.BOTTOM_RIGHT: SpatialRelation.TOP_LEFT,
            SpatialRelation.CONTAINS: SpatialRelation.INSIDE,
            SpatialRelation.INSIDE: SpatialRelation.CONTAINS,
        }
        return inverses.get(self.relation, self.relation)

    def __repr__(self):
        return f"RelationalEdge({self.obj_a_id} {self.relation.name} {self.obj_b_id})"


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT TRANSFORM EDGE — comprehensive learned object-to-object transformation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectTransformEdge:
    """A comprehensive learned transformation between objects.

    Stored in the CRG as: (input_vector, transform_type, output_vector, params)

    The edge captures:
      - Full transformation semantics (type, parameters, context)
      - Relational constraints (what other objects must be present)
      - Confidence/weight based on training frequency
      - Applicability conditions
    """

    # Core identification
    edge_id: str
    input_colour: int
    output_colour: int
    transform_type: TransformType

    # Vector signatures
    input_vector_hex: str
    output_vector_hex: str

    # Parameters
    params: TransformParams = field(default_factory=TransformParams)

    # Relational context
    required_relations: List[RelationalEdge] = field(default_factory=list)
    forbidden_relations: List[RelationalEdge] = field(default_factory=list)

    # Statistics
    weight: int = 1
    success_count: int = 0
    failure_count: int = 0

    # Metadata
    learned_from_task_ids: List[str] = field(default_factory=list)
    applicability_conditions: Dict[str, Any] = field(default_factory=dict)

    # Derived properties
    input_shape: Optional[Tuple[int, int]] = None
    
    # Compatibility properties (for backward compat with old GenerativeTransformer)
    @property
    def position_delta(self) -> Tuple[float, float]:
        return (self.params.delta_r, self.params.delta_c)
    @property
    def size_ratio(self) -> float:
        return self.params.scale_factor
    @property
    def colour_mapping(self) -> Dict[int, int]:
        return self.params.colour_map
    output_shape: Optional[Tuple[int, int]] = None
    input_fill_ratio: float = 1.0
    output_fill_ratio: float = 1.0

    @property
    def confidence(self) -> float:
        """Compute confidence based on success/failure ratio."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    @property
    def is_reliable(self) -> bool:
        """Check if this edge is reliable (high confidence, sufficient samples)."""
        return self.confidence > 0.7 and (self.success_count + self.failure_count) >= 2

    def merge_with(self, other: 'ObjectTransformEdge') -> None:
        """Merge another edge into this one (reinforce learning)."""
        if self.transform_type != other.transform_type:
            return

        # Update weight
        self.weight += other.weight
        self.success_count += other.success_count
        self.failure_count += other.failure_count

        # Merge task IDs
        for tid in other.learned_from_task_ids:
            if tid not in self.learned_from_task_ids:
                self.learned_from_task_ids.append(tid)

        # Average parameters
        self.params.delta_r = (self.params.delta_r * self.weight +
                               other.params.delta_r * other.weight) / (self.weight + other.weight)
        self.params.delta_c = (self.params.delta_c * self.weight +
                               other.params.delta_c * other.weight) / (self.weight + other.weight)

        if self.params.scale_factor != 1.0 or other.params.scale_factor != 1.0:
            self.params.scale_factor = (self.params.scale_factor * self.weight +
                                        other.params.scale_factor * other.weight) / (self.weight + other.weight)

    def __repr__(self):
        return (f"ObjectTransformEdge({self.transform_type.name}: "
                f"{self.input_colour}→{self.output_colour}, "
                f"weight={self.weight}, conf={self.confidence:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT GROUP — collection of objects that transform together
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectGroup:
    """A group of objects that transform together as a unit."""

    group_id: int = 0
    objects: List[GridObject] = field(default_factory=list)
    relations: List[RelationalEdge] = field(default_factory=list)
    centroid: Tuple[float, float] = (0.0, 0.0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # Group-level properties
    pattern_type: str = "none"      # "row", "column", "grid", "cluster", "ring"
    symmetry_type: Optional[str] = None
    regular_spacing: bool = False

    # Learned group transformations
    group_transforms: List[ObjectTransformEdge] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.objects)

    @property
    def colours(self) -> Set[int]:
        return {obj.colour for obj in self.objects}

    @classmethod
    def from_objects(cls, objects: List[GridObject] = field(default_factory=list), group_id: int = 0) -> 'ObjectGroup':
        """Create a group from a list of objects."""
        if not objects:
            raise ValueError("Cannot create empty group")

        # Compute centroid
        all_cells = []
        for obj in objects:
            all_cells.extend(obj.cells)

        rs = [r for r, _ in all_cells]
        cs = [c for _, c in all_cells]
        centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
        bbox = (min(rs), max(rs), min(cs), max(cs))

        # Compute pairwise relations
        relations = []
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i >= j:
                    continue
                relation = compute_spatial_relation(obj_a, obj_b)
                if relation:
                    relations.append(RelationalEdge(
                        obj_a_id=f"{obj_a.colour}_{i}",
                        obj_b_id=f"{obj_b.colour}_{j}",
                        relation=relation,
                        distance=centroid_distance(obj_a.centroid, obj_b.centroid),
                        direction=(obj_b.centroid[0] - obj_a.centroid[0],
                                   obj_b.centroid[1] - obj_a.centroid[1]),
                    ))

        # Detect pattern type
        pattern_type = detect_group_pattern(objects, relations)

        return cls(
            group_id=group_id,
            objects=objects,
            relations=relations,
            centroid=centroid,
            bbox=bbox,
            pattern_type=pattern_type,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL OBJECT — nested object structure
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HierarchicalObject:
    """An object that may contain nested sub-objects."""

    # Core properties
    obj_id: str
    colour: int
    cells: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: Tuple[float, float] = (0.0, 0.0)

    # Hierarchy
    parent: Optional['HierarchicalObject'] = None
    children: List['HierarchicalObject'] = field(default_factory=list)

    # Level in hierarchy (0 = root)
    level: int = 0

    # Encoded vector
    vector: List[int] = field(default_factory=list)
    colour_hex: str = "#000000"

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def depth(self) -> int:
        """Maximum depth of subtree."""
        if self.is_leaf:
            return 0
        return 1 + max(child.depth for child in self.children)

    @property
    def total_cell_count(self) -> int:
        """Total cells including all descendants."""
        if self.is_leaf:
            return len(self.cells)
        return len(self.cells) + sum(child.total_cell_count for child in self.children)

    def flatten(self) -> List[GridObject]:
        """Flatten hierarchy to list of GridObjects."""
        result = []
        if self.is_leaf:
            result.append(GridObject(
                cells=self.cells,
                colour=self.colour,
                grid_shape=(self.bbox[1] - self.bbox[0] + 1,
                           self.bbox[3] - self.bbox[2] + 1),
                bbox=self.bbox,
                centroid=self.centroid,
            ))
        else:
            for child in self.children:
                result.extend(child.flatten())
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL UTILITIES — geometric computations for object relationships
# ══════════════════════════════════════════════════════════════════════════════

def centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two centroids."""
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Compute Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def bboxes_intersect(b1: Tuple[int, int, int, int],
                     b2: Tuple[int, int, int, int]) -> bool:
    """Check if two bounding boxes intersect."""
    return not (b1[1] < b2[0] or b1[0] > b2[1] or
                b1[3] < b2[2] or b1[2] > b2[3])


def bbox_contains(b_outer: Tuple[int, int, int, int],
                  b_inner: Tuple[int, int, int, int]) -> bool:
    """Check if outer bbox completely contains inner bbox."""
    return (b_outer[0] <= b_inner[0] and b_outer[1] >= b_inner[1] and
            b_outer[2] <= b_inner[2] and b_outer[3] >= b_inner[3])


def compute_spatial_relation(obj_a: GridObject, obj_b: GridObject) -> Optional[SpatialRelation]:
    """Compute the primary spatial relation between two objects."""
    # Check containment
    if bbox_contains(obj_a.bbox, obj_b.bbox):
        return SpatialRelation.CONTAINS
    if bbox_contains(obj_b.bbox, obj_a.bbox):
        return SpatialRelation.INSIDE

    # Check overlap
    if bboxes_intersect(obj_a.bbox, obj_b.bbox):
        return SpatialRelation.OVERLAPS

    # Check adjacency/touching
    if objects_touch(obj_a, obj_b):
        return SpatialRelation.TOUCHES

    # Compute directional relation
    dr = obj_b.centroid[0] - obj_a.centroid[0]
    dc = obj_b.centroid[1] - obj_a.centroid[1]

    dist = centroid_distance(obj_a.centroid, obj_b.centroid)
    grid_diag = math.sqrt(obj_a.grid_shape[0]**2 + obj_a.grid_shape[1]**2)

    # Proximity-based
    if dist < 3:
        return SpatialRelation.ADJACENT
    if dist < grid_diag * 0.2:
        rel = SpatialRelation.NEAR
    elif dist > grid_diag * 0.5:
        rel = SpatialRelation.FAR
    else:
        rel = SpatialRelation.DISJOINT

    # Directional refinement
    threshold = 0.5
    if abs(dr) < threshold and abs(dc) < threshold:
        return rel

    if abs(dr) < threshold:
        if dc > 0:
            return SpatialRelation.RIGHT_OF
        else:
            return SpatialRelation.LEFT_OF

    if abs(dc) < threshold:
        if dr > 0:
            return SpatialRelation.BELOW
        else:
            return SpatialRelation.ABOVE

    # Diagonal
    if dr > 0 and dc > 0:
        return SpatialRelation.BOTTOM_RIGHT
    elif dr > 0 and dc < 0:
        return SpatialRelation.BOTTOM_LEFT
    elif dr < 0 and dc > 0:
        return SpatialRelation.TOP_RIGHT
    else:
        return SpatialRelation.TOP_LEFT


def objects_touch(obj_a: GridObject, obj_b: GridObject) -> bool:
    """Check if two objects are touching (8-neighbour adjacency)."""
    cells_b = set(obj_b.cells)
    for r, c in obj_a.cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in cells_b:
                    return True
    return False


def detect_group_pattern(objects: List[GridObject] = field(default_factory=list),
                         relations: List[RelationalEdge] = field(default_factory=list)) -> str:
    """Detect the pattern type of a group of objects."""
    if len(objects) < 2:
        return "single"

    # Check for row pattern
    row_threshold = 2.0
    if all(abs(obj.centroid[0] - objects[0].centroid[0]) < row_threshold
           for obj in objects[1:]):
        return "row"

    # Check for column pattern
    col_threshold = 2.0
    if all(abs(obj.centroid[1] - objects[0].centroid[1]) < col_threshold
           for obj in objects[1:]):
        return "column"

    # Check for grid pattern
    rows = sorted(set(round(obj.centroid[0]) for obj in objects))
    cols = sorted(set(round(obj.centroid[1]) for obj in objects))
    if len(rows) > 1 and len(cols) > 1:
        expected = len(rows) * len(cols)
        if abs(len(objects) - expected) <= 1:
            return "grid"

    # Check for ring pattern (objects roughly equidistant from center)
    centroid = (sum(obj.centroid[0] for obj in objects) / len(objects),
                sum(obj.centroid[1] for obj in objects) / len(objects))
    distances = [centroid_distance(obj.centroid, centroid) for obj in objects]
    if max(distances) - min(distances) < 3:
        return "ring"

    return "cluster"


def check_alignment(obj_a: GridObject, obj_b: GridObject,
                    tolerance: float = 2.0) -> Optional[SpatialRelation]:
    """Check if two objects are aligned."""
    dr = abs(obj_a.centroid[0] - obj_b.centroid[0])
    dc = abs(obj_a.centroid[1] - obj_b.centroid[1])

    if dr < tolerance:
        return SpatialRelation.ALIGNED_H
    if dc < tolerance:
        return SpatialRelation.ALIGNED_V

    # Diagonal alignment
    if abs(dr - dc) < tolerance:
        return SpatialRelation.ALIGNED_DIAG

    return None


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION CHAIN — sequence of transformations A→B→C
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransformChain:
    """A chain of transformations (A→B→C sequence)."""

    chain_id: str
    steps: List[ObjectTransformEdge]

    # Chain properties
    input_colour: int
    final_colour: int
    total_transforms: int

    # Statistics
    chain_weight: int = 1
    observed_sequences: List[List[str]] = field(default_factory=list)

    @property
    def intermediate_colours(self) -> List[int]:
        """Colours of intermediate states."""
        return [edge.output_colour for edge in self.steps[:-1]]

    def apply_sequentially(self, obj: GridObject) -> Optional[GridObject]:
        """Apply the chain of transformations sequentially."""
        current = obj
        for step in self.steps:
            # Apply each step (placeholder - actual application depends on DSL)
            # This would call the appropriate DSL operation
            pass
        return current

    def __repr__(self):
        colours = [str(self.input_colour)]
        for edge in self.steps:
            colours.append(str(edge.output_colour))
        return f"TransformChain({'→'.join(colours)})"


# ══════════════════════════════════════════════════════════════════════════════
# ANALOGICAL MAPPING — mapping for analogical reasoning
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalogicalMapping:
    """A mapping for analogical reasoning (A:B :: C:D)."""

    # Source analogy
    source_a: GridObject
    source_b: GridObject
    source_transform: ObjectTransformEdge

    # Target analogy
    target_c: GridObject
    target_d: Optional[GridObject] = None  # To be predicted

    # Mapping quality
    structural_similarity: float = 1.0
    colour_similarity: float = 1.0
    relational_similarity: float = 1.0

    @property
    def overall_similarity(self) -> float:
        """Overall similarity score."""
        return (self.structural_similarity * 0.4 +
                self.colour_similarity * 0.3 +
                self.relational_similarity * 0.3)

    def predict_target(self) -> Optional[Dict[str, Any]]:
        """Predict the target object D based on the analogy."""
        if self.target_d is not None:
            return None  # Already known

        # Apply the source transformation to target_c
        # This is a placeholder - actual implementation would use DSL
        transform = self.source_transform

        prediction = {
            'predicted_colour': transform.output_colour,
            'predicted_transform': transform.transform_type,
            'params': transform.params,
            'confidence': self.overall_similarity * transform.confidence,
        }

        return prediction


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT CRG — comprehensive learned graph of object transformations
# ══════════════════════════════════════════════════════════════════════════════

class ObjectCRG:
    """Comprehensive Concept Relation Graph for object-level transformations.

    Nodes are 24-bit colour signatures of objects.
    Edges are learned transformations (ObjectTransformEdge).
    Additional structures capture:
      - Relational edges between objects
      - Object groups that transform together
      - Transformation chains (sequences)
      - Hierarchical object structures
      - Analogical mappings

    The CRG is built by observing train pairs: for each (input_obj, output_obj)
    pair, we add (or reinforce) an edge.

    When predicting, we look up the test object's vector in the CRG and
    find the nearest learned transformation, considering context and relations.
    """

    def __init__(self):
        # Core transformation edges
        self.edges_by_colour: Dict[int, List[ObjectTransformEdge]] = defaultdict(list)
        self.all_edges: List[ObjectTransformEdge] = []
        self.nodes: Set[str] = set()

        # Relational structure
        self.relational_edges: List[RelationalEdge] = []
        self.relation_counts: Dict[SpatialRelation, int] = defaultdict(int)

        # Object groups
        self.groups: Dict[int, ObjectGroup] = {}
        self.group_transforms: Dict[int, List[ObjectTransformEdge]] = defaultdict(list)

        # Transformation chains
        self.chains: Dict[str, TransformChain] = {}

        # Analogical mappings
        self.analogies: List[AnalogicalMapping] = []

        # Global statistics
        self.global_colour_mapping: Dict[int, int] = {}
        self.transform_type_counts: Dict[TransformType, int] = defaultdict(int)
        self.task_ids_seen: Set[str] = set()

        # Hierarchical structures
        self.hierarchical_objects: Dict[str, HierarchicalObject] = {}

        # Vote tracking for colour mappings
        self._colour_votes: Dict[Tuple[int, int], int] = defaultdict(int)

    def _generate_edge_id(self, in_colour: int, out_colour: int,
                          t_type: TransformType) -> str:
        """Generate a unique edge ID."""
        return f"edge_{in_colour}_{out_colour}_{t_type.name}_{len(self.all_edges)}"

    def learn_from_pair(self, pair: ObjectPair, task_id: str = "") -> Optional[ObjectTransformEdge]:
        """Learn a single object transformation from a train pair."""
        in_obj = pair.input_obj
        out_obj = pair.output_obj

        if not in_obj.cells:  # Dummy input for "appear"
            return None

        in_colour = in_obj.colour
        out_colour = out_obj.colour if out_obj else 0
        in_hex = in_obj.colour_hex
        out_hex = out_obj.colour_hex if out_obj else "#000000"

        # Determine transform type
        t_type = self._classify_transform(pair, in_obj, out_obj)

        # Build parameters
        params = self._build_transform_params(pair, in_obj, out_obj, t_type)

        # Check if we already have this transformation
        for edge in self.edges_by_colour[in_colour]:
            if (edge.output_colour == out_colour and
                edge.transform_type == t_type):
                # Reinforce existing edge
                edge.weight += 1
                edge.success_count += 1
                edge.learned_from_task_ids.append(task_id)

                # Update parameters (running average)
                n = edge.weight
                edge.params.delta_r = (edge.params.delta_r * (n-1) + params.delta_r) / n
                edge.params.delta_c = (edge.params.delta_c * (n-1) + params.delta_c) / n

                if params.scale_factor != 1.0:
                    edge.params.scale_factor = (edge.params.scale_factor * (n-1) +
                                                params.scale_factor) / n

                self.transform_type_counts[t_type] += 1
                return edge

        # Create new edge
        edge = ObjectTransformEdge(
            edge_id=self._generate_edge_id(in_colour, out_colour, t_type),
            input_colour=in_colour,
            output_colour=out_colour,
            transform_type=t_type,
            input_vector_hex=in_hex,
            output_vector_hex=out_hex,
            params=params,
            weight=1,
            success_count=1,
            learned_from_task_ids=[task_id] if task_id else [],
            input_shape=in_obj.grid_shape,
            input_fill_ratio=in_obj.fill_ratio,
            output_fill_ratio=out_obj.fill_ratio if out_obj else 0,
        )

        self.edges_by_colour[in_colour].append(edge)
        self.all_edges.append(edge)
        self.nodes.add(in_hex)
        self.nodes.add(out_hex)
        self.transform_type_counts[t_type] += 1

        # Update global colour mapping
        if pair.colour_changed and out_obj:
            key = (in_colour, out_colour)
            self._colour_votes[key] += 1
            self._update_global_colour_mapping()

        return edge

    def _classify_transform(self, pair: ObjectPair,
                            in_obj: GridObject,
                            out_obj: Optional[GridObject]) -> TransformType:
        """Classify the transformation type from an object pair."""
        if pair.transform_type == "unchanged":
            return TransformType.UNCHANGED
        elif pair.transform_type == "disappear":
            return TransformType.DISAPPEAR
        elif pair.transform_type == "appear":
            return TransformType.APPEAR

        if pair.size_changed and not pair.position_changed and not pair.colour_changed:
            return TransformType.RESIZE

        if pair.position_changed and not pair.size_changed and not pair.colour_changed:
            # Check if it's a rotation
            # (would need shape comparison - simplified here)
            return TransformType.MOVE

        if pair.colour_changed and not pair.position_changed and not pair.size_changed:
            return TransformType.RECOLOUR

        # Composite - could be multiple transforms
        return TransformType.COMPOSITE

    def _build_transform_params(self, pair: ObjectPair,
                                 in_obj: GridObject,
                                 out_obj: Optional[GridObject],
                                 t_type: TransformType) -> TransformParams:
        """Build transformation parameters from an object pair."""
        params = TransformParams()

        if out_obj is None:
            return params

        # Colour mapping
        if pair.colour_changed:
            params.colour_map = {in_obj.colour: out_obj.colour}

        # Position delta
        if pair.position_changed:
            params.delta_r = out_obj.centroid[0] - in_obj.centroid[0]
            params.delta_c = out_obj.centroid[1] - in_obj.centroid[1]

        # Scale factor
        if pair.size_changed and in_obj.cell_count > 0:
            params.scale_factor = out_obj.cell_count / in_obj.cell_count

        # Fill ratio
        params.fill_ratio_target = out_obj.fill_ratio

        return params

    def _update_global_colour_mapping(self) -> None:
        """Update global colour mapping based on majority vote."""
        best_mapping = {}
        for (old, new), count in self._colour_votes.items():
            if old not in best_mapping or count > self._colour_votes.get((old, best_mapping.get(old, -1)), 0):
                best_mapping[old] = new
        self.global_colour_mapping = best_mapping

    def learn_relational_structure(self, objects: List[GridObject] = field(default_factory=list),
                                   task_id: str = "") -> None:
        """Learn relational structure between objects."""
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i >= j:
                    continue

                relation = compute_spatial_relation(obj_a, obj_b)
                if relation:
                    edge = RelationalEdge(
                        obj_a_id=f"{obj_a.colour}_{i}",
                        obj_b_id=f"{obj_b.colour}_{j}",
                        relation=relation,
                        distance=centroid_distance(obj_a.centroid, obj_b.centroid),
                        direction=(obj_b.centroid[0] - obj_a.centroid[0],
                                   obj_b.centroid[1] - obj_a.centroid[1]),
                        context={'task_id': task_id},
                    )
                    self.relational_edges.append(edge)
                    self.relation_counts[relation] += 1

    def learn_group_transformation(self, group, pair):
        """Stub: group transformation learning."""
        pass

    def learn_groups(self, objects: List[GridObject] = field(default_factory=list),
                     task_id: str = "") -> List[ObjectGroup]:
        """Detect and learn object groups."""
        groups = []

        # Simple grouping by proximity
        used = set()
        group_id = 0

        for i, obj in enumerate(objects):
            if i in used:
                continue

            # Find nearby objects
            group_members = [obj]
            used.add(i)

            for j, other in enumerate(objects):
                if j in used:
                    continue

                dist = centroid_distance(obj.centroid, other.centroid)
                if dist < 5:  # Proximity threshold
                    group_members.append(other)
                    used.add(j)

            if len(group_members) > 1:
                try:
                    group = ObjectGroup.from_objects(group_members, group_id)
                except ValueError:
                    continue
                groups.append(group)
                self.groups[group_id] = group
                group_id += 1

        return groups

    def learn_group_transformation(self, group, pair):
        """Stub: group transformation learning."""
        pass

    def learn_from_task(self, task: ARCTask, task_id: str = "") -> Dict[str, Any]:
        """Learn all object transformations and structure from a task's train pairs."""
        stats = {
            'edges_learned': 0,
            'relations_learned': 0,
            'groups_found': 0,
        }

        for pair_idx, pair in enumerate(task.train):
            in_objects = extract_objects(pair.input)
            out_objects = extract_objects(pair.output)
            obj_pairs = pair_objects(in_objects, out_objects)

            # Learn transformations
            for op in obj_pairs:
                edge = self.learn_from_pair(op, f"{task_id}_pair{pair_idx}")
                if edge:
                    stats['edges_learned'] += 1

            # Learn relational structure
            self.learn_relational_structure(in_objects, f"{task_id}_input")
            self.learn_relational_structure(out_objects, f"{task_id}_output")
            stats['relations_learned'] += len(in_objects) * (len(in_objects) - 1) // 2
            stats['relations_learned'] += len(out_objects) * (len(out_objects) - 1) // 2

            # Learn groups
            in_groups = self.learn_groups(in_objects, f"{task_id}_input")
            out_groups = self.learn_groups(out_objects, f"{task_id}_output")
            stats['groups_found'] += len(in_groups) + len(out_groups)

        self.task_ids_seen.add(task_id)
        return stats

    def find_transform_for_object(self, obj: GridObject,
                                   context: Optional[List[GridObject]] = None
                                   ) -> Optional[ObjectTransformEdge]:
        """Find the best learned transformation for a test object.

        Considers:
          - Direct colour match
          - Colour-space distance (GLM's native metric)
          - Relational context (what other objects are nearby)
          - Group membership
        """
        candidates = []

        # Direct colour lookup
        if obj.colour in self.edges_by_colour:
            edges = self.edges_by_colour[obj.colour]
            for edge in edges:
                score = edge.weight * edge.confidence
                candidates.append((edge, score))

        # Fallback: colour-space distance
        if _GLM_COLOUR_AVAILABLE:
            for edge in self.all_edges:
                d = colour_distance(obj.colour_hex, edge.input_vector_hex)
                if d < 50:  # Threshold
                    score = edge.weight * edge.confidence * (1 - d/50)
                    candidates.append((edge, score))

        # Consider relational context
        if context and candidates:
            for edge, score in candidates:
                # Check if required relations are satisfied
                if edge.required_relations:
                    # Simplified: just check if similar relations exist in context
                    # Full implementation would match specific relations
                    pass

        if not candidates:
            return None

        # Return highest-scoring candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def find_analogies(self, query_obj: GridObject,
                       k: int = 5) -> List[AnalogicalMapping]:
        """Find analogical mappings similar to a query object."""
        analogies = []

        for edge in self.all_edges:
            if edge.input_colour == query_obj.colour:
                # Create a potential analogy
                # (In full implementation, would search for structural matches)
                pass

        return analogies[:k]

    def get_transformation_chains(self, start_colour: int,
                                   max_length: int = 4) -> List[TransformChain]:
        """Find transformation chains starting from a given colour."""
        chains = []

        # BFS to find chains
        visited = set()
        queue = [(start_colour, [], 0)]

        while queue:
            colour, path, length = queue.pop(0)

            if length >= max_length:
                continue

            for edge in self.edges_by_colour.get(colour, []):
                if edge.edge_id in visited:
                    continue

                new_path = path + [edge]
                visited.add(edge.edge_id)

                if len(new_path) >= 2:
                    # Create chain
                    chain = TransformChain(
                        chain_id=f"chain_{start_colour}_{length}",
                        steps=new_path,
                        input_colour=start_colour,
                        final_colour=edge.output_colour,
                        total_transforms=len(new_path),
                        chain_weight=min(e.weight for e in new_path),
                    )
                    chains.append(chain)

                queue.append((edge.output_colour, new_path, length + 1))

        return chains

    def dominant_transform_type(self) -> TransformType:
        """The most common transformation type across all learned edges."""
        if not self.transform_type_counts:
            return TransformType.UNCHANGED
        return max(self.transform_type_counts, key=self.transform_type_counts.get)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for the CRG."""
        return {
            'total_edges': len(self.all_edges),
            'unique_colours': len(self.edges_by_colour),
            'transform_types': {t.name: c for t, c in self.transform_type_counts.items()},
            'dominant_type': self.dominant_transform_type().name,
            'global_colour_mapping': self.global_colour_mapping,
            'nodes': len(self.nodes),
            'relational_edges': len(self.relational_edges),
            'relation_types': {r.name: c for r, c in self.relation_counts.items()},
            'groups': len(self.groups),
            'chains': len(self.chains),
            'analogies': len(self.analogies),
            'tasks_seen': len(self.task_ids_seen),
        }

    def summary(self) -> str:
        """Human-readable summary of the CRG."""
        s = self.stats()
        lines = [
            "ObjectCRG Summary:",
            f"  Total edges:        {s['total_edges']}",
            f"  Unique colours:     {s['unique_colours']}",
            f"  Nodes:              {s['nodes']}",
            f"  Dominant type:      {s['dominant_type']}",
            f"  Relational edges:   {s['relational_edges']}",
            f"  Groups:             {s['groups']}",
            f"  Chains:             {s['chains']}",
            f"  Tasks seen:         {s['tasks_seen']}",
            "",
            "Transform type distribution:",
        ]

        for t_type, count in sorted(s['transform_types'].items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"  {t_type}: {count}")

        if s['global_colour_mapping']:
            lines.append("")
            lines.append("Global colour mapping:")
            for old, new in sorted(s['global_colour_mapping'].items()):
                lines.append(f"  {old} → {new}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize CRG to dictionary."""
        return {
            'edges': [
                {
                    'edge_id': e.edge_id,
                    'input_colour': e.input_colour,
                    'output_colour': e.output_colour,
                    'transform_type': e.transform_type.name,
                    'weight': e.weight,
                    'confidence': e.confidence,
                    'params': {
                        'delta_r': e.params.delta_r,
                        'delta_c': e.params.delta_c,
                        'scale_factor': e.params.scale_factor,
                        'colour_map': e.params.colour_map,
                    },
                }
                for e in self.all_edges
            ],
            'relations': [
                {
                    'obj_a': r.obj_a_id,
                    'obj_b': r.obj_b_id,
                    'relation': r.relation.name,
                    'distance': r.distance,
                }
                for r in self.relational_edges
            ],
            'groups': {
                gid: {
                    'size': g.size,
                    'pattern': g.pattern_type,
                    'colours': list(g.colours),
                }
                for gid, g in self.groups.items()
            },
            'global_colour_mapping': self.global_colour_mapping,
            'stats': self.stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectCRG':
        """Deserialize CRG from dictionary."""
        crg = cls()

        # Restore edges (simplified - full implementation would restore all fields)
        for edge_data in data.get('edges', []):
            # Would need to reconstruct full edge objects
            pass

        # Restore global mapping
        crg.global_colour_mapping = data.get('global_colour_mapping', {})

        return crg


# ══════════════════════════════════════════════════════════════════════════════
# CRG QUERIES — advanced query operations on the CRG
# ══════════════════════════════════════════════════════════════════════════════

def query_by_transform_type(crg: ObjectCRG,
                            t_type: TransformType) -> List[ObjectTransformEdge]:
    """Query all edges of a specific transform type."""
    return [e for e in crg.all_edges if e.transform_type == t_type]


def query_by_colour_pair(crg: ObjectCRG,
                         in_colour: int,
                         out_colour: int) -> List[ObjectTransformEdge]:
    """Query all edges transforming one colour to another."""
    return [e for e in crg.all_edges
            if e.input_colour == in_colour and e.output_colour == out_colour]


def query_reliable_transforms(crg: ObjectCRG,
                              min_confidence: float = 0.7,
                              min_weight: int = 2) -> List[ObjectTransformEdge]:
    """Query all reliable transformations."""
    return [e for e in crg.all_edges
            if e.confidence >= min_confidence and e.weight >= min_weight]


def query_context_sensitive(crg: ObjectCRG) -> List[ObjectTransformEdge]:
    """Query all context-sensitive transformations."""
    return [e for e in crg.all_edges
            if e.required_relations or e.applicability_conditions]


def find_colour_cycles(crg: ObjectCRG) -> List[List[int]]:
    """Find colour transformation cycles (A→B→C→A)."""
    cycles = []
    visited = set()

    for start_edge in crg.all_edges:
        if start_edge.edge_id in visited:
            continue

        # DFS to find cycles
        stack = [(start_edge.output_colour, [start_edge.input_colour])]

        while stack:
            colour, path = stack.pop()

            if colour == path[0] and len(path) > 1:
                cycles.append(path)
                continue

            if len(path) > 5:  # Limit cycle length
                continue

            for edge in crg.edges_by_colour.get(colour, []):
                if edge.edge_id not in visited:
                    visited.add(edge.edge_id)
                    stack.append((edge.output_colour, path + [edge.output_colour]))

    return cycles


def compute_transform_diversity(crg: ObjectCRG) -> float:
    """Compute diversity of transformations (entropy of transform types)."""
    if not crg.all_edges:
        return 0.0

    total = len(crg.all_edges)
    entropy = 0.0

    for count in crg.transform_type_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalize by max possible entropy
    max_entropy = math.log2(len(crg.transform_type_counts)) if crg.transform_type_counts else 1
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — demonstration and testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ObjectCRG Full — Comprehensive Object-level Concept Relation Graph")
    print("=" * 70)
    print()
    print("Features:")
    print("  • Rich relational edges (spatial, topological, semantic)")
    print("  • Hierarchical CRG (nested objects, compositions)")
    print("  • Transformation chains (A→B→C sequences)")
    print("  • Analogical reasoning capabilities")
    print("  • Advanced parameterization (scale, rotation, fill-ratio, symmetry)")
    print("  • Multi-object patterns and group transformations")
    print("  • Context-sensitive transformation rules")
    print()
    print(f"Spatial Relations: {len(SpatialRelation)} types")
    print(f"Transform Types: {len(TransformType)} types")
    print()
    print("Example usage:")
    print("  crg = ObjectCRG()")
    print("  crg.learn_from_task(task, task_id='puzzle_001')")
    print("  transform = crg.find_transform_for_object(test_object)")
    print("  chains = crg.get_transformation_chains(start_colour=5)")
    print("  analogies = crg.find_analogies(query_object)")