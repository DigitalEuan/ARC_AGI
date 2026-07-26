"""
object_crg.py — Object-level Concept Relation Graph
=====================================================

The GLM's CRG learns relationships between concepts by co-occurrence.
This module adapts that mechanism to ARC: it learns relationships between
OBJECTS (connected components) by observing how they transform across
train pairs.

Each learned transformation is stored as an edge:
  (input_object_vector, transform_type, output_object_vector)

When we see a test object, we find the nearest input_object_vector in the
CRG (by colour-space distance, the GLM's native metric) and apply the
learned transformation.

This is GENERATIVE, not enumerative: the CRG tells us WHAT transformation
to apply, and the Φ-grammar generates HOW to apply it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
import sys, os

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
# OBJECT TRANSFORMATION EDGE — a learned object-to-object transformation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectTransformEdge:
    """A learned transformation between objects.

    Stored in the CRG as: (input_vector, transform_type, output_vector)

    The edge captures:
      - input_colour: the source object's colour
      - output_colour: the target object's colour (or 0 if disappeared)
      - transform_type: what kind of transformation (recolour, move, etc.)
      - input_vector_hex: the source object's 24-bit colour signature
      - output_vector_hex: the target object's 24-bit colour signature
      - weight: how many train pairs exhibited this transformation
      - colour_mapping: if recolour, the specific {old: new} mapping
    """
    input_colour: int
    output_colour: int
    transform_type: str          # "recolour", "move", "resize", "appear", "disappear", "unchanged", "composite"
    input_vector_hex: str
    output_vector_hex: str
    weight: int = 1
    colour_mapping: Dict[int, int] = field(default_factory=dict)
    # Spatial delta (for move transforms): (dr, dc)
    position_delta: Tuple[float, float] = (0.0, 0.0)
    # Size ratio (for resize transforms)
    size_ratio: float = 1.0

    def __repr__(self):
        return (f"ObjectTransformEdge({self.transform_type}: "
                f"colour {self.input_colour}→{self.output_colour}, "
                f"weight={self.weight})")


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT CRG — the learned graph of object transformations
# ══════════════════════════════════════════════════════════════════════════════

class ObjectCRG:
    """Concept Relation Graph for object-level transformations.

    Nodes are 24-bit colour signatures of objects.
    Edges are learned transformations (ObjectTransformEdge).

    The CRG is built by observing train pairs: for each (input_obj, output_obj)
    pair, we add (or reinforce) an edge.

    When predicting, we look up the test object's vector in the CRG and
    find the nearest learned transformation.
    """

    def __init__(self):
        # Edges keyed by input_colour → list of edges
        self.edges_by_colour: Dict[int, List[ObjectTransformEdge]] = defaultdict(list)
        # All edges (flat list)
        self.all_edges: List[ObjectTransformEdge] = []
        # Node colours (set of all input/output colour signatures seen)
        self.nodes: Set[str] = set()
        # Global colour mapping learned across all pairs
        self.global_colour_mapping: Dict[int, int] = {}
        # Transform type frequency
        self.transform_type_counts: Dict[str, int] = defaultdict(int)

    def learn_from_pair(self, pair: ObjectPair) -> None:
        """Learn a single object transformation from a train pair."""
        in_obj = pair.input_obj
        out_obj = pair.output_obj

        in_colour = in_obj.colour
        out_colour = out_obj.colour if out_obj else 0
        in_hex = in_obj.colour_hex
        out_hex = out_obj.colour_hex if out_obj else "#000000"

        # Check if we already have this transformation
        for edge in self.edges_by_colour[in_colour]:
            if (edge.output_colour == out_colour
                    and edge.transform_type == pair.transform_type):
                # Reinforce
                edge.weight += 1
                # Update colour mapping
                if pair.colour_changed and out_obj:
                    edge.colour_mapping[in_colour] = out_colour
                # Update position delta (running average)
                if pair.position_changed and out_obj:
                    dr = out_obj.centroid[0] - in_obj.centroid[0]
                    dc = out_obj.centroid[1] - in_obj.centroid[1]
                    n = edge.weight
                    edge.position_delta = (
                        (edge.position_delta[0] * (n - 1) + dr) / n,
                        (edge.position_delta[1] * (n - 1) + dc) / n,
                    )
                # Update size ratio
                if pair.size_changed and out_obj:
                    ratio = out_obj.cell_count / max(in_obj.cell_count, 1)
                    n = edge.weight
                    edge.size_ratio = (edge.size_ratio * (n - 1) + ratio) / n
                self.transform_type_counts[pair.transform_type] += 1
                return

        # New edge
        edge = ObjectTransformEdge(
            input_colour=in_colour,
            output_colour=out_colour,
            transform_type=pair.transform_type,
            input_vector_hex=in_hex,
            output_vector_hex=out_hex,
            weight=1,
            colour_mapping={in_colour: out_colour} if pair.colour_changed and out_obj else {},
            position_delta=(
                (out_obj.centroid[0] - in_obj.centroid[0],
                 out_obj.centroid[1] - in_obj.centroid[1])
                if pair.position_changed and out_obj else (0.0, 0.0)
            ),
            size_ratio=(
                out_obj.cell_count / max(in_obj.cell_count, 1)
                if pair.size_changed and out_obj else 1.0
            ),
        )
        self.edges_by_colour[in_colour].append(edge)
        self.all_edges.append(edge)
        self.nodes.add(in_hex)
        self.nodes.add(out_hex)
        self.transform_type_counts[pair.transform_type] += 1

        # Update global colour mapping (majority vote)
        if pair.colour_changed and out_obj:
            # Count votes for this mapping
            key = (in_colour, out_colour)
            # We need to track votes — use a dict
            if not hasattr(self, '_colour_votes'):
                self._colour_votes: Dict[Tuple[int, int], int] = defaultdict(int)
            self._colour_votes[key] += 1
            # Update global mapping to the majority vote
            best_mapping = {}
            for (old, new), count in self._colour_votes.items():
                if old not in best_mapping or count > self._colour_votes.get((old, best_mapping[old]), 0):
                    best_mapping[old] = new
            self.global_colour_mapping = best_mapping

    def learn_from_task(self, task: ARCTask) -> None:
        """Learn all object transformations from a task's train pairs."""
        for pair in task.train:
            in_objects = extract_objects(pair.input)
            out_objects = extract_objects(pair.output)
            obj_pairs = pair_objects(in_objects, out_objects)
            for op in obj_pairs:
                self.learn_from_pair(op)

    def find_transform_for_object(self, obj: GridObject) -> Optional[ObjectTransformEdge]:
        """Find the best learned transformation for a test object.

        Looks up the object's colour in the CRG. If there are edges for
        that colour, returns the highest-weight one. If no direct colour
        match, falls back to colour-space distance (GLM's native metric).
        """
        # Direct colour lookup
        if obj.colour in self.edges_by_colour:
            edges = self.edges_by_colour[obj.colour]
            # Return the highest-weight edge
            return max(edges, key=lambda e: e.weight)

        # Fallback: find nearest by colour-space distance
        if not _GLM_COLOUR_AVAILABLE or not self.all_edges:
            return None

        best_edge = None
        best_dist = float('inf')
        for edge in self.all_edges:
            d = colour_distance(obj.colour_hex, edge.input_vector_hex)
            if d < best_dist:
                best_dist = d
                best_edge = edge
        return best_edge

    def dominant_transform_type(self) -> str:
        """The most common transformation type across all learned edges."""
        if not self.transform_type_counts:
            return "unknown"
        return max(self.transform_type_counts, key=self.transform_type_counts.get)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for the CRG."""
        return {
            "total_edges": len(self.all_edges),
            "unique_colours": len(self.edges_by_colour),
            "transform_types": dict(self.transform_type_counts),
            "dominant_type": self.dominant_transform_type(),
            "global_colour_mapping": self.global_colour_mapping,
            "nodes": len(self.nodes),
        }

    def summary(self) -> str:
        s = self.stats()
        lines = [
            f"ObjectCRG summary:",
            f"  total edges:       {s['total_edges']}",
            f"  unique colours:    {s['unique_colours']}",
            f"  nodes:             {s['nodes']}",
            f"  dominant type:     {s['dominant_type']}",
            f"  transform types:   {s['transform_types']}",
            f"  colour mapping:    {s['global_colour_mapping']}",
        ]
        return "\n".join(lines)
