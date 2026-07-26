"""
crg_persistence.py — save/load the ObjectCRG across tasks
============================================================

The GLM's "memory" — learned transformations persist across tasks so
the system accumulates knowledge. Each task's CRG edges are saved to
a JSON file and loaded before the next task, building a growing
library of learned transformations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import json
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from generative.object_crg import ObjectCRG, ObjectTransformEdge


CRG_STATE_PATH = os.path.join(_PKG_ROOT, "data", "crg_state.json")


def save_crg(crg: ObjectCRG, path: str = CRG_STATE_PATH) -> None:
    """Save a CRG to a JSON file."""
    data = {
        "edges": [],
        "global_colour_mapping": crg.global_colour_mapping,
        "transform_type_counts": dict(crg.transform_type_counts),
        "nodes": list(crg.nodes),
    }
    for edge in crg.all_edges:
        data["edges"].append({
            "input_colour": edge.input_colour,
            "output_colour": edge.output_colour,
            "transform_type": edge.transform_type,
            "input_vector_hex": edge.input_vector_hex,
            "output_vector_hex": edge.output_vector_hex,
            "weight": edge.weight,
            "colour_mapping": edge.colour_mapping,
            "position_delta": list(edge.position_delta),
            "size_ratio": edge.size_ratio,
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_crg(path: str = CRG_STATE_PATH) -> ObjectCRG:
    """Load a CRG from a JSON file. Returns empty CRG if file doesn't exist."""
    crg = ObjectCRG()
    if not os.path.exists(path):
        return crg

    with open(path, "r") as f:
        data = json.load(f)

    crg.global_colour_mapping = data.get("global_colour_mapping", {})
    crg.transform_type_counts = dict(data.get("transform_type_counts", {}))
    crg.nodes = set(data.get("nodes", []))

    for edge_data in data.get("edges", []):
        edge = ObjectTransformEdge(
            input_colour=edge_data["input_colour"],
            output_colour=edge_data["output_colour"],
            transform_type=edge_data["transform_type"],
            input_vector_hex=edge_data["input_vector_hex"],
            output_vector_hex=edge_data["output_vector_hex"],
            weight=edge_data["weight"],
            colour_mapping=edge_data.get("colour_mapping", {}),
            position_delta=tuple(edge_data.get("position_delta", [0.0, 0.0])),
            size_ratio=edge_data.get("size_ratio", 1.0),
        )
        crg.edges_by_colour[edge.input_colour].append(edge)
        crg.all_edges.append(edge)

    return crg


def merge_crgs(base: ObjectCRG, new: ObjectCRG) -> ObjectCRG:
    """Merge a new CRG into a base CRG (accumulates learning).

    Edges with the same (input_colour, output_colour, transform_type)
    are reinforced (weights added). New edges are added.
    """
    for new_edge in new.all_edges:
        # Check if this edge already exists in base
        found = False
        for base_edge in base.edges_by_colour.get(new_edge.input_colour, []):
            if (base_edge.output_colour == new_edge.output_colour
                    and base_edge.transform_type == new_edge.transform_type):
                # Reinforce
                base_edge.weight += new_edge.weight
                found = True
                break
        if not found:
            base.edges_by_colour[new_edge.input_colour].append(new_edge)
            base.all_edges.append(new_edge)
            base.nodes.add(new_edge.input_vector_hex)
            base.nodes.add(new_edge.output_vector_hex)

    # Merge global colour mapping
    for old_colour, new_colour in new.global_colour_mapping.items():
        base.global_colour_mapping[old_colour] = new_colour

    # Merge transform type counts
    for t, count in new.transform_type_counts.items():
        base.transform_type_counts[t] += count

    return base
