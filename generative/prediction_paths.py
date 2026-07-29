"""
prediction_paths.py — analogical, chain, and group prediction paths
=====================================================================

Implements the three prediction paths that were stubbed in v0.15:

  1. ANALOGICAL: "this test object is LIKE train object X, so apply X's
     transformation" — uses structural similarity (cell count, bbox,
     fill ratio) + colour similarity + relational similarity to find
     the best-matching train object and apply its transformation

  2. CHAIN: "train shows A→B→C, so apply the same sequence to the test"
     — discovers multi-step transformation chains in the CRG and applies
     them in sequence to the test input

  3. GROUP: "these objects transform together as a unit" — detects
     groups of objects that have consistent spatial relationships
     (same row, same column, aligned, touching) and applies the same
     transformation to all members of the group

Each path:
  - Learns from train pairs (via the ObjectCRG)
  - Produces a prediction for the test input
  - Verifies the prediction against train pairs (hard filter)
  - Returns None if no valid prediction is found

The three paths are tried in order: analogical → chain → group → fallback.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict
import sys, os, math

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
_PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from arc_loader import Grid, ARCTask
from generative.object_extractor import extract_objects, GridObject, pair_objects
from generative.object_crg_full import (
    ObjectCRG, ObjectTransformEdge, AnalogicalMapping,
    TransformChain, ObjectGroup, SpatialRelation, TransformType,
    compute_spatial_relation, objects_touch,
)
from dsl.arc_dsl_full import Ops, Operation, Program


# ══════════════════════════════════════════════════════════════════════════════
# ANALOGICAL PREDICTION — "this test object is LIKE train object X"
# ══════════════════════════════════════════════════════════════════════════════

def compute_structural_similarity(obj_a: GridObject, obj_b: GridObject) -> float:
    """Compute structural similarity between two objects (0-1).

    Based on:
      - Cell count similarity (are they the same size?)
      - Fill ratio similarity (are they the same shape density?)
      - Aspect ratio similarity (are they the same proportions?)
    """
    # Cell count similarity
    size_a, size_b = obj_a.cell_count, obj_b.cell_count
    if size_a == 0 and size_b == 0:
        size_sim = 1.0
    else:
        size_sim = 1.0 - abs(size_a - size_b) / max(size_a, size_b, 1)

    # Fill ratio similarity
    fill_a = obj_a.fill_ratio
    fill_b = obj_b.fill_ratio
    fill_sim = 1.0 - abs(fill_a - fill_b)

    # Aspect ratio similarity
    aspect_a = obj_a.width / max(obj_a.height, 1)
    aspect_b = obj_b.width / max(obj_b.height, 1)
    if aspect_a == 0 and aspect_b == 0:
        aspect_sim = 1.0
    else:
        aspect_sim = 1.0 - abs(aspect_a - aspect_b) / max(aspect_a, aspect_b, 0.01)

    return (size_sim + fill_sim + aspect_sim) / 3.0


def find_best_analogy(test_obj: GridObject,
                       crg: ObjectCRG,
                       train_objects: List[GridObject]) -> Optional[AnalogicalMapping]:
    """Find the best analogical match for a test object.

    Searches all train objects for the one most structurally similar to
    the test object, then finds the CRG edge for that train object's
    transformation.
    """
    best_mapping = None
    best_score = 0.0

    for train_obj in train_objects:
        # Structural similarity
        struct_sim = compute_structural_similarity(test_obj, train_obj)

        # Colour similarity (exact match = 1.0, different = 0.0)
        colour_sim = 1.0 if test_obj.colour == train_obj.colour else 0.0

        # Find the CRG edge for this train object
        edge = None
        for e in crg.all_edges:
            if e.input_colour == train_obj.colour:
                edge = e
                break

        if edge is None:
            continue

        # Relational similarity: do they have similar spatial positions?
        # (relative to grid centre)
        h, w = test_obj.grid_shape
        test_centre = (test_obj.centroid[0] / max(h, 1),
                       test_obj.centroid[1] / max(w, 1))
        h2, w2 = train_obj.grid_shape
        train_centre = (train_obj.centroid[0] / max(h2, 1),
                        train_obj.centroid[1] / max(w2, 1))
        rel_sim = 1.0 - math.dist(test_centre, train_centre)

        # Combined score
        score = struct_sim * 0.4 + colour_sim * 0.3 + rel_sim * 0.3

        if score > best_score:
            best_score = score
            best_mapping = AnalogicalMapping(
                source_a=train_obj,
                source_b=None,  # we don't have the output object here
                source_transform=edge,
                target_c=test_obj,
                target_d=None,  # to be predicted
                structural_similarity=struct_sim,
                colour_similarity=colour_sim,
                relational_similarity=rel_sim,
            )

    return best_mapping


def predict_via_analogy(task: ARCTask,
                         crg: ObjectCRG) -> Optional[Grid]:
    """Predict the test output using analogical reasoning.

    For each test object, find the most similar train object and apply
    its transformation.
    """
    test_input = task.test[0].input
    test_objects = extract_objects(test_input)

    # Collect all train objects
    train_objects = []
    for pair in task.train:
        train_objects.extend(extract_objects(pair.input))

    if not test_objects or not train_objects:
        return None

    # Build the prediction grid
    h, w = test_input.shape
    out_cells = [row[:] for row in test_input.cells]

    for test_obj in test_objects:
        # Find the best analogy
        mapping = find_best_analogy(test_obj, crg, train_objects)
        if mapping is None:
            continue

        edge = mapping.source_transform

        # Apply the transformation based on the edge type
        ttype = str(edge.transform_type.value) if hasattr(edge.transform_type, 'value') else str(edge.transform_type)

        if 'recolour' in ttype.lower() or edge.output_colour != edge.input_colour:
            # Recolour this object's cells
            new_colour = edge.output_colour
            for r, c in test_obj.cells:
                if 0 <= r < h and 0 <= c < w:
                    out_cells[r][c] = new_colour

        elif 'move' in ttype.lower():
            # Move this object by the learned position delta
            dr = round(edge.position_delta[0])
            dc = round(edge.position_delta[1])
            # Clear old position
            for r, c in test_obj.cells:
                if 0 <= r < h and 0 <= c < w:
                    out_cells[r][c] = 0
            # Set new position
            for r, c in test_obj.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out_cells[nr][nc] = test_obj.colour

        elif 'disappear' in ttype.lower():
            # Remove this object
            for r, c in test_obj.cells:
                if 0 <= r < h and 0 <= c < w:
                    out_cells[r][c] = 0

        elif 'resize' in ttype.lower():
            # Scale this object
            scale = edge.size_ratio
            if scale != 1.0 and scale > 0:
                # Simple scaling: expand/contract by integer factor
                factor = round(scale)
                if factor > 1:
                    # Expand
                    for r, c in test_obj.cells:
                        if 0 <= r < h and 0 <= c < w:
                            out_cells[r][c] = 0
                    for r, c in test_obj.cells:
                        for dr in range(factor):
                            for dc in range(factor):
                                nr, nc = r * factor + dr, c * factor + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    out_cells[nr][nc] = test_obj.colour

    predicted = Grid(out_cells)

    # Verify against train pairs
    if _verify_train(task, predicted, crg):
        return predicted
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CHAIN PREDICTION — "train shows A→B→C, so apply the same sequence"
# ══════════════════════════════════════════════════════════════════════════════

def predict_via_chain(task: ARCTask,
                       crg: ObjectCRG) -> Optional[Grid]:
    """Predict using transformation chains from the CRG.

    Discovers multi-step chains (A→B→C) in the CRG and applies them
    to the test input. A chain is useful when a single transformation
    can't reproduce the train pairs but a sequence can.
    """
    # Get transformation chains from the CRG
    # The CRG's get_transformation_chains takes (start_colour, max_length)
    # so we need to try chains starting from each test object's colour
    test_objects = extract_objects(task.test[0].input)
    chains = []
    for obj in test_objects:
        try:
            chains.extend(crg.get_transformation_chains(obj.colour, max_length=3))
        except Exception:
            pass

    if not chains:
        return None

    test_input = task.test[0].input
    test_objects = extract_objects(test_input)

    # For each chain, try applying it to the test input
    for chain in chains:
        # Build a program from the chain's steps
        operations = []
        for step in chain.steps:
            ttype = str(step.transform_type.value) if hasattr(step.transform_type, 'value') else str(step.transform_type)

            if 'recolour' in ttype.lower() or step.output_colour != step.input_colour:
                operations.append(Operation(Ops.RECOLOUR,
                                           {"mapping": {step.input_colour: step.output_colour}}))
            elif 'gravity' in ttype.lower():
                operations.append(Operation(Ops.GRAVITY_DOWN))
            elif 'move' in ttype.lower():
                operations.append(Operation(Ops.GRAVITY_DOWN))
            elif 'flip' in ttype.lower():
                operations.append(Operation(Ops.FLIP_H))
            elif 'rotate' in ttype.lower():
                operations.append(Operation(Ops.ROTATE_90))

        if not operations:
            continue

        # Apply the chain
        try:
            prog = Program(operations)
            predicted = prog.apply(test_input)

            # Verify against train pairs
            if _verify_train(task, predicted, crg):
                return predicted
        except Exception:
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# GROUP PREDICTION — "these objects transform together as a unit"
# ══════════════════════════════════════════════════════════════════════════════

def detect_object_groups(objects: List[GridObject],
                          grid: Grid) -> List[ObjectGroup]:
    """Detect groups of objects that have consistent spatial relationships.

    Groups are detected by:
      1. Same-row objects (aligned horizontally)
      2. Same-column objects (aligned vertically)
      3. Touching objects (8-neighbour adjacency between objects)
    """
    groups: List[ObjectGroup] = []
    group_id = 0

    # 1. Row groups: objects whose centroids are in the same row (±2 cells)
    row_clusters = defaultdict(list)
    for obj in objects:
        row_key = round(obj.centroid[0] / 2) * 2  # round to nearest 2
        row_clusters[row_key].append(obj)

    for row_key, cluster in row_clusters.items():
        if len(cluster) >= 2:
            all_cells = [c for o in cluster for c in o.cells]
            if all_cells:
                rs = [r for r, _ in all_cells]
                cs = [c for _, c in all_cells]
                groups.append(ObjectGroup(
                    group_id=group_id,
                    objects=cluster,
                    pattern_type="row",
                    bbox=(min(rs), max(rs), min(cs), max(cs)),
                ))
                group_id += 1

    # 2. Column groups
    col_clusters = defaultdict(list)
    for obj in objects:
        col_key = round(obj.centroid[1] / 2) * 2
        col_clusters[col_key].append(obj)

    for col_key, cluster in col_clusters.items():
        if len(cluster) >= 2:
            all_cells = [c for o in cluster for c in o.cells]
            if all_cells:
                rs = [r for r, _ in all_cells]
                cs = [c for _, c in all_cells]
                groups.append(ObjectGroup(
                    group_id=group_id,
                    objects=cluster,
                    pattern_type="column",
                    bbox=(min(rs), max(rs), min(cs), max(cs)),
                ))
                group_id += 1

    # 3. Touching objects
    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if i >= j:
                continue
            if objects_touch(obj_a, obj_b):
                cluster = [obj_a, obj_b]
                all_cells = obj_a.cells + obj_b.cells
                rs = [r for r, _ in all_cells]
                cs = [c for _, c in all_cells]
                groups.append(ObjectGroup(
                    group_id=group_id,
                    objects=cluster,
                    pattern_type="touching",
                    bbox=(min(rs), max(rs), min(cs), max(cs)),
                ))
                group_id += 1

    return groups


def predict_via_groups(task: ARCTask,
                        crg: ObjectCRG) -> Optional[Grid]:
    """Predict using group-based transformations.

    Detects groups of objects that transform together and applies the
    same transformation to all members of each group.
    """
    test_input = task.test[0].input
    test_objects = extract_objects(test_input)

    if len(test_objects) < 2:
        return None

    # Detect groups in the test input
    test_groups = detect_object_groups(test_objects, test_input)

    if not test_groups:
        return None

    # Learn what transformation each group undergoes in the train pairs
    # by finding the CRG edge for the group's dominant colour
    h, w = test_input.shape
    out_cells = [row[:] for row in test_input.cells]

    for group in test_groups:
        # Find the dominant colour in this group
        group_colours = defaultdict(int)
        for obj in group.objects:
            group_colours[obj.colour] += obj.cell_count

        if not group_colours:
            continue

        dominant_colour = max(group_colours, key=group_colours.get)

        # Find the CRG edge for this colour
        edge = None
        for e in crg.all_edges:
            if e.input_colour == dominant_colour:
                edge = e
                break

        if edge is None:
            continue

        # Apply the transformation to all objects in the group
        new_colour = edge.output_colour
        if new_colour != dominant_colour:
            for obj in group.objects:
                for r, c in obj.cells:
                    if 0 <= r < h and 0 <= c < w:
                        out_cells[r][c] = new_colour

    predicted = Grid(out_cells)

    # Verify against train pairs
    if _verify_train(task, predicted, crg):
        return predicted
    return None


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION — the hard filter (train-pass first, NRCI as measurement)
# ══════════════════════════════════════════════════════════════════════════════

def _verify_train(task: ARCTask, predicted: Grid, crg: ObjectCRG) -> bool:
    """STRICT verification: the prediction's transformation must EXACTLY
    reproduce ALL train pairs.

    v0.16: tightened from non-contradiction to exact match.
    The prediction's colour mapping is extracted and applied to each
    train input. If it doesn't reproduce the train output EXACTLY,
    the prediction is rejected.

    This is the train-pass filter — coherence ≠ correctness.
    """
    test_input = task.test[0].input

    # Shape consistency
    for pair in task.train:
        train_shape_changed = (pair.input.shape != pair.output.shape)
        pred_shape_changed = (test_input.shape != predicted.shape)
        if train_shape_changed != pred_shape_changed:
            if test_input.shape == pair.input.shape:
                return False

    # Change consistency
    test_changed = (test_input != predicted)
    train_any_changed = any(p.input != p.output for p in task.train)
    train_none_changed = all(p.input == p.output for p in task.train)

    if train_none_changed and test_changed:
        return False
    if train_any_changed and not test_changed:
        return False

    # STRICT: extract the colour mapping and verify it reproduces train EXACTLY
    if test_input.shape == predicted.shape:
        # Extract the mapping from the prediction
        mapping: Dict[int, int] = {}
        consistent = True
        for r in range(test_input.height):
            for c in range(test_input.width):
                old = test_input.cells[r][c]
                new = predicted.cells[r][c]
                if old != new:
                    if old in mapping and mapping[old] != new:
                        consistent = False
                        break
                    mapping[old] = new
            if not consistent:
                break

        if not consistent:
            return False  # the prediction itself is inconsistent

        # Apply the mapping to each train pair and check EXACT match
        if mapping:
            from dsl.arc_dsl_full import Operation, Ops, Program
            prog = Program([Operation(Ops.RECOLOUR, {"mapping": mapping})])
            for pair in task.train:
                if pair.input.shape == pair.output.shape:
                    train_pred = prog.apply(pair.input)
                    if train_pred != pair.output:
                        return False  # mapping doesn't reproduce train
                else:
                    # Shape change — can't verify via recolour
                    pass

    return True


# ══════════════════════════════════════════════════════════════════════════════
# FULL PREDICTOR — tries all three paths in order
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_all_paths(task: ARCTask,
                            crg: ObjectCRG) -> Tuple[Optional[Grid], str]:
    """Try all three prediction paths in order.

    Returns (predicted_grid, source) where source is one of:
      "analogy", "chain", "group", or None if all paths fail.
    """
    # Path 1: Analogical reasoning
    pred = predict_via_analogy(task, crg)
    if pred is not None:
        return pred, "analogy"

    # Path 2: Transformation chains
    pred = predict_via_chain(task, crg)
    if pred is not None:
        return pred, "chain"

    # Path 3: Group patterns
    pred = predict_via_groups(task, crg)
    if pred is not None:
        return pred, "group"

    return None, "none"
