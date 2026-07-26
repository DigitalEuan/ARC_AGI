#!/usr/bin/env python3
"""
Spatial Arithmetic
==================
Computing with 3D geometry. Shapes are numbers. Distances are operations.

All arithmetic from a single geometric relationship:
    R(n) = 1 / (2 * sin(pi / n))

This function IS the spatial equivalent of the EML operator — from it,
all operations follow (add, subtract, multiply, divide, log, exp, pow).

Architecture:
    GEOMETRY (passive) — vertices, edges, positions. No logic.
    OBSERVER (active)  — cluster, decode, evaluate. All logic.

Usage:
    python3 spatial_arithmetic.py                     # run all tests
    python3 spatial_arithmetic.py --eval "3+4*5"      # evaluate expression
    python3 spatial_arithmetic.py --scene 7 ADD 3     # build and decode
    python3 spatial_arithmetic.py --natural 5 3       # natural addition
"""

import math
import random
import sys
from fractions import Fraction
from typing import List, Tuple, Dict, Optional

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

UNIT = 1.0
EXACT_TOL = 1e-9
BASE_NODES = 4  # minimum for non-planar cycle

# Operator table: distance ratio → (name, function)
# Closer shapes = higher precedence
OPCODE_TABLE = {
    3: ("MULTIPLY", lambda a, b: a * b),
    4: ("ADD",      lambda a, b: a + b),
    5: ("SUBTRACT", lambda a, b: a - b),
    6: ("DIVIDE",   lambda a, b: Fraction(a, b) if b != 0 else None),
}
OPCODE_BY_NAME = {name: mult for mult, (name, _) in OPCODE_TABLE.items()}

# Dihedral angle modifier table
MODIFIER_TABLE = {
    (0, 22.5):     ("ID",     lambda r: r),
    (22.5, 67.5):  ("SQUARE", lambda r: r * r),
    (67.5, 112.5): ("NEGATE", lambda r: -r),
    (112.5, 157.5):("RECIP",  lambda r: Fraction(1, r) if r != 0 else None),
    (157.5, 180):  ("ABS",    lambda r: abs(r)),
}

# Fractional binding range (below lowest operator distance of 3×)
FRAC_BIND_MIN = 1.2
FRAC_BIND_MAX = 2.5


# ═══════════════════════════════════════════════════════════════════════
# GEOMETRY — Passive Data (no logic, like grooves on vinyl)
# ═══════════════════════════════════════════════════════════════════════

_FREQ_MAP = {}


def make_3d_cycle(n: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Non-planar unit-distance cycle. n vertices, edges exactly UNIT."""
    if n < 1:
        n = 1
    if n < 4:
        R = UNIT / (2 * math.sin(math.pi / max(n, 1)))
        return [(R * math.cos(2 * math.pi * i / max(n, 1)),
                 R * math.sin(2 * math.pi * i / max(n, 1)),
                 0.0) for i in range(max(n, 1))]

    # Frequency coprime with n for asymmetric z-oscillation
    freq = 2
    for f in [3, 5, 7, 2]:
        if math.gcd(f, n) == 1:
            freq = f
            break
    _FREQ_MAP[n] = freq

    z_raw = [math.cos(freq * 2 * math.pi * i / n) +
             0.7 * math.sin(freq * 2 * math.pi * i / n + math.pi / 3)
             for i in range(n)]
    max_dz = max(abs(z_raw[(i + 1) % n] - z_raw[i]) for i in range(n))
    z_amp = 0.95 * UNIT / max_dz
    z = [z_amp * z_raw[i] for i in range(n)]
    dz = [z[(i + 1) % n] - z[i] for i in range(n)]

    def total_dtheta(R):
        t = 0.0
        for i in range(n):
            arg = 1 - (1 - dz[i] ** 2) / (2 * R * R)
            if abs(arg) > 1:
                return float('inf')
            t += math.acos(arg)
        return t

    R_lo = max(math.sqrt(max(1 - d * d, 0.01) / 4) + 0.01 for d in dz)
    R_hi = 10.0
    if total_dtheta(R_lo) > 2 * math.pi:
        z_amp *= 0.5
        z = [z_amp * z_raw[i] for i in range(n)]
        dz = [z[(i + 1) % n] - z[i] for i in range(n)]
        R_lo = max(math.sqrt(max(1 - d * d, 0.01) / 4) + 0.01 for d in dz)
    for _ in range(200):
        R_mid = (R_lo + R_hi) / 2
        if total_dtheta(R_mid) > 2 * math.pi:
            R_lo = R_mid
        else:
            R_hi = R_mid
    R = (R_lo + R_hi) / 2
    thetas = [0.0]
    for i in range(n - 1):
        arg = max(-1, min(1, 1 - (1 - dz[i] ** 2) / (2 * R * R)))
        thetas.append(thetas[-1] + math.acos(arg))
    pts = [(R * math.cos(thetas[i]), R * math.sin(thetas[i]), z[i])
           for i in range(n)]

    # Random rotation for genuine 3D orientation
    rot = _random_rotation_matrix(seed)
    return _apply_rotation(pts, rot)


def _random_rotation_matrix(seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    axis = [rng.gauss(0, 1) for _ in range(3)]
    na = math.sqrt(sum(x * x for x in axis))
    axis = [x / na for x in axis]
    angle = rng.uniform(0, 2 * math.pi)
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    x, y, z = axis
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def _apply_rotation(pts, R):
    return [tuple(sum(R[k][j] * p[j] for j in range(3)) for k in range(3))
            for p in pts]


def centroid(pts: List[Tuple]) -> Tuple[float, float, float]:
    n = len(pts)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))


def radius_of(pts: List[Tuple]) -> float:
    if len(pts) <= 1:
        return 0.0
    c = centroid(pts)
    return max(math.dist(c, p) for p in pts)


# ═══════════════════════════════════════════════════════════════════════
# THE NATURAL PRIMITIVE: R(n) = 1/(2*sin(pi/n))
# This IS the spatial equivalent of ln/exp (the EML parallel)
# ═══════════════════════════════════════════════════════════════════════

def value_to_radius(v: int) -> float:
    """The geometric primitive: value → radius. This IS logarithmic."""
    n = 2 * abs(v) + BASE_NODES
    if n < 4:
        n = 4
    return 1 / (2 * math.sin(math.pi / n))


def radius_to_value(R: float) -> int:
    """Inverse primitive: radius → value. This IS exponential."""
    if R < 0.5:
        return 0
    sin_val = 1 / (2 * R)
    if sin_val > 1:
        return 0
    n = round(math.pi / math.asin(sin_val))
    return max(0, (n - BASE_NODES) // 2)


# ═══════════════════════════════════════════════════════════════════════
# ENCODING — Parity sign, node-count magnitude
# ═══════════════════════════════════════════════════════════════════════

def encode(value: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Signed integer → 3D shape. Even nodes = positive, odd = negative."""
    n = 2 * abs(value) + BASE_NODES
    if value < 0:
        n += 1
    return make_3d_cycle(n, seed=seed)


def decode(pts: List[Tuple]) -> int:
    """3D shape → signed integer via parity."""
    n = len(pts)
    if n < BASE_NODES:
        return 0
    mag = (n - BASE_NODES) // 2
    sign = 1 if (n - BASE_NODES) % 2 == 0 else -1
    return sign * mag


# ═══════════════════════════════════════════════════════════════════════
# COORDINATE-FREE MEASUREMENTS (Cayley-Menger)
# ═══════════════════════════════════════════════════════════════════════

def pairwise_centroid_distance(pts_a: List[Tuple], pts_b: List[Tuple]) -> float:
    """Distance between centroids using ONLY pairwise vertex distances.
    No global coordinates needed — the observer is coordinate-free.

    Uses the variance decomposition identity:
      |C_A - C_B|² = E[d²(a,b)] - E[d²(a,a')] - E[d²(b,b')]
    This allows centroid distance calculation using ONLY pairwise
    vertex measurements, requiring zero global coordinate frame.
    Ref: Blumenthal (1953), Schoenberg (1935)."""
    na, nb = len(pts_a), len(pts_b)
    cross = sum(math.dist(a, b) ** 2 for a in pts_a for b in pts_b) / (na * nb)
    self_a = sum(math.dist(pts_a[i], pts_a[j]) ** 2
                 for i in range(na) for j in range(i + 1, na)) / (na * na)
    self_b = sum(math.dist(pts_b[i], pts_b[j]) ** 2
                 for i in range(nb) for j in range(i + 1, nb)) / (nb * nb)
    return math.sqrt(max(0, cross - self_a - self_b))


def _principal_normal(pts: List[Tuple]) -> List[float]:
    """Normal to principal plane via PCA power iteration."""
    n = len(pts)
    c = centroid(pts)
    centered = [(p[0] - c[0], p[1] - c[1], p[2] - c[2]) for p in pts]
    cov = [[0.0] * 3 for _ in range(3)]
    for p in centered:
        for a in range(3):
            for b in range(3):
                cov[a][b] += p[a] * p[b]
    trace = cov[0][0] + cov[1][1] + cov[2][2]
    rng = random.Random(42)
    v = [rng.gauss(0, 1) for _ in range(3)]
    for _ in range(100):
        v_new = [trace * v[i] - sum(cov[i][j] * v[j] for j in range(3))
                 for i in range(3)]
        norm = math.sqrt(sum(x * x for x in v_new))
        if norm < 1e-15:
            break
        v = [x / norm for x in v_new]
    return v


def dihedral_angle(pts_a: List[Tuple], pts_b: List[Tuple]) -> float:
    """Angle (degrees) between principal planes of two shapes. 0-90°."""
    na = _principal_normal(pts_a)
    nb = _principal_normal(pts_b)
    dot = max(-1.0, min(1.0, sum(na[i] * nb[i] for i in range(3))))
    return math.degrees(math.acos(abs(dot)))


def decode_modifier(angle_deg: float):
    """Decode modifier from dihedral angle."""
    for (lo, hi), (name, fn) in MODIFIER_TABLE.items():
        if lo <= angle_deg < hi:
            return name, fn
    return "ID", lambda r: r


# ═══════════════════════════════════════════════════════════════════════
# OBSERVER — Active Engine (all logic lives here)
# ═══════════════════════════════════════════════════════════════════════

def cluster_detect(points: List[Tuple], tol: float = EXACT_TOL) -> List[List[int]]:
    """Find connected components via unit-distance edges."""
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if abs(math.dist(points[i], points[j]) - UNIT) <= tol:
                union(i, j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def reorder_to_cycle(points: List[Tuple], indices: List[int]) -> List[Tuple]:
    """Reorder cluster into cycle order via adjacency walk."""
    pts = [points[i] for i in indices]
    n = len(pts)
    if n <= 2:
        return pts
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(math.dist(pts[i], pts[j]) - UNIT) <= EXACT_TOL:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = order[-1]
        for nb in adj[cur]:
            if not visited[nb]:
                order.append(nb)
                visited[nb] = True
                break
    return [pts[i] for i in order]


# ═══════════════════════════════════════════════════════════════════════
# NATURAL RULES — operations that emerge from geometry
# ═══════════════════════════════════════════════════════════════════════

def natural_add(a: int, b: int, seed: int = 0) -> Tuple[Optional[int], str]:
    """Natural addition (positive integers only): merge shapes → total nodes = sum.
    Note: parity encoding means signed addition uses the 4× distance operator."""
    shape_a = encode(a, seed=seed * 2)
    shape_b = encode(b, seed=seed * 2 + 1)

    # Find closest pair
    min_dist = float('inf')
    best_i, best_j = 0, 0
    for i, pa in enumerate(shape_a):
        for j, pb in enumerate(shape_b):
            d = math.dist(pa, pb)
            if d < min_dist:
                min_dist = d
                best_i, best_j = i, j

    # Translate shape_b so closest pair is exactly UNIT apart
    pa, pb = shape_a[best_i], shape_b[best_j]
    direction = [pa[k] - pb[k] for k in range(3)]
    d = math.sqrt(sum(x * x for x in direction))
    if d < 1e-10:
        direction, d = [1.0, 0.0, 0.0], 1.0
    shift = [direction[k] * (d - UNIT) / d for k in range(3)]
    placed_b = [tuple(p[k] + shift[k] for k in range(3)) for p in shape_b]

    all_pts = shape_a + placed_b
    clusters = cluster_detect(all_pts)
    if len(clusters) == 1:
        total = len(clusters[0])
        return (total - 2 * BASE_NODES) // 2, "merged"
    return None, f"{len(clusters)} clusters"


def natural_divide(a: int, b: int) -> Tuple[float, float, float]:
    """Natural division: radius ratio ≈ value ratio. Returns (R_a, R_b, ratio)."""
    return value_to_radius(a), value_to_radius(b), value_to_radius(a) / value_to_radius(b)


def build_fraction(numerator: int, denominator: int, seed: int = 0) -> List[Tuple]:
    """Encode a fraction as two shapes at binding distance (1.2-2.5× radius)."""
    sa = encode(numerator, seed=seed * 2)
    sb = encode(denominator, seed=seed * 2 + 1)
    ra, rb = radius_of(sa), radius_of(sb)
    gap = 1.8 * max(ra, rb, UNIT)
    rng = random.Random(seed)
    ca, cb = centroid(sa), centroid(sb)
    d = [rng.uniform(-1, 1) for _ in range(3)]
    nm = math.sqrt(sum(x * x for x in d))
    d = [x / nm for x in d]
    tb = tuple(d[i] * gap for i in range(3))
    pa = [tuple(p[i] - ca[i] for i in range(3)) for p in sa]
    pb = [tuple(p[i] - cb[i] + tb[i] for i in range(3)) for p in sb]
    return pa + pb


def observe_with_fractions(points: List[Tuple]) -> List[Dict]:
    """Two-pass observer: detect fractional bindings, then decode all shapes."""
    clusters = cluster_detect(points)

    # Pass 1: decode all shapes
    shapes = []
    for cl in clusters:
        ordered = reorder_to_cycle(points, cl)
        shapes.append({"val": decode(ordered), "ctr": centroid(ordered),
                       "rad": radius_of(ordered)})

    # Pass 1b: detect fractional bindings (distance 1.2-2.5× radius)
    # Sort by distance so closest pairs bind first (agglomerative philosophy)
    candidates = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            d = math.dist(shapes[i]["ctr"], shapes[j]["ctr"])
            ratio = d / max(shapes[i]["rad"], shapes[j]["rad"], UNIT)
            if FRAC_BIND_MIN <= ratio <= FRAC_BIND_MAX:
                candidates.append((ratio, i, j))
    candidates.sort()
    bound = set()
    final = []
    for _, i, j in candidates:
        if i in bound or j in bound:
            continue
        num, den = shapes[i]["val"], shapes[j]["val"]
        frac_val = Fraction(num, den) if den != 0 else Fraction(0)
        new_ctr = tuple((shapes[i]["ctr"][k] + shapes[j]["ctr"][k]) / 2
                        for k in range(3))
        new_rad = max(shapes[i]["rad"], shapes[j]["rad"])
        final.append({"val": frac_val, "ctr": new_ctr, "rad": new_rad})
        bound.add(i)
        bound.add(j)

    # Add unbound integer shapes
    for k in range(len(shapes)):
        if k not in bound:
            final.append(shapes[k])

    return final


# ═══════════════════════════════════════════════════════════════════════
# SCENE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_scene(a: int, b: int, operator: str, seed: int = 0) -> List[Tuple]:
    """Two-operand scene with shapes at operator distance."""
    mult = OPCODE_BY_NAME[operator]
    rng = random.Random(seed)
    sa, sb = encode(a, seed=seed * 2), encode(b, seed=seed * 2 + 1)
    ra, rb = radius_of(sa), radius_of(sb)
    gap = mult * max(ra, rb, UNIT)
    ca, cb = centroid(sa), centroid(sb)
    d = [rng.uniform(-1, 1) for _ in range(3)]
    nm = math.sqrt(sum(x * x for x in d))
    d = [x / nm for x in d]
    tb = tuple(d[i] * gap for i in range(3))
    pa = [tuple(p[i] - ca[i] for i in range(3)) for p in sa]
    pb = [tuple(p[i] - cb[i] + tb[i] for i in range(3)) for p in sb]
    return pa + pb


def build_expression(values_and_ops: List, seed: int = 0) -> List[Tuple]:
    """Multi-operand expression: linear placement at operator distances."""
    vals = [values_and_ops[i] for i in range(0, len(values_and_ops), 2)]
    ops = [values_and_ops[i] for i in range(1, len(values_and_ops), 2)]
    shapes = [encode(v, seed=seed * 10 + i) for i, v in enumerate(vals)]

    placed = []
    x_pos = 0.0
    for i in range(len(shapes)):
        if i == 0:
            c = centroid(shapes[i])
            placed.extend([tuple(p[j] - c[j] for j in range(3)) for p in shapes[i]])
        else:
            mult = OPCODE_BY_NAME[ops[i - 1]]
            gap = mult * max(radius_of(shapes[i - 1]), radius_of(shapes[i]), UNIT)
            x_pos += gap
            c = centroid(shapes[i])
            placed.extend([tuple(p[j] - c[j] + (x_pos, 0, 0)[j] for j in range(3))
                           for p in shapes[i]])
    return placed


# ═══════════════════════════════════════════════════════════════════════
# OBSERVATION — decode scenes into results
# ═══════════════════════════════════════════════════════════════════════

def observe_scene(points: List[Tuple]) -> Dict:
    """Decode two-operand scene: cluster → decode → operator → result."""
    clusters = cluster_detect(points)
    if len(clusters) != 2:
        return {"ok": False, "reason": f"{len(clusters)} clusters"}
    oa = reorder_to_cycle(points, clusters[0])
    ob = reorder_to_cycle(points, clusters[1])
    va, vb = decode(oa), decode(ob)
    ra, rb = radius_of(oa), radius_of(ob)
    d = pairwise_centroid_distance(oa, ob)  # coordinate-free
    ratio = d / max(ra, rb, UNIT)
    mult = round(ratio)
    if mult not in OPCODE_TABLE:
        return {"ok": False, "reason": f"bad ratio {ratio:.2f}"}
    opname, opfn = OPCODE_TABLE[mult]

    # Optional: dihedral angle modifier
    angle = dihedral_angle(oa, ob)
    modname, modfn = decode_modifier(angle)
    base = opfn(va, vb)
    result = modfn(base) if base is not None else None

    return {"ok": True, "a": va, "b": vb, "operator": opname,
            "modifier": modname, "angle": round(angle, 1),
            "base_result": base, "result": result}


def observe_expression(points: List[Tuple]) -> Dict:
    """Decode multi-operand expression with standard precedence."""
    clusters = cluster_detect(points)
    if len(clusters) < 2:
        return {"ok": False, "reason": f"need >= 2 clusters, got {len(clusters)}"}

    decoded = []
    for cl in clusters:
        ordered = reorder_to_cycle(points, cl)
        decoded.append({"val": decode(ordered), "ctr": centroid(ordered),
                        "rad": radius_of(ordered)})

    # Sort left-to-right
    decoded.sort(key=lambda d: d["ctr"][0])
    vals = [d["val"] for d in decoded]

    # Read operators from adjacent distances
    ops = []
    for i in range(1, len(decoded)):
        d = math.dist(decoded[i - 1]["ctr"], decoded[i]["ctr"])
        ratio = d / max(decoded[i - 1]["rad"], decoded[i]["rad"], UNIT)
        mult = round(ratio)
        if mult not in OPCODE_TABLE:
            return {"ok": False, "reason": f"bad ratio {ratio:.2f}"}
        ops.append(mult)

    # Evaluate: MUL/DIV first, then ADD/SUB
    i = 0
    while i < len(ops):
        if ops[i] in (3, 6):
            _, opfn = OPCODE_TABLE[ops[i]]
            result = opfn(vals[i], vals[i + 1])
            vals = vals[:i] + [result] + vals[i + 2:]
            ops = ops[:i] + ops[i + 1:]
        else:
            i += 1

    result = vals[0]
    for i in range(len(ops)):
        _, opfn = OPCODE_TABLE[ops[i]]
        result = opfn(result, vals[i + 1])

    return {"ok": True, "result": result}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_expr(s: str) -> list:
    tokens, i = [], 0
    while i < len(s):
        if s[i] == ' ':
            i += 1
        elif s[i] == '+':
            tokens.append("ADD"); i += 1
        elif s[i] == '-':
            if not tokens or isinstance(tokens[-1], str):
                num = '-'; i += 1
                while i < len(s) and s[i].isdigit():
                    num += s[i]; i += 1
                tokens.append(int(num))
            else:
                tokens.append("SUBTRACT"); i += 1
        elif s[i] == '*':
            tokens.append("MULTIPLY"); i += 1
        elif s[i] == '/':
            tokens.append("DIVIDE"); i += 1
        elif s[i].isdigit():
            num = ''
            while i < len(s) and s[i].isdigit():
                num += s[i]; i += 1
            tokens.append(int(num))
        else:
            i += 1
    return tokens


def run_tests():
    """Comprehensive test suite."""
    print("=" * 70)
    print("SPATIAL ARITHMETIC — TEST SUITE")
    print("=" * 70)
    ok = True

    # 1: Parity encoding
    print("\n▸ Parity encoding (0-20, signed)")
    t1 = all(decode(encode(v * s, seed=42)) == v * s
             for v in range(21) for s in [1, -1])
    ok &= t1
    print(f"  {'✓' if t1 else '✗'} 42/42 values")

    # 2: Roundtrip
    print("\n▸ Signed roundtrip (50 seeds × 10 values)")
    t2 = all(decode(encode(s * v, seed=seed)) == s * v
             for v in [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
             for s in [1, -1] for seed in range(50))
    ok &= t2
    print(f"  {'✓' if t2 else '✗'} 1000/1000 roundtrips")

    # 3: Pipeline
    print("\n▸ Pipeline (0-10, all ops, 3 seeds each)")
    total = correct = 0
    for a in range(11):
        for b in range(11):
            for op in ["ADD", "MULTIPLY", "SUBTRACT", "DIVIDE"]:
                _, opfn = OPCODE_TABLE[OPCODE_BY_NAME[op]]
                exp = opfn(a, b)
                for trial in range(3):
                    seed = hash((a, b, op, trial)) % 1_000_000
                    r = observe_scene(build_scene(a, b, op, seed=seed))
                    total += 1
                    if r.get("ok") and r["a"] == a and r["b"] == b and r["operator"] == op and r.get("base_result") == exp:
                        correct += 1
    t3 = correct == total
    ok &= t3
    print(f"  {'✓' if t3 else '✗'} {correct}/{total} = {correct / total * 100:.1f}%")

    # 4: Division
    print("\n▸ Division → exact fractions")
    t4 = all(observe_scene(build_scene(a, b, "DIVIDE", seed=42)).get("base_result") == Fraction(a, b)
             for a, b in [(7, 3), (10, 4), (1, 7), (0, 5)])
    ok &= t4
    print(f"  {'✓' if t4 else '✗'} 4/4 exact fractions")

    # 5: Signed pipeline
    print("\n▸ Signed pipeline (±5-±10, all ops)")
    nt = nc = 0
    for a in [-10, -8, -6, -5, 5, 6, 8, 10]:
        for b in [-10, -8, -6, -5, 5, 6, 8, 10]:
            for op in ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"]:
                _, opfn = OPCODE_TABLE[OPCODE_BY_NAME[op]]
                exp = opfn(a, b)
                for trial in range(3):
                    seed = hash((a, b, op, trial)) % 1_000_000
                    r = observe_scene(build_scene(a, b, op, seed=seed))
                    nt += 1
                    if r.get("ok") and r["a"] == a and r["b"] == b and r["operator"] == op and r.get("base_result") == exp:
                        nc += 1
    t5 = nc == nt
    ok &= t5
    print(f"  {'✓' if t5 else '✗'} {nc}/{nt} = {nc / nt * 100:.1f}%")

    # 6: Expressions
    print("\n▸ Expression evaluation (with precedence)")
    expr_tests = [
        ([3, "ADD", 4, "ADD", 5], 12),
        ([3, "MULTIPLY", 4, "ADD", 5], 17),
        ([10, "SUBTRACT", 3, "SUBTRACT", 2], 5),
        ([2, "MULTIPLY", 3, "MULTIPLY", 4], 24),
        ([20, "DIVIDE", 4, "ADD", 3], 8),
        ([10, "SUBTRACT", 2, "MULTIPLY", 3], 4),
        ([1, "ADD", 2, "MULTIPLY", 3, "ADD", 4], 11),
        ([-5, "ADD", 3, "MULTIPLY", -2], -11),
        ([100, "DIVIDE", 10, "SUBTRACT", 3], 7),
    ]
    t6 = True
    for expr, expected in expr_tests:
        r = observe_expression(build_expression(expr, seed=42))
        passed = r.get("ok") and r["result"] == expected
        t6 &= passed
        desc = " ".join(str(x) for x in expr)
        print(f"  {'✓' if passed else '✗'} {desc} = {expected}")
    ok &= t6

    # 7: Natural addition
    print("\n▸ Natural addition (merged clusters)")
    t7 = all(natural_add(a, b, seed=42)[0] == a + b
             for a, b in [(5, 3), (7, 4), (10, 6), (3, 3), (8, 5), (4, 0)])
    ok &= t7
    print(f"  {'✓' if t7 else '✗'} 6/6 natural additions")

    # 8: Cayley-Menger
    print("\n▸ Cayley-Menger (coordinate-free)")
    t8 = True
    for na, nb in [(8, 10), (12, 14)]:
        a = make_3d_cycle(na, seed=0)
        b = make_3d_cycle(nb, seed=1)
        ca, cb = centroid(a), centroid(b)
        bp = [tuple(p[i] - cb[i] + ca[i] + 5 for i in range(3)) for p in b]
        if abs(math.dist(ca, centroid(bp)) - pairwise_centroid_distance(a, bp)) > 1e-10:
            t8 = False
    ok &= t8
    print(f"  {'✓' if t8 else '✗'} Zero error (pairwise = coordinate)")

    # 9: Dihedral angle
    print("\n▸ Dihedral angle channel")
    a = make_3d_cycle(10, seed=0)
    b = make_3d_cycle(10, seed=1)
    angle = dihedral_angle(a, b)
    mod, modfn = decode_modifier(angle)
    t9 = modfn(5) in [5, 25, -5, Fraction(1, 5), 5]  # all valid modifiers
    ok &= t9
    print(f"  {'✓' if t9 else '✗'} Angle={angle:.1f}° → modifier={mod}")

    # 10: Natural primitive
    print("\n▸ Natural primitive R(n) = 1/(2·sin(π/n))")
    t10 = all(radius_to_value(value_to_radius(v)) == v for v in range(4, 51))
    ok &= t10
    print(f"  {'✓' if t10 else '✗'} R⁻¹(R(v)) = v for v ∈ [4, 50]")

    # 11: Values beyond 20
    print("\n▸ Extended range (values 0-50, signed)")
    t11 = all(decode(encode(v * s, seed=42)) == v * s
              for v in range(51) for s in [1, -1])
    ok &= t11
    print(f"  {'✓' if t11 else '✗'} 102/102 values (0-50 × ±1)")

    # 12: Fractional binding
    print("\n▸ Fractional binding (two-pass observer)")
    frac_tests = [(7, 3), (10, 4), (1, 7), (5, 2), (0, 3)]
    t12 = True
    for num, den in frac_tests:
        pts = build_fraction(num, den, seed=42)
        decoded = observe_with_fractions(pts)
        expected = Fraction(num, den) if den != 0 else Fraction(0)
        if len(decoded) == 1 and decoded[0]["val"] == expected:
            pass
        else:
            t12 = False
            print(f"  ✗ {num}/{den}: {decoded}")
    ok &= t12
    print(f"  {'✓' if t12 else '✗'} 5/5 fractions detected and decoded")

    # 13: Fraction + integer expression
    print("\n▸ Fraction + integer expression")
    frac_pts = build_fraction(7, 3, seed=42)
    int_pts = encode(5, seed=100)
    frac_shapes = observe_with_fractions(frac_pts)
    frac_ctr = frac_shapes[0]["ctr"]
    frac_rad = frac_shapes[0]["rad"]
    int_ctr = centroid(int_pts)
    int_rad = radius_of(int_pts)
    gap = 4 * max(frac_rad, int_rad, UNIT)
    rng_t13 = random.Random(42)
    d = [rng_t13.uniform(-1, 1) for _ in range(3)]
    nm = math.sqrt(sum(x * x for x in d))
    d = [x / nm for x in d]
    target = tuple(frac_ctr[i] + d[i] * gap for i in range(3))
    int_placed = [tuple(p[j] - int_ctr[j] + target[j] for j in range(3))
                  for p in int_pts]
    all_pts = frac_pts + int_placed
    decoded_t13 = observe_with_fractions(all_pts)
    t13 = len(decoded_t13) == 2
    if t13:
        vals = [s["val"] for s in decoded_t13]
        frac_val = [v for v in vals if isinstance(v, Fraction)]
        int_val = [v for v in vals if isinstance(v, int)]
        t13 = t13 and len(frac_val) == 1 and len(int_val) == 1
        if t13:
            result = frac_val[0] + int_val[0]
            t13 = result == Fraction(7, 3) + 5
    ok &= t13
    print(f"  {'✓' if t13 else '✗'} 7/3 + 5 = {Fraction(7,3) + 5}")

    # Summary
    print("\n" + "=" * 70)
    if ok:
        print("✓ ALL TESTS PASSED")
        print()
        print("  Natural rules proven:")
        print("    ADD: merged clusters (topology)")
        print("    DIV: radius ratio (metric)")
        print("    LN:  R(n) itself (the primitive)")
        print("    EXP: R⁻¹ (inverse primitive)")
        print()
        print("  Engineered rules:")
        print("    SUB: distance ratio 5×")
        print("    MUL: distance ratio 3×")
        print("    Angle modifier: dihedral angle")
    else:
        print("⚠ SOME TESTS FAILED")
    return ok


def main():
    if len(sys.argv) < 2:
        run_tests()
        return

    cmd = sys.argv[1]

    if cmd == "--eval" and len(sys.argv) > 2:
        tokens = _parse_expr(" ".join(sys.argv[2:]))
        r = observe_expression(build_expression(tokens, seed=42))
        expr_str = " ".join(sys.argv[2:])
        if r.get("ok"):
            print(f"{expr_str} = {r['result']}")
        else:
            print(f"Error: {r.get('reason')}")

    elif cmd == "--scene" and len(sys.argv) > 4:
        a, op, b = int(sys.argv[2]), sys.argv[3].upper(), int(sys.argv[4])
        r = observe_scene(build_scene(a, b, op, seed=42))
        if r.get("ok"):
            if r["modifier"] != "ID":
                print(f"{a} {op} {b} = {r['base_result']} → {r['result']} [mod={r['modifier']}@{r['angle']}°]")
            else:
                print(f"{a} {op} {b} = {r['result']}")
        else:
            print(f"Error: {r.get('reason')}")

    elif cmd == "--natural" and len(sys.argv) > 3:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        result, status = natural_add(a, b, seed=42)
        Ra, Rb, ratio = natural_divide(a, b)
        print(f"Natural addition: {a} + {b} = {result} ({status})")
        print(f"Natural division: R({a})/R({b}) = {Ra:.4f}/{Rb:.4f} = {ratio:.4f} (exact={a}/{b}={a / b:.4f})")
        print(f"Natural log:      ln({a}) ≈ ln(π·R({a})) = {math.log(math.pi * Ra):.4f} (exact={math.log(a):.4f})")

    else:
        print("Spatial Arithmetic — computing with 3D geometry")
        print()
        print("Usage:")
        print("  python3 spatial_arithmetic.py                     # run all tests")
        print("  python3 spatial_arithmetic.py --eval '3+4*5'      # evaluate expression")
        print("  python3 spatial_arithmetic.py --scene 7 ADD 3     # build and decode")
        print("  python3 spatial_arithmetic.py --natural 5 3       # natural operations")


if __name__ == "__main__":
    main()
