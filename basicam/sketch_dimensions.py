"""Smart Dimension (sketch-engine.spec B7), no Qt: what a selection dimensions, a safe
arithmetic parser for typed values, measured values and label geometry."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from basicam.sketch_solver import plane_for, reference_points

LINE_KINDS = ("Line", "Construction line")
_UNITS_MM = {
    "mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "inch": 25.4, "inches": 25.4, '"': 25.4,
    "ft": 304.8, "foot": 304.8, "feet": 304.8, "'": 304.8,
}


# ---------------------------------------------------------------- inference
@dataclass
class Pick:
    """One picked item: an entity, or a point of it (point_index from reference_points)."""

    entity_id: str
    kind: str
    point_index: int | None = None


@dataclass
class DimensionPlan:
    kind: str  # SketchDimension kind
    targets: list[str] = field(default_factory=list)
    point_refs: list[int] = field(default_factory=list)
    orientable: bool = False  # aligned / horizontal / vertical chosen by mouse position
    angular: bool = False

    @property
    def owner(self) -> str:
        return self.targets[-1] if self.targets else ""


def infer_dimension(picks: list[Pick]) -> DimensionPlan | None:
    """The dimension a selection makes (SolidWorks Smart Dimension rules, spec B7)."""
    if not picks:
        return None
    if len(picks) == 1:
        p = picks[0]
        if p.point_index is not None or p.kind == "Point":
            return DimensionPlan("X", [p.entity_id], [p.point_index or 0], orientable=True)  # from the origin
        if p.kind in LINE_KINDS:
            return DimensionPlan("Length", [p.entity_id])
        if p.kind == "Circle":
            return DimensionPlan("Diameter", [p.entity_id])
        if p.kind in ("Curve", "Slot"):
            return DimensionPlan("Radius", [p.entity_id])
        if p.kind == "Rectangle":
            return DimensionPlan("Width", [p.entity_id], orientable=True)
        return None
    a, b = picks[0], picks[1]
    a_point = a.point_index is not None or a.kind == "Point"
    b_point = b.point_index is not None or b.kind == "Point"
    a_centre = a.kind == "Circle" and a.point_index is None
    b_centre = b.kind == "Circle" and b.point_index is None
    if (a_point or a_centre) and (b_point or b_centre):
        return DimensionPlan(
            "Distance", [a.entity_id, b.entity_id], [a.point_index or 0, b.point_index or 0], orientable=True
        )
    if (a_point or a_centre) and b.kind in LINE_KINDS:
        return DimensionPlan("PointLine", [a.entity_id, b.entity_id], [a.point_index or 0, 0])
    if (b_point or b_centre) and a.kind in LINE_KINDS:
        return DimensionPlan("PointLine", [b.entity_id, a.entity_id], [b.point_index or 0, 0])
    if a.kind in LINE_KINDS and b.kind in LINE_KINDS:
        return DimensionPlan("Angle", [a.entity_id, b.entity_id], angular=True)  # parallel: see refine_plan
    return None


def refine_plan(plan: DimensionPlan, features: dict, mouse_uv=None) -> DimensionPlan:
    """Settle choices that depend on geometry or mouse position (parallel lines, H/V/aligned)."""
    if plan.kind == "Angle" and len(plan.targets) == 2:
        a, b = (features[t] for t in plan.targets)
        da, db = (np.asarray(f.points[-1], float) - np.asarray(f.points[0], float) for f in (a, b))
        cross = float(np.linalg.norm(np.cross(da, db)))
        if cross <= 1e-6 * max(float(np.linalg.norm(da) * np.linalg.norm(db)), 1e-12):
            # parallel lines: distance between them (point of the second to the first line)
            return DimensionPlan("PointLine", [plan.targets[1], plan.targets[0]], [0, 0])
        return plan
    if not plan.orientable or mouse_uv is None:
        return plan
    pts = _plan_points(plan, features)
    if plan.kind in ("Width", "Height"):
        pts = list(reference_points(features[plan.targets[0]]))
    if not pts:
        return plan
    sketch_plane = plane_for([features[t] for t in plan.targets])
    uv = np.asarray([sketch_plane.world_to_uv(p) for p in pts], dtype=float)
    mu, mv = float(mouse_uv[0]), float(mouse_uv[1])
    umin, umax = uv[:, 0].min(), uv[:, 0].max()
    vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
    above_below = (mv > vmax or mv < vmin) and umin <= mu <= umax
    left_right = (mu > umax or mu < umin) and vmin <= mv <= vmax
    if plan.kind == "Distance":
        kind = "HDistance" if above_below else "VDistance" if left_right else "Distance"
        return DimensionPlan(kind, plan.targets, plan.point_refs, orientable=True)
    if plan.kind in ("X", "Y"):
        # one point to the origin: label above/below -> X (horizontal), left/right -> Y
        return DimensionPlan("Y" if left_right else "X", plan.targets, plan.point_refs, orientable=True)
    if plan.kind in ("Width", "Height"):
        return DimensionPlan("Height" if left_right else "Width", plan.targets, plan.point_refs, orientable=True)
    return plan


def _plan_points(plan: DimensionPlan, features: dict) -> list[np.ndarray]:
    pts = []
    for index, target in enumerate(plan.targets):
        feat = features.get(target)
        if feat is None:
            continue
        refs = reference_points(feat)
        if feat.kind in ("Rectangle",) and plan.kind in ("Width", "Height"):
            pts.extend(refs[0:2] if plan.kind == "Width" else refs[1:3])
            continue
        point_index = plan.point_refs[index] if index < len(plan.point_refs) else 0
        if refs:
            pts.append(np.asarray(refs[min(point_index, len(refs) - 1)], dtype=float))
    if plan.kind in ("X", "Y"):
        pts.append(np.zeros(3))
    return pts


# ---------------------------------------------------------------- measuring
def measure(kind: str, targets: list[str], point_refs: list[int], features: dict) -> float | None:
    """Current geometric value of a dimension (driven dimensions show this)."""
    feats = [features.get(t) for t in targets]
    if not feats or any(f is None for f in feats):
        return None
    plane = plane_for(feats)

    def point(i):
        refs = reference_points(feats[i])
        index = point_refs[i] if i < len(point_refs) else 0
        return np.asarray(refs[min(index, len(refs) - 1)], dtype=float)

    def uv(p):
        return np.asarray(plane.world_to_uv(p), dtype=float)

    first = feats[0]
    if kind == "Length" and len(first.points) >= 2:
        return float(np.linalg.norm(np.asarray(first.points[-1]) - np.asarray(first.points[0])))
    if kind in ("Diameter", "Radius"):
        refs = reference_points(first)
        centre = np.asarray(refs[0], dtype=float)
        pts = np.asarray(first.points, dtype=float)
        radius = float(np.mean(np.linalg.norm(pts - centre, axis=1))) if first.kind != "Slot" else float(
            np.linalg.norm(np.asarray(refs[2]) - centre)
        )
        return 2.0 * radius if kind == "Diameter" else radius
    if kind in ("Width", "Height") and first.kind == "Rectangle":
        refs = reference_points(first)
        return float(np.linalg.norm(refs[1] - refs[0])) if kind == "Width" else float(np.linalg.norm(refs[2] - refs[1]))
    if kind in ("X", "Y", "Z"):
        p = point(0)
        return float(p["XYZ".index(kind)])
    if kind == "Distance" and len(feats) == 2:
        return float(np.linalg.norm(point(1) - point(0)))
    if kind in ("HDistance", "VDistance") and len(feats) == 2:
        d = uv(point(1)) - uv(point(0))
        return abs(float(d[0] if kind == "HDistance" else d[1]))
    if kind == "PointLine" and len(feats) == 2:
        p = uv(point(0))
        a, b = (uv(np.asarray(feats[1].points[i], dtype=float)) for i in (0, -1))
        ab = b - a
        length = float(np.linalg.norm(ab))
        return abs(float(ab[0] * (p - a)[1] - ab[1] * (p - a)[0])) / length if length > 1e-12 else None
    if kind == "Angle" and len(feats) == 2:
        da, db = (uv(np.asarray(f.points[-1], float)) - uv(np.asarray(f.points[0], float)) for f in feats)
        cosine = float(np.dot(da, db)) / max(float(np.linalg.norm(da) * np.linalg.norm(db)), 1e-12)
        return math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
    return None


# ---------------------------------------------------------------- label geometry
def label_geometry(kind: str, targets: list[str], point_refs: list[int], label_offset, features: dict):
    """(segments [(2, 3) arrays], label world position) for drawing a dimension."""
    feats = [features.get(t) for t in targets]
    if not feats or any(f is None for f in feats):
        return [], None
    plane = plane_for(feats)
    offset = np.asarray(label_offset if label_offset else (0.0, 0.0), dtype=float)[:2]

    def world(u, v):
        return np.asarray(plane.uv_to_world(float(u), float(v)), dtype=float)

    plan = DimensionPlan(kind, targets, point_refs)
    if kind == "Length":
        pts = [np.asarray(feats[0].points[0], float), np.asarray(feats[0].points[-1], float)]
    elif kind in ("Diameter", "Radius"):
        centre = np.asarray(reference_points(feats[0])[0], dtype=float)
        cu, cv = plane.world_to_uv(centre)
        label = world(cu + offset[0], cv + offset[1])
        return [np.vstack((centre, label))], label
    elif kind == "PointLine":
        pts = [np.asarray(reference_points(feats[0])[point_refs[0] if point_refs else 0], float)]
        a, b = np.asarray(feats[1].points[0], float), np.asarray(feats[1].points[-1], float)
        pa, pb, pp = (np.asarray(plane.world_to_uv(x)) for x in (a, b, pts[0]))
        ab = pb - pa
        t = float(np.dot(pp - pa, ab) / max(float(np.dot(ab, ab)), 1e-12))
        pts.append(world(*(pa + ab * t)))
    elif kind == "Angle":
        vertex = np.asarray(feats[0].points[0], float)
        vu, vv = plane.world_to_uv(vertex)
        label = world(vu + offset[0], vv + offset[1])
        return [np.vstack((vertex, label))], label
    else:
        pts = _plan_points(plan, features)
    if len(pts) < 2:
        return [], (pts[0] if pts else None)
    a_uv, b_uv = (np.asarray(plane.world_to_uv(p), dtype=float) for p in pts[:2])
    mid = (a_uv + b_uv) / 2.0
    if kind in ("HDistance", "X", "Width"):
        line_a, line_b = np.array([a_uv[0], mid[1] + offset[1]]), np.array([b_uv[0], mid[1] + offset[1]])
    elif kind in ("VDistance", "Y", "Height"):
        line_a, line_b = np.array([mid[0] + offset[0], a_uv[1]]), np.array([mid[0] + offset[0], b_uv[1]])
    else:
        d = b_uv - a_uv
        length = float(np.linalg.norm(d)) or 1.0
        normal = np.array([-d[1], d[0]]) / length
        shift = normal * float(np.dot(offset, normal))
        line_a, line_b = a_uv + shift, b_uv + shift
    witness_a = np.vstack((pts[0], world(*line_a)))
    witness_b = np.vstack((pts[1], world(*line_b)))
    dim_line = np.vstack((world(*line_a), world(*line_b)))
    label = world(*(mid + offset))  # the label sits where it was placed / dragged
    return [witness_a, witness_b, dim_line], label


def label_anchor_uv(kind: str, targets: list[str], point_refs: list[int], features: dict) -> np.ndarray | None:
    """The uv point a dimension's label_offset is measured from."""
    feats = [features.get(t) for t in targets]
    if not feats or any(f is None for f in feats):
        return None
    plane = plane_for(feats)
    if kind in ("Diameter", "Radius"):
        return np.asarray(plane.world_to_uv(reference_points(feats[0])[0]), dtype=float)
    if kind == "Angle":
        return np.asarray(plane.world_to_uv(np.asarray(feats[0].points[0], float)), dtype=float)
    if kind == "Length":
        pts = [np.asarray(feats[0].points[0], float), np.asarray(feats[0].points[-1], float)]
    elif kind == "PointLine":
        segments, _label = label_geometry(kind, targets, point_refs, (0.0, 0.0), features)
        if not segments:
            return None
        pts = [segments[0][0], segments[1][0]]
    else:
        pts = _plan_points(DimensionPlan(kind, targets, point_refs), features)
    if not pts:
        return None
    uv = np.asarray([plane.world_to_uv(p) for p in pts[:2]], dtype=float)
    return uv.mean(axis=0)


def label_text(kind: str, value: float, driving: bool, unit: str = "mm") -> str:
    if kind == "Angle":
        text = f"{value:.2f}°"
    else:
        shown = value / 25.4 if unit in ("in", "inch", "inches") else value
        prefix = "Ø" if kind == "Diameter" else "R" if kind == "Radius" else ""
        text = f"{prefix}{shown:.4g} {'in' if unit in ('in', 'inch', 'inches') else 'mm'}"
    return text if driving else f"({text})"


# ---------------------------------------------------------------- arithmetic parser
_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<num>(?:\d+\s+)?\d*\.?\d+(?:/\d*\.?\d+)?)\s*(?P<unit>mm|cm|inches|inch|in|ft|feet|foot|m(?![a-z])|\"|')?"
    r"|(?P<op>[-+*/()]))"
)


def _number(text: str) -> float:
    text = text.strip()
    whole = 0.0
    if " " in text:
        head, text = text.split(None, 1)
        whole = float(head)
    if "/" in text:
        num, den = text.split("/")
        return whole + float(num) / float(den)
    return whole + float(text)


def parse_dimension_expression(text: str, default_unit_mm: float = 1.0, angular: bool = False) -> float | None:
    """Evaluate '10+2.5', '1/2in*2', '(3 ft)/4' safely (numbers, units, + - * / and brackets).

    Lengths come back in millimetres (a bare number uses ``default_unit_mm``); angles in degrees.
    No eval: a small recursive-descent parser. Returns None for anything else.
    """
    tokens: list[tuple[str, object]] = []
    position, text = 0, (text or "").strip().lower()
    if not text:
        return None
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            return None
        position = match.end()
        if match.group("op"):
            tokens.append(("op", match.group("op")))
        else:
            try:
                value = _number(match.group("num"))
            except (ValueError, ZeroDivisionError):
                return None
            unit = match.group("unit")
            if unit and angular:
                return None
            tokens.append(("len" if unit else "num", value * _UNITS_MM[unit] if unit else value))
    index = 0

    def peek():
        return tokens[index] if index < len(tokens) else (None, None)

    def take():
        nonlocal index
        index += 1
        return tokens[index - 1]

    def factor():
        kind, value = peek()
        if kind == "op" and value in "+-":
            take()
            is_len, number = factor()
            return is_len, -number if value == "-" else number
        if kind == "op" and value == "(":
            take()
            result = expression()
            if peek() != ("op", ")"):
                raise ValueError("missing )")
            take()
            return result
        if kind in ("num", "len"):
            take()
            return kind == "len", float(value)
        raise ValueError("expected a number")

    def term():
        is_len, value = factor()
        while peek()[0] == "op" and peek()[1] in "*/":
            op = take()[1]
            other_len, other = factor()
            if op == "*":
                if is_len and other_len:
                    raise ValueError("length x length")
                is_len, value = is_len or other_len, value * other
            else:
                if other == 0 or (other_len and not is_len):
                    raise ValueError("bad division")
                is_len, value = is_len and not other_len, value / other
        return is_len, value

    def expression():
        is_len, value = term()
        while peek()[0] == "op" and peek()[1] in "+-":
            op = take()[1]
            other_len, other = term()
            if is_len != other_len:  # a bare number next to a length is in the default unit
                value = value if is_len else value * default_unit_mm
                other = other if other_len else other * default_unit_mm
                is_len = True
            value = value + other if op == "+" else value - other
        return is_len, value

    try:
        is_len, value = expression()
        if index != len(tokens):
            return None
    except (ValueError, ZeroDivisionError):
        return None
    if angular:
        return value
    return value if is_len else value * default_unit_mm


__all__ = [
    "DimensionPlan", "Pick", "infer_dimension", "label_anchor_uv", "label_geometry", "label_text", "measure",
    "parse_dimension_expression", "refine_plan",
]
