"""Sketch drawing interaction (no Qt): rubber-band previews, cursor readouts, relation inference,
line chaining and tangent arcs (sketch-engine.spec B4, B5)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from basicam.sketch_workflow import (
    SketchPlane,
    sample_catmull_rom_spline,
    sample_center_arc,
    sample_circle_points,
    sample_circle_three_points,
    sample_polygon,
    sample_rectangle_points,
    sample_rectangle_three_points,
    sample_slot,
    sample_three_point_arc,
)

HV_INFERENCE_DEG = 3.0  # a line within this angle of horizontal / vertical gets an H / V relation


def _polygon_sides(variant: str) -> int:
    try:
        return max(3, int(str(variant).split()[0]))
    except (ValueError, IndexError):
        return 6


def preview_geometry(tool: str, variant: str, points: list, cursor, plane: SketchPlane) -> np.ndarray | None:
    """Polyline (N, 3) of the entity being drawn, from its placed points and the cursor."""
    if cursor is None or plane is None:
        return None
    cursor = np.asarray(cursor, dtype=float)
    pts = [np.asarray(p, dtype=float) for p in points]
    if not pts:
        return None
    first = pts[0]
    try:
        if tool in ("Line", "Construction line", "Centerline"):
            return np.vstack((pts[-1], cursor))
        if tool == "Rectangle":
            if variant == "3 point":
                if len(pts) == 1:
                    return np.vstack((first, cursor))
                return sample_rectangle_three_points(pts[0], pts[1], cursor, plane)
            return sample_rectangle_points(first, cursor, plane, mode="Center" if variant == "Center" else "Corner")
        if tool == "Circle":
            if variant == "3 point":
                if len(pts) == 1:
                    return np.vstack((first, cursor))
                return sample_circle_three_points(pts[0], pts[1], cursor, plane)
            return sample_circle_points(first, cursor, plane)
        if tool == "Curve":
            if len(pts) == 1:
                return np.vstack((first, cursor))
            if variant == "Center arc":
                return sample_center_arc(pts[0], pts[1], cursor, plane)
            return sample_three_point_arc(pts[0], pts[1], cursor, plane)
        if tool == "Slot":
            if len(pts) == 1:
                return np.vstack((first, cursor))
            u1, u2, uc = (np.asarray(plane.world_to_uv(p)) for p in (pts[0], pts[1], cursor))
            axis = u2 - u1
            length = float(np.linalg.norm(axis))
            if length < 1e-9:
                return None
            half = abs(float(np.dot(uc - u1, np.array([-axis[1], axis[0]]) / length)))
            return sample_slot(pts[0], pts[1], max(2.0 * half, 1e-3), plane)[1]
        if tool == "Polygon":
            return sample_polygon(first, cursor, _polygon_sides(variant), plane)[1]
        if tool == "Spline":
            ctrl = np.asarray(pts + [cursor], dtype=float)
            return sample_catmull_rom_spline(ctrl) if len(ctrl) >= 3 else ctrl
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    return None


def _fmt(value_mm: float, unit: str) -> str:
    if unit in ("in", "inch", "inches"):
        return f"{value_mm / 25.4:.4f} in"
    return f"{value_mm:.3f} mm"


def cursor_readout(tool: str, variant: str, points: list, cursor, plane: SketchPlane, unit: str = "mm") -> str:
    """Text shown next to the mouse: length and angle, width x height, or radius."""
    if cursor is None or plane is None or not points:
        return ""
    cursor = np.asarray(cursor, dtype=float)
    last = np.asarray(points[-1], dtype=float)
    first = np.asarray(points[0], dtype=float)
    ua, va = plane.world_to_uv(last)
    ub, vb = plane.world_to_uv(cursor)
    if tool in ("Line", "Construction line", "Centerline") or (tool in ("Slot", "Curve") and len(points) == 1):
        length = math.hypot(ub - ua, vb - va)
        angle = math.degrees(math.atan2(vb - va, ub - ua))
        return f"{_fmt(length, unit)} · {angle:.1f}°"
    if tool == "Rectangle" and variant != "3 point":
        uf, vf = plane.world_to_uv(first)
        scale = 2.0 if variant == "Center" else 1.0
        return f"{_fmt(abs(ub - uf) * scale, unit)} × {_fmt(abs(vb - vf) * scale, unit)}"
    if tool in ("Circle", "Polygon") and variant != "3 point":
        uf, vf = plane.world_to_uv(first)
        return f"R {_fmt(math.hypot(ub - uf, vb - vf), unit)}"
    if tool == "Slot" and len(points) >= 2:
        u1, u2 = np.asarray(plane.world_to_uv(points[0])), np.asarray(plane.world_to_uv(points[1]))
        axis = u2 - u1
        length = float(np.linalg.norm(axis)) or 1.0
        half = abs(float(np.dot(np.array([ub, vb]) - u1, np.array([-axis[1], axis[0]]) / length)))
        return f"width {_fmt(2 * half, unit)}"
    return ""


# ---------------------------------------------------------------- relation inference
SNAP_RELATION = {
    "Coincident": "coincident",
    "Center": "coincident",
    "Midpoint": "midpoint",
    "OnCurve": "on_curve",
    "ModelVertex": "anchor_model",
    "ModelEdge": "on_edge",
    "Edge": "anchor_model",
}


def relation_for_snap(snap_kind: str | None) -> str | None:
    """Which relation a click on this snap creates (None: no relation)."""
    return SNAP_RELATION.get(snap_kind or "")


def hv_relation(p0, p1, plane: SketchPlane, limit_deg: float = HV_INFERENCE_DEG) -> str | None:
    """"horizontal" / "vertical" for a line within ``limit_deg`` of the plane axes."""
    u0, v0 = plane.world_to_uv(np.asarray(p0, dtype=float))
    u1, v1 = plane.world_to_uv(np.asarray(p1, dtype=float))
    du, dv = u1 - u0, v1 - v0
    if math.hypot(du, dv) < 1e-9:
        return None
    angle = abs(math.degrees(math.atan2(dv, du))) % 180.0
    if min(angle, 180.0 - angle) <= limit_deg:
        return "horizontal"
    if abs(angle - 90.0) <= limit_deg:
        return "vertical"
    return None


# ---------------------------------------------------------------- chaining
@dataclass
class ChainState:
    """A running line chain: each click continues from the last end point."""

    start_point: np.ndarray | None = None
    first_entity: str = ""
    last_entity: str = ""
    entity_ids: list[str] = field(default_factory=list)
    tangent_next: bool = False  # "A": the next segment is a tangent arc

    @property
    def active(self) -> bool:
        return bool(self.last_entity)

    def add(self, entity_id: str, start_point) -> None:
        if not self.entity_ids:
            self.first_entity = entity_id
            self.start_point = np.asarray(start_point, dtype=float)
        self.entity_ids.append(entity_id)
        self.last_entity = entity_id
        self.tangent_next = False

    def closes_at(self, point, tolerance: float) -> bool:
        return (
            self.start_point is not None
            and len(self.entity_ids) >= 2
            and float(np.linalg.norm(np.asarray(point, dtype=float) - self.start_point)) <= max(tolerance, 1e-6)
        )

    def reset(self) -> None:
        self.start_point = None
        self.first_entity = ""
        self.last_entity = ""
        self.entity_ids = []
        self.tangent_next = False


def tangent_arc_points(start, direction, end, plane: SketchPlane, segments: int = 24) -> np.ndarray | None:
    """Arc from ``start`` leaving along ``direction`` (world vector) and ending at ``end``."""
    su, sv = plane.world_to_uv(np.asarray(start, dtype=float))
    eu, ev = plane.world_to_uv(np.asarray(end, dtype=float))
    d = np.asarray(direction, dtype=float)
    tu = float(np.dot(d, plane.u_axis))
    tv = float(np.dot(d, plane.v_axis))
    norm = math.hypot(tu, tv)
    if norm < 1e-12:
        return None
    tu, tv = tu / norm, tv / norm
    nu, nv = -tv, tu  # left normal
    chord_u, chord_v = eu - su, ev - sv
    side = chord_u * nu + chord_v * nv
    if abs(side) < 1e-9:
        return np.asarray([start, end], dtype=float)  # straight ahead: a line
    radius = (chord_u ** 2 + chord_v ** 2) / (2.0 * side)  # signed: + centre on the left
    cu, cv = su + nu * radius, sv + nv * radius
    a0 = math.atan2(sv - cv, su - cu)
    a1 = math.atan2(ev - cv, eu - cu)
    ccw = radius > 0
    sweep = (a1 - a0) % (2 * math.pi) if ccw else -((a0 - a1) % (2 * math.pi))
    r = abs(radius)
    angles = a0 + np.linspace(0.0, sweep, segments + 1)
    return np.asarray([plane.uv_to_world(cu + r * math.cos(a), cv + r * math.sin(a)) for a in angles], dtype=float)


# ---------------------------------------------------------------- typed values while drawing
_UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, '"': 25.4, "ft": 304.8, "'": 304.8}


def parse_typed_length(text: str, default_unit: str = "mm") -> float | None:
    """'25', '25.4mm', '1in', '1/2"', '1 1/2 in' -> millimetres (None if not a length)."""
    text = (text or "").strip().lower()
    if not text:
        return None
    match = re.fullmatch(r"(?:(\d+)\s+)?(\d*\.?\d+)(?:\s*/\s*(\d*\.?\d+))?\s*(mm|cm|m|in|ft|\"|')?", text)
    if not match:
        return None
    whole, number, denominator, unit = match.groups()
    value = float(number)
    if denominator:
        if float(denominator) == 0:
            return None
        value = value / float(denominator)
    if whole:
        value += float(whole)
    unit = unit or ("in" if default_unit in ("in", "inch", "inches") else "mm")
    return value * _UNITS[unit]


__all__ = [
    "ChainState", "HV_INFERENCE_DEG", "cursor_readout", "hv_relation", "parse_typed_length",
    "preview_geometry", "relation_for_snap", "tangent_arc_points",
]
