"""Lathe tools (no Qt): insert shapes, nose radius, holders, grooving / parting width,
threading pitch range, boring bar minimum bore and turret station.

Lathe fields live on ToolDefinition (tolerant loading: old tools load as milling tools);
``lathe_kind`` is empty for milling tools.
"""
from __future__ import annotations

INSERT_SHAPES = {  # ISO insert shape letter -> included angle (degrees)
    "C": 80.0, "D": 55.0, "V": 35.0, "W": 80.0, "T": 60.0, "R": 0.0, "S": 90.0,
}
LATHE_KINDS = ("turning", "boring", "grooving", "parting", "threading", "drill")


def is_lathe_tool(tool) -> bool:
    return str(getattr(tool, "lathe_kind", "") or "") in LATHE_KINDS


def nose_radius(tool) -> float:
    return max(0.0, float(getattr(tool, "nose_radius_mm", 0.0) or 0.0))


def insert_angle(tool) -> float:
    return INSERT_SHAPES.get(str(getattr(tool, "insert_shape", "C") or "C").upper(), 80.0)


def can_thread(tool, pitch_mm: float) -> bool:
    lo = float(getattr(tool, "thread_pitch_min_mm", 0.0) or 0.0)
    hi = float(getattr(tool, "thread_pitch_max_mm", 0.0) or 0.0)
    return str(getattr(tool, "lathe_kind", "")) == "threading" and (hi <= 0 or lo - 1e-9 <= pitch_mm <= hi + 1e-9)


def fits_bore(tool, bore_diameter_mm: float) -> bool:
    minimum = float(getattr(tool, "min_bore_mm", 0.0) or 0.0)
    return str(getattr(tool, "lathe_kind", "")) != "boring" or bore_diameter_mm + 1e-9 >= minimum


__all__ = ["INSERT_SHAPES", "LATHE_KINDS", "can_thread", "fits_bore", "insert_angle", "is_lathe_tool", "nose_radius"]
