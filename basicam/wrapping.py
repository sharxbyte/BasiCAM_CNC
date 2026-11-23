"""Rotary work (no Qt): sketches wrapped onto a cylinder and indexed (3+1) setups.

Wrapped: a sketch drawn on the unwrapped cylinder (X along the rotary axis, Y = arc length on the
surface) is wrapped onto a cylinder of the given radius: Y becomes the rotary angle
(degrees = arc / radius x 180 / pi) and Z the radial position (radius minus the cut depth),
with the tool over the axis. Rotary and linear axes move together; each move carries its
surface distance so the post can give it an inverse-time (G93) feed at the intended surface
speed. Rotary lasers use the same wrap with the beam at the top centre.

Indexed: the part is turned by a fixed angle for the whole setup; ordinary 3-axis passes
computed in the part's frame are turned into machine coordinates, and a guarded index move
starts the setup.
"""
from __future__ import annotations

import math

import numpy as np


def arc_to_degrees(arc_mm: float, radius_mm: float) -> float:
    return math.degrees(arc_mm / max(radius_mm, 1e-9))


def wrap_segments(segments, radius_mm: float, axis_letter: str = "A", centre_yz: tuple[float, float] = (0.0, 0.0),
                  top_z: float | None = None, step_mm: float = 1.0) -> list:
    """Unwrapped segments (X along the axis, Y = arc length, Z = height above the surface ->
    0 at the surface, negative = cut depth) -> rotary segments with metadata["rotary"]."""
    from basicam.cam_dynamics_post import ToolpathSegment

    cy, cz = centre_yz
    surface = (cz + radius_mm) if top_z is None else float(top_z)
    out = []
    for seg in segments:
        a, b = np.asarray(seg.start, dtype=float), np.asarray(seg.end, dtype=float)
        arc = abs(float(b[1] - a[1]))
        pieces = 1 if seg.kind in ("rapid", "retract") else max(1, int(math.ceil(max(arc, abs(b[0] - a[0])) / max(step_mm, 1e-3))))
        prev = None
        for k in range(pieces + 1):
            p = a + (b - a) * (k / pieces)
            angle = arc_to_degrees(float(p[1]), radius_mm)
            point = (float(p[0]), float(cy), float(surface + p[2]))
            if prev is not None:
                surface_mm = abs(float(p[1]) - prev[1]) * (radius_mm + min(float(p[2]), 0.0)) / max(radius_mm, 1e-9)
                meta = dict(seg.metadata or {}, rotary={axis_letter: angle}, surface_mm=surface_mm)
                out.append(ToolpathSegment(seg.kind, prev[0], point, seg.feed_mm_min, seg.rpm, meta))
            prev = (point, float(p[1]))
    return out


def index_segments(segments, angle_deg: float, setup_kinematics, axis_letter: str = "A", safe_z: float = 10.0) -> list:
    """Indexed (3+1) setup: passes in the part frame -> machine coordinates at a fixed rotary
    angle, preceded by the index move at safe Z."""
    from basicam.cam_dynamics_post import ToolpathSegment
    from basicam.kinematics import inverse

    def machine(point):
        joints = inverse(setup_kinematics, np.asarray(point, dtype=float), _axis_for(angle_deg, axis_letter), {axis_letter: angle_deg})
        return (joints.get("X", 0.0), joints.get("Y", 0.0), joints.get("Z", 0.0))

    out = []
    if segments:
        first = machine(segments[0].start)
        out.append(ToolpathSegment("rapid", (first[0], first[1], safe_z), (first[0], first[1], safe_z), 0.0, 0,
                                   {**(segments[0].metadata or {}), "rotary": {axis_letter: float(angle_deg)}, "index": True}))
    for seg in segments:
        meta = dict(seg.metadata or {}, rotary={axis_letter: float(angle_deg)})
        out.append(ToolpathSegment(seg.kind, machine(seg.start), machine(seg.end), seg.feed_mm_min, seg.rpm, meta))
    return out


def _axis_for(angle_deg: float, axis_letter: str) -> np.ndarray:
    a = math.radians(angle_deg)
    if axis_letter == "B":
        return np.array([-math.sin(a), 0.0, math.cos(a)])
    return np.array([0.0, math.sin(a), math.cos(a)])


__all__ = ["arc_to_degrees", "index_segments", "wrap_segments"]
