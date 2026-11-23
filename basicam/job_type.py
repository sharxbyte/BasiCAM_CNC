"""Job type: flat, round or on a 3D object (no Qt in the logic).

Asked when a 2D machine (laser, plasma, waterjet, wire EDM) becomes active for a workspace, or
when an operation is set to Engrave / Etch on any machine. The answer is stored per setup:
  flat     load or create a sheet / plate stock; 2D work on its top face
  round    use the machine's rotary device (open its setup when there is none); bar or tube stock
  surface  sketches are projected onto a selected surface; the head follows it (a laser keeps
           its focus offset, a V-bit keeps its engraving depth)
"""
from __future__ import annotations

import math

import numpy as np

from basicam.machine_types import TWO_D_PROCESSES

WORK_TYPES = {"flat": "Flat (sheet or plate)", "round": "Round (rotary axis)", "surface": "On a 3D object"}
ENGRAVE_KINDS = ("Engrave", "Etch")


def needs_prompt(process: str, op_kind: str | None, current: str) -> bool:
    """Ask when the work type is unknown and a 2D process or an engraving needs it."""
    if current in WORK_TYPES:
        return False
    return process in TWO_D_PROCESSES or (op_kind or "") in ENGRAVE_KINDS


def missing_for(work_type: str, has_sheet: bool, has_rotary: bool, has_surface: bool) -> list[str]:
    """What a work type still needs before toolpaths can be made."""
    if work_type == "flat" and not has_sheet:
        return ["sheet"]
    if work_type == "round" and not has_rotary:
        return ["rotary"]
    if work_type == "surface" and not has_surface:
        return ["surface"]
    return []


def follow_surface(segments, height_at, offset: float, step: float = 1.0, along_normal=None) -> list:
    """Make 2D engrave segments follow a surface: each move is split into ``step`` pieces and its
    Z set to the surface height + ``offset`` (laser focus offset, or minus a V-bit's depth).
    ``height_at(x, y)`` returns the surface height (or None off the surface: Z unchanged)."""
    from basicam.cam_dynamics_post import ToolpathSegment

    out = []
    for seg in segments:
        if seg.kind not in ("cut", "lead_in", "lead_out", "plunge", "ramp"):
            out.append(seg)
            continue
        a, b = np.asarray(seg.start, dtype=float), np.asarray(seg.end, dtype=float)
        length = float(np.linalg.norm(b[:2] - a[:2]))
        pieces = max(1, int(math.ceil(length / max(step, 1e-3))))
        previous = None
        for k in range(pieces + 1):
            p = a + (b - a) * (k / pieces)
            h = height_at(float(p[0]), float(p[1]))
            z = float(h) + offset if h is not None and math.isfinite(h) else float(p[2])
            point = (float(p[0]), float(p[1]), z)
            if previous is not None:
                out.append(ToolpathSegment(seg.kind, previous, point, seg.feed_mm_min, seg.rpm, dict(seg.metadata or {}, surface=True)))
            previous = point
    return out


__all__ = ["ENGRAVE_KINDS", "WORK_TYPES", "follow_surface", "missing_for", "needs_prompt"]
