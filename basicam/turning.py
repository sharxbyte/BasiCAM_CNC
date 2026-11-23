"""Turning toolpaths and the lathe post (shapely, no Qt).

Lathe coordinates: r = radius from the spindle axis, z = along the axis from the chuck / collet
face (positive away from the chuck). Segments carry (r, 0, z) and metadata["lathe"] so the lathe
body checks (machine_bodies) and the lathe post can use them. The finished part is a half
profile [(z, r), ...] (for example from Convert part silhouette).

Operations: facing, rough turning (OD, step-down per side, finish allowance - never cut inside
it), finish turning with nose-radius compensation computed here (the nose centre runs one nose
radius off the profile; no G41/G42), grooving, parting, centre drilling / drilling on the axis,
threading (only when the controller supports G76).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box

CLEAR_MM = 1.0  # air gap before engaging


def profile_polygon(profile: list[tuple[float, float]]) -> Polygon:
    """Region the finished part occupies in the (z, r) half-plane (closed down to the axis)."""
    pts = list(profile)
    if pts[0][1] != 0.0:
        pts.insert(0, (pts[0][0], 0.0))
    if pts[-1][1] != 0.0:
        pts.append((pts[-1][0], 0.0))
    return Polygon(pts).buffer(0)


def _segment(kind, a, b, feed, rpm, meta):
    from basicam.cam_dynamics_post import ToolpathSegment

    return ToolpathSegment(kind, (float(a[1]), 0.0, float(a[0])), (float(b[1]), 0.0, float(b[0])), float(feed), int(rpm), dict(meta, lathe=True))


def facing(stock_r: float, z_end: float, z_face: float, step: float, feed: float, rpm: int, meta: dict) -> list:
    """Face from the stock end down to z_face, passes from outside the stock to the centre."""
    out = []
    z = z_end
    while z > z_face + 1e-9:
        z = max(z_face, z - step)
        out.append(_segment("rapid", (z + CLEAR_MM, stock_r + CLEAR_MM), (z, stock_r + CLEAR_MM), 0, rpm, meta))
        out.append(_segment("cut", (z, stock_r + CLEAR_MM), (z, 0.0), feed, rpm, meta))
        out.append(_segment("retract", (z, 0.0), (z + CLEAR_MM, CLEAR_MM), 0, rpm, meta))
        out.append(_segment("rapid", (z + CLEAR_MM, CLEAR_MM), (z + CLEAR_MM, stock_r + CLEAR_MM), 0, rpm, meta))
    return out


def rough_turn(profile, stock_r: float, z_end: float, step_per_side: float, allowance: float, feed: float, rpm: int, meta: dict) -> list:
    """OD roughing: passes along z at falling radii, stopping at the profile + finish allowance."""
    part = profile_polygon(profile)
    keep = part.buffer(allowance, join_style=2)  # never cut inside the finish allowance
    z_min = min(z for z, _r in profile)
    material = box(z_min, 0.0, z_end, stock_r).difference(keep)
    out = []
    r = stock_r
    while r > 0.0:
        r = max(0.0, r - step_per_side)
        line = LineString([(z_end + CLEAR_MM, r), (z_min - 1e-6, r)])
        cut = line.intersection(material)
        pieces = list(cut.geoms) if isinstance(cut, MultiLineString) else ([cut] if isinstance(cut, LineString) and not cut.is_empty else [])
        for piece in sorted(pieces, key=lambda g: -max(c[0] for c in g.coords)):
            coords = sorted(piece.coords, key=lambda c: -c[0])  # towards the chuck
            start, end = coords[0], coords[-1]
            if math.dist(start, end) < 1e-6:
                continue
            out.append(_segment("rapid", (start[0] + CLEAR_MM, r + CLEAR_MM), (start[0] + CLEAR_MM, r), 0, rpm, meta))
            out.append(_segment("cut", (start[0] + CLEAR_MM, r), (end[0], r), feed, rpm, meta))
            out.append(_segment("retract", (end[0], r), (end[0] + CLEAR_MM * 0.5, r + CLEAR_MM), 0, rpm, meta))
        if r <= 0.0 or not material.intersects(LineString([(z_min, r), (z_end, r)])):
            if r < 0.5 * step_per_side or material.area < 1e-9:
                break
    return out


def finish_path(profile, nose_radius: float) -> list[tuple[float, float]]:
    """Nose-centre path for finishing: the profile offset outwards by the nose radius (so the
    insert's round nose touches the profile everywhere), from the free end to the chuck side."""
    part = profile_polygon(profile)
    grown = part.buffer(max(nose_radius, 1e-6), join_style=1, resolution=32) if nose_radius > 0 else part
    outline = LineString(list(grown.exterior.coords))
    z_min, _rmin, z_max, _rmax = part.bounds
    # the outer part of the outline: points above the axis band, walked from z_max to z_min
    pts = [c for c in outline.coords if c[1] > nose_radius * 0.5 + 1e-6 and z_min - 1e-6 <= c[0] <= z_max + nose_radius + 1e-6]
    pts.sort(key=lambda c: -c[0])
    cleaned = []
    for p in pts:
        if not cleaned or math.dist(p, cleaned[-1]) > 1e-6:
            cleaned.append(p)
    return cleaned


def finish_turn(profile, nose_radius: float, feed: float, rpm: int, meta: dict) -> list:
    path = finish_path(profile, nose_radius)
    out = []
    if len(path) < 2:
        return out
    out.append(_segment("rapid", (path[0][0] + CLEAR_MM, path[0][1] + CLEAR_MM), (path[0][0] + CLEAR_MM, path[0][1]), 0, rpm, meta))
    previous = (path[0][0] + CLEAR_MM, path[0][1])
    for p in path:
        out.append(_segment("cut", previous, p, feed, rpm, dict(meta, nose_centre=True, nose_radius=nose_radius)))
        previous = p
    out.append(_segment("retract", previous, (previous[0], previous[1] + CLEAR_MM), 0, rpm, meta))
    return out


def groove(z_centre: float, stock_r: float, bottom_r: float, width: float, tool_width: float, feed: float, rpm: int, meta: dict) -> list:
    """Plunge passes across the groove width (tool width steps), to the bottom radius."""
    out = []
    left = z_centre - width / 2.0 + tool_width / 2.0
    right = z_centre + width / 2.0 - tool_width / 2.0
    count = max(1, int(math.ceil((right - left) / max(tool_width * 0.8, 1e-3))) + 1)
    for k in range(count):
        z = left + (right - left) * (k / (count - 1) if count > 1 else 0.5)
        out.append(_segment("rapid", (z, stock_r + CLEAR_MM), (z, stock_r + CLEAR_MM), 0, rpm, meta))
        out.append(_segment("cut", (z, stock_r + CLEAR_MM), (z, bottom_r), feed, rpm, meta))
        out.append(_segment("retract", (z, bottom_r), (z, stock_r + CLEAR_MM), 0, rpm, meta))
    return out


def part_off(z: float, stock_r: float, bore_r: float, feed: float, rpm: int, meta: dict) -> list:
    return groove(z, stock_r, bore_r, 0.0, 0.0, feed, rpm, meta)


def drill_on_axis(z_start: float, depth: float, peck: float, feed: float, rpm: int, meta: dict) -> list:
    out = []
    reached = 0.0
    while reached < depth - 1e-9:
        target = min(depth, reached + peck) if peck > 0 else depth
        out.append(_segment("rapid", (z_start + CLEAR_MM, 0.0), (z_start - reached + 0.5 if reached else z_start + CLEAR_MM, 0.0), 0, rpm, meta))
        out.append(_segment("cut", (z_start - reached + (0.5 if reached else CLEAR_MM), 0.0), (z_start - target, 0.0), feed, rpm, meta))
        out.append(_segment("retract", (z_start - target, 0.0), (z_start + CLEAR_MM, 0.0), 0, rpm, meta))
        reached = target
    return out


# ---------------------------------------------------------------- lathe post
def compile_lathe(segments, *, diameter_mode: bool = True, z_zero: float = 0.0, css_m_min: float = 0.0, max_rpm: float = 3000.0,
                  feed_per_rev: float = 0.0, controller_threading: bool = False, manual_speed: bool = False,
                  threads: list[dict] | None = None, controller: str = "grblHAL") -> str:
    """G-code for a lathe: G18, diameter (G7) or radius (G8) mode, G96 constant surface speed
    with a max RPM, G95 feed per revolution where given (otherwise G94 mm/min), X doubled in
    diameter mode, Z measured from ``z_zero`` (the part's front face). A manual-speed spindle
    pauses with a prompt instead of an S word."""
    lines = [f"; Lathe post ({controller})", "G21", "G90", "G18", "G7" if diameter_mode else "G8"]
    rpm_now = None
    for seg in segments:
        rpm = int(seg.rpm or 0)
        if rpm and rpm != rpm_now:
            if manual_speed:
                lines.append(f"M0 (Set the spindle to {rpm} RPM, then cycle start)")
            elif css_m_min > 0:
                lines.append(f"G96 D{max_rpm:.0f} S{css_m_min:.0f} M3")
            else:
                lines.append(f"G97 S{rpm} M3")
            rpm_now = rpm
        r, _y, z = seg.end
        x = 2.0 * r if diameter_mode else r
        if seg.kind in ("rapid", "retract"):
            lines.append(f"G0 X{x:.4f} Z{z - z_zero:.4f}")
        else:
            if feed_per_rev > 0:
                lines.append(f"G95 G1 X{x:.4f} Z{z - z_zero:.4f} F{feed_per_rev:.4f}")
            else:
                lines.append(f"G94 G1 X{x:.4f} Z{z - z_zero:.4f} F{float(seg.feed_mm_min):.1f}")
    for thread in threads or []:
        if not controller_threading:
            raise ValueError("This controller cannot cut threads (G76).")
        lines.append(
            f"G76 P{thread['pitch']:.4f} Z{thread['z_end'] - z_zero:.4f} I{-thread.get('peak_offset', 0.5):.4f} "
            f"J{thread.get('first_cut', 0.1):.4f} K{thread['depth']:.4f}"
        )
    lines += ["M5", "M30"]
    return "\n".join(lines) + "\n"


__all__ = [
    "compile_lathe", "drill_on_axis", "facing", "finish_path", "finish_turn", "groove", "part_off", "profile_polygon", "rough_turn",
]
