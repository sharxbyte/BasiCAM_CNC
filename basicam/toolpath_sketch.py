"""Toolpath sketches generated from a surface (sketch-engine.spec B10), no Qt.

"Assign tool..." on a face or manifold surface: the chosen tool's diameter and tip shape give a
tool envelope OUTSIDE the surface (surfacing.tool_envelope). Pass points are spaced at the
stepover (default 60% of the tool diameter, a 40% overlap), rounded away from the material to
0.001 mm (or 0.0001 in), simplified without ever dropping below the envelope, ordered for the
shortest rapids (zig-zag, nearest neighbour, 2-opt), as roughing layers from the stock top down
to the stock-to-leave, then 3-axis finishing along the surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from basicam.surfacing import SurfaceParams, _env_at, height_map, plan_surface, tool_envelope

ROUND_MM = 0.001
ROUND_IN_MM = 0.0001 * 25.4
SIMPLIFY_MM = 0.01


def tool_shape_of(tool) -> tuple[str, float, float]:
    """(envelope shape, corner radius, included angle) for an inventory tool."""
    kind = str(getattr(tool, "tool_type", "End mill"))
    corner = float(getattr(tool, "corner_radius_mm", 0.0) or 0.0)
    angle = float(getattr(tool, "included_angle_deg", 90.0) or 90.0)
    if kind == "Ball end mill":
        return "Ball", 0.0, angle
    if kind in ("V-bit", "Chamfer mill", "Countersink"):
        return "V-bit", 0.0, angle
    if corner > 0:
        return "Bull", corner, angle
    return "Flat", 0.0, angle


@dataclass
class ToolpathPass:
    points: np.ndarray  # (N, 3) world, tool tip
    kind: str  # "rough" or "finish"
    layer: int = 0


@dataclass
class ToolpathSketchPlan:
    passes: list[ToolpathPass] = field(default_factory=list)
    stepover_mm: float = 0.0
    rounding_mm: float = ROUND_MM
    raw_rapid_mm: float = 0.0
    ordered_rapid_mm: float = 0.0
    envelope_check: object = None  # (hmap, env) for validation

    def kinds(self) -> list[str]:
        return sorted({p.kind for p in self.passes})


def round_away(points: np.ndarray, env, hmap, grid: float) -> np.ndarray:
    """Round x/y to the grid, then z UP to the grid and never below the envelope at the
    rounded position (rounding can never gouge)."""
    out = np.asarray(points, dtype=float).copy()
    out[:, 0] = np.round(out[:, 0] / grid) * grid
    out[:, 1] = np.round(out[:, 1] / grid) * grid
    for i, (x, y, z) in enumerate(out):
        e = _env_at(env, hmap, x, y) if env is not None else -math.inf
        z = max(z, e if math.isfinite(e) else z)
        out[i, 2] = math.ceil(z / grid - 1e-9) * grid
    return out


def simplify_above(points: np.ndarray, tolerance: float = SIMPLIFY_MM) -> np.ndarray:
    """Douglas-Peucker that only drops a point when the chord stays within ``tolerance`` and is
    not below the dropped point (the path never moves into the envelope)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[b] - pts[a]
        length2 = float(np.dot(seg, seg))
        inner = pts[a + 1:b]
        t = np.clip(((inner - pts[a]) @ seg) / length2, 0.0, 1.0) if length2 > 0 else np.zeros(len(inner))
        chord = pts[a] + t[:, None] * seg
        deviation = np.linalg.norm(inner - chord, axis=1)
        below = chord[:, 2] < inner[:, 2] - 1e-9
        worst = int(np.argmax(deviation + below * 1e6))
        if deviation[worst] > tolerance or below.any():
            index = a + 1 + worst
            keep[index] = True
            stack.append((a, index))
            stack.append((index, b))
    return pts[keep]


def _rapid_length(passes: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(passes[i + 1][0, :2] - passes[i][-1, :2]) for i in range(len(passes) - 1)))


def order_passes(passes: list[np.ndarray]) -> list[np.ndarray]:
    """Shortest rapids: nearest neighbour (a pass may be run reversed), then 2-opt."""
    if len(passes) <= 1:
        return list(passes)
    remaining = list(range(len(passes)))
    order: list[tuple[int, bool]] = [(remaining.pop(0), False)]
    while remaining:
        last_index, last_rev = order[-1]
        end = passes[last_index][0 if last_rev else -1][:2]
        best = None
        for index in remaining:
            for rev in (False, True):
                start = passes[index][-1 if rev else 0][:2]
                d = float(np.linalg.norm(start - end))
                if best is None or d < best[0]:
                    best = (d, index, rev)
        remaining.remove(best[1])
        order.append((best[1], best[2]))
    route = [passes[i][::-1] if rev else passes[i] for i, rev in order]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                before = np.linalg.norm(route[i][0, :2] - route[i - 1][-1, :2])
                after_now = np.linalg.norm(route[j + 1][0, :2] - route[j][-1, :2]) if j + 1 < len(route) else 0.0
                before_new = np.linalg.norm(route[j][-1, :2] - route[i - 1][-1, :2])
                after_new = np.linalg.norm(route[j + 1][0, :2] - route[i][0, :2]) if j + 1 < len(route) else 0.0
                if before_new + after_new < before + after_now - 1e-9:
                    route[i:j + 1] = [p[::-1] for p in route[i:j + 1][::-1]]
                    improved = True
    return route


def generate_toolpath_sketch(mesh, tool, params: SurfaceParams, stock_top_z: float | None = None, inch: bool = False) -> ToolpathSketchPlan:
    """Roughing layers + finishing passes for a surface and tool, ready to become sketch entities."""
    shape, corner, angle = tool_shape_of(tool)
    params.tool_shape = shape
    params.corner_radius_mm = corner
    params.tip_angle_deg = angle
    params.tool_diameter_mm = float(getattr(tool, "diameter_mm", params.tool_diameter_mm))
    grid = ROUND_IN_MM if inch else ROUND_MM
    hmap = height_map(mesh, params.resolution(), margin=params.tool_diameter_mm)
    env = tool_envelope(hmap, shape, params.tool_diameter_mm, corner, angle)
    plan = ToolpathSketchPlan(stepover_mm=params.stepover(), rounding_mm=grid, envelope_check=(hmap, env))
    import copy

    rough_params = copy.copy(params)
    rough_params.remove_bulk = True
    finish_params = copy.copy(params)
    finish_params.remove_bulk = False
    rough_passes = (
        [np.asarray(p, dtype=float) for p in plan_surface(mesh, rough_params, stock_top_z=stock_top_z, finish=False).passes]
        if params.remove_bulk else []
    )
    finish_passes = [np.asarray(p, dtype=float) for p in plan_surface(mesh, finish_params, stock_top_z=stock_top_z, finish=True).passes]
    layers: dict[float, list[np.ndarray]] = {}
    for p in rough_passes:
        layers.setdefault(round(float(np.min(p[:, 2])), 6), []).append(p)
    raw_total = ordered_total = 0.0
    for layer_index, z in enumerate(sorted(layers, reverse=True)):  # from the stock top down
        cleaned = [simplify_above(round_away(p, env, hmap, grid)) for p in layers[z]]
        raw_total += _rapid_length(cleaned)
        ordered = order_passes(cleaned)
        ordered_total += _rapid_length(ordered)
        plan.passes.extend(ToolpathPass(p, "rough", layer_index + 1) for p in ordered)
    cleaned = [simplify_above(round_away(p, env, hmap, grid)) for p in finish_passes]
    raw_total += _rapid_length(cleaned)
    ordered = order_passes(cleaned)
    ordered_total += _rapid_length(ordered)
    plan.passes.extend(ToolpathPass(p, "finish", 0) for p in ordered)
    plan.raw_rapid_mm, plan.ordered_rapid_mm = raw_total, ordered_total
    return plan


__all__ = [
    "ROUND_IN_MM", "ROUND_MM", "ToolpathPass", "ToolpathSketchPlan", "generate_toolpath_sketch",
    "order_passes", "round_away", "simplify_above", "tool_shape_of",
]
