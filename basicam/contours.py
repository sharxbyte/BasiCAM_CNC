"""Contours and regions of a sketch as toolpath targets (sketch-engine.spec B11), no Qt.

Closed loops of non-construction entities become regions (nested loops = islands, shapely
polygonize), open chains are trace targets, points and circle centres are drill targets.
check_sketch() finds what blocks a tool assignment: zero-length segments, gaps, self
intersections and overlapping entities. Tool-centre paths for pocket rings, profiles with
tool-radius compensation (computed here, no G41/G42) and drill points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union

from basicam.sketch_solver import plane_for, reference_points

GAP_TOLERANCE_MM = 0.01
GAP_SEARCH_MM = 1.0  # an end this close to another (but not on it) is a gap, not an open end


@dataclass
class SketchProblem:
    message: str
    location: np.ndarray  # world
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class SketchContours:
    plane: object
    regions: list[Polygon] = field(default_factory=list)  # uv, with islands as holes
    chains: list[LineString] = field(default_factory=list)  # open chains (trace)
    points: list[tuple[np.ndarray, str]] = field(default_factory=list)  # drill points (world, entity)
    region_entities: list[set[str]] = field(default_factory=list)

    def to_world(self, uv_points) -> list[tuple[float, float, float]]:
        return [tuple(map(float, self.plane.uv_to_world(float(u), float(v)))) for u, v in uv_points]


def _polyline_uv(feat, plane) -> list[tuple[float, float]]:
    pts = np.asarray(feat.active_points if getattr(feat, "is_projected", False) else feat.points, dtype=float)
    uv = [tuple(map(float, plane.world_to_uv(p))) for p in pts]
    if feat.kind in ("Rectangle", "Polygon", "Slot", "Circle") and uv and uv[0] != uv[-1]:
        uv.append(uv[0])
    return uv


def _cuttable(features):
    return [f for f in features if not f.construction and f.kind != "Point" and len(f.points) >= 2]


def build_contours(features: list) -> SketchContours:
    plane = plane_for(features) if features else None
    result = SketchContours(plane)
    if plane is None:
        return result
    lines, owners = [], []
    for feat in _cuttable(features):
        uv = _polyline_uv(feat, plane)
        if len(uv) >= 2:
            lines.append(LineString(uv))
            owners.append(feat.feature_id)
    for feat in features:
        if feat.construction:
            continue
        if feat.kind == "Point":
            result.points.append((np.asarray(feat.points[0], dtype=float), feat.feature_id))
        elif feat.kind == "Circle":
            result.points.append((np.asarray(reference_points(feat)[0], dtype=float), feat.feature_id))
    if not lines:
        return result
    noded = unary_union(lines)
    faces = list(polygonize(noded))
    result.regions = sorted(faces, key=lambda p: -p.area)
    for face in result.regions:
        outline = face.exterior  # a region belongs to the entities of its outer boundary
        result.region_entities.append({o for o, line in zip(owners, lines) if line.intersection(outline.buffer(1e-6)).length > 1e-6})
    # open chains: linework not on any face boundary
    used = unary_union([f.boundary for f in faces]) if faces else None
    rest = noded if used is None else noded.difference(used.buffer(1e-7))
    merged = linemerge(rest) if not rest.is_empty else rest
    if not merged.is_empty:
        geoms = list(merged.geoms) if isinstance(merged, MultiLineString) else [merged]
        result.chains = [g for g in geoms if isinstance(g, LineString) and g.length > 1e-6]
    return result


def check_sketch(features: list, gap_tolerance: float = GAP_TOLERANCE_MM) -> list[SketchProblem]:
    """Problems that block tool assignment, each with a location (spec B11)."""
    problems: list[SketchProblem] = []
    cuttable = _cuttable(features)
    if not cuttable:
        return problems
    plane = plane_for(features)

    def world(uv):
        return np.asarray(plane.uv_to_world(float(uv[0]), float(uv[1])), dtype=float)

    curves = {}
    for feat in cuttable:
        uv = _polyline_uv(feat, plane)
        for a, b in zip(uv[:-1], uv[1:]):
            if math.dist(a, b) < 1e-9 and feat.kind in ("Line", "Rectangle", "Polygon"):
                problems.append(SketchProblem(f"{feat.name}: zero-length segment", world(a), [feat.feature_id]))
                break
        line = LineString(uv)
        if not line.is_simple and feat.kind not in ("Circle",):
            problems.append(SketchProblem(f"{feat.name} crosses itself", world(line.coords[0]), [feat.feature_id]))
        curves[feat.feature_id] = (feat, line)
    ends = []
    for fid, (feat, line) in curves.items():
        if line.is_ring or math.dist(line.coords[0], line.coords[-1]) < 1e-9:
            continue
        ends.append((fid, line.coords[0]))
        ends.append((fid, line.coords[-1]))
    for i, (fid, p) in enumerate(ends):
        others = [math.dist(p, q) for j, (gid, q) in enumerate(ends) if j != i]
        on_curve = [curves[g][1].distance(Point(p)) for g in curves if g != fid]
        nearest = min(others) if others else math.inf
        nearest_curve = min(on_curve) if on_curve else math.inf
        if nearest <= 1e-6 or nearest_curve <= 1e-6:
            continue
        gap = min(nearest, nearest_curve)
        if gap > gap_tolerance and gap <= GAP_SEARCH_MM:
            problems.append(SketchProblem(
                f"{curves[fid][0].name}: gap of {gap:.3f} mm at its end", world(p), [fid]
            ))
    ids = list(curves)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = curves[ids[i]][1], curves[ids[j]][1]
            shared = a.intersection(b)
            if not shared.is_empty and shared.length > gap_tolerance:
                where = shared.representative_point()
                problems.append(SketchProblem(
                    f"{curves[ids[i]][0].name} and {curves[ids[j]][0].name} overlap",
                    world((where.x, where.y)), [ids[i], ids[j]],
                ))
    return problems


# ---------------------------------------------------------------- tool-centre paths (uv)
def _rings(shape) -> list[list[tuple[float, float]]]:
    polys = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
    rings = []
    for poly in polys:
        if poly.is_empty or not isinstance(poly, Polygon):
            continue
        rings.append(list(poly.exterior.coords))
        rings.extend(list(interior.coords) for interior in poly.interiors)
    return rings


def pocket_rings(region: Polygon, tool_radius: float, stepover: float) -> list[list[tuple[float, float]]]:
    """Offset rings that clear a region (islands kept clear by the tool radius), inside out."""
    stepover = max(1e-3, float(stepover))
    rings: list[list[tuple[float, float]]] = []
    current = region.buffer(-tool_radius, join_style=1)
    while not current.is_empty and current.area > 1e-9:
        rings.extend(_rings(current))
        current = current.buffer(-stepover, join_style=1)
    return rings[::-1]


def profile_path(region: Polygon, side: str, tool_radius: float) -> list[list[tuple[float, float]]]:
    """Outer contour of a region, compensated: Outside (+r), Inside (-r) or On."""
    side = (side or "On").casefold()
    if side.startswith("out"):
        shape = Polygon(region.exterior).buffer(tool_radius, join_style=1)
    elif side.startswith("in"):
        shape = Polygon(region.exterior).buffer(-tool_radius, join_style=1)
    else:
        shape = Polygon(region.exterior)
    return [list(p.exterior.coords) for p in (shape.geoms if hasattr(shape, "geoms") else [shape]) if not p.is_empty]


def region_containing(contours: SketchContours, entity_ids: set[str]) -> Polygon | None:
    """The region an operation targets: the largest one bounded by any of these entities."""
    for region, owners in zip(contours.regions, contours.region_entities):
        if owners & entity_ids:
            return region
    return None


__all__ = [
    "GAP_TOLERANCE_MM", "SketchContours", "SketchProblem", "build_contours", "check_sketch",
    "pocket_rings", "profile_path", "region_containing",
]
