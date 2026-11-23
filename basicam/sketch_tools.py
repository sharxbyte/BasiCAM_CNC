"""Sketch editing tools (sketch-engine.spec B9), no Qt: trim, extend, offset, mirror, fillet,
chamfer, convert entities, move / copy and fully define. Geometry works in the sketch plane's uv
coordinates (shapely for intersections and offsets); results are new entity point arrays plus
the relations / dimensions that keep them tied to their source."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from basicam.sketch_solver import Relation, SketchSolver, plane_for, reference_points
from basicam.sketch_workflow import SketchDimension, SketchPlane

LINE_KINDS = ("Line", "Construction line")
CLOSED_KINDS = ("Rectangle", "Polygon", "Slot", "Circle")


@dataclass
class NewEntity:
    kind: str
    points: np.ndarray  # world (N, 3)
    construction: bool = False
    variant: str = ""
    control_points: np.ndarray | None = None
    dimensions: list[SketchDimension] = field(default_factory=list)


@dataclass
class EditResult:
    """What an editing tool does: entities to add / replace / delete and relations to add.

    Relation refs may name new entities as "new:<index>" (index into ``add``)."""

    add: list[NewEntity] = field(default_factory=list)
    replace: dict[str, np.ndarray] = field(default_factory=dict)  # entity id -> new points
    delete: list[str] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    message: str = ""
    unlink: list[tuple[str, str]] = field(default_factory=list)  # drop coincident relations between these
    join_ends: bool = False  # add coincident relations where new and edited end points touch

    @property
    def empty(self) -> bool:
        return not (self.add or self.replace or self.delete or self.relations or self.unlink)


def _uv(plane: SketchPlane, p) -> np.ndarray:
    return np.asarray(plane.world_to_uv(np.asarray(p, dtype=float)), dtype=float)


def _world(plane: SketchPlane, uv) -> np.ndarray:
    return np.asarray(plane.uv_to_world(float(uv[0]), float(uv[1])), dtype=float)


def _curve_of(feat, plane: SketchPlane) -> LineString | None:
    pts = np.asarray(feat.points, dtype=float)
    if len(pts) < 2:
        return None
    uv = [tuple(_uv(plane, p)) for p in pts]
    if feat.kind in CLOSED_KINDS and uv[0] != uv[-1]:
        uv.append(uv[0])
    return LineString(uv)


def _cut_params(line_uv: tuple[np.ndarray, np.ndarray], others: list[LineString]) -> list[float]:
    """Parameters 0..1 along a line where other curves cross it."""
    a, b = line_uv
    segment = LineString([tuple(a), tuple(b)])
    length = segment.length
    params = []
    for other in others:
        hit = segment.intersection(other)
        if hit.is_empty:
            continue
        points = [hit] if isinstance(hit, Point) else list(getattr(hit, "geoms", []))
        if isinstance(hit, LineString):
            points = [Point(c) for c in hit.coords]
        for p in points:
            if isinstance(p, Point):
                t = segment.project(p) / length if length > 0 else 0.0
                if 1e-9 < t < 1 - 1e-9:
                    params.append(t)
    return sorted(set(round(t, 12) for t in params))


# ---------------------------------------------------------------- trim / extend
def trim(target, click_world, others: list) -> EditResult:
    """Remove the piece of a line between the intersections around the click."""
    if target.kind not in LINE_KINDS:
        return EditResult(message="Trim works on lines.")
    plane = plane_for([target] + list(others))
    a, b = _uv(plane, target.points[0]), _uv(plane, target.points[-1])
    curves = [c for c in (_curve_of(o, plane) for o in others if o.feature_id != target.feature_id) if c is not None]
    cuts = [0.0] + _cut_params((a, b), curves) + [1.0]
    click = _uv(plane, click_world)
    ab = b - a
    t = float(np.dot(click - a, ab) / max(float(np.dot(ab, ab)), 1e-12))
    pieces = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    keep = [(s, e) for s, e in pieces if not (s - 1e-9 <= t <= e + 1e-9)]
    result = EditResult(delete=[target.feature_id])
    if len(cuts) == 2:
        result.message = "Nothing crosses this line: the whole line is removed."
    for s, e in keep:
        pts = np.vstack((_world(plane, a + ab * s), _world(plane, a + ab * e)))
        result.add.append(NewEntity(target.kind, pts, target.construction, target.variant))
    return result


def power_trim(features: list, path_world: list) -> EditResult:
    """Drag across segments: trim every line the drag path crosses, at the crossing."""
    if len(path_world) < 2:
        return EditResult()
    lines = [f for f in features if f.kind in LINE_KINDS]
    if not lines:
        return EditResult()
    plane = plane_for(features)
    path = LineString([tuple(_uv(plane, p)) for p in path_world])
    merged = EditResult()
    for feat in lines:
        curve = _curve_of(feat, plane)
        hit = curve.intersection(path) if curve is not None else None
        if hit is None or hit.is_empty:
            continue
        point = hit if isinstance(hit, Point) else next((g for g in getattr(hit, "geoms", []) if isinstance(g, Point)), None)
        if point is None:
            continue
        piece = trim(feat, _world(plane, (point.x, point.y)), [f for f in features if f is not feat])
        merged.add.extend(piece.add)
        merged.delete.extend(piece.delete)
    return merged


def extend(target, click_world, others: list, reach: float = 1e4) -> EditResult:
    """Extend the line end nearest the click to the first entity in its direction."""
    if target.kind not in LINE_KINDS:
        return EditResult(message="Extend works on lines.")
    plane = plane_for([target] + list(others))
    a, b = _uv(plane, target.points[0]), _uv(plane, target.points[-1])
    click = _uv(plane, click_world)
    move_end = float(np.linalg.norm(click - b)) <= float(np.linalg.norm(click - a))
    start, fixed = (b, a) if move_end else (a, b)
    direction = start - fixed
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return EditResult()
    ray = LineString([tuple(start + direction / norm * 1e-9), tuple(start + direction / norm * reach)])
    best = None
    for other in others:
        if other.feature_id == target.feature_id:
            continue
        curve = _curve_of(other, plane)
        if curve is None:
            continue
        hit = ray.intersection(curve)
        if hit.is_empty:
            continue
        candidates = [hit] if isinstance(hit, Point) else [g for g in getattr(hit, "geoms", []) if isinstance(g, Point)]
        for point in candidates:
            d = Point(tuple(start)).distance(point)
            if best is None or d < best[0]:
                best = (d, np.array([point.x, point.y]))
    if best is None:
        return EditResult(message="Nothing to extend to in that direction.")
    new_end = best[1]
    pts = (np.vstack((_world(plane, fixed), _world(plane, new_end))) if move_end
           else np.vstack((_world(plane, new_end), _world(plane, fixed))))
    return EditResult(replace={target.feature_id: pts})


# ---------------------------------------------------------------- offset
def offset_entities(chain: list, distance: float, side: int = 1, both: bool = False) -> EditResult:
    """Offset a chain of lines (open or closed) or a circle by ``distance`` (spec B9).

    side +1 = outward / left, -1 = inward / right. Lines get parallel relations to their source
    and a driving offset dimension (point-to-line); circles get a concentric relation."""
    result = EditResult()
    if not chain:
        return result
    plane = plane_for(chain)
    sides = (1, -1) if both else (side,)
    if len(chain) == 1 and chain[0].kind == "Circle":
        src = chain[0]
        centre = reference_points(src)[0]
        radius = float(np.mean(np.linalg.norm(np.asarray(src.points[:-1]) - centre, axis=1)))
        for s in sides:
            r = radius + s * distance
            if r <= 1e-9:
                continue
            cu, cv = _uv(plane, centre)
            angles = np.linspace(0.0, 2 * math.pi, max(len(src.points), 49))
            pts = np.asarray([_world(plane, (cu + r * math.cos(t), cv + r * math.sin(t))) for t in angles])
            index = len(result.add)
            result.add.append(NewEntity("Circle", pts, src.construction))
            result.relations.append(Relation("concentric", [(f"new:{index}", 0), (src.feature_id, 0)]))
        return result
    if chain[0].kind == "Rectangle" and len(chain) == 1:
        corners = [tuple(_uv(plane, p)) for p in reference_points(chain[0])]
        segments = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        sources = [None] * 4
    else:
        lines = [f for f in chain if f.kind in LINE_KINDS]
        if len(lines) != len(chain):
            result.message = "Offset works on lines, rectangles and circles."
            return result
        segments = [(tuple(_uv(plane, f.points[0])), tuple(_uv(plane, f.points[-1]))) for f in lines]
        sources = [f.feature_id for f in lines]
    coords = [segments[0][0]] + [s[1] for s in segments]
    closed = math.dist(coords[0], coords[-1]) < 1e-6
    for s in sides:
        if closed:
            polygon = Polygon(coords[:-1])
            grow = s * distance if polygon.exterior.is_ccw else s * distance
            shape = polygon.buffer(grow, join_style=2, mitre_limit=10.0)
            if shape.is_empty or not isinstance(shape, Polygon):
                result.message = "The offset collapses the shape."
                continue
            ring = list(shape.exterior.coords)[:-1]
            # match each new edge to the source edge it is parallel to (same order around)
            new_segments = _match_ring(ring, segments)
        else:
            line = LineString(coords)
            moved = line.offset_curve(s * distance, join_style=2, mitre_limit=10.0)
            pts = list(moved.coords)
            new_segments = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        first_index = len(result.add)
        for i, (p, q) in enumerate(new_segments):
            pts = np.vstack((_world(plane, p), _world(plane, q)))
            index = len(result.add)
            result.add.append(NewEntity("Line", pts))
            source = sources[i] if i < len(sources) else None
            if source is not None:
                result.relations.append(Relation("parallel", [(f"new:{index}", None), (source, None)]))
            if i > 0:
                result.relations.append(Relation("coincident", [(f"new:{index - 1}", 1), (f"new:{index}", 0)]))
        if closed and len(new_segments) > 1:
            result.relations.append(Relation("coincident", [(f"new:{len(result.add) - 1}", 1), (f"new:{first_index}", 0)]))
        if sources and sources[0] is not None and new_segments:
            result.add[first_index].dimensions.append(
                SketchDimension("offset", "Offset", "PointLine", float(distance),
                                target_ids=[f"new:{first_index}", sources[0]], point_refs=[0, 0])
            )
    return result


def _match_ring(ring: list, segments: list) -> list:
    """Order a buffered polygon's edges like the source edges (by direction)."""
    edges = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    if len(edges) != len(segments):
        return edges

    def direction(p, q):
        return math.atan2(q[1] - p[1], q[0] - p[0])

    best_shift, best_score = 0, None
    reversed_ring = [(q, p) for p, q in edges[::-1]]
    for candidate in (edges, reversed_ring):
        for shift in range(len(candidate)):
            rolled = candidate[shift:] + candidate[:shift]
            score = sum(abs(math.remainder(direction(*a) - direction(*b), 2 * math.pi)) for a, b in zip(rolled, segments))
            if best_score is None or score < best_score:
                best_score, best_shift, chosen = score, shift, candidate
    return chosen[best_shift:] + chosen[:best_shift]


# ---------------------------------------------------------------- mirror
def mirror(features: list, centerline) -> EditResult:
    """Mirror entities about a centerline; mirrored points get symmetric relations."""
    plane = plane_for(list(features) + [centerline])
    a, b = _uv(plane, centerline.points[0]), _uv(plane, centerline.points[-1])
    d = b - a
    d = d / max(float(np.linalg.norm(d)), 1e-12)

    def reflect(p):
        uv = _uv(plane, p)
        rel = uv - a
        return _world(plane, a + 2 * np.dot(rel, d) * d - rel)

    result = EditResult()
    for feat in features:
        if feat.feature_id == centerline.feature_id:
            continue
        pts = np.asarray([reflect(p) for p in feat.points])
        control = None if feat.control_points is None else np.asarray([reflect(p) for p in feat.control_points])
        index = len(result.add)
        result.add.append(NewEntity(feat.kind, pts, feat.construction, feat.variant, control))
        if feat.kind in ("Point",) + LINE_KINDS:
            result.relations.append(Relation("symmetric", [(feat.feature_id, None), (f"new:{index}", None), (centerline.feature_id, None)]))
    return result


# ---------------------------------------------------------------- fillet / chamfer
def _corner(line_a, line_b, plane):
    pa = [_uv(plane, line_a.points[0]), _uv(plane, line_a.points[-1])]
    pb = [_uv(plane, line_b.points[0]), _uv(plane, line_b.points[-1])]
    best = min(((i, j) for i in (0, 1) for j in (0, 1)), key=lambda ij: float(np.linalg.norm(pa[ij[0]] - pb[ij[1]])))
    i, j = best
    if float(np.linalg.norm(pa[i] - pb[j])) > 1e-4:
        return None
    corner = (pa[i] + pb[j]) / 2.0
    return corner, pa[1 - i], pb[1 - j], i, j


def fillet(line_a, line_b, radius: float, segments: int = 16) -> EditResult:
    """Round the corner of two lines that meet: trim both, add a tangent arc; the corner stays
    as a construction point."""
    plane = plane_for([line_a, line_b])
    found = _corner(line_a, line_b, plane)
    if found is None:
        return EditResult(message="The two lines must meet at a corner.")
    corner, far_a, far_b, i, j = found
    ua = (far_a - corner) / max(float(np.linalg.norm(far_a - corner)), 1e-12)
    ub = (far_b - corner) / max(float(np.linalg.norm(far_b - corner)), 1e-12)
    angle = math.acos(min(max(float(np.dot(ua, ub)), -1.0), 1.0))
    if angle < 1e-6 or abs(angle - math.pi) < 1e-6:
        return EditResult(message="Parallel lines have no corner to fillet.")
    setback = radius / math.tan(angle / 2.0)
    if setback >= min(float(np.linalg.norm(far_a - corner)), float(np.linalg.norm(far_b - corner))):
        return EditResult(message="The fillet radius is too large for these lines.")
    ta, tb = corner + ua * setback, corner + ub * setback
    bisector = (ua + ub) / max(float(np.linalg.norm(ua + ub)), 1e-12)
    centre = corner + bisector * (radius / math.sin(angle / 2.0))
    a0 = math.atan2(ta[1] - centre[1], ta[0] - centre[0])
    a1 = math.atan2(tb[1] - centre[1], tb[0] - centre[0])
    sweep = math.remainder(a1 - a0, 2 * math.pi)
    arc = np.asarray([_world(plane, centre + radius * np.array([math.cos(a0 + sweep * k / segments), math.sin(a0 + sweep * k / segments)]))
                      for k in range(segments + 1)])
    result = EditResult(unlink=[(line_a.feature_id, line_b.feature_id)], join_ends=True)
    new_a = np.vstack((_world(plane, far_a), _world(plane, ta)))
    new_b = np.vstack((_world(plane, far_b), _world(plane, tb)))
    result.replace[line_a.feature_id] = new_a if i == 1 else new_a[::-1]
    result.replace[line_b.feature_id] = new_b if j == 1 else new_b[::-1]
    result.add.append(NewEntity("Curve", arc, dimensions=[SketchDimension("fillet_r", "Fillet radius", "Radius", float(radius))]))
    result.add.append(NewEntity("Point", _world(plane, corner).reshape(1, 3), construction=True))
    result.relations.append(Relation("tangent", [("new:0", None), (line_a.feature_id, None)]))
    result.relations.append(Relation("tangent", [("new:0", None), (line_b.feature_id, None)]))
    return result


def chamfer(line_a, line_b, distance: float) -> EditResult:
    """Cut the corner of two lines that meet by ``distance`` along each; corner kept as a
    construction point."""
    plane = plane_for([line_a, line_b])
    found = _corner(line_a, line_b, plane)
    if found is None:
        return EditResult(message="The two lines must meet at a corner.")
    corner, far_a, far_b, i, j = found
    la, lb = float(np.linalg.norm(far_a - corner)), float(np.linalg.norm(far_b - corner))
    if distance >= min(la, lb):
        return EditResult(message="The chamfer is longer than the lines.")
    ta = corner + (far_a - corner) / la * distance
    tb = corner + (far_b - corner) / lb * distance
    result = EditResult(unlink=[(line_a.feature_id, line_b.feature_id)], join_ends=True)
    new_a = np.vstack((_world(plane, far_a), _world(plane, ta)))
    new_b = np.vstack((_world(plane, far_b), _world(plane, tb)))
    result.replace[line_a.feature_id] = new_a if i == 1 else new_a[::-1]
    result.replace[line_b.feature_id] = new_b if j == 1 else new_b[::-1]
    result.add.append(NewEntity("Line", np.vstack((_world(plane, ta), _world(plane, tb)))))
    result.add.append(NewEntity("Point", _world(plane, corner).reshape(1, 3), construction=True))
    return result


# ---------------------------------------------------------------- convert entities
def convert_entities(segments: np.ndarray, part_index: int, plane: SketchPlane) -> EditResult:
    """Model edges (M, 2, 3) -> sketch lines anchored to the model at both ends (fully defined)."""
    result = EditResult()
    for segment in np.asarray(segments, dtype=float).reshape(-1, 2, 3):
        a, b = segment
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        pa, pb = plane.project_point_to_plane(a), plane.project_point_to_plane(b)
        index = len(result.add)
        result.add.append(NewEntity("Line", np.vstack((pa, pb))))
        for end, vertex in ((0, a), (1, b)):
            result.relations.append(Relation("anchor_model", [(f"new:{index}", end)], [float(c) for c in vertex], part_index=part_index))
    return result


def face_loop_segments(mesh, cell_ids) -> np.ndarray:
    """Outer boundary edges of a set of mesh cells (a face) as segments (M, 2, 3)."""
    sub = mesh.extract_cells(list(cell_ids)).extract_surface(algorithm="dataset_surface")
    edges = sub.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    if edges.n_lines == 0:
        return np.zeros((0, 2, 3))
    lines = np.asarray(edges.lines).reshape(-1, 3)[:, 1:]
    points = np.asarray(edges.points, dtype=float)
    # merge collinear runs of tessellation edges into straight model edges
    segs = [np.vstack((points[i], points[j])) for i, j in lines]
    return np.asarray(_merge_collinear(segs), dtype=float)


def _merge_collinear(segs: list, tol: float = 1e-6) -> list:
    segs = [s.copy() for s in segs]
    merged = True
    while merged:
        merged = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                for ia in (0, 1):
                    for jb in (0, 1):
                        if float(np.linalg.norm(a[ia] - b[jb])) < tol:
                            da = a[1 - ia] - a[ia]
                            db = b[1 - jb] - b[jb]
                            cross = float(np.linalg.norm(np.cross(da, db)))
                            if cross < tol * max(float(np.linalg.norm(da) * np.linalg.norm(db)), 1e-12) and float(np.dot(da, db)) < 0:
                                segs[i] = np.vstack((a[1 - ia], b[1 - jb]))
                                del segs[j]
                                merged = True
                                break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
    return segs


# ---------------------------------------------------------------- move / copy
def move_copy(features: list, delta_world, copy: bool) -> EditResult:
    delta = np.asarray(delta_world, dtype=float)
    result = EditResult()
    for feat in features:
        pts = np.asarray(feat.points, dtype=float) + delta
        if copy:
            control = None if feat.control_points is None else np.asarray(feat.control_points) + delta
            result.add.append(NewEntity(feat.kind, pts, feat.construction, feat.variant, control))
        else:
            result.replace[feat.feature_id] = pts
    return result


# ---------------------------------------------------------------- fully define
def fully_define(features: list, relations: list) -> tuple[list[Relation], list[tuple[str, SketchDimension]]]:
    """Relations and dimensions that make a sketch fully defined: H/V for nearly H/V lines,
    then X/Y dimensions from the origin for every point still free (spec B9)."""
    from basicam.sketch_interaction import hv_relation

    solver = SketchSolver()
    new_relations: list[Relation] = []
    new_dims: list[tuple[str, SketchDimension]] = []
    plane = plane_for(features)

    def state():
        work = [_copy_feature(f) for f in features]
        dims = {}
        for owner, dim in new_dims:
            dims.setdefault(owner, []).append(dim)
        for f in work:
            f.dimensions = list(f.dimensions) + dims.get(f.feature_id, [])
        return solver.solve(work, list(relations) + new_relations, apply=False)

    for feat in features:
        if feat.kind == "Line" and not feat.construction:
            hv = hv_relation(feat.points[0], feat.points[-1], plane)
            if hv:
                trial = Relation(hv, [(feat.feature_id, None)])
                new_relations.append(trial)
                if state().state == "error":
                    new_relations.pop()
    result = state()
    counter = 0
    world_axes = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
    plane_axes = [name for direction in (plane.u_axis, plane.v_axis) for name, w in world_axes.items()
                  if abs(float(np.dot(w, direction))) > 0.999] or ["X", "Y"]
    for feat in features:
        if result.dof == 0:
            break
        if feat.kind == "Circle":
            measured = float(np.mean(np.linalg.norm(np.asarray(feat.points[:-1]) - reference_points(feat)[0], axis=1))) * 2
            new_dims.append((feat.feature_id, SketchDimension(f"fd_{feat.feature_id}_d", "Diameter", "Diameter", measured)))
            if state().state == "error" or state().dof >= result.dof:
                new_dims.pop()
            else:
                result = state()
        for index, point in enumerate(reference_points(feat)):
            for axis in plane_axes:
                if result.dof == 0:
                    break
                value = float(np.asarray(point, dtype=float)["XYZ".index(axis)])
                counter += 1
                dim = SketchDimension(f"fd_{feat.feature_id}_{axis}{index}_{counter}", axis, axis, value,
                                      target_ids=[feat.feature_id], point_refs=[index])
                new_dims.append((feat.feature_id, dim))
                trial = state()
                if trial.state == "error" or trial.dof >= result.dof:
                    new_dims.pop()
                else:
                    result = trial
    return new_relations, new_dims


def _copy_feature(feat):
    import copy

    clone = copy.copy(feat)
    clone.points = np.asarray(feat.points, dtype=float).copy()
    clone.dimensions = list(feat.dimensions)
    return clone


__all__ = [
    "EditResult", "NewEntity", "chamfer", "convert_entities", "extend", "face_loop_segments", "fillet",
    "fully_define", "mirror", "move_copy", "offset_entities", "power_trim", "trim",
]
