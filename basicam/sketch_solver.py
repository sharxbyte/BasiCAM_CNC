"""Sketch definition: python-solvespace wrapper, relations, last-good store and dependencies.

Each sketch (owner) is solved on its own 2D work plane. States:
  defined       every entity fixed by dimensions/relations and anchored (dof 0)
  underdefined  free geometry or not anchored to the origin, model or coordinates
  error         conflicting dimensions or relations ("Could not solve")
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from python_solvespace import Constraint, Entity, ResultFlag, SolverSystem, make_quaternion

from basicam.sketch_workflow import (
    ParametricSketchFeature,
    SketchPlane,
    SketchWorkflowManager,
    build_standard_sketch_plane,
    slot_outline_uv,
)

STATE_DEFINED = "defined"
STATE_UNDERDEFINED = "underdefined"
STATE_ERROR = "error"

RELATION_KINDS = (
    "coincident", "horizontal", "vertical", "parallel", "perpendicular", "tangent", "equal",
    "concentric", "midpoint", "fix", "anchor_origin", "anchor_model",
    # v2 (sketch-engine.spec B6)
    "collinear",  # two lines on one infinite line
    "symmetric",  # two points (or two entities, point by point) mirrored about a centerline
    "on_edge",  # a point on a model edge: value = edge end points (world xyz x 2), part_index
    "on_curve",  # a point on a sketch line, circle or arc
    "equal_radius",  # two circles / arcs
)
MODEL_RELATION_KINDS = ("anchor_model", "on_edge")

PointRef = tuple[str, int]  # (entity id, point index)


@dataclass
class Relation:
    kind: str
    refs: list[tuple[str, int | None]]  # (entity id, point index or None for the whole entity)
    value: list[float] | None = None  # fix position / model point (world xyz) / model edge (2 x xyz)
    rel_id: str = field(default_factory=lambda: "rel_" + uuid.uuid4().hex[:8])
    part_index: int | None = None  # model references: the part they follow when it moves
    dangling: bool = False  # the referenced part was deleted
    auto: bool = False  # added automatically while drawing (H/V inference); dropped if redundant

    def entity_ids(self) -> list[str]:
        return [ref[0] for ref in self.refs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "refs": [list(r) for r in self.refs], "value": self.value,
            "rel_id": self.rel_id, "part_index": self.part_index, "dangling": self.dangling,
            "auto": self.auto,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        refs = [(str(r[0]), None if r[1] is None else int(r[1])) for r in data.get("refs", [])]
        part_index = data.get("part_index")
        return cls(
            str(data.get("kind", "fix")), refs, data.get("value"),
            str(data.get("rel_id") or "rel_" + uuid.uuid4().hex[:8]),
            None if part_index is None else int(part_index), bool(data.get("dangling", False)),
            bool(data.get("auto", False)),
        )


@dataclass
class SolveResult:
    state: str
    dof: int
    free_ids: list[str] = field(default_factory=list)
    conflicting_ids: list[str] = field(default_factory=list)  # relation / dimension ids
    message: str = ""


# ---------------------------------------------------------------------- plane helpers
def plane_for(features: list[ParametricSketchFeature]) -> SketchPlane:
    name = features[0].plane if features else "XY plane"
    if name in ("XY plane", "XZ plane", "YZ plane"):
        return build_standard_sketch_plane(name)
    pts = np.vstack([np.asarray(f.points, dtype=float) for f in features if len(f.points)]) if features else np.zeros((1, 3))
    center = pts.mean(axis=0)
    if len(pts) >= 3:
        _u, _s, vh = np.linalg.svd(pts - center)
        normal = vh[2]
    else:
        normal = np.array([0.0, 0.0, 1.0])
    if normal[2] < 0:
        normal = -normal
    # Sketch "horizontal" follows world X projected onto the face (Y for faces facing X), so a
    # line along X on a top face is horizontal, as on the XY plane.
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = helper - normal * float(np.dot(helper, normal))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    origin = normal * float(np.dot(center, normal))  # world origin projected onto the sketch plane
    return SketchPlane(name=name, origin=origin, normal=normal, u_axis=u, v_axis=v)


def _circle_through(a, b, c):
    (x1, y1), (x2, y2), (x3, y3) = a, b, c
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return cx, cy, math.hypot(x1 - cx, y1 - cy)


# ---------------------------------------------------------------------- system builder
class _Builder:
    """Builds a SolverSystem from one sketch's entities, dimensions and relations."""

    def __init__(
        self,
        features: list[ParametricSketchFeature],
        relations: list[Relation],
        plane: SketchPlane,
        fixed_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.fixed_ids = frozenset(fixed_ids)  # entities built as fixed parameters (free-entity probe)
        self._involved: set[str] = set()
        self.features = features
        self.relations = relations
        self.plane = plane
        self.sys = SolverSystem()
        self.wp = self.sys.create_2d_base()
        self.points: dict[str, list] = {}  # entity id -> solver point entities
        self.lines: dict[str, list] = {}
        self.circle: dict[str, Any] = {}
        self.kinds: dict[str, str] = {}
        self.uv_init: dict[str, list[tuple[float, float]]] = {}
        self.cons_owner: list[str] = []  # constraint handle (1-based) -> owner id
        self.extra: dict[str, Any] = {}
        self._origin = None

    def uv(self, world) -> tuple[float, float]:
        return self.plane.world_to_uv(np.asarray(world, dtype=float))

    def _track(self, owner: str, fn, *args) -> None:
        if self.fixed_ids and self._involved and self._involved <= self.fixed_ids:
            return  # only fixed parameters: nothing to solve (SolveSpace would call it inconsistent)
        before = self.sys.cons_len()
        fn(*args)
        self.cons_owner.extend([owner] * (self.sys.cons_len() - before))

    def _fixed_point(self, u: float, v: float):
        previous = self.sys.group()
        self.sys.set_group(1)
        point = self.sys.add_point_2d(float(u), float(v), self.wp)
        self.sys.set_group(previous)
        return point

    def normal(self):
        """Fixed 3D normal of the 2D base work plane (circles and arcs need one)."""
        if "_normal" not in self.extra:
            previous = self.sys.group()
            self.sys.set_group(1)
            self.extra["_normal"] = self.sys.add_normal_3d(*make_quaternion(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
            self.sys.set_group(previous)
        return self.extra["_normal"]

    def origin(self):
        if self._origin is None:
            self._origin = self._fixed_point(*self.uv(np.zeros(3)))
        return self._origin

    def build(self) -> "_Builder":
        s, wp = self.sys, self.wp
        s.set_group(2)
        for feat in self.features:
            fid = feat.feature_id
            pts = np.asarray(feat.points, dtype=float)
            kind = feat.kind
            self.kinds[fid] = kind
            self._involved = {fid}
            s.set_group(1 if fid in self.fixed_ids else 2)
            if kind == "Point" or len(pts) == 1:
                uv = [self.uv(pts[0])]
                self.points[fid] = [s.add_point_2d(*uv[0], wp)]
            elif kind in ("Line", "Construction line"):
                uv = [self.uv(pts[0]), self.uv(pts[-1])]
                p = [s.add_point_2d(*c, wp) for c in uv]
                self.points[fid] = p
                self.lines[fid] = [s.add_line_2d(p[0], p[1], wp)]
            elif kind == "Rectangle" and len(pts) >= 4:
                uv = [self.uv(c) for c in pts[:4]]
                p = [s.add_point_2d(*c, wp) for c in uv]
                lines = [s.add_line_2d(p[i], p[(i + 1) % 4], wp) for i in range(4)]
                self.points[fid], self.lines[fid] = p, lines
                if feat.variant == "3 point":
                    self._track(fid, s.perpendicular, lines[0], lines[1], wp)
                    self._track(fid, s.parallel, lines[0], lines[2], wp)
                    self._track(fid, s.parallel, lines[1], lines[3], wp)
                else:
                    for i in (0, 2):
                        self._track(fid, s.horizontal, lines[i], wp)
                    for i in (1, 3):
                        self._track(fid, s.vertical, lines[i], wp)
            elif kind == "Slot" and feat.control_points is not None and len(feat.control_points) >= 3:
                ctrl = [self.uv(c) for c in np.asarray(feat.control_points, dtype=float)[:3]]
                uv, _outline = slot_outline_uv(*ctrl)
                p = [s.add_point_2d(*c, wp) for c in uv]
                nm = self.normal()
                arc1 = s.add_arc(nm, p[0], p[2], p[3], wp)
                arc2 = s.add_arc(nm, p[1], p[4], p[5], wp)
                lines = [s.add_line_2d(p[3], p[4], wp), s.add_line_2d(p[5], p[2], wp)]
                # Semicircle ends: each arc's centre on its diameter, both diameters perpendicular to
                # one side, equal radii (arc ends stay on their circle implicitly). Tangent
                # constraints give a rank-deficient system at an exact slot, so they are not used.
                d1, d2 = s.add_line_2d(p[2], p[3], wp), s.add_line_2d(p[4], p[5], wp)
                self._track(fid, s.equal, arc1, arc2, wp)
                self._track(fid, s.coincident, p[0], d1, wp)
                self._track(fid, s.coincident, p[1], d2, wp)
                self._track(fid, s.perpendicular, lines[0], d1, wp)
                self._track(fid, s.perpendicular, lines[0], d2, wp)
                self.points[fid], self.lines[fid] = p, lines
                self.circle[fid] = arc1
                self.extra[fid] = {"arcs": [arc1, arc2]}
            elif kind == "Polygon" and feat.control_points is not None and len(feat.control_points) >= 4:
                uv = [self.uv(c) for c in np.asarray(feat.control_points, dtype=float)]
                p = [s.add_point_2d(*c, wp) for c in uv]
                centre, verts = p[0], p[1:]
                radius = float(np.hypot(uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]))
                dist = s.add_distance(radius, wp)
                ring = s.add_circle(self.normal(), centre, dist, wp)
                lines = [s.add_line_2d(verts[i], verts[(i + 1) % len(verts)], wp) for i in range(len(verts))]
                for vertex in verts:
                    self._track(fid, s.coincident, vertex, ring, wp)
                for line in lines[1:]:
                    self._track(fid, s.equal, lines[0], line, wp)
                self.points[fid], self.lines[fid] = p, lines
                self.extra[fid] = {"radius_entity": dist, "ring": ring}
            elif kind == "Circle" and len(pts) >= 4:
                ring = np.asarray([self.uv(c) for c in pts[:-1]])
                center = ring.mean(axis=0)
                radius = float(np.mean(np.linalg.norm(ring - center, axis=1)))
                uv = [tuple(center)]
                ct = s.add_point_2d(*uv[0], wp)
                nm = self.normal()
                dist = s.add_distance(radius, wp)
                self.points[fid] = [ct]
                self.circle[fid] = s.add_circle(nm, ct, dist, wp)
                self.extra[fid] = {"radius_entity": dist}
            elif kind == "Curve" and len(pts) >= 3:
                a, m, b = self.uv(pts[0]), self.uv(pts[len(pts) // 2]), self.uv(pts[-1])
                circle = _circle_through(a, m, b)
                if circle is None:
                    uv = [a, b]
                    p = [s.add_point_2d(*c, wp) for c in uv]
                    self.points[fid] = p
                    self.lines[fid] = [s.add_line_2d(p[0], p[1], wp)]
                    self.kinds[fid] = "Line"
                else:
                    cx, cy, _r = circle
                    ccw = ((a[0] - cx) * (m[1] - cy) - (a[1] - cy) * (m[0] - cx)) > 0
                    start, end = (a, b) if ccw else (b, a)
                    uv = [(cx, cy), start, end]
                    p = [s.add_point_2d(*c, wp) for c in uv]
                    nm = self.normal()
                    self.points[fid] = p
                    self.circle[fid] = s.add_arc(nm, p[0], p[1], p[2], wp)
                    self.extra[fid] = {"ccw": ccw}
            else:  # spline / projected spline: rigid shape, position by its first point
                uv = [self.uv(pts[0])]
                self.points[fid] = [s.add_point_2d(*uv[0], wp)]
                self.kinds[fid] = "Rigid"
            self.uv_init[fid] = [tuple(map(float, c)) for c in uv]
            s.set_group(2)
        self._add_dimensions()
        self._add_relations()
        return self

    def _add_dimensions(self) -> None:
        s, wp = self.sys, self.wp
        axes = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
        for feat in self.features:
            fid = feat.feature_id
            pts = self.points.get(fid, [])
            if not pts:
                continue
            for dim in feat.dimensions:
                if not dim.driving:
                    continue
                self._involved = {fid, *dim.target_ids}
                kind = dim.kind
                if kind in axes:
                    axis = axes[kind]
                    du, dv = float(np.dot(axis, self.plane.u_axis)), float(np.dot(axis, self.plane.v_axis))
                    target = np.zeros(3)
                    target[list(axes).index(kind)] = dim.value
                    tu, tv = self.uv(target)
                    point = pts[0]
                    start_uv = self.uv_init[fid][0]
                    if dim.point_refs and dim.target_ids:
                        picked = self._dim_point(dim, 0)
                        if picked is not None:
                            point = picked
                            start_uv = tuple(s.params(picked.params))
                    if abs(du) > 0.999:
                        ref = self._fixed_point(tu, start_uv[1] + 1.0)
                        line = s.add_line_2d(ref, point, wp)
                        self._track(dim.dim_id, s.vertical, line, wp)
                    elif abs(dv) > 0.999:
                        ref = self._fixed_point(start_uv[0] + 1.0, tv)
                        line = s.add_line_2d(ref, point, wp)
                        self._track(dim.dim_id, s.horizontal, line, wp)
                elif kind == "Length" and len(pts) >= 2:
                    self._track(dim.dim_id, s.distance, pts[0], pts[1], dim.value, wp)
                elif kind == "Diameter" and fid in self.circle:
                    self._track(dim.dim_id, s.diameter, self.circle[fid], dim.value)
                elif kind == "Width" and len(pts) >= 4:
                    self._track(dim.dim_id, s.distance, pts[0], pts[1], dim.value, wp)
                elif kind == "Height" and len(pts) >= 4:
                    self._track(dim.dim_id, s.distance, pts[1], pts[2], dim.value, wp)
                elif kind == "Distance" and len(dim.target_ids) == 2 and dim.point_refs:
                    a, b = self._dim_point(dim, 0), self._dim_point(dim, 1)
                    if a is not None and b is not None:
                        self._track(dim.dim_id, s.distance, a, b, dim.value, wp)
                elif kind == "Distance" and len(dim.target_ids) == 2:
                    other = self.points.get(dim.target_ids[0]) if dim.target_ids[1] == fid else self.points.get(dim.target_ids[1])
                    if other:
                        self._track(dim.dim_id, s.distance, other[0], pts[0], dim.value, wp)
                elif kind == "Angle" and len(dim.target_ids) == 2:
                    a, b = (self.lines.get(t, [None])[0] for t in dim.target_ids)
                    if a is not None and b is not None:
                        self._track(dim.dim_id, s.angle, a, b, float(dim.value), wp)
                elif kind == "Radius" and fid in self.circle:
                    self._track(dim.dim_id, s.diameter, self.circle[fid], 2.0 * float(dim.value))
                elif kind == "PointLine" and len(dim.target_ids) == 2:
                    point = self._dim_point(dim, 0)
                    line = self.lines.get(dim.target_ids[1], [None])[0]
                    if point is not None and line is not None:
                        self._track(dim.dim_id, s.distance, point, line, float(dim.value), wp)
                elif kind in ("HDistance", "VDistance") and len(dim.target_ids) == 2:
                    a, b = self._dim_point(dim, 0), self._dim_point(dim, 1)
                    only_fixed = bool(self.fixed_ids) and self._involved <= self.fixed_ids
                    if a is not None and b is not None and not only_fixed:
                        ua, va = s.params(a.params)
                        ub, vb = s.params(b.params)
                        horizontal = kind == "HDistance"
                        corner = s.add_point_2d(ub if horizontal else ua, va if horizontal else vb, wp)
                        first, second = s.add_line_2d(a, corner, wp), s.add_line_2d(corner, b, wp)
                        self._track(dim.dim_id, s.horizontal if horizontal else s.vertical, first, wp)
                        self._track(dim.dim_id, s.vertical if horizontal else s.horizontal, second, wp)
                        self._track(dim.dim_id, s.distance, a, corner, float(dim.value), wp)

    def _arc_line_tangent(self, id_a: str, id_b: str):
        """(line, radius line to the arc end the line touches) for an arc-line tangency.

        python-solvespace's ARC_LINE_TANGENT always uses the arc's start point, so a line
        touching the arc's end would be solved wrongly; tangency is expressed instead as the line
        perpendicular to the radius at the shared end."""
        for arc_id, line_id in ((id_a, id_b), (id_b, id_a)):
            arc = self.circle.get(arc_id)
            if arc is None or not arc.is_arc() or line_id not in self.lines or self.kinds.get(line_id) not in ("Line", "Construction line"):
                continue
            centre, start, end = self.points[arc_id][:3]
            ends = self.points[line_id]
            line_uv = [np.asarray(self.sys.params(p.params)) for p in ends]
            best = None
            for arc_point in (start, end):
                uv = np.asarray(self.sys.params(arc_point.params))
                d = min(float(np.linalg.norm(uv - q)) for q in line_uv)
                if best is None or d < best[0]:
                    best = (d, arc_point)
            spoke = self.sys.add_line_2d(centre, best[1], self.wp)
            return self.lines[line_id][0], spoke
        return None

    def _dim_point(self, dim, index: int):
        if index >= len(dim.target_ids):
            return None
        point_index = dim.point_refs[index] if index < len(dim.point_refs) else 0
        return self._point((dim.target_ids[index], point_index))

    def _point(self, ref) -> Any | None:
        fid, index = ref
        pts = self.points.get(fid)
        if not pts:
            return None
        return pts[min(index or 0, len(pts) - 1)]

    def _curve(self, fid):
        if fid in self.lines:
            return self.lines[fid][0]
        return self.circle.get(fid)

    def _add_relations(self) -> None:
        s, wp = self.sys, self.wp
        for rel in self.relations:
            refs = [r for r in rel.refs if r[0] in self.points]
            if len(refs) != len(rel.refs):
                continue  # references another sketch or a deleted entity (handled as broken elsewhere)
            k, rid = rel.kind, rel.rel_id
            self._involved = set(rel.entity_ids())
            try:
                if k in ("coincident", "concentric") and len(refs) == 2:
                    a, b = self._point(refs[0]), self._point(refs[1])
                    self._track(rid, s.coincident, a, b, wp)
                elif k in ("horizontal", "vertical") and refs:
                    fn = s.horizontal if k == "horizontal" else s.vertical
                    if len(refs) == 2:
                        line = s.add_line_2d(self._point(refs[0]), self._point(refs[1]), wp)
                    else:
                        line = self._curve(refs[0][0])
                    if line is not None:
                        self._track(rid, fn, line, wp)
                elif k in ("parallel", "perpendicular") and len(refs) == 2:
                    a, b = self._curve(refs[0][0]), self._curve(refs[1][0])
                    if a is not None and b is not None:
                        self._track(rid, s.parallel if k == "parallel" else s.perpendicular, a, b, wp)
                elif k == "equal" and len(refs) == 2:
                    a, b = self._curve(refs[0][0]), self._curve(refs[1][0])
                    if a is not None and b is not None:
                        self._track(rid, s.equal, a, b, wp)
                elif k == "tangent" and len(refs) == 2:
                    a, b = self._curve(refs[0][0]), self._curve(refs[1][0])
                    if a is not None and b is not None:
                        arc_line = self._arc_line_tangent(refs[0][0], refs[1][0])
                        if arc_line is not None:
                            line, spoke = arc_line
                            self._track(rid, s.perpendicular, line, spoke, wp)
                        else:
                            self._track(rid, s.tangent, a, b, wp)
                elif k == "midpoint" and len(refs) == 2:
                    point, line = self._point(refs[0]), self._curve(refs[1][0])
                    if point is not None and line is not None:
                        self._track(rid, s.midpoint, point, line, wp)
                elif k == "fix":
                    targets = [self._point(r)] if refs[0][1] is not None else list(self.points[refs[0][0]])
                    for index, point in enumerate(targets):
                        u, v = self.sys.params(point.params)
                        if rel.value and refs[0][1] is not None:
                            u, v = self.uv(rel.value)
                        self._track(rid, s.coincident, self._fixed_point(u, v), point, wp)
                elif k == "anchor_origin" and refs:
                    self._track(rid, s.coincident, self.origin(), self._point(refs[0]), wp)
                elif k == "anchor_model" and refs and rel.value and not rel.dangling:
                    u, v = self.uv(rel.value[:3])
                    self._track(rid, s.coincident, self._fixed_point(u, v), self._point(refs[0]), wp)
                elif k == "collinear" and len(refs) == 2:
                    a, b = self._curve(refs[0][0]), self._curve(refs[1][0])
                    other = self.points.get(refs[1][0])
                    if a is not None and b is not None and other:
                        self._track(rid, s.parallel, a, b, wp)
                        self._track(rid, s.coincident, other[0], a, wp)
                elif k == "symmetric" and len(refs) == 3:
                    centre = self._curve(refs[2][0])
                    if centre is None or refs[2][0] not in self.lines:
                        continue
                    if refs[0][1] is not None and refs[1][1] is not None:
                        pairs = [(self._point(refs[0]), self._point(refs[1]))]
                    else:
                        pa, pb = self.points[refs[0][0]], self.points[refs[1][0]]
                        pairs = list(zip(pa, pb))
                        ca, cb = self.circle.get(refs[0][0]), self.circle.get(refs[1][0])
                        if ca is not None and cb is not None and len(pa) == 1:
                            self._track(rid, s.equal, ca, cb, wp)
                    for pa, pb in pairs:
                        self._track(rid, s.symmetric, pa, pb, centre, wp)
                elif k == "on_edge" and refs and rel.value and len(rel.value) >= 6 and not rel.dangling:
                    a = self._fixed_point(*self.uv(rel.value[:3]))
                    b = self._fixed_point(*self.uv(rel.value[3:6]))
                    previous = s.group()
                    s.set_group(1)
                    edge = s.add_line_2d(a, b, wp)
                    s.set_group(previous)
                    self._track(rid, s.coincident, self._point(refs[0]), edge, wp)
                elif k == "on_curve" and len(refs) == 2:
                    point, curve = self._point(refs[0]), self._curve(refs[1][0])
                    if point is None or curve is None:
                        continue
                    if curve.is_arc():
                        self._track(
                            rid, s.add_constraint, int(Constraint.PT_ON_CIRCLE), wp, 0.0,
                            point, Entity.NONE, curve, Entity.NONE,
                        )
                    else:
                        self._track(rid, s.coincident, point, curve, wp)
                elif k == "equal_radius" and len(refs) == 2:
                    a, b = self.circle.get(refs[0][0]), self.circle.get(refs[1][0])
                    if a is not None and b is not None:
                        self._track(rid, s.equal, a, b, wp)
            except (TypeError, ValueError):
                continue


def _solve_system(builder: _Builder) -> tuple[int, int, list[str]]:
    flag = builder.sys.solve()
    failures = builder.sys.failures()
    owners = [builder.cons_owner[h - 1] for h in failures if 0 < h <= len(builder.cons_owner)]
    return flag, builder.sys.dof(), list(dict.fromkeys(owners))


class SketchSolver:
    """Solve one sketch: state, degrees of freedom, free entities and conflicting constraints."""

    def solve(
        self,
        features: list[ParametricSketchFeature],
        relations: Iterable[Relation] = (),
        apply: bool = True,
        dragged: tuple[str, int, tuple[float, float]] | None = None,
    ) -> SolveResult:
        """Solve one sketch. ``dragged`` = (entity id, point index, (u, v) on the sketch plane):
        drag to explore (spec B8) - the point is held at the cursor where the sketch allows it,
        otherwise the sketch settles as close to the cursor as its relations permit."""
        features = [f for f in features]
        if not features:
            return SolveResult(STATE_DEFINED, 0)
        relations = list(relations)
        plane = plane_for(features)
        builder = _Builder(features, relations, plane).build()
        if dragged is not None:
            builder = self._drag_builder(features, relations, plane, dragged) or builder
        flag, dof, conflicts = _solve_system(builder)
        if flag != ResultFlag.OKAY:
            names = ", ".join(conflicts) or "constraints"
            message = "Could not solve" + (f": conflicting {names}" if conflicts else "")
            return SolveResult(STATE_ERROR, dof, [], conflicts, message)
        if apply:
            _write_back(builder, features)
        free = self._free_entities(features, relations, dof) if dof > 0 else []
        return SolveResult(STATE_UNDERDEFINED if dof > 0 else STATE_DEFINED, dof, free, [], "")

    @staticmethod
    def _drag_builder(features, relations, plane, dragged) -> "_Builder | None":
        fid, index, (u, v) = dragged
        for hold in (True, False):  # hold at the cursor; else start there and settle nearby
            builder = _Builder(features, relations, plane).build()
            point = builder._point((fid, index))
            if point is None:
                return None
            builder.sys.set_params(point.params, (float(u), float(v)))
            if hold:
                builder._involved = {fid}
                builder._track(f"drag_{fid}", builder.sys.dragged, point, builder.wp)
            flag, _dof, _conflicts = _solve_system(builder)
            if flag == ResultFlag.OKAY:
                # rebuild with the solved start values so the caller's solve repeats this result
                final = _Builder(features, relations, plane).build()
                for entity_id, solver_points in builder.points.items():
                    for source, target in zip(solver_points, final.points.get(entity_id, [])):
                        final.sys.set_params(target.params, builder.sys.params(source.params))
                for entity_id, extra in builder.extra.items():
                    radius = extra.get("radius_entity") if isinstance(extra, dict) else None
                    target = final.extra.get(entity_id, {}).get("radius_entity") if isinstance(final.extra.get(entity_id), dict) else None
                    if radius is not None and target is not None:
                        final.sys.set_params(target.params, builder.sys.params(radius.params))
                return final
        return None

    def _free_entities(self, features, relations, dof: int) -> list[str]:
        """An entity is free when fixing all its points still removes degrees of freedom."""
        free = []
        plane = plane_for(features)
        for feat in features:
            # build the entity as fixed parameters; if that removes freedom, the entity could move
            builder = _Builder(features, list(relations), plane, fixed_ids=frozenset({feat.feature_id})).build()
            flag, probe_dof, _ = _solve_system(builder)
            if flag == ResultFlag.OKAY and probe_dof < dof:
                free.append(feat.feature_id)
        return free


def _write_back(builder: _Builder, features: list[ParametricSketchFeature]) -> None:
    plane, s = builder.plane, builder.sys
    for feat in features:
        fid = feat.feature_id
        pts = builder.points.get(fid)
        if not pts:
            continue
        uv = [tuple(s.params(p.params)) for p in pts]
        if all(math.dist(a, b) < 1e-9 for a, b in zip(uv, builder.uv_init[fid])) and fid not in builder.circle:
            continue
        world = [plane.uv_to_world(u, v) for u, v in uv]
        kind = builder.kinds[fid]
        if kind == "Point":
            feat.points = np.asarray(world[:1], dtype=float)
        elif kind in ("Line", "Construction line") and fid in builder.lines:
            if kind == "Line" and feat.kind == "Curve":
                continue
            new = np.asarray(feat.points, dtype=float).copy()
            new[0], new[-1] = world[0], world[1]
            feat.points = new
        elif kind == "Rectangle":
            feat.points = np.asarray(world + [world[0]], dtype=float)
        elif kind == "Slot":
            _pts, outline = slot_outline_uv(uv[0], uv[1], uv[2])
            feat.control_points = np.asarray(world[:3], dtype=float)
            feat.points = np.asarray([plane.uv_to_world(u, v) for u, v in outline], dtype=float)
        elif kind == "Polygon":
            feat.control_points = np.asarray(world, dtype=float)
            feat.points = np.asarray(world[1:] + [world[1]], dtype=float)
        elif kind == "Circle":
            radius = float(s.params(builder.extra[fid]["radius_entity"].params)[0])
            cu, cv = uv[0]
            n = max(len(feat.points) - 1, 48)
            angles = np.linspace(0.0, 2.0 * math.pi, n + 1)
            feat.points = np.asarray([plane.uv_to_world(cu + radius * math.cos(a), cv + radius * math.sin(a)) for a in angles])
        elif kind == "Curve":
            (cu, cv), start, end = uv
            r = math.hypot(start[0] - cu, start[1] - cv)
            a0 = math.atan2(start[1] - cv, start[0] - cu)
            sweep = (math.atan2(end[1] - cv, end[0] - cu) - a0) % (2 * math.pi) or 2 * math.pi
            n = max(len(feat.points) - 1, 16)
            angles = a0 + np.linspace(0.0, sweep, n + 1)
            arc = [plane.uv_to_world(cu + r * math.cos(a), cv + r * math.sin(a)) for a in angles]
            if not builder.extra[fid]["ccw"]:
                arc = arc[::-1]
            feat.points = np.asarray(arc, dtype=float)
        elif kind == "Rigid":
            delta = np.asarray(world[0]) - np.asarray(feat.points[0], dtype=float)
            feat.points = np.asarray(feat.points, dtype=float) + delta
            if feat.projected_points is not None:
                feat.projected_points = np.asarray(feat.projected_points, dtype=float) + delta


def reference_points(feat: ParametricSketchFeature) -> list[np.ndarray]:
    """World points in the order the solver indexes them (relation point refs use these indices)."""
    pts = np.asarray(feat.points, dtype=float)
    if feat.kind == "Point" or len(pts) == 1:
        return [pts[0]]
    if feat.kind in ("Line", "Construction line"):
        return [pts[0], pts[-1]]
    if feat.kind == "Rectangle" and len(pts) >= 4:
        return [pts[i] for i in range(4)]
    control = getattr(feat, "control_points", None)
    if feat.kind == "Slot" and control is not None and len(control) >= 3:
        plane = plane_for([feat])
        uv_pts, _outline = slot_outline_uv(*(plane.world_to_uv(c) for c in np.asarray(control, dtype=float)[:3]))
        return [np.asarray(plane.uv_to_world(u, v), dtype=float) for u, v in uv_pts]
    if feat.kind == "Polygon" and control is not None and len(control) >= 4:
        return [np.asarray(c, dtype=float) for c in control]
    if feat.kind == "Circle" and len(pts) >= 4:
        return [pts[:-1].mean(axis=0)]
    if feat.kind == "Curve" and len(pts) >= 3:
        plane = plane_for([feat])
        a, m, b = (plane.world_to_uv(pts[i]) for i in (0, len(pts) // 2, -1))
        circle = _circle_through(a, m, b)
        if circle is None:
            return [pts[0], pts[-1]]
        cx, cy, _r = circle
        ccw = ((a[0] - cx) * (m[1] - cy) - (a[1] - cy) * (m[0] - cx)) > 0
        start, end = (pts[0], pts[-1]) if ccw else (pts[-1], pts[0])
        return [plane.uv_to_world(cx, cy), start, end]
    return [pts[0]]


def closest_reference_points(a: ParametricSketchFeature, b: ParametricSketchFeature) -> tuple[int, int]:
    pa, pb = reference_points(a), reference_points(b)
    best = min(((i, j) for i in range(len(pa)) for j in range(len(pb))), key=lambda ij: float(np.linalg.norm(pa[ij[0]] - pb[ij[1]])))
    return best


# ---------------------------------------------------------------------- manager-level helpers
def relations_of(manager: SketchWorkflowManager) -> list[Relation]:
    if not hasattr(manager, "relations"):
        manager.relations = []
    return manager.relations


def sketch_groups(manager: SketchWorkflowManager) -> dict[str, list[ParametricSketchFeature]]:
    groups: dict[str, list[ParametricSketchFeature]] = {}
    for feat in manager.features:
        groups.setdefault(feat.owner_sketch_id or f"_solo_{feat.feature_id}", []).append(feat)
    return groups


class LastGoodStateStore:
    """Per sketch: copies of its entities and relations at the last time it had no error."""

    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}

    def store(self, owner: str, features: list[ParametricSketchFeature], relations: list[Relation]) -> None:
        ids = {f.feature_id for f in features}
        self.states[owner] = {
            "features": copy.deepcopy(features),
            "relations": [copy.deepcopy(r) for r in relations if set(r.entity_ids()) & ids],
        }

    def has(self, owner: str) -> bool:
        return owner in self.states

    def revert(self, manager: SketchWorkflowManager, owner: str) -> bool:
        state = self.states.get(owner)
        if state is None:
            return False
        restored = copy.deepcopy(state["features"])
        ids = {f.feature_id for f in restored} | {f.feature_id for f in manager.features if _owner(f) == owner}
        manager.features = [f for f in manager.features if _owner(f) != owner] + restored
        rels = relations_of(manager)
        manager.relations = [r for r in rels if not set(r.entity_ids()) & ids] + copy.deepcopy(state["relations"])
        return True


def _owner(feat: ParametricSketchFeature) -> str:
    return feat.owner_sketch_id or f"_solo_{feat.feature_id}"


def solve_all(manager: SketchWorkflowManager, store: LastGoodStateStore | None = None) -> dict[str, SolveResult]:
    """Re-solve every sketch and set each entity's state, colour flags and messages."""
    solver = SketchSolver()
    rels = relations_of(manager)
    known = {f.feature_id for f in manager.features}
    results: dict[str, SolveResult] = {}
    for owner, feats in sketch_groups(manager).items():
        ids = {f.feature_id for f in feats}
        sketch_rels = [r for r in rels if set(r.entity_ids()) & ids]
        broken = [f for f in feats if getattr(f, "broken_reason", "")]
        missing = [r for r in sketch_rels if not set(r.entity_ids()) <= known]
        missing += [r for r in sketch_rels if getattr(r, "dangling", False) and r not in missing]
        invalid = []
        for feat in feats:
            manager._validate_feature(feat)
            if feat.has_error:
                invalid.append(feat)
        if invalid:
            result = SolveResult(STATE_ERROR, 0, [], [f.feature_id for f in invalid], f"Could not solve: {invalid[0].error_message}")
        elif broken or missing:
            if broken:
                message = broken[0].broken_reason
            elif any(getattr(r, "dangling", False) for r in missing):
                message = "Dangling model reference: the part it was attached to was deleted"
            else:
                message = "Reference to a deleted entity"
            result = SolveResult(STATE_ERROR, 0, [], [r.rel_id for r in missing], message)
        else:
            result = solver.solve(feats, sketch_rels)
            if result.state == STATE_ERROR:
                # This solver reports redundant constraints as inconsistent: an automatic
                # (inferred) relation made redundant by a later one is dropped, as SolidWorks would.
                for rel in sorted((r for r in sketch_rels if r.auto), key=lambda r: r.rel_id not in result.conflicting_ids):
                    trial = [r for r in sketch_rels if r is not rel]
                    retry = solver.solve(feats, trial)
                    if retry.state != STATE_ERROR:
                        manager.relations = [r for r in manager.relations if r is not rel]
                        sketch_rels, result = trial, retry
                        break
        results[owner] = result
        for feat in feats:
            if result.state == STATE_ERROR:
                feat.state = STATE_ERROR
                feat.has_error = True
                feat.error_message = result.message or "Could not solve"
                feat.defined = False
            else:
                feat.state = STATE_UNDERDEFINED if feat.feature_id in result.free_ids else STATE_DEFINED
                feat.has_error = False
                feat.error_message = ""
                feat.defined = feat.state == STATE_DEFINED
        if store is not None and result.state != STATE_ERROR:
            store.store(owner, feats, sketch_rels)
    return results


def sketch_status_text(name: str, result: SolveResult | None) -> str:
    """One status line per sketch (sketch-engine.spec B8)."""
    if result is None:
        return f"{name}: Not solved"
    if result.state == STATE_ERROR:
        conflicts = ", ".join(result.conflicting_ids)
        detail = result.message or "Could not solve"
        if conflicts and "conflicting" not in detail:
            detail += f" (conflicting: {conflicts})"
        return f"{name}: {detail}"
    if result.state == STATE_UNDERDEFINED or result.dof > 0:
        return f"{name}: Under defined ({result.dof} DOF, {len(result.free_ids)} free)"
    return f"{name}: Fully defined"


def model_reference_shift(relations: Iterable[Relation], part_index: int, transform) -> int:
    """Move anchor_model / on_edge values of relations attached to `part_index` by `transform`
    (a callable mapping a world xyz point). Returns the number of relations moved."""
    moved = 0
    for rel in relations:
        if rel.kind in MODEL_RELATION_KINDS and rel.part_index == part_index and rel.value and not rel.dangling:
            values = list(rel.value)
            rel.value = [
                float(c) for i in range(0, len(values) - len(values) % 3, 3)
                for c in transform(np.asarray(values[i:i + 3], dtype=float))
            ]
            moved += 1
    return moved


def mark_part_deleted(relations: Iterable[Relation], part_index: int) -> int:
    """References to a deleted part become dangling; later parts shift down by one index."""
    dangling = 0
    for rel in relations:
        if rel.part_index is None:
            continue
        if rel.part_index == part_index:
            rel.dangling = True
            dangling += 1
        elif rel.part_index > part_index:
            rel.part_index -= 1
    return dangling


def dependents_of(manager: SketchWorkflowManager, entity_id: str) -> list[str]:
    """Entities that depend on `entity_id` through relations, bound dimensions or parent links."""
    deps: list[str] = []
    for rel in relations_of(manager):
        ids = rel.entity_ids()
        if entity_id in ids:
            deps.extend(i for i in ids if i != entity_id)
    for feat in manager.features:
        if feat.feature_id == entity_id:
            continue
        if feat.parent_feature_id == entity_id:
            deps.append(feat.feature_id)
        for dim in feat.dimensions:
            if entity_id in dim.target_ids:
                deps.append(feat.feature_id)
    known = {f.feature_id for f in manager.features}
    return [d for d in dict.fromkeys(deps) if d in known]


def delete_entities(manager: SketchWorkflowManager, entity_ids: Iterable[str]) -> list[str]:
    """Delete entities; their dependents are marked broken (red). Returns the newly broken ids."""
    doomed = set(entity_ids)
    broken: list[str] = []
    for eid in doomed:
        for dep in dependents_of(manager, eid):
            if dep not in doomed:
                broken.append(dep)
    manager.features = [f for f in manager.features if f.feature_id not in doomed]
    manager.relations = [r for r in relations_of(manager) if not set(r.entity_ids()) & doomed]
    for feat in manager.features:
        if feat.feature_id in broken:
            feat.broken_reason = "Lost a reference to a deleted entity"
            feat.dimensions = [d for d in feat.dimensions if not set(d.target_ids) & doomed]
            if feat.parent_feature_id in doomed:
                feat.parent_feature_id = None
    return list(dict.fromkeys(broken))


def broken_ids(manager: SketchWorkflowManager) -> list[str]:
    return [f.feature_id for f in manager.features if getattr(f, "broken_reason", "")]
