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
from python_solvespace import ResultFlag, SolverSystem, make_quaternion

from basicam.sketch_workflow import (
    ParametricSketchFeature,
    SketchPlane,
    SketchWorkflowManager,
    build_standard_sketch_plane,
)

STATE_DEFINED = "defined"
STATE_UNDERDEFINED = "underdefined"
STATE_ERROR = "error"

RELATION_KINDS = (
    "coincident", "horizontal", "vertical", "parallel", "perpendicular", "tangent", "equal",
    "concentric", "midpoint", "fix", "anchor_origin", "anchor_model",
)

PointRef = tuple[str, int]  # (entity id, point index)


@dataclass
class Relation:
    kind: str
    refs: list[tuple[str, int | None]]  # (entity id, point index or None for the whole entity)
    value: list[float] | None = None  # fix position / model point (world xyz)
    rel_id: str = field(default_factory=lambda: "rel_" + uuid.uuid4().hex[:8])

    def entity_ids(self) -> list[str]:
        return [ref[0] for ref in self.refs]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "refs": [list(r) for r in self.refs], "value": self.value, "rel_id": self.rel_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        refs = [(str(r[0]), None if r[1] is None else int(r[1])) for r in data.get("refs", [])]
        return cls(str(data.get("kind", "fix")), refs, data.get("value"), str(data.get("rel_id") or "rel_" + uuid.uuid4().hex[:8]))


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
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, normal)
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
                    if abs(du) > 0.999:
                        ref = self._fixed_point(tu, self.uv_init[fid][0][1] + 1.0)
                        line = s.add_line_2d(ref, pts[0], wp)
                        self._track(dim.dim_id, s.vertical, line, wp)
                    elif abs(dv) > 0.999:
                        ref = self._fixed_point(self.uv_init[fid][0][0] + 1.0, tv)
                        line = s.add_line_2d(ref, pts[0], wp)
                        self._track(dim.dim_id, s.horizontal, line, wp)
                elif kind == "Length" and len(pts) >= 2:
                    self._track(dim.dim_id, s.distance, pts[0], pts[1], dim.value, wp)
                elif kind == "Diameter" and fid in self.circle:
                    self._track(dim.dim_id, s.diameter, self.circle[fid], dim.value)
                elif kind == "Width" and len(pts) >= 4:
                    self._track(dim.dim_id, s.distance, pts[0], pts[1], dim.value, wp)
                elif kind == "Height" and len(pts) >= 4:
                    self._track(dim.dim_id, s.distance, pts[1], pts[2], dim.value, wp)
                elif kind == "Distance" and len(dim.target_ids) == 2:
                    other = self.points.get(dim.target_ids[0]) if dim.target_ids[1] == fid else self.points.get(dim.target_ids[1])
                    if other:
                        self._track(dim.dim_id, s.distance, other[0], pts[0], dim.value, wp)

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
                elif k == "anchor_model" and refs and rel.value:
                    u, v = self.uv(rel.value)
                    self._track(rid, s.coincident, self._fixed_point(u, v), self._point(refs[0]), wp)
            except (TypeError, ValueError):
                continue


def _solve_system(builder: _Builder) -> tuple[int, int, list[str]]:
    flag = builder.sys.solve()
    failures = builder.sys.failures()
    owners = [builder.cons_owner[h - 1] for h in failures if 0 < h <= len(builder.cons_owner)]
    return flag, builder.sys.dof(), list(dict.fromkeys(owners))


class SketchSolver:
    """Solve one sketch: state, degrees of freedom, free entities and conflicting constraints."""

    def solve(self, features: list[ParametricSketchFeature], relations: Iterable[Relation] = (), apply: bool = True) -> SolveResult:
        features = [f for f in features]
        if not features:
            return SolveResult(STATE_DEFINED, 0)
        relations = list(relations)
        plane = plane_for(features)
        builder = _Builder(features, relations, plane).build()
        flag, dof, conflicts = _solve_system(builder)
        if flag != ResultFlag.OKAY:
            names = ", ".join(conflicts) or "constraints"
            message = "Could not solve" + (f": conflicting {names}" if conflicts else "")
            return SolveResult(STATE_ERROR, dof, [], conflicts, message)
        if apply:
            _write_back(builder, features)
        free = self._free_entities(features, relations, dof) if dof > 0 else []
        return SolveResult(STATE_UNDERDEFINED if dof > 0 else STATE_DEFINED, dof, free, [], "")

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
        invalid = []
        for feat in feats:
            manager._validate_feature(feat)
            if feat.has_error:
                invalid.append(feat)
        if invalid:
            result = SolveResult(STATE_ERROR, 0, [], [f.feature_id for f in invalid], f"Could not solve: {invalid[0].error_message}")
        elif broken or missing:
            message = (broken[0].broken_reason if broken else "Reference to a deleted entity")
            result = SolveResult(STATE_ERROR, 0, [], [r.rel_id for r in missing], message)
        else:
            result = solver.solve(feats, sketch_rels)
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
