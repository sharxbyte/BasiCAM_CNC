from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyvista as pv

from basicam.setups import MachiningSetup
from basicam.theme import TOKENS as _THEME_TOKENS


SKETCH_COLOR_UNDERDEFINED = _THEME_TOKENS["sketch_underdefined"]  # yellow (warning token)
SKETCH_COLOR_DEFINED = _THEME_TOKENS["sketch_defined"]
SKETCH_COLOR_ERROR = _THEME_TOKENS["error"]  # only errors use red


@dataclass
class SketchPlane:
    """Active reference plane or curved-surface projection envelope for a sketch session."""

    name: str
    origin: np.ndarray  # shape (3,)
    normal: np.ndarray  # unit vector shape (3,)
    u_axis: np.ndarray  # unit vector shape (3,)
    v_axis: np.ndarray  # unit vector shape (3,)
    is_curved_surface: bool = False
    source_part_index: int | None = None
    source_feature_ids: list[int] = field(default_factory=list)
    surface_mesh: pv.PolyData | None = None

    def project_point_to_plane(self, world_pt: np.ndarray) -> np.ndarray:
        p = np.asarray(world_pt, dtype=float)
        dist = float(np.dot(p - self.origin, self.normal))
        return p - dist * self.normal

    def world_to_uv(self, world_pt: np.ndarray) -> tuple[float, float]:
        rel = np.asarray(world_pt, dtype=float) - self.origin
        return float(np.dot(rel, self.u_axis)), float(np.dot(rel, self.v_axis))

    def uv_to_world(self, u: float, v: float) -> np.ndarray:
        return self.origin + float(u) * self.u_axis + float(v) * self.v_axis


@dataclass
class SnapSettings:
    """Toggleable sketch snapping and geometric constraint modes."""

    endpoints: bool = True
    midpoints: bool = True
    centers: bool = True
    edges: bool = True
    coincident: bool = True
    tangent: bool = True
    horizontal: bool = True
    vertical: bool = True
    snap_radius_mm: float = 1.5


@dataclass
class SnapCandidate:
    point: np.ndarray
    kind: str  # "Endpoint", "Midpoint", "Center", "Edge", "Coincident", "Tangent", "Horizontal", "Vertical"
    label: str = ""


@dataclass
class SketchDimension:
    """Parametric dimension / constraint driving a sketch feature."""

    dim_id: str
    name: str
    kind: str  # "X", "Y", "Z", "Length", "Offset", "Diameter", "Width", "Height"
    value: float
    tolerance: float = 0.0
    valid: bool = True
    error_message: str = ""
    target_ids: list[str] = field(default_factory=list)  # entity ids this dimension is bound to
    driving: bool = True  # driving dimensions set geometry; driven ones only report it
    unit_text: str = "mm"  # unit the user typed (for display)
    sig_figs: int = 4  # significant figures of the typed value


@dataclass
class ParametricSketchFeature:
    """A parametric sketch or surface-projected spline feature in the workflow tree."""

    feature_id: str
    name: str
    kind: str  # "Point", "Line", "Construction line", "Rectangle", "Circle", "Curve", "Spline", "Projected Spline"
    plane: str
    points: np.ndarray  # shape (N, 3)
    construction: bool = False
    parent_feature_id: str | None = None
    dimensions: list[SketchDimension] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    defined: bool = False
    has_error: bool = False
    error_message: str = ""
    assigned_operation_name: str | None = None
    assigned_setup_name: str | None = None
    assigned_setup_color: str | None = None
    projected_points: np.ndarray | None = None  # shape (M, 3) when projected onto curved surface
    is_projected: bool = False
    owner_sketch_id: str = ""  # the sketch (support plane/face) this entity belongs to
    variant: str = ""  # drawing method, e.g. "Center" rectangle
    state: str = "underdefined"  # "defined", "underdefined" or "error" (set by the solver)
    broken_reason: str = ""  # set when a referenced entity was deleted
    status: str = ""  # "future" = Future operation (no tool needed yet)

    @property
    def entity_id(self) -> str:
        """Stable id, never reused; relations and tool assignments refer to it."""
        return self.feature_id

    @property
    def active_points(self) -> np.ndarray:
        if self.is_projected and self.projected_points is not None and len(self.projected_points) > 0:
            return self.projected_points
        return self.points

    @property
    def is_underdefined(self) -> bool:
        return not self.has_error and self.state != "defined"

    @property
    def display_color(self) -> str:
        """Red = error (also construction); yellow = under-defined, except construction features."""
        if self.has_error:
            return SKETCH_COLOR_ERROR
        if self.is_underdefined and not self.construction:
            return SKETCH_COLOR_UNDERDEFINED
        if self.assigned_setup_color:
            return self.assigned_setup_color
        return SKETCH_COLOR_DEFINED

    @property
    def status_label(self) -> str:
        if self.has_error:
            message = self.error_message or "Could not solve"
            return message if message.startswith("Could not solve") else f"Could not solve: {message}"
        proj_tag = " [3D Spline]" if self.is_projected else ""
        if self.is_underdefined:
            return ("Under-defined construction" if self.construction else "Under-defined") + proj_tag
        if self.assigned_setup_name:
            op = f" ({self.assigned_operation_name})" if self.assigned_operation_name else ""
            return f"Tooling · {self.assigned_setup_name}{op}{proj_tag}"
        return f"Defined{proj_tag}"


def build_standard_sketch_plane(plane_name: str, origin: np.ndarray | None = None) -> SketchPlane:
    org = np.asarray(origin, dtype=float) if origin is not None else np.zeros(3, dtype=float)
    if plane_name == "XZ plane":
        return SketchPlane(
            name="XZ plane",
            origin=org,
            normal=np.array([0.0, 1.0, 0.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
        )
    if plane_name == "YZ plane":
        return SketchPlane(
            name="YZ plane",
            origin=org,
            normal=np.array([1.0, 0.0, 0.0]),
            u_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
        )
    return SketchPlane(
        name="XY plane",
        origin=org,
        normal=np.array([0.0, 0.0, 1.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
    )


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 1.0, 0.0]) if abs(float(np.dot(n, [0.0, 0.0, 1.0]))) > 0.92 else np.array([0.0, 0.0, 1.0])
    u = np.cross(ref, n)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return n, u, v


def flood_select_manifold_features(
    mesh: pv.PolyData,
    seed_feature_id: int,
    max_angle_deg: float = 30.0,
) -> tuple[set[int], np.ndarray]:
    """Flood-selects connected CAD/mesh feature_ids across edges where dihedral angle <= max_angle_deg."""
    if mesh is None or mesh.n_cells == 0 or "feature_id" not in mesh.cell_data:
        return {seed_feature_id}, np.empty(0, dtype=int)

    fids = np.asarray(mesh.cell_data["feature_id"], dtype=int)
    with_normals = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=False,
        inplace=False,
    )
    normals = np.asarray(with_normals.cell_data["Normals"], dtype=float)
    cos_threshold = math.cos(math.radians(max_angle_deg))

    # Build feature-to-feature adjacency with minimum normal alignment across shared vertices
    vert_to_cells: dict[int, list[int]] = {}
    for cid in range(mesh.n_cells):
        pts = mesh.get_cell(cid).point_ids
        for pid in pts:
            vert_to_cells.setdefault(int(pid), []).append(cid)

    adj_features: dict[int, set[int]] = {}
    for cids in vert_to_cells.values():
        if len(cids) < 2:
            continue
        for i in range(len(cids)):
            c1 = cids[i]
            f1 = int(fids[c1])
            n1 = normals[c1]
            for j in range(i + 1, len(cids)):
                c2 = cids[j]
                f2 = int(fids[c2])
                if f1 == f2:
                    continue
                n2 = normals[c2]
                if float(np.dot(n1, n2)) >= cos_threshold:
                    adj_features.setdefault(f1, set()).add(f2)
                    adj_features.setdefault(f2, set()).add(f1)

    visited: set[int] = {int(seed_feature_id)}
    queue = [int(seed_feature_id)]
    while queue:
        curr = queue.pop(0)
        for nxt in adj_features.get(curr, ()):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)

    cell_mask = np.isin(fids, list(visited))
    return visited, np.flatnonzero(cell_mask)


def build_sketch_plane_from_surface(
    mesh: pv.PolyData,
    cell_indices: np.ndarray,
    *,
    part_index: int,
    feature_ids: Iterable[int],
    label: str = "Surface plane",
) -> SketchPlane:
    """Creates a flat sketch plane from selected face(s); for curved surfaces sets Z-max plane for projection."""
    sub = mesh.extract_cells(cell_indices).extract_surface()
    with_normals = sub.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=False,
        inplace=False,
    )
    normals = np.asarray(with_normals.cell_data["Normals"], dtype=float)
    mean_n = normals.mean(axis=0) if len(normals) else np.array([0.0, 0.0, 1.0])
    mean_n /= max(float(np.linalg.norm(mean_n)), 1e-12)
    angular_spread = (
        float(np.max(np.degrees(np.arccos(np.clip(normals @ mean_n, -1.0, 1.0)))))
        if len(normals)
        else 0.0
    )
    is_curved = angular_spread > 2.5
    pts = np.asarray(sub.points, dtype=float)
    center = pts.mean(axis=0) if len(pts) else np.zeros(3, dtype=float)

    if is_curved and len(pts):
        max_z = float(np.max(pts[:, 2]))
        origin = np.array([float(center[0]), float(center[1]), max_z], dtype=float)
        n, u, v = _basis_from_normal(np.array([0.0, 0.0, 1.0]))
    else:
        origin = center
        n, u, v = _basis_from_normal(mean_n)

    return SketchPlane(
        name=label,
        origin=origin,
        normal=n,
        u_axis=u,
        v_axis=v,
        is_curved_surface=is_curved,
        source_part_index=part_index,
        source_feature_ids=sorted(int(f) for f in feature_ids),
        surface_mesh=sub,
    )


def sample_rectangle_points(p1: np.ndarray, p2: np.ndarray, plane: SketchPlane, mode: str = "Corner") -> np.ndarray:
    u1, v1 = plane.world_to_uv(p1)
    u2, v2 = plane.world_to_uv(p2)
    if mode == "Center":
        du = abs(u2 - u1)
        dv = abs(v2 - v1)
        umin, umax = u1 - du, u1 + du
        vmin, vmax = v1 - dv, v1 + dv
    else:
        umin, umax = min(u1, u2), max(u1, u2)
        vmin, vmax = min(v1, v2), max(v1, v2)
    corners = [
        plane.uv_to_world(umin, vmin),
        plane.uv_to_world(umax, vmin),
        plane.uv_to_world(umax, vmax),
        plane.uv_to_world(umin, vmax),
        plane.uv_to_world(umin, vmin),
    ]
    return np.asarray(corners, dtype=float)


def sample_circle_points(center: np.ndarray, edge_pt: np.ndarray, plane: SketchPlane, segments: int = 48) -> np.ndarray:
    uc, vc = plane.world_to_uv(center)
    ue, ve = plane.world_to_uv(edge_pt)
    radius = max(math.hypot(ue - uc, ve - vc), 0.05)
    angles = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    pts = [
        plane.uv_to_world(uc + radius * math.cos(a), vc + radius * math.sin(a))
        for a in angles
    ]
    return np.asarray(pts, dtype=float)


def _circle_through(uv1, uv2, uv3) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = uv1, uv2, uv3
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return cx, cy, math.hypot(x1 - cx, y1 - cy)


def sample_circle_three_points(p1, p2, p3, plane: SketchPlane, segments: int = 48) -> np.ndarray | None:
    """Circle through three points; None when they are collinear."""
    circle = _circle_through(plane.world_to_uv(p1), plane.world_to_uv(p2), plane.world_to_uv(p3))
    if circle is None:
        return None
    cx, cy, r = circle
    angles = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    return np.asarray([plane.uv_to_world(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles], dtype=float)


def sample_center_arc(center, start, end, plane: SketchPlane, segments: int = 32) -> np.ndarray:
    """Counter-clockwise arc around `center` from `start` to the direction of `end`."""
    uc, vc = plane.world_to_uv(center)
    us, vs = plane.world_to_uv(start)
    ue, ve = plane.world_to_uv(end)
    r = max(math.hypot(us - uc, vs - vc), 0.05)
    a0 = math.atan2(vs - vc, us - uc)
    a1 = math.atan2(ve - vc, ue - uc)
    sweep = (a1 - a0) % (2.0 * math.pi) or 2.0 * math.pi
    angles = a0 + np.linspace(0.0, sweep, segments + 1)
    return np.asarray([plane.uv_to_world(uc + r * math.cos(a), vc + r * math.sin(a)) for a in angles], dtype=float)


def sample_rectangle_three_points(p1, p2, p3, plane: SketchPlane) -> np.ndarray:
    """p1-p2 is one edge (any angle); p3 sets the width perpendicular to it."""
    a = np.asarray(plane.world_to_uv(p1), dtype=float)
    b = np.asarray(plane.world_to_uv(p2), dtype=float)
    c = np.asarray(plane.world_to_uv(p3), dtype=float)
    edge = b - a
    length = float(np.linalg.norm(edge))
    if length < 1e-9:
        return sample_rectangle_points(p1, p3, plane)
    normal = np.array([-edge[1], edge[0]]) / length
    offset = normal * float(np.dot(c - a, normal))
    corners = [a, b, b + offset, a + offset, a]
    return np.asarray([plane.uv_to_world(float(u), float(v)) for u, v in corners], dtype=float)


def sample_three_point_arc(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, plane: SketchPlane, segments: int = 32) -> np.ndarray:
    """Circular arc from p1 through p2 to p3 on the active sketch plane (straight line if collinear)."""
    uv1, uv2, uv3 = plane.world_to_uv(p1), plane.world_to_uv(p2), plane.world_to_uv(p3)
    circle = _circle_through(uv1, uv2, uv3)
    if circle is not None:
        cx, cy, r = circle
        a1 = math.atan2(uv1[1] - cy, uv1[0] - cx)
        a2 = math.atan2(uv2[1] - cy, uv2[0] - cx)
        a3 = math.atan2(uv3[1] - cy, uv3[0] - cx)
        ccw_to_3 = (a3 - a1) % (2.0 * math.pi)
        ccw_to_2 = (a2 - a1) % (2.0 * math.pi)
        sweep = ccw_to_3 if ccw_to_2 <= ccw_to_3 else ccw_to_3 - 2.0 * math.pi
        angles = a1 + np.linspace(0.0, sweep, segments + 1)
        return np.asarray([plane.uv_to_world(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles], dtype=float)
    u1, v1 = uv1
    u2, v2 = uv2
    u3, v3 = uv3
    ts = np.linspace(0.0, 1.0, segments + 1)
    # Lagrange quadratic interpolation passing strictly through p1 (t=0), p2 (t=0.5), p3 (t=1)
    pts = []
    for t in ts:
        l0 = 2.0 * (t - 0.5) * (t - 1.0)
        l1 = -4.0 * t * (t - 1.0)
        l2 = 2.0 * t * (t - 0.5)
        u = l0 * u1 + l1 * u2 + l2 * u3
        v = l0 * v1 + l1 * v2 + l2 * v3
        pts.append(plane.uv_to_world(u, v))
    return np.asarray(pts, dtype=float)


def sample_catmull_rom_spline(control_pts: np.ndarray, samples_per_seg: int = 16) -> np.ndarray:
    pts = np.asarray(control_pts, dtype=float)
    if len(pts) < 3:
        return pts
    padded = np.vstack([pts[0], pts, pts[-1]])
    out: list[np.ndarray] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for t in np.linspace(0.0, 1.0, samples_per_seg, endpoint=(i == len(padded) - 3)):
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(pt)
    return np.asarray(out, dtype=float)


def project_points_onto_surface(
    points: np.ndarray,
    surface_mesh: pv.PolyData,
    projection_dir: np.ndarray | None = None,
    dense_step_mm: float = 1.0,
) -> np.ndarray:
    """Densifies a polyline/spline and projects each sample vertically/normally onto surface_mesh."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0 or surface_mesh is None or surface_mesh.n_cells == 0:
        return pts

    # Densify segments so projected splines conform smoothly to curvature
    dense: list[np.ndarray] = [pts[0]]
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        dist = float(np.linalg.norm(seg))
        steps = max(1, int(math.ceil(dist / max(dense_step_mm, 0.1))))
        for s in range(1, steps + 1):
            dense.append(pts[i] + seg * (s / steps))
    dense_arr = np.asarray(dense, dtype=float)

    pdir = np.asarray(projection_dir if projection_dir is not None else [0.0, 0.0, -1.0], dtype=float)
    pdir /= max(float(np.linalg.norm(pdir)), 1e-12)
    bmin, bmax = np.asarray(surface_mesh.bounds).reshape(3, 2).T
    span = float(np.linalg.norm(bmax - bmin)) * 2.5 + 20.0

    projected: list[np.ndarray] = []
    for pt in dense_arr:
        start = pt - pdir * span
        end = pt + pdir * span
        hit_pts, _ = surface_mesh.ray_trace(start, end)
        if len(hit_pts) > 0:
            # Choose hit closest to original point along projection axis
            idx = int(np.argmin(np.linalg.norm(hit_pts - pt, axis=1)))
            projected.append(hit_pts[idx])
        else:
            closest_id = surface_mesh.find_closest_point(pt)
            if closest_id >= 0:
                projected.append(np.asarray(surface_mesh.points[closest_id], dtype=float))
            else:
                projected.append(pt)
    return np.asarray(projected, dtype=float)


class SketchWorkflowManager:
    """Tracks sketch features, active sketch plane, snaps/constraints, dimensions, and DXF export."""

    def __init__(self) -> None:
        self.features: list[ParametricSketchFeature] = []
        self._snapshots: list[list[ParametricSketchFeature]] = []
        self._counter = 0
        self.active_plane: SketchPlane | None = None
        self.snap_settings = SnapSettings()
        self.sketches: dict[str, dict[str, Any]] = {}  # sketch id -> {"name", "plane"}
        self.active_owner: str | None = None
        self._sketch_counter = 0
        self.relations: list = []  # basicam.sketch_solver.Relation objects
        self.last_good = None  # basicam.sketch_solver.LastGoodStateStore (created by the window)

    # ---------------------------------------------------------------- ids and sketches
    def _next_number(self) -> int:
        """Entity numbers only grow, so ids are never reused (also across save/open)."""
        highest = self._counter
        for feat in self.features:
            tail = feat.feature_id.rsplit("_", 1)[-1]
            if tail.isdigit():
                highest = max(highest, int(tail))
        self._counter = highest + 1
        return self._counter

    def begin_sketch(self, plane_name: str) -> str:
        """Start a sketch locked to its support (plane, planar face or curved surface)."""
        self._sketch_counter = max(
            [self._sketch_counter] + [int(k[1:]) for k in self.sketches if k[1:].isdigit()]
        ) + 1
        sketch_id = f"S{self._sketch_counter:03d}"
        self.sketches[sketch_id] = {"name": f"Sketch {self._sketch_counter}", "plane": plane_name}
        self.active_owner = sketch_id
        return sketch_id

    def end_sketch(self) -> None:
        self.active_owner = None

    def can_edit(self, feat: ParametricSketchFeature) -> bool:
        """Entities can only be edited within their own sketch."""
        return self.active_owner is None or not feat.owner_sketch_id or feat.owner_sketch_id == self.active_owner

    def state_dict(self) -> dict[str, Any]:
        return {"counter": self._counter, "sketch_counter": self._sketch_counter, "sketches": dict(self.sketches)}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self._counter = max(int(data.get("counter", 0)), self._counter)
        self._sketch_counter = max(int(data.get("sketch_counter", 0)), self._sketch_counter)
        self.sketches = {str(k): dict(v) for k, v in dict(data.get("sketches", {})).items()}

    def set_construction(self, feature_id: str, construction: bool) -> bool:
        feat = self.get_feature(feature_id)
        if feat is None or feat.kind == "Point" or not self.can_edit(feat):
            return False
        self.save_snapshot()
        feat.construction = bool(construction)
        return True

    def set_entity_dimension(self, feature_id: str, value_mm: float, unit_text: str = "mm", sig_figs: int = 4) -> SketchDimension | None:
        """Create or edit the natural dimension of one entity (length, diameter or width)."""
        feat = self.get_feature(feature_id)
        if feat is None or not self.can_edit(feat) or value_mm <= 0:
            return None
        kind = {"Line": "Length", "Construction line": "Length", "Circle": "Diameter", "Rectangle": "Width"}.get(feat.kind)
        if kind is None:
            return None
        dim = next((d for d in feat.dimensions if d.kind == kind), None)
        if dim is None:
            dim = SketchDimension(f"{feat.feature_id}_{kind.lower()}", kind, kind, value_mm)
            feat.dimensions.append(dim)
        dim.target_ids = [feat.feature_id]
        dim.unit_text = unit_text
        dim.sig_figs = sig_figs
        dim.driving = True
        self.update_dimension_value(feat.feature_id, dim.dim_id, value_mm)
        return dim

    def set_pair_distance(self, id_a: str, id_b: str, value_mm: float, unit_text: str = "mm", sig_figs: int = 4) -> SketchDimension | None:
        """Distance between two entities' reference points; moves the second entity."""
        a, b = self.get_feature(id_a), self.get_feature(id_b)
        if a is None or b is None or a is b or value_mm < 0 or not (self.can_edit(a) and self.can_edit(b)):
            return None
        self.save_snapshot()
        pa, pb = np.asarray(a.points[0], dtype=float), np.asarray(b.points[0], dtype=float)
        direction = pb - pa
        length = float(np.linalg.norm(direction))
        direction = direction / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])
        delta = pa + direction * value_mm - pb
        b.points = np.asarray(b.points, dtype=float) + delta
        dim_id = f"{a.feature_id}_{b.feature_id}_dist"
        dim = next((d for d in b.dimensions if d.dim_id == dim_id), None)
        if dim is None:
            dim = SketchDimension(dim_id, f"Distance to {a.name}", "Distance", value_mm)
            b.dimensions.append(dim)
        dim.value = float(value_mm)
        dim.target_ids = [a.feature_id, b.feature_id]
        dim.unit_text = unit_text
        dim.sig_figs = sig_figs
        self._validate_feature(b)
        self._propagate_children(b.feature_id, delta_override=delta)
        return dim

    def has_errors(self) -> bool:
        return any(f.has_error for f in self.features)

    def blocking_errors(self) -> list[str]:
        return [f"{f.name}: {f.error_message}" for f in self.features if f.has_error]

    def can_revert(self) -> bool:
        return len(self._snapshots) > 0

    def save_snapshot(self) -> None:
        self._snapshots.append(copy.deepcopy(self.features))
        if len(self._snapshots) > 30:
            self._snapshots.pop(0)

    def revert_last(self) -> bool:
        if not self._snapshots:
            return False
        self.features = self._snapshots.pop()
        return True

    def clear(self) -> None:
        self.save_snapshot()
        self.features.clear()

    def get_feature(self, feature_id: str) -> ParametricSketchFeature | None:
        for feat in self.features:
            if feat.feature_id == feature_id:
                return feat
        return None

    def resolve_snap(
        self,
        raw_point: np.ndarray,
        *,
        anchor_point: np.ndarray | None = None,
        part_meshes: Iterable[pv.PolyData] = (),
    ) -> tuple[np.ndarray, SnapCandidate | None, list[str]]:
        """Resolves active snap targets and geometric constraints (H/V/Coincident/Center/Edge/Tangent)."""
        pt = np.asarray(raw_point, dtype=float)
        if self.active_plane is not None:
            pt = self.active_plane.project_point_to_plane(pt)

        tol = max(float(self.snap_settings.snap_radius_mm), 0.05)
        best: SnapCandidate | None = None
        best_dist = tol
        applied_constraints: list[str] = []

        # 1. Snap to existing sketch endpoints, midpoints, circle centers, and edges
        for feat in self.features:
            fpts = np.asarray(feat.points, dtype=float)
            if len(fpts) == 0:
                continue
            if self.snap_settings.endpoints or self.snap_settings.coincident:
                for ep in fpts:
                    proj_ep = self.active_plane.project_point_to_plane(ep) if self.active_plane else ep
                    d = float(np.linalg.norm(pt - proj_ep))
                    if d < best_dist:
                        best_dist = d
                        best = SnapCandidate(proj_ep, "Coincident", f"Coincident ({feat.name})")

            if self.snap_settings.centers and feat.kind in {"Circle", "Rectangle"} and len(fpts) >= 3:
                center = fpts[:-1].mean(axis=0)
                if self.active_plane:
                    center = self.active_plane.project_point_to_plane(center)
                d = float(np.linalg.norm(pt - center))
                if d < best_dist:
                    best_dist = d
                    best = SnapCandidate(center, "Center", f"Center ({feat.name})")

            if self.snap_settings.midpoints and len(fpts) >= 2:
                for i in range(len(fpts) - 1):
                    mid = 0.5 * (fpts[i] + fpts[i + 1])
                    if self.active_plane:
                        mid = self.active_plane.project_point_to_plane(mid)
                    d = float(np.linalg.norm(pt - mid))
                    if d < best_dist:
                        best_dist = d
                        best = SnapCandidate(mid, "Midpoint", f"Midpoint ({feat.name})")

            if self.snap_settings.tangent and feat.kind == "Circle" and anchor_point is not None and len(fpts) >= 3:
                center = fpts[:-1].mean(axis=0)
                radius = float(np.linalg.norm(fpts[0] - center))
                to_anchor = anchor_point - center
                dist_ac = float(np.linalg.norm(to_anchor))
                if dist_ac > radius + 1e-4 and self.active_plane is not None:
                    ua, va = self.active_plane.world_to_uv(anchor_point)
                    uc, vc = self.active_plane.world_to_uv(center)
                    base_ang = math.atan2(va - vc, ua - uc)
                    alpha = math.acos(min(max(radius / dist_ac, -1.0), 1.0))
                    for sign in (-1.0, 1.0):
                        t_pt = self.active_plane.uv_to_world(
                            uc + radius * math.cos(base_ang + sign * alpha),
                            vc + radius * math.sin(base_ang + sign * alpha),
                        )
                        d = float(np.linalg.norm(pt - t_pt))
                        if d < best_dist:
                            best_dist = d
                            best = SnapCandidate(t_pt, "Tangent", f"Tangent ({feat.name})")

        # 2. Snap to part vertices, edge midpoints, and round-feature arc centers
        if best is None and (self.snap_settings.edges or self.snap_settings.centers):
            for mesh in part_meshes:
                if mesh is None or mesh.n_points == 0:
                    continue
                cid = mesh.find_closest_point(pt)
                if cid >= 0:
                    v_pt = np.asarray(mesh.points[cid], dtype=float)
                    if self.active_plane:
                        v_pt = self.active_plane.project_point_to_plane(v_pt)
                    d = float(np.linalg.norm(pt - v_pt))
                    if d < best_dist:
                        best_dist = d
                        best = SnapCandidate(v_pt, "Edge", "Part vertex/edge")

            # Detect circular boundary loop center on active surface mesh
            if best is None and self.snap_settings.centers and self.active_plane and self.active_plane.surface_mesh is not None:
                surf = self.active_plane.surface_mesh
                if surf.n_points >= 6:
                    c_pt = self.active_plane.project_point_to_plane(np.asarray(surf.points).mean(axis=0))
                    d = float(np.linalg.norm(pt - c_pt))
                    if d < best_dist * 1.4:
                        best = SnapCandidate(c_pt, "Center", "Surface centerpoint")

        if best is not None:
            pt = best.point.copy()
            applied_constraints.append(best.kind)

        # 3. Horizontal / Vertical alignment relative to anchor_point on active plane
        if anchor_point is not None and self.active_plane is not None:
            u0, v0 = self.active_plane.world_to_uv(anchor_point)
            u1, v1 = self.active_plane.world_to_uv(pt)
            if self.snap_settings.horizontal and abs(v1 - v0) < tol * 0.75 and abs(u1 - u0) > tol:
                v1 = v0
                pt = self.active_plane.uv_to_world(u1, v1)
                applied_constraints.append("Horizontal")
            elif self.snap_settings.vertical and abs(u1 - u0) < tol * 0.75 and abs(v1 - v0) > tol:
                u1 = u0
                pt = self.active_plane.uv_to_world(u1, v1)
                applied_constraints.append("Vertical")

        return pt, best, applied_constraints

    def create_feature(
        self,
        kind: str,
        plane: str,
        points: np.ndarray,
        *,
        construction: bool = False,
        parent_feature_id: str | None = None,
        auto_dimension: bool = False,
        constraints: list[str] | None = None,
        variant: str = "",
    ) -> ParametricSketchFeature:
        self.save_snapshot()
        number = self._next_number()
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        fid = f"sk_{number:03d}"
        feat = ParametricSketchFeature(
            feature_id=fid,
            name=f"Sketch {number:02d} · {kind}",
            kind=kind,
            plane=plane,
            points=pts,
            construction=bool(construction) and kind != "Point",
            parent_feature_id=parent_feature_id,
            constraints=list(constraints or []),
            owner_sketch_id=self.active_owner or "",
            variant=variant,
        )
        # Auto-project onto curved surface when active_plane is a curved manifold region
        if (
            self.active_plane is not None
            and self.active_plane.is_curved_surface
            and self.active_plane.surface_mesh is not None
            and not construction
        ):
            feat.projected_points = project_points_onto_surface(
                pts,
                self.active_plane.surface_mesh,
                projection_dir=-self.active_plane.normal,
            )
            feat.is_projected = True

        if auto_dimension:
            self._populate_default_dimensions(feat)
            feat.defined = True
        self._validate_feature(feat)
        self.features.append(feat)
        return feat

    def project_feature_to_active_surface(self, feature_id: str) -> bool:
        feat = self.get_feature(feature_id)
        if feat is None or self.active_plane is None or self.active_plane.surface_mesh is None:
            return False
        self.save_snapshot()
        feat.projected_points = project_points_onto_surface(
            feat.points,
            self.active_plane.surface_mesh,
            projection_dir=-self.active_plane.normal,
        )
        feat.is_projected = True
        return True

    def _populate_default_dimensions(self, feat: ParametricSketchFeature) -> None:
        feat.dimensions.clear()
        p0 = feat.points[0]
        feat.dimensions.extend(
            [
                SketchDimension(f"{feat.feature_id}_x", "Origin X", "X", float(p0[0])),
                SketchDimension(f"{feat.feature_id}_y", "Origin Y", "Y", float(p0[1])),
                SketchDimension(f"{feat.feature_id}_z", "Origin Z", "Z", float(p0[2])),
            ]
        )
        if feat.kind in {"Line", "Construction line"} and len(feat.points) >= 2:
            length = float(np.linalg.norm(feat.points[1] - feat.points[0]))
            feat.dimensions.append(
                SketchDimension(f"{feat.feature_id}_len", "Length", "Length", max(length, 0.001))
            )
        elif feat.kind == "Circle" and len(feat.points) >= 4:
            center = feat.points[:-1].mean(axis=0)
            diam = float(np.linalg.norm(feat.points[0] - center)) * 2.0
            feat.dimensions.append(
                SketchDimension(f"{feat.feature_id}_dia", "Diameter", "Diameter", max(diam, 0.001))
            )
        elif feat.kind == "Rectangle" and len(feat.points) >= 4:
            w = float(np.linalg.norm(feat.points[1] - feat.points[0]))
            h = float(np.linalg.norm(feat.points[2] - feat.points[1]))
            feat.dimensions.append(
                SketchDimension(f"{feat.feature_id}_w", "Width", "Width", max(w, 0.001))
            )
            feat.dimensions.append(
                SketchDimension(f"{feat.feature_id}_h", "Height", "Height", max(h, 0.001))
            )

    def define_feature(self, feature_id: str, dimensions: list[SketchDimension] | None = None) -> bool:
        feat = self.get_feature(feature_id)
        if feat is None:
            return False
        self.save_snapshot()
        if dimensions is not None:
            feat.dimensions = dimensions
        elif not feat.dimensions:
            self._populate_default_dimensions(feat)
        self._apply_dimensions_to_geometry(feat)
        self._validate_feature(feat)
        if not feat.has_error:
            feat.defined = True
        self._propagate_children(feat.feature_id)
        return not feat.has_error

    def update_dimension_value(self, feature_id: str, dim_id: str, new_value: float) -> bool:
        feat = self.get_feature(feature_id)
        if feat is None or not self.can_edit(feat):
            return False
        self.save_snapshot()
        old_p0 = feat.points[0].copy()
        for dim in feat.dimensions:
            if dim.dim_id == dim_id:
                dim.value = float(new_value)
                break
        self._apply_dimensions_to_geometry(feat)
        self._validate_feature(feat)
        if not feat.has_error:
            feat.defined = True
            delta = feat.points[0] - old_p0
            self._propagate_children(feat.feature_id, delta_override=delta)
        return not feat.has_error

    def move_feature_point(self, feature_id: str, point_index: int, new_point: np.ndarray) -> bool:
        feat = self.get_feature(feature_id)
        if feat is None or not 0 <= point_index < len(feat.points) or not self.can_edit(feat):
            return False
        self.save_snapshot()
        old_p0 = feat.points[0].copy()
        pt = np.asarray(new_point, dtype=float)
        if self.active_plane is not None:
            pt = self.active_plane.project_point_to_plane(pt)
        feat.points[point_index] = pt
        if feat.is_projected and self.active_plane is not None and self.active_plane.surface_mesh is not None:
            feat.projected_points = project_points_onto_surface(
                feat.points,
                self.active_plane.surface_mesh,
                projection_dir=-self.active_plane.normal,
            )
        if feat.dimensions:
            self._sync_dimensions_from_geometry(feat)
        self._validate_feature(feat)
        delta = feat.points[0] - old_p0
        self._propagate_children(feat.feature_id, delta_override=delta)
        return not feat.has_error

    def _apply_dimensions_to_geometry(self, feat: ParametricSketchFeature) -> None:
        if len(feat.points) == 0:
            return
        old_p0 = feat.points[0].copy()
        target_p0 = old_p0.copy()
        target_len: float | None = None
        target_dia: float | None = None

        for dim in feat.dimensions:
            if dim.kind == "X":
                target_p0[0] = dim.value
            elif dim.kind == "Y":
                target_p0[1] = dim.value
            elif dim.kind == "Z":
                target_p0[2] = dim.value
            elif dim.kind == "Length":
                target_len = dim.value
            elif dim.kind == "Diameter":
                target_dia = dim.value

        shift = target_p0 - old_p0
        feat.points = feat.points + shift

        if target_len is not None and len(feat.points) >= 2:
            vec = feat.points[1] - feat.points[0]
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                vec = np.array([1.0, 0.0, 0.0])
                norm = 1.0
            feat.points[1] = feat.points[0] + (vec / norm) * target_len

        if target_dia is not None and feat.kind == "Circle" and len(feat.points) >= 4:
            center = feat.points[:-1].mean(axis=0)
            curr_r = float(np.linalg.norm(feat.points[0] - center))
            if curr_r > 1e-9:
                scale = (target_dia * 0.5) / curr_r
                feat.points = center + (feat.points - center) * scale

    def _sync_dimensions_from_geometry(self, feat: ParametricSketchFeature) -> None:
        p0 = feat.points[0]
        for dim in feat.dimensions:
            if dim.kind == "X":
                dim.value = float(p0[0])
            elif dim.kind == "Y":
                dim.value = float(p0[1])
            elif dim.kind == "Z":
                dim.value = float(p0[2])
            elif dim.kind == "Length" and len(feat.points) >= 2:
                dim.value = float(np.linalg.norm(feat.points[1] - feat.points[0]))

    def _validate_feature(self, feat: ParametricSketchFeature) -> None:
        feat.has_error = False
        feat.error_message = ""
        if feat.points.size == 0 or not np.all(np.isfinite(feat.points)):
            feat.has_error = True
            feat.error_message = "Non-finite or empty coordinates"
            return

        if feat.parent_feature_id:
            parent = self.get_feature(feat.parent_feature_id)
            if parent is None:
                feat.has_error = True
                feat.error_message = f"Missing parent reference ({feat.parent_feature_id})"
                return
            if parent.has_error:
                feat.has_error = True
                feat.error_message = f"Parent '{parent.name}' has an unresolved error"
                return

        for dim in feat.dimensions:
            dim.valid = True
            dim.error_message = ""
            if not math.isfinite(dim.value):
                dim.valid = False
                dim.error_message = "Invalid numeric value"
                feat.has_error = True
                feat.error_message = f"{dim.name}: invalid value"
                return
            if dim.kind in {"Length", "Diameter", "Width", "Height"} and dim.value <= 0.0:
                dim.valid = False
                dim.error_message = f"{dim.kind} must be > 0"
                feat.has_error = True
                feat.error_message = f"{dim.name} must be > 0 mm"
                return

        if feat.kind in {"Line", "Construction line"} and len(feat.points) >= 2:
            seg_len = float(np.linalg.norm(feat.points[1] - feat.points[0]))
            if seg_len < 1e-6:
                feat.has_error = True
                feat.error_message = "Zero-length line segment"
                return

    def _propagate_children(
        self, parent_id: str, delta_override: np.ndarray | None = None
    ) -> None:
        for child in self.features:
            if child.parent_feature_id != parent_id:
                continue
            if delta_override is not None and np.linalg.norm(delta_override) > 0:
                child.points = child.points + delta_override
                if child.dimensions:
                    self._sync_dimensions_from_geometry(child)
            self._validate_feature(child)
            self._propagate_children(child.feature_id, delta_override=delta_override)

    def sync_tooling_assignments(self, setups: Iterable[MachiningSetup]) -> None:
        """Links sketch features to assigned setup operations and updates their setup colors."""
        for feat in self.features:
            feat.assigned_operation_name = None
            feat.assigned_setup_name = None
            feat.assigned_setup_color = None

        for setup in setups:
            for op in setup.operations:
                for feat in self.features:
                    if op.target == feat.name or op.target.startswith(f"{feat.name} ·"):
                        feat.assigned_operation_name = op.name
                        feat.assigned_setup_name = setup.name
                        feat.assigned_setup_color = setup.color


def export_sketches_to_dxf(
    features: Iterable[ParametricSketchFeature],
    output_path: str | Path,
    *,
    include_construction: bool = False,
) -> int:
    """Exports sketch features (including projected 3D splines) to an ASCII R12-compatible DXF file."""
    exported_count = 0
    lines: list[str] = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",  # millimeters
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "TABLES",
        "0", "TABLE",
        "2", "LAYER",
        "70", "3",
        "0", "LAYER", "2", "SKETCH_DEFINED", "70", "0", "62", "7", "6", "CONTINUOUS",
        "0", "LAYER", "2", "SKETCH_UNDERDEFINED", "70", "0", "62", "5", "6", "CONTINUOUS",
        "0", "LAYER", "2", "SKETCH_CONSTRUCTION", "70", "0", "62", "8", "6", "DASHED",
        "0", "ENDTAB",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]

    for feat in features:
        if feat.has_error:
            continue
        if feat.construction and not include_construction:
            continue
        pts = np.asarray(feat.active_points, dtype=float)
        if pts.size == 0:
            continue

        if feat.construction:
            layer = "SKETCH_CONSTRUCTION"
        elif feat.defined or feat.assigned_setup_color:
            layer = "SKETCH_DEFINED"
        else:
            layer = "SKETCH_UNDERDEFINED"

        if feat.kind == "Point" or len(pts) == 1:
            p = pts[0]
            lines.extend([
                "0", "POINT",
                "8", layer,
                "10", f"{p[0]:.6f}",
                "20", f"{p[1]:.6f}",
                "30", f"{p[2]:.6f}",
            ])
            exported_count += 1
        else:
            for idx in range(len(pts) - 1):
                p1, p2 = pts[idx], pts[idx + 1]
                lines.extend([
                    "0", "LINE",
                    "8", layer,
                    "10", f"{p1[0]:.6f}",
                    "20", f"{p1[1]:.6f}",
                    "30", f"{p1[2]:.6f}",
                    "11", f"{p2[0]:.6f}",
                    "21", f"{p2[1]:.6f}",
                    "31", f"{p2[2]:.6f}",
                ])
                exported_count += 1

    lines.extend(["0", "ENDSEC", "0", "EOF", ""])
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return exported_count
