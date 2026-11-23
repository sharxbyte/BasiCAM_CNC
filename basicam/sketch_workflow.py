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
    # "Coincident", "Center", "Midpoint", "Tangent", "OnCurve", "ModelVertex", "ModelEdge", "Edge"
    kind: str
    label: str = ""
    entity_id: str = ""  # sketch entity snapped to
    part_index: int | None = None  # model snaps: the part
    segment: np.ndarray | None = None  # ModelEdge: the edge segment (2, 3) in world coordinates


def _snap_points_of(feat) -> list[np.ndarray]:
    """End and corner points a cursor can snap to (centres are separate)."""
    pts = np.asarray(feat.points, dtype=float)
    if len(pts) == 0:
        return []
    kind = feat.kind
    control = getattr(feat, "control_points", None)
    if kind == "Point" or len(pts) == 1:
        return [pts[0]]
    if kind == "Rectangle":
        return [pts[i] for i in range(min(4, len(pts)))]
    if kind == "Polygon" and control is not None:
        return [np.asarray(c, dtype=float) for c in control[1:]]
    if kind in ("Circle", "Slot"):
        return []
    return [pts[0], pts[-1]]


def _centres_of(feat) -> list[np.ndarray]:
    pts = np.asarray(feat.points, dtype=float)
    control = getattr(feat, "control_points", None)
    if feat.kind == "Circle" and len(pts) >= 4:
        return [pts[:-1].mean(axis=0)]
    if feat.kind == "Polygon" and control is not None:
        return [np.asarray(control[0], dtype=float)]
    if feat.kind == "Slot" and control is not None:
        return [np.asarray(control[0], dtype=float), np.asarray(control[1], dtype=float)]
    if feat.kind == "Curve" and len(pts) >= 3:
        a, m, b = pts[0], pts[len(pts) // 2], pts[-1]
        ab, ac = m - a, b - a
        normal = np.cross(ab, ac)
        denom = 2.0 * float(np.dot(normal, normal))
        if denom < 1e-18:
            return []
        centre = a + (np.cross(normal, ab) * float(np.dot(ac, ac)) + np.cross(ac, normal) * float(np.dot(ab, ab))) / denom
        return [centre]
    return []


def _closest_on_segments(point: np.ndarray, segments: np.ndarray):
    """(closest point, its segment) on segments (M, 2, 3), or None."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if len(segments) == 0:
        return None
    a, b = segments[:, 0], segments[:, 1]
    ab = b - a
    lengths = np.einsum("ij,ij->i", ab, ab)
    t = np.where(lengths > 1e-18, np.einsum("ij,ij->i", np.asarray(point) - a, ab) / np.where(lengths > 1e-18, lengths, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    feet = a + ab * t[:, None]
    index = int(np.argmin(np.linalg.norm(feet - point, axis=1)))
    return feet[index], segments[index]


@dataclass
class SketchDimension:
    """Parametric dimension / constraint driving a sketch feature."""

    dim_id: str
    name: str
    # "X", "Y", "Z" (from the origin), "Length", "Offset", "Diameter", "Width", "Height",
    # "Distance" (two entities), "Angle" (two lines, degrees), "Radius" (circle/arc),
    # "PointLine" (point to line, perpendicular), "HDistance" / "VDistance" (two points along the
    # sketch u / v axis). target_ids name the entities; point_refs the point index on each target.
    kind: str
    value: float
    tolerance: float = 0.0
    valid: bool = True
    error_message: str = ""
    target_ids: list[str] = field(default_factory=list)  # entity ids this dimension is bound to
    driving: bool = True  # driving dimensions set geometry; driven ones only report it
    unit_text: str = "mm"  # unit the user typed (for display)
    sig_figs: int = 4  # significant figures of the typed value
    point_refs: list[int] = field(default_factory=list)  # point index per target id (default 0)
    label_offset: list[float] = field(default_factory=list)  # (u, v) of the label from its anchor


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
    # Defining points of composite entities (world xyz): Slot = [centre 1, centre 2, arc 1 start];
    # Polygon = [centre, vertex 0, ..., vertex n-1]. `points` stays the drawn outline.
    control_points: np.ndarray | None = None

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


def slot_outline_uv(c1, c2, a1s, segments: int = 16) -> tuple[list, list]:
    """Slot from its two arc centres and arc 1's start point (all uv).

    Returns (solver points [c1, c2, a1s, a1e, a2s, a2e], closed outline uv). Arc 1 runs CCW from
    a1s = c1 + n*r to a1e = c1 - n*r around the far end of c1; arc 2 from c2 - n*r to c2 + n*r.
    """
    c1, c2, a1s = (np.asarray(p, dtype=float)[:2] for p in (c1, c2, a1s))
    d = c2 - c1
    length = float(np.linalg.norm(d))
    d = d / length if length > 1e-12 else np.array([1.0, 0.0])
    n = np.array([-d[1], d[0]])
    r = max(float(np.linalg.norm(a1s - c1)), 1e-6)
    pts = [c1, c2, c1 + n * r, c1 - n * r, c2 - n * r, c2 + n * r]
    a_n = math.atan2(n[1], n[0])
    outline = [c1 + r * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(a_n, a_n + math.pi, segments + 1)]
    outline += [c2 + r * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(a_n + math.pi, a_n + 2 * math.pi, segments + 1)]
    outline.append(outline[0])
    return [tuple(p) for p in pts], [tuple(p) for p in outline]


def sample_slot(c1: np.ndarray, c2: np.ndarray, width: float, plane: SketchPlane) -> tuple[np.ndarray, np.ndarray]:
    """(control_points, outline) of a straight slot with arc centres c1, c2 and the given width."""
    u1, u2 = np.asarray(plane.world_to_uv(c1)), np.asarray(plane.world_to_uv(c2))
    d = u2 - u1
    length = float(np.linalg.norm(d))
    d = d / length if length > 1e-12 else np.array([1.0, 0.0])
    a1s = u1 + np.array([-d[1], d[0]]) * (width / 2.0)
    pts, outline = slot_outline_uv(u1, u2, a1s)
    control = np.asarray([plane.uv_to_world(*pts[i]) for i in (0, 1, 2)], dtype=float)
    return control, np.asarray([plane.uv_to_world(u, v) for u, v in outline], dtype=float)


def sample_polygon(centre: np.ndarray, vertex: np.ndarray, sides: int, plane: SketchPlane) -> tuple[np.ndarray, np.ndarray]:
    """(control_points [centre, vertices...], closed outline) of a regular polygon (inscribed)."""
    sides = max(3, int(sides))
    cu, cv = plane.world_to_uv(centre)
    vu, vv = plane.world_to_uv(vertex)
    r = math.hypot(vu - cu, vv - cv) or 1.0
    a0 = math.atan2(vv - cv, vu - cu)
    verts = [plane.uv_to_world(cu + r * math.cos(a0 + 2 * math.pi * i / sides), cv + r * math.sin(a0 + 2 * math.pi * i / sides)) for i in range(sides)]
    control = np.asarray([plane.uv_to_world(cu, cv)] + verts, dtype=float)
    return control, np.asarray(verts + [verts[0]], dtype=float)


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
        tolerance: float | None = None,
        model_vertices: Iterable[tuple[int, np.ndarray]] = (),
        model_edges: Iterable[tuple[int, np.ndarray]] = (),
        exclude_ids: Iterable[str] = (),
    ) -> tuple[np.ndarray, SnapCandidate | None, list[str]]:
        """Snap a cursor point (sketch-engine.spec B5).

        Priority, nearest within ``tolerance`` (mm; the window passes its pixel radius converted
        at the cursor) inside each group: sketch points and centres > model vertices > midpoints
        > tangent points > model edges and sketch curves. Then horizontal / vertical inference
        from ``anchor_point``. ``model_vertices`` / ``model_edges`` are (part index, points) and
        (part index, segments of shape (M, 2, 3)) in the part's current position. A model vertex
        keeps its own 3D coordinates (the point is defined there).
        """
        pt = np.asarray(raw_point, dtype=float)
        plane = self.active_plane
        if plane is not None:
            pt = plane.project_point_to_plane(pt)

        def project(p):
            return plane.project_point_to_plane(p) if plane is not None else np.asarray(p, dtype=float)

        tol = max(float(tolerance if tolerance is not None else self.snap_settings.snap_radius_mm), 1e-4)
        settings = self.snap_settings
        excluded = set(exclude_ids)
        model_vertices = list(model_vertices)
        groups: dict[int, tuple[float, SnapCandidate]] = {}

        def offer(priority: int, point, candidate: SnapCandidate, *, on_plane: bool = True) -> None:
            point = project(point) if on_plane else np.asarray(point, dtype=float)
            distance = float(np.linalg.norm(project(point) - pt))
            if distance <= tol and (priority not in groups or distance < groups[priority][0]):
                candidate.point = point
                groups[priority] = (distance, candidate)

        for feat in self.features:
            if feat.feature_id in excluded:
                continue
            fpts = np.asarray(feat.points, dtype=float)
            if len(fpts) == 0:
                continue
            closed = feat.kind in {"Circle", "Rectangle", "Slot", "Polygon"}
            if settings.endpoints or settings.coincident:
                for ep in _snap_points_of(feat):
                    offer(0, ep, SnapCandidate(ep, "Coincident", f"Coincident ({feat.name})", feat.feature_id))
            if settings.centers:
                for center in _centres_of(feat):
                    offer(0, center, SnapCandidate(center, "Center", f"Center ({feat.name})", feat.feature_id))
            if settings.midpoints and len(fpts) >= 2 and feat.kind in {"Line", "Construction line", "Rectangle", "Polygon"}:
                for i in range(len(fpts) - 1):
                    mid = 0.5 * (fpts[i] + fpts[i + 1])
                    offer(2, mid, SnapCandidate(mid, "Midpoint", f"Midpoint ({feat.name})", feat.feature_id))
            if settings.tangent and feat.kind == "Circle" and anchor_point is not None and len(fpts) >= 3 and plane is not None:
                center = fpts[:-1].mean(axis=0)
                radius = float(np.linalg.norm(fpts[0] - center))
                dist_ac = float(np.linalg.norm(anchor_point - center))
                if dist_ac > radius + 1e-4:
                    ua, va = plane.world_to_uv(anchor_point)
                    uc, vc = plane.world_to_uv(center)
                    base_ang = math.atan2(va - vc, ua - uc)
                    alpha = math.acos(min(max(radius / dist_ac, -1.0), 1.0))
                    for sign in (-1.0, 1.0):
                        t_pt = plane.uv_to_world(
                            uc + radius * math.cos(base_ang + sign * alpha), vc + radius * math.sin(base_ang + sign * alpha)
                        )
                        offer(3, t_pt, SnapCandidate(t_pt, "Tangent", f"Tangent ({feat.name})", feat.feature_id))
            if settings.edges and len(fpts) >= 2 and feat.kind != "Point":
                segments = np.stack((fpts[:-1], fpts[1:]), axis=1)
                if closed and not np.allclose(fpts[0], fpts[-1]):
                    segments = np.vstack((segments, [[fpts[-1], fpts[0]]]))
                hit = _closest_on_segments(pt, segments)
                if hit is not None:
                    offer(4, hit[0], SnapCandidate(hit[0], "OnCurve", f"On {feat.name}", feat.feature_id))

        if settings.edges or settings.endpoints:
            for part_index, verts in model_vertices:
                verts = np.asarray(verts, dtype=float).reshape(-1, 3)
                if len(verts) == 0:
                    continue
                flat = np.asarray([project(v) for v in verts]) if plane is not None else verts
                along = np.linalg.norm(flat - pt, axis=1)
                off_plane = np.linalg.norm(verts - flat, axis=1)
                # Vertices that line up on screen (top and bottom corner): prefer the one on the plane.
                nearest = int(np.lexsort((off_plane, np.round(along, 6)))[0])
                vertex = verts[nearest]
                offer(1, vertex, SnapCandidate(vertex, "ModelVertex", "Model vertex", "", part_index), on_plane=False)
        if settings.edges:
            for part_index, segments in model_edges:
                segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
                hit = _closest_on_segments(pt, segments)
                if hit is not None:
                    point, segment = hit
                    offer(4, point, SnapCandidate(point, "ModelEdge", "Model edge", "", part_index, segment))
        # Older callers: plain part meshes (vertex snap only).
        if not groups and not model_vertices and (settings.edges or settings.centers):
            for mesh in part_meshes:
                if mesh is None or mesh.n_points == 0:
                    continue
                cid = mesh.find_closest_point(pt)
                if cid >= 0:
                    v_pt = np.asarray(mesh.points[cid], dtype=float)
                    offer(1, v_pt, SnapCandidate(v_pt, "Edge", "Part vertex/edge"))

        best: SnapCandidate | None = groups[min(groups)][1] if groups else None
        applied_constraints: list[str] = []
        if best is not None:
            pt = np.asarray(best.point, dtype=float).copy()
            applied_constraints.append(best.kind)

        # Horizontal / vertical alignment relative to anchor_point (free cursor only).
        if anchor_point is not None and plane is not None and best is None:
            u0, v0 = plane.world_to_uv(anchor_point)
            u1, v1 = plane.world_to_uv(pt)
            if settings.horizontal and abs(v1 - v0) < tol * 0.75 and abs(u1 - u0) > tol:
                pt = plane.uv_to_world(u1, v0)
                applied_constraints.append("Horizontal")
            elif settings.vertical and abs(u1 - u0) < tol * 0.75 and abs(v1 - v0) > tol:
                pt = plane.uv_to_world(u0, v1)
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
        control_points: np.ndarray | None = None,
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
            control_points=None if control_points is None else np.asarray(control_points, dtype=float).reshape(-1, 3),
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
            if dim.kind in {"Length", "Diameter", "Width", "Height", "Radius"} and dim.value <= 0.0:
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
