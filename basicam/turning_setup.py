"""Turning setup (no Qt): stock, the turning axis, off-centre (eccentric) turning and the
part's half profile.

The picked turning axis (a cylindrical face's axis, an edge, or a typed point and direction) is
placed on the spindle centre line: world X is the spindle axis, X = 0 is the chuck / collet face,
the work extends towards +X. The side view is the XZ plane (Z = radius). Off-centre turning
offsets the picked axis from the stock's own centre; a 4-jaw independent chuck then needs a
different offset on each jaw, and the stock plus the offset must still turn inside the throw.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np


@dataclass
class TurningSetup:
    stock_kind: str = "bar"  # "bar" (round / hex / square), "tube" or "blank"
    stock_diameter_mm: float = 30.0
    stock_length_mm: float = 100.0
    stick_out_mm: float = 60.0  # from the jaw / collet face
    axis_point: tuple[float, float, float] = (0.0, 0.0, 0.0)  # part coordinates
    axis_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    eccentric_mm: tuple[float, float] = (0.0, 0.0)  # picked axis offset from the stock centre (across the axis)
    jaw_count: int = 3
    jaw_spacing_deg: float = 0.0
    jaw_width_mm: float = 12.0
    jaw_height_mm: float = 18.0
    jaw_step_mm: float = 0.0
    grip_diameter_mm: float = 30.0
    inside_clamp: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "TurningSetup":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        for key in ("axis_point", "axis_direction", "eccentric_mm"):
            if key in values:
                values[key] = tuple(float(v) for v in values[key])
        return cls(**values)


def axis_from_cylinder(points, normals) -> tuple[np.ndarray, np.ndarray]:
    """Axis of a cylindrical face from its points and normals: the direction is the one the
    normals never point along; the point is the least-squares centre of the normal lines."""
    n = np.asarray(normals, dtype=float)
    p = np.asarray(points, dtype=float)
    _u, _s, vh = np.linalg.svd(n - n.mean(axis=0) * 0.0)
    direction = vh[-1] / np.linalg.norm(vh[-1])
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    # least squares: the point c minimising the distance to every normal line p + t n
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for point, normal in zip(p, n):
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        m = np.eye(3) - np.outer(normal, normal) - np.outer(direction, direction)
        a += m
        b += m @ point
    centre = np.linalg.lstsq(a, b, rcond=None)[0]
    centre = centre - direction * float(np.dot(centre - p.mean(axis=0), direction))
    return centre, direction


def longest_axis(bounds) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    spans = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    direction = np.zeros(3)
    direction[int(np.argmax(spans))] = 1.0
    return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]), direction


def placement_matrix(axis_point, axis_direction, face_point_along: float = 0.0, eccentric=(0.0, 0.0)) -> np.ndarray:
    """4x4 matrix putting the picked axis on the spindle line (world X), with the part's end
    ``face_point_along`` (distance along the axis from axis_point) at X = 0."""
    d = np.asarray(axis_direction, dtype=float)
    d = d / np.linalg.norm(d)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(d, helper)
    v /= np.linalg.norm(v)
    w = np.cross(d, v)
    rot = np.vstack((d, v, w))  # rows: new X, Y, Z
    m = np.eye(4)
    m[:3, :3] = rot
    origin = np.asarray(axis_point, dtype=float) + d * face_point_along
    m[:3, 3] = -rot @ origin
    return m


def jaw_offsets(eccentric, jaw_angles_deg: list[float]) -> list[float]:
    """Radial adjustment of each jaw (mm, + = jaw moves out) to hold the stock off-centre by
    ``eccentric`` (v, w): an independent 4-jaw chuck."""
    ev, ew = (float(c) for c in eccentric)
    out = []
    for angle in jaw_angles_deg:
        a = math.radians(angle)
        out.append(-(ev * math.cos(a) + ew * math.sin(a)))  # the jaw on the far side moves out
    return out


def throw_problems(stock_diameter: float, eccentric, throw_mm: float, clamp_min: float, clamp_max: float) -> list[str]:
    offset = math.hypot(*eccentric)
    problems = []
    swing = stock_diameter + 2.0 * offset
    if swing > throw_mm + 1e-9:
        problems.append(f"Off-centre by {offset:.3f} mm the stock swings {swing:.2f} mm, more than the {throw_mm:g} mm throw.")
    if not (clamp_min - 1e-9 <= stock_diameter <= clamp_max + 1e-9):
        problems.append(f"The chuck grips {clamp_min:g}-{clamp_max:g} mm, not {stock_diameter:g} mm.")
    return problems


def half_profile(mesh, samples: int = 200) -> list[tuple[float, float]]:
    """Revolved silhouette of a part already on the spindle line (world X): the outer radius at
    each X, closed down to the axis at both ends. Grooves and cones keep their shape."""
    points = np.asarray(mesh.points, dtype=float)
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    breaks = np.unique(np.round(points[:, 0], 6))  # profile corners sit at mesh vertex positions
    if len(breaks) > 4 * samples:
        breaks = breaks[:: int(math.ceil(len(breaks) / (4 * samples)))]
    xs = np.unique(np.concatenate((np.linspace(x0, x1, samples + 1), breaks)))
    radii = []
    for x in xs:
        cut = mesh.slice(normal="x", origin=(float(np.clip(x, x0 + 1e-5, x1 - 1e-5)), 0.0, 0.0))
        if cut.n_points:
            pts = np.asarray(cut.points)
            radii.append(float(np.max(np.hypot(pts[:, 1], pts[:, 2]))))
        else:
            radii.append(radii[-1] if radii else 0.0)
    profile = [(float(xs[0]), 0.0)]
    for x, r in zip(xs, radii):
        profile.append((float(x), float(r)))
    profile.append((float(xs[-1]), 0.0))
    return _simplify(profile, 0.005)


def _simplify(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    from shapely.geometry import LineString

    line = LineString(points).simplify(tolerance, preserve_topology=False)
    return [(float(x), float(y)) for x, y in line.coords]


__all__ = [
    "TurningSetup", "axis_from_cylinder", "half_profile", "jaw_offsets", "longest_axis", "placement_matrix", "throw_problems",
]
