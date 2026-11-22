"""Part positioning: translate, rotate, align, and work-frame construction.

Pure functions that build new geometry (meshes and Open CASCADE shapes are copied,
never edited in place) so undo snapshots that reference the old objects stay valid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pyvista as pv

Vec = Sequence[float]


# ---------------------------------------------------------------------- matrices
def matrix_translate(delta: Vec) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(delta, dtype=float)
    return m


def _rodrigues(axis: Vec, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return np.eye(3)
    a = a / norm
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def matrix_rotate(axis: Vec, angle_deg: float, pivot: Vec = (0.0, 0.0, 0.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rodrigues(axis, math.radians(angle_deg))
    p = np.asarray(pivot, dtype=float)
    m[:3, 3] = p - m[:3, :3] @ p
    return m


def rotation_between(a: Vec, b: Vec) -> np.ndarray:
    """3x3 rotation taking direction a onto direction b."""
    a = np.asarray(a, dtype=float) / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(b, dtype=float) / max(np.linalg.norm(b), 1e-12)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return _rodrigues(np.cross(a, helper), math.pi)
    return _rodrigues(cross, math.acos(dot))


def apply_to_point(matrix: np.ndarray, point: Vec) -> np.ndarray:
    p = np.append(np.asarray(point, dtype=float), 1.0)
    return (matrix @ p)[:3]


def apply_to_points(matrix: np.ndarray, points) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts @ matrix[:3, :3].T + matrix[:3, 3]


# ---------------------------------------------------------------------- geometry
def transform_mesh(mesh: pv.PolyData | None, matrix: np.ndarray) -> pv.PolyData | None:
    if mesh is None:
        return None
    return mesh.transform(matrix, inplace=False)


def transform_shape(shape: Any, matrix: np.ndarray) -> Any:
    """Copy an Open CASCADE shape with a rigid transform applied."""
    if shape is None:
        return None
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    trsf = gp_Trsf()
    m = matrix
    trsf.SetValues(m[0, 0], m[0, 1], m[0, 2], m[0, 3], m[1, 0], m[1, 1], m[1, 2], m[1, 3], m[2, 0], m[2, 1], m[2, 2], m[2, 3])
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


@dataclass
class TransformResult:
    matrix: np.ndarray
    mesh: pv.PolyData | None
    edge_mesh: pv.PolyData | None
    brep_shape: Any
    solid_shape: Any


def transform_part(part, matrix: np.ndarray) -> TransformResult:
    brep = transform_shape(getattr(part, "brep_shape", None), matrix)
    solid_src = getattr(part, "solid_shape", None)
    solid = brep if solid_src is not None and solid_src is getattr(part, "brep_shape", None) else transform_shape(solid_src, matrix)
    return TransformResult(
        matrix=matrix,
        mesh=transform_mesh(part.mesh, matrix),
        edge_mesh=transform_mesh(getattr(part, "edge_mesh", None), matrix),
        brep_shape=brep,
        solid_shape=solid,
    )


def part_center(part) -> np.ndarray:
    b = np.asarray(part.mesh.bounds, dtype=float)
    return np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])


def translate_part(part, delta: Vec) -> TransformResult:
    return transform_part(part, matrix_translate(delta))


def translate_part_to(part, target: Vec, reference: Vec | None = None) -> TransformResult:
    """Absolute move: the reference point (default part centre) ends at `target`."""
    ref = np.asarray(reference if reference is not None else part_center(part), dtype=float)
    return translate_part(part, np.asarray(target, dtype=float) - ref)


def rotate_part(part, axis: Vec, angle_deg: float, pivot: Vec | None = None) -> TransformResult:
    pivot = part_center(part) if pivot is None else pivot
    return transform_part(part, matrix_rotate(axis, angle_deg, pivot))


def align_faces_matrix(face_point: Vec, face_normal: Vec, target_point: Vec, target_normal: Vec, offset: float = 0.0, flip: bool = False) -> np.ndarray:
    """Face to face: the moving face ends coplanar with the target face, normals opposed (touching)."""
    goal = -np.asarray(target_normal, dtype=float) if not flip else np.asarray(target_normal, dtype=float)
    rot = rotation_between(face_normal, goal)
    p = np.asarray(face_point, dtype=float)
    tn = np.asarray(target_normal, dtype=float) / max(np.linalg.norm(target_normal), 1e-12)
    # rotate about the face point, then move along the target normal onto its plane (+ offset)
    distance = float(np.dot(np.asarray(target_point, dtype=float) - p, tn)) + offset
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = p - rot @ p + tn * distance
    return m


def align_faces(part, face_point, face_normal, target_point, target_normal, offset: float = 0.0, flip: bool = False) -> TransformResult:
    return transform_part(part, align_faces_matrix(face_point, face_normal, target_point, target_normal, offset, flip))


def align_to_plane(part, face_point, face_normal, plane_point, plane_normal, offset: float = 0.0, flip: bool = False) -> TransformResult:
    """Face to plane: the face lies on the plane (its normal opposite the plane normal unless flipped)."""
    return align_faces(part, face_point, face_normal, plane_point, plane_normal, offset, flip)


def align_vertices(part, vertex: Vec, target: Vec, offset: Vec = (0.0, 0.0, 0.0)) -> TransformResult:
    """Vertex to vertex (or reference point to coordinates / to a sketched point)."""
    delta = np.asarray(target, dtype=float) + np.asarray(offset, dtype=float) - np.asarray(vertex, dtype=float)
    return translate_part(part, delta)


# ---------------------------------------------------------------------- work frame
@dataclass
class Frame:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray

    def matrix(self) -> np.ndarray:
        """Frame -> world."""
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = self.x_axis, self.y_axis, self.z_axis, self.origin
        return m

    def to_frame(self, point: Vec) -> np.ndarray:
        d = np.asarray(point, dtype=float) - self.origin
        return np.array([np.dot(d, self.x_axis), np.dot(d, self.y_axis), np.dot(d, self.z_axis)])

    def to_dict(self) -> dict[str, list[float]]:
        return {k: [float(v) for v in getattr(self, k)] for k in ("origin", "x_axis", "y_axis", "z_axis")}

    @classmethod
    def from_dict(cls, data: dict | None) -> "Frame | None":
        if not data:
            return None
        return cls(*(np.asarray(data[k], dtype=float) for k in ("origin", "x_axis", "y_axis", "z_axis")))


def frame_from_origin_and_edge(origin: Vec, edge_direction: Vec, normal_hint: Vec = (0.0, 0.0, 1.0), flip: bool = False) -> Frame:
    """X along the picked edge, Z as close to the hint as possible, Y completes a right-handed frame."""
    x = np.asarray(edge_direction, dtype=float)
    x = x / max(np.linalg.norm(x), 1e-12)
    if flip:
        x = -x
    hint = np.asarray(normal_hint, dtype=float)
    z = hint - np.dot(hint, x) * x
    if np.linalg.norm(z) < 1e-9:
        alt = np.array([0.0, 1.0, 0.0]) if abs(x[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        z = alt - np.dot(alt, x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return Frame(np.asarray(origin, dtype=float), x, y, z)
