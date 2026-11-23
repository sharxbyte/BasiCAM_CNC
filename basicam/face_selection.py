"""Box selection by whole faces (no Qt).

A face is the set of cells sharing one ``feature_id`` (a B-rep face, or a planar facet group
for meshes). Window selection (dragged left to right) takes a face only when every cell of it
is inside the box; crossing selection (right to left) takes every face with a touched cell.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pyvista as pv


def project_points_to_display(points, camera, aspect: float, viewport, size) -> np.ndarray:
    """World points (N, 3) -> display (x, y, depth), matching vtkRenderer.WorldToDisplay.

    Depth is in the 0..1 range used by ``vtkRenderer.GetZ``.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    vtk_matrix = camera.GetCompositeProjectionTransformMatrix(aspect, 0.0, 1.0)
    matrix = np.array([[vtk_matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
    homogeneous = np.column_stack((pts, np.ones(len(pts)))) @ matrix.T
    w = homogeneous[:, 3:4]
    w = np.where(np.abs(w) < 1e-300, 1e-300, w)
    view = homogeneous[:, :3] / w
    width, height = float(size[0]), float(size[1])
    vx0, vy0, vx1, vy1 = (float(v) for v in viewport)
    out = np.empty_like(view)
    out[:, 0] = (view[:, 0] + 1.0) * (width * (vx1 - vx0)) / 2.0 + width * vx0
    out[:, 1] = (view[:, 1] + 1.0) * (height * (vy1 - vy0)) / 2.0 + height * vy0
    out[:, 2] = view[:, 2]
    return out


def _cell_connectivity(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (offsets, connectivity) with offsets of length n_cells + 1, in cell-id order."""
    if isinstance(mesh, pv.PolyData) and mesh.n_cells and mesh.is_all_triangles and not (
        mesh.n_verts or mesh.n_lines or getattr(mesh, "n_strips", 0)
    ):
        tris = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
        offsets = np.arange(0, 3 * len(tris) + 1, 3, dtype=np.int64)
        return offsets, tris.ravel()
    grid = mesh if isinstance(mesh, pv.UnstructuredGrid) else mesh.cast_to_unstructured_grid()
    return np.asarray(grid.offset, dtype=np.int64), np.asarray(grid.cell_connectivity, dtype=np.int64)


def box_select_faces(
    mesh,
    project: Callable[[np.ndarray], np.ndarray],
    box: tuple[float, float, float, float],
    height: float,
    contained: bool,
    get_z: Callable[[int, int], float] | None = None,
    depth_tolerance: float = 0.015,
    width: float | None = None,
) -> dict[int, list[int]]:
    """Faces hit by a screen box: {feature_id: [all cell ids of that face]}.

    ``project`` maps world points to display coordinates (origin bottom-left); ``box`` is
    (left, right, top, bottom) in top-left screen pixels; ``height`` is the window height.
    ``get_z`` (renderer.GetZ) enables visible-only selection; None selects through.
    """
    if mesh is None or mesh.n_cells == 0 or mesh.n_points == 0:
        return {}
    left, right, top, bottom = box
    screen = project(np.asarray(mesh.points))
    sx = screen[:, 0]
    sy = height - screen[:, 1]
    sz = screen[:, 2]
    offsets, conn = _cell_connectivity(mesh)
    n_cells = len(offsets) - 1
    counts = np.diff(offsets)
    valid = counts > 0
    starts = offsets[:-1][valid]
    cx, cy, cz = sx[conn], sy[conn], sz[conn]
    minx = np.full(n_cells, np.inf)
    maxx = np.full(n_cells, -np.inf)
    miny = np.full(n_cells, np.inf)
    maxy = np.full(n_cells, -np.inf)
    center = np.zeros((n_cells, 3))
    if len(starts):
        minx[valid] = np.minimum.reduceat(cx, starts)
        maxx[valid] = np.maximum.reduceat(cx, starts)
        miny[valid] = np.minimum.reduceat(cy, starts)
        maxy[valid] = np.maximum.reduceat(cy, starts)
        n = counts[valid].astype(float)
        center[valid, 0] = np.add.reduceat(cx, starts) / n
        center[valid, 1] = np.add.reduceat(cy, starts) / n
        center[valid, 2] = np.add.reduceat(cz, starts) / n
    inside = valid & (minx >= left) & (maxx <= right) & (miny >= top) & (maxy <= bottom)
    touched = valid & (maxx >= left) & (minx <= right) & (maxy >= top) & (miny <= bottom)

    raw_ids = mesh.cell_data.get("feature_id") if hasattr(mesh, "cell_data") else None
    feature_ids = (
        np.asarray(raw_ids, dtype=np.int64) if raw_ids is not None and len(raw_ids) == n_cells
        else np.arange(n_cells, dtype=np.int64)
    )
    order = np.argsort(feature_ids, kind="stable")
    sorted_ids = feature_ids[order]
    unique_ids, first = np.unique(sorted_ids, return_index=True)
    bounds = np.append(first, len(order))

    def visible(cell_id: int) -> bool:
        if get_z is None:
            return True
        x = int(round(center[cell_id, 0]))
        y = int(round(height - center[cell_id, 1]))
        # GetZ outside the render window reads past the depth buffer (access violation).
        if x < 0 or y < 0 or y >= height or (width is not None and x >= width):
            return False
        depth = get_z(x, y)
        return depth >= 0 and center[cell_id, 2] <= depth + depth_tolerance

    result: dict[int, list[int]] = {}
    for index, feature_id in enumerate(unique_ids.tolist()):
        cells = order[bounds[index]:bounds[index + 1]]
        face_valid = cells[valid[cells]]
        if len(face_valid) == 0:
            continue
        if contained:
            if not bool(np.all(inside[face_valid])):
                continue
            candidates = face_valid
        else:
            candidates = face_valid[touched[face_valid]]
            if len(candidates) == 0:
                continue
        if get_z is not None and not any(visible(int(c)) for c in candidates):
            continue
        result[int(feature_id)] = sorted(int(c) for c in cells)
    return result


__all__ = ["box_select_faces", "project_points_to_display"]
