"""3-axis surface roughing and finishing ("Trace curved face").

Height map: vertical rays onto the selected surface (trimesh). Tool envelope: the
tool-tip height at each sample is the highest the tool can sit without touching
the surface (flat: max height under the tool disc; ball: drop-cutter formula), so
passes never cut into the surface. Region: the projected outline of the selected
faces (shapely), grown by the stock clearance; raster lines are clipped to it.
Pass spacing defaults to 40% overlap (stepover = 60% of the tool diameter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import trimesh
from shapely.geometry import LineString, MultiLineString, MultiPoint, Polygon
from shapely.ops import unary_union

DEFAULT_OVERLAP = 0.40
PATTERNS = ("Rectilinear", "Grid", "Linear carve")
TOOL_SHAPES = ("Flat", "Ball")


@dataclass
class SurfaceParams:
    tool_shape: str = "Ball"
    tool_diameter_mm: float = 6.0
    stepover_mm: float | None = None  # default: 60% of the diameter (40% overlap)
    stepdown_mm: float = 2.0
    pattern: str = "Rectilinear"
    stock_to_leave_mm: float = 0.3  # roughing allowance
    clearance_mm: float = 1.0
    remove_bulk: bool = True
    resolution_mm: float | None = None

    def stepover(self) -> float:
        return float(self.stepover_mm) if self.stepover_mm else self.tool_diameter_mm * (1.0 - DEFAULT_OVERLAP)

    def resolution(self) -> float:
        return float(self.resolution_mm) if self.resolution_mm else max(0.2, min(self.stepover() / 2.0, self.tool_diameter_mm / 6.0))

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SurfaceParams":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(data or {}).items() if k in known})


@dataclass
class HeightMap:
    xs: np.ndarray
    ys: np.ndarray
    z: np.ndarray  # shape (len(ys), len(xs)); -inf where there is no surface
    step: float

    def at(self, x: float, y: float) -> float:
        i = int(round((y - self.ys[0]) / self.step))
        j = int(round((x - self.xs[0]) / self.step))
        if 0 <= i < len(self.ys) and 0 <= j < len(self.xs):
            return float(self.z[i, j])
        return -math.inf


@dataclass
class SurfacePlan:
    passes: list[list[tuple[float, float, float]]] = field(default_factory=list)
    roughing_layers: int = 0
    detail_deviation_mm: float = 0.0
    suggestion: str = ""


def _surface_mesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    try:
        tri = mesh.triangulate().extract_surface(algorithm="dataset_surface")
    except TypeError:  # older pyvista without the keyword
        tri = mesh.triangulate().extract_surface()
    faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(vertices=np.asarray(tri.points), faces=faces, process=False)


def region_polygon(mesh: pv.PolyData, clearance: float = 0.0) -> Polygon:
    tm = _surface_mesh(mesh)
    if len(tm.faces) <= 20000:
        tris = [Polygon(tm.vertices[f][:, :2]) for f in tm.faces]
        tris = [t for t in tris if t.is_valid and t.area > 1e-12]
        shape = unary_union(tris) if tris else MultiPoint(tm.vertices[:, :2]).convex_hull
    else:
        shape = MultiPoint(tm.vertices[:, :2]).convex_hull
    return shape.buffer(clearance) if clearance else shape


def height_map(mesh: pv.PolyData, step: float, margin: float) -> HeightMap:
    """Top-down height of the surface on a grid (highest triangle over each grid point)."""
    tm = _surface_mesh(mesh)
    lo, hi = tm.bounds
    xs = np.arange(lo[0] - margin, hi[0] + margin + step * 0.5, step)
    ys = np.arange(lo[1] - margin, hi[1] + margin + step * 0.5, step)
    z = np.full((len(ys), len(xs)), -np.inf)
    tri = tm.vertices[tm.faces]  # (n, 3, 3)
    for (ax, ay, az), (bx, by, bz), (cx, cy, cz) in tri:
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue  # vertical triangle: its edges are covered by neighbours
        j0 = max(int(math.floor((min(ax, bx, cx) - xs[0]) / step)), 0)
        j1 = min(int(math.ceil((max(ax, bx, cx) - xs[0]) / step)) + 1, len(xs))
        i0 = max(int(math.floor((min(ay, by, cy) - ys[0]) / step)), 0)
        i1 = min(int(math.ceil((max(ay, by, cy) - ys[0]) / step)) + 1, len(ys))
        if j0 >= j1 or i0 >= i1:
            continue
        gx, gy = np.meshgrid(xs[j0:j1], ys[i0:i1])
        l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
        l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if not inside.any():
            continue
        zz = np.where(inside, l1 * az + l2 * bz + l3 * cz, -np.inf)
        z[i0:i1, j0:j1] = np.maximum(z[i0:i1, j0:j1], zz)
    return HeightMap(xs, ys, z, step)


def tool_envelope(hmap: HeightMap, shape: str, diameter: float) -> np.ndarray:
    """Lowest tool-tip height at each grid point that does not touch the surface."""
    r = diameter / 2.0
    n = int(math.ceil(r / hmap.step))
    env = np.full(hmap.z.shape, -np.inf)
    padded = np.pad(hmap.z, n, constant_values=-np.inf)
    rows, cols = hmap.z.shape
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            d = math.hypot(di, dj) * hmap.step
            if d > r + 1e-9:
                continue
            window = padded[n + di : n + di + rows, n + dj : n + dj + cols]
            if shape == "Ball":
                lift = math.sqrt(max(r * r - d * d, 0.0)) - r
                env = np.maximum(env, window + lift)
            else:
                env = np.maximum(env, window)
    return env


def _raster_lines(region: Polygon, stepover: float, pattern: str) -> list[list[tuple[float, float]]]:
    minx, miny, maxx, maxy = region.bounds
    lines: list[list[tuple[float, float]]] = []
    y = miny
    flip = False
    while y <= maxy + 1e-9:
        clipped = region.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
        parts = list(clipped.geoms) if isinstance(clipped, MultiLineString) else ([clipped] if not clipped.is_empty else [])
        for part in parts:
            coords = list(part.coords)
            if pattern != "Linear carve" and flip:
                coords = coords[::-1]
            lines.append([(float(cx), float(cy)) for cx, cy in coords])
        flip = not flip
        y += stepover
    if pattern == "Grid":
        x = minx
        while x <= maxx + 1e-9:
            clipped = region.intersection(LineString([(x, miny - 1), (x, maxy + 1)]))
            parts = list(clipped.geoms) if isinstance(clipped, MultiLineString) else ([clipped] if not clipped.is_empty else [])
            lines.extend([[(float(cx), float(cy)) for cx, cy in p.coords] for p in parts])
            x += stepover
    return lines


def _sample_line(line, step: float):
    (x0, y0), (x1, y1) = line[0], line[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(math.ceil(length / step)))
    return [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1)]


def _env_at(env: np.ndarray, hmap: HeightMap, x: float, y: float) -> float:
    i = int(round((y - hmap.ys[0]) / hmap.step))
    j = int(round((x - hmap.xs[0]) / hmap.step))
    i = min(max(i, 0), len(hmap.ys) - 1)
    j = min(max(j, 0), len(hmap.xs) - 1)
    # highest neighbouring value: protects the surface between grid points
    window = env[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
    return float(np.max(window))


def plan_surface(mesh: pv.PolyData, params: SurfaceParams, stock_top_z: float | None = None, finish: bool = True) -> SurfacePlan:
    """Roughing layers (optional) then the 3-axis finishing passes."""
    step = params.resolution()
    hmap = height_map(mesh, step, margin=params.tool_diameter_mm)
    env = tool_envelope(hmap, params.tool_shape, params.tool_diameter_mm)
    region = region_polygon(mesh, params.clearance_mm)
    lines = _raster_lines(region, params.stepover(), params.pattern)
    plan = SurfacePlan()
    top = float(stock_top_z) if stock_top_z is not None else float(np.nanmax(hmap.z[np.isfinite(hmap.z)]))
    floor = float(np.min(env[np.isfinite(env)])) if np.isfinite(env).any() else top
    if params.remove_bulk:
        z_layer = top - params.stepdown_mm
        while z_layer > floor + params.stock_to_leave_mm - 1e-9:
            plan.roughing_layers += 1
            for line in lines:
                pts = []
                for x, y in _sample_line(line, step):
                    e = _env_at(env, hmap, x, y)
                    pts.append((x, y, max(z_layer, (e if math.isfinite(e) else floor) + params.stock_to_leave_mm)))
                if any(p[2] <= z_layer + 1e-9 for p in pts):  # only where material is removed at this layer
                    plan.passes.append(pts)
            z_layer -= params.stepdown_mm
    if finish:
        for line in lines:
            pts = []
            for x, y in _sample_line(line, step):
                e = _env_at(env, hmap, x, y)
                pts.append((x, y, e if math.isfinite(e) else floor))
            plan.passes.append(pts)
    finite = np.isfinite(hmap.z)
    plan.detail_deviation_mm = float(np.max(env[finite] - hmap.z[finite])) if finite.any() else 0.0
    return plan


def suggest_finer_tool(mesh: pv.PolyData, params: SurfaceParams, inventory_diameters: list[float], tolerance_mm: float = 0.1) -> tuple[float | None, float]:
    """If the tool cannot reach the detail, the largest smaller tool that can (or the smallest available)."""
    step = params.resolution()
    hmap = height_map(mesh, step, margin=params.tool_diameter_mm)
    finite = np.isfinite(hmap.z)

    def deviation(d):
        env = tool_envelope(hmap, params.tool_shape, d)
        lift = 0.0 if params.tool_shape == "Flat" else 0.0
        return float(np.max(env[finite] - hmap.z[finite])) - lift if finite.any() else 0.0

    current = deviation(params.tool_diameter_mm)
    if current <= tolerance_mm:
        return None, current
    for d in sorted((d for d in inventory_diameters if d < params.tool_diameter_mm), reverse=True):
        if deviation(d) <= tolerance_mm:
            return d, current
    smaller = [d for d in inventory_diameters if d < params.tool_diameter_mm]
    return (min(smaller) if smaller else None), current


def min_clearance_to_surface(passes, hmap: HeightMap) -> float:
    """Smallest (path z - surface z under the tool centre); >= 0 means no gouge."""
    worst = math.inf
    for line in passes:
        for x, y, z in line:
            s = hmap.at(x, y)
            if math.isfinite(s):
                worst = min(worst, z - s)
    return worst


# ---------------------------------------------------------------------- dialog
def trace_surface_dialog(parent, tools, default_diameter: float):
    """Trace curved face: tool, stepover/stepdown, pattern, remove bulk; returns SurfaceParams or None."""
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QLabel,
    )

    dialog = QDialog(parent)
    dialog.setWindowTitle("Trace curved face")
    form = QFormLayout(dialog)
    shape = QComboBox()
    shape.addItems(list(TOOL_SHAPES))
    form.addRow("Tool type", shape)
    diameter = QDoubleSpinBox()
    diameter.setRange(0.1, 100.0)
    diameter.setValue(default_diameter)
    diameter.setSuffix(" mm")
    form.addRow("Tool diameter", diameter)
    stepover = QDoubleSpinBox()
    stepover.setRange(0.01, 100.0)
    stepover.setValue(default_diameter * (1.0 - DEFAULT_OVERLAP))
    stepover.setSuffix(" mm")
    stepover.setToolTip("Default leaves 40% overlap between passes.")
    form.addRow("Stepover", stepover)
    stepdown = QDoubleSpinBox()
    stepdown.setRange(0.01, 100.0)
    stepdown.setValue(2.0)
    stepdown.setSuffix(" mm")
    form.addRow("Stepdown", stepdown)
    pattern = QComboBox()
    pattern.addItems(list(PATTERNS))
    form.addRow("Pattern", pattern)
    bulk = QCheckBox("Remove bulk first, then 3-axis contour")
    bulk.setChecked(True)
    form.addRow("", bulk)
    note = QLabel("Passes follow the tool envelope and never cut into the surface.")
    note.setWordWrap(True)
    form.addRow(note)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return SurfaceParams(
        tool_shape=shape.currentText(), tool_diameter_mm=diameter.value(), stepover_mm=stepover.value(),
        stepdown_mm=stepdown.value(), pattern=pattern.currentText(), remove_bulk=bulk.isChecked(),
    )
