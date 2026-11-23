"""True-shape nesting on sheet / plate (shapely, no Qt).

Parts are footprints (their outline seen from above, holes kept so small parts can sit inside
bigger parts' holes). Bottom-left fill: largest parts first, each placed at the lowest, then
left-most position where it fits inside the usable sheet area and keeps the spacing from every
placed part. Rotation is locked unless allowed (per nest and per part); a grain-locked part never
rotates. Parts that do not fit open another sheet; different materials / thicknesses nest on
separate stocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


@dataclass
class NestItem:
    key: str  # part id / name
    footprint: Polygon  # at its current position
    quantity: int = 1
    allow_rotation: bool = False
    grain_locked: bool = False
    group: str = ""  # material / thickness group


@dataclass
class Placement:
    key: str
    copy: int
    sheet: int
    angle_deg: float
    dx: float
    dy: float
    polygon: Polygon  # placed footprint


@dataclass
class NestResult:
    placements: list[Placement] = field(default_factory=list)
    unplaced: list[tuple[str, int]] = field(default_factory=list)
    sheets: int = 0
    utilisation: list[float] = field(default_factory=list)  # per sheet, 0..1
    group: str = ""


@dataclass
class NestSettings:
    sheet_length: float = 1220.0
    sheet_width: float = 2440.0
    margin: float = 5.0
    spacing: float = 0.4  # default 2 x kerf
    allow_rotation: bool = False  # off by default
    rotation_steps: int = 4  # 0/90/180/270; more = free rotation in N steps
    max_sheets: int = 20
    origin: tuple[float, float] = (0.0, 0.0)


def footprint_of_mesh(mesh) -> Polygon:
    """Outline of a part seen from above (union of its projected triangles), holes kept."""
    pts = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = []
    i = 0
    while i < len(faces):
        n = int(faces[i])
        ids = faces[i + 1:i + 1 + n]
        if n >= 3:
            poly = Polygon([(pts[j][0], pts[j][1]) for j in ids])
            if poly.is_valid and poly.area > 1e-9:
                triangles.append(poly)
        i += n + 1
    if not triangles:
        b = mesh.bounds
        return box(b[0], b[2], b[1], b[3])
    shape = unary_union(triangles).buffer(0)
    if hasattr(shape, "geoms"):
        shape = max(shape.geoms, key=lambda g: g.area)
    return shape.simplify(0.01)


def _angles(item: NestItem, settings: NestSettings) -> list[float]:
    if item.grain_locked or not (settings.allow_rotation or item.allow_rotation):
        return [0.0]
    steps = max(1, int(settings.rotation_steps))
    return [360.0 * k / steps for k in range(steps)]


def nest(items: list[NestItem], settings: NestSettings) -> NestResult:
    result = NestResult(group=items[0].group if items else "")
    ox, oy = settings.origin
    usable = box(ox + settings.margin, oy + settings.margin, ox + settings.sheet_length - settings.margin, oy + settings.sheet_width - settings.margin)
    queue = []
    for item in items:
        for copy in range(max(1, int(item.quantity))):
            queue.append((item, copy))
    queue.sort(key=lambda entry: -entry[0].footprint.area)  # largest first
    sheets: list[list[Polygon]] = []
    half = settings.spacing / 2.0

    def try_sheet(placed: list[Polygon], item: NestItem):
        best = None
        for angle in _angles(item, settings):
            shape = affinity.rotate(item.footprint, angle, origin="centroid") if angle else item.footprint
            minx, miny, _maxx, _maxy = shape.bounds
            shape0 = affinity.translate(shape, -minx, -miny)  # bottom-left at (0, 0)
            grown = shape0.buffer(half, join_style=2)
            xs = {usable.bounds[0]} | {p.bounds[2] + settings.spacing for p in placed}
            ys = {usable.bounds[1]} | {p.bounds[3] + settings.spacing for p in placed}
            for p in placed:  # against other parts' corners (lets L shapes interlock)
                for vx, vy in list(p.exterior.coords)[:-1]:
                    xs.add(vx + settings.spacing)
                    ys.add(vy + settings.spacing)
            for p in placed:  # inside other parts' holes
                for interior in getattr(p, "interiors", []):
                    hx, hy, _hx1, _hy1 = Polygon(interior).bounds
                    xs.add(hx + settings.spacing)
                    ys.add(hy + settings.spacing)
            others = unary_union([p.buffer(half, join_style=2) for p in placed]) if placed else None
            for y in sorted(ys):
                for x in sorted(xs):
                    candidate = affinity.translate(grown, x, y)
                    if not usable.buffer(half + 1e-9, join_style=2).contains(candidate):
                        continue
                    if others is not None and candidate.intersection(others).area > 1e-9:
                        continue
                    key = (y, x)
                    if best is None or key < best[0]:
                        best = (key, angle, x - minx, y - miny, affinity.translate(shape0, x, y))
                    break  # lowest x for this y found
                if best is not None and best[0][0] <= y:
                    break
        return best

    for item, copy in queue:
        placed_here = False
        for sheet_index in range(len(sheets) + 1):
            if sheet_index == len(sheets):
                if len(sheets) >= settings.max_sheets:
                    break
                sheets.append([])
            found = try_sheet(sheets[sheet_index], item)
            if found is not None:
                _key, angle, dx, dy, polygon = found
                sheets[sheet_index].append(polygon)
                result.placements.append(Placement(item.key, copy, sheet_index, angle, dx, dy, polygon))
                placed_here = True
                break
            if not sheets[sheet_index]:
                sheets.pop()  # an empty new sheet did not help: the part is too big
                break
        if not placed_here:
            result.unplaced.append((item.key, copy))
    result.sheets = len(sheets)
    area = usable.area
    result.utilisation = [sum(p.area for p in sheet) / area if area > 0 else 0.0 for sheet in sheets]
    return result


def nest_groups(items: list[NestItem], settings: NestSettings) -> list[NestResult]:
    """Different materials / thicknesses nest on separate stocks."""
    groups: dict[str, list[NestItem]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item)
    return [nest(group, settings) for _key, group in sorted(groups.items())]


def placement_matrix(item_footprint: Polygon, placement: Placement) -> np.ndarray:
    """4x4 matrix moving a part from where it is to its placement (rotate about the footprint
    centroid, then translate)."""
    c = item_footprint.centroid
    a = math.radians(placement.angle_deg)
    rot = np.eye(4)
    rot[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    to_origin = np.eye(4)
    to_origin[:2, 3] = [-c.x, -c.y]
    back = np.eye(4)
    back[:2, 3] = [c.x, c.y]
    rotated = affinity.rotate(item_footprint, placement.angle_deg, origin="centroid")
    minx, miny, _x1, _y1 = rotated.bounds
    target_minx, target_miny, _a, _b = placement.polygon.bounds
    shift = np.eye(4)
    shift[:2, 3] = [target_minx - minx, target_miny - miny]
    return shift @ back @ rot @ to_origin


__all__ = ["NestItem", "NestResult", "NestSettings", "Placement", "footprint_of_mesh", "nest", "nest_groups", "placement_matrix"]
