"""Stock definitions (no Qt): block, sheet / plate, bar and tube, and their meshes.

Lengths in mm. Sheets and plates lie on XY from (origin_x, origin_y); their top is at Z = 0 when
``top_at_z0`` is set (usual for laser, plasma and waterjet), otherwise their bottom is at Z = 0.
Bars and tubes lie along X with their axis at (y, z) = (origin_y + r, origin_z + r).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyvista as pv

STOCK_KINDS = ("block", "sheet", "plate", "bar", "tube")
BAR_SHAPES = ("round", "hex", "square")
GRAIN = ("none", "x", "y")


@dataclass
class StockSpec:
    kind: str = "block"
    length: float = 100.0  # X
    width: float = 80.0  # Y
    thickness: float = 20.0  # Z (block height, sheet/plate thickness)
    material: str = "Unspecified"
    grain: str = "none"
    margin: float = 0.0  # unusable border of a sheet/plate
    bar_shape: str = "round"
    diameter: float = 25.0  # bar diameter or across-flats; tube outer diameter
    wall: float = 2.0  # tube wall
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0
    top_at_z0: bool = False

    @property
    def is_flat(self) -> bool:
        return self.kind in ("sheet", "plate")

    def size(self) -> tuple[float, float, float]:
        """Envelope X, Y, Z."""
        if self.kind in ("bar", "tube"):
            d = self.diameter
            if self.kind == "bar" and self.bar_shape == "hex":
                return self.length, d * 2.0 / np.sqrt(3.0), d  # flats vertical (Z), corners in Y
            return self.length, d, d
        return self.length, self.width, self.thickness

    def usable_area(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) inside the margin (sheet/plate nesting area)."""
        m = max(0.0, self.margin)
        return (self.origin_x + m, self.origin_y + m, self.origin_x + self.length - m, self.origin_y + self.width - m)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "StockSpec":
        spec = cls()
        for key, value in dict(data or {}).items():
            if hasattr(spec, key):
                default = getattr(spec, key)
                try:
                    setattr(spec, key, type(default)(value))
                except (TypeError, ValueError):
                    pass
        if spec.kind not in STOCK_KINDS:
            spec.kind = "block"
        if spec.bar_shape not in BAR_SHAPES:
            spec.bar_shape = "round"
        if spec.grain not in GRAIN:
            spec.grain = "none"
        return spec

    def validate(self) -> list[str]:
        problems = []
        if self.length <= 0:
            problems.append("Length must be greater than zero.")
        if self.kind in ("block", "sheet", "plate") and (self.width <= 0 or self.thickness <= 0):
            problems.append("Width and thickness must be greater than zero.")
        if self.kind in ("bar", "tube") and self.diameter <= 0:
            problems.append("Diameter must be greater than zero.")
        if self.kind == "tube" and not (0 < self.wall < self.diameter / 2.0):
            problems.append("Tube wall must be between zero and half the outer diameter.")
        if self.is_flat and 2 * self.margin >= min(self.length, self.width):
            problems.append("The margin leaves no usable area.")
        return problems


def _box(xmin, ymin, zmin, sx, sy, sz) -> pv.PolyData:
    return pv.Box(bounds=(xmin, xmin + sx, ymin, ymin + sy, zmin, zmin + sz)).triangulate()


def build_stock_mesh(spec: StockSpec) -> pv.PolyData:
    """Closed surface mesh of the stock."""
    x0, y0, z0 = spec.origin_x, spec.origin_y, spec.origin_z
    if spec.kind in ("block", "sheet", "plate"):
        zmin = z0 - spec.thickness if (spec.is_flat and spec.top_at_z0) else z0
        return _box(x0, y0, zmin, spec.length, spec.width, spec.thickness)
    sx, sy, sz = spec.size()
    centre_y, centre_z = y0 + sy / 2.0, z0 + sz / 2.0
    if spec.kind == "bar" and spec.bar_shape == "square":
        return _box(x0, y0, z0, spec.length, spec.diameter, spec.diameter)
    if spec.kind == "bar":
        sides = 6 if spec.bar_shape == "hex" else 64
        radius = spec.diameter / 2.0 if sides == 64 else spec.diameter / np.sqrt(3.0)  # hex: across corners / 2
        mesh = pv.Cylinder(
            center=(x0 + spec.length / 2.0, centre_y, centre_z),
            direction=(1.0, 0.0, 0.0),
            radius=radius,
            height=spec.length,
            resolution=sides,
            capping=True,
        ).triangulate()  # hex: pv.Cylinder already has flats top and bottom (corners on Y)
        return mesh
    # tube: revolve the wall cross-section (an annulus) about X.
    r_out = spec.diameter / 2.0
    r_in = r_out - spec.wall
    profile = np.array(
        [[0.0, r_in, 0.0], [spec.length, r_in, 0.0], [spec.length, r_out, 0.0], [0.0, r_out, 0.0], [0.0, r_in, 0.0]]
    )
    line = pv.lines_from_points(profile)
    tube = line.extrude_rotate(resolution=64, rotation_axis=(1.0, 0.0, 0.0), capping=False).triangulate()
    tube = tube.clean()
    return tube.translate((x0, centre_y, centre_z), inplace=False)


__all__ = ["BAR_SHAPES", "GRAIN", "STOCK_KINDS", "StockSpec", "build_stock_mesh"]
