"""Viewport display modes and helpers (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

DISPLAY_MODES: tuple[str, ...] = ("Shaded", "Unshaded solid", "Transparent shaded", "Hidden lines", "Wireframe")

# Older names saved in settings or projects.
LEGACY_NAMES = {"Transparent hidden lines": "Hidden lines"}


@dataclass(frozen=True)
class ModeFlags:
    show_surface: bool
    lighting: bool
    opacity: float
    surface_is_mask: bool  # surface drawn in background colour to hide what is behind it
    show_edges: bool
    dashed_hidden_edges: bool


def normalize_mode(mode: str) -> str:
    mode = LEGACY_NAMES.get(mode, mode)
    return mode if mode in DISPLAY_MODES else "Shaded"


def mode_flags(mode: str) -> ModeFlags:
    mode = normalize_mode(mode)
    if mode == "Unshaded solid":
        return ModeFlags(True, False, 1.0, False, True, False)
    if mode == "Transparent shaded":
        return ModeFlags(True, True, 0.35, False, True, False)
    if mode == "Hidden lines":
        return ModeFlags(True, False, 1.0, True, True, True)
    if mode == "Wireframe":
        return ModeFlags(False, False, 0.0, False, True, False)
    return ModeFlags(True, True, 1.0, False, True, False)


@dataclass(frozen=True)
class LightingParams:
    key_intensity: float
    fill_intensity: float
    key_position: tuple[float, float, float]  # camera-relative direction (camera looks down -Z)
    ambient: float
    diffuse: float
    specular: float


LIGHTING_DEFAULTS = {"brightness": 100, "contrast": 50, "azimuth": 35, "elevation": 45}


def lighting_params(brightness: float, contrast: float, azimuth_deg: float, elevation_deg: float) -> LightingParams:
    """Off-screen key light that follows the camera plus a weaker fill light.

    brightness 10-200 (%) scales both lights; contrast 0-100 (%) moves light from the
    fill/ambient into the key light so faces at different angles differ more.
    """
    b = max(0.1, min(2.0, brightness / 100.0))
    c = max(0.0, min(1.0, contrast / 100.0))
    az = np.radians(azimuth_deg)
    el = np.radians(max(-89.0, min(89.0, elevation_deg)))
    direction = (float(np.sin(az) * np.cos(el)), float(np.sin(el)), float(np.cos(az) * np.cos(el)))
    return LightingParams(
        key_intensity=b * (0.75 + 0.5 * c),
        fill_intensity=b * (0.6 - 0.35 * c),
        key_position=direction,
        ambient=0.12 + 0.2 * (1.0 - c),
        diffuse=0.75 + 0.2 * c,
        specular=0.05 + 0.25 * c,
    )


def dash_lines(lines: pv.PolyData, dash: float, gap: float) -> pv.PolyData:
    """Split every line segment into dashes of length `dash` separated by `gap`."""
    if lines is None or lines.n_cells == 0 or dash <= 0:
        return pv.PolyData()
    points: list[np.ndarray] = []
    pts = np.asarray(lines.points, dtype=float)
    period = dash + max(gap, 0.0)
    for cell_index in range(lines.n_cells):
        ids = lines.get_cell(cell_index).point_ids
        for a, b in zip(ids[:-1], ids[1:]):
            p0, p1 = pts[a], pts[b]
            length = float(np.linalg.norm(p1 - p0))
            if length <= 1e-12:
                continue
            direction = (p1 - p0) / length
            start = 0.0
            while start < length:
                end = min(start + dash, length)
                points.append(p0 + direction * start)
                points.append(p0 + direction * end)
                start += period
    if not points:
        return pv.PolyData()
    arr = np.asarray(points)
    n = len(arr) // 2
    cells = np.column_stack([np.full(n, 2), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)]).ravel()
    return pv.PolyData(arr, lines=cells)
