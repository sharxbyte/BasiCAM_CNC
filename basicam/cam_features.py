from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyvista as pv

from basicam.operations import Operation
from basicam.setups import MachiningSetup
from basicam.sketch_workflow import (
    ParametricSketchFeature,
    SketchDimension,
    SketchWorkflowManager,
)


WCS_CODES = ["G54", "G55", "G56", "G57", "G58", "G59"]


@dataclass
class ToolpathDynamics:
    """Lead-in, lead-out, plunge, ramp, and retract heights/parameters (mm & mm/min)."""

    clearance_z_mm: float = 10.0
    retract_z_mm: float = 3.0
    plunge_feed_mm_min: float = 150.0
    ramp_angle_deg: float = 3.0
    lead_in_mm: float = 2.0
    lead_out_mm: float = 2.0
    entry_mode: str = "Ramp"  # "Plunge", "Ramp", "Helix"


@dataclass
class GCodeValidationIssue:
    severity: str  # "error" or "warning"
    line_number: int
    message: str


@dataclass
class GrblModalState:
    motion: str = "G0"
    wcs: str = "G54"
    plane: str = "G17"
    distance: str = "G90"
    feed_mode: str = "G94"
    units: str = "G21"
    cutter_comp: str = "G40"
    tool_offset: str = "G49"
    program: str = "M0"
    spindle: str = "M5"
    coolant: str = "M9"
    tool_number: int = 0
    feed_rate: float = 0.0
    spindle_rpm: float = 0.0

    def summary_text(self) -> str:
        return (
            f"Modal: {self.wcs} · {self.motion} · {self.distance} · {self.units} · "
            f"{self.spindle} · F{self.feed_rate:g} S{self.spindle_rpm:g}"
        )


def parse_grbl_parser_state(line: str, current: GrblModalState | None = None) -> GrblModalState | None:
    """Parses GRBL `$G` response: `[GC:G0 G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]`."""
    match = re.search(r"\[GC:([^\]]+)\]", line.strip())
    if not match:
        return None
    state = current or GrblModalState()
    tokens = match.group(1).strip().split()
    for tok in tokens:
        if tok in {"G0", "G1", "G2", "G3", "G38.2", "G38.3", "G38.4", "G38.5", "G80"}:
            state.motion = tok
        elif tok in WCS_CODES:
            state.wcs = tok
        elif tok in {"G17", "G18", "G19"}:
            state.plane = tok
        elif tok in {"G90", "G91"}:
            state.distance = tok
        elif tok in {"G93", "G94"}:
            state.feed_mode = tok
        elif tok in {"G20", "G21"}:
            state.units = tok
        elif tok in {"G40"}:
            state.cutter_comp = tok
        elif tok in {"G43.1", "G49"}:
            state.tool_offset = tok
        elif tok in {"M0", "M1", "M2", "M30"}:
            state.program = tok
        elif tok in {"M3", "M4", "M5"}:
            state.spindle = tok
        elif tok in {"M7", "M8", "M9"}:
            state.coolant = tok
        elif tok.startswith("T") and tok[1:].isdigit():
            state.tool_number = int(tok[1:])
        elif tok.startswith("F"):
            try:
                state.feed_rate = float(tok[1:])
            except ValueError:
                pass
        elif tok.startswith("S"):
            try:
                state.spindle_rpm = float(tok[1:])
            except ValueError:
                pass
    return state


def auto_fit_stock_bounds(
    meshes: Iterable[pv.PolyData],
    margin_xy_mm: float = 2.0,
    margin_z_top_mm: float = 0.5,
    margin_z_bottom_mm: float = 1.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Computes `(center_xyz, size_xyz)` for a stock box enclosing all target part meshes plus margins."""
    valid = [m for m in meshes if m is not None and m.n_cells > 0]
    if not valid:
        return None
    all_bounds = np.array([m.bounds for m in valid], dtype=float)
    xmin = float(np.min(all_bounds[:, 0])) - margin_xy_mm
    xmax = float(np.max(all_bounds[:, 1])) + margin_xy_mm
    ymin = float(np.min(all_bounds[:, 2])) - margin_xy_mm
    ymax = float(np.max(all_bounds[:, 3])) + margin_xy_mm
    zmin = float(np.min(all_bounds[:, 4])) - margin_z_bottom_mm
    zmax = float(np.max(all_bounds[:, 5])) + margin_z_top_mm
    center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5)
    size = (max(xmax - xmin, 0.1), max(ymax - ymin, 0.1), max(zmax - zmin, 0.1))
    return center, size


def map_sketch_to_operation_params(
    feat: ParametricSketchFeature,
    tool_diameter_mm: float = 6.0,
    default_depth_mm: float = 5.0,
) -> dict[str, Any]:
    """Maps a sketch feature (including rectangles, circles, arcs, and 3D projected splines) to CAM operation inputs."""
    pts = np.asarray(feat.active_points, dtype=float)
    z_top = float(np.max(pts[:, 2])) if len(pts) else 0.0
    z_bot = float(np.min(pts[:, 2])) if len(pts) else 0.0
    depth = max(default_depth_mm, (z_top - z_bot) if feat.is_projected else default_depth_mm)

    if feat.is_projected:
        suggested_kind = "3D Surface Carve"
        strategy = f"3-axis surface-projected spline follow ({len(pts)} pts)"
    elif feat.kind == "Point":
        suggested_kind = "Drill"
        strategy = "Point center plunge / spot drill"
    elif feat.kind == "Circle":
        suggested_kind = "Pocket"
        strategy = "Closed circular pocket / bore"
    elif feat.kind == "Rectangle":
        suggested_kind = "Pocket"
        strategy = "Closed rectangular pocket / face clear"
    elif feat.kind in {"Curve", "Spline"}:
        suggested_kind = "Contour"
        strategy = f"Multi-point {feat.kind.lower()} contour follow"
    else:
        suggested_kind = "Contour"
        strategy = "Linear profile / slot pass"

    return {
        "target": feat.name,
        "suggested_kind": suggested_kind,
        "depth_mm": round(depth, 3),
        "strategy": strategy,
        "summary": f"{feat.name} ({feat.kind}{', 3D projected' if feat.is_projected else ''})",
        "points": pts.tolist(),
    }

def build_verified_toolpath_segments(
    setups: list[MachiningSetup],
    parts_bounds: tuple[float, float, float, float, float, float],
    dynamics: ToolpathDynamics,
    sketch_features: list[ParametricSketchFeature] | None = None,
) -> list[dict[str, Any]]:
    """Generates multi-pass Z stepdown, lead-in/lead-out, ramp/plunge, and retract polylines for verification."""
    xmin, xmax, ymin, ymax, zmin, zmax = parts_bounds
    inset_x = (xmax - xmin) * 0.12
    inset_y = (ymax - ymin) * 0.12
    sketch_lookup = {f.name: f for f in (sketch_features or [])}
    segments: list[dict[str, Any]] = []

    for s_idx, setup in enumerate(setups):
        ox, oy, oz = setup.origin_mm
        for op in setup.operations:
            stepdown = max(float(op.stepdown_mm), 0.2)
            total_depth = max(float(op.depth_mm), 0.1)
            n_passes = max(1, int(math.ceil(total_depth / stepdown)))
            top_z = oz + zmax
            clear_z = top_z + dynamics.clearance_z_mm
            retract_z = top_z + dynamics.retract_z_mm

            feat = sketch_lookup.get(op.target)
            if feat is not None and len(feat.points) > 0:
                base_pts = np.asarray(feat.points, dtype=float)
            else:
                base_pts = np.array(
                    [
                        [ox + xmin + inset_x, oy + ymin + inset_y, top_z],
                        [ox + xmax - inset_x, oy + ymin + inset_y, top_z],
                        [ox + xmax - inset_x, oy + ymax - inset_y, top_z],
                        [ox + xmin + inset_x, oy + ymax - inset_y, top_z],
                        [ox + xmin + inset_x, oy + ymin + inset_y, top_z],
                    ],
                    dtype=float,
                )

            rapid_pts: list[np.ndarray] = []
            cut_pts: list[np.ndarray] = []
            entry_pts: list[np.ndarray] = []

            start_xy = base_pts[0, :2]
            rapid_pts.append(np.array([start_xy[0], start_xy[1], clear_z]))
            rapid_pts.append(np.array([start_xy[0], start_xy[1], retract_z]))

            for p_idx in range(1, n_passes + 1):
                pass_z = top_z - min(p_idx * stepdown, total_depth)
                prev_z = retract_z if p_idx == 1 else (top_z - min((p_idx - 1) * stepdown, total_depth))

                if len(base_pts) == 1 or op.kind in {"Drill", "Ream", "Tap"}:
                    entry_pts.extend(
                        [
                            np.array([start_xy[0], start_xy[1], prev_z]),
                            np.array([start_xy[0], start_xy[1], pass_z]),
                        ]
                    )
                    rapid_pts.extend(
                        [
                            np.array([start_xy[0], start_xy[1], pass_z]),
                            np.array([start_xy[0], start_xy[1], retract_z]),
                        ]
                    )
                else:
                    lead_vec = base_pts[1, :2] - base_pts[0, :2]
                    norm = float(np.linalg.norm(lead_vec))
                    perp = (
                        np.array([-lead_vec[1], lead_vec[0]]) / norm
                        if norm > 1e-6
                        else np.array([1.0, 0.0])
                    )
                    lead_in_xy = start_xy + perp * dynamics.lead_in_mm
                    lead_out_xy = base_pts[-1, :2] + perp * dynamics.lead_out_mm

                    if dynamics.entry_mode == "Ramp" and len(base_pts) >= 2:
                        mid_xy = 0.5 * (start_xy + base_pts[1, :2])
                        entry_pts.extend(
                            [
                                np.array([lead_in_xy[0], lead_in_xy[1], prev_z]),
                                np.array([start_xy[0], start_xy[1], 0.5 * (prev_z + pass_z)]),
                                np.array([mid_xy[0], mid_xy[1], pass_z]),
                            ]
                        )
                    else:
                        entry_pts.extend(
                            [
                                np.array([lead_in_xy[0], lead_in_xy[1], prev_z]),
                                np.array([lead_in_xy[0], lead_in_xy[1], pass_z]),
                                np.array([start_xy[0], start_xy[1], pass_z]),
                            ]
                        )

                    for pt in base_pts:
                        cut_pts.append(np.array([pt[0], pt[1], pass_z]))
                    cut_pts.append(np.array([lead_out_xy[0], lead_out_xy[1], pass_z]))

            last_xy = base_pts[-1, :2]
            rapid_pts.extend(
                [
                    np.array([last_xy[0], last_xy[1], retract_z]),
                    np.array([last_xy[0], last_xy[1], clear_z]),
                ]
            )
            segments.append(
                {
                    "setup_index": s_idx,
                    "setup_color": setup.color,
                    "operation_name": op.name,
                    "rapid": np.array(rapid_pts, dtype=float) if rapid_pts else np.empty((0, 3)),
                    "entry": np.array(entry_pts, dtype=float) if entry_pts else np.empty((0, 3)),
                    "cut": np.array(cut_pts, dtype=float) if cut_pts else np.empty((0, 3)),
                }
            )
    return segments


def validate_gcode_program(
    gcode_text: str,
    stock_bounds: tuple[float, float, float, float, float, float] | None = None,
    expected_units: str = "G21",
) -> list[GCodeValidationIssue]:
    """Validates generated G-code for units, WCS, spindle/feed safety, and stock envelope bounds."""
    issues: list[GCodeValidationIssue] = []
    lines = gcode_text.splitlines()
    has_units = False
    has_wcs = False
    spindle_running = False
    current_feed = 0.0
    motion_mode = "G0"
    distance_mode = "G90"
    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}

    word_re = re.compile(r"([A-Z])([+-]?\d+(?:\.\d+)?)")

    for idx, raw in enumerate(lines, start=1):
        clean = raw.split(";")[0].strip()
        if "(" in clean:
            clean = re.sub(r"\([^)]*\)", "", clean).strip()
        if not clean:
            continue
        words = [(m.group(1), float(m.group(2))) for m in word_re.finditer(clean.upper())]
        for letter, val in words:
            ival = int(round(val))
            if letter == "G":
                if ival in (20, 21):
                    has_units = True
                    unit_str = f"G{ival}"
                    if unit_str != expected_units:
                        issues.append(
                            GCodeValidationIssue(
                                "error", idx, f"Unit mismatch: found {unit_str}, expected {expected_units}."
                            )
                        )
                elif ival in (54, 55, 56, 57, 58, 59):
                    has_wcs = True
                elif ival in (0, 1, 2, 3):
                    motion_mode = f"G{ival}"
                elif ival in (90, 91):
                    distance_mode = f"G{ival}"
            elif letter == "M":
                if ival in (3, 4):
                    spindle_running = True
                elif ival == 5:
                    spindle_running = False
            elif letter == "F":
                current_feed = val
                if val <= 0:
                    issues.append(GCodeValidationIssue("error", idx, "Feed rate F must be > 0."))
            elif letter in pos:
                pos[letter] = val if distance_mode == "G90" else pos[letter] + val

        if motion_mode in ("G1", "G2", "G3") and any(l in ("X", "Y", "Z") for l, _ in words):
            if current_feed <= 0:
                issues.append(
                    GCodeValidationIssue("error", idx, f"Cutting move ({motion_mode}) with zero feed rate.")
                )
            if not spindle_running:
                issues.append(
                    GCodeValidationIssue("warning", idx, f"Cutting move ({motion_mode}) while spindle is off (M5).")
                )
            if stock_bounds is not None:
                xmin, xmax, ymin, ymax, zmin, _zmax = stock_bounds
                if pos["Z"] < zmin - 0.05:
                    issues.append(
                        GCodeValidationIssue(
                            "warning", idx, f"Z={pos['Z']:.3f} extends below stock bottom ({zmin:.3f})."
                        )
                    )
                if not (xmin - 5.0 <= pos["X"] <= xmax + 5.0 and ymin - 5.0 <= pos["Y"] <= ymax + 5.0):
                    issues.append(
                        GCodeValidationIssue(
                            "warning",
                            idx,
                            f"XY ({pos['X']:.2f}, {pos['Y']:.2f}) exceeds stock envelope.",
                        )
                    )

    if not has_units:
        issues.append(GCodeValidationIssue("error", 1, "Missing units declaration (G20/G21)."))
    if not has_wcs:
        issues.append(GCodeValidationIssue("warning", 1, "No active WCS (G54–G59) declared in program."))
    return issues


def serialize_project_state(
    *,
    job_name: str,
    stock_material: str,
    stock_dims: dict[str, float],
    parts: list[Any],
    setups: list[MachiningSetup],
    sketch_workflow: SketchWorkflowManager,
    dynamics: ToolpathDynamics,
) -> dict[str, Any]:
    """Serializes the current BasiCAM project state to a JSON-compatible dictionary."""
    return {
        "version": 1,
        "job_name": job_name,
        "stock_material": stock_material,
        "stock_dims": stock_dims,
        "dynamics": asdict(dynamics),
        "parts": [
            {
                "name": p.name,
                "source_path": str(p.source_path) if p.source_path else None,
                "imported": bool(p.imported),
                "role": p.role.value,
                "note": p.note,
                "feature_notes": {str(k): v for k, v in p.feature_notes.items()},
            }
            for p in parts
        ],
        "setups": [
            {
                "name": s.name,
                "orientation": s.orientation,
                "origin_mm": list(s.origin_mm),
                "color": s.color,
                "wcs": getattr(s, "wcs", "G54"),
                "operations": [asdict(op) for op in s.operations],
            }
            for s in setups
        ],
        "sketches": [
            {
                "feature_id": f.feature_id,
                "name": f.name,
                "kind": f.kind,
                "plane": f.plane,
                "points": np.asarray(f.points, dtype=float).tolist(),
                "construction": f.construction,
                "parent_feature_id": f.parent_feature_id,
                "defined": f.defined,
                "has_error": f.has_error,
                "error_message": f.error_message,
                "dimensions": [asdict(d) for d in f.dimensions],
            }
            for f in sketch_workflow.features
        ],
    }


def save_project_file(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_project_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
