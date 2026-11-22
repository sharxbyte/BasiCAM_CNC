from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from basicam.models import PartModel, PartRole
from basicam.operations import Operation, pass_depths
from basicam.setups import MachiningSetup
from basicam.sketch_workflow import ParametricSketchFeature, SketchDimension


@dataclass
class DynamicFeatureRecognitionResult:
    """Result of scanning geometry/sketch features for pocket, hole, slot, and contour candidates."""

    pockets: list[dict[str, Any]] = field(default_factory=list)
    holes: list[dict[str, Any]] = field(default_factory=list)
    slots: list[dict[str, Any]] = field(default_factory=list)
    contours: list[dict[str, Any]] = field(default_factory=list)
    source_summary: str = ""


class DynamicFeatureRecognizer:
    """
    Scans imported meshes and parametric sketch features to suggest candidate
    machining features (pockets, holes, slots, contours).
    """

    def analyze(
        self,
        part_meshes: list[pv.PolyData] | None = None,
        stock_mesh: pv.PolyData | None = None,
        sketch_features: list[ParametricSketchFeature] | None = None,
    ) -> DynamicFeatureRecognitionResult:
        result = DynamicFeatureRecognitionResult()
        meshes = [m for m in (part_meshes or []) if m is not None and m.n_points > 0]
        sketches = list(sketch_features or [])

        for idx, mesh in enumerate(meshes):
            bounds = mesh.bounds  # xmin, xmax, ymin, ymax, zmin, zmax
            dx = max(0.0, float(bounds[1] - bounds[0]))
            dy = max(0.0, float(bounds[3] - bounds[2]))
            dz = max(0.0, float(bounds[5] - bounds[4]))
            if dx > 0 and dy > 0 and dz > 0:
                result.contours.append(
                    {
                        "name": f"Outer Contour · Part {idx + 1}",
                        "kind": "Contour",
                        "x_mm": round(dx, 3),
                        "y_mm": round(dy, 3),
                        "z_mm": round(dz, 3),
                        "suggested_depth_mm": round(min(dz, max(1.0, dz * 0.5)), 2),
                        "suggested_tool_mm": round(min(12.0, max(2.0, min(dx, dy) * 0.1)), 2),
                    }
                )
                if dz > 0.5 and min(dx, dy) > 4.0:
                    result.pockets.append(
                        {
                            "name": f"Pocket Candidate · Part {idx + 1}",
                            "kind": "Pocket",
                            "x_mm": round(dx * 0.7, 3),
                            "y_mm": round(dy * 0.7, 3),
                            "suggested_depth_mm": round(max(1.0, dz * 0.35), 2),
                            "suggested_tool_mm": round(min(6.0, max(1.5, min(dx, dy) * 0.08)), 2),
                        }
                    )

        for feat in sketches:
            kind_l = str(feat.kind).lower()
            pts = np.asarray(feat.points, dtype=float)
            span_x = float(np.ptp(pts[:, 0])) if pts.ndim == 2 and pts.shape[0] > 0 else 0.0
            span_y = float(np.ptp(pts[:, 1])) if pts.ndim == 2 and pts.shape[0] > 0 else 0.0
            dim_vals = [float(d.value) for d in feat.dimensions if float(d.value) > 0]
            nom_size = dim_vals[0] if dim_vals else max(span_x, span_y, 6.0)

            if "circle" in kind_l or "hole" in kind_l or "arc" in kind_l:
                result.holes.append(
                    {
                        "name": f"Hole · {feat.name}",
                        "kind": "Hole",
                        "diameter_mm": round(nom_size, 3),
                        "suggested_depth_mm": round(max(2.0, nom_size * 1.2), 2),
                        "suggested_tool_mm": round(max(1.0, min(nom_size * 0.8, 10.0)), 2),
                        "sketch_id": feat.feature_id,
                    }
                )
            elif "slot" in kind_l or "rect" in kind_l:
                result.slots.append(
                    {
                        "name": f"Slot · {feat.name}",
                        "kind": "Slot",
                        "length_mm": round(max(span_x, span_y, nom_size), 3),
                        "width_mm": round(min(span_x, span_y) if min(span_x, span_y) > 0 else nom_size * 0.5, 3),
                        "suggested_depth_mm": round(max(1.5, nom_size * 0.4), 2),
                        "suggested_tool_mm": round(max(1.5, min(6.0, nom_size * 0.25)), 2),
                        "sketch_id": feat.feature_id,
                    }
                )
            else:
                result.contours.append(
                    {
                        "name": f"Sketch Profile · {feat.name}",
                        "kind": "Contour",
                        "length_mm": round(max(span_x, span_y, nom_size), 3),
                        "suggested_depth_mm": 2.0,
                        "suggested_tool_mm": 4.0,
                        "sketch_id": feat.feature_id,
                    }
                )

        result.source_summary = (
            f"{len(result.pockets)} pocket(s), {len(result.holes)} hole(s), "
            f"{len(result.slots)} slot(s), {len(result.contours)} contour(s)"
        )
        return result


@dataclass
class ToolpathDynamicsConfig:
    """Configurable toolpath dynamics, corner smoothing, lead-in/out, and clearance parameters."""

    clearance_z_mm: float = 5.0
    retract_z_mm: float = 2.0
    corner_smoothing_radius_mm: float = 0.5
    lead_in_radius_mm: float = 1.5
    lead_out_radius_mm: float = 1.5
    plunge_feed_ratio: float = 0.35
    finish_feed_ratio: float = 0.75
    trochoidal_stepover_ratio: float = 0.12
    rapid_feed_mm_min: float = 3000.0


@dataclass
class ToolpathSegment:
    """Single motion segment with feed override and dynamic metadata."""

    kind: str  # "rapid", "plunge", "cut", "lead_in", "lead_out", "retract"
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    feed_mm_min: float
    rpm: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolpathDynamicsEngine:
    """
    Generates and refines multi-pass toolpath motion segments from Operations and
    MachiningSetups, applying clearance planes, lead-in/lead-out transitions,
    and corner smoothing.
    """

    def __init__(self, config: ToolpathDynamicsConfig | None = None) -> None:
        self.config = config or ToolpathDynamicsConfig()

    @property
    def clearance_z_mm(self) -> float:
        return float(self.config.clearance_z_mm)

    @clearance_z_mm.setter
    def clearance_z_mm(self, value: float) -> None:
        self.config.clearance_z_mm = float(value)

    def build_operation_segments(
        self,
        operation: Operation,
        setup: MachiningSetup | None = None,
        stock_top_z_mm: float = 0.0,
        bounds_xy: tuple[float, float, float, float] = (-40.0, 40.0, -40.0, 40.0),
        path_xy: list[tuple[float, float]] | None = None,
        surface_passes: list[list[tuple[float, float, float]]] | None = None,
    ) -> list[ToolpathSegment]:
        """Segments for one operation.

        Path: the target sketch's closed contour (path_xy) or, without one, the bounds rectangle.
        Depths: light first passes, roughing at the step-down, then the final pass (pass_depths).
        Entry: plunge, or a ramp by angle or length along the path; optional straight lead-in/out.
        Direction: climb runs the contour clockwise, conventional counter-clockwise; the first pass
        of each new layer is marked full-width (slotting) and fed slower.
        Surface passes (from basicam.surfacing) are emitted as given, never below the surface.
        """
        ox, oy, oz = setup.origin_mm if setup is not None else (0.0, 0.0, 0.0)
        safe_clear = float(getattr(setup, "safe_z_mm", self.config.clearance_z_mm)) if setup is not None else float(self.config.clearance_z_mm)
        retract_clear = float(getattr(setup, "retract_clearance_mm", self.config.retract_z_mm)) if setup is not None else float(self.config.retract_z_mm)
        top = stock_top_z_mm + oz
        safe_z = top + max(1.0, safe_clear)
        retract_z = top + max(0.5, min(retract_clear, safe_clear))
        cut_feed = max(10.0, float(operation.feed_mm_min))
        plunge_feed = max(5.0, cut_feed * float(self.config.plunge_feed_ratio))
        rapid_feed = max(cut_feed, float(self.config.rapid_feed_mm_min))
        rpm = int(operation.rpm)
        meta = {"op": operation.name, "tool": operation.tool}
        if setup is not None:
            meta["setup"] = f"{setup.name} · {setup.orientation}"
        segments: list[ToolpathSegment] = []

        def add(kind, start, end, feed, **extra):
            extra = {("pass" if key == "pass_" else key): value for key, value in extra.items()}
            segments.append(ToolpathSegment(kind, tuple(map(float, start)), tuple(map(float, end)), feed, rpm, {**meta, **extra}))

        if surface_passes:
            first = surface_passes[0][0]
            add("rapid", (first[0], first[1], safe_z), (first[0], first[1], safe_z), rapid_feed, pass_=0)
            for index, line in enumerate(surface_passes, start=1):
                start = line[0]
                add("rapid", (start[0], start[1], safe_z if index == 1 else retract_z), (start[0], start[1], retract_z), rapid_feed, pass_=index)
                add("plunge", (start[0], start[1], retract_z), start, plunge_feed, pass_=index)
                for a, b in zip(line[:-1], line[1:]):
                    add("cut", a, b, cut_feed, pass_=index, surface=True)
                add("retract", line[-1], (line[-1][0], line[-1][1], retract_z), rapid_feed, pass_=index)
            last = surface_passes[-1][-1]
            add("rapid", (last[0], last[1], retract_z), (last[0], last[1], safe_z), rapid_feed, final=True)
            return segments

        if path_xy and len(path_xy) >= 2:
            loop = [(float(x) + ox, float(y) + oy) for x, y in path_xy]
            if math.dist(loop[0], loop[-1]) > 1e-6 and len(loop) > 2:
                loop.append(loop[0])
        else:
            xmin, xmax, ymin, ymax = bounds_xy
            loop = [(xmin + ox, ymin + oy), (xmax + ox, ymin + oy), (xmax + ox, ymax + oy), (xmin + ox, ymax + oy), (xmin + ox, ymin + oy)]
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(loop[:-1], loop[1:]))
        climb = str(operation.direction).casefold().startswith("climb")
        if (climb and area > 0) or (not climb and area < 0):
            loop = loop[::-1]  # climb = clockwise, conventional = counter-clockwise

        depths = pass_depths(
            operation.depth_mm, operation.stepdown_mm,
            getattr(operation, "first_pass_count", 0), getattr(operation, "first_pass_depth_mm", 0.5),
            getattr(operation, "final_pass_depth_mm", 0.0),
        )
        lead_in = max(0.0, float(getattr(operation, "lead_in_mm", 0.0)))
        lead_out = max(0.0, float(getattr(operation, "lead_out_mm", 0.0)))
        ramp_mode = str(getattr(operation, "ramp_mode", "Plunge"))
        p0, p1 = loop[0], loop[1]
        seg_len = max(math.dist(p0, p1), 1e-9)
        direction = ((p1[0] - p0[0]) / seg_len, (p1[1] - p0[1]) / seg_len)
        entry = (p0[0] - direction[0] * lead_in, p0[1] - direction[1] * lead_in)
        add("rapid", (entry[0], entry[1], safe_z), (entry[0], entry[1], safe_z), rapid_feed, pass_=0)

        previous_z = top
        for pass_idx, depth in enumerate(depths, start=1):
            z_pass = top - depth
            add("rapid", (entry[0], entry[1], safe_z if pass_idx == 1 else retract_z), (entry[0], entry[1], retract_z), rapid_feed, pass_=pass_idx)
            start_z = previous_z if ramp_mode != "Plunge" else z_pass
            add("plunge", (entry[0], entry[1], retract_z), (entry[0], entry[1], start_z), plunge_feed, pass_=pass_idx)
            if lead_in > 0:
                add("lead_in", (entry[0], entry[1], start_z), (p0[0], p0[1], start_z), cut_feed * float(self.config.finish_feed_ratio), pass_=pass_idx)
            if ramp_mode != "Plunge" and start_z > z_pass + 1e-9:
                drop = start_z - z_pass
                if ramp_mode == "Angle":
                    run = drop / math.tan(math.radians(max(0.5, float(getattr(operation, "ramp_angle_deg", 3.0)))))
                else:
                    run = max(0.1, float(getattr(operation, "ramp_length_mm", 10.0)))
                # zig-zag along the first edge until the ramp distance is covered
                legs = max(1, int(math.ceil(run / seg_len)))
                z = start_z
                here = p0
                for leg in range(legs):
                    target = p1 if leg % 2 == 0 else p0
                    leg_len = min(seg_len, run - leg * seg_len)
                    frac = leg_len / seg_len
                    end_xy = (here[0] + (target[0] - here[0]) * frac, here[1] + (target[1] - here[1]) * frac)
                    z_next = start_z - drop * min(1.0, (leg * seg_len + leg_len) / run)
                    add("ramp", (here[0], here[1], z), (end_xy[0], end_xy[1], z_next), plunge_feed, pass_=pass_idx)
                    here, z = end_xy, z_next
                if math.dist(here, p0) > 1e-9:
                    add("cut", (here[0], here[1], z_pass), (p0[0], p0[1], z_pass), cut_feed, pass_=pass_idx)
            full_width = pass_idx == 1 or depth == depths[0]
            for i in range(len(loop) - 1):
                add(
                    "cut", (loop[i][0], loop[i][1], z_pass), (loop[i + 1][0], loop[i + 1][1], z_pass),
                    cut_feed * (0.7 if full_width else 1.0), pass_=pass_idx,
                    direction="full width" if full_width else ("climb" if climb else "conventional"),
                )
            end = loop[-1]
            exit_xy = end
            if lead_out > 0:
                last_a, last_b = loop[-2], loop[-1]
                length = max(math.dist(last_a, last_b), 1e-9)
                exit_xy = (end[0] + (last_b[0] - last_a[0]) / length * lead_out, end[1] + (last_b[1] - last_a[1]) / length * lead_out)
                add("lead_out", (end[0], end[1], z_pass), (exit_xy[0], exit_xy[1], z_pass), cut_feed * float(self.config.finish_feed_ratio), pass_=pass_idx)
            add("retract", (exit_xy[0], exit_xy[1], z_pass), (exit_xy[0], exit_xy[1], retract_z), rapid_feed, pass_=pass_idx)
            add("rapid", (exit_xy[0], exit_xy[1], retract_z), (entry[0], entry[1], retract_z), rapid_feed, pass_=pass_idx)
            previous_z = z_pass

        add("rapid", (entry[0], entry[1], retract_z), (entry[0], entry[1], safe_z), rapid_feed, final=True)
        return segments

    def estimate_cycle_time_seconds(self, segments: list[ToolpathSegment]) -> float:
        total_sec = 0.0
        for seg in segments:
            dx = seg.end[0] - seg.start[0]
            dy = seg.end[1] - seg.start[1]
            dz = seg.end[2] - seg.start[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            feed_mm_sec = max(1e-3, float(seg.feed_mm_min) / 60.0)
            total_sec += dist / feed_mm_sec
        return total_sec


@dataclass
class SimulationFrame:
    """Single playback frame along a simulated toolpath."""

    index: int
    time_s: float
    position: tuple[float, float, float]
    segment_kind: str
    feed_mm_min: float
    rpm: int
    operation_name: str


class ToolpathSimulationPlayer:
    """Steps through toolpath segments at a configurable playback speed for live or stepped preview."""

    def __init__(self, segments: list[ToolpathSegment] | None = None) -> None:
        self.segments: list[ToolpathSegment] = list(segments or [])
        self.current_index: int = 0
        self.playback_rate: float = 1.0
        self.elapsed_s: float = 0.0

    def set_segments(self, segments: list[ToolpathSegment]) -> None:
        self.segments = list(segments)
        self.current_index = 0
        self.elapsed_s = 0.0

    def reset(self) -> SimulationFrame | None:
        self.current_index = 0
        self.elapsed_s = 0.0
        return self.current_frame()

    def current_frame(self) -> SimulationFrame | None:
        if not self.segments:
            return None
        idx = max(0, min(self.current_index, len(self.segments) - 1))
        seg = self.segments[idx]
        return SimulationFrame(
            index=idx,
            time_s=round(self.elapsed_s, 3),
            position=seg.end,
            segment_kind=seg.kind,
            feed_mm_min=seg.feed_mm_min,
            rpm=seg.rpm,
            operation_name=str(seg.metadata.get("op", "")),
        )

    def step(self, delta_steps: int = 1) -> SimulationFrame | None:
        if not self.segments:
            return None
        prev_idx = self.current_index
        self.current_index = max(0, min(self.current_index + int(delta_steps), len(self.segments) - 1))
        for i in range(min(prev_idx, self.current_index), max(prev_idx, self.current_index)):
            seg = self.segments[i]
            dx = seg.end[0] - seg.start[0]
            dy = seg.end[1] - seg.start[1]
            dz = seg.end[2] - seg.start[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            dt = (dist / max(1e-3, seg.feed_mm_min / 60.0)) / max(0.05, self.playback_rate)
            self.elapsed_s += dt if self.current_index >= prev_idx else -dt
        self.elapsed_s = max(0.0, self.elapsed_s)
        return self.current_frame()


@dataclass
class PostProcessorSpec:
    """Machine/controller post-processor dialect definition."""

    name: str
    controller: str  # "grbl", "linuxcnc", "mach3", "fanuc"
    line_ending: str = "\n"
    use_line_numbers: bool = False
    line_number_step: int = 10
    coolant_command: str = "M8"
    spindle_cw_command: str = "M3"
    spindle_off_command: str = "M5"
    program_header: list[str] = field(default_factory=lambda: ["G90", "G94"])
    program_footer: list[str] = field(default_factory=lambda: ["M5", "M9", "G0 Z10.0", "M30"])


class PostProcessorCompiler:
    """Compiles toolpath segments and operations into controller-specific G-code."""

    def __init__(self, spec: PostProcessorSpec | None = None) -> None:
        self.spec = spec or PostProcessorSpec(name="Generic GRBL / 3-Axis", controller="grbl")

    @staticmethod
    def _scale_words(line: str, scale: float, digits: int) -> str:
        """Convert X/Y/Z/F values written in millimetres (header/footer text) to the program units."""
        import re

        return re.sub(
            r"([XYZF])([-+]?\d*\.?\d+)",
            lambda m: f"{m.group(1)}{float(m.group(2)) * scale:.{digits}f}",
            line,
        )

    def compile_segments(
        self,
        segments: list[ToolpathSegment],
        units_code: str = "G21",
        manual_spindle: bool = False,
        coolant: bool = False,
        work_offset: str = "G54",
    ) -> str:
        """Segments are millimetres; under G20 every coordinate and feed is converted to inches."""
        inches = str(units_code).upper() == "G20"
        scale = 1.0 / 25.4 if inches else 1.0
        digits = 5 if inches else 4
        lines: list[str] = []
        lines.append(f"; Post-processor: {self.spec.name} ({self.spec.controller})")
        lines.append(units_code)
        lines.append("G90")
        lines.append(work_offset if work_offset in ("G54", "G55", "G56", "G57", "G58", "G59") else "G54")
        for hdr in self.spec.program_header:
            lines.append(hdr)

        active_rpm: int | None = None
        active_op: str | None = None
        if coolant and self.spec.coolant_command:
            lines.append(self.spec.coolant_command)

        for seg in segments:
            op_name = seg.metadata.get("op") if seg.metadata else None
            if op_name and op_name != active_op:
                tool = seg.metadata.get("tool", "")
                lines.append(f"; Operation: {op_name}" + (f" · Tool: {tool}" if tool else ""))
                active_op = op_name
            if not manual_spindle and seg.rpm > 0 and seg.rpm != active_rpm:
                lines.append(f"{self.spec.spindle_cw_command} S{int(seg.rpm)}")
                active_rpm = int(seg.rpm)

            x, y, z = (v * scale for v in seg.end)
            gcode_cmd = "G0" if seg.kind in ("rapid", "retract") else "G1"
            feed = float(seg.feed_mm_min) * scale
            if gcode_cmd == "G0":
                lines.append(f"G0 X{x:.{digits}f} Y{y:.{digits}f} Z{z:.{digits}f}")
            else:
                lines.append(f"G1 X{x:.{digits}f} Y{y:.{digits}f} Z{z:.{digits}f} F{feed:.{2 if inches else 1}f}")

        for ftr in self.spec.program_footer:
            lines.append(self._scale_words(ftr, scale, digits) if inches else ftr)

        if self.spec.use_line_numbers:
            numbered: list[str] = []
            n = int(self.spec.line_number_step)
            for idx, ln in enumerate(lines):
                if ln.startswith(";"):
                    numbered.append(ln)
                else:
                    numbered.append(f"N{(idx + 1) * n} {ln}")
            lines = numbered

        return self.spec.line_ending.join(lines) + self.spec.line_ending


@dataclass
class SetupSheetSummary:
    """Structured CAM setup/job sheet data for export or preview."""

    job_name: str
    job_notes: str
    stock_material: str
    stock_dimensions: dict[str, float]
    setup_count: int
    operation_count: int
    estimated_total_time_s: float
    operations_table: list[dict[str, Any]]
    post_processor_name: str


def build_setup_sheet_summary(
    job_name: str,
    job_notes: str,
    stock_material: str,
    stock_dimensions: dict[str, float],
    setups: list[MachiningSetup],
    dynamics: ToolpathDynamicsEngine | None = None,
    post_name: str = "Generic GRBL / 3-Axis",
) -> SetupSheetSummary:
    eng = dynamics or ToolpathDynamicsEngine()
    rows: list[dict[str, Any]] = []
    total_time = 0.0
    for s_idx, setup in enumerate(setups):
        for op in setup.operations:
            segs = eng.build_operation_segments(op, setup=setup)
            est_s = eng.estimate_cycle_time_seconds(segs)
            total_time += est_s
            rows.append(
                {
                    "setup": setup.name,
                    "setup_index": s_idx,
                    "operation": op.name,
                    "kind": op.kind,
                    "tool": op.tool,
                    "diameter_mm": op.tool_diameter_mm,
                    "rpm": op.rpm,
                    "feed_mm_min": op.feed_mm_min,
                    "stepdown_mm": op.stepdown_mm,
                    "depth_mm": op.depth_mm,
                    "direction": op.direction,
                    "est_time_s": round(est_s, 1),
                }
            )
    return SetupSheetSummary(
        job_name=job_name,
        job_notes=job_notes,
        stock_material=stock_material,
        stock_dimensions=dict(stock_dimensions),
        setup_count=len(setups),
        operation_count=len(rows),
        estimated_total_time_s=round(total_time, 1),
        operations_table=rows,
        post_processor_name=post_name,
    )


def render_setup_sheet_markdown(summary: SetupSheetSummary) -> str:
    dims = summary.stock_dimensions
    lines = [
        f"# CAM Setup Sheet — {summary.job_name}",
        "",
        f"- **Stock Material:** {summary.stock_material}",
        f"- **Stock Dimensions:** {dims.get('X', 0):.2f} × {dims.get('Y', 0):.2f} × {dims.get('Z', 0):.2f} mm",
        f"- **Setups:** {summary.setup_count} | **Operations:** {summary.operation_count}",
        f"- **Estimated Cycle Time:** {summary.estimated_total_time_s:.1f} s ({summary.estimated_total_time_s / 60.0:.2f} min)",
        f"- **Post-Processor:** {summary.post_processor_name}",
    ]
    if summary.job_notes:
        lines.extend(["", "## Notes", "", summary.job_notes])
    lines.extend(
        [
            "",
            "## Operations",
            "",
            "| Setup | Operation | Kind | Tool | Ø (mm) | RPM | Feed (mm/min) | Stepdown (mm) | Depth (mm) | Est. Time (s) |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in summary.operations_table:
        lines.append(
            f"| {r['setup']} | {r['operation']} | {r['kind']} | {r['tool']} | "
            f"{r['diameter_mm']:.2f} | {r['rpm']} | {r['feed_mm_min']:.1f} | "
            f"{r['stepdown_mm']:.2f} | {r['depth_mm']:.2f} | {r['est_time_s']:.1f} |"
        )
    return "\n".join(lines)


def write_setup_sheet_files(summary: SetupSheetSummary, out_dir: str | Path) -> dict[str, Path]:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    md_path = dest / "setup_sheet.md"
    json_path = dest / "setup_sheet.json"
    md_path.write_text(render_setup_sheet_markdown(summary), encoding="utf-8")
    payload = {
        "job_name": summary.job_name,
        "job_notes": summary.job_notes,
        "stock_material": summary.stock_material,
        "stock_dimensions": summary.stock_dimensions,
        "setup_count": summary.setup_count,
        "operation_count": summary.operation_count,
        "estimated_total_time_s": summary.estimated_total_time_s,
        "post_processor_name": summary.post_processor_name,
        "operations": summary.operations_table,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}
