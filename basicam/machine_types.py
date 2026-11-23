"""Machine processes, kinematic layouts, axes, heads and controllers (no Qt).

Every machine has a process (what it cuts with), a kinematic layout (how its axes move the tool
relative to the work), an axis list (linear in mm, rotary in degrees or continuous) and one or
more heads (a router with a laser module, a mill-turn...). Machines saved before v2 load as a
3-axis gantry mill.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

PROCESSES = {
    "mill": "Mill / router",
    "laser": "Laser",
    "plasma": "Plasma",
    "waterjet": "Waterjet",
    "wire_edm": "Wire EDM",
    "lathe": "Lathe",
    "mill_turn": "Mill-turn",
    "rotary_laser": "Rotary laser",
    "laser_lathe": "Laser lathe",
}
TWO_D_PROCESSES = ("laser", "plasma", "waterjet", "wire_edm")
LATHE_PROCESSES = ("lathe", "mill_turn", "laser_lathe")

LAYOUTS = {
    "gantry3": ("X", "Y", "Z"),
    "gantry3_rotary": ("X", "Y", "Z", "A"),  # 4th axis: A about X (or B about Y, see rotary_axis)
    "table_table5": ("X", "Y", "Z", "A", "C"),  # A/C trunnion (or B/C)
    "head_table5": ("X", "Y", "Z", "B", "C"),  # B tilting head, C rotary table
    "lathe2": ("X", "Z"),  # X in diameter (or radius) mode
    "lathe_live": ("X", "Z", "C"),  # live tooling: C = spindle as a rotary axis (optional Y)
    "uv_taper": ("X", "Y", "U", "V"),  # wire EDM / tilting head taper
}
LAYOUT_LABELS = {
    "gantry3": "3-axis gantry",
    "gantry3_rotary": "3 axes + rotary (4th axis)",
    "table_table5": "5-axis table-table (trunnion)",
    "head_table5": "5-axis head-table",
    "lathe2": "2-axis lathe (X/Z)",
    "lathe_live": "Lathe with live tooling (X/Z/C)",
    "uv_taper": "Taper head (X/Y/U/V)",
}
ROTARY_NAMES = ("A", "B", "C")


@dataclass
class ControllerSpec:
    name: str
    max_axes: int
    streams: bool  # BasiCAM sends the program over serial (GRBL protocol); else export only
    lathe_mode: bool = False
    inverse_time: bool = True  # G93
    threading: bool = False  # G76 / G33


CONTROLLERS = {
    "GRBL": ControllerSpec("GRBL", 3, True, lathe_mode=False, inverse_time=True),
    "grblHAL": ControllerSpec("grblHAL", 6, True, lathe_mode=True, inverse_time=True, threading=True),
    "FluidNC": ControllerSpec("FluidNC", 6, True, lathe_mode=False, inverse_time=True),
    "Export only (LinuxCNC style)": ControllerSpec("Export only (LinuxCNC style)", 9, False, True, True, True),
    "Export only (industrial laser / waterjet / EDM)": ControllerSpec("Export only (industrial laser / waterjet / EDM)", 9, False),
}


def controller_spec(name: str) -> ControllerSpec:
    return CONTROLLERS.get(name) or CONTROLLERS["GRBL"]


@dataclass
class AxisSpec:
    name: str
    kind: str = "linear"  # "linear" (mm) or "rotary" (degrees)
    minimum: float = 0.0
    maximum: float = 300.0
    continuous: bool = False  # rotary with no end stops
    home: float = 0.0
    max_rate: float = 3000.0  # mm/min or deg/min
    acceleration: float = 500.0  # mm/s^2 or deg/s^2

    @classmethod
    def from_dict(cls, data: dict) -> "AxisSpec":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**values)


@dataclass
class Head:
    name: str = "Spindle"
    process: str = "mill"
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # from the main spindle

    @classmethod
    def from_dict(cls, data: dict) -> "Head":
        data = dict(data or {})
        return cls(str(data.get("name", "Spindle")), str(data.get("process", "mill")),
                   tuple(float(v) for v in data.get("offset_mm", (0.0, 0.0, 0.0))))


def default_axes(layout: str, travel_mm=(300.0, 300.0, 80.0), homing_direction: str = "max") -> list[AxisSpec]:
    names = LAYOUTS.get(layout, LAYOUTS["gantry3"])
    out = []
    travel = dict(zip(("X", "Y", "Z"), travel_mm))
    for name in names:
        if name in ROTARY_NAMES:
            if name == "C" or layout in ("gantry3_rotary",):
                out.append(AxisSpec(name, "rotary", -360.0, 360.0, continuous=True, max_rate=10800.0))
            else:
                out.append(AxisSpec(name, "rotary", -120.0, 120.0, max_rate=5400.0))
        elif name in ("U", "V"):
            out.append(AxisSpec(name, "linear", -10.0, 10.0, max_rate=600.0))
        else:
            span = float(travel.get(name, 300.0))
            lo, hi = (-span, 0.0) if homing_direction == "max" else (0.0, span)
            out.append(AxisSpec(name, "linear", lo, hi, max_rate=3000.0))
    return out


def axis_problems(axes: list[AxisSpec], controller: str) -> list[str]:
    """Why a machine cannot run on its controller (more axes than the controller has...)."""
    spec = controller_spec(controller)
    problems = []
    if len(axes) > spec.max_axes:
        problems.append(f"{spec.name} drives at most {spec.max_axes} axes; this machine has {len(axes)} ({', '.join(a.name for a in axes)}).")
    names = [a.name for a in axes]
    if len(set(names)) != len(names):
        problems.append("Two axes share a name.")
    for axis in axes:
        if axis.kind == "rotary" and not axis.continuous and axis.minimum >= axis.maximum:
            problems.append(f"Rotary axis {axis.name} needs minimum < maximum (or continuous).")
    return problems


__all__ = [
    "AxisSpec", "CONTROLLERS", "ControllerSpec", "Head", "LATHE_PROCESSES", "LAYOUTS", "LAYOUT_LABELS",
    "PROCESSES", "ROTARY_NAMES", "TWO_D_PROCESSES", "axis_problems", "controller_spec", "default_axes",
]
