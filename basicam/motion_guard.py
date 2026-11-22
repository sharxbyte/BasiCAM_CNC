"""Motion guard: one set of checks for every motion source.

Results:
  HARD      envelope would reach or cross a hard limit      -> always refused
  KEEPOUT   envelope would enter a keep-out zone            -> always refused
  COLLISION other predicted collision (fixture, or rapid through part/stock) -> alarm
  SOFT      envelope would cross a soft limit               -> refused unless that axis
            direction is Released and the `release_soft_limits` gate is on

Coordinates are machine coordinates (GRBL MPos) of the tool tip, in mm. The tool
envelope is a cylinder of radius diameter/2 + 5 mm from 5 mm below the tip up the
stick-out length. Programs are interpreted with GrblModalState (G90/G91, G20/G21,
G0-G3, G53, G54-G59); work coordinates are converted with the work offset (WCO).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Sequence

from basicam.cam_features import WCS_CODES, GrblModalState
from basicam.machine_profiles import AXES, MachineProfile, tool_stick_out

ENVELOPE_CLEARANCE_MM = 5.0
INCH_MM = 25.4

Vec = tuple[float, float, float]


class ViolationKind(str, Enum):
    HARD = "HARD"
    KEEPOUT = "KEEPOUT"
    COLLISION = "COLLISION"
    SOFT = "SOFT"


BLOCKING_ALWAYS = frozenset({ViolationKind.HARD, ViolationKind.KEEPOUT})


@dataclass(frozen=True)
class Violation:
    kind: ViolationKind
    message: str
    axis: str | None = None
    line: int | None = None

    def text(self) -> str:
        where = f" (line {self.line})" if self.line is not None else ""
        return f"{self.kind.value}{where}: {self.message}"


@dataclass(frozen=True)
class ToolEnvelope:
    radius: float
    below_tip: float = ENVELOPE_CLEARANCE_MM
    height: float = 40.0  # stick-out above the tip

    @classmethod
    def from_tool(cls, tool) -> "ToolEnvelope":
        diameter = float(getattr(tool, "diameter_mm", 6.0) or 6.0) if tool is not None else 6.0
        stick = tool_stick_out(tool) if tool is not None else 40.0
        return cls(radius=diameter / 2.0 + ENVELOPE_CLEARANCE_MM, height=max(stick, 1.0))


@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned box in machine coordinates. kind: 'keepout', 'fixture', 'stock' or 'part'."""

    name: str
    min_xyz: Vec
    max_xyz: Vec
    kind: str = "fixture"


@dataclass
class GuardState:
    position: Vec = (0.0, 0.0, 0.0)  # machine coordinates
    modal: GrblModalState = field(default_factory=GrblModalState)
    work_offsets: dict[str, Vec] = field(default_factory=dict)  # per WCS code; G54 is the reported WCO

    def wco(self) -> Vec:
        return self.work_offsets.get(self.modal.wcs, self.work_offsets.get("G54", (0.0, 0.0, 0.0)))


# ---------------------------------------------------------------------- geometry
def _envelope_hits_box(tip: Sequence[float], env: ToolEnvelope, lo: Sequence[float], hi: Sequence[float]) -> bool:
    z0, z1 = tip[2] - env.below_tip, tip[2] + env.height
    if z1 < lo[2] or z0 > hi[2]:
        return False
    dx = max(lo[0] - tip[0], 0.0, tip[0] - hi[0])
    dy = max(lo[1] - tip[1], 0.0, tip[1] - hi[1])
    return dx * dx + dy * dy < env.radius * env.radius


def _samples(start: Sequence[float], end: Sequence[float], step: float) -> list[Vec]:
    length = math.dist(start, end)
    n = max(1, int(math.ceil(length / max(step, 0.1))))
    return [tuple(start[i] + (end[i] - start[i]) * k / n for i in range(3)) for k in range(n + 1)]


def _limit_violations(machine: MachineProfile, tool, env: ToolEnvelope, point: Vec, release_allowed: bool) -> list[Violation]:
    out: list[Violation] = []
    for index, axis in enumerate(AXES):
        lo, hi = machine.hard_bounds(axis)
        if axis == "Z":
            lo = machine.z_min_for(tool)
            low_extent, high_extent = point[2] - env.below_tip, point[2]
        else:
            low_extent, high_extent = point[index] - env.radius, point[index] + env.radius
        if low_extent <= lo:
            out.append(Violation(ViolationKind.HARD, f"{axis} envelope reaches the {axis} min hard limit ({lo:.2f})", axis))
            continue
        if high_extent >= hi:
            out.append(Violation(ViolationKind.HARD, f"{axis} envelope reaches the {axis} max hard limit ({hi:.2f})", axis))
            continue
        limit = machine.soft_limits.get(axis)
        margin_min = limit.margin_min if limit else 0.0
        margin_max = limit.margin_max if limit else 0.0
        released_min = bool(limit and limit.released_min and release_allowed)
        released_max = bool(limit and limit.released_max and release_allowed)
        if not released_min and low_extent < lo + margin_min:
            out.append(Violation(ViolationKind.SOFT, f"{axis} envelope crosses the {axis} min soft limit ({lo + margin_min:.2f})", axis))
        elif not released_max and high_extent > hi - margin_max:
            out.append(Violation(ViolationKind.SOFT, f"{axis} envelope crosses the {axis} max soft limit ({hi - margin_max:.2f})", axis))
    return out


def check_move(
    machine: MachineProfile | None,
    tool,
    start: Sequence[float],
    end: Sequence[float],
    modal_state: GrblModalState | None = None,
    *,
    obstacles: Iterable[Obstacle] = (),
    release_allowed: bool = False,
    line: int | None = None,
) -> list[Violation]:
    """Check one straight move of the tool tip from start to end (machine coordinates)."""
    env = ToolEnvelope.from_tool(tool)
    cutting = bool(modal_state is not None and modal_state.motion in {"G1", "G2", "G3"})
    obstacles = list(obstacles)
    if machine is not None:
        obstacles += [Obstacle(z.name, tuple(z.min_xyz), tuple(z.max_xyz), "keepout") for z in machine.keep_out_zones]
    found: dict[tuple[ViolationKind, str | None], Violation] = {}
    for point in _samples(start, end, env.radius / 2.0):
        if machine is not None:
            for v in _limit_violations(machine, tool, env, point, release_allowed):
                found.setdefault((v.kind, v.axis), replace(v, line=line))
        for ob in obstacles:
            if not _envelope_hits_box(point, env, ob.min_xyz, ob.max_xyz):
                continue
            if ob.kind == "keepout":
                found.setdefault((ViolationKind.KEEPOUT, ob.name), Violation(ViolationKind.KEEPOUT, f"enters keep-out zone '{ob.name}'", None, line))
            elif ob.kind == "fixture":
                found.setdefault((ViolationKind.COLLISION, ob.name), Violation(ViolationKind.COLLISION, f"collides with fixture '{ob.name}'", None, line))
            elif not cutting:
                found.setdefault((ViolationKind.COLLISION, ob.name), Violation(ViolationKind.COLLISION, f"rapid move through {ob.kind} '{ob.name}'", None, line))
    # A HARD violation on an axis supersedes SOFT on the same axis.
    hard_axes = {v.axis for v in found.values() if v.kind is ViolationKind.HARD}
    return [v for v in found.values() if not (v.kind is ViolationKind.SOFT and v.axis in hard_axes)]


# ---------------------------------------------------------------------- program interpreter
_WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")


def _strip(line: str) -> str:
    line = re.sub(r"\([^)]*\)", "", line)
    return line.split(";", 1)[0].strip().upper()


def _arc_points(start: Vec, end: Vec, words: dict[str, float], clockwise: bool, scale: float) -> list[Vec]:
    sx, sy, sz = start
    ex, ey, ez = end
    if "R" in words:
        r = words["R"] * scale
        dx, dy = ex - sx, ey - sy
        d = math.hypot(dx, dy)
        if d == 0 or abs(r) < d / 2:
            return [end]
        h = math.sqrt(max(r * r - (d / 2) ** 2, 0.0))
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        sign = (-1 if clockwise else 1) * (1 if r > 0 else -1)
        cx, cy = mx - sign * h * dy / d, my + sign * h * dx / d
    else:
        cx, cy = sx + words.get("I", 0.0) * scale, sy + words.get("J", 0.0) * scale
    radius = math.hypot(sx - cx, sy - cy)
    a0, a1 = math.atan2(sy - cy, sx - cx), math.atan2(ey - cy, ex - cx)
    sweep = a1 - a0
    if clockwise and sweep >= 0:
        sweep -= 2 * math.pi
    elif not clockwise and sweep <= 0:
        sweep += 2 * math.pi
    n = max(4, int(abs(sweep) * max(radius, 1.0) / 2.0))
    return [
        (cx + radius * math.cos(a0 + sweep * k / n), cy + radius * math.sin(a0 + sweep * k / n), sz + (ez - sz) * k / n)
        for k in range(1, n + 1)
    ]


def interpret_line(raw: str, state: GuardState) -> tuple[list[Vec], GuardState]:
    """Apply one G-code line; returns the path points (machine coords) after the current position."""
    text = _strip(raw)
    if not text or text.startswith("%"):
        return [], state
    jog = text.startswith("$J=")
    if jog:
        text = text[3:]
    elif text.startswith("$"):
        return [], state  # system commands do not move (homing is handled by the caller)
    words: dict[str, float] = {}
    gcodes: list[str] = []
    for letter, value in _WORD.findall(text):
        if letter == "G":
            num = float(value)
            gcodes.append(f"G{int(num)}" if num.is_integer() else f"G{num:g}")
        elif letter == "M":
            continue
        else:
            words[letter] = float(value)
    modal = replace(state.modal)
    machine_coords = False
    for g in gcodes:
        if g in {"G0", "G1", "G2", "G3"}:
            modal.motion = g
        elif g in {"G90", "G91"}:
            modal.distance = g
        elif g in {"G20", "G21"}:
            modal.units = g
        elif g in WCS_CODES:
            modal.wcs = g
        elif g == "G53":
            machine_coords = True
    if "F" in words:
        modal.feed_rate = words["F"]
    new_state = replace(state, modal=modal)
    if jog:
        modal_for_move = replace(modal, motion="G1")
    else:
        modal_for_move = modal
    if not any(axis in words for axis in AXES) or any(g in {"G28", "G30", "G92", "G10"} for g in gcodes):
        return [], new_state  # no axis words, or homing/offset commands handled elsewhere
    scale = INCH_MM if modal.units == "G20" else 1.0
    wco = new_state.wco()
    target = list(state.position)
    for index, axis in enumerate(AXES):
        if axis not in words:
            continue
        value = words[axis] * scale
        if modal.distance == "G91" and not machine_coords:
            target[index] = state.position[index] + value
        elif machine_coords:
            target[index] = value
        else:
            target[index] = value + wco[index]
    end: Vec = tuple(target)
    if modal_for_move.motion in {"G2", "G3"} and not jog:
        points = _arc_points(state.position, end, words, modal_for_move.motion == "G2", scale)
    else:
        points = [end]
    new_state.position = end
    if jog:
        new_state.modal = replace(state.modal)  # $J= modal words apply to the jog only
    return points, new_state


def check_program(
    machine: MachineProfile | None,
    tool,
    lines: Iterable[str],
    start_state: GuardState | None = None,
    *,
    obstacles: Iterable[Obstacle] = (),
    release_allowed: bool = False,
    stop_after: int = 200,
) -> list[Violation]:
    """Full pre-check of a program; returns violations with 1-based line numbers."""
    state = start_state or GuardState()
    obstacles = list(obstacles)
    violations: list[Violation] = []
    for number, raw in enumerate(lines, start=1):
        before = state
        points, state = interpret_line(raw, state)
        prev = before.position
        moving_modal = state.modal if not _strip(raw).startswith("$J=") else replace(state.modal, motion="G1")
        for point in points:
            violations.extend(
                check_move(machine, tool, prev, point, moving_modal, obstacles=obstacles, release_allowed=release_allowed, line=number)
            )
            prev = point
        if len(violations) >= stop_after:
            break
    return violations


def blocking(violations: Iterable[Violation]) -> bool:
    """Every violation must be fixed before proceeding (SOFT is already filtered for released axes)."""
    return any(True for _ in violations)


# ---------------------------------------------------------------------- window adapter
class MotionGuard:
    """Binds the checks to the main window's machine, tool, position, fixtures and gates."""

    def __init__(self, window) -> None:
        self.window = window

    # context ----------------------------------------------------------
    def machine(self) -> MachineProfile | None:
        return getattr(self.window, "current_machine", None)

    def tool(self):
        w = self.window
        index = w.tool_name.currentIndex() if hasattr(w, "tool_name") else -1
        tools = getattr(w, "tools", [])
        return tools[index] if 0 <= index < len(tools) else None

    def release_allowed(self) -> bool:
        return bool(self.window.safety.allows("release_soft_limits"))

    def work_offset(self) -> Vec:
        return tuple(getattr(self.window, "last_wco", None) or (0.0, 0.0, 0.0))

    def position(self) -> Vec:
        status = getattr(self.window, "grbl_status", None)
        if status is not None and status.machine_position:
            return tuple(status.machine_position[:3])
        if status is not None and status.work_position:
            wco = self.work_offset()
            return tuple(status.work_position[i] + wco[i] for i in range(3))
        tracked = getattr(self.window, "tracked_work_pos", [0.0, 0.0, 0.0])
        wco = self.work_offset()
        return tuple(float(tracked[i]) + wco[i] for i in range(3))

    def obstacles(self) -> list[Obstacle]:
        out: list[Obstacle] = []
        manager = getattr(self.window, "fixture_manager", None)
        wco = self.work_offset()
        for fixture in getattr(manager, "fixtures", []) or []:
            if not getattr(fixture, "enabled", True):
                continue
            px, py, pz = fixture.position_mm
            dx, dy, dz = fixture.dimensions_mm
            lo = (px - dx / 2 + wco[0], py - dy / 2 + wco[1], pz + wco[2])
            hi = (px + dx / 2 + wco[0], py + dy / 2 + wco[1], pz + dz + wco[2])
            kind = "keepout" if getattr(fixture, "keep_out", False) else "fixture"
            out.append(Obstacle(fixture.name, lo, hi, kind))
        return out

    def start_state(self) -> GuardState:
        modal = getattr(self.window, "grbl_modal_state", None) or GrblModalState()
        return GuardState(position=self.position(), modal=replace(modal), work_offsets={"G54": self.work_offset()})

    # checks -----------------------------------------------------------
    def check_jog(self, axis: str, distance: float) -> list[Violation]:
        start = self.position()
        end = list(start)
        end[AXES.index(axis)] += float(distance)
        return check_move(
            self.machine(), self.tool(), start, tuple(end), replace(GrblModalState(), motion="G1"),
            obstacles=self.obstacles(), release_allowed=self.release_allowed(),
        )

    def check_line(self, command: str) -> list[Violation]:
        return check_program(
            self.machine(), self.tool(), [command], self.start_state(),
            obstacles=self.obstacles(), release_allowed=self.release_allowed(),
        )

    def work_offset_known(self) -> bool:
        return getattr(self.window, "last_wco", None) is not None

    def check_program_text(self, text: str, start: GuardState | None = None) -> list[Violation]:
        """Pre-check a whole program. Without a known work offset the machine limits cannot be
        placed, so only fixtures (in work coordinates) are checked; streams re-check when connected."""
        machine = self.machine() if self.work_offset_known() else None
        if start is None:
            wco = self.work_offset()
            start = GuardState(position=wco, work_offsets={"G54": wco})
        return check_program(
            machine, self.tool(), text.splitlines(), start,
            obstacles=self.obstacles(), release_allowed=self.release_allowed(),
        )

    def report(self, violations: list[Violation], title: str, show_dialog: bool = True) -> bool:
        """Log violations and raise an alarm; returns True when the motion may proceed."""
        if not violations:
            return True
        w = self.window
        lines = [v.text() for v in violations[:12]]
        if len(violations) > 12:
            lines.append(f"... and {len(violations) - 12} more")
        for line in lines:
            w.terminal_core.log("error" if violations[0].kind in BLOCKING_ALWAYS else "warning", f"{title}: {line}")
        if any(v.kind is ViolationKind.SOFT for v in violations):
            lines.append("Check that the part fits the work area and that the operation is possible.")
        if show_dialog:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(w, f"Motion guard · {title}", "Refused; fix this before proceeding:\n\n" + "\n".join(lines))
        return False
