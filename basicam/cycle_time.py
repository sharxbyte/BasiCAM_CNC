"""Live cycle-time estimate and the speed-multiplier rules (no Qt).

Time = remaining distance per segment / that segment's feed, scaled by the current
speed multiplier; rapids use the machine rapid rate and are never sped up. When the
multiplier changes only the remaining projection changes; elapsed time is kept.
Tool changes without a reposition add 2 minutes each; manual repositions to change
the part orientation add 7 minutes each.

GRBL 1.1 overrides: feed and spindle 10-200% (realtime 0x90-0x9D), rapids 25/50/100%.
Beyond 200% the app scales the F words it streams, up to 10x, clamped to the machine
maximum rate; that needs the `speed_multiplier_over_limit` gate.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

TOOL_CHANGE_S = 120.0
REPOSITION_S = 420.0
GRBL_OVERRIDE_MAX = 2.0
GRBL_OVERRIDE_MIN = 0.1
SOFTWARE_MAX = 10.0
RAPID_KINDS = {"rapid", "retract"}

FEED_RESET, FEED_PLUS_10, FEED_MINUS_10, FEED_PLUS_1, FEED_MINUS_1 = b"\x90", b"\x91", b"\x92", b"\x93", b"\x94"
SPINDLE_RESET, SPINDLE_PLUS_10, SPINDLE_MINUS_10, SPINDLE_PLUS_1, SPINDLE_MINUS_1 = b"\x99", b"\x9a", b"\x9b", b"\x9c", b"\x9d"


def _length(seg) -> float:
    """Travel of a move; wrapped rotary moves add their distance on the cylinder surface."""
    straight = math.dist(seg.start, seg.end)
    surface = float((getattr(seg, "metadata", None) or {}).get("surface_mm", 0.0) or 0.0)
    return math.hypot(straight, surface) if surface > 0 else straight


@dataclass
class Allowance:
    index: int  # applied before this segment
    seconds: float
    reason: str


class CycleEstimator:
    def __init__(self, segments: list, machine=None, default_rapid_mm_min: float = 3000.0) -> None:
        self.segments = list(segments)
        rapid = getattr(machine, "rapid_mm_min", None) if machine is not None else None
        self.rapid_mm_min = float(rapid or default_rapid_mm_min)
        changer = getattr(machine, "tool_changer", None) if machine is not None else None
        self.tool_change_s = float(changer.time_s()) if changer is not None else TOOL_CHANGE_S
        self.allowances = self._detect_allowances()
        self._cut_s = [self._seg_time(s) if s.kind not in RAPID_KINDS else 0.0 for s in self.segments]
        self._rapid_s = [_length(s) / self.rapid_mm_min * 60.0 if s.kind in RAPID_KINDS else 0.0 for s in self.segments]
        self.elapsed_s = 0.0
        self._last_tick: float | None = None

    @staticmethod
    def _seg_time(seg) -> float:
        return _length(seg) / max(float(seg.feed_mm_min), 1e-3) * 60.0

    def _detect_allowances(self) -> list[Allowance]:
        out: list[Allowance] = []
        prev_tool = prev_setup = None
        for index, seg in enumerate(self.segments):
            meta = seg.metadata or {}
            tool, setup = meta.get("tool"), meta.get("setup")
            if index and setup is not None and prev_setup is not None and setup != prev_setup:
                out.append(Allowance(index, REPOSITION_S, f"reposition to {setup}"))
            elif index and tool and prev_tool and tool != prev_tool:
                out.append(Allowance(index, getattr(self, "tool_change_s", TOOL_CHANGE_S), f"tool change to {tool}"))
            prev_tool = tool or prev_tool
            prev_setup = setup if setup is not None else prev_setup
        return out

    def remaining(self, position_index: int = 0, multiplier: float = 1.0) -> float:
        """Seconds left from segment `position_index`; cutting moves scale with the multiplier."""
        m = max(float(multiplier), 1e-3)
        i = max(0, int(position_index))
        cut = sum(self._cut_s[i:]) / m
        rapid = sum(self._rapid_s[i:])
        extra = sum(a.seconds for a in self.allowances if a.index >= max(i, 1))
        return cut + rapid + extra

    def total(self, multiplier: float = 1.0) -> float:
        return self.remaining(0, multiplier)

    def tick(self, now: float | None = None) -> None:
        """Accumulate wall-clock elapsed time while running (independent of the multiplier)."""
        now = time.monotonic() if now is None else now
        if self._last_tick is not None:
            self.elapsed_s += max(0.0, now - self._last_tick)
        self._last_tick = now

    def pause(self) -> None:
        self._last_tick = None

    def operation_bounds(self, position_index: int) -> tuple[int, int, str]:
        if not self.segments:
            return 0, 0, ""
        i = min(max(position_index, 0), len(self.segments) - 1)
        op = (self.segments[i].metadata or {}).get("op", "")
        start = i
        while start > 0 and (self.segments[start - 1].metadata or {}).get("op", "") == op:
            start -= 1
        end = i
        while end + 1 < len(self.segments) and (self.segments[end + 1].metadata or {}).get("op", "") == op:
            end += 1
        return start, end + 1, op

    def operation_times(self, position_index: int, multiplier: float = 1.0) -> tuple[str, float, float]:
        """(operation name, elapsed estimate, remaining) for the operation at `position_index`."""
        start, end, op = self.operation_bounds(position_index)
        m = max(float(multiplier), 1e-3)
        done = sum(self._cut_s[start:position_index]) + sum(self._rapid_s[start:position_index])
        left = sum(self._cut_s[position_index:end]) / m + sum(self._rapid_s[position_index:end])
        return op, done, left


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------- multiplier rules
def clamp_multiplier(requested: float, gate_allows: bool, machine_max_feed: float | None = None, program_max_feed: float | None = None) -> float:
    """Effective multiplier: 10-200% by GRBL override; beyond 200% only with the gate, up to 10x,
    and never so far that the fastest programmed feed exceeds the machine maximum."""
    value = max(GRBL_OVERRIDE_MIN, float(requested))
    limit = SOFTWARE_MAX if gate_allows else GRBL_OVERRIDE_MAX
    value = min(value, limit)
    if value > GRBL_OVERRIDE_MAX and machine_max_feed and program_max_feed:
        value = min(value, max(GRBL_OVERRIDE_MAX, machine_max_feed / max(program_max_feed, 1e-6)))
    return round(value, 3)


def split_multiplier(effective: float) -> tuple[float, float]:
    """(GRBL override fraction <= 2.0, software F-word scale applied when streaming)."""
    override = min(max(effective, GRBL_OVERRIDE_MIN), GRBL_OVERRIDE_MAX)
    software = effective / override if effective > GRBL_OVERRIDE_MAX else 1.0
    return override, software


def override_bytes(target: float, kind: str = "feed") -> list[bytes]:
    """Realtime bytes that set the GRBL feed or spindle override to `target` (fraction, 0.1-2.0)."""
    pct = int(round(min(max(target, GRBL_OVERRIDE_MIN), GRBL_OVERRIDE_MAX) * 100))
    if kind == "spindle":
        reset, p10, m10, p1, m1 = SPINDLE_RESET, SPINDLE_PLUS_10, SPINDLE_MINUS_10, SPINDLE_PLUS_1, SPINDLE_MINUS_1
    else:
        reset, p10, m10, p1, m1 = FEED_RESET, FEED_PLUS_10, FEED_MINUS_10, FEED_PLUS_1, FEED_MINUS_1
    out = [reset]
    delta = pct - 100
    tens, ones = divmod(abs(delta), 10)
    out += [p10 if delta > 0 else m10] * tens
    out += [p1 if delta > 0 else m1] * ones
    return out


def override_percent(sequence: Iterable[bytes], kind: str = "feed") -> int:
    pct = 100
    table = (
        {SPINDLE_RESET: None, SPINDLE_PLUS_10: 10, SPINDLE_MINUS_10: -10, SPINDLE_PLUS_1: 1, SPINDLE_MINUS_1: -1}
        if kind == "spindle"
        else {FEED_RESET: None, FEED_PLUS_10: 10, FEED_MINUS_10: -10, FEED_PLUS_1: 1, FEED_MINUS_1: -1}
    )
    for b in sequence:
        step = table.get(b, 0)
        pct = 100 if step is None else max(10, min(200, pct + step))
    return pct


_F_WORD = re.compile(r"F([-+]?\d*\.?\d+)", re.IGNORECASE)


def scale_feed_line(line: str, factor: float, max_feed: float | None = None) -> str:
    """Scale F words of a streamed line (software override beyond 200%), clamped to max_feed."""
    if factor == 1.0 or line.lstrip().startswith((";", "(")) or "G0" in line.upper().replace("G01", "G1").split(";")[0].split():
        return line

    def repl(match):
        value = float(match.group(1)) * factor
        if max_feed:
            value = min(value, max_feed)
        return f"F{value:.1f}"

    return _F_WORD.sub(repl, line)


def parse_grbl_settings(lines: Iterable[str]) -> dict[int, float]:
    """`$$` output -> {110: 5000.0, ...}."""
    out: dict[int, float] = {}
    for line in lines:
        m = re.match(r"^\$(\d+)=([-+]?\d*\.?\d+)", line.strip())
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def machine_max_feed(grbl_settings: dict[int, float], machine=None) -> float | None:
    """Slowest of the axis maximum rates ($110-$112) and the machine profile maximum feed."""
    values = [grbl_settings[k] for k in (110, 111, 112) if k in grbl_settings]
    profile = getattr(machine, "max_feed_mm_min", None) if machine is not None else None
    if profile:
        values.append(min(profile) if isinstance(profile, (tuple, list)) else float(profile))
    return min(values) if values else None


@dataclass
class MultiplierState:
    """Per-session confirmation memory for multipliers above 100%."""

    confirmed_above_100: bool = False
    history: list[float] = field(default_factory=list)
