"""Bar feeders and automatic advance for lathes (no Qt).

A machine's bar-feed option is none, a bar puller (a gripper in a turret station, moved by
G-code) or an external bar feeder (a user-set M-code or output starts it; BasiCAM waits for the
"bar fed" input, with a timeout). A production run repeats the program: part off, retract, stop
the spindle, open the chuck, advance the bar to the stock stop, close the chuck, continue -
looped by BasiCAM's sender (GRBL has no M99) and counted. The chuck only ever opens with the
spindle stopped.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields


@dataclass
class BarFeed:
    kind: str = "none"  # "none", "puller", "feeder"
    bar_length_mm: float = 1000.0
    remnant_mm: float = 100.0  # unusable end left in the chuck
    part_length_mm: float = 40.0
    parting_width_mm: float = 3.0
    stock_stop_z_mm: float = 5.0  # where the bar end stops (part front face side), lathe z from the work zero
    puller_station: int = 0
    puller_x_mm: float = 0.0  # gripper centre on the axis
    chuck_open: str = "M10"
    chuck_close: str = "M11"
    feeder_trigger: str = "M64 P7"  # starts an external feeder
    feeder_done_input: int = 3  # input that reports "bar fed"
    feeder_timeout_s: float = 30.0
    spindle_stop_wait_s: float = 2.0

    def parts_per_bar(self) -> int:
        usable = self.bar_length_mm - self.remnant_mm
        pitch = self.part_length_mm + self.parting_width_mm
        return max(0, int(math.floor((usable + 1e-9) / pitch))) if pitch > 0 else 0

    @classmethod
    def from_dict(cls, data: dict | None) -> "BarFeed":
        known = {f.name for f in fields(cls)}
        feed = cls(**{k: v for k, v in dict(data or {}).items() if k in known})
        if feed.kind not in ("none", "puller", "feeder"):
            feed.kind = "none"
        return feed

    def to_dict(self) -> dict:
        return asdict(self)


def advance_lines(feed: BarFeed, safe_x_mm: float = 60.0) -> list[str]:
    """Between parts: spindle stopped before the chuck opens, bar advanced, chuck closed."""
    pitch = feed.part_length_mm + feed.parting_width_mm
    lines = ["(Bar advance)", f"G0 X{safe_x_mm:.3f}", "M5", f"G4 P{feed.spindle_stop_wait_s:.2f}"]
    if feed.kind == "puller":
        lines += [
            f"T{feed.puller_station} (bar puller)" if feed.puller_station else "(bar puller)",
            f"G0 X{feed.puller_x_mm:.3f} Z{feed.stock_stop_z_mm - pitch:.3f}",
            feed.chuck_open,
            "G4 P0.5",
            f"G1 Z{feed.stock_stop_z_mm:.3f} F500",
            feed.chuck_close,
            "G4 P0.5",
            f"G0 X{safe_x_mm:.3f}",
        ]
    elif feed.kind == "feeder":
        lines += [
            feed.chuck_open,
            "G4 P0.5",
            feed.feeder_trigger,
            f"M66 P{feed.feeder_done_input} L3 Q{feed.feeder_timeout_s:.1f}",  # wait for "bar fed"
            feed.chuck_close,
            "G4 P0.5",
        ]
    return lines


@dataclass
class ProductionRun:
    feed: BarFeed
    target: int = 0  # 0 = until the bar ends
    made: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def bar_end(self) -> bool:
        return self.made >= self.feed.parts_per_bar()

    def part_done(self) -> bool:
        """Count a finished part; True when another cycle should run."""
        self.made += 1
        if self.bar_end:
            self.log.append(f"Bar end after {self.made} parts.")
            return False
        if self.target and self.made >= self.target:
            return False
        return True

    def bar_used_mm(self) -> float:
        return self.made * (self.feed.part_length_mm + self.feed.parting_width_mm)

    def remnant_mm(self) -> float:
        return self.feed.bar_length_mm - self.bar_used_mm()


__all__ = ["BarFeed", "ProductionRun", "advance_lines"]
