"""Consumable and usage tracking per machine and per job (no Qt).

Counted from the program that was actually streamed (acknowledged lines), estimated before a
run from the program. Per machine (QSettings JSON ``usage/<machine id>``): spindle hours, laser
beam-on and lens hours, plasma pierces and arc-on time, waterjet pump, nozzle hours and abrasive
used, EDM wire length, tool changes and gas time; maintenance thresholds per item raise a
warning once passed; "Reset counter" keeps a dated log.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field

ITEMS = {
    "spindle_h": ("Spindle", "h"),
    "beam_h": ("Laser beam on", "h"),
    "lens_h": ("Lens since cleaning", "h"),
    "pierces": ("Pierces", ""),
    "arc_h": ("Plasma arc on", "h"),
    "pump_h": ("Waterjet pump", "h"),
    "nozzle_h": ("Nozzle / orifice", "h"),
    "abrasive_kg": ("Abrasive used", "kg"),
    "wire_m": ("EDM wire used", "m"),
    "tool_changes": ("Tool changes", ""),
    "gas_h": ("Air / gas", "h"),
}
PROCESS_ITEMS = {
    "mill": ("spindle_h", "tool_changes"),
    "lathe": ("spindle_h", "tool_changes"),
    "mill_turn": ("spindle_h", "tool_changes"),
    "laser": ("beam_h", "lens_h", "pierces", "gas_h"),
    "rotary_laser": ("beam_h", "lens_h", "pierces", "gas_h"),
    "laser_lathe": ("beam_h", "lens_h", "spindle_h"),
    "plasma": ("arc_h", "pierces", "gas_h"),
    "waterjet": ("pump_h", "nozzle_h", "abrasive_kg", "pierces"),
    "wire_edm": ("wire_m",),
}
_WORD = re.compile(r"([A-Z])\s*(-?\d+(?:\.\d*)?|-?\.\d+)")


@dataclass
class JobUsage:
    items: dict[str, float] = field(default_factory=dict)
    cut_minutes: float = 0.0

    def add(self, other: "JobUsage") -> None:
        for key, value in other.items.items():
            self.items[key] = self.items.get(key, 0.0) + value
        self.cut_minutes += other.cut_minutes


def usage_from_program(lines, process: str = "mill", abrasive_kg_min: float = 0.35, wire_m_min: float = 0.1) -> JobUsage:
    """Usage of the given program lines: beam / arc / jet / spindle time while it is on, pierces
    (each switch-on), tool changes, abrasive from the flow rate and wire from its feed."""
    x = y = z = 0.0
    feed = 0.0
    on = False
    on_minutes = 0.0
    spindle_minutes = 0.0
    pierces = 0
    changes = 0
    inches = False
    for raw in lines:
        code = re.sub(r"\([^)]*\)", "", str(raw)).split(";", 1)[0].upper()
        if not code.strip():
            continue
        words = {k: float(v) for k, v in _WORD.findall(code)}
        if "G20" in code:
            inches = True
        if "G21" in code:
            inches = False
        scale = 25.4 if inches else 1.0
        if re.search(r"\bM0?6\b", code) or re.search(r"\bM0\b", code) and "CHANGE TOOL" in str(raw).upper():
            changes += 1
        if re.search(r"\bM0?[34]\b", code):
            power = words.get("S", 1.0)
            if not on and power > 0:
                pierces += 1
            on = power > 0
        if re.search(r"\bM0?5\b", code):
            on = False
        if "F" in words:
            feed = words["F"] * scale
        if any(k in words for k in "XYZ"):
            nx, ny, nz = (words.get(k, v / scale) * scale for k, v in (("X", x), ("Y", y), ("Z", z)))
            distance = math.dist((x, y, z), (nx, ny, nz))
            moving = re.search(r"\bG0?[123]\b", code) is not None and not re.search(r"\bG0?0\b", code)
            if moving and feed > 0:
                minutes = distance / feed
                if on:
                    on_minutes += minutes
                spindle_minutes += minutes if on else 0.0
            x, y, z = nx, ny, nz
    usage = JobUsage(cut_minutes=on_minutes)
    hours = on_minutes / 60.0
    items = PROCESS_ITEMS.get(process, PROCESS_ITEMS["mill"])
    values = {
        "spindle_h": spindle_minutes / 60.0, "beam_h": hours, "lens_h": hours, "arc_h": hours,
        "pump_h": hours, "nozzle_h": hours, "abrasive_kg": on_minutes * abrasive_kg_min,
        "wire_m": on_minutes * wire_m_min, "pierces": float(pierces), "tool_changes": float(changes), "gas_h": hours,
    }
    usage.items = {k: values[k] for k in items}
    if changes and "tool_changes" not in usage.items:
        usage.items["tool_changes"] = float(changes)
    return usage


@dataclass
class MachineUsage:
    counters: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)  # item -> limit (warn once passed)
    reset_log: list[dict] = field(default_factory=list)

    def add(self, usage: JobUsage) -> list[str]:
        """Add a job; returns the warnings for thresholds this job passed."""
        warnings = []
        for key, value in usage.items.items():
            before = self.counters.get(key, 0.0)
            after = before + value
            self.counters[key] = after
            limit = self.thresholds.get(key)
            if limit and before < limit <= after:
                label, unit = ITEMS.get(key, (key, ""))
                warnings.append(f"{label}: {after:.2f} {unit} - maintenance due (threshold {limit:g} {unit}).".replace("  ", " "))
        return warnings

    def due(self) -> list[str]:
        return [k for k, limit in self.thresholds.items() if limit and self.counters.get(k, 0.0) >= limit]

    def reset(self, key: str, note: str = "") -> None:
        self.reset_log.append({"item": key, "value": self.counters.get(key, 0.0), "time": time.strftime("%Y-%m-%d %H:%M"), "note": note})
        self.counters[key] = 0.0


class UsageStore:
    PREFIX = "usage/"

    def __init__(self, settings) -> None:
        self.settings = settings

    def get(self, machine_id: str) -> MachineUsage:
        raw = self.settings.value(self.PREFIX + machine_id, "") if self.settings is not None else ""
        try:
            data = json.loads(raw) if isinstance(raw, str) and raw else {}
        except ValueError:
            data = {}
        return MachineUsage(
            {k: float(v) for k, v in dict(data.get("counters", {})).items()},
            {k: float(v) for k, v in dict(data.get("thresholds", {})).items()},
            list(data.get("reset_log", [])),
        )

    def save(self, machine_id: str, usage: MachineUsage) -> None:
        self.settings.setValue(self.PREFIX + machine_id, json.dumps(asdict(usage)))


def format_usage(usage: JobUsage) -> str:
    parts = []
    for key, value in usage.items.items():
        label, unit = ITEMS.get(key, (key, ""))
        shown = f"{value:.0f}" if unit == "" else f"{value:.3f} {unit}"
        parts.append(f"{label} {shown}")
    return "; ".join(parts) or "no usage"


__all__ = ["ITEMS", "JobUsage", "MachineUsage", "PROCESS_ITEMS", "UsageStore", "format_usage", "usage_from_program"]
