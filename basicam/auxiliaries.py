"""Auxiliary outputs: air assist, process gas, coolant, cutting fluid, water, vacuum, dust and
fume extraction (no Qt).

Each machine lists the auxiliaries it has and the commands that switch them (M7 / M8 / M9, a
grblHAL or FluidNC digital output M62-M65 / M64 P<n>, or a user M-code), with optional delays
before and after (gas purge, pump spin-up) and a "required" flag: a program that needs a
required auxiliary is refused when its feedback input says it is not running.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

AUX_KINDS = {
    "air_assist": "Air assist",
    "process_gas": "Process gas",
    "flood": "Flood coolant",
    "mist": "Mist coolant",
    "cutting_fluid": "Cutting fluid / through-spindle coolant",
    "water": "Water flow (waterjet, water table, EDM flushing)",
    "vacuum": "Vacuum hold-down",
    "dust": "Dust collection",
    "fume": "Fume extraction",
}
GASES = ("Air", "O2", "N2", "Argon mix")
DEFAULT_COMMANDS = {
    "flood": ("M8", "M9"),
    "mist": ("M7", "M9"),
    "air_assist": ("M8", "M9"),  # GRBL laser builds usually wire air assist to the coolant output
    "process_gas": ("M64 P1", "M65 P1"),
    "cutting_fluid": ("M64 P2", "M65 P2"),
    "water": ("M64 P3", "M65 P3"),
    "vacuum": ("M64 P4", "M65 P4"),
    "dust": ("M64 P5", "M65 P5"),
    "fume": ("M64 P6", "M65 P6"),
}


@dataclass
class AuxOutput:
    kind: str
    on_command: str = ""
    off_command: str = ""
    pre_delay_s: float = 0.0  # wait after switching on (purge / spin-up)
    post_delay_s: float = 0.0  # wait after switching off
    required: bool = False
    feedback_input: str = ""  # controller pin that reports it running (e.g. "P1")
    gas: str = "Air"
    pressure_bar: float = 0.0

    def commands(self) -> tuple[str, str]:
        default = DEFAULT_COMMANDS.get(self.kind, ("", ""))
        return (self.on_command or default[0], self.off_command or default[1])

    @classmethod
    def from_dict(cls, data: dict) -> "AuxOutput":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(data or {}).items() if k in known})


def default_aux_for(process: str, material_family: str = "") -> list[str]:
    """Auxiliaries an operation starts with, from the process and the material family."""
    family = (material_family or "").casefold()
    if process in ("laser", "rotary_laser", "laser_lathe"):
        return ["air_assist", "fume"]
    if process == "plasma":
        return ["process_gas", "fume"]
    if process == "waterjet":
        return ["water"]
    if process == "wire_edm":
        return ["water"]
    if family in ("wood", "foam", "composite"):
        return ["dust"]
    if family == "metal":
        return ["flood"]
    return []


def aux_lines(outputs: list[AuxOutput], kinds: list[str], on: bool) -> list[str]:
    """G-code that switches the given auxiliaries on (or off), with their delays (G4 P<s>)."""
    by_kind = {o.kind: o for o in outputs}
    lines: list[str] = []
    seen: set[str] = set()
    for kind in kinds:
        output = by_kind.get(kind)
        if output is None:
            continue
        command = output.commands()[0 if on else 1]
        if command and command not in seen:
            lines.append(command)
            seen.add(command)
        delay = output.pre_delay_s if on else output.post_delay_s
        if delay > 0:
            lines.append(f"G4 P{delay:.2f}")
    return lines


def missing_required(outputs: list[AuxOutput], kinds: list[str], pins: str) -> list[str]:
    """Required auxiliaries whose feedback input is not active in the status report's Pn field."""
    active = set((pins or "").upper())
    out = []
    for output in outputs:
        if output.kind in kinds and output.required and output.feedback_input:
            if output.feedback_input.upper().lstrip("P") not in active:
                out.append(AUX_KINDS.get(output.kind, output.kind))
    return out


__all__ = ["AUX_KINDS", "AuxOutput", "DEFAULT_COMMANDS", "GASES", "aux_lines", "default_aux_for", "missing_required"]
