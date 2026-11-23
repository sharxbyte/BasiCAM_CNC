"""Automatic tool changer, tool table and tool-change program blocks (no Qt).

A machine's tool changer has a type (none / manual, carousel, rack, lathe turret or gang), a
pocket (station) count - how many tools can be pre-loaded -, a change position, whether the
controller runs M6 itself, and how tool lengths are measured. The tool table says which
inventory tool (and holder) sits in each pocket; tool number = pocket by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

CHANGER_TYPES = ("none", "carousel", "rack", "turret", "gang")
CHANGER_LABELS = {
    "none": "None (manual change)", "carousel": "Carousel ATC", "rack": "Rack ATC",
    "turret": "Lathe turret", "gang": "Gang tooling",
}
ATC_CHANGE_S = 8.0
MANUAL_CHANGE_S = 60.0


@dataclass
class ToolChanger:
    kind: str = "none"
    pockets: int = 0
    change_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # machine coordinates
    controller_m6: bool = True  # the controller runs M6; else BasiCAM sends the change macro
    length_measure: str = "setter"  # "setter" (tool setter probe) or "table" (offsets from the tool table)
    setter_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    length_offsets: bool = True  # G43 H<n> after a change, where supported
    change_time_s: float = 0.0  # 0 = default (8 s ATC, 60 s manual)

    @property
    def automatic(self) -> bool:
        return self.kind != "none" and self.pockets > 0

    def time_s(self) -> float:
        if self.change_time_s > 0:
            return float(self.change_time_s)
        return ATC_CHANGE_S if self.automatic else MANUAL_CHANGE_S

    @classmethod
    def from_dict(cls, data: dict | None) -> "ToolChanger":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        for key in ("change_position_mm", "setter_position_mm"):
            if key in values:
                values[key] = tuple(float(v) for v in values[key])
        changer = cls(**values)
        if changer.kind not in CHANGER_TYPES:
            changer.kind = "none"
        return changer


@dataclass
class PocketEntry:
    tool: str = ""  # inventory tool name
    holder: str = ""  # holder name (its envelope comes from the tool's holder fields)
    length_offset_mm: float = 0.0


@dataclass
class ToolTable:
    pockets: dict[int, PocketEntry] = field(default_factory=dict)

    def number_of(self, tool_name: str) -> int | None:
        for number, entry in sorted(self.pockets.items()):
            if entry.tool == tool_name:
                return number
        return None

    def load(self, pocket: int, tool_name: str, holder: str = "") -> None:
        for number, entry in list(self.pockets.items()):
            if entry.tool == tool_name and number != pocket:
                del self.pockets[number]  # one pocket per tool
        self.pockets[int(pocket)] = PocketEntry(tool_name, holder)

    def unload(self, pocket: int) -> None:
        self.pockets.pop(int(pocket), None)

    def to_dict(self) -> dict:
        return {str(k): asdict(v) for k, v in self.pockets.items()}

    @classmethod
    def from_dict(cls, data: dict | None) -> "ToolTable":
        table = cls()
        for key, value in dict(data or {}).items():
            try:
                table.pockets[int(key)] = PocketEntry(**{k: v for k, v in dict(value).items() if k in ("tool", "holder", "length_offset_mm")})
            except (TypeError, ValueError):
                continue
        return table


def tool_change_block(changer: ToolChanger, table: ToolTable, tool_name: str, safe_z_mm: float, inches: bool = False) -> list[str]:
    """G-code for one tool change. With an ATC: T<n> M6 (+ G43 H<n>). Without: a guarded
    pause - spindle off, up to safe Z, M0 with a prompt - then an optional setter probe."""
    scale = 1.0 / 25.4 if inches else 1.0
    lines = ["M5", f"G53 G0 Z{0.0:.4f}" if changer.automatic else f"G0 Z{safe_z_mm * scale:.4f}"]
    number = table.number_of(tool_name)
    if changer.automatic and number is not None:
        x, y, z = (v * scale for v in changer.change_position_mm)
        if not changer.controller_m6:
            lines.append(f"G53 G0 X{x:.4f} Y{y:.4f}")
        lines.append(f"T{number} M6")
        if changer.length_offsets and changer.length_measure == "table":
            lines.append(f"G43 H{number}")
    else:
        lines.append(f"M0 (Change tool: {tool_name})")
    if changer.length_measure == "setter" and (not changer.automatic or not changer.controller_m6):
        sx, sy, sz = (v * scale for v in changer.setter_position_mm)
        lines.append(f"(Measure the tool on the setter at X{sx:.3f} Y{sy:.3f})")
    return lines


def tools_not_loaded(changer: ToolChanger, table: ToolTable, tool_names) -> list[str]:
    if not changer.automatic:
        return []
    return [name for name in dict.fromkeys(tool_names) if name and table.number_of(name) is None]


__all__ = [
    "ATC_CHANGE_S", "CHANGER_LABELS", "CHANGER_TYPES", "MANUAL_CHANGE_S", "PocketEntry", "ToolChanger",
    "ToolTable", "tool_change_block", "tools_not_loaded",
]
