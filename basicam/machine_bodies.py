"""Lathe machine bodies and moving keep-outs (shapely in the XZ half-plane, no Qt).

Lathe coordinates here: r = distance from the spindle axis (mm), z = along the axis from the
chuck / collet face (positive towards the tailstock). Revolved bodies are profiles in the (z, r)
half-plane. Jaws turn with the spindle, so their keep-out is the whole disc they sweep. Bodies
linked to an axis move with it: the cross slide / turret with X and Z, the tailstock with its own
position. Every move is checked: the tool, its holder and the turret against the chuck body,
jaws, collet nose, tailstock and steady rest (KEEPOUT, never overridable), and the holder and
turret against the stock (COLLISION; only the cutting edge may touch the stock).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from basicam.rotary_devices import RotaryDevice


@dataclass
class LatheTool:
    """Tool outline in the half-plane, relative to its tip: the insert and holder reach outwards
    (+r) from the tip; ``width`` is the size along z."""

    name: str = "Turning tool"
    insert_width_mm: float = 6.0
    insert_length_mm: float = 10.0  # radial size of the insert / cutting part
    holder_width_mm: float = 20.0
    holder_length_mm: float = 40.0
    turret_width_mm: float = 80.0
    turret_height_mm: float = 60.0
    boring: bool = False  # boring bars reach inwards along z instead (into a bore)

    @classmethod
    def from_tool(cls, tool) -> "LatheTool":
        if tool is None:
            return cls()
        width = float(getattr(tool, "groove_width_mm", 0.0) or 0.0) or float(getattr(tool, "diameter_mm", 6.0) or 6.0)
        return cls(
            str(getattr(tool, "name", "tool")), insert_width_mm=width,
            insert_length_mm=max(float(getattr(tool, "flute_length_mm", 10.0) or 10.0), 1.0),
            holder_width_mm=max(float(getattr(tool, "holder_diameter_mm", 0.0) or 0.0), width + 10.0),
            holder_length_mm=max(float(getattr(tool, "holder_length_mm", 0.0) or 0.0), 40.0),
            boring=str(getattr(tool, "lathe_kind", "")) == "boring",
        )

    def parts(self, z: float, r: float) -> tuple[Polygon, Polygon]:
        """(cutting part, holder + turret) placed with the tip at (z, r)."""
        half = self.insert_width_mm / 2.0
        if self.boring:
            cutting = box(z - 1.0, r, z + self.insert_length_mm, r + self.insert_width_mm)
            holder = box(z + self.insert_length_mm, r, z + self.insert_length_mm + self.holder_length_mm, r + self.holder_width_mm)
            return cutting, holder
        # OD tools work towards the chuck: the tip is the insert's chuck-side corner and the
        # holder and turret trail on the free-end side, above the insert
        cutting = box(z, r, z + self.insert_width_mm, r + self.insert_length_mm)
        holder = box(z, r + self.insert_length_mm, z + self.holder_width_mm, r + self.insert_length_mm + self.holder_length_mm)
        top = r + self.insert_length_mm + self.holder_length_mm
        turret = box(z, top, z + self.turret_width_mm, top + self.turret_height_mm)
        _ = half
        return cutting, unary_union([holder, turret])


@dataclass
class BodyProfile:
    name: str
    polygon: Polygon
    keepout: bool = True
    moves_with: str = ""  # "" fixed, "tailstock"


def lathe_bodies(device: RotaryDevice) -> list[BodyProfile]:
    """Keep-out profiles of the headstock side and the tailstock / steady rest."""
    bodies = [BodyProfile("Chuck body", box(-device.body_length_mm, 0.0, 0.0, device.headstock_diameter_mm / 2.0))]
    clamp_r = device.clamp_diameter_mm / 2.0
    if device.mount == "Collet":
        bodies.append(BodyProfile("Collet nose", box(0.0, clamp_r, min(15.0, device.jaw_depth_mm), max(clamp_r + 6.0, clamp_r * 1.4))))
    elif device.effective_jaw_count() > 0:
        r0 = clamp_r - device.jaw_height_mm if device.inside_clamp else clamp_r
        r1 = clamp_r if device.inside_clamp else clamp_r + device.jaw_height_mm
        # jaws turn with the spindle: they sweep the whole annulus
        bodies.append(BodyProfile("Jaws (swept)", box(0.0, max(r0, 0.0), device.jaw_depth_mm, r1)))
    if device.tailstock:
        tip = device.tailstock_position_mm
        centre = Polygon([(tip, 0.0), (tip + 12.0, 0.0), (tip + 12.0, 8.0), (tip + 6.0, 8.0)])
        quill = box(tip + 12.0, 0.0, tip + 12.0 + device.quill_length_mm, 20.0)
        bodies.append(BodyProfile("Tailstock", unary_union([centre, quill]), moves_with="tailstock"))
    if device.steady_rest:
        u = device.steady_rest_position_mm
        inner = device.steady_rest_diameter_mm / 2.0
        bodies.append(BodyProfile("Steady rest", box(u - 10.0, inner, u + 10.0, inner + 80.0)))
    return bodies


def stock_profile(diameter: float, stick_out: float, bore: float = 0.0) -> Polygon:
    """Bar / tube in the half-plane, from the jaw face out to its stick-out."""
    return box(0.0, bore / 2.0, stick_out, diameter / 2.0)


@dataclass
class LatheViolation:
    kind: str  # "KEEPOUT" or "COLLISION"
    text: str
    move: int


def check_lathe_moves(device: RotaryDevice, moves: list[tuple[tuple[float, float], tuple[float, float], str, bool]],
                      tools: dict | None = None, stock: Polygon | None = None, step_mm: float = 0.5) -> list[LatheViolation]:
    """Check moves [((z0, r0), (z1, r1), tool name, cutting)] against the lathe bodies; the tool
    and holder outline changes with the tool of each move (turret changes)."""
    tools = tools or {}
    bodies = lathe_bodies(device)
    found: dict[tuple, LatheViolation] = {}
    remaining = stock  # stock removal: what the cutting edge has swept is gone
    for index, (a, b, tool_name, cutting) in enumerate(moves, start=1):
        tool = tools.get(tool_name) or LatheTool(tool_name or "tool")
        length = math.dist(a, b)
        steps = max(1, int(math.ceil(length / max(step_mm, 0.05))))
        for k in range(steps + 1):
            z = a[0] + (b[0] - a[0]) * k / steps
            r = a[1] + (b[1] - a[1]) * k / steps
            cutting_part, holder = tool.parts(z, r)
            whole = unary_union([cutting_part, holder])
            for body in bodies:
                if whole.intersection(body.polygon).area > 1e-9:
                    key = ("KEEPOUT", body.name)
                    found.setdefault(key, LatheViolation("KEEPOUT", f"move {index}: {tool.name} would hit the {body.name.lower()}", index))
            if remaining is not None and not remaining.is_empty:
                if cutting:
                    remaining = remaining.difference(cutting_part)
                if holder.intersection(remaining).area > 1e-6:
                    found.setdefault(("COLLISION", "holder"), LatheViolation("COLLISION", f"move {index}: the holder of {tool.name} would hit the stock", index))
                if not cutting and cutting_part.intersection(remaining).area > 1e-6:
                    found.setdefault(("COLLISION", "rapid"), LatheViolation("COLLISION", f"move {index}: rapid move of {tool.name} through the stock", index))
    return list(found.values())


def segments_to_moves(segments) -> list:
    """Lathe segments (x = radius, z axial from the chuck face, metadata tool) -> moves."""
    out = []
    for seg in segments:
        cutting = seg.kind not in ("rapid", "retract")
        out.append(((seg.start[2], seg.start[0]), (seg.end[2], seg.end[0]), (seg.metadata or {}).get("tool", ""), cutting))
    return out


__all__ = ["BodyProfile", "LatheTool", "LatheViolation", "check_lathe_moves", "lathe_bodies", "segments_to_moves", "stock_profile"]
