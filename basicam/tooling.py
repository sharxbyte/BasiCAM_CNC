from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math

import numpy as np
import pyvista as pv

COLLET_GRIP_MM = 12.0  # length of tool held in the collet


TOOL_TYPES = (
    "End mill",
    "Ball end mill",
    "Drill",
    "Tap",
    "Reamer",
    "Countersink",
    "Groove cutter",
    "Fly cutter",
    "Spiral flute cutter",
    "Router bit",
    "T-slot cutter",
    "Dovetail cutter",
    "V-bit",
    "Chamfer mill",
    "Custom",
)


@dataclass
class ToolDefinition:
    name: str
    tool_type: str
    diameter_mm: float = 6.0
    tip_diameter_mm: float = 0.0
    included_angle_deg: float = 118.0
    flute_count: int = 2
    overall_length_mm: float = 60.0
    flute_length_mm: float = 20.0
    shank_diameter_mm: float = 6.0
    cutting_length_mm: float = 18.0
    profile_points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.profile_points:
            self.profile_points = default_profile(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ToolDefinition:
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in dict(data).items() if key in known}
        values.setdefault("name", "Tool")
        values.setdefault("tool_type", "End mill")
        values["profile_points"] = [tuple(point) for point in values.get("profile_points", [])]
        return cls(**values)

    @property
    def stick_out_mm(self) -> float:
        """Length out of the collet: overall length minus the 12 mm held in the collet."""
        return max(0.0, float(self.overall_length_mm) - COLLET_GRIP_MM)


def default_profile(tool: ToolDefinition) -> list[tuple[float, float]]:
    radius = max(tool.diameter_mm / 2, 0.05)
    shank_radius = max(tool.shank_diameter_mm / 2, 0.05)
    total = max(tool.overall_length_mm, 1.0)
    flute = min(max(tool.flute_length_mm, 0.5), total * 0.9)
    tip_radius = min(max(tool.tip_diameter_mm / 2, 0.0), radius)

    if tool.tool_type == "Ball end mill":
        profile = [(0.0, 0.0)]
        for angle in np.linspace(0.0, math.pi / 2, 9)[1:]:
            profile.append((radius * math.sin(float(angle)), radius * (1 - math.cos(float(angle)))))
        profile.extend([(radius, flute), (shank_radius, total)])
        return profile

    if tool.tool_type in {"Drill", "V-bit", "Countersink", "Chamfer mill"}:
        angle = math.radians(min(max(tool.included_angle_deg, 10.0), 170.0) / 2)
        cone_height = min(max((radius - tip_radius) / max(math.tan(angle), 0.05), 0.5), flute)
        start_radius = tip_radius if tool.tool_type == "Countersink" else 0.0
        return [(start_radius, 0.0), (radius, cone_height), (radius, flute), (shank_radius, total)]

    if tool.tool_type in {"T-slot cutter", "Groove cutter"}:
        neck_radius = min(shank_radius, radius * 0.35)
        head_height = min(max(tool.cutting_length_mm * 0.35, 1.0), flute * 0.6)
        return [
            (0.0, 0.0),
            (radius, 0.0),
            (radius, head_height),
            (neck_radius, head_height),
            (neck_radius, flute),
            (shank_radius, total),
        ]

    if tool.tool_type == "Dovetail cutter":
        neck_radius = min(shank_radius, radius * 0.35)
        head_height = min(max(tool.cutting_length_mm * 0.4, 1.0), flute * 0.65)
        return [
            (0.0, 0.0),
            (radius, 0.0),
            (radius * 0.85, head_height),
            (neck_radius, head_height + 0.4),
            (neck_radius, flute),
            (shank_radius, total),
        ]

    return [(0.0, 0.0), (radius, 0.0), (radius, flute), (shank_radius, total)]


def create_tool_mesh(tool: ToolDefinition, resolution: int = 48) -> pv.PolyData:
    if len(tool.profile_points) < 2:
        raise ValueError("A tool profile needs at least two radius/length points.")
    if any(radius < 0 or height < 0 for radius, height in tool.profile_points):
        raise ValueError("Tool profile radius and axial length must be non-negative.")

    angular_steps = max(12, resolution)
    angles = np.linspace(0.0, 2 * math.pi, angular_steps, endpoint=False)
    points: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for radius, height in tool.profile_points:
        if radius == 0:
            rings.append([len(points)])
            points.append((0.0, 0.0, height))
        else:
            ring = []
            for angle in angles:
                ring.append(len(points))
                points.append(
                    (radius * float(np.cos(angle)), radius * float(np.sin(angle)), height)
                )
            rings.append(ring)

    faces: list[int] = []
    for first_ring, second_ring in zip(rings, rings[1:]):
        if len(first_ring) == 1:
            apex = first_ring[0]
            for index in range(angular_steps):
                next_index = (index + 1) % angular_steps
                faces.extend((3, apex, second_ring[next_index], second_ring[index]))
        elif len(second_ring) == 1:
            apex = second_ring[0]
            for index in range(angular_steps):
                next_index = (index + 1) % angular_steps
                faces.extend((3, first_ring[index], first_ring[next_index], apex))
        else:
            for index in range(angular_steps):
                next_index = (index + 1) % angular_steps
                faces.extend(
                    (
                        3,
                        first_ring[index],
                        first_ring[next_index],
                        second_ring[next_index],
                    )
                )
                faces.extend(
                    (
                        3,
                        first_ring[index],
                        second_ring[next_index],
                        second_ring[index],
                    )
                )

    return pv.PolyData(np.asarray(points), np.asarray(faces, dtype=np.int64))


def default_tool_inventory() -> list[ToolDefinition]:
    specs = [
        ("6 mm end mill", "End mill", 6, 0, 90, 4, 60, 20, 6, 18),
        ("6 mm ball end", "Ball end mill", 6, 0, 90, 2, 60, 18, 6, 18),
        ("6 mm drill", "Drill", 6, 0, 118, 2, 75, 35, 6, 32),
        ("M6 spiral tap", "Tap", 6, 4.8, 118, 3, 70, 24, 6, 20),
        ("6 mm reamer", "Reamer", 6, 0, 90, 6, 70, 28, 6, 24),
        ("12 mm countersink", "Countersink", 12, 3, 90, 3, 55, 12, 10, 10),
        ("10 mm groove cutter", "Groove cutter", 10, 0, 90, 4, 55, 14, 6, 8),
        ("50 mm fly cutter", "Fly cutter", 50, 0, 90, 1, 80, 8, 12, 8),
        ("6 mm spiral flute", "Spiral flute cutter", 6, 0, 90, 2, 65, 24, 6, 22),
        ("6 mm router bit", "Router bit", 6, 0, 90, 2, 60, 20, 6, 18),
        ("12 mm T-slot", "T-slot cutter", 12, 0, 90, 3, 60, 16, 8, 10),
        ("12 mm dovetail", "Dovetail cutter", 12, 0, 60, 3, 60, 18, 8, 12),
        ("60 degree V-bit", "V-bit", 12, 0, 60, 2, 55, 16, 8, 12),
        ("90 degree chamfer", "Chamfer mill", 10, 0, 90, 3, 55, 14, 8, 10),
    ]
    return [
        ToolDefinition(
            name,
            tool_type,
            diameter,
            tip_diameter,
            angle,
            flutes,
            overall,
            flute_length,
            shank,
            cutting_length,
        )
        for (
            name,
            tool_type,
            diameter,
            tip_diameter,
            angle,
            flutes,
            overall,
            flute_length,
            shank,
            cutting_length,
        ) in specs
    ]