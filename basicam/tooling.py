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
    corner_radius_mm: float = 0.0  # bull nose end mills (0 = sharp corner)
    holder_diameter_mm: float = 0.0  # tool holder / collet nut: part of the collision envelope
    holder_length_mm: float = 0.0
    # lathe tools (v2 Block 21); lathe_kind "" = a milling tool
    lathe_kind: str = ""  # "turning", "boring", "grooving", "parting", "threading", "drill"
    insert_shape: str = "C"  # C D V W T R S
    nose_radius_mm: float = 0.4
    holder_angle_deg: float = 95.0
    orientation: str = "OD"  # "OD", "ID" or "face"
    groove_width_mm: float = 0.0
    thread_pitch_min_mm: float = 0.0
    thread_pitch_max_mm: float = 0.0
    min_bore_mm: float = 0.0
    station: int = 0  # turret station (0 = any)
    # v3 Block 4: every length is measured from the tip (Z = 0) upward.
    # cutting_length_mm = end of the cutting edge; flute_length_mm = end of the flute (>= cutting);
    # reach_mm = end of a reduced neck (0 = no neck); taper_angle_deg is per side.
    neck_diameter_mm: float = 0.0
    reach_mm: float = 0.0
    taper_angle_deg: float = 0.0
    helix_angle_deg: float = 30.0
    profile_mode: str = "generated"  # "generated" from the dimensions, or "custom" (edited points)

    def __post_init__(self) -> None:
        if self.tool_type == "Custom":
            self.profile_mode = "custom"
        if self.profile_mode != "custom" or not self.profile_points:
            self.profile_points = default_profile(self)

    def max_radius_mm(self) -> float:
        """Widest radius of the tool body (cutting edge, neck or shank), for the collision envelope."""
        radii = [float(r) for r, _z in self.profile_points] or [self.diameter_mm / 2.0]
        return max(max(radii), float(self.shank_diameter_mm) / 2.0, float(self.diameter_mm) / 2.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ToolDefinition:
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in dict(data).items() if key in known}
        values.setdefault("name", "Tool")
        values.setdefault("tool_type", "End mill")
        values["profile_points"] = [tuple(point) for point in values.get("profile_points", [])]
        if "profile_mode" not in values and values["profile_points"]:
            # Tools saved before v3: a profile the owner edited stays as a custom profile; an
            # untouched default profile is regenerated from the dimensions.
            probe = cls(**{**values, "profile_points": [], "profile_mode": "generated"})
            legacy = legacy_default_profile(probe)
            stored = values["profile_points"]
            same = len(stored) == len(legacy) and all(
                abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6 for a, b in zip(stored, legacy)
            )
            values["profile_mode"] = "generated" if same else "custom"
        return cls(**values)

    @property
    def stick_out_mm(self) -> float:
        """Length out of the collet: overall length minus the 12 mm held in the collet."""
        return max(0.0, float(self.overall_length_mm) - COLLET_GRIP_MM)


SHANK_CHAMFER_DEG = 30.0  # step between neck / flutes and the shank


def _upper_body(tool: ToolDefinition, points: list[tuple[float, float]], radius: float, z: float) -> None:
    """Flute end -> neck -> 30 degree chamfer -> shank up to the overall length."""
    total = max(float(tool.overall_length_mm), z + 0.5)
    shank = max(float(tool.shank_diameter_mm) / 2.0, 0.05)
    neck = float(tool.neck_diameter_mm) / 2.0
    reach = float(tool.reach_mm)
    if neck > 0.0 and reach > z + 1e-6:
        neck = min(neck, radius) if radius > 0 else neck
        if abs(neck - radius) > 1e-9:
            points.append((neck, z))
        points.append((neck, min(reach, total)))
        radius, z = neck, min(reach, total)
    if abs(shank - radius) > 1e-9 and z < total:
        rise = abs(shank - radius) * math.tan(math.radians(SHANK_CHAMFER_DEG))
        z = min(z + rise, total)
        points.append((shank, z))
    if z < total:
        points.append((shank, total))
    elif points[-1] != (shank, total):
        points.append((shank, total))


def _dedupe(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for r, z in points:
        point = (round(float(r), 6), round(float(z), 6))
        if not out or out[-1] != point:
            out.append(point)
    return out


def default_profile(tool: ToolDefinition) -> list[tuple[float, float]]:
    """Revolved half-profile (radius, Z from the tip) generated from the tool's dimensions.

    The cutting edge runs from the tip up to the cutting length; the flute continues to the flute
    length; then the optional neck to the reach, a short 30 degree chamfer and the shank."""
    radius = max(tool.diameter_mm / 2.0, 0.05)
    total = max(float(tool.overall_length_mm), 1.0)
    cut = min(max(float(tool.cutting_length_mm), 0.1), total)
    flute = min(max(float(tool.flute_length_mm), cut), total)
    tip_radius = min(max(tool.tip_diameter_mm / 2.0, 0.0), radius)
    taper = math.tan(math.radians(max(0.0, min(float(tool.taper_angle_deg), 30.0))))
    kind = tool.tool_type
    pts: list[tuple[float, float]] = []

    if kind == "Ball end mill":
        pts.append((0.0, 0.0))
        for angle in np.linspace(0.0, math.pi / 2, 9)[1:]:
            pts.append((radius * math.sin(float(angle)), radius * (1 - math.cos(float(angle)))))
        top = radius + max(cut - radius, 0.0) * taper
        pts.append((top, cut))
        if flute > cut:
            pts.append((top, flute))
        _upper_body(tool, pts, top, flute)
        return _dedupe(pts)

    if kind in {"Drill", "V-bit", "Countersink", "Chamfer mill"}:
        angle = math.radians(min(max(tool.included_angle_deg, 10.0), 170.0) / 2)
        cone = (radius - tip_radius) / max(math.tan(angle), 0.05)
        start = tip_radius if kind in {"Countersink", "Chamfer mill"} else 0.0
        if start > 0:
            pts.append((0.0, 0.0))
        pts.append((start, 0.0))
        cone_top = min(cone, flute)
        pts.append((start + (radius - start) * (cone_top / cone if cone > 0 else 1.0), cone_top))
        if flute > cone_top:
            pts.append((radius, flute))
        _upper_body(tool, pts, pts[-1][0], flute)
        return _dedupe(pts)

    if kind in {"T-slot cutter", "Groove cutter", "Fly cutter"}:
        # the cutting length is the head thickness; a neck joins the head to the shank
        head = cut
        neck_r = float(tool.neck_diameter_mm) / 2.0 or min(max(tool.shank_diameter_mm / 2.0, 0.05), radius * 0.35)
        if kind == "Fly cutter":
            neck_r = max(tool.shank_diameter_mm / 2.0, 0.05)
        pts += [(0.0, 0.0), (radius, 0.0), (radius, head), (min(neck_r, radius), head)]
        z = max(flute, head)
        pts.append((min(neck_r, radius), z))
        _upper_body(tool, pts, min(neck_r, radius), z)
        return _dedupe(pts)

    if kind == "Dovetail cutter":
        side = math.radians(min(max(tool.included_angle_deg, 30.0), 89.0))
        top = max(radius - cut / math.tan(side), 0.05)
        pts += [(0.0, 0.0), (radius, 0.0), (top, cut)]
        neck_r = min(float(tool.neck_diameter_mm) / 2.0 or top, radius)  # neck joins the head to the shank
        if abs(neck_r - top) > 1e-9:
            pts.append((neck_r, cut))
        if flute > cut:
            pts.append((neck_r, flute))
        _upper_body(tool, pts, neck_r, flute)
        return _dedupe(pts)

    if kind in {"Tap", "Reamer"}:
        lead = min(radius * 0.35, cut * 0.25)
        pts += [(0.0, 0.0), (max(radius - lead, tip_radius), 0.0), (radius, lead), (radius, cut)]
        if flute > cut:
            pts.append((radius, flute))
        _upper_body(tool, pts, radius, flute)
        return _dedupe(pts)

    # End mill, bull nose (corner radius), router bit, spiral flute cutter, tapered end mill
    corner = min(max(float(tool.corner_radius_mm), 0.0), radius)
    pts.append((0.0, 0.0))
    if corner > 0:
        pts.append((radius - corner, 0.0))
        for angle in np.linspace(0.0, math.pi / 2, 6)[1:]:
            pts.append((radius - corner + corner * math.sin(float(angle)), corner * (1 - math.cos(float(angle)))))
    else:
        pts.append((radius, 0.0))
    top = radius + cut * taper
    pts.append((top, cut))
    if flute > cut:
        pts.append((top, flute))
    _upper_body(tool, pts, top, flute)
    return _dedupe(pts)


def validate_tool(tool: ToolDefinition) -> list[str]:
    """Contradictory dimensions, in plain words (empty when the tool is consistent)."""
    problems: list[str] = []
    if tool.diameter_mm <= 0:
        problems.append("The cutting diameter must be more than 0.")
    if tool.cutting_length_mm > tool.flute_length_mm + 1e-9:
        problems.append("The cutting length cannot be longer than the flute length.")
    if tool.flute_length_mm > tool.overall_length_mm + 1e-9:
        problems.append("The flute length cannot be longer than the overall length.")
    if tool.reach_mm > tool.overall_length_mm + 1e-9:
        problems.append("The reach cannot be longer than the overall length.")
    if tool.neck_diameter_mm > 0 and tool.tool_type not in {"T-slot cutter", "Groove cutter", "Dovetail cutter"}:
        if tool.neck_diameter_mm >= tool.diameter_mm:
            problems.append("A neck must be narrower than the cutting diameter.")
        if 0 < tool.reach_mm <= tool.flute_length_mm:
            problems.append("The reach (end of the neck) must be beyond the flute length.")
    if tool.tip_diameter_mm > tool.diameter_mm + 1e-9:
        problems.append("The tip diameter cannot be larger than the cutting diameter.")
    if tool.corner_radius_mm * 2 > tool.diameter_mm + 1e-9:
        problems.append("The corner radius cannot be more than half the cutting diameter.")
    if tool.shank_diameter_mm <= 0:
        problems.append("The shank diameter must be more than 0.")
    return problems


def legacy_default_profile(tool: ToolDefinition) -> list[tuple[float, float]]:
    """Profile generator before v3 (used only to recognise untouched old profiles)."""
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


def tool_envelope_mesh(tool: ToolDefinition, resolution: int = 48) -> pv.PolyData:
    """Revolved envelope of the tool (no flutes): used by the motion guard and the simulation."""
    return _revolve_profile(tool, resolution)


def create_tool_mesh(tool: ToolDefinition, resolution: int = 48) -> pv.PolyData:
    """Alias of tool_envelope_mesh for existing callers."""
    return _revolve_profile(tool, resolution)


def _revolve_profile(tool: ToolDefinition, resolution: int = 48) -> pv.PolyData:
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
        ("10 mm groove cutter", "Groove cutter", 10, 0, 90, 4, 55, 14, 6, 3),
        ("50 mm fly cutter", "Fly cutter", 50, 0, 90, 1, 80, 8, 12, 8),
        ("6 mm spiral flute", "Spiral flute cutter", 6, 0, 90, 2, 65, 24, 6, 22),
        ("6 mm router bit", "Router bit", 6, 0, 90, 2, 60, 20, 6, 18),
        ("12 mm T-slot", "T-slot cutter", 12, 0, 90, 3, 60, 16, 8, 5),
        ("12 mm dovetail", "Dovetail cutter", 12, 0, 60, 3, 60, 18, 8, 6),
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


# Old "Tools & Speeds" list kinds -> inventory tool types (v3 Block 3 migration).
LIBRARY_KIND_TYPES = {
    "End mill": "End mill",
    "Ball mill": "Ball end mill",
    "Drill": "Drill",
    "Chamfer mill": "Chamfer mill",
    "Face mill": "End mill",
}


def migrate_library_entries(tools: list[ToolDefinition], entries) -> list[ToolDefinition]:
    """Add each old library entry that has no inventory twin (same type and diameter)."""
    result = list(tools)
    for entry in entries:
        tool_type = LIBRARY_KIND_TYPES.get(str(entry.kind), "End mill")
        diameter = float(entry.diameter_mm)
        twin = any(t.tool_type == tool_type and abs(t.diameter_mm - diameter) < 1e-3 for t in result)
        if twin or any(t.name == entry.name for t in result):
            continue
        flute = max(2.0 * diameter, 6.0)
        result.append(
            ToolDefinition(
                name=str(entry.name),
                tool_type=tool_type,
                diameter_mm=diameter,
                flute_count=int(entry.flutes),
                shank_diameter_mm=diameter,
                flute_length_mm=flute,
                cutting_length_mm=flute,
                overall_length_mm=flute + 40.0,
            )
        )
    return result


# ---------------------------------------------------------------------- display mesh with flutes
ZONE_CUTTING, ZONE_FLUTE, ZONE_BODY = 0, 1, 2


def _densify(profile: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for (r0, z0), (r1, z1) in zip(profile, profile[1:]):
        length = math.hypot(r1 - r0, z1 - z0)
        count = max(1, int(math.ceil(length / max(step, 1e-3))))
        for i in range(count):
            f = i / count
            out.append((r0 + (r1 - r0) * f, z0 + (z1 - z0) * f))
    out.append(profile[-1])
    return out


def tool_preview_mesh(tool: ToolDefinition, angular_steps: int = 144) -> pv.PolyData:
    """Display mesh: the revolved body with one helical groove per flute over the flute length.

    Point data "zone": 0 cutting edge (up to the cutting length), 1 flute, 2 neck and shank."""
    profile = [(float(r), float(z)) for r, z in tool.profile_points]
    if len(profile) < 2:
        raise ValueError("A tool profile needs at least two radius/length points.")
    radius = max(tool.diameter_mm / 2.0, 0.05)
    cut = float(tool.cutting_length_mm)
    flute = max(float(tool.flute_length_mm), cut)
    flutes = max(int(tool.flute_count), 0)
    helix = math.tan(math.radians(max(0.0, min(float(tool.helix_angle_deg), 60.0))))
    step = max(min(flute, max(tool.overall_length_mm, 1.0)) / 60.0, 0.05)
    rows = _densify(profile, step)
    angles = np.linspace(0.0, 2 * math.pi, angular_steps, endpoint=False)
    runout = max(flute * 0.08, 0.5)
    points = []
    zones = []
    for r, z in rows:
        if r <= 1e-9:
            ring = np.zeros((angular_steps, 3))
            ring[:, 2] = z
        else:
            scale = np.ones(angular_steps)
            if flutes > 0 and z <= flute and tool.profile_mode != "custom":
                fade = 1.0 if z <= flute - runout else max(0.0, (flute - z) / runout)
                phase = flutes * (angles - z * helix / max(r, 1e-3))
                groove = ((1.0 + np.cos(phase)) / 2.0) ** 6  # narrow valleys between the cutting edges
                scale = 1.0 - 0.22 * fade * groove
            ring = np.column_stack((r * scale * np.cos(angles), r * scale * np.sin(angles), np.full(angular_steps, z)))
        points.append(ring)
        zone = ZONE_CUTTING if z <= cut + 1e-9 else (ZONE_FLUTE if z <= flute + 1e-9 else ZONE_BODY)
        zones.append(np.full(angular_steps, zone))
    pts = np.vstack(points)
    n = angular_steps
    faces = []
    for row in range(len(rows) - 1):
        a, b = row * n, (row + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.extend((3, a + i, a + j, b + j, 3, a + i, b + j, b + i))
    mesh = pv.PolyData(pts, np.asarray(faces, dtype=np.int64))
    mesh.point_data["zone"] = np.concatenate(zones).astype(np.int8)
    mesh = mesh.clean(tolerance=1e-9)
    return mesh
