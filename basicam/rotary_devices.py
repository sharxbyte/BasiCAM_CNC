"""Rotary devices: 4th axis, rotary laser, lathe headstock, lathe-mill and laser-lathe (no Qt).

A device sits on the machine with its spin axis along X (or Y) through ``centre_mm`` (machine
coordinates); its face (the chuck / collet face) is at ``centre_mm`` and work extends towards +axis.
Geometry here (chuck body, jaws, collet nose, rollers, tailstock, steady rest) is used for the
viewport (red keep-out geometry) and, in Block 19, for the moving keep-out checks.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

DEVICE_KINDS = {
    "rotary_4th": "4th-axis rotary (mill / router)",
    "rotary_laser": "Rotary laser (rollers or chuck)",
    "lathe_headstock": "Lathe headstock",
    "lathe_mill": "Lathe and mill combination",
    "laser_lathe": "Laser lathe",
}
MOUNTS = ("3-jaw chuck", "4-jaw independent chuck", "6-jaw chuck", "Collet", "Faceplate", "Between centres", "Rollers")
SPEED_CONTROL = ("automatic", "manual")


@dataclass
class RotaryDevice:
    name: str = "Rotary"
    kind: str = "rotary_4th"
    axis: str = "A"
    orientation: str = "X"  # spin axis along machine X or Y
    centre_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # face centre, machine coordinates
    headstock_diameter_mm: float = 100.0  # chuck / headstock body diameter
    body_length_mm: float = 60.0  # chuck body depth behind the face
    throw_mm: float = 80.0  # maximum swing (largest diameter that turns clear)
    mount: str = "3-jaw chuck"
    clamp_min_mm: float = 2.0  # smallest / largest diameter the mount grips
    clamp_max_mm: float = 80.0
    clamp_diameter_mm: float = 25.0  # what it is gripping now
    inside_clamp: bool = False  # jaws grip a bore from the inside
    jaw_count: int = 3
    jaw_spacing_deg: float = 0.0  # 0 = evenly spaced
    jaw_width_mm: float = 12.0
    jaw_height_mm: float = 18.0  # radial size of a jaw
    jaw_depth_mm: float = 20.0  # how far the jaw stands proud of the face (along the axis)
    jaw_step_mm: float = 0.0  # stepped jaws: step depth
    jaw_type: str = "hard"  # "hard", "soft", "stepped"
    roller_spacing_mm: float = 60.0  # rotary laser rollers
    roller_diameter_mm: float = 30.0
    tailstock: bool = False
    tailstock_travel_mm: float = 300.0
    tailstock_position_mm: float = 200.0  # centre point distance from the face
    quill_length_mm: float = 60.0
    live_centre: bool = True
    steady_rest: bool = False
    steady_rest_position_mm: float = 100.0
    steady_rest_diameter_mm: float = 25.0
    speed_control: str = "automatic"  # "manual": the program pauses and asks the operator for RPM
    max_rpm: float = 3000.0
    max_rate_deg_min: float = 10800.0
    indexing_deg: float = 0.0  # 0 = continuous positioning; else the index step
    continuous: bool = True  # can turn (lathe / wrapping); False = positioning only

    def jaw_angles(self) -> list[float]:
        count = self.effective_jaw_count()
        if count == 0:
            return []
        step = self.jaw_spacing_deg if self.jaw_spacing_deg > 0 else 360.0 / count
        return [i * step for i in range(count)]

    def effective_jaw_count(self) -> int:
        if self.mount.startswith("3-jaw"):
            return 3
        if self.mount.startswith("4-jaw"):
            return 4
        if self.mount.startswith("6-jaw"):
            return 6
        if self.mount == "Faceplate":
            return max(0, int(self.jaw_count))
        return 0

    def problems(self) -> list[str]:
        out = []
        if self.clamp_min_mm > self.clamp_max_mm:
            out.append("The smallest clamping diameter is larger than the largest.")
        if self.mount not in ("Rollers", "Between centres") and not (self.clamp_min_mm - 1e-9 <= self.clamp_diameter_mm <= self.clamp_max_mm + 1e-9):
            out.append(f"The {self.mount.lower()} grips {self.clamp_min_mm:g}-{self.clamp_max_mm:g} mm, not {self.clamp_diameter_mm:g} mm.")
        if self.throw_mm <= 0:
            out.append("The throw (largest diameter that turns clear) must be greater than zero.")
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RotaryDevice":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        if "centre_mm" in values:
            values["centre_mm"] = tuple(float(v) for v in values["centre_mm"])
        device = cls(**values)
        if device.kind not in DEVICE_KINDS:
            device.kind = "rotary_4th"
        if device.mount not in MOUNTS:
            device.mount = "3-jaw chuck"
        return device


@dataclass
class Body:
    """A keep-out body in the device frame (axis = +u): a cylinder or a box."""

    name: str
    shape: str  # "cylinder" or "box"
    u0: float  # along the axis
    u1: float
    radius: float = 0.0  # cylinder
    centre_vw: tuple[float, float] = (0.0, 0.0)  # box centre across the axis
    size_vw: tuple[float, float] = (0.0, 0.0)  # box size across the axis
    angle_deg: float = 0.0  # box rotation about the axis
    rotates: bool = True  # turns with the spindle


def device_bodies(device: RotaryDevice) -> list[Body]:
    """Chuck body, jaws, collet nose, rollers, tailstock and steady rest (device frame)."""
    bodies = [Body("Chuck body" if device.mount != "Collet" else "Collet body", "cylinder", -device.body_length_mm, 0.0,
                   radius=device.headstock_diameter_mm / 2.0)]
    clamp_r = device.clamp_diameter_mm / 2.0
    if device.mount == "Collet":
        bodies.append(Body("Collet nose", "cylinder", 0.0, min(15.0, device.jaw_depth_mm), radius=max(clamp_r + 6.0, clamp_r * 1.4)))
    for index, angle in enumerate(device.jaw_angles()):
        r0 = clamp_r - device.jaw_height_mm if device.inside_clamp else clamp_r
        r1 = clamp_r if device.inside_clamp else clamp_r + device.jaw_height_mm
        bodies.append(Body(f"Jaw {index + 1}", "box", 0.0, device.jaw_depth_mm, centre_vw=((r0 + r1) / 2.0, 0.0),
                           size_vw=(abs(r1 - r0), device.jaw_width_mm), angle_deg=angle))
    if device.mount == "Rollers":
        half = device.roller_spacing_mm / 2.0
        rr = device.roller_diameter_mm / 2.0
        for side in (-1, 1):
            bodies.append(Body(f"Roller {'front' if side < 0 else 'back'}", "box", 0.0, 300.0,
                               centre_vw=(side * half, -(clamp_r + rr)), size_vw=(device.roller_diameter_mm, device.roller_diameter_mm),
                               rotates=False))
    if device.tailstock:
        tip = device.tailstock_position_mm
        bodies.append(Body("Tailstock centre", "cylinder", tip, tip + 25.0, radius=8.0, rotates=False))
        bodies.append(Body("Tailstock quill", "cylinder", tip + 25.0, tip + 25.0 + device.quill_length_mm, radius=20.0, rotates=False))
    if device.steady_rest:
        u = device.steady_rest_position_mm
        bodies.append(Body("Steady rest", "box", u - 10.0, u + 10.0, centre_vw=(0.0, -(device.steady_rest_diameter_mm / 2.0 + 30.0)),
                           size_vw=(device.steady_rest_diameter_mm + 80.0, 60.0), rotates=False))
    return bodies


def body_meshes(device: RotaryDevice):
    """pyvista meshes of the bodies placed in machine coordinates (for drawing)."""
    import pyvista as pv

    centre = np.asarray(device.centre_mm, dtype=float)
    axis = np.array([1.0, 0.0, 0.0]) if device.orientation == "X" else np.array([0.0, 1.0, 0.0])
    out = []
    for body in device_bodies(device):
        mid = (body.u0 + body.u1) / 2.0
        length = max(body.u1 - body.u0, 1e-3)
        if body.shape == "cylinder":
            mesh = pv.Cylinder(center=centre + axis * mid, direction=axis, radius=body.radius, height=length, resolution=48)
        else:
            sv, sw = body.size_vw
            cv, cw = body.centre_vw
            box = pv.Box(bounds=(-length / 2.0, length / 2.0, cv - sv / 2.0, cv + sv / 2.0, cw - sw / 2.0, cw + sw / 2.0))
            box = box.rotate_x(body.angle_deg, inplace=False)
            if device.orientation == "Y":
                box = box.rotate_z(90.0, inplace=False)
            mesh = box.translate(centre + axis * mid, inplace=False)
        out.append((body.name, mesh))
    return out


__all__ = ["Body", "DEVICE_KINDS", "MOUNTS", "RotaryDevice", "SPEED_CONTROL", "body_meshes", "device_bodies"]
