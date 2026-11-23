"""Forward and inverse kinematics for the supported machine layouts (numpy, no Qt).

Joints are a dict of axis name -> value (mm for linear axes, degrees for rotary axes).
forward() returns the tool tip position and the tool axis (unit vector, from the tip towards
the spindle) in WORKPIECE coordinates; inverse() returns the joints that put the tip there with
that tool axis. Rotary centres are the points the rotary axes turn about, in machine
coordinates. Lathes: the spindle axis is Z, X is the cross slide; in diameter mode X reads the
diameter (twice the radius).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z_AXIS = np.array([0.0, 0.0, 1.0])


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass
class KinematicSetup:
    layout: str = "gantry3"
    rotary_axis: str = "A"  # gantry3_rotary: "A" (about X) or "B" (about Y)
    tilt_axis: str = "A"  # table_table5: "A" or "B" trunnion
    centre_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rotary / table centre
    pivot_length_mm: float = 0.0  # head_table5: pivot to tool tip
    diameter_mode: bool = True  # lathes
    wire_height_mm: float = 50.0  # uv_taper: distance between the guides

    @classmethod
    def of(cls, machine) -> "KinematicSetup":
        if machine is None:
            return cls()
        return cls(
            layout=str(getattr(machine, "layout", "gantry3") or "gantry3"),
            rotary_axis=str(getattr(machine, "rotary_axis", "A") or "A"),
            tilt_axis=str(getattr(machine, "tilt_axis", "A") or "A"),
            centre_mm=tuple(float(v) for v in getattr(machine, "rotary_centre_mm", (0.0, 0.0, 0.0))),
            pivot_length_mm=float(getattr(machine, "pivot_length_mm", 0.0) or 0.0),
            diameter_mode=bool(getattr(machine, "diameter_mode", True)),
            wire_height_mm=float(getattr(machine, "wire_height_mm", 50.0) or 50.0),
        )


def _rot(axis: str, deg: float) -> np.ndarray:
    return {"A": _rx, "B": _ry, "C": _rz}[axis](deg)


def forward(setup: KinematicSetup, joints: dict) -> tuple[np.ndarray, np.ndarray]:
    """Joints -> (tool tip, tool axis) in workpiece coordinates."""
    j = {k: float(v) for k, v in joints.items()}
    c = np.asarray(setup.centre_mm, dtype=float)
    layout = setup.layout
    if layout in ("lathe2", "lathe_live"):
        x = j.get("X", 0.0)
        radius = x / 2.0 if setup.diameter_mode else x
        z = j.get("Z", 0.0)
        y = j.get("Y", 0.0)
        tip_machine = np.array([radius, y, z])  # spindle along Z, X radial
        if layout == "lathe_live":
            tip = _rz(-j.get("C", 0.0)) @ tip_machine  # the part turns by C under the tool
            return tip, _rz(-j.get("C", 0.0)) @ np.array([1.0, 0.0, 0.0])
        return tip_machine, np.array([1.0, 0.0, 0.0])
    xyz = np.array([j.get("X", 0.0), j.get("Y", 0.0), j.get("Z", 0.0)])
    if layout == "gantry3":
        return xyz, Z_AXIS.copy()
    if layout == "gantry3_rotary":
        axis = setup.rotary_axis
        r = _rot(axis, j.get(axis, 0.0))
        return r.T @ (xyz - c) + c, r.T @ Z_AXIS
    if layout == "table_table5":
        tilt = setup.tilt_axis
        r = _rot(tilt, j.get(tilt, 0.0)) @ _rz(j.get("C", 0.0))  # machine = R (work - c) + c
        return r.T @ (xyz - c) + c, r.T @ Z_AXIS
    if layout == "head_table5":
        head_axis = _ry(j.get("B", 0.0)) @ Z_AXIS
        tip_machine = xyz - setup.pivot_length_mm * (head_axis - Z_AXIS)
        rc = _rz(j.get("C", 0.0))
        return rc.T @ (tip_machine - c) + c, rc.T @ head_axis
    if layout == "uv_taper":
        lower = np.array([j.get("X", 0.0), j.get("Y", 0.0), 0.0])
        upper = lower + np.array([j.get("U", 0.0), j.get("V", 0.0), setup.wire_height_mm])
        direction = upper - lower
        return lower, direction / np.linalg.norm(direction)
    return xyz, Z_AXIS.copy()


def inverse(setup: KinematicSetup, tip, tool_axis=None, previous: dict | None = None) -> dict:
    """(tool tip, tool axis) in workpiece coordinates -> joints. With two solutions the one
    closest to ``previous`` (shortest rotary move) is returned."""
    tip = np.asarray(tip, dtype=float)
    a = np.asarray(tool_axis if tool_axis is not None else Z_AXIS, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    c = np.asarray(setup.centre_mm, dtype=float)
    layout = setup.layout
    previous = previous or {}
    if layout in ("lathe2", "lathe_live"):
        if layout == "lathe_live":
            angle = math.degrees(math.atan2(tip[1], tip[0]))
            radius = math.hypot(tip[0], tip[1])
            return {"X": radius * 2.0 if setup.diameter_mode else radius, "Z": float(tip[2]), "C": _nearest(-angle, previous.get("C"))}
        return {"X": float(tip[0]) * (2.0 if setup.diameter_mode else 1.0), "Z": float(tip[2])}
    if layout == "gantry3":
        return {"X": float(tip[0]), "Y": float(tip[1]), "Z": float(tip[2])}
    if layout == "gantry3_rotary":
        axis = setup.rotary_axis
        if axis == "A":  # a = R_x(-A) z = (0, sin A, cos A)
            angle = math.degrees(math.atan2(a[1], a[2]))
        else:  # a = R_y(-B) z = (-sin B, 0, cos B)
            angle = math.degrees(math.atan2(-a[0], a[2]))
        angle = _nearest(angle, previous.get(axis))
        xyz = _rot(axis, angle) @ (tip - c) + c
        return {"X": float(xyz[0]), "Y": float(xyz[1]), "Z": float(xyz[2]), axis: angle}
    if layout == "table_table5":
        tilt = setup.tilt_axis
        if tilt == "A":
            c_deg = math.degrees(math.atan2(a[0], a[1])) if abs(a[0]) + abs(a[1]) > 1e-12 else previous.get("C", 0.0)
            ac = _rz(c_deg) @ a
            t_deg = math.degrees(math.atan2(ac[1], ac[2]))
        else:
            c_deg = math.degrees(math.atan2(-a[1], a[0])) if abs(a[0]) + abs(a[1]) > 1e-12 else previous.get("C", 0.0)
            ac = _rz(c_deg) @ a
            t_deg = math.degrees(math.atan2(-ac[0], ac[2]))
        c_deg = _nearest(c_deg, previous.get("C"))
        r = _rot(tilt, t_deg) @ _rz(c_deg)
        xyz = r @ (tip - c) + c
        return {"X": float(xyz[0]), "Y": float(xyz[1]), "Z": float(xyz[2]), tilt: t_deg, "C": c_deg}
    if layout == "head_table5":
        c_deg = -math.degrees(math.atan2(a[1], a[0])) if abs(a[0]) + abs(a[1]) > 1e-12 else previous.get("C", 0.0)
        c_deg = _nearest(c_deg, previous.get("C"))
        ar = _rz(c_deg) @ a
        b_deg = math.degrees(math.atan2(ar[0], ar[2]))
        tip_machine = _rz(c_deg) @ (tip - c) + c
        head_axis = _ry(b_deg) @ Z_AXIS
        xyz = tip_machine + setup.pivot_length_mm * (head_axis - Z_AXIS)
        return {"X": float(xyz[0]), "Y": float(xyz[1]), "Z": float(xyz[2]), "B": b_deg, "C": c_deg}
    if layout == "uv_taper":
        scale = setup.wire_height_mm / max(float(a[2]), 1e-9)
        return {"X": float(tip[0]), "Y": float(tip[1]), "U": float(a[0] * scale), "V": float(a[1] * scale)}
    return {"X": float(tip[0]), "Y": float(tip[1]), "Z": float(tip[2])}


def _nearest(angle: float, previous: float | None) -> float:
    """The equivalent angle (+/- 360) closest to the previous one: the shortest rotary move."""
    if previous is None:
        return float(angle)
    turns = round((float(previous) - float(angle)) / 360.0)
    return float(angle) + 360.0 * turns


def tool_points(setup: KinematicSetup, joints: dict, length: float, samples: int = 4) -> np.ndarray:
    """Points along the tool from the tip up its axis (workpiece coordinates), for collision checks."""
    tip, axis = forward(setup, joints)
    return np.asarray([tip + axis * (length * i / max(samples - 1, 1)) for i in range(samples)])


# ---------------------------------------------------------------- 3+2 positional (Block 18)
def tilted_frame(normal) -> np.ndarray:
    """3x3 rotation whose third column is the face normal (the tool axis for the setup); its
    first column follows world X projected onto the face where possible."""
    n = np.asarray(normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = helper - n * float(np.dot(helper, n))
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.column_stack((x, y, n))


def orient_points(setup: KinematicSetup, points, normal, origin=(0.0, 0.0, 0.0), previous: dict | None = None) -> tuple[list[dict], dict]:
    """3-axis points in the tilted setup frame (Z = face normal, from ``origin``) -> joints with
    fixed rotary angles (positional 3+2). Returns (joints per point, rotary angles)."""
    frame = tilted_frame(normal)
    o = np.asarray(origin, dtype=float)
    axis = frame[:, 2]
    out, angles = [], {}
    prev = dict(previous or {})
    for p in points:
        world = o + frame @ np.asarray(p, dtype=float)
        joints = inverse(setup, world, axis, prev)
        if not angles:
            angles = {k: v for k, v in joints.items() if k in ("A", "B", "C")}
        prev = joints
        out.append(joints)
    return out, angles


__all__ = ["KinematicSetup", "forward", "inverse", "orient_points", "tilted_frame", "tool_points"]
