"""Material library (no Qt): names, family, density and which processes can cut them.

Stored as JSON in QSettings (``materials/library``) like the machine inventory. The cut charts
of later blocks key on ``Material.name``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

FAMILIES = ("metal", "wood", "plastic", "composite", "foam", "stone", "glass", "other")
PROCESSES = ("mill", "laser", "plasma", "waterjet", "wire_edm", "lathe")


@dataclass
class Material:
    name: str
    family: str = "other"
    density_g_cm3: float = 1.0
    processes: dict[str, bool] = field(default_factory=lambda: {p: True for p in PROCESSES})
    conductive: bool = False  # plasma and wire EDM need a conductive material
    notes: str = ""

    def suits(self, process: str) -> bool:
        if process in ("plasma", "wire_edm") and not self.conductive:
            return False
        return bool(self.processes.get(process, True))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Material":
        data = dict(data or {})
        processes = {p: True for p in PROCESSES}
        processes.update({str(k): bool(v) for k, v in dict(data.get("processes", {}) or {}).items()})
        family = str(data.get("family", "other"))
        return cls(
            name=str(data.get("name", "Unnamed")),
            family=family if family in FAMILIES else "other",
            density_g_cm3=float(data.get("density_g_cm3", 1.0) or 1.0),
            processes=processes,
            conductive=bool(data.get("conductive", False)),
            notes=str(data.get("notes", "")),
        )


def _m(name, family, density, conductive=False, no=()):
    processes = {p: p not in no for p in PROCESSES}
    return Material(name, family, density, processes, conductive)


DEFAULT_MATERIALS: list[Material] = [
    _m("Unspecified", "other", 1.0),
    _m("Aluminum", "metal", 2.70, True),
    _m("Brass", "metal", 8.50, True),
    _m("Bronze", "metal", 8.80, True),
    _m("Copper", "metal", 8.96, True),
    _m("Mild steel", "metal", 7.85, True),
    _m("Stainless steel", "metal", 8.00, True),
    _m("Titanium", "metal", 4.43, True),
    _m("Plastic", "plastic", 1.20, no=("plasma", "wire_edm")),
    _m("Acrylic (PMMA)", "plastic", 1.18, no=("plasma", "wire_edm")),
    _m("Polycarbonate", "plastic", 1.20, no=("plasma", "wire_edm", "laser")),  # laser: fumes/poor edge
    _m("PVC", "plastic", 1.40, no=("plasma", "wire_edm", "laser")),  # laser: releases chlorine gas
    _m("Wood (soft)", "wood", 0.45, no=("plasma", "wire_edm")),
    _m("Wood (hard)", "wood", 0.75, no=("plasma", "wire_edm")),
    _m("Plywood", "wood", 0.60, no=("plasma", "wire_edm")),
    _m("MDF", "wood", 0.75, no=("plasma", "wire_edm")),
    _m("Carbon fibre", "composite", 1.60, no=("plasma", "wire_edm")),
    _m("Foam", "foam", 0.05, no=("plasma", "wire_edm", "waterjet")),
    _m("Glass", "glass", 2.50, no=("plasma", "wire_edm", "mill", "lathe")),
    _m("Stone", "stone", 2.70, no=("plasma", "wire_edm", "laser")),
    _m("Other", "other", 1.0),
]


class MaterialStore:
    KEY = "materials/library"

    def __init__(self, settings) -> None:
        self.settings = settings

    def list(self) -> list[Material]:
        raw = self.settings.value(self.KEY, "") if self.settings is not None else ""
        try:
            items = json.loads(raw) if isinstance(raw, str) and raw else list(raw or [])
        except (TypeError, ValueError):
            items = []
        materials = [Material.from_dict(item) for item in items if isinstance(item, dict)]
        return materials or [Material.from_dict(m.to_dict()) for m in DEFAULT_MATERIALS]

    def names(self) -> list[str]:
        return [m.name for m in self.list()]

    def get(self, name: str) -> Material | None:
        return next((m for m in self.list() if m.name == name), None)

    def save_all(self, materials: list[Material]) -> None:
        self.settings.setValue(self.KEY, json.dumps([m.to_dict() for m in materials]))

    def save(self, material: Material) -> None:
        materials = [m for m in self.list() if m.name != material.name]
        materials.append(material)
        self.save_all(materials)

    def delete(self, name: str) -> None:
        self.save_all([m for m in self.list() if m.name != name])


__all__ = ["DEFAULT_MATERIALS", "FAMILIES", "PROCESSES", "Material", "MaterialStore"]
