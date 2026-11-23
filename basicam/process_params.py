"""Cutting-process parameters and cut charts for laser, plasma, waterjet and wire EDM (no Qt).

A cut chart entry is keyed on (process, material, thickness); lookups take the nearest entry of
the same process and material (thicker wins on ties). Everything is editable and stored as JSON
in QSettings (``process/cut_chart``); the built-in rows are starting points, not guarantees.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class ProcessParams:
    process: str = "laser"
    material: str = "Mild steel"
    thickness_mm: float = 3.0
    kerf_mm: float = 0.2  # beam / kerf / jet / wire+gap width
    feed_mm_min: float = 1500.0
    passes: int = 1
    # laser
    power_pct: float = 100.0
    min_power_pct: float = 0.0
    frequency_hz: float = 0.0  # PWM / pulse frequency where supported (0 = controller default)
    focus_offset_mm: float = 0.0
    dynamic_power: bool = True  # GRBL laser mode ($32=1): M4 scales power with speed
    hatch_spacing_mm: float = 0.1  # engraving fill
    hatch_angle_deg: float = 0.0
    # plasma
    cut_height_mm: float = 1.5
    pierce_height_mm: float = 3.8
    pierce_delay_s: float = 0.5
    thc: bool = True
    thc_lock_near_corners: bool = True
    hole_speed_pct: float = 60.0  # holes smaller than hole_limit run slower
    hole_limit_mm: float = 25.0
    arc_retries: int = 2
    # waterjet
    pierce_type: str = "dynamic"  # "stationary", "low-pressure", "dynamic", "wiggle"
    pierce_time_s: float = 2.0
    abrasive: bool = True
    abrasive_kg_min: float = 0.35
    corner_slowdown_pct: float = 50.0
    # wire EDM
    wire_diameter_mm: float = 0.25
    spark_gap_mm: float = 0.02
    skim_passes: int = 0
    taper_deg: float = 0.0
    # pierce / peck
    peck_count: int = 0  # thick plate: pierce in steps
    peck_depth_mm: float = 0.0
    peck_dwell_s: float = 0.2
    # leads
    lead_type: str = "line+arc"  # "line", "arc", "line+arc", "none"
    lead_in_mm: float = 3.0
    lead_out_mm: float = 1.0
    lead_angle_deg: float = 90.0
    overcut_mm: float = 0.5
    lead_safety_mm: float = 0.1  # extra clearance to neighbouring parts (Preferences)
    # tabs / micro-joints
    tab_count: int = 0
    tab_spacing_mm: float = 0.0
    tab_width_mm: float = 0.0  # 0 = 2 x kerf
    tab_height_mm: float = 0.0  # router partial-depth tabs
    micro_joints: bool = False
    # bevel / taper
    bevel_deg: float = 0.0
    good_side: str = "bottom"  # natural taper: the side that is kept to size ("top" or "bottom")
    # flash cutting
    flash_cut: bool = False

    @property
    def effective_kerf(self) -> float:
        if self.process == "wire_edm":
            return self.wire_diameter_mm + 2.0 * self.spark_gap_mm
        return self.kerf_mm

    def tab_width(self) -> float:
        return self.tab_width_mm if self.tab_width_mm > 0 else 2.0 * self.effective_kerf

    def part_spacing(self) -> float:
        return 2.0 * self.effective_kerf

    @classmethod
    def from_dict(cls, data: dict | None) -> "ProcessParams":
        known = {f.name for f in fields(cls)}
        params = cls()
        for key, value in dict(data or {}).items():
            if key in known:
                default = getattr(params, key)
                try:
                    setattr(params, key, type(default)(value) if not isinstance(default, bool) else bool(value))
                except (TypeError, ValueError):
                    pass
        return params

    def to_dict(self) -> dict:
        return asdict(self)


def _row(process, material, thickness, kerf, feed, **extra) -> ProcessParams:
    return ProcessParams(process=process, material=material, thickness_mm=thickness, kerf_mm=kerf, feed_mm_min=feed, **extra)


DEFAULT_CHART: list[ProcessParams] = [
    _row("laser", "Acrylic (PMMA)", 3.0, 0.15, 900.0, power_pct=80.0),
    _row("laser", "Plywood", 3.0, 0.18, 1200.0, power_pct=90.0),
    _row("laser", "MDF", 3.0, 0.2, 900.0, power_pct=95.0),
    _row("laser", "Mild steel", 1.0, 0.12, 3000.0, power_pct=100.0),
    _row("laser", "Stainless steel", 1.0, 0.12, 2500.0, power_pct=100.0),
    _row("plasma", "Mild steel", 6.0, 1.5, 2000.0, cut_height_mm=1.5, pierce_height_mm=3.8, pierce_delay_s=0.5),
    _row("plasma", "Mild steel", 12.0, 1.9, 1000.0, cut_height_mm=1.5, pierce_height_mm=4.5, pierce_delay_s=1.0),
    _row("plasma", "Aluminum", 6.0, 1.7, 2500.0),
    _row("waterjet", "Mild steel", 10.0, 0.9, 250.0, pierce_time_s=4.0),
    _row("waterjet", "Glass", 6.0, 0.8, 600.0, pierce_type="low-pressure"),
    _row("waterjet", "Stone", 20.0, 1.0, 120.0),
    _row("wire_edm", "Mild steel", 20.0, 0.29, 3.0, wire_diameter_mm=0.25, spark_gap_mm=0.02),
]


class CutChart:
    KEY = "process/cut_chart"

    def __init__(self, settings=None) -> None:
        self.settings = settings

    def rows(self) -> list[ProcessParams]:
        raw = self.settings.value(self.KEY, "") if self.settings is not None else ""
        try:
            items = json.loads(raw) if isinstance(raw, str) and raw else []
        except ValueError:
            items = []
        rows = [ProcessParams.from_dict(r) for r in items if isinstance(r, dict)]
        return rows or [ProcessParams.from_dict(r.to_dict()) for r in DEFAULT_CHART]

    def save_rows(self, rows: list[ProcessParams]) -> None:
        self.settings.setValue(self.KEY, json.dumps([r.to_dict() for r in rows]))

    def lookup(self, process: str, material: str, thickness_mm: float) -> ProcessParams:
        same = [r for r in self.rows() if r.process == process and r.material == material]
        if not same:
            same = [r for r in self.rows() if r.process == process]
        if not same:
            return ProcessParams(process=process, material=material, thickness_mm=thickness_mm)
        best = min(same, key=lambda r: (abs(r.thickness_mm - thickness_mm), -r.thickness_mm))
        params = ProcessParams.from_dict(best.to_dict())
        params.material, params.thickness_mm = material, thickness_mm
        return params


__all__ = ["CutChart", "DEFAULT_CHART", "ProcessParams"]
