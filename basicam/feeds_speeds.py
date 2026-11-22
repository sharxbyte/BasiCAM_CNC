from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MaterialCuttingSpec:
    sfm: float
    chip_load_mm_per_tooth: float
    stepdown_factor: float


MATERIAL_CUTTING_SPECS: dict[str, MaterialCuttingSpec] = {
    "Wood (soft)": MaterialCuttingSpec(sfm=1200.0, chip_load_mm_per_tooth=0.150, stepdown_factor=0.75),
    "Wood (hard)": MaterialCuttingSpec(sfm=900.0, chip_load_mm_per_tooth=0.100, stepdown_factor=0.50),
    "Plastic": MaterialCuttingSpec(sfm=800.0, chip_load_mm_per_tooth=0.080, stepdown_factor=0.50),
    "Aluminum": MaterialCuttingSpec(sfm=500.0, chip_load_mm_per_tooth=0.050, stepdown_factor=0.35),
    "Brass": MaterialCuttingSpec(sfm=350.0, chip_load_mm_per_tooth=0.045, stepdown_factor=0.30),
    "Bronze": MaterialCuttingSpec(sfm=220.0, chip_load_mm_per_tooth=0.035, stepdown_factor=0.25),
    "Mild steel": MaterialCuttingSpec(sfm=120.0, chip_load_mm_per_tooth=0.025, stepdown_factor=0.15),
    "Steel": MaterialCuttingSpec(sfm=120.0, chip_load_mm_per_tooth=0.025, stepdown_factor=0.15),
    "Stainless steel": MaterialCuttingSpec(sfm=80.0, chip_load_mm_per_tooth=0.020, stepdown_factor=0.10),
    "Other": MaterialCuttingSpec(sfm=300.0, chip_load_mm_per_tooth=0.040, stepdown_factor=0.25),
}


@dataclass(frozen=True)
class CuttingRecommendation:
    rpm: int
    feed_mm_min: float
    stepdown_mm: float


def recommend_cutting_params(
    material: str,
    diameter_mm: float,
    flutes: int = 2,
    multiplier: float = 1.0,
    operation_kind: str = "Rough",
) -> CuttingRecommendation:
    spec = MATERIAL_CUTTING_SPECS.get(material, MATERIAL_CUTTING_SPECS["Other"])
    safe_diameter_mm = max(float(diameter_mm), 0.1)
    diameter_in = safe_diameter_mm / 25.4
    safe_flutes = max(int(flutes), 1)
    factor = min(max(float(multiplier), 0.1), 10.0)

    raw_rpm = (spec.sfm * 3.82) / diameter_in
    op_lower = operation_kind.casefold()
    if op_lower == "finish":
        raw_rpm *= 1.15
        feed_factor = 0.75
        doc_factor = spec.stepdown_factor * 0.35
    elif op_lower in {"drill", "ream", "tap"}:
        raw_rpm *= 0.70
        feed_factor = 0.50
        doc_factor = spec.stepdown_factor * 0.50
    else:
        feed_factor = 1.0
        doc_factor = spec.stepdown_factor

    rpm = max(1, min(60000, int(round(raw_rpm * factor))))
    feed_mm_min = max(1.0, round(rpm * safe_flutes * spec.chip_load_mm_per_tooth * feed_factor * factor, 1))
    stepdown_mm = max(0.05, round(safe_diameter_mm * doc_factor, 2))
    return CuttingRecommendation(rpm=rpm, feed_mm_min=feed_mm_min, stepdown_mm=stepdown_mm)


def slider_to_multiplier(slider_value: int) -> float:
    """Map slider integer [-100..100] logarithmically onto [0.1..10.0] with 0 -> 1.0."""
    clamped = max(-100, min(100, int(slider_value)))
    return round(10.0 ** (clamped / 100.0), 2)


def multiplier_to_slider(multiplier: float) -> int:
    clamped = min(max(float(multiplier), 0.1), 10.0)
    return int(round(math.log10(clamped) * 100.0))
