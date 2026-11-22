from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from basicam.models import PartModel, PartRole
from basicam.operations import Operation
from basicam.setups import MachiningSetup


@dataclass
class WorkholdingFixture:
    """Represents a vise, clamp, strap, or fixture block in the CAM setup."""

    fixture_id: str
    name: str
    kind: str  # "Vise", "Strap Clamp", "Fixture Plate", "Soft Jaws", "Sub-plate"
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions_mm: tuple[float, float, float] = (120.0, 80.0, 25.0)
    clearance_margin_mm: float = 3.0
    enabled: bool = True
    keep_out: bool = False  # marked "keep out": the tool envelope may never enter it


@dataclass
class CollisionCheckResult:
    """Result of checking toolpath segments against stock, fixtures, and bounds."""

    has_collisions: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checked_segments: int = 0


class CollisionAndFixtureManager:
    """Manages setup workholding fixtures and checks toolpath motion against bounding boxes."""

    def __init__(self) -> None:
        self.fixtures: list[WorkholdingFixture] = []

    def add_fixture(self, fixture: WorkholdingFixture) -> None:
        self.fixtures.append(fixture)

    def remove_fixture(self, fixture_id: str) -> bool:
        before = len(self.fixtures)
        self.fixtures = [f for f in self.fixtures if f.fixture_id != fixture_id]
        return len(self.fixtures) < before

    def check_segments(
        self,
        segments: list[Any],
        tool_diameter_mm: float = 6.0,
        stock_dimensions: dict[str, float] | None = None,
    ) -> CollisionCheckResult:
        warnings: list[str] = []
        errors: list[str] = []
        radius = max(0.1, float(tool_diameter_mm) * 0.5)
        dims = stock_dimensions or {"X": 100.0, "Y": 80.0, "Z": 20.0}
        stock_z = float(dims.get("Z", 20.0))

        for idx, seg in enumerate(segments):
            sx, sy, sz = getattr(seg, "start", (0.0, 0.0, 0.0))
            ex, ey, ez = getattr(seg, "end", (0.0, 0.0, 0.0))
            kind = str(getattr(seg, "kind", "cut"))

            if kind == "rapid" and (ez < -0.5 or sz < -0.5):
                errors.append(
                    f"Segment {idx}: Rapid move drops below Z=0 (start Z={sz:.2f}, end Z={ez:.2f})."
                )

            if stock_z > 0 and abs(ez) > stock_z * 1.25:
                warnings.append(
                    f"Segment {idx} ({kind}): Z={ez:.2f} exceeds 125% of stock Z ({stock_z:.2f})."
                )

            for fix in self.fixtures:
                if not fix.enabled:
                    continue
                fx, fy, fz = fix.position_mm
                dx, dy, dz = fix.dimensions_mm
                pad = radius + float(fix.clearance_margin_mm)
                xmin, xmax = fx - dx * 0.5 - pad, fx + dx * 0.5 + pad
                ymin, ymax = fy - dy * 0.5 - pad, fy + dy * 0.5 + pad
                zmax = fz + dz + pad

                in_xy = (xmin <= ex <= xmax and ymin <= ey <= ymax) or (xmin <= sx <= xmax and ymin <= sy <= ymax)
                in_z = min(sz, ez) <= zmax
                if in_xy and in_z:
                    errors.append(
                        f"Segment {idx} ({kind}): Toolpath approaches fixture '{fix.name}' ({fix.fixture_id})."
                    )

        return CollisionCheckResult(
            has_collisions=len(errors) > 0,
            warnings=warnings,
            errors=errors,
            checked_segments=len(segments),
        )


@dataclass
class FeedsAndSpeedsRecommendation:
    """Calculated spindle speed and feed rate recommendation."""

    rpm: int
    feed_mm_min: float
    plunge_mm_min: float
    stepdown_mm: float
    surface_speed_m_min: float
    chip_load_mm: float
    notes: str = ""


class FeedsAndSpeedsCalculator:
    """Computes RPM, feed rate, and stepdown from material, tool geometry, and operation kind."""

    MATERIAL_SFM: dict[str, tuple[float, float]] = {
        "aluminum": (180.0, 0.04),
        "6061": (200.0, 0.045),
        "brass": (150.0, 0.035),
        "steel": (70.0, 0.02),
        "stainless": (50.0, 0.015),
        "wood": (350.0, 0.08),
        "delrin": (250.0, 0.06),
        "acrylic": (220.0, 0.05),
    }

    def recommend(
        self,
        material: str = "Aluminum",
        tool_diameter_mm: float = 6.0,
        flutes: int = 2,
        operation_kind: str = "Rough",
        max_rpm: int = 24000,
    ) -> FeedsAndSpeedsRecommendation:
        mat_l = (material or "").lower()
        sfm, base_chip = (120.0, 0.03)
        for key, params in self.MATERIAL_SFM.items():
            if key in mat_l:
                sfm, base_chip = params
                break

        op_l = (operation_kind or "Rough").lower()
        if "finish" in op_l:
            sfm *= 1.15
            base_chip *= 0.65
            stepdown = max(0.25, tool_diameter_mm * 0.15)
        elif "drill" in op_l:
            sfm *= 0.65
            base_chip *= 0.7
            stepdown = max(0.5, tool_diameter_mm * 0.5)
        elif "face" in op_l:
            sfm *= 1.05
            base_chip *= 0.85
            stepdown = max(0.25, tool_diameter_mm * 0.05)
        else:
            stepdown = max(0.4, tool_diameter_mm * 0.35)

        dia = max(0.5, float(tool_diameter_mm))
        n_flutes = max(1, int(flutes))
        # RPM ≈ (SFM * 1000) / (pi * D_mm)
        raw_rpm = (sfm * 1000.0) / (math.pi * dia)
        rpm = int(max(500, min(int(max_rpm), round(raw_rpm / 100.0) * 100)))
        feed = round(rpm * n_flutes * base_chip, 1)
        plunge = round(feed * 0.35, 1)

        return FeedsAndSpeedsRecommendation(
            rpm=rpm,
            feed_mm_min=max(20.0, feed),
            plunge_mm_min=max(10.0, plunge),
            stepdown_mm=round(stepdown, 2),
            surface_speed_m_min=round((math.pi * dia * rpm) / 1000.0, 1),
            chip_load_mm=round(base_chip, 4),
            notes=f"Material '{material or 'General'}', {n_flutes} flutes, Ø{dia:.2f} mm ({operation_kind}).",
        )


def export_cam_package(
    out_dir: str | Path,
    job_name: str,
    gcode_text: str,
    setup_sheet_md: str,
    setup_sheet_json: dict[str, Any],
    collision_report: CollisionCheckResult | None = None,
) -> dict[str, Path]:
    """Exports G-code, Markdown/JSON setup sheets, and optional collision report to a folder."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    safe_slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (job_name or "job")).strip("_") or "job"

    gcode_path = dest / f"{safe_slug}.nc"
    md_path = dest / f"{safe_slug}_setup_sheet.md"
    json_path = dest / f"{safe_slug}_setup_sheet.json"

    gcode_path.write_text(gcode_text, encoding="utf-8")
    md_path.write_text(setup_sheet_md, encoding="utf-8")
    json_path.write_text(json.dumps(setup_sheet_json, indent=2), encoding="utf-8")

    outputs: dict[str, Path] = {
        "gcode": gcode_path,
        "setup_sheet_md": md_path,
        "setup_sheet_json": json_path,
    }

    if collision_report is not None:
        col_path = dest / f"{safe_slug}_collision_report.json"
        col_payload = {
            "has_collisions": collision_report.has_collisions,
            "checked_segments": collision_report.checked_segments,
            "errors": collision_report.errors,
            "warnings": collision_report.warnings,
        }
        col_path.write_text(json.dumps(col_payload, indent=2), encoding="utf-8")
        outputs["collision_report"] = col_path

    return outputs
