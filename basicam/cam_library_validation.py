from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basicam.operations import Operation
from basicam.setups import MachiningSetup


@dataclass
class ToolLibraryEntry:
    """CNC cutting tool definition with default cutting parameters."""

    tool_id: str
    name: str
    kind: str  # "End mill", "Ball mill", "Drill", "Chamfer mill", "Face mill"
    diameter_mm: float
    flutes: int = 2
    default_rpm: int = 8000
    default_feed_mm_min: float = 500.0
    default_stepdown_mm: float = 1.5
    material_tag: str = "HSS / Carbide"


DEFAULT_TOOL_LIBRARY: list[ToolLibraryEntry] = [
    ToolLibraryEntry("T1", "6mm 2F Flat End Mill", "End mill", 6.0, 2, 10000, 750.0, 1.5, "Aluminum / Wood"),
    ToolLibraryEntry("T2", "3mm 2F Flat End Mill", "End mill", 3.0, 2, 12000, 450.0, 0.8, "Aluminum / Plastics"),
    ToolLibraryEntry("T3", "6mm 2F Ball Nose", "Ball mill", 6.0, 2, 10000, 600.0, 1.0, "3D Contouring"),
    ToolLibraryEntry("T4", "5mm Jobber Drill", "Drill", 5.0, 2, 4000, 200.0, 2.0, "General"),
    ToolLibraryEntry("T5", "25mm Face Mill", "Face mill", 25.0, 4, 6000, 900.0, 0.5, "Facing"),
]


@dataclass
class GCodeValidationIssue:
    """Single validation diagnostic found during pre-export G-code/toolpath inspection."""

    severity: str  # "error", "warning", "info"
    code: str
    message: str
    line_or_op: str = ""


class GCodeValidator:
    """Checks toolpaths, feeds/speeds, depths, and G-code text for common CNC hazards."""

    def validate(
        self,
        gcode_text: str = "",
        operations: list[Operation] | None = None,
        setups: list[MachiningSetup] | None = None,
        stock_dimensions: dict[str, float] | None = None,
    ) -> list[GCodeValidationIssue]:
        issues: list[GCodeValidationIssue] = []
        ops = list(operations or [])
        if not ops and setups:
            for s in setups:
                ops.extend(s.operations)

        stock_z = float((stock_dimensions or {}).get("Z", 20.0))

        for op in ops:
            if op.tool_diameter_mm <= 0:
                issues.append(
                    GCodeValidationIssue("error", "TOOL_DIA", "Tool diameter must be greater than 0 mm.", op.name)
                )
            if op.rpm <= 0:
                issues.append(
                    GCodeValidationIssue("warning", "SPINDLE_RPM", "Spindle RPM is 0; ensure manual spindle is intended.", op.name)
                )
            if op.feed_mm_min <= 0:
                issues.append(
                    GCodeValidationIssue("error", "FEED_RATE", "Feed rate must be greater than 0 mm/min.", op.name)
                )
            if op.stepdown_mm <= 0:
                issues.append(
                    GCodeValidationIssue("error", "STEPDOWN", "Stepdown must be greater than 0 mm.", op.name)
                )
            elif op.depth_mm > 0 and op.stepdown_mm > op.depth_mm:
                issues.append(
                    GCodeValidationIssue("warning", "STEPDOWN_GT_DEPTH", "Stepdown exceeds total operation depth.", op.name)
                )
            if stock_z > 0 and abs(op.depth_mm) > stock_z:
                issues.append(
                    GCodeValidationIssue(
                        "warning",
                        "DEPTH_EXCEEDS_STOCK",
                        f"Depth ({op.depth_mm:.2f} mm) exceeds stock Z ({stock_z:.2f} mm).",
                        op.name,
                    )
                )

        if gcode_text.strip():
            lines = gcode_text.splitlines()
            has_units = any(("G20" in ln or "G21" in ln) for ln in lines)
            has_end = any(("M30" in ln or "M2" in ln or "M02" in ln) for ln in lines)
            if not has_units:
                issues.append(
                    GCodeValidationIssue("warning", "NO_UNITS", "Program does not specify G20 (inch) or G21 (mm).", "Header")
                )
            if not has_end:
                issues.append(
                    GCodeValidationIssue("info", "NO_M30", "Program does not end with M30 / M2.", "Footer")
                )

        return issues
