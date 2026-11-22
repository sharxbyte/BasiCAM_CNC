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
class MachineKinematicsProfile:
    """CNC machine travel limits, axis naming, and home/park coordinates."""

    name: str = "3-Axis Benchtop Mill"
    x_min_mm: float = -200.0
    x_max_mm: float = 200.0
    y_min_mm: float = -200.0
    y_max_mm: float = 200.0
    z_min_mm: float = -80.0
    z_max_mm: float = 80.0
    max_feed_mm_min: float = 3500.0
    park_position_mm: tuple[float, float, float] = (0.0, 0.0, 40.0)


@dataclass
class SoftLimitCheckResult:
    """Result of checking toolpath segments against machine travel envelopes and feed limits."""

    within_limits: bool
    violations: list[str] = field(default_factory=list)
    checked_segments: int = 0


class MachineEnvelopeChecker:
    """Validates toolpath coordinates and feeds against a MachineKinematicsProfile."""

    def __init__(self, profile: MachineKinematicsProfile | None = None) -> None:
        self.profile = profile or MachineKinematicsProfile()

    def check_segments(self, segments: list[Any]) -> SoftLimitCheckResult:
        p = self.profile
        violations: list[str] = []
        for idx, seg in enumerate(segments):
            for pt_label, pt in (("start", getattr(seg, "start", None)), ("end", getattr(seg, "end", None))):
                if pt is None or len(pt) < 3:
                    continue
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                if not (p.x_min_mm <= x <= p.x_max_mm):
                    violations.append(
                        f"Segment {idx} ({pt_label}): X={x:.3f} outside [{p.x_min_mm}, {p.x_max_mm}]."
                    )
                if not (p.y_min_mm <= y <= p.y_max_mm):
                    violations.append(
                        f"Segment {idx} ({pt_label}): Y={y:.3f} outside [{p.y_min_mm}, {p.y_max_mm}]."
                    )
                if not (p.z_min_mm <= z <= p.z_max_mm):
                    violations.append(
                        f"Segment {idx} ({pt_label}): Z={z:.3f} outside [{p.z_min_mm}, {p.z_max_mm}]."
                    )
            feed = float(getattr(seg, "feed_mm_min", 0.0))
            if feed > p.max_feed_mm_min:
                violations.append(
                    f"Segment {idx}: Feed {feed:.1f} mm/min exceeds machine max {p.max_feed_mm_min:.1f} mm/min."
                )

        return SoftLimitCheckResult(
            within_limits=len(violations) == 0,
            violations=violations,
            checked_segments=len(segments),
        )


@dataclass
class StockRemovalEstimate:
    """Approximate volumetric material removal breakdown across operations."""

    stock_volume_mm3: float
    estimated_removed_mm3: float
    remaining_volume_mm3: float
    removal_ratio: float
    per_operation_mm3: list[dict[str, Any]] = field(default_factory=list)


class StockRemovalEstimator:
    """Estimates swept material removal volume from operation dimensions and stock bounds."""

    def estimate(
        self,
        stock_dimensions: dict[str, float] | None = None,
        operations: list[Operation] | None = None,
        setups: list[MachiningSetup] | None = None,
    ) -> StockRemovalEstimate:
        dims = stock_dimensions or {"X": 100.0, "Y": 80.0, "Z": 20.0}
        sx = max(0.0, float(dims.get("X", 100.0)))
        sy = max(0.0, float(dims.get("Y", 80.0)))
        sz = max(0.0, float(dims.get("Z", 20.0)))
        stock_vol = sx * sy * sz

        ops = list(operations or [])
        if not ops and setups:
            for s in setups:
                ops.extend(s.operations)

        per_op: list[dict[str, Any]] = []
        total_removed = 0.0
        for op in ops:
            dia = max(0.2, float(op.tool_diameter_mm))
            depth = max(0.0, abs(float(op.depth_mm)))
            kind_l = (op.kind or "").lower()
            if "drill" in kind_l:
                area = math.pi * (dia * 0.5) ** 2
            elif "face" in kind_l:
                area = sx * sy
            elif "pocket" in kind_l:
                area = (sx * 0.6) * (sy * 0.6)
            else:
                perimeter = 2.0 * (sx + sy)
                area = perimeter * dia

            vol = min(stock_vol, area * depth)
            total_removed += vol
            per_op.append(
                {
                    "operation": op.name,
                    "kind": op.kind,
                    "removed_mm3": round(vol, 2),
                }
            )

        removed_capped = min(stock_vol, total_removed)
        remaining = max(0.0, stock_vol - removed_capped)
        ratio = (removed_capped / stock_vol) if stock_vol > 0 else 0.0

        return StockRemovalEstimate(
            stock_volume_mm3=round(stock_vol, 2),
            estimated_removed_mm3=round(removed_capped, 2),
            remaining_volume_mm3=round(remaining, 2),
            removal_ratio=round(ratio, 4),
            per_operation_mm3=per_op,
        )
