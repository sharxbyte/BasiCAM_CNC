from dataclasses import dataclass

from basicam.models import PartModel
from basicam.operations import Operation


@dataclass(frozen=True)
class ReadinessItem:
    title: str
    ready: bool
    detail: str


def gcode_readiness(
    parts: list[PartModel],
    stock_defined: bool,
    material: str,
    operations: list[Operation],
    *,
    work_origin_set: bool = False,
    machine_profile_set: bool = False,
    postprocessor_set: bool = False,
    workholding_checked: bool = False,
    setup=None,
    untooled_sketches: list[str] | None = None,
) -> list[ReadinessItem]:
    valid_geometry = any(not part.mesh.is_empty for part in parts)
    valid_operations = bool(operations) and all(
        operation.tool.strip()
        and operation.target.strip()
        and operation.tool_diameter_mm > 0
        and operation.rpm > 0
        and operation.feed_mm_min > 0
        and operation.stepdown_mm > 0
        and operation.depth_mm > 0
        for operation in operations
    )
    extra: list[ReadinessItem] = []
    if setup is not None:
        safe = float(getattr(setup, "safe_z_mm", 0.0))
        retract = float(getattr(setup, "retract_clearance_mm", 0.0))
        fields_ok = (
            str(getattr(setup, "work_offset", "")) in {"G54", "G55", "G56", "G57", "G58", "G59"}
            and safe >= 1.0 and 0.5 <= retract <= safe and float(getattr(setup, "stock_clearance_mm", -1.0)) >= 0.0
        )
        extra.append(ReadinessItem(
            "Setup heights and work offset", fields_ok,
            "Set the work offset (G54-G59), safe Z, a retract clearance no higher than safe Z, and stock clearance.",
        ))
    if untooled_sketches is not None:
        extra.append(ReadinessItem(
            "Sketches tooled or marked", not untooled_sketches,
            "Give every sketch a tool path, mark it Construction or Future operation"
            + (f": {', '.join(untooled_sketches[:5])}" if untooled_sketches else "."),
        ))
    return extra + [
        ReadinessItem("Part geometry", valid_geometry, "Import or create the part to machine."),
        ReadinessItem("Stock shape", stock_defined, "Create a stock box or import stock geometry."),
        ReadinessItem(
            "Stock material",
            material != "Unspecified",
            "Choose a material to select and validate cutting data.",
        ),
        ReadinessItem(
            "Machining operations",
            valid_operations,
            "Set a tool diameter, spindle speed, feed, pass depth, final depth, and target for every operation.",
        ),
        ReadinessItem(
            "Work origin and orientation",
            work_origin_set,
            "Set the work coordinate origin and verify axis orientation.",
        ),
        ReadinessItem(
            "Machine profile",
            machine_profile_set,
            "Select a controller profile and verify axis travel and spindle limits.",
        ),
        ReadinessItem(
            "Postprocessor",
            postprocessor_set,
            "Choose a postprocessor matching the target controller.",
        ),
        ReadinessItem(
            "Workholding and clearances",
            workholding_checked,
            "Confirm clamps, safe heights, tool reach, and collision clearances.",
        ),
    ]