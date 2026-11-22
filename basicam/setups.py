from __future__ import annotations

from dataclasses import dataclass, field

from basicam.operations import Operation
from basicam.theme import SETUP_PALETTE_DEFAULT

# lime, sky blue, warm amber, soft violet, teal, tan (no red: red is reserved for errors)
SETUP_PALETTE: tuple[str, ...] = SETUP_PALETTE_DEFAULT


@dataclass
class MachiningSetup:
    name: str
    orientation: str = "Top (+Z)"
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: str = SETUP_PALETTE[0]
    operations: list[Operation] = field(default_factory=list)
    work_frame: dict | None = None  # {"origin", "x_axis", "y_axis", "z_axis"} from Position > Work origin and axes
    work_offset: str = "G54"  # work coordinate system (G54-G59)
    safe_z_mm: float = 5.0  # rapid height above the stock top
    retract_clearance_mm: float = 2.0  # retract height above the stock top between passes
    stock_clearance_mm: float = 1.0  # extra margin around the part/stock for toolpaths
