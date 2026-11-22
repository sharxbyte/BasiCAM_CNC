from dataclasses import dataclass
from enum import Enum


class LengthUnit(str, Enum):
    MILLIMETERS = "mm"
    INCHES = "in"


MM_PER_INCH = 25.4


@dataclass
class UnitSettings:
    display: LengthUnit = LengthUnit.MILLIMETERS
    imported_as: LengthUnit = LengthUnit.MILLIMETERS

    def display_value(self, millimeters: float) -> float:
        if self.display is LengthUnit.INCHES:
            return millimeters / MM_PER_INCH
        return millimeters

    def reinterpretation_factor(self, new_unit: LengthUnit) -> float:
        """Return the scale multiplier when raw file coordinates get a new unit."""
        old_mm_per_unit = self._millimeters_per_unit(self.imported_as)
        new_mm_per_unit = self._millimeters_per_unit(new_unit)
        return new_mm_per_unit / old_mm_per_unit

    def set_import_unit(self, new_unit: LengthUnit) -> float:
        factor = self.reinterpretation_factor(new_unit)
        self.imported_as = new_unit
        return factor

    @staticmethod
    def _millimeters_per_unit(unit: LengthUnit) -> float:
        return MM_PER_INCH if unit is LengthUnit.INCHES else 1.0
