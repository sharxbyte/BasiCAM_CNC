import unittest

from basicam.operations import Operation, operation_warnings
from basicam.models import PartModel
from basicam.readiness import gcode_readiness
from basicam.units import LengthUnit, UnitSettings
import pyvista as pv


class UnitSettingsTests(unittest.TestCase):
    def test_display_units_do_not_change_import_scale(self) -> None:
        units = UnitSettings()
        units.display = LengthUnit.INCHES
        self.assertAlmostEqual(units.display_value(25.4), 1.0)
        self.assertEqual(units.imported_as, LengthUnit.MILLIMETERS)

    def test_reinterpreting_mm_coordinates_as_inches_scales_physical_size(self) -> None:
        units = UnitSettings()
        factor = units.set_import_unit(LengthUnit.INCHES)
        self.assertAlmostEqual(factor, 25.4)
        self.assertEqual(units.imported_as, LengthUnit.INCHES)


class OperationWarningTests(unittest.TestCase):
    def test_tap_before_drill_and_roughing_is_flagged(self) -> None:
        tap = Operation("Tap M6", "tap", "M6 tap", 0, 100, 1, 10, "conventional")
        messages = operation_warnings([tap])
        self.assertEqual(len(messages), 2)

    def test_drill_then_tap_does_not_report_missing_drill(self) -> None:
        drill = Operation("Drill", "drill", "5 mm drill", 1800, 80, 2, 10, "conventional")
        tap = Operation("Tap", "tap", "M6 tap", 0, 100, 1, 10, "conventional")
        messages = operation_warnings([drill, tap])
        self.assertFalse(any("drilling operation" in message for message in messages))


class ReadinessTests(unittest.TestCase):
    def test_missing_machine_setup_is_spelled_out(self) -> None:
        part = PartModel("Block", pv.Cube())
        operation = Operation("Rough", "rough", "6 mm end mill", 6000, 400, 2, 8, "climb")
        items = gcode_readiness([part], True, "Aluminum", [operation])
        missing = [item.title for item in items if not item.ready]
        self.assertEqual(
            missing,
            ["Work origin and orientation", "Machine profile", "Postprocessor", "Workholding and clearances"],
        )

    def test_empty_job_lists_geometry_stock_material_and_operations(self) -> None:
        items = gcode_readiness([], False, "Unspecified", [])
        missing = [item.title for item in items if not item.ready]
        self.assertEqual(
            missing,
            ["Part geometry", "Stock shape", "Stock material", "Machining operations",
             "Work origin and orientation", "Machine profile", "Postprocessor",
             "Workholding and clearances"],
        )


if __name__ == "__main__":
    unittest.main()
