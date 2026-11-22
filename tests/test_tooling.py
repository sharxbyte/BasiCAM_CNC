import unittest

import numpy as np

from basicam.sketches import construction_line
from basicam.tooling import ToolDefinition, create_tool_mesh, default_tool_inventory


class ToolingTests(unittest.TestCase):
    def test_default_cutter_families_generate_3d_envelopes(self) -> None:
        tools = default_tool_inventory()
        self.assertGreaterEqual(len(tools), 14)
        for tool in tools:
            with self.subTest(tool=tool.tool_type):
                mesh = create_tool_mesh(tool)
                self.assertGreater(mesh.n_cells, 0)
                self.assertAlmostEqual(mesh.bounds[1] - mesh.bounds[0], tool.diameter_mm)

    def test_custom_profile_round_trips(self) -> None:
        original = ToolDefinition(
            "Custom dovetail",
            "Custom",
            profile_points=[(0, 0), (3, 0), (2, 4), (3, 12)],
        )
        restored = ToolDefinition.from_dict(original.to_dict())
        self.assertEqual(restored.profile_points, original.profile_points)
        self.assertEqual(restored.name, original.name)

    def test_construction_line_is_centered_and_has_requested_extent(self) -> None:
        points = construction_line(np.array([1.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]), 20)
        self.assertTrue(np.allclose(points.mean(axis=0), [2.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(points[1] - points[0])), 20.0)

    def test_construction_line_rejects_identical_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            construction_line(np.zeros(3), np.zeros(3), 10)


if __name__ == "__main__":
    unittest.main()