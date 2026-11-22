import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ezdxf
import trimesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.IGESControl import IGESControl_Writer
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer

from basicam.importers import load_part, load_part_geometry


class ImporterTests(unittest.TestCase):
    def test_stl_import_applies_import_unit_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.stl"
            trimesh.creation.box(extents=(1, 2, 3)).export(path)

            geometry = load_part_geometry(path, 25.4)
            part = geometry.mesh

            self.assertAlmostEqual(part.bounds[1] - part.bounds[0], 25.4)
            self.assertAlmostEqual(part.bounds[3] - part.bounds[2], 50.8)
            self.assertIsNotNone(geometry.solid_shape)
            self.assertEqual(geometry.geometry_type, "Solid")
            self.assertTrue(TopExp_Explorer(geometry.solid_shape, TopAbs_SOLID).More())
            self.assertIn("Converted watertight", geometry.solid_conversion_message)

    def test_open_stl_remains_mesh_with_conversion_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open_triangle.stl"
            open_triangle = trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 2]],
                process=False,
            )
            open_triangle.export(path)

            geometry = load_part_geometry(path, 1.0)

            self.assertIsNone(geometry.solid_shape)
            self.assertEqual(geometry.geometry_type, "Mesh / drawing")
            self.assertIn("open or non-manifold", geometry.solid_conversion_message)

    def test_disconnected_closed_stl_bodies_convert_to_compound_solid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_blocks.stl"
            first = trimesh.creation.box(extents=(2, 2, 2))
            second = trimesh.creation.box(extents=(2, 2, 2))
            second.apply_translation((5, 0, 0))
            trimesh.util.concatenate([first, second]).export(path)

            geometry = load_part_geometry(path, 1.0)
            explorer = TopExp_Explorer(geometry.solid_shape, TopAbs_SOLID)
            solid_count = 0
            while explorer.More():
                solid_count += 1
                explorer.Next()

            self.assertEqual(solid_count, 2)
            self.assertAlmostEqual(geometry.mesh.bounds[1] - geometry.mesh.bounds[0], 7.0)

    def test_dxf_circle_import_creates_closed_linework(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "circle.dxf"
            document = ezdxf.new()
            document.modelspace().add_circle((0, 0), radius=10)
            document.saveas(path)

            part = load_part(path, 1.0)

            self.assertEqual(part.n_points, 64)
            self.assertEqual(part.n_lines, 64)

    def test_step_import_preserves_selectable_cad_faces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.step"
            writer = STEPControl_Writer()
            shape = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
            self.assertEqual(writer.Transfer(shape, STEPControl_AsIs), IFSelect_RetDone)
            self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)

            geometry = load_part_geometry(path, 1.0)

            self.assertEqual(len(set(geometry.mesh.cell_data["feature_id"].tolist())), 6)
            self.assertAlmostEqual(geometry.mesh.bounds[5] - geometry.mesh.bounds[4], 30.0)
            self.assertGreater(geometry.edge_mesh.n_lines, 0)
            self.assertIsNotNone(geometry.solid_shape)
            self.assertTrue(TopExp_Explorer(geometry.solid_shape, TopAbs_SOLID).More())

    def test_iges_import_reads_open_cascade_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.igs"
            writer = IGESControl_Writer("MM")
            self.assertTrue(writer.AddShape(BRepPrimAPI_MakeBox(10, 20, 30).Shape()))
            self.assertTrue(writer.Write(str(path)))

            part = load_part(path, 1.0)

            self.assertGreater(part.n_cells, 0)
            self.assertAlmostEqual(part.bounds[3] - part.bounds[2], 20.0, delta=0.1)

    def test_sldprt_explains_linux_neutral_format_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.sldprt"
            path.touch()
            with patch("basicam.importers.sys.platform", "linux"):
                with self.assertRaisesRegex(ValueError, "proprietary format"):
                    load_part(path, 1.0)


if __name__ == "__main__":
    unittest.main()
