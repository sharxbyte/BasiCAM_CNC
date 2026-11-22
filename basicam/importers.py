from pathlib import Path
from dataclasses import dataclass
import sys
import tempfile

import ezdxf
import numpy as np
import pyvista as pv
import trimesh


SUPPORTED_SUFFIXES = {".sldprt", ".step", ".stp", ".igs", ".iges", ".stl", ".dxf", ".dwg"}
HANDLED_DXF_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE"}

_dwg_converter_path: str | None = None


def set_dwg_converter(path: str | None) -> None:
    """User-installed ODA File Converter executable (Settings > Preferences)."""
    global _dwg_converter_path
    _dwg_converter_path = path or None


@dataclass
class ImportedGeometry:
    mesh: pv.PolyData
    brep_shape: object | None = None
    solid_shape: object | None = None
    edge_mesh: pv.PolyData | None = None
    solid_conversion_message: str = ""
    native_brep: bool = False
    unsupported: list[str] | None = None  # drawing entity types that were not imported

    @property
    def geometry_type(self) -> str:
        if self.solid_shape is not None:
            return "Solid"
        if self.brep_shape is not None:
            return "Surface model"
        return "Mesh / drawing"


def load_part(path: str | Path, millimeters_per_unit: float) -> pv.PolyData:
    return load_part_geometry(path, millimeters_per_unit).mesh


def load_part_geometry(path: str | Path, millimeters_per_unit: float) -> ImportedGeometry:
    source = Path(path)
    suffix = source.suffix.lower()
    brep_shape = None
    solid_shape = None
    edge_mesh = None
    native_brep = False
    solid_conversion_message = ""
    unsupported: list[str] | None = None

    if suffix == ".stl":
        imported = trimesh.load(source, force="mesh")
        if isinstance(imported, trimesh.Scene):
            imported = imported.dump(concatenate=True)
        if not isinstance(imported, trimesh.Trimesh) or imported.is_empty:
            raise ValueError("The STL file does not contain a triangle mesh.")
        faces = np.column_stack(
            (np.full(len(imported.faces), 3, dtype=np.int64), imported.faces)
        ).ravel()
        part = pv.PolyData(imported.vertices, faces)
        part.cell_data["feature_id"] = np.arange(len(imported.faces), dtype=np.int64)
        solid_shape, solid_conversion_message = _mesh_to_solid(imported)
        brep_shape = solid_shape
    elif suffix == ".dxf":
        part, unsupported = _dxf_document_to_mesh(ezdxf.readfile(source))
    elif suffix == ".dwg":
        part, unsupported = _dxf_document_to_mesh(_read_dwg(source))
        solid_conversion_message = "Converted from DWG with the ODA File Converter."
    elif suffix in {".step", ".stp", ".igs", ".iges"}:
        part, brep_shape, edge_mesh = _load_open_cascade(source)
        native_brep = True
        if _contains_solid(brep_shape):
            solid_shape = brep_shape
    elif suffix == ".sldprt":
        geometry = _load_sldprt(source, millimeters_per_unit)
        return geometry
    else:
        raise ValueError(
            f"{suffix or 'This file'} is not supported. "
            "Supported formats are SLDPRT (Windows with SolidWorks), STEP, IGES, STL, DXF and DWG (with the ODA File Converter)."
        )

    part.scale((millimeters_per_unit,) * 3, inplace=True)
    if brep_shape is not None:
        brep_shape = scale_solid_shape(brep_shape, millimeters_per_unit)
        if solid_shape is not None:
            solid_shape = brep_shape
    if edge_mesh is not None:
        edge_mesh.scale((millimeters_per_unit,) * 3, inplace=True)
    return ImportedGeometry(
        part, brep_shape, solid_shape, edge_mesh, solid_conversion_message, native_brep, unsupported
    )


def _read_dwg(path: Path):
    """DWG -> ezdxf document through ezdxf.addons.odafc and the user-installed ODA File Converter."""
    from ezdxf.addons import odafc

    if _dwg_converter_path:
        ezdxf.options.set("odafc-addon", "win_exec_path", _dwg_converter_path)
        ezdxf.options.set("odafc-addon", "unix_exec_path", _dwg_converter_path)
    if not odafc.is_installed():
        raise ValueError(
            "DWG import needs the free ODA File Converter. Install it, then set its path in "
            "Options > Preferences > DWG converter."
        )
    try:
        return odafc.readfile(str(path))
    except odafc.ODAFCError as error:
        raise ValueError(f"The ODA File Converter could not convert this DWG: {error}") from error


def _load_dxf(path: Path) -> pv.PolyData:
    return _dxf_document_to_mesh(ezdxf.readfile(path))[0]


def _dxf_document_to_mesh(document) -> tuple[pv.PolyData, list[str]]:
    """Linework from LINE, LWPOLYLINE, POLYLINE and CIRCLE; other entity types are listed as unsupported."""
    points: list[tuple[float, float, float]] = []
    line_cells: list[int] = []
    feature_ids: list[int] = []

    def add_polyline(
        vertices: list[tuple[float, float, float]],
        feature_id: int,
        closed: bool = False,
    ) -> None:
        if len(vertices) < 2:
            return
        start = len(points)
        points.extend(vertices)
        count = len(vertices)
        segment_count = count if closed else count - 1
        for index in range(segment_count):
            line_cells.extend((2, start + index, start + ((index + 1) % count)))
            feature_ids.append(feature_id)

    modelspace = document.modelspace()
    skipped: dict[str, int] = {}
    for feature_id, entity in enumerate(modelspace):
        kind = entity.dxftype()
        if kind not in HANDLED_DXF_TYPES:
            skipped[kind] = skipped.get(kind, 0) + 1
            continue
        if kind == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            add_polyline(
                [(start.x, start.y, start.z), (end.x, end.y, end.z)], feature_id
            )
        elif kind == "LWPOLYLINE":
            vertices = [
                (float(vertex[0]), float(vertex[1]), 0.0)
                for vertex in entity.get_points("xy")
            ]
            add_polyline(vertices, feature_id, entity.closed)
        elif kind == "POLYLINE":
            vertices = [
                (vertex.dxf.location.x, vertex.dxf.location.y, vertex.dxf.location.z)
                for vertex in entity.vertices
            ]
            add_polyline(vertices, feature_id, entity.is_closed)
        elif kind == "CIRCLE":
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            angles = np.linspace(0.0, 2.0 * np.pi, 65)[:-1]
            vertices = [
                (
                    center.x + radius * float(np.cos(angle)),
                    center.y + radius * float(np.sin(angle)),
                    center.z,
                )
                for angle in angles
            ]
            add_polyline(vertices, feature_id, closed=True)

    if not points:
        raise ValueError("No LINE, POLYLINE, or CIRCLE entities were found in this DXF.")

    part = pv.PolyData(np.asarray(points), lines=np.asarray(line_cells, dtype=np.int64))
    part.cell_data["feature_id"] = np.asarray(feature_ids, dtype=np.int64)
    return part, [f"{kind} × {count}" for kind, count in sorted(skipped.items())]


def _load_open_cascade(path: Path) -> tuple[pv.PolyData, object, pv.PolyData]:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GeomAbs import GeomAbs_Line
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.IGESControl import IGESControl_Reader
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as error:
        raise ValueError(
            "STEP/IGES support needs the optional Open CASCADE bindings. "
            "Install the project's declared dependencies and retry."
        ) from error

    reader = IGESControl_Reader() if path.suffix.lower() in {".igs", ".iges"} else STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"Open CASCADE could not read {path.name}.")
    if reader.TransferRoots() == 0:
        raise ValueError(f"No geometry could be transferred from {path.name}.")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"{path.name} contains no usable solid or surface geometry.")

    BRepMesh_IncrementalMesh(shape, 0.1, False, 0.35, True)
    points: list[tuple[float, float, float]] = []
    faces: list[int] = []
    feature_ids: list[int] = []
    edge_points: list[tuple[float, float, float]] = []
    edge_lines: list[int] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    feature_id = 0

    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            start = len(points)
            transform = location.Transformation()
            for node_index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_index).Transformed(transform)
                points.append((point.X(), point.Y(), point.Z()))
            reverse = face.Orientation() == TopAbs_REVERSED
            for triangle_index in range(1, triangulation.NbTriangles() + 1):
                first, second, third = triangulation.Triangle(triangle_index).Get()
                if reverse:
                    second, third = third, second
                faces.extend((3, start + first - 1, start + second - 1, start + third - 1))
                feature_ids.append(feature_id)
        feature_id += 1
        explorer.Next()

    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge(edge_explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        first_parameter = curve.FirstParameter()
        last_parameter = curve.LastParameter()
        if np.isfinite(first_parameter) and np.isfinite(last_parameter) and last_parameter > first_parameter:
            sample_count = (
                2
                if curve.GetType() == GeomAbs_Line
                else max(12, min(96, int(np.ceil((last_parameter - first_parameter) * 8)) + 1))
            )
            point_start = len(edge_points)
            for parameter in np.linspace(first_parameter, last_parameter, sample_count):
                point = curve.Value(float(parameter))
                edge_points.append((point.X(), point.Y(), point.Z()))
            edge_lines.extend((sample_count, *range(point_start, point_start + sample_count)))
        edge_explorer.Next()

    if not faces:
        raise ValueError(f"No triangulatable faces were found in {path.name}.")
    part = pv.PolyData(np.asarray(points), np.asarray(faces, dtype=np.int64))
    part.cell_data["feature_id"] = np.asarray(feature_ids, dtype=np.int64)
    edge_mesh = pv.PolyData(np.asarray(edge_points), lines=np.asarray(edge_lines, dtype=np.int64))
    return part, shape, edge_mesh


def scale_solid_shape(shape: object, factor: float) -> object:
    if factor == 1.0:
        return shape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Pnt, gp_Trsf

    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0.0, 0.0, 0.0), factor)
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _contains_solid(shape: object) -> bool:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    return TopExp_Explorer(shape, TopAbs_SOLID).More()


def _mesh_to_solid(mesh: trimesh.Trimesh) -> tuple[object | None, str]:
    maximum_faces = 100_000
    if not mesh.is_watertight:
        return None, "Mesh is open or non-manifold; kept as a mesh."
    if not mesh.is_winding_consistent:
        return None, "Mesh face winding is inconsistent; kept as a mesh."
    if len(mesh.faces) > maximum_faces:
        return None, (
            f"Mesh has {len(mesh.faces):,} faces; automatic solid conversion is limited to "
            f"{maximum_faces:,} faces to avoid excessive import time and memory."
        )

    try:
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_Sewing,
        )
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRep import BRep_Builder
        from OCP.TopAbs import TopAbs_SHELL
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound
        from OCP.gp import gp_Pnt
    except ImportError as error:
        return None, f"Open CASCADE unavailable for mesh solidification: {error}"

    tolerance = max(float(np.linalg.norm(mesh.extents)) * 1e-8, 1e-7)
    sewing = BRepBuilderAPI_Sewing(tolerance)
    try:
        for triangle in mesh.faces:
            polygon = BRepBuilderAPI_MakePolygon()
            for vertex_index in triangle:
                point = mesh.vertices[int(vertex_index)]
                polygon.Add(gp_Pnt(float(point[0]), float(point[1]), float(point[2])))
            polygon.Close()
            face_builder = BRepBuilderAPI_MakeFace(polygon.Wire())
            if not face_builder.IsDone():
                return None, "A mesh facet could not be represented as a planar B-rep face."
            sewing.Add(face_builder.Face())

        sewing.Perform()
        sewed_shape = sewing.SewedShape()
        shell_explorer = TopExp_Explorer(sewed_shape, TopAbs_SHELL)
        if not shell_explorer.More():
            return None, "Mesh facets did not sew into a closed shell; kept as a mesh."
        if sewing.NbFreeEdges() or sewing.NbMultipleEdges():
            return None, "Mesh did not sew into a manifold shell; kept as a mesh."
        solids = []
        while shell_explorer.More():
            shell = TopoDS.Shell(shell_explorer.Current())
            solid = BRepBuilderAPI_MakeSolid(shell).Solid()
            if not BRepCheck_Analyzer(solid).IsValid():
                return None, "Open CASCADE rejected a sewn mesh solid as invalid; kept as a mesh."
            solids.append(solid)
            shell_explorer.Next()
        if len(solids) == 1:
            result = solids[0]
        else:
            result = TopoDS_Compound()
            compound_builder = BRep_Builder()
            compound_builder.MakeCompound(result)
            for solid in solids:
                compound_builder.Add(result, solid)
        return result, f"Converted watertight STL mesh to {len(solids)} faceted solid(s)."
    except Exception as error:
        return None, f"Mesh solidification failed ({error}); kept as a mesh."


def _load_sldprt(path: Path, millimeters_per_unit: float) -> ImportedGeometry:
    if sys.platform != "win32":
        raise ValueError(
            "SolidWorks .sldprt is a proprietary format and cannot be read natively on Linux. "
            "Export it to STEP from SolidWorks (or another licensed converter), then import the STEP file."
        )
    try:
        import win32com.client
    except ImportError as error:
        raise ValueError(
            "Importing .sldprt directly requires Windows, an installed licensed SolidWorks "
            "application, and the optional pywin32 package. Export to STEP as an alternative."
        ) from error

    application = win32com.client.Dispatch("SldWorks.Application")
    document = None
    try:
        opened = application.OpenDoc6(str(path), 1, 1, "", 0, 0)
        document = opened[0] if isinstance(opened, tuple) else opened
        if document is None:
            raise ValueError("SolidWorks could not open this part. Check the file and license.")
        with tempfile.TemporaryDirectory(prefix="basicam-sldprt-") as temporary_directory:
            step_path = Path(temporary_directory) / f"{path.stem}.step"
            saved = document.Extension.SaveAs(str(step_path), 0, 0, None, 0, 0)
            succeeded = saved[0] if isinstance(saved, tuple) else saved
            if not succeeded or not step_path.exists():
                raise ValueError("SolidWorks opened the part but could not export it to STEP.")
            part, brep_shape, edge_mesh = _load_open_cascade(step_path)
            part.scale((millimeters_per_unit,) * 3, inplace=True)
            edge_mesh.scale((millimeters_per_unit,) * 3, inplace=True)
            brep_shape = scale_solid_shape(brep_shape, millimeters_per_unit)
            solid_shape = brep_shape if _contains_solid(brep_shape) else None
            return ImportedGeometry(part, brep_shape, solid_shape, edge_mesh, native_brep=True)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"SolidWorks conversion failed: {error}") from error
    finally:
        if document is not None:
            try:
                application.CloseDoc(document.GetTitle())
            except Exception:
                pass
