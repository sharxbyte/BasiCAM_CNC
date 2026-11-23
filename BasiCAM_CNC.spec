from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files


project_root = Path(SPECPATH)
diagnostic_mode = os.environ.get("BASICAM_DIAGNOSTIC") == "1"
datas = [
    (str(project_root / "ThisIsNotMalware.txt"), "."),
    (str(project_root / "BCCIcon.ico"), "."),  # window/taskbar icon at run time
]
binaries = []
hiddenimports = [
    "basicam.app",
    "basicam.importers",
    "basicam.face_selection",
    "basicam.materials",
    "basicam.stock",
    "basicam.part_materials_dialog",
    "basicam.sketch_interaction",
    "basicam.sketch_dimensions",
    "basicam.sketch_tools",
    "basicam.contours",
    "basicam.toolpath_sketch",
    "basicam.machine_types",
    "basicam.kinematics",
    "basicam.tool_changer",
    "basicam.auxiliaries",
    "basicam.usage_tracking",
    "basicam.process_params",
    "basicam.cut_features",
    "basicam.rotary_devices",
    "basicam.job_type",
    "basicam.nesting",
    "basicam.wrapping",
    "basicam.machine_bodies",
    "basicam.turning_setup",
    "basicam.turning",
    "basicam.lathe_tools",
    "basicam.bar_feed",
    "basicam.setups",
    "basicam.tooling_ui",
    "basicam.gcode",
    "basicam.feeds_speeds",
    "basicam.stock_analysis",
    "basicam.machine",
    "basicam.models",
    "basicam.operations",
    "basicam.readiness",
    "basicam.sketches",
    "basicam.sketch_workflow",
    "basicam.cam_features",
    "basicam.tooling",
    "basicam.units",
    "basicam.project_bundle",
    "basicam.cam_dynamics_post",
    "basicam.cam_library_validation",
    "basicam.cam_fixtures_speeds_export",
    "basicam.cam_envelope_stock_removal",
    "basicam.safety",
    "basicam.terminal",
    "basicam.operation_stack",
    "basicam.ui_tips",
    "basicam.theme",
    "basicam.display_modes",
    "basicam.machine_profiles",
    "basicam.motion_guard",
    "basicam.sketch_solver",
    "basicam.sketch_dialogs",
    "basicam.transforms",
    "basicam.assignments",
    "basicam.surfacing",
    "basicam.cycle_time",
    "basicam.keyboard_control",
    "basicam.gcode_viewer",
    "basicam.gcode_dialogs",
    "basicam.exporters",
    "ezdxf.addons.odafc",
    "shapely",
    "shapely.geometry",
    "shapely.ops",
    "python_solvespace",
    "python_solvespace.slvs",
    "vtkmodules.qt",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "OCP.STEPControl",
    "OCP.IGESControl",
    "OCP.BRep",
    "OCP.BRepBuilderAPI",
    "OCP.BRepAdaptor",
    "OCP.BRepCheck",
    "OCP.BRepMesh",
    "OCP.GeomAbs",
    "OCP.IFSelect",
    "OCP.IntCurvesFace",
    "OCP.TopAbs",
    "OCP.TopExp",
    "OCP.TopLoc",
    "OCP.TopoDS",
    "OCP.gp",
    "serial",
    "serial.tools.list_ports",
    "vtkmodules.vtkRenderingCore",
    "vtkmodules.vtkInteractionStyle",
    "vtkmodules.vtkInteractionWidgets",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "win32com.client",
    "pythoncom",
    "pywintypes",
]
datas += collect_data_files("pyvista")

for package in (
    "pyvistaqt",
    "trimesh",
    "ezdxf",
    "shapely",  # GEOS DLLs
    "python_solvespace",  # compiled solver extension
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

analysis = Analysis(
    [str(project_root / "basicam" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tests", "tools", "helpers"],  # development-only folders are never bundled
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="BasiCAM CNC Diagnostic" if diagnostic_mode else "BasiCAM CNC",
    debug=diagnostic_mode,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=diagnostic_mode,
    disable_windowed_traceback=False,
    icon=str(project_root / "BCCIcon.ico"),
)