from __future__ import annotations

import copy
import math
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
try:
    import pyvista_cad  # registers .cad accessor, STEP/BREP readers, and CAD view helpers
    HAS_PYVISTA_CAD = True
except ImportError:
    pyvista_cad = None
    HAS_PYVISTA_CAD = False
from PySide6.QtCore import QEvent, QPoint, QRect, QSettings, QUrl, QTimer, Qt
from PySide6.QtGui import QAction, QFont, QKeySequence, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QKeySequenceEdit,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QTextEdit,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.vtkRenderingCore import vtkCellPicker
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser

from basicam.importers import (
    SUPPORTED_SUFFIXES,
    load_part,
    load_part_geometry,
    scale_solid_shape,
)
from basicam.machine import (
    GrblCapabilities,
    GrblController,
    GrblStatus,
    available_serial_ports,
    describe_grbl_message,
    parse_grbl_capabilities,
    parse_grbl_status,
)
from basicam.importers import set_dwg_converter
from basicam.exporters import export_parts
from basicam.models import PartModel, PartRole
from basicam.operations import Operation, operation_warnings
from basicam.readiness import gcode_readiness
from basicam.sketches import SketchEntity, construction_line
from basicam.tooling_ui import ToolLibraryDialog
from basicam.units import LengthUnit, UnitSettings
from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
from basicam.gcode import GCodeSettings, generate_gcode
from basicam.feeds_speeds import (
    multiplier_to_slider,
    recommend_cutting_params,
    slider_to_multiplier,
)
from basicam.setups import SETUP_PALETTE, MachiningSetup
from basicam.stock_analysis import RoughingAnalysis
from basicam.tooling import ToolDefinition, create_tool_mesh, default_tool_inventory
from basicam.sketch_workflow import (
    ParametricSketchFeature,
    SketchDimension,
    SketchWorkflowManager,
    build_sketch_plane_from_surface,
    build_standard_sketch_plane,
    export_sketches_to_dxf,
    flood_select_manifold_features,
    sample_catmull_rom_spline,
    sample_center_arc,
    sample_circle_points,
    sample_circle_three_points,
    sample_rectangle_points,
    sample_rectangle_three_points,
    sample_three_point_arc,
)
from basicam.cam_features import validate_gcode_program
from basicam.project_bundle import (
    brep_bytes_to_shape,
    shape_to_brep_bytes,
    CrashResilientSessionManager,
    WorkspaceHistoryManager,
    WorkspaceSnapshot,
    _deserialize_sketch_feature,
    _serialize_sketch_feature,
    load_bcam_bundle,
    save_bcam_bundle,
)
from basicam.cam_dynamics_post import (
    DynamicFeatureRecognizer,
    PostProcessorCompiler,
    PostProcessorSpec,
    ToolpathDynamicsConfig,
    ToolpathDynamicsEngine,
    ToolpathSimulationPlayer,
    build_setup_sheet_summary,
    write_setup_sheet_files,
)
from basicam.cam_library_validation import (
    DEFAULT_TOOL_LIBRARY,
    GCodeValidationIssue,
    GCodeValidator,
    ToolLibraryEntry,
)
from basicam.cam_fixtures_speeds_export import (
    CollisionAndFixtureManager,
    FeedsAndSpeedsCalculator,
    WorkholdingFixture,
    export_cam_package,
)
from basicam.cam_envelope_stock_removal import (
    MachineEnvelopeChecker,
    MachineKinematicsProfile,
    StockRemovalEstimator,
)
from basicam.safety import AdvancedSettingsDialog, SafetyGates, run_first_run_dialog
from basicam.terminal import CommandValidator, TerminalCore, TerminalDock, TerminalView
from basicam.operation_stack import OperationDropArea, OperationRow, move_operation, renumber
from basicam.ui_tips import apply_tooltips
from basicam.machine_profiles import (
    BedFitDialog,
    MachineProfile,
    MachineStore,
    PreflightDialog,
    SoftLimitWizard,
    compatible_machines,
    fit_report,
    generation_fit_rule,
)
from basicam.motion_guard import MotionGuard
from basicam.sketch_solver import (
    RELATION_KINDS,
    LastGoodStateStore,
    Relation,
    broken_ids,
    closest_reference_points,
    delete_entities,
    dependents_of,
    sketch_groups,
    solve_all,
)
from basicam.sketch_dialogs import CascadeDeleteDialog, CascadeStepDialog, DeleteBadSketchDialog
from basicam.assignments import (
    AssignmentLedger,
    RetainAssignmentsDialog,
    SketchStatus,
    can_assign_tool,
    check_assignments_walk,
    entity_operations,
    generation_gate,
    mark_future,
    sketch_status,
    untooled,
)
from basicam.operations import default_entry_params
from basicam.keyboard_control import KeyboardMachineControl, MoveRequest
from basicam.gcode_viewer import AdvancedEditorWindow, build_resume_preamble, parse_program, section_setup_sheet
from basicam.gcode_dialogs import ResumeDialog, StartDisclaimerDialog
from basicam.cycle_time import (
    CycleEstimator,
    MultiplierState,
    clamp_multiplier,
    format_duration,
    machine_max_feed,
    override_bytes,
    parse_grbl_settings,
    scale_feed_line,
    split_multiplier,
)
from basicam.surfacing import SurfaceParams, plan_surface, suggest_finer_tool, trace_surface_dialog
from basicam.transforms import (
    Frame,
    TransformResult,
    align_faces_matrix,
    apply_to_point,
    apply_to_points,
    frame_from_origin_and_edge,
    matrix_rotate,
    matrix_translate,
    part_center,
    transform_mesh,
    transform_part,
)
from basicam.theme import COLORS, TOKENS, activate as activate_theme, color as theme_color, is_red
from basicam.display_modes import (
    DISPLAY_MODES,
    LIGHTING_DEFAULTS,
    dash_lines,
    lighting_params,
    mode_flags,
    normalize_mode,
)

# Kill-button behaviour (see BasiCAMWindow._kill_machine)
KILL_Z_RETRACT_MM = 20.0        # retract distance, reduced if the Z stop is nearer
KILL_Z_FEED_MM_MIN = 600.0      # retract feed rate (controller limits jogs to 5000 mm/min)
KILL_Z_MACHINE_TOP_MM = 0.0     # machine-coordinate Z at the top stop (GRBL default: 0, travel negative)
KILL_Z_STOP_MARGIN_MM = 1.0     # stay this far below the stop
KILL_UNLOCKABLE_ALARMS = {3, 6}  # reset-caused alarms only: 3 = reset while in motion, 6 = homing reset
KILL_POLL_MS = 100
GRID_LINE_WIDTH = 1.0           # origin axes use the same line width as the grid
KILL_HOLD_TIMEOUT_S = 2.0
KILL_RESET_SETTLE_S = 0.6
KILL_RESET_TIMEOUT_S = 5.0
KILL_UNLOCK_TIMEOUT_S = 3.0

# ---- length parsing for dimension entry -------------------------------------------------
_UNIT_MM = {
    "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "millimetre": 1.0, "millimetres": 1.0,
    "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0, "centimetre": 10.0, "centimetres": 10.0,
    "m": 1000.0, "meter": 1000.0, "meters": 1000.0, "metre": 1000.0, "metres": 1000.0,
    "in": 25.4, "inch": 25.4, "inches": 25.4, '"': 25.4, "\u2033": 25.4,
    "ft": 304.8, "foot": 304.8, "feet": 304.8, "'": 304.8, "\u2032": 304.8,
}
_QUANTITY = re.compile(
    r"\s*(?P<num>\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d+\.?\d*|\.\d+)\s*"
    r"(?P<unit>mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|inches|inch|in|feet|foot|ft|[\"'\u2033\u2032])?",
    re.IGNORECASE,
)


def _number(token: str) -> float:
    token = token.strip()
    if "/" in token:
        parts = token.split()
        whole = float(parts[0]) if len(parts) == 2 else 0.0
        numerator, denominator = (float(v) for v in parts[-1].replace(" ", "").split("/"))
        return whole + numerator / denominator
    return float(token)


def parse_length_to_mm(text: str, default_unit_mm: float = 1.0) -> float | None:
    """Parse '12.5 mm', '1/2 in', '1 1/4"', '3 ft', '5\\' 3 1/2"' into millimetres.

    A bare number uses ``default_unit_mm`` (the current display unit). Returns None if invalid.
    """
    text = text.strip()
    if not text:
        return None
    sign = 1.0
    if text[0] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    position, total, matched, last_unit = 0, 0.0, 0, None
    while position < len(text):
        found = _QUANTITY.match(text, position)
        if found is None:
            return None
        try:
            value = _number(found.group("num"))
        except (ValueError, ZeroDivisionError):
            return None
        unit = found.group("unit")
        if unit is None:
            factor = 25.4 if last_unit in ("'", "\u2032", "ft", "foot", "feet") and matched else default_unit_mm
        else:
            factor = _UNIT_MM[unit.lower()]
        last_unit = unit.lower() if unit else last_unit
        total += value * factor
        matched += 1
        position = found.end()
    return sign * total if matched else None


def describe_length_text(text: str, display_unit: str = "mm") -> tuple[str, int]:
    """Unit the user typed (display unit if bare) and the significant figures of the number."""
    found = re.search(r"[a-zA-Z\"'′″]+", text)
    unit = found.group(0).lower() if found else display_unit
    unit = {'"': "in", "″": "in", "'": "ft", "′": "ft"}.get(unit, unit)
    digits = re.sub(r"[^0-9.]", "", text.split("/")[0] if "/" in text and " " not in text.strip() else text)
    core = digits.replace(".", "").lstrip("0") or "0"
    return unit, max(1, min(len(core), 10))


def format_length(value_mm: float, display_unit: str = "mm", sig_figs: int = 4) -> str:
    """Show a length in the display unit, keeping significant figures."""
    if display_unit in ("in", "inch", "inches"):
        value, suffix = value_mm / 25.4, "in"
    else:
        value, suffix = value_mm, "mm"
    return f"{float(f'{value:.{max(1, sig_figs)}g}'):g} {suffix}"


class WorkspaceBundleMixin:
    """Drop-in integration helpers for .BCAM bundles, crash recovery, autosave, and undo/redo."""

    def init_bundle_and_history(self) -> None:
        self.session_manager = CrashResilientSessionManager()
        settings = getattr(self, "settings", None)
        try:
            limit = int(settings.value("history/limit", 50)) if settings is not None else 50
        except (TypeError, ValueError):
            limit = 50
        self.HISTORY_LIMIT = max(1, limit)
        self.history_manager = WorkspaceHistoryManager(max_steps=self.HISTORY_LIMIT)
        self.current_bcam_path: Path | None = None
        self.session_manager.mark_session_started()
        self._bind_bundle_actions()
        self._check_crash_recovery()

    def _bind_bundle_actions(self) -> None:
        self.save_bcam_action = QAction("Save Project (.bcam)", self)
        self.save_bcam_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_bcam_action.triggered.connect(self.save_project_bcam)

        self.save_bcam_as_action = QAction("Save Project As (.bcam)…", self)
        self.save_bcam_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_bcam_as_action.triggered.connect(self.save_project_bcam_as)

        self.open_bcam_action = QAction("Open Project (.bcam)…", self)
        self.open_bcam_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_bcam_action.triggered.connect(self.open_project_bcam)

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo_workspace)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo_workspace)

        self._refresh_history_action_states()

    def _refresh_history_action_states(self) -> None:
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(self.history_manager.can_undo())
        if hasattr(self, "redo_action"):
            self.redo_action.setEnabled(self.history_manager.can_redo())

    def _part_meta(self, part) -> dict:
        solid_is_brep = part.solid_shape is not None and part.solid_shape is part.brep_shape
        return {
            "name": part.name,
            "role": part.role.name,
            "source_path": str(part.source_path) if getattr(part, "source_path", None) else None,
            "imported": bool(getattr(part, "imported", False)),
            "note": getattr(part, "note", "") or "",
            "feature_notes": {str(k): v for k, v in getattr(part, "feature_notes", {}).items()},
            "native_brep": bool(getattr(part, "native_brep", False)),
            "solid_conversion_message": getattr(part, "solid_conversion_message", "") or "",
            "solid_is_brep": solid_is_brep,
        }

    def _collect_extras(self) -> dict:
        """Project data added by later roadmap blocks; each block adds its own key here."""
        extras = dict(getattr(self, "_loaded_extras", {}) or {})
        extras["sketch_notes"] = {k: v for k, v in getattr(self, "sketch_notes", {}).items() if v}
        machine = getattr(self, "current_machine", None)
        extras["machine"] = {"id": machine.machine_id, "profile": machine.to_dict()} if machine else None
        extras["sketch_state"] = self.sketch_workflow.state_dict()
        extras["sketch_relations"] = [r.to_dict() for r in getattr(self.sketch_workflow, "relations", [])]
        if hasattr(self, "assignment_ledger"):
            extras["tool_assignments"] = self.assignment_ledger.to_dict()
        if getattr(self, "gcode_progress", None):
            extras["gcode_progress"] = dict(self.gcode_progress)
        store = getattr(self.sketch_workflow, "last_good", None)
        extras["sketch_last_good"] = {
            owner: {
                "features": [_serialize_sketch_feature(f) for f in state["features"]],
                "relations": [r.to_dict() for r in state["relations"]],
            }
            for owner, state in (store.states.items() if store is not None else [])
        }
        return extras

    def _apply_extras(self, extras: dict) -> None:
        """Restore project data stored under `extras`; unknown keys are kept for round trips."""
        self._loaded_extras = dict(extras or {})
        self.sketch_notes = {str(k): str(v) for k, v in dict(self._loaded_extras.get("sketch_notes", {})).items()}
        if hasattr(self, "notes_list"):
            self._refresh_notes_list()
        if hasattr(self, "sketch_workflow"):
            manager = self.sketch_workflow
            manager.load_state_dict(dict(self._loaded_extras.get("sketch_state", {}) or {}))
            manager.relations = [Relation.from_dict(r) for r in self._loaded_extras.get("sketch_relations", []) or []]
            store = LastGoodStateStore()
            for owner, state in dict(self._loaded_extras.get("sketch_last_good", {}) or {}).items():
                store.states[owner] = {
                    "features": [_deserialize_sketch_feature(f) for f in state.get("features", [])],
                    "relations": [Relation.from_dict(r) for r in state.get("relations", [])],
                }
            manager.last_good = store
            self.assignment_ledger = AssignmentLedger.from_dict(self._loaded_extras.get("tool_assignments"))
            progress = self._loaded_extras.get("gcode_progress")
            self.gcode_progress = dict(progress) if isinstance(progress, dict) else None
            if self.gcode_progress and self.gcode_progress.get("acked", 0) < self.gcode_progress.get("lines", 0) + self.gcode_progress.get("start_line", 1) - 1:
                self.terminal_core.log(
                    "warning",
                    f"The last program stopped after line {self.gcode_progress.get('acked', 0)}; "
                    "use Advanced G-code > Resume code here to continue.",
                )
            if hasattr(self, "sketch_tree"):
                self._refresh_sketch_tree()
        saved = self._loaded_extras.get("machine")
        if isinstance(saved, dict) and saved.get("id") and hasattr(self, "machine_store"):
            machine = self.machine_store.get(saved["id"])
            if machine is None and saved.get("profile"):
                machine = MachineProfile.from_dict(saved["profile"])  # deleted from inventory: use the project copy
                self.terminal_core.log("warning", f"Machine '{machine.name}' is not in the inventory; using the project copy.")
            if machine is not None:
                self._save_machine_state()
                self.current_machine = machine
                self.machine_store.default_id = machine.machine_id if self.machine_store.get(machine.machine_id) else self.machine_store.default_id
                self._load_machine_state()
                self._sync_machine_profile()
                self._refresh_machine_combo()

    def capture_workspace_snapshot(self) -> WorkspaceSnapshot:
        """Everything needed to rebuild the workspace, including exact B-rep geometry."""
        stock_index = self.parts.index(self.stock_part) if self.stock_part in self.parts else None
        parts = list(self.parts)
        return WorkspaceSnapshot(
            job_name=self.project_name_label.text(),
            job_notes=getattr(self, "job_notes", ""),
            stock_material=self.stock_material.currentText(),
            stock_dimensions={axis: float(spin.value()) for axis, spin in self.stock_dimensions.items()},
            stock_note=self.stock_note,
            stock_imported=bool(self.stock_imported),
            stock_part_index=stock_index,
            active_setup_index=int(self.active_setup_index),
            setups=list(self.setups),
            sketch_features=list(self.sketch_workflow.features),
            active_sketch_id=self.active_sketch_id,
            parts_meta=[self._part_meta(part) for part in parts],
            part_meshes=[part.mesh if part.mesh is not None else pv.PolyData() for part in parts],
            part_edge_meshes=[getattr(part, "edge_mesh", None) for part in parts],
            part_breps=[shape_to_brep_bytes(part.brep_shape) for part in parts],
            part_solids=[
                None
                if part.solid_shape is None or part.solid_shape is part.brep_shape
                else shape_to_brep_bytes(part.solid_shape)
                for part in parts
            ],
            stock_mesh=self.stock_mesh,
            form_draft={
                "display_unit": self.units.display.value,
                "finish_quality": self.finish_quality.currentText(),
                "tool": self.tool_name.currentText(),
            },
            extras=self._collect_extras(),
        )

    def apply_workspace_snapshot(self, snapshot: WorkspaceSnapshot) -> None:
        """Rebuild parts (with B-rep), stock, setups, sketches, and form state from a snapshot."""
        parts = []
        for index, meta in enumerate(snapshot.parts_meta):
            mesh = snapshot.part_meshes[index] if index < len(snapshot.part_meshes) else pv.PolyData()
            edge_mesh = snapshot.part_edge_meshes[index] if index < len(snapshot.part_edge_meshes) else None
            brep = brep_bytes_to_shape(snapshot.part_breps[index]) if index < len(snapshot.part_breps) else None
            solid_blob = snapshot.part_solids[index] if index < len(snapshot.part_solids) else None
            solid = brep_bytes_to_shape(solid_blob) if solid_blob else (brep if meta.get("solid_is_brep") else None)
            source = meta.get("source_path")
            try:
                role = PartRole[str(meta.get("role", "TARGET"))]
            except KeyError:
                role = PartRole.TARGET
            part = PartModel(
                str(meta.get("name", f"Part {index + 1}")),
                mesh,
                Path(source) if source else None,
                imported=bool(meta.get("imported", False)),
                brep_shape=brep,
                solid_shape=solid,
                edge_mesh=edge_mesh,
                solid_conversion_message=str(meta.get("solid_conversion_message", "")),
                native_brep=bool(meta.get("native_brep", False)),
                role=role,
            )
            part.note = str(meta.get("note", ""))
            part.feature_notes = {int(k): v for k, v in dict(meta.get("feature_notes", {})).items()}
            parts.append(part)

        self.parts = parts
        index = snapshot.stock_part_index
        self.stock_part = parts[index] if index is not None and 0 <= index < len(parts) else None
        self.stock_mesh = self.stock_part.mesh if self.stock_part is not None else None
        self.stock_imported = bool(snapshot.stock_imported)
        self.stock_note = snapshot.stock_note
        self.job_notes = snapshot.job_notes
        self.project_name_label.setText(snapshot.job_name)
        self.stock_material.setCurrentText(snapshot.stock_material)
        for axis, spin in self.stock_dimensions.items():
            if axis in snapshot.stock_dimensions:
                spin.setValue(float(snapshot.stock_dimensions[axis]))
        self.stock_status.setText(
            f"Stock · {self.stock_part.name}" if self.stock_part is not None else "No stock geometry"
        )
        self.setups = list(snapshot.setups) or self.setups
        self.active_setup_index = min(max(snapshot.active_setup_index, 0), max(len(self.setups) - 1, 0))
        self.sketch_workflow.features = list(snapshot.sketch_features)
        self.active_sketch_id = snapshot.active_sketch_id
        self.selected_cells.clear()
        self.selected_features.clear()
        self.selected_feature = None
        self.pending_sketch_points.clear()

        draft = snapshot.form_draft or {}
        wants_inches = draft.get("display_unit") == LengthUnit.INCHES.value
        if hasattr(self, "display_button") and self.display_button.isChecked() != wants_inches:
            self.display_button.setChecked(wants_inches)
        if draft.get("finish_quality"):
            self.finish_quality.setCurrentText(str(draft["finish_quality"]))
        position = self.tool_name.findText(str(draft.get("tool", "")))
        if position >= 0:
            self.tool_name.setCurrentIndex(position)

        has_ops = self._has_any_operations()
        self.generate_action.setEnabled(has_ops)
        self.export_button.setEnabled(has_ops)
        self._refresh_parts()
        self._refresh_note_targets()
        self._refresh_operation_stack()
        self._refresh_sketch_tree()
        self._apply_extras(getattr(snapshot, "extras", {}) or {})
        self._refresh_scene()
        self._undo_stack, self._redo_stack = [], []
        self._refresh_history_action_states()

    def _project_title(self) -> str:
        name = self.project_name_label.text() if hasattr(self, "project_name_label") else "Untitled job"
        star = " *" if getattr(self, "_project_dirty", False) else ""
        return f"{name}{star} · BasiCAM CNC"

    def _update_window_title(self) -> None:
        self.setWindowTitle(self._project_title())

    def _mark_dirty(self) -> None:
        if not getattr(self, "_project_dirty", False):
            self._project_dirty = True
            self._update_window_title()

    def _write_project(self, path) -> bool:
        if self.sketch_workflow.has_errors():
            message = "Save refused: a sketch is in the error state. Solve it or press Esc to revert first."
            self.terminal_core.log("error", message)
            self.statusBar().showMessage(message, 6000)
            return False
        try:
            saved = save_bcam_bundle(self.capture_workspace_snapshot(), path)
        except Exception as error:
            self._critical_box(self, "Save project", f"Could not save the project:\n{error}")
            return False
        self.current_bcam_path = Path(saved)
        self._project_dirty = False
        self._update_window_title()
        self.statusBar().showMessage(f"Saved {Path(saved).name}", 4000)
        return True

    def save_project_bcam(self) -> None:
        if self.current_bcam_path is None:
            self.save_project_bcam_as()
            return
        self._write_project(self.current_bcam_path)

    def save_project_bcam_as(self) -> None:
        safe_name = re.sub(r'[\\/:*?"<>|]+', "_", self.project_name_label.text().strip()) or "Untitled job"
        start = self.current_bcam_path or (Path.home() / f"{safe_name}.bcam")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project Bundle", str(start), "BasiCAM Bundle (*.bcam);;All Files (*)"
        )
        if not file_path:
            return
        self._write_project(file_path)

    def open_project_bcam(self) -> None:
        start_dir = str(self.current_bcam_path.parent if self.current_bcam_path else Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project Bundle", start_dir, "BasiCAM Bundle (*.bcam);;All Files (*)"
        )
        if not file_path:
            return
        try:
            snapshot = load_bcam_bundle(file_path)
            self.apply_workspace_snapshot(snapshot)
        except Exception as error:
            self._critical_box(self, "Open project", f"Could not open the project:\n{error}")
            return
        self.current_bcam_path = Path(file_path)
        self._project_dirty = False
        self._update_window_title()
        self.statusBar().showMessage(f"Opened {Path(file_path).name}", 4000)

    def _autosave_if_due(self) -> None:
        try:
            if self.session_manager.should_autosave():
                self.session_manager.autosave_atomic(self.capture_workspace_snapshot())
        except Exception:
            pass  # autosave must never interrupt editing

    def _check_crash_recovery(self) -> None:
        if not self.session_manager.had_unclean_exit():
            return
        reply = QMessageBox.question(
            self,
            "Recover Unsaved Session?",
            "The previous session did not close cleanly. Restore the last autosaved state?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            recovered = self.session_manager.recover_last_good_snapshot()
            if recovered is not None:
                self.apply_workspace_snapshot(recovered)

    def handle_clean_shutdown(self) -> None:
        """Call inside your main window's closeEvent after any final save/prompt."""
        self.session_manager.mark_clean_exit()
class CamWorkspaceTabsMixin:
    """Tabbed CAM dock exposing Tool Library/Speeds, Fixtures/Envelope, Simulation/Post, and Setup Sheet."""

    def init_cam_tabs_dock(self) -> None:
        self.cam_tabs = QTabWidget(self)
        self.cam_tabs.addTab(self._build_tools_speeds_tab(), "Tools & Speeds")
        self.cam_tabs.addTab(self._build_fixtures_envelope_tab(), "Fixtures & Envelope")
        self.cam_tabs.addTab(self._build_sim_post_tab(), "Sim & Post")
        self.cam_tabs.addTab(self._build_setup_export_tab(), "Setup & Export")

        self.cam_tabs_dock = QDockWidget("CAM Workflows", self)
        self.cam_tabs_dock.setObjectName("CamWorkflowsDock")
        self.cam_tabs_dock.setWidget(self.cam_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.cam_tabs_dock)

    def _build_tools_speeds_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tool_library_combo = QComboBox()
        for entry in getattr(self, "tool_library", []):
            self.tool_library_combo.addItem(f"{entry.tool_id} · {entry.name} (Ø{entry.diameter_mm:g}mm)", entry.tool_id)
        layout.addWidget(QLabel("Tool Library"))
        layout.addWidget(self.tool_library_combo)

        apply_tool_btn = QPushButton("Load Selected Tool")
        apply_tool_btn.clicked.connect(self._on_apply_library_tool_clicked)
        layout.addWidget(apply_tool_btn)

        flutes_row = QHBoxLayout()
        flutes_row.addWidget(QLabel("Flutes:"))
        self.flutes_spin = QSpinBox()
        self.flutes_spin.setRange(1, 12)
        self.flutes_spin.setValue(2)
        flutes_row.addWidget(self.flutes_spin)
        layout.addLayout(flutes_row)

        calc_btn = QPushButton("Calculate Feeds && Speeds")
        calc_btn.clicked.connect(self._on_calc_feeds_speeds_clicked)
        layout.addWidget(calc_btn)

        self.speeds_readout = QTextEdit()
        self.speeds_readout.setReadOnly(True)
        self.speeds_readout.setMaximumHeight(110)
        layout.addWidget(self.speeds_readout)
        layout.addStretch()
        return tab

    def _build_fixtures_envelope_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.fixture_name_combo = QComboBox()
        self.fixture_name_combo.addItems(["Vise", "Strap Clamp", "Fixture Plate", "Soft Jaws", "Sub-plate"])
        layout.addWidget(QLabel("Workholding Fixture Kind"))
        layout.addWidget(self.fixture_name_combo)

        coords_row = QHBoxLayout()
        self.fix_x = QDoubleSpinBox()
        self.fix_y = QDoubleSpinBox()
        self.fix_z = QDoubleSpinBox()
        for sp in (self.fix_x, self.fix_y, self.fix_z):
            sp.setRange(-2000.0, 2000.0)
            sp.setDecimals(2)
        coords_row.addWidget(QLabel("X:"))
        coords_row.addWidget(self.fix_x)
        coords_row.addWidget(QLabel("Y:"))
        coords_row.addWidget(self.fix_y)
        coords_row.addWidget(QLabel("Z:"))
        coords_row.addWidget(self.fix_z)
        layout.addLayout(coords_row)
        self.fix_keep_out = QCheckBox("Keep out")
        self.fix_keep_out.setChecked(True)
        self.fix_keep_out.setToolTip("The tool envelope may never enter this fixture; no setting can override it.")
        layout.addWidget(self.fix_keep_out)

        add_fix_btn = QPushButton("Add Fixture")
        add_fix_btn.clicked.connect(self._on_add_fixture_clicked)
        layout.addWidget(add_fix_btn)

        self.fixtures_list = QListWidget()
        self.fixtures_list.setMaximumHeight(100)
        layout.addWidget(self.fixtures_list)

        check_btn = QPushButton("Check Collisions && Envelope")
        check_btn.clicked.connect(self._on_check_collisions_envelope_clicked)
        layout.addWidget(check_btn)

        self.safety_log = QTextEdit()
        self.safety_log.setReadOnly(True)
        layout.addWidget(self.safety_log)
        return tab

    def _build_sim_post_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.post_combo = QComboBox()
        self.post_combo.addItems([
            "Generic GRBL / 3-Axis (grbl)",
            "LinuxCNC (linuxcnc)",
            "Mach3 (mach3)",
            "Fanuc (fanuc)",
        ])
        layout.addWidget(QLabel("Post-Processor Dialect"))
        layout.addWidget(self.post_combo)

        apply_post_btn = QPushButton("Set Post-Processor")
        apply_post_btn.clicked.connect(self._on_set_post_clicked)
        layout.addWidget(apply_post_btn)

        step_row = QHBoxLayout()
        prev_btn = QPushButton("Step -1")
        next_btn = QPushButton("Step +1")
        reset_btn = QPushButton("Reset")
        prev_btn.clicked.connect(lambda: self._on_step_sim(-1))
        next_btn.clicked.connect(lambda: self._on_step_sim(1))
        reset_btn.clicked.connect(self._on_reset_sim)
        step_row.addWidget(prev_btn)
        step_row.addWidget(next_btn)
        step_row.addWidget(reset_btn)
        layout.addLayout(step_row)

        self.sim_status = QTextEdit()
        self.sim_status.setReadOnly(True)
        layout.addWidget(self.sim_status)
        return tab

    def _build_setup_export_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        setup_box = QGroupBox("Active setup")
        setup_form = QFormLayout(setup_box)
        self.setup_work_offset = QComboBox()
        self.setup_work_offset.addItems(["G54", "G55", "G56", "G57", "G58", "G59"])
        self.setup_work_offset.setToolTip("Work coordinate system written at the start of the program.")
        setup_form.addRow("Work offset", self.setup_work_offset)
        self.setup_safe_z = self._double_spin(5.0, 1.0, 500.0, 2)
        self.setup_safe_z.setToolTip("Rapid moves travel this far above the stock top.")
        setup_form.addRow("Safe Z · mm above stock", self.setup_safe_z)
        self.setup_retract = self._double_spin(2.0, 0.5, 500.0, 2)
        self.setup_retract.setToolTip("Retract height above the stock top between passes.")
        setup_form.addRow("Retract clearance · mm", self.setup_retract)
        self.setup_stock_clearance = self._double_spin(1.0, 0.0, 100.0, 2)
        self.setup_stock_clearance.setToolTip("Extra margin around the part for toolpaths.")
        setup_form.addRow("Stock clearance · mm", self.setup_stock_clearance)
        for widget in (self.setup_safe_z, self.setup_retract, self.setup_stock_clearance):
            widget.valueChanged.connect(self._store_setup_fields)
        self.setup_work_offset.currentTextChanged.connect(self._store_setup_fields)
        layout.addWidget(setup_box)

        stock_est_btn = QPushButton("Estimate Stock Removal && Features")
        stock_est_btn.clicked.connect(self._on_estimate_stock_and_features)
        layout.addWidget(stock_est_btn)

        export_pkg_btn = QPushButton("Export CAM Package (.nc + Sheets)")
        export_pkg_btn.clicked.connect(self._on_export_cam_package_clicked)
        layout.addWidget(export_pkg_btn)

        self.setup_sheet_preview = QTextEdit()
        self.setup_sheet_preview.setReadOnly(True)
        layout.addWidget(self.setup_sheet_preview)
        return tab

    def _on_apply_library_tool_clicked(self) -> None:
        tool_id = self.tool_library_combo.currentData()
        if tool_id and hasattr(self, "apply_tool_library_entry"):
            entry = self.apply_tool_library_entry(str(tool_id))
            if entry is not None:
                self.speeds_readout.setPlainText(
                    f"Loaded {entry.name}\nRPM: {entry.default_rpm} | Feed: {entry.default_feed_mm_min} mm/min | Stepdown: {entry.default_stepdown_mm} mm"
                )

    def _on_calc_feeds_speeds_clicked(self) -> None:
        if hasattr(self, "recommend_feeds_and_speeds"):
            rec = self.recommend_feeds_and_speeds(flutes=int(self.flutes_spin.value()))
            self.speeds_readout.setPlainText(
                f"RPM: {rec.rpm}\nFeed: {rec.feed_mm_min} mm/min (Plunge: {rec.plunge_mm_min})\n"
                f"Stepdown: {rec.stepdown_mm} mm | SFM: {rec.surface_speed_m_min} m/min"
            )

    def _on_add_fixture_clicked(self) -> None:
        kind = self.fixture_name_combo.currentText()
        idx = len(self.fixture_manager.fixtures) + 1
        fix = WorkholdingFixture(
            fixture_id=f"FIX_{idx:02d}",
            name=f"{kind} {idx}",
            kind=kind,
            position_mm=(float(self.fix_x.value()), float(self.fix_y.value()), float(self.fix_z.value())),
            keep_out=self.fix_keep_out.isChecked(),
        )
        self.fixture_manager.add_fixture(fix)
        tag = " · KEEP OUT" if fix.keep_out else ""
        self.fixtures_list.addItem(f"{fix.fixture_id}: {fix.name} @ {fix.position_mm}{tag}")
    def _init_preview_and_sketch_assign_ui(self) -> None:
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(120)
        self._sim_timer.timeout.connect(self._on_sim_timer_tick)
        self._toolpath_preview_actors: list = []
        self._sim_marker_actor = None

        # Append Sketch -> Toolpath Assignment & 3D Preview controls to Sim & Post tab
        sim_tab = self.cam_tabs.widget(2)
        if sim_tab is not None and sim_tab.layout() is not None:
            lay = sim_tab.layout()

            play_btn = QPushButton("Play / Pause Sim")
            play_btn.clicked.connect(self._on_toggle_sim_playback)
            lay.addWidget(play_btn)

            preview_tp_btn = QPushButton("Show / Refresh 3D Toolpath Preview")
            preview_tp_btn.clicked.connect(self.render_toolpath_preview)
            lay.addWidget(preview_tp_btn)

            clear_tp_btn = QPushButton("Clear 3D Toolpath Preview")
            clear_tp_btn.clicked.connect(self.clear_toolpath_preview)
            lay.addWidget(clear_tp_btn)

            lay.addWidget(QLabel("Assign Operation to Sketch"))
            self.sketch_assign_combo = QComboBox()
            self.op_assign_combo = QComboBox()
            lay.addWidget(self.sketch_assign_combo)
            lay.addWidget(self.op_assign_combo)

            assign_btn = QPushButton("Assign Toolpath to Sketch")
            assign_btn.clicked.connect(self._on_assign_toolpath_to_sketch)
            lay.addWidget(assign_btn)

            self.refresh_sketch_and_op_combos()

        # Append Tool Preview button/readout to Tools & Speeds tab
        tools_tab = self.cam_tabs.widget(0)
        if tools_tab is not None and tools_tab.layout() is not None:
            t_lay = tools_tab.layout()
            tool_prev_btn = QPushButton("Preview Tool Cylinder in Scene")
            tool_prev_btn.clicked.connect(self.render_tool_preview)
            t_lay.addWidget(tool_prev_btn)

    def refresh_sketch_and_op_combos(self) -> None:
        if not hasattr(self, "sketch_assign_combo") or not hasattr(self, "op_assign_combo"):
            return
        self.sketch_assign_combo.clear()
        for feat in getattr(getattr(self, "sketch_workflow", None), "features", []):
            fid = getattr(feat, "feature_id", "")
            fname = getattr(feat, "name", fid)
            assigned = getattr(feat, "assigned_operation_name", None) or "Unassigned"
            self.sketch_assign_combo.addItem(f"{fname} [{assigned}]", fid)

        self.op_assign_combo.clear()
        ops = list(getattr(self, "operations", []))
        if not ops:
            for s in getattr(self, "setups", []):
                ops.extend(getattr(s, "operations", []))
        for op in ops:
            self.op_assign_combo.addItem(getattr(op, "name", "Operation"), getattr(op, "name", ""))

    def _on_assign_toolpath_to_sketch(self) -> None:
        sketch_id = self.sketch_assign_combo.currentData()
        op_name = self.op_assign_combo.currentData()
        if not sketch_id or not op_name:
            return
        setups = list(getattr(self, "setups", []))
        active_setup = setups[getattr(self, "active_setup_index", 0)] if setups else None
        for feat in getattr(getattr(self, "sketch_workflow", None), "features", []):
            if getattr(feat, "feature_id", None) == sketch_id:
                feat.assigned_operation_name = str(op_name)
                if active_setup is not None:
                    feat.assigned_setup_name = getattr(active_setup, "name", None)
                    feat.assigned_setup_color = getattr(active_setup, "color", None)
                if hasattr(self, "record_workspace_edit"):
                    self.record_workspace_edit(f"Assign {op_name} to {getattr(feat, 'name', sketch_id)}")
                break
        self.refresh_sketch_and_op_combos()
        if hasattr(self, "_refresh_sketch_tree"):
            self._refresh_sketch_tree()

    def render_tool_preview(self) -> None:
        plotter = getattr(self, "plotter", None) or getattr(self, "engine", None)
        if plotter is None or not hasattr(plotter, "add_mesh"):
            return
        dia = float(self._tool_diameter_mm()) if hasattr(self, "tool_diameter") else 6.0
        length = max(10.0, dia * 4.0)
        radius = max(0.25, dia * 0.5)
        pos = (0.0, 0.0, length * 0.5 + 5.0)
        frame = self.toolpath_player.current_frame()
        if frame is not None:
            pos = (frame.position[0], frame.position[1], frame.position[2] + length * 0.5)
        cyl = pv.Cylinder(center=pos, direction=(0.0, 0.0, 1.0), radius=radius, height=length)
        actor = plotter.add_mesh(cyl, name="cam_tool_preview", color=theme_color("tool"), opacity=0.75)
        self._toolpath_preview_actors.append(actor)
        if hasattr(plotter, "render"):
            plotter.render()

    def render_toolpath_preview(self) -> None:
        plotter = getattr(self, "plotter", None)
        if plotter is None or not hasattr(plotter, "add_mesh"):
            return
        self.clear_toolpath_preview()
        segs = self._current_toolpath_segments()
        if not segs:
            return
        self.toolpath_player.set_segments(segs)

        cut_pts, cut_lines = [], []
        rapid_pts, rapid_lines = [], []
        for seg in segs:
            s = list(seg.start)
            e = list(seg.end)
            if seg.kind in ("rapid", "retract"):
                idx = len(rapid_pts)
                rapid_pts.extend([s, e])
                rapid_lines.append([2, idx, idx + 1])
            else:
                idx = len(cut_pts)
                cut_pts.extend([s, e])
                cut_lines.append([2, idx, idx + 1])

        if cut_pts and cut_lines:
            cut_poly = pv.PolyData(cut_pts)
            cut_poly.lines = [v for row in cut_lines for v in row]
            self._toolpath_preview_actors.append(
                plotter.add_mesh(cut_poly, name="cam_tp_cut", color=theme_color("toolpath_cut"), line_width=2.5)
            )
        if rapid_pts and rapid_lines:
            rapid_poly = pv.PolyData(rapid_pts)
            rapid_poly.lines = [v for row in rapid_lines for v in row]
            self._toolpath_preview_actors.append(
                plotter.add_mesh(rapid_poly, name="cam_tp_rapid", color=theme_color("toolpath_rapid"), line_width=1.2, opacity=0.65)
            )
        self._update_sim_marker()
        if hasattr(plotter, "render"):
            plotter.render()

    def clear_toolpath_preview(self) -> None:
        plotter = getattr(self, "plotter", None)
        if plotter is not None and hasattr(plotter, "remove_actor"):
            for name in ("cam_tp_cut", "cam_tp_rapid", "cam_sim_marker", "cam_tool_preview"):
                try:
                    plotter.remove_actor(name)
                except Exception:
                    pass
        self._toolpath_preview_actors.clear()
        self._sim_marker_actor = None

    def _update_sim_marker(self) -> None:
        plotter = getattr(self, "plotter", None)
        frame = self.toolpath_player.current_frame()
        if plotter is None or frame is None or not hasattr(plotter, "add_mesh"):
            return
        dia = float(self._tool_diameter_mm()) if hasattr(self, "tool_diameter") else 6.0
        sphere = pv.Sphere(radius=max(0.5, dia * 0.4), center=frame.position)
        self._sim_marker_actor = plotter.add_mesh(sphere, name="cam_sim_marker", color=theme_color("sim_marker"))

    def _on_toggle_sim_playback(self) -> None:
        if self._sim_timer.isActive():
            self._sim_timer.stop()
            self.sim_status.append("Playback paused.")
        else:
            if not self.toolpath_player.segments:
                self.toolpath_player.set_segments(self._current_toolpath_segments())
            self._sim_timer.start()
            self.sim_status.append("Playback running...")

    def _on_sim_timer_tick(self) -> None:
        if not self.toolpath_player.segments:
            self._sim_timer.stop()
            return
        if self.toolpath_player.current_index >= len(self.toolpath_player.segments) - 1:
            self._sim_timer.stop()
            self.sim_status.append("Playback complete.")
            return
        frame = self.toolpath_player.step(1)
        if frame is not None:
            x, y, z = frame.position
            self.sim_status.setPlainText(
                f"Frame {frame.index + 1}/{len(self.toolpath_player.segments)} | t={frame.time_s:.2f}s\n"
                f"Kind: {frame.segment_kind} | Op: {frame.operation_name}\n"
                f"Pos: X={x:.3f}, Y={y:.3f}, Z={z:.3f} | F={frame.feed_mm_min:.1f} | S={frame.rpm}"
            )
            self._update_sim_marker()

    def _current_toolpath_segments(self):
        segs = list(getattr(self.toolpath_player, "segments", []))
        if segs:
            return segs
        ops = list(getattr(self, "operations", []))
        setups = list(getattr(self, "setups", []))
        active_setup = setups[getattr(self, "active_setup_index", 0)] if setups else None
        out = []
        for op in ops:
            out.extend(self.toolpath_dynamics.build_operation_segments(op, setup=active_setup))
        return out

    def _on_check_collisions_envelope_clicked(self) -> None:
        segs = self._current_toolpath_segments()
        col = self.check_toolpath_collisions(segs) if hasattr(self, "check_toolpath_collisions") else None
        env = self.check_machine_envelope(segs) if hasattr(self, "check_machine_envelope") else None
        issues = self.validate_gcode_and_toolpaths() if hasattr(self, "validate_gcode_and_toolpaths") else []

        lines = [f"Checked {len(segs)} segment(s):"]
        if col is not None:
            lines.append(f"- Fixture collisions: {'YES' if col.has_collisions else 'None'} ({len(col.errors)} errors, {len(col.warnings)} warnings)")
            lines.extend(f"  ERR: {e}" for e in col.errors)
            lines.extend(f"  WARN: {w}" for w in col.warnings)
        if env is not None:
            lines.append(f"- Machine envelope: {'OK' if env.within_limits else 'VIOLATIONS'} ({len(env.violations)})")
            lines.extend(f"  ENV: {v}" for v in env.violations)
        if issues:
            lines.append(f"- Validator diagnostics: {len(issues)}")
            lines.extend(f"  [{i.severity.upper()}] {i.code}: {i.message}" for i in issues)
        self.safety_log.setPlainText("\n".join(lines))

    def _on_set_post_clicked(self) -> None:
        label = self.post_combo.currentText()
        ctrl = label.split("(")[-1].rstrip(")").strip()
        self.post_compiler.spec.name = label
        self.post_compiler.spec.controller = ctrl
        self.sim_status.setPlainText(f"Active post-processor: {label}")

    def _on_step_sim(self, delta: int) -> None:
        if not self.toolpath_player.segments:
            self.toolpath_player.set_segments(self._current_toolpath_segments())
        frame = self.toolpath_player.step(delta)
        if frame is not None:
            x, y, z = frame.position
            self.sim_status.setPlainText(
                f"Frame {frame.index + 1}/{len(self.toolpath_player.segments)} | t={frame.time_s:.2f}s\n"
                f"Kind: {frame.segment_kind} | Op: {frame.operation_name}\n"
                f"Pos: X={x:.3f}, Y={y:.3f}, Z={z:.3f} | F={frame.feed_mm_min:.1f} | S={frame.rpm}"
            )

    def _on_reset_sim(self) -> None:
        frame = self.toolpath_player.reset()
        self.sim_status.setPlainText("Simulation reset." if frame is None else f"Frame 1: {frame.position}")

    def _on_estimate_stock_and_features(self) -> None:
        est = self.estimate_stock_removal() if hasattr(self, "estimate_stock_removal") else None
        feats = self.recognize_dynamic_features() if hasattr(self, "recognize_dynamic_features") else None
        parts = []
        if est is not None:
            parts.append(
                f"Stock Vol: {est.stock_volume_mm3:.1f} mm³ | Removed: {est.estimated_removed_mm3:.1f} mm³ "
                f"({est.removal_ratio * 100:.1f}%) | Remaining: {est.remaining_volume_mm3:.1f} mm³"
            )
        if feats is not None:
            parts.append(f"Detected features: {feats.source_summary}")
        self.setup_sheet_preview.setPlainText("\n".join(parts))

    def build_setup_sheet(self, program_map, start: int = 1, end: int | None = None, estimate_s: float | None = None) -> str:
        """Setup sheet for what a program (or a section of it) contains: operations, tools,
        time estimate, work offsets and notes. Used by the export tab and the advanced editor."""
        end = end or len(program_map.lines)
        info = section_setup_sheet(program_map, start, end, estimate_s)
        lines = [
            f"# Setup sheet · {self.project_name_label.text() if hasattr(self, 'project_name_label') else 'Job'}",
            "",
            f"- Program lines: {info['lines'][0]}-{info['lines'][1]}",
            f"- Units: {info['units']}",
            f"- Work offsets: {', '.join(info['work_offsets']) or 'G54'}",
            f"- Estimated time: {format_duration(estimate_s) if estimate_s is not None else 'not estimated'}",
            f"- Stock: {self._stock_material_name()} · " + " × ".join(f"{v:g}" for v in self._stock_dims_mm().values()) + " mm",
            "",
            "## Operations",
        ]
        lines += [f"{i}. {name}" for i, name in enumerate(info["operations"], start=1)] or ["(none in this section)"]
        lines += ["", "## Tools"] + ([f"- {t}" for t in info["tools"]] or ["- (not stated in the code)"])
        notes = getattr(self, "job_notes", "") or ""
        if notes:
            lines += ["", "## Notes", notes]
        return "\n".join(lines) + "\n"

    def _load_setup_fields(self) -> None:
        if not hasattr(self, "setup_safe_z") or not self.setups:
            return
        setup = self.active_setup
        widgets = (self.setup_work_offset, self.setup_safe_z, self.setup_retract, self.setup_stock_clearance)
        for widget in widgets:
            widget.blockSignals(True)
        self.setup_work_offset.setCurrentText(getattr(setup, "work_offset", "G54"))
        self.setup_safe_z.setValue(float(getattr(setup, "safe_z_mm", 5.0)))
        self.setup_retract.setValue(float(getattr(setup, "retract_clearance_mm", 2.0)))
        self.setup_stock_clearance.setValue(float(getattr(setup, "stock_clearance_mm", 1.0)))
        for widget in widgets:
            widget.blockSignals(False)

    def _store_setup_fields(self, *_args) -> None:
        if not self.setups:
            return
        setup = self.active_setup
        setup.work_offset = self.setup_work_offset.currentText()
        setup.safe_z_mm = self.setup_safe_z.value()
        setup.retract_clearance_mm = min(self.setup_retract.value(), self.setup_safe_z.value())
        setup.stock_clearance_mm = self.setup_stock_clearance.value()
        self._mark_dirty()

    def _on_export_cam_package_clicked(self) -> None:
        prepared = self._prepare_gcode()  # same checks as Generate G-code (sketches, gate, fit, guard)
        if prepared is None:
            return
        gcode_text, _settings = prepared
        out_dir = QFileDialog.getExistingDirectory(self, "Select CAM Export Folder", str(Path.home()))
        if not out_dir:
            return
        segs = list(self.toolpath_player.segments)
        paths = self.export_full_cam_package(out_dir, gcode_text=gcode_text, segments=segs)
        self.setup_sheet_preview.setPlainText(
            "Exported CAM package:\n" + "\n".join(f"- {k}: {v}" for k, v in paths.items())
        )
class BasiCAMWindow(CamWorkspaceTabsMixin, WorkspaceBundleMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("BasiCAM", "BasiCAM CNC")
        activate_theme(self.settings)
        self._sync_terminal_colors()
        set_dwg_converter(str(self.settings.value("import/dwg_converter", "") or ""))
        self.safety = SafetyGates(self.settings)
        self.machine_store = MachineStore(self.settings)
        self.current_machine: MachineProfile | None = self.machine_store.get(self.machine_store.default_id)
        self.last_wco: tuple[float, float, float] | None = None
        self.preflight_complete = False
        self._preflight_pending = False
        self.terminal_core = TerminalCore(self.settings)
        self.motion_guard = MotionGuard(self)
        self.setWindowTitle("BasiCAM CNC")
        self.resize(1500, 930)
        self.init_bundle_and_history()
        self.feature_recognizer = DynamicFeatureRecognizer()
        self.tool_library: list[ToolLibraryEntry] = list(DEFAULT_TOOL_LIBRARY)
        self.gcode_validator = GCodeValidator()
        self.fixture_manager = CollisionAndFixtureManager()
        self.feeds_speeds_calc = FeedsAndSpeedsCalculator()
        self.envelope_checker = MachineEnvelopeChecker(
            self.current_machine.kinematics_profile() if self.current_machine else MachineKinematicsProfile()
        )
        self.init_cam_tabs_dock()
        self._init_preview_and_sketch_assign_ui()
        self.stock_removal_estimator = StockRemovalEstimator()
        self.toolpath_dynamics = ToolpathDynamicsEngine(ToolpathDynamicsConfig())
        self.toolpath_player = ToolpathSimulationPlayer()
        self.post_compiler = PostProcessorCompiler(
            PostProcessorSpec(name="Generic GRBL / 3-Axis", controller="grbl")
        )
        self.tools = self._load_tool_inventory()
        self.command_actions: dict[str, QAction] = {}
        _raw_orbit = self.settings.value("view/orbit_sensitivity", 0.45, type=float)
        self.orbit_sensitivity = float(_raw_orbit if isinstance(_raw_orbit, (int, float)) else 0.45)
        self.units = UnitSettings()
        self.parts: list[PartModel] = []
        self.stock_part: PartModel | None = None
        self.stock_mesh: pv.PolyData | None = None
        self.stock_imported = False
        self.stock_note = ""
        self.finish_allowance = 0.5
        self.roughing_analysis: RoughingAnalysis | None = None
        self.setups: list[MachiningSetup] = [
            MachiningSetup(name="Setup 1", orientation="Top (+Z)", color=SETUP_PALETTE[0])
        ]
        self.active_setup_index = 0
        self.sketch_workflow = SketchWorkflowManager()
        self.sketch_workflow.last_good = LastGoodStateStore()
        self.assignment_ledger = AssignmentLedger()
        self.sketch_entities: list[SketchEntity] = []
        self.pending_sketch_points: list[np.ndarray] = []
        self.active_sketch_id: str | None = None
        self.dragging_sketch: tuple[str, int] | None = None
        self.measure_points: list[np.ndarray] = []
        self.measure_active = False
        self.feature_pick_active = False
        self.selected_feature: tuple[int, int] | None = None
        self.selected_features: set[tuple[int, int]] = set()
        self.operation_target = "Whole part"
        self.highlight_feature_ids: set[tuple] = set()
        self.sketch_notes: dict[str, str] = {}
        self.selected_operation: tuple[int, int] | None = None
        self.known_feature_targets: set[tuple[int, int]] = set()
        self.dataset_part_indexes: dict[str, int] = {}
        self.stock_dataset_keys: set[str] = set()
        self.selected_cells: dict[int, set[int]] = {}
        self.drag_origin = QPoint()
        self.drag_current = QPoint()
        self.drag_mode = ""
        self.drag_rubber_band: QRubberBand | None = None
        self.drag_threshold = 5
        self.drag_additive = False
        self.last_hover_note = ""
        self.work_origin_set = False
        self.machine_profile_set = False
        self.postprocessor_set = False
        self.workholding_checked = False
        self.grbl_controller: GrblController | None = None
        self.grbl_capabilities: GrblCapabilities | None = None
        self.grbl_status: GrblStatus | None = None
        self.grbl_identification_lines: list[str] = []
        self.machine_poll_ticks = 0
        self.position_verified = False
        self.homing_in_progress = False
        self.speed_multiplier = 1.0
        self.keyboard_control = KeyboardMachineControl()
        self.ghost_tool_pose: tuple[float, float, float] | None = None
        self.ghost_path: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self.multiplier_state = MultiplierState()
        self.stream_feed_scale = 1.0  # software F-word scale beyond GRBL's 200% override
        self.grbl_settings: dict[int, float] = {}
        self.cycle_estimator: CycleEstimator | None = None
        self.stream_index = 0
        self.stream_lines: list[str] = []
        self.stream_source_offset = 0
        self.stream_waiting = False
        self.program_running = False
        self.streaming_program = False
        self.gcode_progress: dict | None = None
        self.advanced_editor = None
        self._build_ui()
        self._load_machine_state()
        self.parts_list.installEventFilter(self)
        self._refresh_scene(reset_camera=False)
        self.zero_reference_mpos: tuple[float, float, float] | None = None
        self.tracked_work_pos: list[float] = [0.0, 0.0, 0.0]
    def _toggle_tooltips(self, enabled: bool) -> None:
        self.settings.setValue("ui/tooltips_enabled", enabled)
        for widget in QApplication.allWidgets():
            if not enabled:
                tip = widget.toolTip()
                if tip:
                    widget.setProperty("_saved_tooltip", tip)
                    widget.setToolTip("")
            else:
                saved = widget.property("_saved_tooltip")
                if saved:
                    widget.setToolTip(str(saved))

    def render_cad_or_mesh(
        plotter,
        part_model,
        *,
        name_prefix: str,
        color: str = "#8a9296",
        opacity: float = 1.0,
        show_surface: bool = True,
        show_topological_edges: bool = True,
        edge_color: str = "#9aa7ab",
        line_width: float = 1.4,
    ) -> None:
        """Renders a CAD/B-rep part via pyvista-cad when available, hiding triangle-mesh noise."""
        mesh = getattr(part_model, "mesh", None)
        brep_shape = getattr(part_model, "brep_shape", None)

        if HAS_PYVISTA_CAD and show_surface and hasattr(plotter, "cad"):
            cad_source = brep_shape if brep_shape is not None else mesh
            if cad_source is not None:
                plotter.cad.add(
                    cad_source,
                    color=color,
                    edge_color=edge_color if show_topological_edges else None,
                    line_width=line_width,
                    opacity=opacity,
                    name=f"{name_prefix}-cad",
                )
                return

        if show_surface and mesh is not None and mesh.n_cells > 0:
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                show_edges=False,
                name=f"{name_prefix}-surface",
            )

        edge_mesh = getattr(part_model, "edge_mesh", None)
        if show_topological_edges and edge_mesh is not None and edge_mesh.n_cells > 0:
            plotter.add_mesh(
                edge_mesh,
                color=edge_color,
                opacity=0.82,
                line_width=line_width,
                name=f"{name_prefix}-edges",
            )
            
        
    def _draw_sketch_entities(self) -> None:
        self.sketch_workflow.sync_tooling_assignments(self.setups)
        if not self.sketch_workflow.features:
            return
        if hasattr(self, "sketch_visible") and not self.sketch_visible.isChecked():
            return
        show_construction = (
            self.construction_visible.isChecked()
            if hasattr(self, "construction_visible")
            else True
        )

        for idx, feat in enumerate(self.sketch_workflow.features):
            if feat.construction and not show_construction:
                continue
            pts = np.asarray(feat.active_points, dtype=float)
            if pts.size == 0:
                continue
            color = feat.display_color
            is_selected = feat.feature_id == self.active_sketch_id
            if ("sketch", feat.feature_id, -1) in getattr(self, "highlight_feature_ids", set()):
                color = theme_color("highlight")
                is_selected = True
            if feat.kind == "Point" or len(pts) == 1:
                cloud = pv.PolyData(pts[:1])
                self.plotter.add_mesh(
                    cloud,
                    color=color,
                    point_size=12 if is_selected else 9,
                    render_points_as_spheres=True,
                    name=f"sketch-point-{idx}",
                )
            else:
                line_mesh = self._sketch_polyline(pts)
                if feat.construction:  # pyvista has no line-style option: build dashes as geometry
                    line_mesh = self._dashed(line_mesh)
                self.plotter.add_mesh(
                    line_mesh,
                    color=color,
                    line_width=3.2 if is_selected else (1.5 if feat.construction else 2.4),
                    name=f"sketch-entity-{idx}",
                )
                ctrl_pts = np.asarray(feat.points, dtype=float)
                endpoints = pv.PolyData(ctrl_pts)
                self.plotter.add_mesh(
                    endpoints,
                    color=color,
                    point_size=10 if is_selected else 7,
                    render_points_as_spheres=True,
                    name=f"sketch-handles-{idx}",
                )
                if feat.is_projected and len(ctrl_pts) >= 2:
                    flat_mesh = self._dashed(self._sketch_polyline(ctrl_pts))
                    self.plotter.add_mesh(
                        flat_mesh,
                        color=theme_color("reference"),
                        line_width=1.2,
                        opacity=0.65,
                        name=f"sketch-plane-ref-{idx}",
                    )

    @staticmethod
    def _sketch_polyline(points) -> pv.PolyData:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return pv.lines_from_points(pts) if len(pts) >= 2 else pv.PolyData(pts)

    def _dashed(self, mesh: pv.PolyData) -> pv.PolyData:
        if mesh.n_cells == 0 or mesh.n_points < 2:
            return mesh
        size = float(np.linalg.norm(np.ptp(np.asarray(mesh.points), axis=0))) or 10.0
        dashed = dash_lines(mesh, dash=max(size / 40.0, 0.5), gap=max(size / 60.0, 0.3))
        return dashed if dashed.n_cells else mesh
    @property
    def active_setup(self) -> MachiningSetup:
        if not self.setups:
            self.setups.append(
                MachiningSetup(name="Setup 1", orientation="Top (+Z)", color=SETUP_PALETTE[0])
            )
            self.active_setup_index = 0
        self.active_setup_index = min(max(self.active_setup_index, 0), len(self.setups) - 1)
        return self.setups[self.active_setup_index]

    @property
    def operations(self) -> list[Operation]:
        return [op for setup in self.setups for op in setup.operations]

    def _has_any_operations(self) -> bool:
        return any(setup.operations for setup in self.setups)

    def _build_ui(self) -> None:
        self.setStyleSheet(self._stylesheet())
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_project_panel())
        splitter.addWidget(self._build_view_panel())
        splitter.setSizes([245, 900])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self._install_workflow_tabs()
        self._build_machine_dock()
        self.terminal = TerminalDock(self.terminal_core, self)
        self.terminal.view.submitted.connect(lambda text: self._send_console_command(text, self.terminal.view))
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.terminal)
        self.machine_dock.visibilityChanged.connect(self._terminal_takeover)
        self._build_menus()
        apply_tooltips(self)
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet(
            f"background:{COLORS['panel']};color:{COLORS['muted']};padding:5px 10px"
        )

    def _load_tool_inventory(self) -> list[ToolDefinition]:
        serialized = self.settings.value("tools/inventory", "")
        if serialized:
            try:
                return [ToolDefinition.from_dict(item) for item in json.loads(serialized)]
            except (TypeError, ValueError, KeyError):
                self.settings.remove("tools/inventory")
        return default_tool_inventory()

    def _refresh_tool_combo(self) -> None:
        if not hasattr(self, "tool_name"):
            return
        current = self.tool_name.currentText()
        self.tool_name.blockSignals(True)
        self.tool_name.clear()
        for index, tool in enumerate(self.tools):
            self.tool_name.addItem(tool.name, index)
        if current and self.tool_name.findText(current) >= 0:
            self.tool_name.setCurrentText(current)
        elif current:
            self.tool_name.setEditText(current)
        self.tool_name.blockSignals(False)

    def _apply_tool_to_operation(self, index: int) -> None:
        if not 0 <= index < len(self.tools) or not hasattr(self, "tool_diameter"):
            return
        tool = self.tools[index]
        self._set_tool_diameter_mm(tool.diameter_mm)
        if tool.tool_type in {"Drill", "Tap", "Reamer"}:
            self.operation_kind.setCurrentText(tool.tool_type)
        if hasattr(self, "tool_description"):
            flutes = getattr(tool, "flute_count", None)
            parts = [str(tool.tool_type), f"Ø{tool.diameter_mm:g} mm"]
            if flutes:
                parts.append(f"{flutes} flute")
            self.tool_description.setText(" · ".join(parts))
        self._refresh_tool_preview()
        self._apply_recommended_defaults()

    def _refresh_tool_preview(self) -> None:
        """Show the selected inventory tool's 3D envelope in the operation panel."""
        plotter = getattr(self, "tool_preview_plotter", None)
        index = self.tool_name.currentIndex() if hasattr(self, "tool_name") else -1
        if plotter is None or not 0 <= index < len(self.tools):
            return
        try:
            mesh = create_tool_mesh(self.tools[index])
        except (ValueError, TypeError, AttributeError) as error:
            self.tool_description.setText(f"Preview unavailable: {error}")
            return
        plotter.clear_actors()
        plotter.add_mesh(mesh, color=theme_color("tool"), show_edges=False, name="tool_preview")
        self.tool_preview_mesh = mesh
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.render()

    # --- tool diameter is stored in mm but shown in the current display unit -------------
    def _tool_diameter_mm(self) -> float:
        value = float(self.tool_diameter.value())
        return value * 25.4 if self.units.display is LengthUnit.INCHES else value

    def _set_tool_diameter_mm(self, mm: float) -> None:
        inches = self.units.display is LengthUnit.INCHES
        self.tool_diameter.blockSignals(True)
        self.tool_diameter.setValue(mm / 25.4 if inches else mm)
        self.tool_diameter.blockSignals(False)

    def _refresh_unit_labels(self, previous_mm: float | None = None) -> None:
        if not hasattr(self, "tool_diameter_label"):
            return
        inches = self.units.display is LengthUnit.INCHES
        mm = previous_mm if previous_mm is not None else self._tool_diameter_mm()
        self.tool_diameter_label.setText(f"Diameter ({self.units.display.value})")
        self.tool_diameter.blockSignals(True)
        self.tool_diameter.setDecimals(3 if inches else 2)
        self.tool_diameter.setRange(0.004 if inches else 0.1, 4.0 if inches else 100.0)
        self.tool_diameter.setValue(mm / 25.4 if inches else mm)
        self.tool_diameter.blockSignals(False)

    # --- overlap defaults ---------------------------------------------------------------
    HARD_MATERIALS = {"Mild steel", "Stainless steel", "Bronze"}
    QUALITY_OVERLAP = {"Draft": 40.0, "Standard": 50.0, "Fine": 70.0}

    def _default_overlap_pct(self) -> float:
        base = self.QUALITY_OVERLAP.get(self.finish_quality.currentText(), 50.0)
        if self.stock_material.currentText() in self.HARD_MATERIALS:
            base += 10.0
        return min(base, 90.0)

    def _apply_default_overlap(self, *_args) -> None:
        if not hasattr(self, "overlap") or getattr(self, "_overlap_user_edited", False):
            return
        self.overlap.blockSignals(True)
        self.overlap.setValue(self._default_overlap_pct())
        self.overlap.blockSignals(False)

    def _on_overlap_edited(self, _value: float) -> None:
        self._overlap_user_edited = abs(self.overlap.value() - self._default_overlap_pct()) > 0.01

    # --- pass/end depth only matter when a cut is aimed at a sketch ----------------------
    def _operation_targets_sketch(self) -> bool:
        target = getattr(self, "operation_target", "")
        return any(f.name == target for f in self.sketch_workflow.features)

    def _update_operation_form_visibility(self) -> None:
        if not hasattr(self, "operation_form"):
            return
        visible = self._operation_targets_sketch()
        for widget in (self.stepdown, self.depth):
            self.operation_form.setRowVisible(widget, visible)

    def open_tool_library(self) -> None:
        dialog = ToolLibraryDialog(self.tools, self, machine_store=self.machine_store)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.tools = dialog.tools
            self.settings.setValue(
                "tools/inventory", json.dumps([tool.to_dict() for tool in self.tools])
            )
            self._refresh_tool_combo()
        self._refresh_machine_combo()
        if self.current_machine is not None:
            updated = self.machine_store.get(self.current_machine.machine_id)
            if updated is not None:
                self.current_machine = updated
                self._sync_machine_profile()

    # ------------------------------------------------------------ machines (Block 5)
    def _refresh_machine_combo(self) -> None:
        if not hasattr(self, "machine_combo"):
            return
        self.machine_combo.blockSignals(True)
        self.machine_combo.clear()
        self.machine_combo.addItem("No machine selected", None)
        machines = self.machine_store.list()
        current_id = self.current_machine.machine_id if self.current_machine else None
        if self.current_machine is not None and all(m.machine_id != current_id for m in machines):
            machines.append(self.current_machine)  # project copy of a machine deleted from the inventory
        for machine in machines:
            self.machine_combo.addItem(machine.name, machine.machine_id)
        index = self.machine_combo.findData(current_id)
        self.machine_combo.setCurrentIndex(max(index, 0))
        self.machine_combo.blockSignals(False)

    def _machine_state(self) -> dict:
        state = {
            "soft_limits": {a: dict(vars(l)) for a, l in self.current_machine.soft_limits.items()},
            "z_reference_tool": self.current_machine.z_reference_tool,
            "z_reference_stick_out_mm": self.current_machine.z_reference_stick_out_mm,
            "tool": self.tool_name.currentText() if hasattr(self, "tool_name") else "",
            "wco": list(self.last_wco) if self.last_wco is not None else None,
            "home": list(self.zero_reference_mpos) if getattr(self, "zero_reference_mpos", None) else None,
        }
        if hasattr(self, "jog_distance"):
            state["jog_distance"] = self.jog_distance.value()
            state["jog_feed"] = self.jog_feed.value()
        return state

    def _save_machine_state(self) -> None:
        if self.current_machine is not None:
            self.machine_store.save_state(self.current_machine.machine_id, self._machine_state())

    def _load_machine_state(self, override: dict | None = None) -> None:
        """Restore the machine's last good state (soft limits, tool, work offset, home, jog)."""
        if self.current_machine is None:
            return
        from basicam.machine_profiles import SoftLimit

        state = override if override is not None else self.machine_store.load_state(self.current_machine.machine_id)
        for axis, raw in dict(state.get("soft_limits", {})).items():
            self.current_machine.soft_limits[axis] = SoftLimit.from_dict(raw)
        if state.get("z_reference_stick_out_mm"):
            self.current_machine.z_reference_tool = str(state.get("z_reference_tool", ""))
            self.current_machine.z_reference_stick_out_mm = float(state["z_reference_stick_out_mm"])
        wco = state.get("wco")
        self.last_wco = tuple(float(v) for v in wco) if wco else None
        home = state.get("home")
        self.zero_reference_mpos = tuple(float(v) for v in home) if home else None
        if hasattr(self, "jog_distance") and "jog_distance" in state:
            self.jog_distance.setValue(float(state["jog_distance"]))
            self.jog_feed.setValue(float(state["jog_feed"]))
        if state.get("tool") and hasattr(self, "tool_name"):
            index = self.tool_name.findText(str(state["tool"]))
            if index >= 0:
                self.tool_name.setCurrentIndex(index)

    def _sync_machine_profile(self) -> None:
        profile = self.current_machine.kinematics_profile() if self.current_machine else MachineKinematicsProfile()
        self.envelope_checker = MachineEnvelopeChecker(profile)

    def _set_current_machine(self, machine_id: str | None, load_state: bool = True) -> None:
        """Select a machine for this project and as the default for new projects."""
        self._save_machine_state()
        machine = self.machine_store.get(machine_id) if machine_id else None
        if machine_id and machine is None and self.current_machine and self.current_machine.machine_id == machine_id:
            machine = self.current_machine
        self.current_machine = machine
        self.machine_store.default_id = machine.machine_id if machine else None
        self.last_wco = None
        if load_state:
            self._load_machine_state()
        self._sync_machine_profile()
        self._refresh_machine_combo()
        self._mark_dirty()
        self.terminal_core.log("info", f"Machine: {machine.name if machine else 'none selected'}")

    def _open_soft_limit_wizard(self) -> None:
        if self.current_machine is None:
            QMessageBox.information(self, "Soft limits", "Select or add a machine first (Tool inventory > Machines).")
            return

        def position():
            status = self.grbl_status
            return status.machine_position if status is not None and status.machine_position else None

        wizard = SoftLimitWizard(copy.deepcopy(self.current_machine), self.safety, self.tools, position, self)
        if wizard.exec() == QDialog.DialogCode.Accepted:
            self.current_machine = wizard.machine
            self.machine_store.save(self.current_machine)
            self._save_machine_state()
            self._sync_machine_profile()
            self.terminal_core.log("info", f"Soft limits saved for {self.current_machine.name}.")

    def _start_preflight(self) -> None:
        """Run after connecting, after an alarm and after a hard lock; motion stays off until complete."""
        self.preflight_complete = False
        if getattr(self, "_preflight_dialog", None) is not None and self._preflight_dialog.isVisible():
            return
        axes = self.grbl_capabilities.axes if self.grbl_capabilities else "XYZ"
        dialog = PreflightDialog(self.current_machine, axes, self)
        dialog.finished.connect(lambda result, d=dialog: self._preflight_finished(d, result))
        self._preflight_dialog = dialog
        self.terminal_core.log("warning", "Preflight: move each axis through its travel and tick it to enable motion.")
        dialog.open()

    def _preflight_finished(self, dialog, result: int) -> None:
        self._preflight_dialog = None
        if result == QDialog.DialogCode.Accepted and dialog.is_complete():
            self.preflight_complete = True
            self.terminal_core.log("info", "Preflight complete; motion enabled.")
            if dialog.wants_soft_limit_setup:
                self._open_soft_limit_wizard()
        else:
            self.terminal_core.log("warning", "Preflight not completed; motion stays disabled.")
        self._set_machine_controls()

    def _check_bed_fit(self, part) -> None:
        """Warn when a part is larger than the selected machine, or suggest a machine that fits."""
        if part is None or part.mesh is None or part.mesh.n_points == 0:
            return
        b = part.mesh.bounds
        size = (b[1] - b[0], b[3] - b[2], b[5] - b[4])
        if self.current_machine is not None:
            axes = fit_report(size, self.current_machine)
            if axes:
                message = (
                    f"{part.name} is larger than {self.current_machine.name} in {', '.join(axes)}. "
                    "It may still be machined on an improvised fixture; G-code is refused if all three exceed."
                )
                self.terminal_core.log("warning", message)
                QMessageBox.warning(self, "Part larger than the work area", message)
            return
        machines = self.machine_store.list()
        if not machines:
            return
        fits = compatible_machines(machines, size)
        if not fits:
            message = f"No machine in the inventory is large enough for {part.name}."
            self.terminal_core.log("warning", message)
            QMessageBox.warning(self, "No machine large enough", message)
            return
        dialog = BedFitDialog(fits, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.choice == "use" and dialog.selected_id():
                self._set_current_machine(dialog.selected_id())
            elif dialog.choice == "add":
                self.open_tool_library()

    def _part_size_for_generation(self) -> tuple[float, float, float] | None:
        targets = [p for p in self.parts if p is not self.stock_part and p.mesh is not None and p.mesh.n_points]
        if not targets:
            return None
        bounds = np.array([p.mesh.bounds for p in targets])
        return (
            float(bounds[:, 1].max() - bounds[:, 0].min()),
            float(bounds[:, 3].max() - bounds[:, 2].min()),
            float(bounds[:, 5].max() - bounds[:, 4].min()),
        )

    def _generation_fit_allowed(self) -> bool:
        size = self._part_size_for_generation()
        if size is None:
            return True
        ok, message = generation_fit_rule(size, self.current_machine, self.safety.allows("generate_oversize_part"))
        if message:
            self.terminal_core.log("error" if not ok else "warning", message)
            (self._critical_box if not ok else QMessageBox.warning)(self, "Part size and work area", message)
        return ok

    def _add_command(self, menu, name: str, title: str, shortcut: str, callback) -> QAction:
        action = QAction(title, self)
        stored_shortcut = self.settings.value(f"keybinds/{name}", shortcut)
        action.setShortcut(QKeySequence(stored_shortcut))
        action.triggered.connect(callback)
        menu.addAction(action)
        self.addAction(action)
        self.command_actions[name] = action
        self.default_shortcuts[name] = shortcut
        return action

    def _build_menus(self) -> None:
        self.command_actions.clear()
        self.default_shortcuts: dict[str, str] = {}
        file_menu = self.menuBar().addMenu("File")
        help_menu = file_menu.addMenu("Help")
        repo_action = QAction("GitHub Repository", self)
        repo_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/sharxbyte/BasiCAM_CNC"))
        )
        help_menu.addAction(repo_action)
        file_menu.addSeparator()

        self.open_bcam_action.setShortcut(QKeySequence("Ctrl+Alt+O"))
        for project_action in (self.open_bcam_action, self.save_bcam_action, self.save_bcam_as_action):
            file_menu.addAction(project_action)
            self.addAction(project_action)
        file_menu.addSeparator()
        self._add_command(file_menu, "open_part", "Open part...", "Ctrl+O", self.open_part)
        self._add_command(file_menu, "import_part", "Import part...", "Ctrl+Shift+I", self.import_part)
        self._add_command(file_menu, "export_part", "Export part...", "", self.export_part)
        self._add_command(file_menu, "import_stock", "Import stock...", "Ctrl+Shift+O", self.open_stock)
        self._add_command(file_menu, "readiness", "G-code checklist...", "Ctrl+G", self.show_gcode_readiness)
        self.generate_action = QAction("Generate G-code...", self)
        self.generate_action.setEnabled(False)
        self.generate_action.triggered.connect(self.export_gcode)
        file_menu.addAction(self.generate_action)
        self._add_command(file_menu, "export_dxf", "Export sketches as DXF...", "Ctrl+Shift+D", self.export_sketches_dxf_dialog)
        self._add_command(file_menu, "exit", "Exit", "Alt+F4", self.close)

        edit_menu = self.menuBar().addMenu("Edit")
        self.undo_action.setShortcuts([QKeySequence("Ctrl+Z")])
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        self.addAction(self.undo_action)
        self.addAction(self.redo_action)
        edit_menu.addSeparator()
        self._add_command(edit_menu, "cut", "Cut", "Ctrl+X", self.cut_selection)
        self._add_command(edit_menu, "copy", "Copy", "Ctrl+C", self.copy_selection)
        self._add_command(edit_menu, "paste", "Paste", "Ctrl+V", self.paste_selection)
        self._add_command(edit_menu, "delete", "Delete", "Delete", self.delete_layered)
        self._add_command(edit_menu, "cancel", "Cancel / back out", "Esc", self.escape_layered)
        edit_menu.addSeparator()
        self._add_command(edit_menu, "copy_note", "Copy note", "Ctrl+Shift+C", self._copy_note)
        self._add_command(edit_menu, "paste_note", "Paste note", "Ctrl+Shift+V", self._paste_note)

        sketch_menu = self.menuBar().addMenu("Sketch")
        self._add_command(sketch_menu, "sketch_new", "New sketch / finish sketch", "S", self.sketch_mode.toggle)
        self._add_command(sketch_menu, "sketch_dimension", "Dimension…", "D", self._focus_dimension_input)
        sketch_menu.addSeparator()
        self._add_command(sketch_menu, "sketch_line", "Line", "L", lambda *_: self._sketch_hotkey("Line", "Two point"))
        self._add_command(sketch_menu, "sketch_circle", "Circle (center point)", "C", lambda *_: self._sketch_hotkey("Circle", "Center"))
        self._add_command(sketch_menu, "sketch_rectangle", "Rectangle (corner)", "R", lambda *_: self._sketch_hotkey("Rectangle", "Corner"))
        self._add_command(sketch_menu, "sketch_point", "Point", "P", lambda *_: self._sketch_hotkey("Point"))
        self._add_command(sketch_menu, "sketch_arc", "Arc (3 point)", "A", lambda *_: self._sketch_hotkey("Curve", "3 point arc"))

        view_menu = self.menuBar().addMenu("View")
        self._add_command(view_menu, "fit_view", "Fit model", "F", self._fit_view)
        standard_views = view_menu.addMenu("Standard view")
        for name, shortcut in (
            ("Front", "1"), ("Back", "Ctrl+1"), ("Left", "2"),
            ("Right", "Ctrl+2"), ("Top", "3"), ("Bottom", "Ctrl+3"),
            ("Isometric", "0"),
        ):
            self._add_command(
                standard_views, f"view_{name.casefold()}", name, shortcut,
                lambda checked=False, view=name: self.set_standard_view(view),
            )
        self._add_command(view_menu, "normal_view", "Normal to selected", "N", self.view_normal_to_selected)
        grid_action = QAction("Visible grid", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(True)
        grid_action.toggled.connect(self.grid_toggle.setChecked)
        self.grid_toggle.toggled.connect(grid_action.setChecked)
        view_menu.addAction(grid_action)

        tools_menu = self.menuBar().addMenu("Tools")
        self._add_command(tools_menu, "tool_inventory", "Tool inventory...", "Ctrl+T", self.open_tool_library)
        self.advanced_gcode_action = self._add_command(
            tools_menu, "advanced_gcode", "Advanced G-code...", "", lambda *_: self.open_advanced_gcode()
        )
        self.advanced_gcode_action.setToolTip("Edit, preview, export and resume G-code; needs the gate in Settings > Advanced.")
        self._refresh_advanced_gcode_action()

        options_menu = self.menuBar().addMenu("Options")
        self._add_command(options_menu, "key_bindings", "Key bindings...", "", self._show_key_bindings)
        self._add_command(options_menu, "preferences", "Preferences...", "", self._show_preferences)
        self._add_command(options_menu, "job_info", "Job information...", "", self._show_job_information)
        colorblind_action = QAction("Color-blind friendly axes", self)
        colorblind_action.setCheckable(True)
        colorblind_action.setChecked(self.settings.value("view/colorblind_axes", False, type=bool))
        colorblind_action.toggled.connect(self._toggle_colorblind_axes)
        options_menu.addAction(colorblind_action)
        tooltips_action = QAction("Toggle tool tips", self)
        tooltips_action.setCheckable(True)
        tooltips_action.setChecked(self.settings.value("ui/tooltips_enabled", True, type=bool))
        tooltips_action.toggled.connect(self._toggle_tooltips)
        options_menu.addAction(tooltips_action)
        self.keyboard_confirm_action = QAction("Confirm keyboard operations", self)
        self.keyboard_confirm_action.setCheckable(True)
        self.keyboard_confirm_action.setChecked(self.extra_keyboard_confirm)
        self.keyboard_confirm_action.setEnabled(not self.safety.beginner_mode)
        self.keyboard_confirm_action.setToolTip("Ask before keyboard moves and jogs are applied; always on in Beginner mode.")
        self.keyboard_confirm_action.toggled.connect(self._set_extra_keyboard_confirm)
        options_menu.addAction(self.keyboard_confirm_action)
        self.keyboard_machine_action = QAction("Keyboard machine control", self)
        self.keyboard_machine_action.setCheckable(True)
        self.keyboard_machine_action.setToolTip("Drive the machine (or the virtual ghost) from the keyboard; same as the Machine tab toggle.")
        self.keyboard_machine_action.toggled.connect(self.keyboard_control_toggle.setChecked)
        self.keyboard_control_toggle.toggled.connect(self.keyboard_machine_action.setChecked)
        options_menu.addAction(self.keyboard_machine_action)
        self.advanced_settings_action = self._add_command(
            options_menu, "advanced_settings", "Settings > Advanced...", "", self._show_advanced_settings
        )
        self.advanced_settings_action.setToolTip("Safety overrides; each starts off and needs its warning acknowledged.")
        self.advanced_settings_action.setVisible(not self.safety.beginner_mode)

        machine_menu = self.menuBar().addMenu("Machine")
        for axis in "XYZ":
            for sign, label in ((-1, "−"), (1, "+")):
                self._add_command(
                    machine_menu, f"machine_jog_{axis.lower()}{'p' if sign > 0 else 'm'}", f"Jog {axis}{label}", "",
                    lambda *_a, a=axis, s=sign: self._jog_axis(a, s),
                )
        machine_menu.addSeparator()
        self._add_command(machine_menu, "machine_home", "Home machine", "", self._home_machine)
        self._add_command(machine_menu, "machine_hold", "Feed hold", "", lambda *_: self._send_realtime(b"!", "Feed hold (!)"))
        self._add_command(machine_menu, "machine_resume", "Resume", "", lambda *_: self._send_realtime(b"~", "Resume (~)"))
        self._add_command(machine_menu, "machine_soft_limits", "Soft limits...", "", self._open_soft_limit_wizard)
        self._add_command(view_menu, "position_part", "Position part...", "", lambda *_: self._show_tab("Position"))
        self._add_command(view_menu, "work_axes", "Assign work axes...", "", self._show_axes_assignment)

        window_menu = self.menuBar().addMenu("Window")
        window_menu.addAction(self.machine_dock.toggleViewAction())
        window_menu.addAction(self.terminal.toggleViewAction())
        self.terminal_collapse_action = QAction("Collapse terminal", self)
        self.terminal_collapse_action.setCheckable(True)
        self.terminal_collapse_action.setToolTip("Shrink the bottom terminal to its title bar; it stays collapsed until unticked.")
        self.terminal_collapse_action.toggled.connect(self._toggle_terminal_collapsed)
        window_menu.addAction(self.terminal_collapse_action)

    def _copy_note(self) -> None:
        QApplication.clipboard().setText(self.note_editor.toPlainText())

    def _paste_note(self) -> None:
        self.note_editor.insertPlainText(QApplication.clipboard().text())

    def _build_machine_dock(self) -> None:
        self.machine_dock = QDockWidget("Machine & Console", self)
        self.machine_dock.setObjectName("machineConsoleDock")
        self.machine_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        tabs = QTabWidget()
        machine = QWidget()
        machine_layout = QHBoxLayout(machine)
        connection_panel = QVBoxLayout()
        machine_row = QHBoxLayout()
        machine_row.addWidget(QLabel("Machine"))
        self.machine_combo = QComboBox()
        self.machine_combo.setToolTip("Machine profile used for travel, limits and keep-out zones; add machines in Tool inventory > Machines.")
        self.machine_combo.activated.connect(lambda _i: self._set_current_machine(self.machine_combo.currentData()))
        machine_row.addWidget(self.machine_combo, 1)
        connection_panel.addLayout(machine_row)
        port_row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(210)
        port_row.addWidget(self.port_combo, 1)
        refresh_ports = QPushButton("Refresh")
        refresh_ports.clicked.connect(self._refresh_machine_ports)
        port_row.addWidget(refresh_ports)
        connection_panel.addLayout(port_row)
        connect_row = QHBoxLayout()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["115200", "57600", "250000"])
        connect_row.addWidget(self.baud_combo)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._toggle_machine_connection)
        connect_row.addWidget(self.connect_button)
        connection_panel.addLayout(connect_row)
        self.machine_status_label = QLabel("Disconnected · GRBL-compatible controllers only")
        self.machine_status_label.setWordWrap(True)
        connection_panel.addWidget(self.machine_status_label)
        self.machine_info_label = QLabel("No controller identified")
        self.machine_info_label.setObjectName("mutedText")
        self.machine_info_label.setWordWrap(True)
        connection_panel.addWidget(self.machine_info_label)
        self.position_verified_box = QCheckBox("Machine position homed / verified")
        self.position_verified_box.toggled.connect(self._set_position_verified)
        connection_panel.addWidget(self.position_verified_box)
        self.dro_label = QLabel("DRO ΔX: +0.000  ΔY: +0.000  ΔZ: +0.000 mm")
        self.dro_label.setObjectName("panelTitle")
        connection_panel.addWidget(self.dro_label)

        zero_btn = QPushButton("Zero work (G10)")
        zero_btn.clicked.connect(self._set_work_zero_reference)
        connection_panel.addWidget(zero_btn)
        machine_layout.addLayout(connection_panel, 2)

        jog_group = QGroupBox("Jog · mm")
        jog_layout = QGridLayout(jog_group)
        self.jog_distance = self._double_spin(1.0, 0.01, 25.0, 2)
        self.jog_feed = self._double_spin(300.0, 1.0, 5000.0, 1)
        jog_layout.addWidget(QLabel("Step"), 0, 0)
        jog_layout.addWidget(self.jog_distance, 0, 1, 1, 2)
        jog_layout.addWidget(QLabel("Feed"), 1, 0)
        jog_layout.addWidget(self.jog_feed, 1, 1, 1, 2)
        self.jog_buttons: dict[str, QPushButton] = {}
        for axis_index, axis in enumerate("XYZ"):
            for direction_index, sign in enumerate((-1, 1)):
                label = f"{axis}{'−' if sign < 0 else '+'}"
                button = QPushButton(label)
                button.clicked.connect(
                    lambda checked=False, a=axis, s=sign: self._jog_axis(a, s)
                )
                self.jog_buttons[label] = button
                jog_layout.addWidget(button, 2 + axis_index, direction_index)
        self.keyboard_control_toggle = QPushButton("Keyboard control")
        self.keyboard_control_toggle.setCheckable(True)
        self.keyboard_control_toggle.setToolTip(
            "Arrows X/Y, PgUp/PgDn Z, Home homes, End stops; or X/Y/Z, +/-, distance, Enter (preview), Enter (send)."
        )
        self.keyboard_control_toggle.toggled.connect(self._on_keyboard_control_toggled)
        jog_layout.addWidget(self.keyboard_control_toggle, 6, 0, 1, 2)
        self.keyboard_virtual_box = QCheckBox("Virtual (ghost only)")
        self.keyboard_virtual_box.setChecked(True)
        self.keyboard_virtual_box.setToolTip("Virtual moves only the ghost tool and tracked position; nothing is sent.")
        jog_layout.addWidget(self.keyboard_virtual_box, 7, 0, 1, 2)
        self.keyboard_prompt = QLabel("")
        self.keyboard_prompt.setObjectName("mutedText")
        self.keyboard_prompt.setWordWrap(True)
        jog_layout.addWidget(self.keyboard_prompt, 8, 0, 1, 2)
        self.soft_limits_button = QPushButton("Soft limits...")
        self.soft_limits_button.setToolTip("Record hard limits at the endstops and set soft-limit margins for the selected machine.")
        self.soft_limits_button.clicked.connect(self._open_soft_limit_wizard)
        jog_layout.addWidget(self.soft_limits_button, 5, 0, 1, 2)
        machine_layout.addWidget(jog_group)

        actions_layout = QVBoxLayout()
        self.home_button = QPushButton("Home ($H)")
        self.home_button.clicked.connect(self._home_machine)
        actions_layout.addWidget(self.home_button)
        self.unlock_button = QPushButton("Unlock alarm ($X)")
        self.unlock_button.clicked.connect(self._unlock_machine)
        actions_layout.addWidget(self.unlock_button)
        self.hold_button = QPushButton("Feed hold")
        self.hold_button.clicked.connect(lambda: self._send_realtime(b"!", "Feed hold"))
        actions_layout.addWidget(self.hold_button)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(lambda: self._send_realtime(b"~", "Cycle start"))
        actions_layout.addWidget(self.resume_button)
        kill_row = QHBoxLayout()
        self.kill_button = QPushButton("Kill")
        self.kill_button.setObjectName("killButton")
        self.kill_button.setStyleSheet(
            """
            QPushButton#killButton {
                background-color: #3a4245;
                color: #6b787a;
                border: 1px solid #3a4245;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton#killButton:hover {
                background-color: #4a5255;
                color: #8b989a;
            }
            QPushButton#killButton:pressed {
                background-color: #313739;
            }
        """
        )
        self.kill_button.clicked.connect(self._kill_machine)
        kill_row.addWidget(self.kill_button, 1)
        actions_layout.addLayout(kill_row)
        self.reset_controller_button = QPushButton("Soft reset")
        self.reset_controller_button.clicked.connect(self._reset_machine_controller)
        actions_layout.addWidget(self.reset_controller_button)
        spindle_row = QHBoxLayout()
        self.spindle_rpm = self._int_spin(5000, 1, 60000)
        spindle_row.addWidget(self.spindle_rpm)
        self.spindle_start_button = QPushButton("Spindle on")
        self.spindle_start_button.clicked.connect(self._start_spindle)
        spindle_row.addWidget(self.spindle_start_button)
        self.spindle_stop_button = QPushButton("Off")
        self.spindle_stop_button.clicked.connect(lambda: self._send_machine_line("M5"))
        spindle_row.addWidget(self.spindle_stop_button)
        actions_layout.addLayout(spindle_row)
        machine_layout.addLayout(actions_layout)
        manual_cut_group = QGroupBox("Manual cut · mm")
        manual_layout = QFormLayout(manual_cut_group)
        self.manual_cut_dir = QComboBox()
        self.manual_cut_dir.addItems(["X+", "X-", "Y+", "Y-", "Z+", "Z-"])
        manual_layout.addRow("Axis", self.manual_cut_dir)
        self.manual_cut_distance = self._double_spin(5.0, 0.01, 500.0, 2)
        manual_layout.addRow("Move", self.manual_cut_distance)
        self.manual_cut_feed = self._double_spin(300.0, 1.0, 5000.0, 1)
        manual_layout.addRow("Feed", self.manual_cut_feed)
        self.manual_cut_rpm = self._int_spin(5000, 0, 60000)
        manual_layout.addRow("RPM", self.manual_cut_rpm)
        self.manual_cut_spindle_running_box = QCheckBox("Spindle already running")
        manual_layout.addRow("", self.manual_cut_spindle_running_box)
        run_cut_btn = QPushButton("Run cut")
        run_cut_btn.clicked.connect(self._execute_manual_cut)
        manual_layout.addRow(run_cut_btn)
        machine_layout.addWidget(manual_cut_group)
        tabs.addTab(machine, "Machine")

        self.console_view = TerminalView(self.terminal_core)
        self.console_view.submitted.connect(lambda text: self._send_console_command(text, self.console_view))
        self.console_output = self.console_view.output
        self.console_input = self.console_view.input_line
        self.console_send_button = self.console_view.send_button
        self.terminal_core.log(
            "info", "Disconnected. Connect a GRBL-compatible controller to query status or send commands."
        )
        tabs.addTab(self.console_view, "Console")

        self.machine_timer = QTimer(self)
        self.machine_timer.setInterval(100)
        self.machine_timer.timeout.connect(self._poll_machine)

        self.machine_dock.setWidget(tabs)
        tabs.installEventFilter(self)  # keyboard machine control while the Machine tab has focus
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.machine_dock)
        self.machine_dock.hide()
        self._refresh_machine_ports()
        self._refresh_machine_combo()
        self._set_machine_controls()

    def _refresh_machine_ports(self) -> None:
        current_port = self.port_combo.currentData()
        self.port_combo.clear()
        for device, description in available_serial_ports():
            self.port_combo.addItem(f"{device} · {description}", device)
        if current_port:
            index = self.port_combo.findData(current_port)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("No serial ports found", "")

    def _toggle_machine_connection(self) -> None:
        if self.grbl_controller is not None:
            self.machine_timer.stop()
            self._save_machine_state()
            self.preflight_complete = False
            self.grbl_controller.close()
            self.grbl_controller = None
            self.grbl_capabilities = None
            self.grbl_status = None
            self.position_verified = False
            self.homing_in_progress = False
            self.position_verified_box.setChecked(False)
            self.machine_status_label.setText("Disconnected · GRBL-compatible controllers only")
            self.machine_info_label.setText("No controller identified")
            self._console_log("Disconnected.")
            self._set_machine_controls()
            return

        port = self.port_combo.currentData()
        if not port:
            self.machine_status_label.setText("Select an available serial port first.")
            return
        answer = QMessageBox.question(
            self,
            "Connect to GRBL controller",
            "Only GRBL 1.1-compatible controllers are supported. Opening a serial port can reset some controllers.\n\n"
            "Confirm the machine is clear, the emergency stop is accessible, and the selected port is correct.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            baudrate = int(self.baud_combo.currentText())
            self.grbl_controller = GrblController(str(port), baudrate)
            self.grbl_identification_lines.clear()
            self.grbl_capabilities = None
            self.grbl_status = None
            self.position_verified = False
            self.homing_in_progress = False
            self.position_verified_box.setChecked(False)
            self.preflight_complete = False
            self._preflight_pending = True  # preflight repeats after every (re)connect
            self.grbl_controller.send_line("$I")
            self.grbl_controller.send_line("$$")  # read-only: axis maximum rates $110-$112 for the multiplier clamp
            self.grbl_controller.send_realtime(b"?")
            self.machine_poll_ticks = 0
            self.machine_status_label.setText(f"Connected to {port} · identifying GRBL firmware…")
            self._console_log(f"Connected to {port} at {baudrate} baud; sent $I and status query.")
            self.machine_timer.start()
        except Exception as error:
            if self.grbl_controller is not None:
                self.grbl_controller.close()
            self.grbl_controller = None
            self._critical_box(self, "Controller connection failed", str(error))
        self._set_machine_controls()

    def _poll_machine(self) -> None:
        if self.grbl_controller is None:
            return
        try:
            self.machine_poll_ticks += 1
            if self.machine_poll_ticks % 5 == 0:
                self.grbl_controller.send_realtime(b"?")
            lines = self.grbl_controller.read_lines()
        except Exception as error:
            self._console_log(f"Serial error: {error}")
            self.machine_timer.stop()
            self.grbl_controller.close()
            self.grbl_controller = None
            self.grbl_capabilities = None
            self.grbl_status = None
            self.position_verified = False
            self.position_verified_box.setChecked(False)
            self._update_dro_readout()
            self._set_machine_controls()
            return

        for line in lines:
            self.terminal_core.log("error" if line.upper().startswith(("ALARM", "ERROR")) else "response", line)
            self._record_fault(line)
            if line.startswith("$") and "=" in line:
                self.grbl_settings.update(parse_grbl_settings([line]))
            self._stream_ack(line)
            self.grbl_identification_lines.append(line)
            if len(self.grbl_identification_lines) > 100:
                self.grbl_identification_lines = self.grbl_identification_lines[-100:]
            status = parse_grbl_status(line)
            if status is not None:
                self.grbl_status = status
                if status.work_offset is not None:
                    self.last_wco = tuple(status.work_offset)
                if status.state.casefold().startswith("alarm") and self.preflight_complete:
                    self.preflight_complete = False  # alarm or hard lock: preflight again
                    self._preflight_pending = True
                self._update_dro_readout()
                if status.state.casefold() == "home":
                    self.homing_in_progress = True
                elif status.state.casefold() == "idle" and self.homing_in_progress:
                    self.homing_in_progress = False
                    self.position_verified = True
                    self.position_verified_box.setChecked(True)
                elif status.state.casefold().startswith("alarm"):
                    self.position_verified = False
                    self.position_verified_box.setChecked(False)
            elif line.upper().startswith("ALARM:"):
                self.grbl_status = GrblStatus(line)
                self.preflight_complete = False
                self._preflight_pending = True
                self.position_verified = False
                self.position_verified_box.setChecked(False)
                self._update_dro_readout()
        self.grbl_capabilities = parse_grbl_capabilities(self.grbl_identification_lines)
        identified = self.grbl_capabilities is not None and self.grbl_capabilities.version.startswith("1.1")
        state = self.grbl_status.state.casefold() if self.grbl_status is not None else ""
        if identified and getattr(self, "_preflight_pending", False) and not state.startswith("alarm"):
            self._preflight_pending = False
            self._start_preflight()
        self._set_machine_controls()

    def _set_machine_controls(self) -> None:
        connected = self.grbl_controller is not None
        identified = bool(
            connected
            and self.grbl_capabilities is not None
            and self.grbl_capabilities.version.startswith("1.1")
        )
        state = self.grbl_status.state if self.grbl_status is not None else "Unknown"
        controller_idle = identified and state.casefold() == "idle"
        idle = controller_idle and self.position_verified
        axes = self.grbl_capabilities.axes if identified else ""
        self.connect_button.setText("Disconnect" if connected else "Connect")
        # The console stays usable without a machine (help/status); sending is refused by the validator.
        self.console_input.setEnabled(True)
        self.console_send_button.setEnabled(True)
        motion_ok = bool(idle and self.preflight_complete)
        for label, button in self.jog_buttons.items():
            button.setEnabled(bool(motion_ok and label[0] in axes))
        self.home_button.setEnabled(bool(controller_idle))
        self.position_verified_box.setEnabled(bool(controller_idle))
        self.unlock_button.setEnabled(bool(identified and state.casefold().startswith("alarm")))
        self.hold_button.setEnabled(bool(identified and state.casefold() in {"run", "jog", "home"}))
        self.resume_button.setEnabled(bool(identified and state.casefold().startswith("hold")))
        self.kill_button.setEnabled(bool(connected))
        self.reset_controller_button.setEnabled(bool(connected))
        self.spindle_start_button.setEnabled(motion_ok)
        self.spindle_stop_button.setEnabled(bool(idle or state.casefold() in {"run", "jog"}))

        self._update_status_indicator(connected, identified, state)
        if not connected:
            self.machine_status_label.setText("Disconnected · GRBL-compatible controllers only")
            self.machine_badge.setText("●  SIMULATION ONLY")
        elif not identified:
            self.machine_status_label.setText("Serial connected · waiting for a GRBL firmware response")
            self.machine_badge.setText("●  IDENTIFYING")
        else:
            position_state = "position verified" if self.position_verified else "home/verify position to enable motion"
            self.machine_status_label.setText(
                f"GRBL {self.grbl_capabilities.version} · {state} · {position_state}"
            )
            self.machine_badge.setText("●  GRBL CONNECTED")
        if identified:
            capabilities = self.grbl_capabilities
            options = f" · options {capabilities.options}" if capabilities.options else ""
            position = ""
            if self.grbl_status and self.grbl_status.machine_position:
                xyz = self.grbl_status.machine_position
                position = f"\nMachine XYZ: {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f} mm"
            pins = (
                f"\nActive input pins: {self.grbl_status.pins}"
                if self.grbl_status and self.grbl_status.pins
                else ""
            )
            self.machine_info_label.setText(
                f"Axes: {capabilities.axes}{options}{position}{pins}"
            )
        else:
            self.machine_info_label.setText("Motion commands remain disabled until GRBL is identified and Idle.")

    @staticmethod
    def machine_state_text(connected: bool, identified: bool, state: str, verified: bool, running: bool) -> str:
        if not connected:
            return "Disconnected / simulation only"
        if state.casefold().startswith("alarm"):
            return "ALARM"
        if running or state.casefold() in {"run", "jog", "home"}:
            return "RUNNING"
        if identified and verified:
            return "Connected - ready"
        return "Connected - not homed"

    def _record_fault(self, line: str) -> str | None:
        """GRBL alarm/error: timestamped log line, kept for the status indicator."""
        description = describe_grbl_message(line)
        if description is None:
            return None
        stamp = time.strftime("%H:%M:%S")
        self.last_fault = description
        self.fault_log = (getattr(self, "fault_log", []) + [f"[{stamp}] {description}"])[-200:]
        self.terminal_core.log("error", f"[{stamp}] {description}")
        if hasattr(self, "machine_state_indicator"):
            self.machine_state_indicator.setToolTip(f"Last fault at {stamp}: {description}")
        return description

    def _update_status_indicator(self, connected: bool, identified: bool, state: str) -> None:
        if not hasattr(self, "machine_state_indicator"):
            return
        text = self.machine_state_text(connected, identified, state, self.position_verified, self._program_running())
        fault = getattr(self, "last_fault", "")
        if text == "ALARM" and fault.startswith("ALARM"):
            text = fault.split(":")[0]  # e.g. "ALARM 1"
        colors = {"RUNNING": theme_color("warning"), "Connected - ready": theme_color("ok")}
        color = theme_color("error") if text.startswith("ALARM") else colors.get(text, "#9fb0b3")
        self.machine_state_indicator.setText(text)
        self.machine_state_indicator.setStyleSheet(f"color: {color}; font-weight: 700;")

    def _console_log(self, line: str, level: str = "info") -> None:
        self.terminal_core.log(level, line)

    def _critical_box(self, parent, title: str, text: str, *args):
        """QMessageBox.critical that also logs the failure in the terminal."""
        if hasattr(self, "terminal_core"):
            self.terminal_core.log("error", f"{title}: {text}")
        return QMessageBox.critical(parent, title, text, *args)

    def _terminal_takeover(self, machine_dock_visible: bool) -> None:
        if hasattr(self, "terminal"):
            self.terminal.takeover(bool(machine_dock_visible))

    def _toggle_terminal_collapsed(self, collapsed: bool) -> None:
        self.terminal.set_collapsed(bool(collapsed), manual=True)

    def _program_running(self) -> bool:
        if getattr(self, "program_running", False):
            return True
        state = self.grbl_status.state.casefold() if self.grbl_status is not None else ""
        return state in {"run", "hold", "hold:0", "hold:1"} and bool(getattr(self, "streaming_program", False))

    def _terminal_status_lines(self) -> list[str]:
        """Status report; every line is a comment so nothing printed can be executed."""
        lines = [f"Last result: {self.terminal_core.last_result}"]
        if self.grbl_controller is None:
            lines.append("Machine: not connected (simulation only)")
        else:
            caps = self.grbl_capabilities
            version = caps.version if caps is not None else "unidentified"
            lines.append(f"Machine: {getattr(self.grbl_controller, 'port', 'serial')} · GRBL {version}")
            if self.grbl_status is not None:
                lines.append(f"Controller state: {self.grbl_status.state}")
                if self.grbl_status.machine_position:
                    x, y, z = self.grbl_status.machine_position[:3]
                    lines.append(f"Machine XYZ: {x:.3f}, {y:.3f}, {z:.3f} mm")
                if self.grbl_status.pins:
                    lines.append(f"Input pins: {self.grbl_status.pins}")
            lines.append(f"Position verified: {'yes' if self.position_verified else 'no'}")
            for raw in [line for line in self.grbl_identification_lines if line.startswith(("[VER", "[OPT", "[GC:"))][-3:]:
                lines.append(f"Controller: {raw}")
        lines.append(f"Tool: {self.tool_name.currentText() if hasattr(self, 'tool_name') else '-'}")
        setup = self.setups[self.active_setup_index] if self.setups else None
        if setup is not None:
            ops = setup.operations
            lines.append(f"Setup: {setup.name} · {len(ops)} operation(s)")
            lines.append(f"Operation: {ops[-1].name if ops else 'none'}")
        lines.append(f"Program running: {'yes' if self._program_running() else 'no'}")
        return ["; " + line.replace("\n", " ") for line in lines]

    def _console_motion_check(self, command: str) -> tuple[bool, str]:
        """Motion guard for console lines (called by CommandValidator before the confirmation)."""
        if command.upper().startswith("$H"):
            return True, ""
        if not self.preflight_complete:
            return False, "Finish the preflight check before sending motion."
        violations = self.motion_guard.check_line(command)
        if violations:
            self.motion_guard.report(violations, "Console")
            return False, violations[0].text()
        return True, ""

    def _send_machine_line(self, command: str) -> None:
        if self.grbl_controller is None or self.grbl_capabilities is None:
            self.terminal_core.log("warning", "No machine connected; nothing was sent.")
            return
        try:
            self.terminal_core.log("command", command)
            self.grbl_controller.send_line(command)
            self.terminal_core.record_success(f"Sent: {command}")
        except Exception as error:
            self.terminal_core.log("error", f"Local error: {error}")

    def _send_console_command(self, command: str | None = None, view=None) -> None:
        if command is None or isinstance(command, bool):
            command = self.console_input.text()
        view = view or self.console_view
        command = command.strip()
        if not command:
            return
        self.terminal_core.add_history(command)
        validator = CommandValidator(motion_check=getattr(self, "_console_motion_check", None))
        result = validator.validate(
            command,
            connected=self.grbl_controller is not None and self.grbl_capabilities is not None,
            running=self._program_running(),
            position_verified=self.position_verified,
        )
        if result.action == "help":
            self.terminal_core.log("info", command)
            for line in CommandValidator.HELP_TEXT:
                self.terminal_core.log("info", line)
            view.clear_input()
            return
        if result.action == "status":
            self.terminal_core.log("info", command)
            for line in self._terminal_status_lines():
                self.terminal_core.log("info", line)
            if self.grbl_controller is not None:
                try:  # read-only diagnostics; responses arrive through the poll loop
                    self.grbl_controller.send_realtime(b"?")
                    if not self._program_running():
                        self.grbl_controller.send_line("$G")
                    self.terminal_core.log("info", "; Queried controller status and parser state (read-only).")
                except Exception as error:
                    self.terminal_core.log("error", f"; Diagnostic query failed: {error}")
            view.clear_input()
            return
        if result.action == "refuse":
            self.terminal_core.log("warning", f"{command}: {result.message}")
            self.statusBar().showMessage(result.message, 6000)
            return
        if result.action == "realtime":
            if result.realtime == b"\x18":
                self._reset_machine_controller()  # keeps its confirmation
            else:
                self._send_realtime(result.realtime, command)
            view.clear_input()
            return
        if command.upper() == "$X":
            self._unlock_machine()  # keeps its confirmation
            view.clear_input()
            return
        if result.moves:
            answer = QMessageBox.question(
                self,
                "Send motion command",
                "This command may move an axis or start the spindle. Verify the work area is clear.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._send_machine_line(command)
        view.clear_input()

    def _jog_axis(self, axis: str, sign: int) -> None:
        if self.grbl_controller is None or self.grbl_status is None or self.grbl_status.state.casefold() != "idle":
            return
        if not self.preflight_complete:
            self.terminal_core.log("warning", "Jog refused: finish the preflight check first.")
            return
        try:
            distance = self.jog_distance.value() * sign
            feed = self.jog_feed.value()
            if not self.motion_guard.report(self.motion_guard.check_jog(axis, distance), "Jog"):
                return
            self.grbl_controller.jog(axis, distance, feed)
            self._record_relative_move(axis, distance)
            self._console_log(f"> $J=G91 G21 {axis}{distance:.4f} F{feed:.1f}")
        except Exception as error:
            self._console_log(f"Jog rejected: {error}")

    # ------------------------------------------------------------ Advanced G-code mode (Block 12)
    def _advanced_gcode_allowed(self) -> bool:
        return bool(self.safety.allows("advanced_gcode_mode")) and not self.safety.beginner_mode

    def _refresh_advanced_gcode_action(self) -> None:
        allowed = self._advanced_gcode_allowed()
        for widget in (getattr(self, "advanced_gcode_action", None), getattr(self, "advanced_editor_button", None)):
            if widget is not None:
                widget.setEnabled(allowed)

    def open_advanced_gcode(self, text: str | None = None) -> AdvancedEditorWindow | None:
        if not self._advanced_gcode_allowed():
            QMessageBox.information(
                self, "Advanced G-code",
                "Advanced G-code mode is off. Turn it on in Options > Settings > Advanced (not available in Beginner mode).",
            )
            return None
        if isinstance(text, bool):
            text = None
        if text is None:
            prepared = self._prepare_gcode() if self.operations else None
            text = prepared[0] if prepared else ""
        window = AdvancedEditorWindow(self, text)
        self.advanced_editor = window
        window.show()
        return window

    def _machine_and_spindle_stopped(self) -> bool:
        if self.grbl_controller is None or self.grbl_status is None:
            return not self._program_running()
        state = self.grbl_status.state.casefold()
        spindle = self.grbl_status.spindle_rpm or 0.0
        return state in {"idle", "alarm", "hold:0", "door:0", "sleep"} and spindle <= 0.0

    def start_program_stream(self, lines: list[str], label: str, start_line: int = 1, editor=None, start_state=None) -> bool:
        """Stream a program with a full motion-guard pre-check before the first line is sent."""
        if self._program_running():
            QMessageBox.warning(self, label, "A program is already running.")
            return False
        if self.grbl_controller is None or self.grbl_capabilities is None:
            QMessageBox.warning(self, label, "Connect a GRBL controller first.")
            return False
        if not (self.preflight_complete and self.position_verified):
            QMessageBox.warning(self, label, "Finish the preflight check and verify the machine position first.")
            return False
        violations = self.motion_guard.check_program_text("\n".join(lines), start_state or self.motion_guard.start_state())
        if violations:
            self.motion_guard.report(violations, label)
            return False
        dialog = StartDisclaimerDialog(label, f"{label}: {len(lines)} lines from line {start_line}.", self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        program = [l for l in lines]
        self.stream_lines = program
        self.stream_index = 0
        self.stream_source_offset = start_line - 1
        self.stream_waiting = False
        self.program_running = True
        self.streaming_program = True
        self.gcode_progress = {"label": label, "start_line": start_line, "acked": 0, "lines": len(program)}
        self.cycle_estimator = None
        self.terminal_core.log("info", f"{label}: streaming {len(program)} lines.")
        self._stream_next()
        self._set_machine_controls()
        return True

    def _stream_next(self) -> None:
        """Send the next non-empty line (F words scaled beyond 200% when the gate allows)."""
        while self.stream_index < len(self.stream_lines):
            raw = self.stream_lines[self.stream_index]
            code = re.sub(r"\([^)]*\)", "", raw).split(";", 1)[0].strip()
            if code:
                line = scale_feed_line(code, self.stream_feed_scale, machine_max_feed(self.grbl_settings, self.current_machine))
                try:
                    self.grbl_controller.send_line(line)
                except Exception as error:
                    self.terminal_core.log("error", f"Stream stopped: {error}")
                    self._finish_stream(False)
                    return
                self.terminal_core.log("command", f"[{self.stream_source_offset + self.stream_index + 1}] {line}")
                self.stream_waiting = True
                return
            self.stream_index += 1
        self._finish_stream(True)

    def _stream_ack(self, line: str) -> None:
        if not getattr(self, "streaming_program", False) or not getattr(self, "stream_waiting", False):
            return
        if line.lower() == "ok":
            self.stream_waiting = False
            self.stream_index += 1
            self.gcode_progress["acked"] = self.stream_source_offset + self.stream_index
            self._stream_next()
        elif line.lower().startswith(("error", "alarm")):
            self.terminal_core.log("error", f"Stream stopped at line {self.stream_source_offset + self.stream_index + 1}: {line}")
            self._finish_stream(False)

    def _finish_stream(self, completed: bool) -> None:
        self.program_running = False
        self.streaming_program = False
        self.stream_waiting = False
        self._mark_dirty()  # progress is saved with the project for a resume suggestion
        self.terminal_core.log("info" if completed else "warning", "Program finished." if completed else
                               f"Program stopped; resume is suggested from line {self.gcode_progress.get('acked', 0) + 1}.")
        self._set_machine_controls()

    def stop_program_stream(self) -> None:
        if self.grbl_controller is not None:
            self._send_realtime(b"!", "Feed hold (!)")
        if getattr(self, "streaming_program", False):
            self._finish_stream(False)

    def preview_program_segments(self, segments: list, label: str) -> None:
        self.toolpath_player.set_segments(segments)
        self.cycle_estimator = None
        self.render_toolpath_preview()
        self._update_cycle_estimate()
        self.terminal_core.log("info", f"Preview: {label} ({len(segments)} moves).")

    def show_section_setup_sheet(self, program, start: int, end: int) -> str:
        estimate = CycleEstimator(program.segments(start, end), self.current_machine).total(self.speed_multiplier)
        text = self.build_setup_sheet(program, start, end, estimate)
        if hasattr(self, "setup_sheet_preview"):
            self.setup_sheet_preview.setPlainText(text)
        return text

    def resume_program_at(self, program, line: int, editor=None) -> bool:
        """Check the tool, ask about the home location, then stream preamble + remaining lines."""
        info = program.info(line)
        if info is None:
            return False
        estimate = format_duration(CycleEstimator(program.segments(line), self.current_machine).total(self.speed_multiplier))
        dialog = ResumeDialog(line, info.operation, info.tool, estimate, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        if dialog.wants_new_home():
            self.machine_dock.show()
            self.machine_dock.raise_()
            self.terminal_core.log("warning", "Set the new home (Zero work), then choose Resume code here again.")
            return False
        safe_z = float(getattr(self.active_setup, "safe_z_mm", 5.0)) + max((p.position[2] for p in program.lines[:line]), default=0.0)
        preamble = build_resume_preamble(program, line, safe_z, manual_spindle=self.manual_spindle_box.isChecked())
        remaining = [l.text for l in program.lines[line - 1 :]]
        self.terminal_core.log("info", f"Resume at line {line}: preamble {len(preamble)} lines + {len(remaining)} program lines.")
        return self.start_program_stream(preamble + remaining, "Resume code here", start_line=line, editor=editor)

    # ------------------------------------------------------------ keyboard machine control (Block 11)
    def _on_keyboard_control_toggled(self, enabled: bool) -> None:
        self.keyboard_control.reset()
        self.keyboard_prompt.setText("Keyboard control on: viewport or Machine tab must have focus." if enabled else "")
        if enabled:
            self.ghost_tool_pose = tuple(self.tracked_work_pos)
        else:
            self.ghost_tool_pose = None
            self.ghost_path = None
        self._refresh_scene(reset_camera=False)

    def _keyboard_machine_active(self) -> bool:
        if not getattr(self, "keyboard_control_toggle", None) or not self.keyboard_control_toggle.isChecked():
            return False
        if self.keyboard_virtual_box.isChecked():
            return True
        state = self.grbl_status.state.casefold() if self.grbl_status is not None else ""
        return bool(self.grbl_controller is not None and self.grbl_capabilities is not None and self.preflight_complete
                    and self.position_verified and state in {"idle", "jog"})

    def _qt_key_name(self, event) -> str:
        key = event.key()
        names = {
            Qt.Key.Key_Left: "Left", Qt.Key.Key_Right: "Right", Qt.Key.Key_Up: "Up", Qt.Key.Key_Down: "Down",
            Qt.Key.Key_PageUp: "PageUp", Qt.Key.Key_PageDown: "PageDown", Qt.Key.Key_Home: "Home", Qt.Key.Key_End: "End",
            Qt.Key.Key_Escape: "Escape", Qt.Key.Key_Return: "Enter", Qt.Key.Key_Enter: "Enter",
            Qt.Key.Key_Backspace: "Backspace", Qt.Key.Key_Plus: "+", Qt.Key.Key_Minus: "-", Qt.Key.Key_Period: ".",
        }
        return names.get(key, event.text())

    def _handle_machine_key(self, event) -> bool:
        """Route a key to the keyboard state machine; returns True when consumed."""
        name = self._qt_key_name(event)
        if not name:
            return False
        mods = event.modifiers()
        control = self.keyboard_control
        control.inches = self.units.display is LengthUnit.INCHES
        control.jog_feed = self.jog_feed.value()
        control.cut_feed = self.feed.value() if hasattr(self, "feed") else control.cut_feed
        control.spindle_on = bool(self.grbl_status and self.grbl_status.spindle_rpm)
        result = control.handle(
            name, shift=bool(mods & Qt.KeyboardModifier.ShiftModifier), ctrl=bool(mods & Qt.KeyboardModifier.ControlModifier)
        )
        if result.action == "none":
            return name in ("Left", "Right", "Up", "Down", "PageUp", "PageDown", "Enter", "Escape")
        self.keyboard_prompt.setText(result.message)
        if result.action == "preview" and result.request is not None:
            self._preview_keyboard_move(result.request)
        elif result.action == "cancel":
            self.ghost_path = None
            self.ghost_tool_pose = tuple(self.tracked_work_pos)
            self._refresh_scene(reset_camera=False)
        elif result.action == "send" and result.request is not None:
            self._send_keyboard_move(result.request)
        elif result.action == "home":
            if self.keyboard_virtual_box.isChecked():
                self.terminal_core.log("info", "Virtual: home ignored (nothing sent).")
            else:
                self._home_machine()
        elif result.action == "stop":
            if not self.keyboard_virtual_box.isChecked():
                self._send_realtime(b"!", "Feed hold (!) from End key")
            self.ghost_path = None
        return True

    def _preview_keyboard_move(self, request: MoveRequest) -> None:
        start = tuple(self.tracked_work_pos)
        end = list(start)
        end[{"X": 0, "Y": 1, "Z": 2}[request.axis]] += request.distance
        self.ghost_tool_pose = tuple(end)
        self.ghost_path = (start, tuple(end))
        self._refresh_scene(reset_camera=False)

    def _send_keyboard_move(self, request: MoveRequest) -> bool:
        """Guard, confirm, then send (real) or move the ghost (virtual); always logged."""
        violations = self.motion_guard.check_jog(request.axis, request.distance)
        if not self.motion_guard.report(violations, "Keyboard move"):
            self.ghost_path = None
            self.ghost_tool_pose = tuple(self.tracked_work_pos)
            self._refresh_scene(reset_camera=False)
            return False
        label = f"{request.axis}{request.distance:+.3f} mm at F{request.feed:.0f}" + (" (cut)" if request.is_cut else "")
        if self.extra_keyboard_confirm:
            answer = QMessageBox.question(self, "Keyboard move", f"Move {label}?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        virtual = self.keyboard_virtual_box.isChecked()
        if not virtual:
            if self.grbl_controller is None:
                self.terminal_core.log("warning", "No machine connected; nothing was sent.")
                return False
            try:
                self.grbl_controller.jog(request.axis, request.distance, request.feed)
            except Exception as error:
                self.terminal_core.log("error", f"Keyboard move rejected: {error}")
                return False
        self._record_relative_move(request.axis, request.distance)
        self.terminal_core.log("command" if not virtual else "info", ("Virtual: " if virtual else "Keyboard: ") + label)
        self.ghost_tool_pose = tuple(self.tracked_work_pos)
        self.ghost_path = None
        self._refresh_scene(reset_camera=False)
        return True

    def _draw_ghost_tool(self) -> None:
        if self.ghost_tool_pose is None:
            return
        index = self.tool_name.currentIndex() if hasattr(self, "tool_name") else -1
        if not 0 <= index < len(self.tools):
            return
        mesh = create_tool_mesh(self.tools[index]).translate(self.ghost_tool_pose, inplace=False)
        self.plotter.add_mesh(mesh, color=theme_color("highlight"), opacity=0.35, name="ghost-tool", pickable=False)
        if self.ghost_path is not None:
            self.plotter.add_mesh(
                pv.Line(self.ghost_path[0], self.ghost_path[1]), color=theme_color("toolpath_cut"),
                line_width=3, name="ghost-path", pickable=False,
            )

    def _send_realtime(self, command: bytes, description: str) -> None:
        if self.grbl_controller is None:
            return
        try:
            self.grbl_controller.send_realtime(command)
            self.terminal_core.log("command", description)
        except Exception as error:
            self._console_log(f"Controller error: {error}")

    def _home_machine(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Home machine",
            "Homing moves the machine toward its configured switches. Clear the work area and confirm the machine's homing setup is correct.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.homing_in_progress = True
            self.position_verified = False
            self.position_verified_box.setChecked(False)
            self._send_machine_line("$H")
            self._set_machine_controls()

    def _set_position_verified(self, checked: bool) -> None:
        if not checked:
            self.position_verified = False
            self._set_machine_controls()
            return
        if self.grbl_capabilities is None or self.grbl_status is None or self.grbl_status.state.casefold() != "idle":
            self.position_verified_box.blockSignals(True)
            self.position_verified_box.setChecked(False)
            self.position_verified_box.blockSignals(False)
            return
        answer = QMessageBox.question(
            self,
            "Verify machine position",
            "Confirm that the machine is homed or that you have independently verified its current work position.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.position_verified = True
        else:
            self.position_verified_box.blockSignals(True)
            self.position_verified_box.setChecked(False)
            self.position_verified_box.blockSignals(False)
        self._set_machine_controls()

    def _unlock_machine(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Unlock GRBL alarm",
            "$X clears the GRBL alarm lock and can bypass a limit-related lockout. Inspect the machine and clear the fault before continuing.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._send_machine_line("$X")

    def _reset_machine_controller(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Soft reset controller",
            "A soft reset interrupts current motion and may lose the controller's position state. It is not an emergency stop.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.position_verified = False
            self.position_verified_box.setChecked(False)
            self._send_realtime(b"\x18", "Soft reset (Ctrl-X)")

    # ------------------------------------------------------------------
    # Undo / redo (15 steps), layered delete, and part cut/copy/paste
    # ------------------------------------------------------------------
    HISTORY_LIMIT = 50  # replaced per instance from the "history/limit" setting

    def _history_snapshot(self) -> dict:
        def _dc(value):
            try:
                return copy.deepcopy(value)
            except Exception:
                return value

        stock_index = self.parts.index(self.stock_part) if self.stock_part in self.parts else None
        return {
            "parts": [copy.copy(part) for part in self.parts],
            "stock_index": stock_index,
            "stock_imported": self.stock_imported,
            "stock_note": self.stock_note,
            "setups": _dc(self.setups),
            "active_setup_index": self.active_setup_index,
            "sketch_workflow": _dc(self.sketch_workflow),
            "sketch_entities": _dc(self.sketch_entities),
            "active_sketch_id": self.active_sketch_id,
        }

    def _history_restore(self, snap: dict) -> None:
        self.parts = list(snap["parts"])
        index = snap["stock_index"]
        self.stock_part = self.parts[index] if index is not None and index < len(self.parts) else None
        self.stock_mesh = self.stock_part.mesh if self.stock_part is not None else None
        self.stock_imported = snap["stock_imported"]
        self.stock_note = snap["stock_note"]
        self.setups = snap["setups"]
        self.active_setup_index = min(snap["active_setup_index"], max(len(self.setups) - 1, 0))
        self.sketch_workflow = snap["sketch_workflow"]
        self.sketch_entities = snap["sketch_entities"]
        self.active_sketch_id = snap["active_sketch_id"]
        self.selected_cells.clear()
        self.selected_features.clear()
        self.selected_feature = None
        self.stock_status.setText(
            f"Stock · {self.stock_part.name}" if self.stock_part is not None else "No stock geometry"
        )
        has_ops = self._has_any_operations()
        self.generate_action.setEnabled(has_ops)
        self.export_button.setEnabled(has_ops)
        self._refresh_parts()
        self._refresh_note_targets()
        self._refresh_operation_stack()
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)

    def _push_history(self, label: str) -> None:
        """Call BEFORE a user action changes the workspace."""
        if getattr(self, "_history_suspended", False):
            return
        if not hasattr(self, "_undo_stack"):
            self._undo_stack: list[tuple[str, dict]] = []
            self._redo_stack: list[tuple[str, dict]] = []
        self._undo_stack.append((label, self._history_snapshot()))
        if len(self._undo_stack) > self.HISTORY_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._refresh_history_action_states()
        self._mark_dirty()
        if hasattr(self, "session_manager"):
            self.session_manager.record_change(1)
            QTimer.singleShot(0, self._autosave_if_due)  # after the edit has been applied

    def record_workspace_edit(self, label: str, before_snapshot=None) -> None:
        # Legacy hook used by older code paths; the caller has already changed state,
        # so it cannot capture "before". Kept as a no-op to avoid recording a wrong state.
        self._refresh_history_action_states()

    def _refresh_history_action_states(self) -> None:
        undo = getattr(self, "_undo_stack", [])
        redo = getattr(self, "_redo_stack", [])
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(bool(undo))
            self.undo_action.setText(f"Undo {undo[-1][0]}" if undo else "Undo")
        if hasattr(self, "redo_action"):
            self.redo_action.setEnabled(bool(redo))
            self.redo_action.setText(f"Redo {redo[-1][0]}" if redo else "Redo")

    def undo_workspace(self) -> None:
        undo = getattr(self, "_undo_stack", [])
        if not undo:
            self.statusBar().showMessage("Nothing to undo.", 2500)
            return
        label, snap = undo.pop()
        self._redo_stack.append((label, self._history_snapshot()))
        self._history_restore(snap)
        self._refresh_history_action_states()
        self.statusBar().showMessage(f"Undid: {label}", 3500)

    def redo_workspace(self) -> None:
        redo = getattr(self, "_redo_stack", [])
        if not redo:
            self.statusBar().showMessage("Nothing to redo.", 2500)
            return
        label, snap = redo.pop()
        self._undo_stack.append((label, self._history_snapshot()))
        self._history_restore(snap)
        self._refresh_history_action_states()
        self.statusBar().showMessage(f"Redid: {label}", 3500)

    # --- layered delete: one layer per key press ---------------------------------
    def delete_layered(self) -> None:
        """Delete one layer of the current selection per press.

        Order for a sketch with an assigned tool: tool operation, then the sketch, then the
        selected object. A press never removes more than one layer.
        """
        feat = (
            self.sketch_workflow.get_feature(self.active_sketch_id)
            if self.active_sketch_id
            else None
        )
        if feat is not None:
            op_name = getattr(feat, "assigned_operation_name", None)
            if op_name:
                self._delete_operation_by_name(op_name, feat)
                return
            self._delete_sketch_feature(feat)
            return
        if 0 <= self.parts_list.currentRow() < len(self.parts):
            self._push_history(f"Delete {self.parts[self.parts_list.currentRow()].name}")
            self.delete_selected_part()
            return
        self.statusBar().showMessage("Nothing selected to delete.", 2500)

    def _delete_operation_by_name(self, op_name: str, feat=None) -> None:
        for setup in self.setups:
            for index, operation in enumerate(setup.operations):
                if operation.name == op_name:
                    self._push_history(f"Delete tool operation {op_name}")
                    setup.operations.pop(index)
                    for i, op in enumerate(setup.operations, start=1):
                        op.name = f"{i:02d} · {op.kind}"
                    if feat is not None:
                        feat.assigned_operation_name = None
                        for attr in ("assigned_setup_name", "assigned_setup_color"):
                            if hasattr(feat, attr):
                                setattr(feat, attr, None)
                    has_ops = self._has_any_operations()
                    self.generate_action.setEnabled(has_ops)
                    self.export_button.setEnabled(has_ops)
                    self._refresh_operation_stack()
                    self._refresh_sketch_tree()
                    self._refresh_scene(reset_camera=False)
                    self.statusBar().showMessage(
                        f"Removed tool operation {op_name}. Press Delete again to remove the sketch.", 4000
                    )
                    return
        if feat is not None:  # assignment pointed at an operation that no longer exists
            self._push_history("Clear stale tool assignment")
            feat.assigned_operation_name = None
            self._refresh_sketch_tree()

    def _entity_names(self, ids) -> list[str]:
        names = []
        for eid in ids:
            feat = self.sketch_workflow.get_feature(eid)
            names.append(feat.name if feat else eid)
        return names

    def _history_depth(self) -> int:
        return len(getattr(self, "_undo_stack", []))

    def _delete_sketch_feature(self, feat) -> None:
        """Delete with cascade: dependents turn red; one undo step per cascade step."""
        manager = self.sketch_workflow
        deps = dependents_of(manager, feat.feature_id)
        if deps:
            dialog = CascadeDeleteDialog(feat.name, self._entity_names(deps), self)
            dialog.exec()
            if dialog.choice != "continue":
                return
        start_depth = self._history_depth()
        self._push_history(f"Delete sketch {feat.name}")
        broken = delete_entities(self.sketch_workflow, [feat.feature_id])
        self._after_sketch_delete()
        self.statusBar().showMessage(f"Deleted sketch {feat.name}.", 3500)
        step = 1
        while broken:
            self.terminal_core.log("error", "Broken sketch entities: " + ", ".join(self._entity_names(broken)))
            dialog = CascadeStepDialog(self._entity_names(broken), step, self)
            dialog.exec()
            if dialog.choice == "undo_step":
                self.undo_workspace()
                step -= 1
                broken = broken_ids(self.sketch_workflow) if step > 0 else []
            elif dialog.choice == "undo_all":
                while self._history_depth() > start_depth:
                    self.undo_workspace()
                broken = []
            elif dialog.choice == "continue":
                self._push_history("Delete broken entities")
                broken = delete_entities(self.sketch_workflow, broken)
                self._after_sketch_delete()
                step += 1
            else:  # fix by hand: entities stay red; saving and generation are refused until fixed
                break

    def _after_sketch_delete(self) -> None:
        features = self.sketch_workflow.features
        self.active_sketch_id = features[-1].feature_id if features else None
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)

    def _check_assignment_memory(self) -> None:
        """When a tooled sketch was changed and is fully defined again, ask whether to keep its tools."""
        if getattr(self, "_assignment_check_busy", False) or not hasattr(self, "assignment_ledger"):
            return
        ledger = self.assignment_ledger
        ask = ledger.observe(self.sketch_workflow.features, self.setups)
        if not ask:
            return
        self._assignment_check_busy = True
        try:
            for fid in ask:
                feat = self.sketch_workflow.get_feature(fid)
                ops = ledger.operations.get(fid, [])
                if feat is None or not ops:
                    ledger.forget(fid)
                    continue
                dialog = RetainAssignmentsDialog(feat.name, ops, self)
                dialog.exec()
                if dialog.choice == "no":
                    self._push_history("Remove changed sketch assignments")
                    for setup in self.setups:
                        setup.operations = [op for op in setup.operations if op.target not in (feat.name, feat.feature_id)]
                    ledger.forget(fid)
                    self._refresh_operation_stack_only()
                elif dialog.choice == "check":
                    self._walk_check_assignments([feat.name, feat.feature_id])
                    ledger.confirm(feat, self.setups)
                else:
                    ledger.confirm(feat, self.setups)
        finally:
            self._assignment_check_busy = False

    def _walk_check_assignments(self, target_ids: list[str]) -> None:
        """Confirm each operation from the first assignment of this sketch downward."""
        for s_idx, o_idx in check_assignments_walk(target_ids, self.setups):
            op = self.setups[s_idx].operations[o_idx]
            answer = QMessageBox.question(
                self, "Check assignments",
                f"{op.name}\nTool: {op.tool} · Ø{op.tool_diameter_mm:g} mm\nTarget: {op.target}\n\nIs this tool still right?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._jump_to_operation((s_idx, o_idx))
                self.terminal_core.log("warning", f"Check assignments stopped at {op.name}: change its tool, then continue.")
                return
        self.terminal_core.log("info", "All tool assignments for the changed sketch were confirmed.")

    def _refresh_operation_stack_only(self) -> None:
        busy = getattr(self, "_assignment_check_busy", False)
        self._assignment_check_busy = True
        try:
            self._refresh_operation_stack()
        finally:
            self._assignment_check_busy = busy

    def _set_future_operation(self, future: bool) -> None:
        ids = self._selected_entity_ids()
        feats = [f for f in (self.sketch_workflow.get_feature(i) for i in ids) if f is not None]
        if not feats:
            return
        self._push_history("Future operation" if future else "Clear future operation")
        for feat in feats:
            feat.status = "future" if future else ""
        self._refresh_sketch_tree()

    def _solve_sketches(self) -> dict:
        """Re-solve every sketch after an edit; keeps the last good state of each sketch."""
        manager = self.sketch_workflow
        if getattr(manager, "last_good", None) is None:
            manager.last_good = LastGoodStateStore()
        return solve_all(manager, manager.last_good)

    def _revert_sketch_errors(self) -> bool:
        """Esc in the error state: revert each broken sketch to its last good state (or offer to delete it)."""
        manager = self.sketch_workflow
        bad_owners = [owner for owner, feats in sketch_groups(manager).items() if any(f.has_error for f in feats)]
        if not bad_owners:
            return False
        self._push_history("Revert sketch error")
        for owner in bad_owners:
            if manager.last_good is not None and manager.last_good.has(owner):
                manager.last_good.revert(manager, owner)
                self.terminal_core.log("info", f"Reverted {manager.sketches.get(owner, {}).get('name', owner)} to its last good state.")
            else:
                name = manager.sketches.get(owner, {}).get("name", owner)
                dialog = DeleteBadSketchDialog(name, self)
                dialog.exec()
                if dialog.choice == "delete":
                    ids = [f.feature_id for f in manager.features if (f.owner_sketch_id or f"_solo_{f.feature_id}") == owner]
                    delete_entities(manager, ids)
        self._after_sketch_delete()
        return True

    # --- cut / copy / paste -----------------------------------------------------
    def _focus_text_widget(self):
        widget = QApplication.focusWidget()
        if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return widget
        return None

    def _selected_part_indexes(self) -> list[int]:
        rows = sorted({self.parts_list.row(item) for item in self.parts_list.selectedItems()})
        return [row for row in rows if 0 <= row < len(self.parts)]

    def copy_selection(self) -> None:
        text_widget = self._focus_text_widget()
        if text_widget is not None:
            text_widget.copy()
            return
        indexes = self._selected_part_indexes()
        if not indexes:
            self.statusBar().showMessage("Select a part to copy.", 2500)
            return
        self._part_clipboard = [self._clone_part(self.parts[i]) for i in indexes]
        self.statusBar().showMessage(f"Copied {len(indexes)} part(s).", 2500)

    def cut_selection(self) -> None:
        text_widget = self._focus_text_widget()
        if text_widget is not None:
            text_widget.cut()
            return
        indexes = self._selected_part_indexes()
        if not indexes:
            return
        self._part_clipboard = [self._clone_part(self.parts[i]) for i in indexes]
        self._push_history(f"Cut {len(indexes)} part(s)")
        self._history_suspended = True
        try:
            for index in sorted(indexes, reverse=True):
                self.parts_list.setCurrentRow(index)
                self.delete_selected_part()
        finally:
            self._history_suspended = False
        self.statusBar().showMessage(f"Cut {len(indexes)} part(s).", 2500)

    def paste_selection(self) -> None:
        text_widget = self._focus_text_widget()
        if text_widget is not None:
            if hasattr(text_widget, "paste"):
                text_widget.paste()
            return
        clipboard = getattr(self, "_part_clipboard", [])
        if not clipboard:
            self.statusBar().showMessage("Nothing to paste.", 2500)
            return
        self._push_history(f"Paste {len(clipboard)} part(s)")
        for template in clipboard:
            part = self._clone_part(template)
            existing = {p.name for p in self.parts}
            base = template.name.removesuffix(" (copy)")
            part.name = f"{base} (copy)"
            number = 2
            while part.name in existing:
                part.name = f"{base} (copy {number})"
                number += 1
            if part.role is PartRole.STOCK:
                part.role = PartRole.TARGET
            self.parts.append(part)
        self._refresh_parts()
        self.parts_list.setCurrentRow(len(self.parts) - 1)
        self._refresh_note_targets()
        self._refresh_scene(reset_camera=False)
        self.statusBar().showMessage(f"Pasted {len(clipboard)} part(s).", 2500)

    @staticmethod
    def _clone_part(part):
        clone = copy.copy(part)
        for attr in ("mesh", "edge_mesh"):
            value = getattr(part, attr, None)
            if value is not None:
                setattr(clone, attr, value.copy(deep=True))
        return clone

    # ------------------------------------------------------------------
    # Kill sequence: stop X/Y motion, retract Z, stop a GRBL-controlled spindle.
    # Software stop only: NOT a substitute for the machine's physical E-stop.
    # ------------------------------------------------------------------
    def _kill_machine(self) -> None:
        """Immediate (no confirmation) multi-step stop.

        1. Feed hold (!) decelerates every axis to a stop, so X and Y stop moving.
        2. Soft reset (Ctrl-X) flushes queued motion and switches the spindle output off.
        3. If the controller is left in an alarm lock that was caused by the reset itself
           (and no limit switch is active), it is unlocked ($X) so Z can move.
        4. A GRBL-controlled spindle is sent M5 (skipped when "Manual spindle speed" or
           "Spindle already running" is checked: that spindle is run by hand).
        5. Z jogs up by KILL_Z_RETRACT_MM, or less if the machine's Z stop is nearer
           (travel is only checked when the position was verified before the kill).
        """
        if self.grbl_controller is None:
            return
        if getattr(self, "_kill_stage", None) is not None:
            self._send_realtime(b"!", "KILL (repeat): feed hold")
            return
        try:
            self.grbl_controller.send_realtime(b"!")
            self._console_log("> KILL: feed hold (!)")
        except Exception as error:
            self._console_log(f"Kill command error (feed hold): {error}")
            # Still try the reset below so motion is flushed even if the hold failed.
        self._kill_was_verified = bool(self.position_verified)
        self._kill_manual_spindle = bool(
            getattr(self, "manual_spindle_box", None) is not None and self.manual_spindle_box.isChecked()
            or getattr(self, "manual_cut_spindle_running_box", None) is not None
            and self.manual_cut_spindle_running_box.isChecked()
        )
        self._kill_mpos_z = None
        self._kill_stage = "hold"
        self._kill_deadline = time.monotonic() + KILL_HOLD_TIMEOUT_S
        self._kill_stage_started = time.monotonic()
        self.position_verified = False
        self.homing_in_progress = False
        self.position_verified_box.blockSignals(True)
        self.position_verified_box.setChecked(False)
        self.position_verified_box.blockSignals(False)
        self.statusBar().showMessage(
            "KILL: stopping motion. Software stop only: use the physical E-stop in an emergency.", 10000
        )
        self._set_machine_controls()
        QTimer.singleShot(KILL_POLL_MS, self._kill_step)

    def _kill_finish(self, message: str) -> None:
        self._kill_stage = None
        self._console_log(f"KILL: {message}")
        self.statusBar().showMessage(f"KILL: {message}", 15000)
        self._set_machine_controls()

    def _kill_state(self) -> str:
        return self.grbl_status.state.casefold() if self.grbl_status is not None else ""

    def _kill_step(self) -> None:
        stage = getattr(self, "_kill_stage", None)
        if stage is None:
            return
        controller = self.grbl_controller
        if controller is None:
            self._kill_finish("controller disconnected during kill sequence.")
            return
        now = time.monotonic()
        state = self._kill_state()
        try:
            if stage == "hold":
                if state.startswith("hold") or state in {"idle"} or state.startswith("alarm") or now > self._kill_deadline:
                    if self.grbl_status is not None and self.grbl_status.machine_position:
                        self._kill_mpos_z = float(self.grbl_status.machine_position[2])
                    controller.send_realtime(b"\x18")
                    self._console_log("> KILL: soft reset (Ctrl-X)")
                    self._kill_stage = "reset"
                    self._kill_stage_started = now
                    self._kill_deadline = now + KILL_RESET_TIMEOUT_S
            elif stage == "reset":
                # Give the controller time to reboot and report a fresh status.
                if now - self._kill_stage_started >= KILL_RESET_SETTLE_S:
                    if state == "idle":
                        self._kill_stage = "retract"
                    elif state.startswith("alarm"):
                        if self._kill_unlock_allowed():
                            controller.send_line("$X")
                            self._console_log("> KILL: $X (alarm caused by the reset)")
                            self._kill_stage = "unlock"
                            self._kill_stage_started = now
                            self._kill_deadline = now + KILL_UNLOCK_TIMEOUT_S
                        else:
                            self._kill_finish(
                                "stopped, but Z was NOT retracted: controller is in an alarm that was not "
                                "caused by the reset (or a limit switch is active). Clear the fault, then jog Z up."
                            )
                            return
                if self._kill_stage == "reset" and now > self._kill_deadline:
                    self._kill_finish("stopped, but the controller did not report a status after reset. Z NOT retracted.")
                    return
            elif stage == "unlock":
                if state == "idle":
                    self._kill_stage = "retract"
                elif now > self._kill_deadline:
                    self._kill_finish("stopped, but the alarm did not clear. Z NOT retracted.")
                    return
            if self._kill_stage == "retract":
                self._kill_do_retract(controller)
                return
            if self._kill_stage == "retract_wait":
                if (state == "idle" and now - self._kill_stage_started > 0.4) or now > self._kill_deadline:
                    self._kill_finish(self._kill_summary)
                    return
        except Exception as error:
            self._kill_finish(f"error during kill sequence: {error}")
            return
        QTimer.singleShot(KILL_POLL_MS, self._kill_step)

    def _kill_unlock_allowed(self) -> bool:
        """Only auto-unlock an alarm that the reset itself caused, and never with a limit switch active."""
        pins = (self.grbl_status.pins or "") if self.grbl_status is not None else ""
        if any(axis in pins.upper() for axis in "XYZ"):
            return False
        codes = re.findall(r"ALARM:(\d+)", "\n".join(self.grbl_identification_lines[-20:]), flags=re.IGNORECASE)
        if codes and int(codes[-1]) not in KILL_UNLOCKABLE_ALARMS:
            return False
        return True

    def _kill_do_retract(self, controller) -> None:
        notes: list[str] = []
        if not self._kill_manual_spindle:
            controller.send_line("M5")
            self._console_log("> KILL: M5 (spindle stop)")
            notes.append("spindle stopped")
        else:
            notes.append("manual spindle left to you")
        distance = KILL_Z_RETRACT_MM
        if self._kill_was_verified and self._kill_mpos_z is not None:
            z_top = self.current_machine.z_top() if self.current_machine is not None else KILL_Z_MACHINE_TOP_MM
            headroom = z_top - self._kill_mpos_z - KILL_Z_STOP_MARGIN_MM
            distance = max(0.0, min(KILL_Z_RETRACT_MM, headroom))
            if distance < 0.1:
                self._kill_summary = "X/Y stopped; Z already at its stop; " + "; ".join(notes) + "."
                self._kill_stage = "retract_wait"
                self._kill_stage_started = time.monotonic()
                self._kill_deadline = time.monotonic()
                QTimer.singleShot(KILL_POLL_MS, self._kill_step)
                return
        else:
            notes.append("position unverified: Z retracted without a travel check")
        controller.jog("Z", distance, KILL_Z_FEED_MM_MIN)
        self._record_relative_move("Z", distance)
        self._console_log(f"> KILL: $J=G91 G21 Z{distance:.3f} F{KILL_Z_FEED_MM_MIN:.0f}")
        self._kill_summary = f"X/Y stopped; Z retracted {distance:.1f} mm; " + "; ".join(notes) + "."
        self._kill_stage = "retract_wait"
        self._kill_stage_started = time.monotonic()
        self._kill_deadline = time.monotonic() + distance / KILL_Z_FEED_MM_MIN * 60.0 + 3.0
        QTimer.singleShot(KILL_POLL_MS, self._kill_step)

    def _start_spindle(self) -> None:
        if self.grbl_status is None or self.grbl_status.state.casefold() != "idle":
            return
        self._send_machine_line(f"M3 S{self.spindle_rpm.value()}")

    def _show_key_bindings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Key bindings")
        form = QFormLayout(dialog)
        editors: dict[str, QKeySequenceEdit] = {}
        for name, action in self.command_actions.items():
            editor = QKeySequenceEdit(action.shortcut(), dialog)
            editors[name] = editor
            form.addRow(action.text(), editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for name, editor in editors.items():
                sequence = editor.keySequence()
                self.command_actions[name].setShortcut(sequence)
                self.settings.setValue(
                    f"keybinds/{name}",
                    sequence.toString(QKeySequence.SequenceFormat.PortableText),
                )

    def _show_preferences(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Preferences")
        form = QFormLayout(dialog)
        display_mode = QComboBox()
        display_mode.addItems(
            list(DISPLAY_MODES)
        )
        display_mode.setCurrentText(self.display_mode.currentText())
        form.addRow("Default display", display_mode)
        sensitivity = QDoubleSpinBox()
        sensitivity.setRange(0.1, 2.0)
        sensitivity.setSingleStep(0.05)
        sensitivity.setValue(self.orbit_sensitivity)
        form.addRow("Orbit sensitivity", sensitivity)
        parallel = QCheckBox("Orthographic projection")
        parallel.setChecked(bool(self.plotter.camera.GetParallelProjection()))
        form.addRow("Camera", parallel)
        history_limit = QSpinBox()
        history_limit.setRange(5, 500)
        history_limit.setValue(int(self.HISTORY_LIMIT))
        history_limit.setToolTip("Number of undo steps kept; applies immediately.")
        form.addRow("Undo history", history_limit)
        beginner = QCheckBox("Beginner mode")
        beginner.setChecked(self.safety.beginner_mode)
        beginner.setToolTip("All warnings on, safety overrides off and advanced features hidden.")
        form.addRow("Experience", beginner)
        grid_font = QSpinBox()
        grid_font.setRange(6, 48)
        grid_font.setSuffix(" px")
        grid_font.setValue(self._grid_font_px())
        grid_font.setToolTip("On-screen size of the grid numbers; stays the same when zooming.")
        form.addRow("Interface font · grid numbers", grid_font)
        dwg_row = QHBoxLayout()
        dwg_path = QLineEdit(str(self.settings.value("import/dwg_converter", "") or ""))
        dwg_path.setPlaceholderText("ODAFileConverter.exe")
        dwg_path.setToolTip("The free ODA File Converter, needed to import DWG files.")
        dwg_browse = QPushButton("Browse...")
        dwg_browse.clicked.connect(
            lambda: dwg_path.setText(QFileDialog.getOpenFileName(dialog, "ODA File Converter", "", "Programs (*.exe);;All files (*)")[0] or dwg_path.text())
        )
        dwg_row.addWidget(dwg_path, 1)
        dwg_row.addWidget(dwg_browse)
        form.addRow("DWG converter", dwg_row)
        colour_group, pending_colours = self._build_colour_overrides_group(dialog)
        form.addRow(colour_group)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._set_history_limit(history_limit.value())
            self.settings.setValue("interface/grid_font_px", grid_font.value())
            self.settings.setValue("import/dwg_converter", dwg_path.text().strip())
            set_dwg_converter(dwg_path.text().strip())
            for token, value in pending_colours.items():
                self.settings.setValue(f"colors/{token}", value)
            self._apply_theme()
            if beginner.isChecked() != self.safety.beginner_mode:
                self._set_beginner_mode(beginner.isChecked())
            self.display_mode.setCurrentText(display_mode.currentText())
            self.orbit_sensitivity = sensitivity.value()
            self.settings.setValue("view/orbit_sensitivity", self.orbit_sensitivity)
            self.plotter.camera.SetParallelProjection(parallel.isChecked())
            self.settings.setValue("view/parallel_projection", parallel.isChecked())
            self.plotter.render()

    OVERRIDABLE_COLOURS = (
        ("face", "Part faces"), ("edge", "Part edges"), ("highlight", "Highlight"), ("selection", "Selection"),
        ("toolpath_cut", "Cutting moves"), ("toolpath_rapid", "Rapid moves"), ("grid", "Grid"),
        ("viewport_bg", "Viewport background"), ("accent", "Accent"), ("warning", "Warning"), ("error", "Error"),
    )

    def _build_colour_overrides_group(self, parent):
        """Colour override buttons; returns (group, pending dict filled on change). Red only for Error."""
        group = QGroupBox("Colours")
        grid = QGridLayout(group)
        pending: dict[str, str] = {}

        def paint(button, value):
            button.setText(value)
            button.setStyleSheet(f"background: {value}; color: #101010;")

        def choose(token, button):
            current = QColor(pending.get(token, theme_color(token)))
            chosen = QColorDialog.getColor(current, parent, "Choose colour")
            if not chosen.isValid():
                return
            value = chosen.name()
            if token != "error" and is_red(value):
                QMessageBox.information(parent, "Colours", "Red is reserved for errors; choose another colour.")
                return
            pending[token] = value
            paint(button, value)

        for row, (token, title) in enumerate(self.OVERRIDABLE_COLOURS):
            grid.addWidget(QLabel(title), row // 2, (row % 2) * 2)
            button = QPushButton()
            paint(button, theme_color(token))
            button.clicked.connect(lambda _=False, t=token, b=button: choose(t, b))
            grid.addWidget(button, row // 2, (row % 2) * 2 + 1)
        reset = QPushButton("Reset colours")
        reset.setToolTip("Return every colour to the default palette when you press Save.")

        def reset_all():
            for token, _title in self.OVERRIDABLE_COLOURS:
                pending[token] = ""

        reset.clicked.connect(reset_all)
        grid.addWidget(reset, (len(self.OVERRIDABLE_COLOURS) + 1) // 2, 0, 1, 4)
        return group, pending

    def _set_history_limit(self, limit: int) -> None:
        self.HISTORY_LIMIT = max(1, int(limit))
        self.settings.setValue("history/limit", self.HISTORY_LIMIT)
        self.history_manager.max_steps = self.HISTORY_LIMIT
        stack = getattr(self, "_undo_stack", None)
        if stack is not None and len(stack) > self.HISTORY_LIMIT:
            del stack[: len(stack) - self.HISTORY_LIMIT]
        self._refresh_history_action_states()

    def _set_beginner_mode(self, enabled: bool, confirm: bool = True) -> bool:
        """Turn Beginner mode on or off; turning it off asks for confirmation."""
        if not enabled and confirm:
            answer = QMessageBox.question(
                self,
                "Turn off Beginner mode",
                "Turning off Beginner mode shows Settings > Advanced and advanced features. "
                "Safety overrides stay off until you enable them. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self.safety.set_beginner_mode(enabled)
        if hasattr(self, "advanced_settings_action"):
            self.advanced_settings_action.setVisible(not enabled)
        if hasattr(self, "keyboard_confirm_action"):
            self.keyboard_confirm_action.setEnabled(not enabled)
            self.keyboard_confirm_action.setChecked(self.extra_keyboard_confirm)
        return True

    @property
    def extra_keyboard_confirm(self) -> bool:
        """Extra confirmation for keyboard operations; forced on in Beginner mode."""
        if self.safety.beginner_mode:
            return True
        return bool(self.settings.value("ui/extra_keyboard_confirm", True, type=bool))

    def _set_extra_keyboard_confirm(self, enabled: bool) -> None:
        self.settings.setValue("ui/extra_keyboard_confirm", bool(enabled))

    def _show_advanced_settings(self) -> None:
        if self.safety.beginner_mode:
            QMessageBox.information(self, "Settings > Advanced", "Not available in Beginner mode.")
            return
        AdvancedSettingsDialog(self.safety, self).exec()
        self._refresh_advanced_gcode_action()

    def _show_job_information(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Job information")
        form = QFormLayout(dialog)
        name = QLineEdit(self.project_name_label.text())
        notes = QTextEdit(str(self.settings.value("job/notes", "")))
        notes.setFixedHeight(120)
        form.addRow("Job name", name)
        form.addRow("Job notes", notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.project_name_label.setText(name.text().strip() or "Untitled job")
            self.settings.setValue("job/name", self.project_name_label.text())
            self.settings.setValue("job/notes", notes.toPlainText())

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 12, 18, 12)
        brand = QLabel("BasiCAM CNC")
        brand.setObjectName("brand")
        subtitle = QLabel("PART SETUP  /  TOOL PLANNING")
        subtitle.setObjectName("eyebrow")
        brand_col = QVBoxLayout()
        brand_col.setSpacing(2)
        brand_col.addWidget(brand)
        brand_col.addWidget(subtitle)
        layout.addLayout(brand_col)
        layout.addStretch(1)
        # Connection state is still tracked here but no longer shown; the Machine dock has it.
        self.machine_badge = QLabel("")
        self.machine_badge.hide()
        self.machine_state_indicator = QLabel("Disconnected / simulation only")
        self.machine_state_indicator.setObjectName("machineStateIndicator")
        self.machine_state_indicator.setToolTip("Machine connection state; open Window > Machine & Console for details.")
        layout.addWidget(self.machine_state_indicator)
        layout.addSpacing(12)
        open_button = QPushButton("Open part")
        open_button.setObjectName("primaryButton")
        open_button.clicked.connect(self.open_part)
        layout.addWidget(open_button)
        tools_button = QPushButton("Tool inventory")
        tools_button.clicked.connect(self.open_tool_library)
        layout.addWidget(tools_button)
        donate_button = QPushButton("Donate")
        donate_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://paypal.me/sharxbyte"))
        )
        layout.addWidget(donate_button)
        return header

    def _build_project_panel(self) -> QWidget:
        panel = self._panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self._section_label("PROJECT"))
        self.project_name_label = QLabel(str(self.settings.value("job/name", "Untitled job")))
        self.project_name_label.setObjectName("panelTitle")
        layout.addWidget(self.project_name_label)

        layout.addWidget(self._section_label("PARTS"))
        self.parts_list = QListWidget()
        self.parts_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.parts_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.parts_list.customContextMenuRequested.connect(self._show_parts_context_menu)
        self.parts_list.currentRowChanged.connect(self._select_part_row)
        layout.addWidget(self.parts_list, 1)
        add_part = QPushButton("＋  Import part")
        add_part.setVisible(False)  # use File > Import part
        add_part.clicked.connect(self.import_part)
        layout.addWidget(add_part)

        layout.addWidget(self._divider())
        layout.addWidget(self._section_label("STOCK ENVELOPE · MM"))
        stock_form = QFormLayout()
        stock_form.setContentsMargins(0, 0, 0, 0)
        stock_form.setHorizontalSpacing(8)
        stock_form.setVerticalSpacing(8)
        self.stock_dimensions: dict[str, QDoubleSpinBox] = {}
        for axis, value in (("X", 100), ("Y", 80), ("Z", 20)):
            field = self._double_spin(value, 0.1, 2000, 1)
            self.stock_dimensions[axis] = field
            stock_form.addRow(f"{axis} size", field)
        layout.addLayout(stock_form)
        self.stock_material = QComboBox()
        self.stock_material.addItems(
            [
                "Unspecified",
                "Wood (soft)",
                "Wood (hard)",
                "Plastic",
                "Aluminum",
                "Brass",
                "Bronze",
                "Mild steel",
                "Stainless steel",
                "Other",
            ]
        )
        self.stock_material.currentTextChanged.connect(self._apply_recommended_defaults)
        layout.addWidget(self.stock_material)
        stock_actions = QHBoxLayout()
        create_stock = QPushButton("Create box")
        create_stock.clicked.connect(self.create_stock_box)
        stock_actions.addWidget(create_stock)
        import_stock = QPushButton("Import billet")
        import_stock.clicked.connect(self.open_stock)
        stock_actions.addWidget(import_stock)
        layout.addLayout(stock_actions)
        self.stock_status = QLabel("No stock geometry")
        self.stock_status.setObjectName("mutedText")
        self.stock_status.setWordWrap(True)
        layout.addWidget(self.stock_status)

        layout.addStretch(1)
        return panel

    def _build_view_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setObjectName("subToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)
        view_label = QLabel("MODEL VIEW")
        view_label.setObjectName("eyebrow")
        toolbar_layout.addWidget(view_label)
        toolbar_layout.addStretch(1)
        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self._fit_view)
        toolbar_layout.addWidget(fit_button)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Isometric", "Front", "Back", "Left", "Right", "Top", "Bottom"])
        self.view_combo.currentTextChanged.connect(self.set_standard_view)
        toolbar_layout.addWidget(self.view_combo)
        self.display_mode = QComboBox()
        self.display_mode.addItems(
            list(DISPLAY_MODES)
        )
        self.display_mode.currentTextChanged.connect(self._on_display_mode_changed)
        self.lighting_button = QPushButton("Lighting…")
        self.lighting_button.setToolTip("Adjust light brightness, contrast and direction to make faces easier to tell apart.")
        self.lighting_button.clicked.connect(self._show_lighting_panel)
        toolbar_layout.addWidget(self.lighting_button)
        toolbar_layout.addWidget(self.display_mode)
        self.grid_toggle = QPushButton("Grid")
        self.grid_toggle.setCheckable(True)
        self.grid_toggle.setChecked(True)
        self.grid_toggle.toggled.connect(self._toggle_grid)
        toolbar_layout.addWidget(self.grid_toggle)
        self.through_toggle = QPushButton("Select through")
        self.through_toggle.setCheckable(True)
        self.through_toggle.setToolTip(
            "When enabled, box selection includes geometry behind the visible surface."
        )
        self.through_toggle.setEnabled(False)
        toolbar_layout.addWidget(self.through_toggle)
        layout.addWidget(toolbar)

        self.plotter = QtInteractor(panel)
        self.plotter.set_background(theme_color("viewport_bg"))
        self.plotter.show_grid(color=theme_color("grid"), xtitle="X", ytitle="Y", ztitle="Z")
        self.plotter.interactor.SetInteractorStyle(vtkInteractorStyleUser())
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.installEventFilter(self)
        self.cell_picker = vtkCellPicker()
        self.cell_picker.SetTolerance(0.003)
        layout.addWidget(self.plotter.interactor, 1)
        self.normal_button = QPushButton("Normal", self.plotter.interactor)
        self.normal_button.setToolTip("Look straight at the selected face or curve (N)")
        self.normal_button.clicked.connect(self.view_normal_to_selected)
        self.orient_button = QPushButton("Axes", self.plotter.interactor)
        self.orient_button.setToolTip("Assign the work origin and axes from a picked point and edge (Position tab).")
        self.orient_button.clicked.connect(self._show_axes_assignment)
        for overlay_button in (self.normal_button, self.orient_button):
            overlay_button.setFixedSize(58, 24)
            overlay_button.setStyleSheet(
                f"QPushButton{{background:{COLORS['panel_alt']};color:{COLORS['text']};"
                f"border:1px solid {COLORS['line']};border-radius:4px;padding:0;font-size:11px}}"
                f"QPushButton:hover{{border-color:{COLORS['accent']}}}"
            )
            overlay_button.show()
        self.orientation_widget = vtkCameraOrientationWidget()
        self.orientation_widget.SetParentRenderer(self.plotter.renderer)
        self.orientation_widget.SetInteractor(self.plotter.render_window.GetInteractor())
        self.orientation_widget.AnimateOff()
        self.orientation_widget.SetEnabled(True)

        footer = QFrame()
        footer.setObjectName("viewFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 9, 12, 9)
        footer_layout.setSpacing(8)
        self.measure_button = QPushButton("Measure")
        self.measure_button.clicked.connect(self.start_measurement)
        footer_layout.addWidget(self.measure_button)
        self.measure_result = QLabel("Select Measure, then pick two points")
        self.measure_result.setObjectName("mutedText")
        footer_layout.addWidget(self.measure_result, 1)
        footer_layout.addWidget(QLabel("Display"))
        self.display_button = QPushButton("mm")
        self.display_button.setCheckable(True)
        self.display_button.setFixedWidth(58)
        self.display_button.setToolTip("Changes labels only; model size is unchanged.")
        self.display_button.clicked.connect(self._toggle_display_unit)
        footer_layout.addWidget(self.display_button)
        footer_layout.addWidget(QLabel("Import as"))
        self.import_button = QPushButton("mm")
        self.import_button.setCheckable(True)
        self.import_button.setFixedWidth(58)
        self.import_button.setToolTip(
            "Reinterprets imported coordinates and scales model geometry."
        )
        self.import_button.clicked.connect(self._toggle_import_unit)
        footer_layout.addWidget(self.import_button)
        layout.addWidget(footer)
        return panel

    # ------------------------------------------------------------------
    # CAM workflow tabs (Sketch, Operations, Notes, Preview & G-code)
    # ------------------------------------------------------------------
    @staticmethod
    def _scroll_wrap(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _install_workflow_tabs(self) -> None:
        tabs = self.cam_tabs
        tabs.insertTab(0, self._scroll_wrap(self._build_sketch_tab()), "Sketch")
        tabs.insertTab(1, self._scroll_wrap(self._build_operation_panel()), "Operations")
        # existing order is now: Sketch, Operations, Tools & Speeds, Fixtures, Sim & Post, Setup & Export
        tabs.insertTab(3, self._build_notes_tab(), "Notes")
        tabs.insertTab(5, self._scroll_wrap(self._build_preview_gcode_tab()), "Preview & G-code")
        tabs.insertTab(1, self._scroll_wrap(self._build_position_tab()), "Position")
        tabs.setCurrentIndex(0)
        self.cam_tabs_dock.setMinimumWidth(380)
        self.resizeDocks([self.cam_tabs_dock], [430], Qt.Orientation.Horizontal)
        self._refresh_notes_list()
        self._update_sketch_tab_state(False)

    def _position_view_overlay(self) -> None:
        """Place Normal / Orient just under the VTK orientation cube (top right)."""
        if not hasattr(self, "orient_button") or not hasattr(self, "orientation_widget"):
            return
        interactor = self.plotter.interactor
        try:
            rep = self.orientation_widget.GetRepresentation()
            size = rep.GetSize()[0]
            pad = rep.GetPadding()
            dpr = interactor.devicePixelRatioF() or 1.0
            cube = int(size / dpr)
            pad_x, pad_y = int(pad[0] / dpr), int(pad[1] / dpr)
        except Exception:
            cube, pad_x, pad_y = 120, 10, 10
        left = interactor.width() - pad_x - cube
        top = pad_y + cube + 4
        self.normal_button.move(left, top)
        self.orient_button.move(left + 62, top)
        self.normal_button.raise_()
        self.orient_button.raise_()

    # --- Sketch tab -------------------------------------------------------------------
    SKETCH_TOOLS = (
        ("Point", "Point", ("Click",)),
        ("Line", "Line", ("Two point",)),
        ("Rectangle", "Rectangle", ("Corner", "Center", "3 point")),
        ("Circle", "Circle", ("Center", "3 point")),
        ("Curve", "Curve", ("3 point arc", "Center arc")),
        ("Spline", "Spline", ("Fit points",)),
    )
    SNAP_OPTIONS = ("Grid", "Endpoint", "Midpoint", "Center", "Intersection", "On surface")

    def _build_sketch_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._section_label("SKETCH"))
        session_row = QHBoxLayout()
        self.sketch_mode = QPushButton("New sketch")
        self.sketch_mode.setObjectName("primaryButton")
        self.sketch_mode.setCheckable(True)
        self.sketch_mode.setToolTip("Start a sketch, then pick a plane or select a face first.")
        self.sketch_mode.toggled.connect(self._toggle_sketch_mode)
        self.sketch_mode.toggled.connect(self._update_sketch_tab_state)
        session_row.addWidget(self.sketch_mode, 1)
        self.clear_sketch_btn = QPushButton("Clear sketch")
        self.clear_sketch_btn.setToolTip("Remove every entity in the sketch being edited")
        self.clear_sketch_btn.clicked.connect(self.clear_sketch)
        self.clear_sketch_btn.setVisible(False)  # only while a sketch is being edited
        session_row.addWidget(self.clear_sketch_btn)
        layout.addLayout(session_row)

        plane_row = QHBoxLayout()
        plane_row.addWidget(QLabel("Plane"))
        self.work_plane = QComboBox()
        self.work_plane.addItems(["XY plane", "XZ plane", "YZ plane"])
        self.work_plane.currentTextChanged.connect(self.set_work_plane)
        plane_row.addWidget(self.work_plane, 1)
        self.use_face_button = QPushButton("Use selected face")
        self.use_face_button.setToolTip("Lock the sketch to the face or curved surface selected in the viewer")
        self.use_face_button.clicked.connect(self._sketch_use_selected_face)
        plane_row.addWidget(self.use_face_button)
        layout.addLayout(plane_row)
        self.sketch_status = QLabel("Click New sketch, then pick a plane or select a face.")
        self.sketch_status.setObjectName("mutedText")
        self.sketch_status.setWordWrap(True)
        layout.addWidget(self.sketch_status)

        # ---- drawing tools (enabled only while a sketch is active)
        self.sketch_kind = QComboBox()
        self.sketch_kind.addItems([tool[1] for tool in self.SKETCH_TOOLS] + ["Construction line"])
        self.sketch_kind.hide()
        self.sketch_tools_box = QGroupBox("Draw")
        tools_layout = QVBoxLayout(self.sketch_tools_box)
        grid = QGridLayout()
        grid.setSpacing(6)
        self.sketch_tool_group = QButtonGroup(self)
        self.sketch_tool_group.setExclusive(True)
        self.sketch_tool_buttons: dict[str, QPushButton] = {}
        for index, (label, kind, _variants) in enumerate(self.SKETCH_TOOLS):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, k=kind: self._select_sketch_tool(k))
            self.sketch_tool_group.addButton(button)
            self.sketch_tool_buttons[kind] = button
            grid.addWidget(button, index // 3, index % 3)
        tools_layout.addLayout(grid)
        variant_row = QHBoxLayout()
        variant_row.addWidget(QLabel("Mode"))
        self.sketch_variant = QComboBox()
        variant_row.addWidget(self.sketch_variant, 1)
        self.finish_spline_btn = QPushButton("Finish spline")
        self.finish_spline_btn.clicked.connect(self._finish_pending_spline)
        self.finish_spline_btn.hide()
        variant_row.addWidget(self.finish_spline_btn)
        tools_layout.addLayout(variant_row)
        layout.addWidget(self.sketch_tools_box)

        # ---- options: construction + snaps
        self.sketch_options_box = QGroupBox("Options")
        options_layout = QVBoxLayout(self.sketch_options_box)
        self.sketch_construction_toggle = QCheckBox("Construction")
        self.sketch_construction_toggle.setToolTip(
            "Applies to the selected sketch entities and to new ones. Points cannot be construction."
        )
        self.sketch_construction_toggle.toggled.connect(self._apply_construction_to_selection)
        options_layout.addWidget(self.sketch_construction_toggle)
        snap_grid = QGridLayout()
        self.snap_toggles: dict[str, QCheckBox] = {}
        for index, name in enumerate(self.SNAP_OPTIONS):
            box = QCheckBox(name)
            box.setChecked(self.settings.value(f"sketch/snap_{name}", name != "On surface", type=bool))
            box.toggled.connect(lambda checked, n=name: self.settings.setValue(f"sketch/snap_{n}", checked))
            self.snap_toggles[name] = box
            snap_grid.addWidget(box, index // 3, index % 3)
        options_layout.addWidget(QLabel("Snap"))
        options_layout.addLayout(snap_grid)
        layout.addWidget(self.sketch_options_box)

        # ---- dimensions
        self.sketch_dimension_box = QGroupBox("Dimension  (hotkey D)")
        dim_layout = QHBoxLayout(self.sketch_dimension_box)
        self.dimension_input = QLineEdit()
        self.dimension_input.setPlaceholderText('e.g. 12.5 mm, 1/2 in, 3 ft, 2.4 cm, 1 1/4"')
        self.dimension_input.returnPressed.connect(self._apply_dimension_text)
        dim_layout.addWidget(self.dimension_input, 1)
        apply_dim = QPushButton("Apply")
        apply_dim.clicked.connect(self._apply_dimension_text)
        dim_layout.addWidget(apply_dim)
        layout.addWidget(self.sketch_dimension_box)

        # ---- visibility
        visibility_row = QHBoxLayout()
        self.sketch_visible = QCheckBox("Show sketches")
        self.sketch_visible.setChecked(True)
        self.sketch_visible.toggled.connect(lambda _checked: self._refresh_scene(reset_camera=False))
        visibility_row.addWidget(self.sketch_visible)
        self.construction_visible = QCheckBox("Show construction")
        self.construction_visible.setChecked(True)
        self.construction_visible.toggled.connect(lambda _checked: self._refresh_scene(reset_camera=False))
        visibility_row.addWidget(self.construction_visible)
        layout.addLayout(visibility_row)

        # ---- feature tree (right-click an entity to assign a tool operation)
        layout.addWidget(self._divider())
        tree_header = QHBoxLayout()
        tree_header.addWidget(self._section_label("SKETCH / FEATURE TREE"))
        self.revert_sketch_btn = QPushButton("Revert")
        self.revert_sketch_btn.setToolTip("Revert last sketch edit or error")
        self.revert_sketch_btn.clicked.connect(self.revert_sketch_change)
        tree_header.addWidget(self.revert_sketch_btn)
        layout.addLayout(tree_header)
        self.sketch_tree = QTreeWidget()
        self.sketch_tree.setHeaderLabels(["Feature / Dimension", "State"])
        self.sketch_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.sketch_tree.setMinimumHeight(160)
        self.sketch_tree.itemClicked.connect(self._on_sketch_tree_clicked)
        self.sketch_tree.itemDoubleClicked.connect(self._on_sketch_tree_double_clicked)
        layout.addWidget(self.sketch_tree, 1)
        tree_buttons = QHBoxLayout()
        define_btn = QPushButton("Define / Dim")
        define_btn.setToolTip("Define or edit dimensions for the selected sketch feature")
        define_btn.clicked.connect(self.edit_selected_sketch_dimensions)
        tree_buttons.addWidget(define_btn)
        target_btn = QPushButton("Target op")
        target_btn.setToolTip("Use the selected sketch feature as the target of a new operation")
        target_btn.clicked.connect(self.assign_selected_sketch_to_target)
        tree_buttons.addWidget(target_btn)
        dxf_btn = QPushButton("DXF")
        dxf_btn.setToolTip("Export sketches as DXF")
        dxf_btn.clicked.connect(self.export_sketches_dxf_dialog)
        tree_buttons.addWidget(dxf_btn)
        layout.addLayout(tree_buttons)
        self.sketch_error_banner = QLabel("")
        self.sketch_error_banner.setObjectName("warningText")
        self.sketch_error_banner.setWordWrap(True)
        self.sketch_error_banner.hide()
        layout.addWidget(self.sketch_error_banner)
        self._select_sketch_tool("Line")
        return tab

    # ------------------------------------------------------------ Position tab (Block 8)
    def _build_position_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._section_label("POSITION SELECTED PART"))
        self.position_points: list[np.ndarray] = []
        self.position_preview: TransformResult | None = None

        move_box = QGroupBox("Translate")
        move_form = QFormLayout(move_box)
        self.move_mode = QComboBox()
        self.move_mode.addItems(["Relative", "Absolute"])
        self.move_mode.setToolTip("Relative moves by the amounts; Absolute puts the pivot at the coordinates.")
        move_form.addRow("Mode", self.move_mode)
        self.move_xyz = [self._double_spin(0.0, -100000.0, 100000.0, 3) for _ in range(3)]
        for axis, spin in zip("XYZ", self.move_xyz):
            move_form.addRow(f"{axis} · mm", spin)
        layout.addWidget(move_box)

        rot_box = QGroupBox("Rotate")
        rot_form = QFormLayout(rot_box)
        self.rotate_angle = self._double_spin(0.0, -360.0, 360.0, 3)
        rot_form.addRow("Angle · °", self.rotate_angle)
        self.rotate_axis = QComboBox()
        self.rotate_axis.addItems(["Z", "X", "Y", "Two picked points"])
        rot_form.addRow("Axis", self.rotate_axis)
        self.pivot_mode = QComboBox()
        self.pivot_mode.addItems(["Selection centre", "Last picked point", "Origin"])
        rot_form.addRow("Pivot", self.pivot_mode)
        layout.addWidget(rot_box)

        row = QHBoxLayout()
        preview = QPushButton("Preview")
        preview.setToolTip("Show the result as a ghost before applying.")
        preview.clicked.connect(lambda: self._position_preview(self._position_matrix()))
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("primaryButton")
        apply_btn.setToolTip("Move the part (one undo step); sketches and stock on it follow.")
        apply_btn.clicked.connect(self._position_apply)
        reset = QPushButton("Reset")
        reset.setToolTip("Clear the typed values, picked points and preview.")
        reset.clicked.connect(self._position_reset)
        for button in (preview, apply_btn, reset):
            row.addWidget(button)
        layout.addLayout(row)

        pick_box = QGroupBox("Picked points")
        pick_layout = QVBoxLayout(pick_box)
        self.pick_point_button = QPushButton("Pick points")
        self.pick_point_button.setCheckable(True)
        self.pick_point_button.setToolTip("Click points on the model; the last two are used by rotation axis, alignment and axes.")
        self.pick_point_button.toggled.connect(self._toggle_position_picking)
        pick_layout.addWidget(self.pick_point_button)
        self.picked_label = QLabel("No points picked")
        self.picked_label.setObjectName("mutedText")
        self.picked_label.setWordWrap(True)
        pick_layout.addWidget(self.picked_label)
        layout.addWidget(pick_box)

        align_box = QGroupBox("Align (one undoable step each)")
        align_form = QFormLayout(align_box)
        self.align_mode = QComboBox()
        self.align_mode.addItems([
            "Face to face", "Vertex to vertex", "Face to plane", "Point to coordinates", "Point to sketch point",
        ])
        self.align_mode.setToolTip(
            "Faces: select a face on the part, then a target face (Ctrl+click). Vertices/points: pick points. "
            "Plane: choose it below."
        )
        align_form.addRow("Mode", self.align_mode)
        self.align_plane = QComboBox()
        self.align_plane.addItems(["XY plane", "XZ plane", "YZ plane"])
        align_form.addRow("Plane", self.align_plane)
        self.align_offset = self._double_spin(0.0, -10000.0, 10000.0, 3)
        align_form.addRow("Offset · mm", self.align_offset)
        self.align_flip = QCheckBox("Flip")
        self.align_flip.setToolTip("Point the aligned face the other way.")
        align_form.addRow("", self.align_flip)
        align_row = QHBoxLayout()
        align_preview = QPushButton("Preview")
        align_preview.clicked.connect(lambda: self._position_preview(self._alignment_matrix()))
        align_apply = QPushButton("Align")
        align_apply.setToolTip("Apply the alignment as one undoable move.")
        align_apply.clicked.connect(lambda: self._apply_matrix(self._alignment_matrix(), "Align part"))
        align_row.addWidget(align_preview)
        align_row.addWidget(align_apply)
        align_form.addRow(align_row)
        layout.addWidget(align_box)

        self.frame_group = QGroupBox("Work origin and axes")
        frame_form = QFormLayout(self.frame_group)
        frame_note = QLabel("Pick the origin point, then a second point along the X edge. Z stays as close to up as possible.")
        frame_note.setWordWrap(True)
        frame_form.addRow(frame_note)
        self.frame_flip = QCheckBox("Reverse X")
        frame_form.addRow("", self.frame_flip)
        frame_row = QHBoxLayout()
        frame_preview = QPushButton("Preview axes")
        frame_preview.clicked.connect(self._preview_work_frame)
        frame_apply = QPushButton("Apply axes")
        frame_apply.setToolTip("Set the active setup's work coordinate frame (undoable).")
        frame_apply.clicked.connect(self._apply_work_frame)
        frame_row.addWidget(frame_preview)
        frame_row.addWidget(frame_apply)
        frame_form.addRow(frame_row)
        self.frame_label = QLabel("Work frame: machine axes")
        self.frame_label.setObjectName("mutedText")
        self.frame_label.setWordWrap(True)
        frame_form.addRow(self.frame_label)
        layout.addWidget(self.frame_group)
        layout.addStretch(1)
        return tab

    def _position_part_index(self) -> int | None:
        row = self.parts_list.currentRow() if hasattr(self, "parts_list") else -1
        if 0 <= row < len(self.parts):
            return row
        if self.selected_features:
            return sorted(self.selected_features)[0][0]
        return 0 if len(self.parts) == 1 else None

    def _position_pivot(self, part) -> np.ndarray:
        mode = self.pivot_mode.currentText()
        if mode == "Last picked point" and self.position_points:
            return self.position_points[-1]
        if mode == "Origin":
            return np.zeros(3)
        return part_center(part)

    def _position_matrix(self) -> np.ndarray | None:
        index = self._position_part_index()
        if index is None:
            self.statusBar().showMessage("Select a part to position.", 4000)
            return None
        part = self.parts[index]
        values = np.array([s.value() for s in self.move_xyz])
        if self.move_mode.currentText() == "Absolute":
            matrix = matrix_translate(values - self._position_pivot(part))
        else:
            matrix = matrix_translate(values)
        if abs(self.rotate_angle.value()) > 1e-12:
            axis_name = self.rotate_axis.currentText()
            if axis_name == "Two picked points":
                if len(self.position_points) < 2:
                    self.statusBar().showMessage("Pick two points to define the rotation axis.", 4000)
                    return None
                axis, pivot = self.position_points[-1] - self.position_points[-2], self.position_points[-2]
            else:
                axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis_name]
                pivot = self._position_pivot(part)
            matrix = matrix @ matrix_rotate(axis, self.rotate_angle.value(), pivot)
        return matrix

    def _feature_face(self, part_index: int, feature_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        mesh = self.parts[part_index].mesh
        if mesh is None or mesh.n_cells == 0:
            return None
        cells = (
            np.flatnonzero(mesh.cell_data["feature_id"] == feature_id)
            if "feature_id" in mesh.cell_data
            else np.array([feature_id])
        )
        if cells.size == 0:
            return None
        sub = mesh.extract_cells(cells).extract_surface().compute_normals(cell_normals=True, point_normals=False)
        normal = np.asarray(sub.cell_data["Normals"]).mean(axis=0)
        return np.asarray(sub.points).mean(axis=0), normal / max(np.linalg.norm(normal), 1e-12)

    def _alignment_matrix(self) -> np.ndarray | None:
        index = self._position_part_index()
        if index is None:
            self.statusBar().showMessage("Select a part to align.", 4000)
            return None
        mode = self.align_mode.currentText()
        offset, flip = self.align_offset.value(), self.align_flip.isChecked()
        faces = sorted(self.selected_features)
        if mode in ("Face to face", "Face to plane"):
            own = [f for f in faces if f[0] == index]
            if not own:
                self.statusBar().showMessage("Select a face on the part to move.", 4000)
                return None
            face = self._feature_face(*own[0])
            if face is None:
                return None
            if mode == "Face to plane":
                normal = {"XY plane": (0, 0, 1), "XZ plane": (0, 1, 0), "YZ plane": (1, 0, 0)}[self.align_plane.currentText()]
                return align_faces_matrix(face[0], face[1], (0, 0, 0), normal, offset, flip)
            others = [f for f in faces if f != own[0]]
            if not others:
                self.statusBar().showMessage("Ctrl+click a target face as well.", 4000)
                return None
            target = self._feature_face(*others[0])
            return align_faces_matrix(face[0], face[1], target[0], target[1], offset, flip) if target else None
        if not self.position_points:
            self.statusBar().showMessage("Pick a reference point on the part first.", 4000)
            return None
        if mode == "Vertex to vertex":
            if len(self.position_points) < 2:
                self.statusBar().showMessage("Pick the part vertex, then the target vertex.", 4000)
                return None
            return matrix_translate(self.position_points[-1] - self.position_points[-2] + np.array([0, 0, offset]))
        reference = self.position_points[-1]
        if mode == "Point to coordinates":
            target = np.array([s.value() for s in self.move_xyz])
        else:
            feat = self.sketch_workflow.get_feature(self.active_sketch_id) if self.active_sketch_id else None
            if feat is None:
                self.statusBar().showMessage("Select a sketch entity as the target.", 4000)
                return None
            target = np.asarray(feat.points[0], dtype=float)
        return matrix_translate(target - reference + np.array([0, 0, offset]))

    def _position_preview(self, matrix: np.ndarray | None) -> None:
        index = self._position_part_index()
        self.plotter.remove_actor("position-preview", render=False)
        if matrix is None or index is None:
            self.plotter.render()
            return
        ghost = transform_mesh(self.parts[index].mesh, matrix)
        self.plotter.add_mesh(ghost, color=theme_color("highlight"), opacity=0.3, name="position-preview", pickable=False)
        self.plotter.render()

    def _position_apply(self) -> None:
        self._apply_matrix(self._position_matrix(), "Move part")

    def _apply_matrix(self, matrix: np.ndarray | None, label: str) -> None:
        index = self._position_part_index()
        if matrix is None or index is None:
            return
        self._apply_transform(self.parts[index], transform_part(self.parts[index], matrix), label)

    def _position_reset(self) -> None:
        for spin in self.move_xyz:
            spin.setValue(0.0)
        self.rotate_angle.setValue(0.0)
        self.position_points.clear()
        self.picked_label.setText("No points picked")
        self.plotter.remove_actor("position-preview", render=False)
        self._clear_frame_preview()
        self.plotter.render()

    def _toggle_position_picking(self, enabled: bool) -> None:
        try:
            if enabled:
                self.plotter.enable_point_picking(
                    callback=self._on_position_point, show_message=False, use_mesh=True, show_point=True, left_clicking=True
                )
            else:
                self.plotter.disable_picking()
        except (AttributeError, RuntimeError) as error:  # no active picker to switch
            self.terminal_core.log("warning", f"Point picking unavailable: {error}")

    def _on_position_point(self, point) -> None:
        if point is None:
            return
        self.position_points.append(np.asarray(point, dtype=float))
        self.position_points = self.position_points[-4:]
        shown = "; ".join(f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in self.position_points[-2:])
        self.picked_label.setText(f"Last points: {shown}")

    def _work_frame_from_picks(self) -> Frame | None:
        if len(self.position_points) < 2:
            self.statusBar().showMessage("Pick the origin, then a point along the X edge.", 4000)
            return None
        origin, along = self.position_points[-2], self.position_points[-1]
        return frame_from_origin_and_edge(origin, along - origin, flip=self.frame_flip.isChecked())

    def _preview_work_frame(self) -> None:
        frame = self._work_frame_from_picks()
        self._clear_frame_preview()
        if frame is None:
            return
        size = max(self._scene_extent() * 0.15, 10.0)
        for axis, color in ((frame.x_axis, "axis_x"), (frame.y_axis, "axis_y"), (frame.z_axis, "axis_z")):
            self.plotter.add_mesh(
                pv.Line(frame.origin, frame.origin + axis * size), color=theme_color(color), line_width=4,
                name=f"frame-preview-{color}", pickable=False,
            )
        self.plotter.render()

    def _clear_frame_preview(self) -> None:
        for color in ("axis_x", "axis_y", "axis_z"):
            self.plotter.remove_actor(f"frame-preview-{color}", render=False)

    def _apply_work_frame(self) -> None:
        frame = self._work_frame_from_picks()
        if frame is None:
            return
        self._push_history("Assign work axes")
        self.active_setup.work_frame = frame.to_dict()
        o = frame.origin
        self.frame_label.setText(f"Work frame origin ({o[0]:.2f}, {o[1]:.2f}, {o[2]:.2f}); X along the picked edge.")
        self.terminal_core.log("info", f"Work frame set for {self.active_setup.name}.")
        self._refresh_scene(reset_camera=False)

    def _select_sketch_tool(self, kind: str) -> None:
        self.sketch_kind.setCurrentText(kind)
        button = self.sketch_tool_buttons.get(kind)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        variants = next((v for _l, k, v in self.SKETCH_TOOLS if k == kind), ("Default",))
        self.sketch_variant.clear()
        self.sketch_variant.addItems(list(variants))
        self.finish_spline_btn.setVisible(kind == "Spline")
        self.pending_sketch_points.clear()

    def _update_sketch_tab_state(self, editing: bool) -> None:
        """Tools are only live while a sketch is being edited on a plane or face."""
        for widget in (self.sketch_tools_box, self.sketch_options_box, self.sketch_dimension_box):
            widget.setEnabled(bool(editing))
        self.use_face_button.setEnabled(True)
        self.clear_sketch_btn.setVisible(bool(editing))
        plane = self.sketch_workflow.active_plane
        if editing:
            where = plane.name if plane is not None else self.work_plane.currentText()
            self.sketch_status.setText(f"Sketching on {where}. Pick a tool and click in the viewer.")
        else:
            self.sketch_status.setText("Click New sketch, then pick a plane or select a face.")

    def _apply_construction_to_selection(self, construction: bool) -> None:
        """Construction applies to the selected entities (not points) and to new ones."""
        ids = [
            item.data(0, Qt.ItemDataRole.UserRole)[1]
            for item in self.sketch_tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole) and item.data(0, Qt.ItemDataRole.UserRole)[0] == "feature"
        ]
        if len(ids) <= 1:  # single selection follows the active entity (tree click or last drawn)
            ids = [self.active_sketch_id] if self.active_sketch_id else ids
        targets = [f for f in (self.sketch_workflow.get_feature(i) for i in ids) if f is not None and f.kind != "Point"]
        targets = [f for f in targets if f.construction != bool(construction)]
        if not targets:
            return
        self._push_history("Toggle construction")
        for feat in targets:
            self.sketch_workflow.set_construction(feat.feature_id, construction)
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)

    def _sketch_use_selected_face(self) -> None:
        if not self.sketch_mode.isChecked():
            self.sketch_mode.setChecked(True)  # starts the sketch on the current selection
        else:
            self.start_sketch_on_selection()
            self._update_sketch_tab_state(True)

    def _focus_dimension_input(self) -> None:
        self.cam_tabs.setCurrentIndex(0)
        if self.sketch_dimension_box.isEnabled():
            self.dimension_input.setFocus()
            self.dimension_input.selectAll()
        else:
            self.statusBar().showMessage("Start a sketch (New sketch) before entering dimensions.", 3000)

    def _apply_dimension_text(self) -> None:
        """Enter confirms: parse the typed length (mm, cm, m, in, ft, fractions) and apply it."""
        text = self.dimension_input.text().strip()
        if not text:
            return
        inches = self.units.display is LengthUnit.INCHES
        millimeters = parse_length_to_mm(text, 25.4 if inches else 1.0)
        if millimeters is None:
            self.statusBar().showMessage(
                f"Could not read '{text}'. Try 12.5 mm, 1/2 in, 1 1/4\", 3 ft, 2.4 cm.", 5000
            )
            return
        unit_text, sig_figs = describe_length_text(text, self.units.display.value)
        payloads = [
            item.data(0, Qt.ItemDataRole.UserRole)
            for item in self.sketch_tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole)
        ]
        current = self.sketch_tree.currentItem()
        if not payloads and current is not None and current.data(0, Qt.ItemDataRole.UserRole):
            payloads = [current.data(0, Qt.ItemDataRole.UserRole)]
        if not payloads and self.active_sketch_id:
            payloads = [("feature", self.active_sketch_id, "")]
        entity_ids = list(dict.fromkeys(p[1] for p in payloads if p[0] == "feature"))
        if not payloads:
            self.statusBar().showMessage(
                "Select an entity, two entities, or a dimension, then type its value and press Enter.", 4500
            )
            return
        self._push_history("Edit sketch dimension")
        if len(entity_ids) == 2:
            ok = self.sketch_workflow.set_pair_distance(entity_ids[0], entity_ids[1], millimeters, unit_text, sig_figs) is not None
        elif payloads[0][0] == "dim":
            _kind, feature_id, dim_id = payloads[0]
            feat = self.sketch_workflow.get_feature(feature_id)
            ok = self.sketch_workflow.update_dimension_value(feature_id, dim_id, millimeters)
            for dim in feat.dimensions if feat else []:
                if dim.dim_id == dim_id:
                    dim.unit_text, dim.sig_figs = unit_text, sig_figs
        else:
            ok = self.sketch_workflow.set_entity_dimension(entity_ids[0], millimeters, unit_text, sig_figs) is not None
            if not ok:
                feat = self.sketch_workflow.get_feature(entity_ids[0])
                if feat is not None and not self.sketch_workflow.can_edit(feat):
                    self.statusBar().showMessage("That entity belongs to another sketch; edit it inside its own sketch.", 5000)
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)
        shown = millimeters / 25.4 if inches else millimeters
        unit = self.units.display.value
        if ok:
            self.dimension_input.clear()
            self.statusBar().showMessage(f"Dimension set to {shown:.4g} {unit} ({millimeters:.4g} mm).", 4000)
        else:
            self.sketch_error_banner.setText("That dimension caused a sketch error. Fix the value or click Revert.")
            self.sketch_error_banner.show()

    # --- Esc: back out one layer per press ------------------------------------------------
    def escape_layered(self) -> None:
        if self.sketch_workflow.has_errors() and self._revert_sketch_errors():
            self.statusBar().showMessage("Sketch error reverted.", 3000)
            return
        if self.pending_sketch_points:
            self.pending_sketch_points.clear()
            self._refresh_scene(reset_camera=False)
            self.statusBar().showMessage("Cancelled the sketch entity in progress.", 2500)
            return
        if self.dimension_input.text():
            self.dimension_input.clear()
            self.plotter.interactor.setFocus()
            self.statusBar().showMessage("Dimension entry cleared.", 2000)
            return
        if self.feature_pick_active:
            self.feature_pick_active = False
            self.pick_feature_button.setChecked(False)
            self.pick_feature_button.setText("Pick sketch feature")
            self.statusBar().showMessage("Cancelled picking.", 2000)
            return
        if self.measure_active:
            self.measure_active = False
            self.measure_points.clear()
            self.measure_result.setText("Select Measure, then pick two points")
            self._refresh_scene(reset_camera=False)
            self.statusBar().showMessage("Measurement cancelled.", 2000)
            return
        if self.sketch_mode.isChecked():
            self.sketch_mode.setChecked(False)
            self.statusBar().showMessage("Finished sketch.", 2000)
            return
        if self.selected_cells or self.selected_features or self.selected_feature:
            self.selected_cells.clear()
            self.selected_features.clear()
            self.selected_feature = None
            self._refresh_scene(reset_camera=False)
            self.statusBar().showMessage("Selection cleared.", 2000)
            return
        focus = QApplication.focusWidget()
        if focus is not None and focus is not self.plotter.interactor:
            self.plotter.interactor.setFocus()
            return
        self.statusBar().showMessage("Nothing to cancel.", 1500)

    # --- sketch tool hotkeys -----------------------------------------------------------------
    def _sketch_hotkey(self, kind: str, variant: str | None = None) -> None:
        if not self.sketch_mode.isChecked():
            self.sketch_mode.setChecked(True)  # starts on the selected face, else the chosen plane
        self.cam_tabs.setCurrentIndex(0)
        self._select_sketch_tool(kind)
        if variant:
            position = self.sketch_variant.findText(variant)
            if position >= 0:
                self.sketch_variant.setCurrentIndex(position)

    # --- Notes tab ----------------------------------------------------------------------
    def _build_notes_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._section_label("NOTES · HOVER IN VIEW"))
        self.note_target = QComboBox()
        self.note_target.currentIndexChanged.connect(self._load_note_target)
        layout.addWidget(self.note_target)
        self.note_editor = QTextEdit()
        self.note_editor.setPlaceholderText("Add setup, inspection, or feature notes…")
        self.note_editor.setMinimumHeight(110)
        layout.addWidget(self.note_editor)
        save_note = QPushButton("Save note")
        save_note.clicked.connect(self.save_note)
        layout.addWidget(save_note)
        layout.addWidget(self._divider())
        layout.addWidget(self._section_label("NOTE TREE"))
        self.notes_list = QTreeWidget()
        self.notes_list.setHeaderLabels(["Target", "Note"])
        self.notes_list.setToolTip("Hover a note to highlight its target; double-click to select the target.")
        self.notes_list.setMouseTracking(True)
        self.notes_list.itemClicked.connect(self._on_note_item_clicked)
        self.notes_list.itemEntered.connect(self._on_note_item_hovered)
        self.notes_list.viewportEntered.connect(lambda: self._set_note_highlight(None))
        self.notes_list.itemDoubleClicked.connect(self._jump_to_note_target)
        layout.addWidget(self.notes_list, 1)
        return tab

    def _set_note_highlight(self, payload) -> None:
        new = {tuple(payload)} if payload else set()
        if new == self.highlight_feature_ids:
            return
        self.highlight_feature_ids = new
        self._draw_note_highlight()
        self._draw_sketch_entities()
        self.plotter.render()

    def _on_note_item_hovered(self, item, _column: int = 0) -> None:
        self._set_note_highlight(item.data(0, Qt.ItemDataRole.UserRole))

    def _draw_note_highlight(self) -> None:
        """Overlay the part/feature named by the hovered note (sketches are drawn in _draw_sketch_entities)."""
        self.plotter.remove_actor("note-highlight", render=False)
        for kind, index, feature_id in self.highlight_feature_ids:
            if kind not in {"part", "feature"} or not isinstance(index, int) or not 0 <= index < len(self.parts):
                continue
            mesh = self.parts[index].mesh
            if mesh is None or mesh.n_cells == 0:
                continue
            if kind == "feature" and "feature_id" in mesh.cell_data:
                cells = np.flatnonzero(mesh.cell_data["feature_id"] == feature_id)
            elif kind == "feature":
                cells = np.array([feature_id]) if 0 <= feature_id < mesh.n_cells else np.array([], dtype=int)
            else:
                cells = np.arange(mesh.n_cells)
            if cells.size == 0:
                continue
            overlay = mesh.extract_cells(cells)
            self.plotter.add_mesh(overlay, color=theme_color("highlight"), opacity=0.6, name="note-highlight", pickable=False)

    def _jump_to_note_target(self, item, _column: int = 0) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind, index, feature_id = payload
        if kind == "sketch":
            self.active_sketch_id = index
            self.cam_tabs.setCurrentIndex(0)
            self._refresh_sketch_tree()
            self._refresh_scene(reset_camera=False)
        elif kind == "feature" and isinstance(index, int) and 0 <= index < len(self.parts):
            self.selected_features = {(index, feature_id)}
            self.selected_feature = (index, feature_id)
            self.known_feature_targets.add((index, feature_id))
            self.operation_target = f"{self.parts[index].name} · Feature {feature_id + 1}"
            self._refresh_scene(reset_camera=False)
        elif kind == "part" and isinstance(index, int) and 0 <= index < len(self.parts):
            self.parts_list.setCurrentRow(index)
        self.statusBar().showMessage(f"Selected note target: {item.text(0)}", 4000)

    def _open_note_for(self, payload: tuple) -> None:
        """Show the Notes tab with the given target loaded in the editor."""
        self._refresh_note_targets()
        position = self.note_target.findData(payload)
        if position >= 0:
            self.note_target.setCurrentIndex(position)
        notes_index = next((i for i in range(self.cam_tabs.count()) if self.cam_tabs.tabText(i) == "Notes"), -1)
        if notes_index >= 0:
            self.cam_tabs.setCurrentIndex(notes_index)
        self.note_editor.setFocus()

    def _refresh_notes_list(self) -> None:
        if not hasattr(self, "notes_list"):
            return
        self.notes_list.clear()
        for index, part in enumerate(self.parts):
            if getattr(part, "note", ""):
                item = QTreeWidgetItem([f"Part · {part.name}", part.note.splitlines()[0][:80]])
                item.setData(0, Qt.ItemDataRole.UserRole, ("part", index, -1))
                self.notes_list.addTopLevelItem(item)
            for feature_id, note in sorted(getattr(part, "feature_notes", {}).items()):
                if note:
                    item = QTreeWidgetItem([f"Feature {feature_id + 1} · {part.name}", note.splitlines()[0][:80]])
                    item.setData(0, Qt.ItemDataRole.UserRole, ("feature", index, feature_id))
                    self.notes_list.addTopLevelItem(item)
        for feat in self.sketch_workflow.features:
            note = self.sketch_notes.get(feat.feature_id, "")
            if note:
                item = QTreeWidgetItem([f"Sketch · {feat.name}", note.splitlines()[0][:80]])
                item.setData(0, Qt.ItemDataRole.UserRole, ("sketch", feat.feature_id, -1))
                self.notes_list.addTopLevelItem(item)

    def _on_note_item_clicked(self, item, _column: int = 0) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        position = self.note_target.findData(payload)
        if position >= 0:
            self.note_target.setCurrentIndex(position)

    # --- Preview & G-code tab -----------------------------------------------------------
    def _build_preview_gcode_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._section_label("PREVIEW CUTS"))
        for text, slot in (
            ("Show / refresh 3D toolpath preview", self.render_toolpath_preview),
            ("Clear 3D toolpath preview", self.clear_toolpath_preview),
            ("Play / pause simulation", self._on_toggle_sim_playback),
            ("Preview tool in scene", self.render_tool_preview),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)
        layout.addWidget(self._section_label("CYCLE TIME"))
        self.cycle_current_label = QLabel("Current operation: -")
        self.cycle_current_label.setWordWrap(True)
        self.cycle_current_label.setToolTip("Elapsed and remaining time of the operation at the run or preview position.")
        layout.addWidget(self.cycle_current_label)
        self.cycle_total_label = QLabel("Estimated total: generate or preview a program")
        self.cycle_total_label.setWordWrap(True)
        self.cycle_total_label.setToolTip(
            "Remaining distance / feed at the current speed multiplier; rapids at the machine rate; "
            "+2 min per tool change, +7 min per reposition."
        )
        layout.addWidget(self.cycle_total_label)
        self.cycle_timer = QTimer(self)
        self.cycle_timer.setInterval(1000)
        self.cycle_timer.timeout.connect(self._update_cycle_estimate)
        self.cycle_timer.start()
        layout.addWidget(self._divider())
        layout.addWidget(self._section_label("G-CODE"))
        checklist = QPushButton("G-code checklist…")
        checklist.clicked.connect(self.show_gcode_readiness)
        layout.addWidget(checklist)
        self.export_button = QPushButton("Generate G-code…")
        self.export_button.setObjectName("primaryButton")
        self.export_button.setEnabled(self._has_any_operations())
        self.export_button.clicked.connect(self.export_gcode)
        layout.addWidget(self.export_button)
        self.advanced_editor_button = QPushButton("Open in advanced editor")
        self.advanced_editor_button.setToolTip("Open the generated program in the Advanced G-code editor (gate required).")
        self.advanced_editor_button.clicked.connect(lambda: self.open_advanced_gcode())
        self.advanced_editor_button.setEnabled(self._advanced_gcode_allowed())
        layout.addWidget(self.advanced_editor_button)
        layout.addStretch(1)
        return tab

    def _build_operation_panel(self) -> QWidget:
        panel = self._panel()
        panel.setMinimumWidth(290)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(11)
        layout.addWidget(self._section_label("NEW OPERATION"))
        self.operation_kind = QComboBox()
        self.operation_kind.addItems(["Rough", "Finish", "Drill", "Ream", "Tap"])
        layout.addWidget(self.operation_kind)
        self.target_label = QLabel("Target: Whole part")
        self.target_label.setObjectName("mutedText")
        self.target_label.setWordWrap(True)
        layout.addWidget(self.target_label)
        self.pick_feature_button = QPushButton("Pick sketch feature")
        self.pick_feature_button.setCheckable(True)
        self.pick_feature_button.clicked.connect(self.start_feature_pick)
        layout.addWidget(self.pick_feature_button)

        form = QFormLayout()
        self.operation_form = form
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.tool_name = QComboBox()
        self.tool_name.setEditable(False)
        self._refresh_tool_combo()
        self.tool_name.currentIndexChanged.connect(self._apply_tool_to_operation)
        form.addRow("Tool Type", self.tool_name)
        self.tool_description = QLabel("")
        self.tool_description.setObjectName("mutedText")
        self.tool_description.setWordWrap(True)
        form.addRow("", self.tool_description)
        self.tool_preview_plotter = QtInteractor()
        self.tool_preview_plotter.set_background(theme_color("viewport_bg"))
        self.tool_preview_plotter.interactor.setFixedHeight(150)
        self.tool_preview_plotter.interactor.setToolTip("3D envelope of the selected inventory tool.")
        form.addRow("", self.tool_preview_plotter.interactor)
        self.tool_diameter = self._double_spin(6, 0.1, 100, 2)
        self.tool_diameter_label = QLabel("Diameter (mm)")
        form.addRow(self.tool_diameter_label, self.tool_diameter)
        self.spindle = self._int_spin(6000, 0, 60000)
        form.addRow("Spindle · rpm", self.spindle)
        self.manual_spindle_box = QCheckBox("Manual spindle speed")
        self.manual_spindle_box.toggled.connect(self._on_manual_spindle_toggled)
        form.addRow("", self.manual_spindle_box)
        self.recommended_rpm_label = QLabel("Recommended RPM: --")
        self.recommended_rpm_label.setObjectName("mutedText")
        self.recommended_rpm_label.setWordWrap(True)
        self.recommended_rpm_label.hide()
        form.addRow("", self.recommended_rpm_label)

        speed_row_widget = QWidget()
        speed_row = QHBoxLayout(speed_row_widget)
        speed_row.setContentsMargins(0, 0, 0, 0)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(-100, 100)
        self.speed_slider.setValue(multiplier_to_slider(1.0))
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        speed_row.addWidget(self.speed_slider, 1)
        self.speed_factor_label = QLabel("1.00x")
        self.speed_factor_label.setFixedWidth(48)
        speed_row.addWidget(self.speed_factor_label)
        form.addRow("Speed/feed ×", speed_row_widget)


        self.feed = self._double_spin(450, 0.1, 50000, 1)
        form.addRow("Feed · mm/min", self.feed)
        self.finish_quality = QComboBox()
        self.finish_quality.addItems(["Draft", "Standard", "Fine"])
        self.finish_quality.setCurrentText("Standard")
        self.finish_quality.setToolTip("Sets the default overlap between passes.")
        self.finish_quality.currentTextChanged.connect(self._apply_default_overlap)
        form.addRow("Finish quality", self.finish_quality)
        self.overlap = self._double_spin(50, 0, 90, 0)
        self.overlap.setSuffix(" %")
        self.overlap.setToolTip(
            "Share of the tool diameter that each pass overlaps the previous one.\n"
            "Defaults: Draft 40%, Standard 50%, Fine 70%; hard materials +10% (lighter radial cut)."
        )
        self.overlap.valueChanged.connect(self._on_overlap_edited)
        form.addRow("Overlap", self.overlap)
        self.stepdown = self._double_spin(2, 0.01, 100, 2)
        form.addRow("Pass depth · mm", self.stepdown)
        self.depth = self._double_spin(8, 0.01, 1000, 2)
        form.addRow("End depth · mm", self.depth)

        self.tool_diameter.valueChanged.connect(self._apply_recommended_defaults)
        self.operation_kind.currentTextChanged.connect(self._apply_recommended_defaults)
        self.direction = QComboBox()
        self.direction.addItems(["Conventional", "Climb"])
        form.addRow("Cut direction", self.direction)
        layout.addLayout(form)

        entry_box = QGroupBox("Depth and entry")
        entry_form = QFormLayout(entry_box)
        self.first_pass_count = self._int_spin(0, 0, 20)
        self.first_pass_count.setToolTip("Light passes that even the face before roughing starts.")
        entry_form.addRow("First passes", self.first_pass_count)
        self.first_pass_depth = self._double_spin(0.5, 0.01, 50, 3)
        entry_form.addRow("First pass depth · mm", self.first_pass_depth)
        self.final_pass_depth = self._double_spin(0.0, 0.0, 50, 3)
        self.final_pass_depth.setToolTip("Depth of the last, finishing pass; 0 for none.")
        entry_form.addRow("Final pass depth · mm", self.final_pass_depth)
        self.ramp_mode = QComboBox()
        self.ramp_mode.addItems(["Plunge", "Angle", "Length"])
        self.ramp_mode.setToolTip("How the tool enters each layer: straight down, or ramping along the path.")
        entry_form.addRow("Entry", self.ramp_mode)
        self.ramp_angle = self._double_spin(3.0, 0.5, 45.0, 2)
        entry_form.addRow("Ramp angle · °", self.ramp_angle)
        self.ramp_length = self._double_spin(10.0, 0.1, 500.0, 2)
        entry_form.addRow("Ramp length · mm", self.ramp_length)
        self.lead_in = self._double_spin(0.0, 0.0, 100.0, 2)
        entry_form.addRow("Lead-in · mm", self.lead_in)
        self.lead_out = self._double_spin(0.0, 0.0, 100.0, 2)
        entry_form.addRow("Lead-out · mm", self.lead_out)
        entry_defaults = QPushButton("Defaults from material")
        entry_defaults.setToolTip("Fill these from the stock material, finish quality and tool diameter.")
        entry_defaults.clicked.connect(self._apply_entry_defaults)
        entry_form.addRow(entry_defaults)
        layout.addWidget(entry_box)
        trace_btn = QPushButton("Trace curved face...")
        trace_btn.setToolTip("3-axis surface roughing then finishing on the selected face(s); never cuts into the surface.")
        trace_btn.clicked.connect(self.trace_curved_face)
        layout.addWidget(trace_btn)

        add_operation = QPushButton("Add operation to stack")
        add_operation.setObjectName("primaryButton")
        add_operation.setToolTip("Add an operation with these settings to the active setup.")
        add_operation.clicked.connect(self.add_operation)
        op_buttons = QHBoxLayout()
        op_buttons.addWidget(add_operation, 1)
        duplicate_operation = QPushButton("Duplicate")
        duplicate_operation.setToolTip("Copy the selected operation (or the last one) to the end of its setup.")
        duplicate_operation.clicked.connect(self.duplicate_operation)
        op_buttons.addWidget(duplicate_operation)
        assign_many = QPushButton("Assign to selection")
        assign_many.setToolTip("Copy the selected operation once for every selected feature or sketch.")
        assign_many.clicked.connect(self.assign_operation_to_selection)
        op_buttons.addWidget(assign_many)
        layout.addLayout(op_buttons)
        layout.addWidget(self._divider())
        layout.addWidget(self._divider())
        setup_header = QHBoxLayout()
        setup_header.addWidget(self._section_label("SETUPS & OPERATION STACK"))
        add_setup_btn = QPushButton("+ New setup")
        add_setup_btn.clicked.connect(self._add_new_setup)
        setup_header.addWidget(add_setup_btn)
        layout.addLayout(setup_header)

        setup_meta_row = QHBoxLayout()
        self.setup_orientation_combo = QComboBox()
        self.setup_orientation_combo.addItems(
            ["Top (+Z)", "Bottom (-Z)", "Front (-Y)", "Back (+Y)", "Left (-X)", "Right (+X)", "Custom face"]
        )
        self.setup_orientation_combo.currentTextChanged.connect(self._on_setup_orientation_changed)
        setup_meta_row.addWidget(self.setup_orientation_combo, 1)

        use_picked_btn = QPushButton("From selection")
        use_picked_btn.setToolTip("Set active setup orientation & origin from selected face/feature")
        use_picked_btn.clicked.connect(self._apply_selection_to_active_setup)
        setup_meta_row.addWidget(use_picked_btn)
        layout.addLayout(setup_meta_row)
        layout.addWidget(self._section_label("OPERATION STACK"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        stack_contents = OperationDropArea(self._drop_operation)
        stack_contents.setToolTip("Drag an operation card onto another position to reorder the stack.")
        self.operation_stack_area = stack_contents
        self.operation_stack = QVBoxLayout(stack_contents)
        self.operation_stack.setContentsMargins(0, 0, 0, 0)
        self.operation_stack.setSpacing(6)
        self.stack_empty = QLabel("No operations yet")
        self.stack_empty.setObjectName("mutedText")
        self.stack_empty.setWordWrap(True)
        self.operation_stack.addWidget(self.stack_empty)
        self.warning_label = QLabel("")
        self.warning_label.setObjectName("warningText")
        self.warning_label.setWordWrap(True)
        self.operation_stack.addWidget(self.warning_label)
        self.operation_stack.addStretch(1)
        scroll.setWidget(stack_contents)
        layout.addWidget(scroll, 1)

        self._apply_tool_to_operation(self.tool_name.currentIndex())
        self._update_operation_form_visibility()
        return panel

    def open_part(self) -> None:
        path = self._choose_part_path("Open part in new workspace")
        if path:
            self._load_part_path(path, replace_workspace=True)

    def import_part(self) -> None:
        path = self._choose_part_path("Import part into current workspace")
        if path:
            self._load_part_path(path, replace_workspace=False)

    def _choose_part_path(self, title: str) -> str:
        extensions = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            f"Supported now ({extensions});;All files (*.*)",
        )
        return path

    def _load_part_path(self, path: str, replace_workspace: bool) -> None:
        default_name = Path(path).stem
        meta = self._prompt_import_metadata(default_name, replace_workspace)
        if meta is None:
            return
        job_name, part_name, material_name = meta
        self.project_name_label.setText(job_name)
        if self.stock_material.findText(material_name) >= 0:
            self.stock_material.setCurrentText(material_name)
        unit_factor = 25.4 if self.units.imported_as is LengthUnit.INCHES else 1.0
        try:
            geometry = load_part_geometry(path, unit_factor)
        except Exception as error:
            self._critical_box(self, "Import failed", str(error))
            return
        self._push_history(f"Import {part_name}")
        if replace_workspace:
            self._reset_workspace()
        role = PartRole.TARGET
        self.parts.append(
            PartModel(
                part_name,
                geometry.mesh,
                Path(path),
                imported=True,
                brep_shape=geometry.brep_shape,
                solid_shape=geometry.solid_shape,
                edge_mesh=geometry.edge_mesh,
                solid_conversion_message=geometry.solid_conversion_message,
                native_brep=geometry.native_brep,
                role=role,
            )
        )
        if role is PartRole.STOCK:
            self.stock_part = self.parts[-1]
            self.stock_mesh = geometry.mesh
            self.stock_imported = True
            self.stock_status.setText(f"Stock · {part_name}")
        else:
            self._check_bed_fit(self.parts[-1])
        self._refresh_parts()
        self.parts_list.setCurrentRow(len(self.parts) - 1)
        self._refresh_note_targets()
        self._refresh_scene()
        status = f"Imported {Path(path).name} as {geometry.geometry_type.lower()}"
        if geometry.solid_conversion_message:
            status += f" · {geometry.solid_conversion_message}"
        self.statusBar().showMessage(status, 8000)
        if getattr(geometry, "unsupported", None):
            message = "Not imported (unsupported entities): " + ", ".join(geometry.unsupported)
            self.terminal_core.log("warning", message)
            QMessageBox.information(self, "Some entities were not imported", message)

    def export_part(self) -> None:
        """Export the selected part, or every part, as STL or STEP; anything not exported is listed."""
        if not self.parts:
            QMessageBox.information(self, "Export part", "There is nothing to export.")
            return
        row = self.parts_list.currentRow()
        targets = [self.parts[row]] if 0 <= row < len(self.parts) else list(self.parts)
        name = targets[0].name if len(targets) == 1 else "workspace"
        path, _ = QFileDialog.getSaveFileName(self, "Export part", f"{name}.step", "STEP (*.step *.stp);;STL (*.stl)")
        if not path:
            return
        try:
            saved, skipped = export_parts(targets, path)
        except Exception as error:
            self._critical_box(self, "Export failed", str(error))
            return
        self.terminal_core.log("info", f"Exported {len(targets) - len(skipped)} part(s) to {saved}")
        if skipped:
            message = "Not exported:\n" + "\n".join(skipped)
            self.terminal_core.log("warning", message.replace("\n", "; "))
            QMessageBox.information(self, "Export part", message)
        else:
            self.statusBar().showMessage(f"Exported {saved.name}", 5000)

    def _reset_workspace(self) -> None:
        self.parts.clear()
        self.stock_part = None
        self.stock_mesh = None
        self.stock_imported = False
        self.setups = [
            MachiningSetup(name="Setup 1", orientation="Top (+Z)", color=SETUP_PALETTE[0])
        ]
        self.active_setup_index = 0
        self.sketch_workflow = SketchWorkflowManager()
        self.sketch_entities.clear()
        self.pending_sketch_points.clear()
        self.active_sketch_id = None
        self.dragging_sketch = None
        if hasattr(self, "sketch_tree"):
            self._refresh_sketch_tree()
        self.stock_mesh = None
        self.stock_part = None
        self.stock_imported = False
        self.stock_note = ""
        self.dataset_part_indexes.clear()
        self.stock_dataset_keys.clear()
        self.selected_cells.clear()
        self.selected_features.clear()
        self.selected_feature = None
        self.known_feature_targets.clear()
        self.operation_target = "Whole part"
        self.target_label.setText("Target: Whole part")
        self.stock_status.setText("No stock geometry")
        self.generate_action.setEnabled(False)
        self.export_button.setEnabled(False)
        self._refresh_parts()
        self._refresh_note_targets()
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)

    def open_stock(self) -> None:
        extensions = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import stock geometry",
            "",
            f"Supported formats ({extensions});;All files (*.*)",
        )
        if not path:
            return
        scale = 25.4 if self.units.imported_as is LengthUnit.INCHES else 1.0
        try:
            geometry = load_part_geometry(path, scale)
        except Exception as error:
            self._critical_box(self, "Stock import failed", str(error))
            return
        part_name = Path(path).name
        self._set_stock_geometry(
            part_name,
            geometry.mesh,
            Path(path),
            imported=True,
            brep_shape=geometry.brep_shape,
            solid_shape=geometry.solid_shape,
            edge_mesh=geometry.edge_mesh,
            native_brep=geometry.native_brep,
            conversion_message=geometry.solid_conversion_message,
        )

    def create_stock_box(self) -> None:
        """Add stock block(s) that fully bound the target part(s); never replaces a part.

        If some (but not all) parts are selected, one box bounds that selection. If nothing or
        everything is selected, each part gets its own box.
        """
        targets = [
            (i, p) for i, p in enumerate(self.parts)
            if p.role is not PartRole.STOCK and p.mesh is not None and p.mesh.n_cells > 0
        ]
        if not targets:
            QMessageBox.information(self, "Create box", "Import a part first, then create its stock box.")
            return
        selected = set(self._selected_part_indexes()) | set(self.selected_cells.keys())
        chosen = [(i, p) for i, p in targets if i in selected]
        if not chosen or len(chosen) == len(targets):
            groups = [[(i, p)] for i, p in targets]
        else:
            groups = [chosen]

        self._push_history("Create stock box")
        # Drop stock boxes made earlier so repeated clicks do not stack duplicates.
        self.parts = [
            p for p in self.parts
            if not (p.role is PartRole.STOCK and not p.imported and p.name.startswith("Stock ·"))
        ]
        material = self.stock_material.currentText() if hasattr(self.stock_material, "currentText") else ""
        first_box = None
        for group in groups:
            bounds = np.array([part.mesh.bounds for _, part in group], dtype=float)
            xmin, ymin, zmin = bounds[:, 0::2].min(axis=0)
            xmax, ymax, zmax = bounds[:, 1::2].max(axis=0)
            box = pv.Cube(
                center=((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2),
                x_length=max(xmax - xmin, 1e-3),
                y_length=max(ymax - ymin, 1e-3),
                z_length=max(zmax - zmin, 1e-3),
            )
            label = group[0][1].name if len(group) == 1 else f"{len(group)} parts"
            stock = PartModel(f"Stock · {label}", box)
            stock.role = PartRole.STOCK
            stock.imported = False
            self.parts.append(stock)
            if first_box is None:
                first_box = stock
        self.stock_part = first_box
        self.stock_mesh = first_box.mesh
        self.stock_imported = False
        sx, sy, sz = (first_box.mesh.bounds[1] - first_box.mesh.bounds[0],
                      first_box.mesh.bounds[3] - first_box.mesh.bounds[2],
                      first_box.mesh.bounds[5] - first_box.mesh.bounds[4])
        for axis, value in zip("XYZ", (sx, sy, sz)):
            self.stock_dimensions[axis].setValue(float(value))
        self.stock_status.setText(
            f"{len(groups)} stock box(es) · {material}".rstrip(" ·")
        )
        self._refresh_parts()
        self._refresh_note_targets()
        self._refresh_scene()

    def _set_stock_geometry(
        self,
        name: str,
        mesh: pv.PolyData,
        source_path: Path | None,
        *,
        imported: bool,
        brep_shape: object | None = None,
        solid_shape: object | None = None,
        edge_mesh: pv.PolyData | None = None,
        native_brep: bool = False,
        conversion_message: str = "",
    ) -> None:
        self._push_history("Set stock")
        if self.stock_part is None:
            self.stock_part = PartModel(name, mesh)
            self.parts.append(self.stock_part)
        self.stock_part.name = name
        self.stock_part.mesh = mesh
        self.stock_part.source_path = source_path
        self.stock_part.imported = imported
        self.stock_part.brep_shape = brep_shape
        self.stock_part.solid_shape = solid_shape
        self.stock_part.edge_mesh = edge_mesh
        self.stock_part.native_brep = native_brep
        self.stock_part.solid_conversion_message = conversion_message
        self.stock_part.role = PartRole.STOCK
        self.stock_mesh = mesh
        self.stock_imported = imported
        self._refresh_parts()
        self._refresh_note_targets()
        self._refresh_scene()

    def _refresh_parts(self) -> None:
        current_row = self.parts_list.currentRow()
        self.parts_list.blockSignals(True)
        self.parts_list.clear()
        for index, part in enumerate(self.parts):
            item = QListWidgetItem(f"{part.role.value}  ·  {part.geometry_type}  ·  {part.name}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            tooltip = f"{part.name}\nType: {part.geometry_type}"
            if part.solid_conversion_message:
                tooltip += f"\n{part.solid_conversion_message}"
            item.setToolTip(tooltip)
            self.parts_list.addItem(item)
        if self.parts:
            self.parts_list.setCurrentRow(min(max(current_row, 0), len(self.parts) - 1))
        self.parts_list.blockSignals(False)
        if self.parts:
            self._select_part_row(self.parts_list.currentRow())

    def _select_part_row(self, row: int) -> None:
        if 0 <= row < len(self.parts):
            self._select_whole_object(row)

    def _show_parts_context_menu(self, position: QPoint) -> None:
        item = self.parts_list.itemAt(position)
        if item is None:
            return
        self.parts_list.setCurrentItem(item)
        menu = QMenu(self.parts_list)
        select_action = menu.addAction("Select whole object")
        delete_action = menu.addAction("Delete object")
        role_menu = menu.addMenu("Set role")
        role_actions = {role_menu.addAction(role.value): role for role in PartRole}
        selected_action = menu.exec(self.parts_list.mapToGlobal(position))
        if selected_action == select_action:
            self._select_whole_object(self.parts_list.row(item))
        elif selected_action == delete_action:
            self.delete_selected_part()
        elif selected_action in role_actions:
            self._set_part_role(self.parts_list.row(item), role_actions[selected_action])

    def _set_part_role(self, part_index: int, role: PartRole) -> None:
        if not 0 <= part_index < len(self.parts):
            return
        part = self.parts[part_index]
        if role is PartRole.STOCK:
            if self.stock_part is not None and self.stock_part is not part:
                self.stock_part.role = PartRole.TARGET
            self.stock_part = part
            self.stock_mesh = part.mesh
            self.stock_imported = part.imported
            self.stock_status.setText(f"Stock · {part.name}")
        elif self.stock_part is part:
            self.stock_part = None
            self.stock_mesh = None
            self.stock_imported = False
            self.stock_status.setText("No stock geometry")
        part.role = role
        self._refresh_parts()
        self._refresh_scene(reset_camera=False)
        self.statusBar().showMessage(f"{part.name} role set to {role.value}", 4000)

    def _select_whole_object(self, part_index: int) -> None:
        if not 0 <= part_index < len(self.parts):
            return
        part = self.parts[part_index]
        self.selected_cells = {part_index: set(range(part.mesh.n_cells))}
        self.selected_feature = None
        self.selected_features.clear()
        self.operation_target = f"{part.name} · Whole object"
        self.target_label.setText(f"Target: {self.operation_target}")
        note_data = ("part", part_index, -1)
        note_index = self.note_target.findData(note_data)
        if note_index >= 0:
            self.note_target.setCurrentIndex(note_index)
        self.statusBar().showMessage(f"Selected whole object: {part.name}", 3000)
        self._refresh_scene(reset_camera=False)

    def delete_selected_part(self) -> None:
        index = self.parts_list.currentRow()
        if not 0 <= index < len(self.parts):
            return
        removed = self.parts.pop(index)
        if removed is self.stock_part:
            self.stock_part = None
            self.stock_mesh = None
            self.stock_imported = False
            self.stock_status.setText("No stock geometry")
        removed_operation_count = sum(
            operation.target.startswith(f"{removed.name} ·") for operation in self.operations
        )
        for setup in self.setups:
            setup.operations = [
                operation
                for operation in setup.operations
                if not operation.target.startswith(f"{removed.name} ·")
            ]
            for op_num, operation in enumerate(setup.operations, start=1):
                operation.name = f"{op_num:02d} · {operation.kind}"
        self.known_feature_targets = {
            (part_index - 1 if part_index > index else part_index, feature_id)
            for part_index, feature_id in self.known_feature_targets
            if part_index != index
        }
        self.selected_cells = {
            (part_index - 1 if part_index > index else part_index): cells
            for part_index, cells in self.selected_cells.items()
            if part_index != index
        }
        self.selected_feature = None
        self.selected_features = {
            (part_index - 1 if part_index > index else part_index, feature_id)
            for part_index, feature_id in self.selected_features
            if part_index != index
        }
        self.operation_target = "Whole part"
        self.target_label.setText("Target: Whole part")
        self._refresh_parts()
        self._refresh_note_targets()
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)
        message = f"Deleted {removed.name}"
        if removed_operation_count:
            message += f" and {removed_operation_count} assigned operation(s)"
        self.statusBar().showMessage(message, 5000)

    @staticmethod
    def _hex_to_rgb(hex_color: str, fallback: tuple[int, int, int] = (194, 223, 84)) -> tuple[int, int, int]:
        qcolor = QColor(hex_color)
        if not qcolor.isValid():
            return fallback
        return (qcolor.red(), qcolor.green(), qcolor.blue())

    def _setup_color_for_feature(self, part_index: int, feature_id: int) -> str | None:
        target_exact = f"Part {part_index + 1} · Feature {feature_id}"
        for setup in self.setups:
            for op in setup.operations:
                if op.target == target_exact or op.target.endswith(f"Feature {feature_id}"):
                    return setup.color
        return None

    def _build_smooth_display_mesh(self, mesh: pv.PolyData, smooth: bool) -> pv.PolyData:
        display_mesh = mesh.copy(deep=True)
        if smooth and display_mesh.n_points > 0 and display_mesh.n_cells > 0:
            try:
                display_mesh = display_mesh.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    split_vertices=False,
                    feature_angle=45.0,
                    auto_orient_normals=True,
                    inplace=False,
                )
            except Exception:
                pass
        return display_mesh

    def _record_relative_move(self, axis: str, delta: float) -> None:
        axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis.upper())
        if axis_idx is not None:
            self.tracked_work_pos[axis_idx] += float(delta)
        self._update_dro_readout()

    JOG_STEPS_MM = {"medium": 1.0, "fine": 0.1, "large": 10.0}
    JOG_STEPS_IN = {"medium": 0.1, "fine": 0.01, "large": 1.0}

    def _view_is_square(self, tolerance_deg: float = 2.0) -> bool:
        """Arrow jogging only when the camera is within ~2 degrees of normal to a principal or selected face."""
        cam = self.plotter.camera
        d = np.asarray(cam.focal_point, dtype=float) - np.asarray(cam.position, dtype=float)
        d /= max(np.linalg.norm(d), 1e-12)
        normals = [np.eye(3)[i] for i in range(3)]
        if self.selected_features:
            face = self._feature_face(*sorted(self.selected_features)[0])
            if face is not None:
                normals.append(face[1])
        limit = math.cos(math.radians(tolerance_deg))
        return any(abs(float(np.dot(d, n))) >= limit for n in normals)

    def _screen_axes(self) -> tuple[np.ndarray, np.ndarray]:
        cam = self.plotter.camera
        d = np.asarray(cam.focal_point, dtype=float) - np.asarray(cam.position, dtype=float)
        d /= max(np.linalg.norm(d), 1e-12)
        up = np.asarray(cam.up, dtype=float)
        up = up - np.dot(up, d) * d
        up /= max(np.linalg.norm(up), 1e-12)
        right = np.cross(d, up)

        def snap(v):
            index = int(np.argmax(np.abs(v)))
            out = np.zeros(3)
            out[index] = math.copysign(1.0, v[index])
            return out if abs(v[index]) > 0.999 else v

        return snap(right), snap(up)

    def _jog_step_mm(self, modifiers) -> float:
        size = "medium"
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            size = "fine"
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            size = "large"
        if self.units.display is LengthUnit.INCHES:
            return self.JOG_STEPS_IN[size] * 25.4
        return self.JOG_STEPS_MM[size]

    def _confirm_keyboard_move(self, key: str) -> bool:
        """Extra confirmation for keyboard operations (once per selection within 10 s)."""
        if not self.extra_keyboard_confirm:
            return True
        last = getattr(self, "_keyboard_confirmed", None)
        if last is not None and last[0] == key and time.monotonic() - last[1] < 10.0:
            self._keyboard_confirmed = (key, time.monotonic())
            return True
        answer = QMessageBox.question(self, "Keyboard move", f"Move {key} with the arrow keys?")
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._keyboard_confirmed = (key, time.monotonic())
        return True

    def _jog_selection(self, dx: int, dy: int, step_mm: float) -> bool:
        """Arrow keys: move the selected part (or sketch entity) in the screen plane."""
        if not self._view_is_square():
            self.statusBar().showMessage("Arrow moves need a view square to a face (within 2°); use a standard view.", 4000)
            return False
        right, up = self._screen_axes()
        delta = (right * dx + up * dy) * step_mm
        index = self._position_part_index() if self.parts else None
        if index is not None and (self.parts_list.currentRow() >= 0 or len(self.parts) == 1):
            part = self.parts[index]
            if not self._confirm_keyboard_move(part.name):
                return False
            self._apply_transform(part, transform_part(part, matrix_translate(delta)), "Move part")
            return True
        feat = self.sketch_workflow.get_feature(self.active_sketch_id) if self.active_sketch_id else None
        if feat is not None and self.sketch_workflow.can_edit(feat):
            if not self._confirm_keyboard_move(feat.name):
                return False
            self._push_history("Move sketch entity")
            feat.points = np.asarray(feat.points, dtype=float) + delta
            if feat.projected_points is not None:
                feat.projected_points = np.asarray(feat.projected_points, dtype=float) + delta
            self._refresh_sketch_tree()
            self._refresh_scene(reset_camera=False)
            return True
        self.statusBar().showMessage("Select a part or sketch entity to move.", 3000)
        return False

    def _apply_transform(self, part, result: TransformResult, label: str = "Move part") -> None:
        """Replace the part's geometry with new transformed geometry; dependents follow (one undo step)."""
        self._push_history(label)
        index = self.parts.index(part)
        old_mesh = part.mesh
        old_bounds = np.asarray(old_mesh.bounds, dtype=float) if old_mesh is not None and old_mesh.n_points else None
        m = result.matrix
        part.mesh, part.edge_mesh = result.mesh, result.edge_mesh
        part.brep_shape, part.solid_shape = result.brep_shape, result.solid_shape
        if part is self.stock_part:
            self.stock_mesh = part.mesh
        # stock boxes that contained the part follow it
        if old_bounds is not None and part is not self.stock_part:
            for other in self.parts:
                if other is part or other.role is not PartRole.STOCK or other.mesh is None or not other.mesh.n_points:
                    continue
                ob = np.asarray(other.mesh.bounds, dtype=float)
                if all(ob[2 * i] <= old_bounds[2 * i] + 1e-6 and ob[2 * i + 1] >= old_bounds[2 * i + 1] - 1e-6 for i in range(3)):
                    moved = transform_part(other, m)
                    other.mesh, other.edge_mesh = moved.mesh, moved.edge_mesh
                    other.brep_shape, other.solid_shape = moved.brep_shape, moved.solid_shape
                    if other is self.stock_part:
                        self.stock_mesh = other.mesh
        # sketches drawn on this part, and anchors to its geometry
        manager = self.sketch_workflow
        owners = {sid for sid, info in manager.sketches.items() if info.get("part_index") == index}
        for feat in manager.features:
            if feat.owner_sketch_id in owners:
                feat.points = apply_to_points(m, feat.points)
                if feat.projected_points is not None:
                    feat.projected_points = apply_to_points(m, feat.projected_points)
        if old_mesh is not None and old_mesh.n_points:
            verts = np.asarray(old_mesh.points)
            for rel in manager.relations:
                if rel.value and rel.kind in ("anchor_model", "fix"):
                    if float(np.min(np.linalg.norm(verts - np.asarray(rel.value), axis=1))) < 1e-6:
                        rel.value = apply_to_point(m, rel.value).tolist()
        self.plotter.remove_actor("position-preview", render=False)
        self._check_bed_fit(part)
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)
        self.statusBar().showMessage(f"{label}: {part.name}", 3000)

    def _set_work_zero_reference(self) -> None:
        if self.grbl_status and self.grbl_status.machine_position:
            self.zero_reference_mpos = tuple(float(v) for v in self.grbl_status.machine_position)
        else:
            self.zero_reference_mpos = (0.0, 0.0, 0.0)
        self.tracked_work_pos = [0.0, 0.0, 0.0]
        self.work_origin_set = True
        if self.grbl_controller is not None:
            self._send_machine_line("G10 L20 P1 X0 Y0 Z0")
        self._update_dro_readout()

    def _update_dro_readout(self) -> None:
        if not hasattr(self, "dro_label"):
            return
        if self.grbl_status and self.grbl_status.machine_position:
            mx, my, mz = self.grbl_status.machine_position
            zx, zy, zz = self.zero_reference_mpos or (0.0, 0.0, 0.0)
            dx, dy, dz = mx - zx, my - zy, mz - zz
            self.tracked_work_pos = [dx, dy, dz]
            source = "MPos"
        else:
            dx, dy, dz = self.tracked_work_pos
            source = "est"
        self.dro_label.setText(
            f"DRO ({source}) ΔX: {dx:+.3f}  ΔY: {dy:+.3f}  ΔZ: {dz:+.3f} mm"
        )

    def _execute_manual_cut(self) -> None:
        if self.grbl_controller is None or not self.position_verified:
            self.statusBar().showMessage("Connect, verify position, and set zero before running a manual cut.", 5000)
            return
        axis_dir = self.manual_cut_dir.currentText()
        axis = axis_dir[0]
        sign = 1.0 if axis_dir.endswith("+") else -1.0
        delta = sign * float(self.manual_cut_distance.value())
        feed = float(self.manual_cut_feed.value())
        rpm = int(self.manual_cut_rpm.value())
        manual_spindle = self.manual_cut_spindle_running_box.isChecked()

        cmds: list[str] = []
        if not manual_spindle and rpm > 0:
            cmds.append(f"M3 S{rpm}")
        cmds.append("G91")
        cmds.append(f"G1 {axis}{delta:.3f} F{feed:.1f}")
        cmds.append("G90")

        confirm = QMessageBox.question(
            self,
            "Confirm Manual Cut",
            "Send the following manual cut commands to the controller?\n\n" + "\n".join(cmds),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            for line in cmds:
                self._send_machine_line(line)
            self._record_relative_move(axis, delta)
    def closeEvent(self, event) -> None:
        self.handle_clean_shutdown()
        super().closeEvent(event)
    def _scene_extent(self) -> float:
        extent = 0.0
        for part in self.parts:
            if part.mesh is not None and part.mesh.n_cells > 0:
                b = np.asarray(part.mesh.bounds, dtype=float)
                extent = max(extent, float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])))
        return extent if extent > 0 else 100.0

    def _axis_colors(self) -> tuple[str, str, str]:
        # Colour-blind palette (Okabe-Ito) is applied through the theme; X is orange, never red.
        return (theme_color("axis_x"), theme_color("axis_y"), theme_color("axis_z"))

    def _toggle_colorblind_axes(self, enabled: bool) -> None:
        """Switch the whole palette (axes, setups, highlights, error/warning) to colour-blind safe colours."""
        self.settings.setValue("view/colorblind_axes", bool(enabled))
        self._apply_theme()

    def _apply_theme(self) -> None:
        activate_theme(self.settings)
        self.setStyleSheet(self._stylesheet())
        if hasattr(self, "terminal_core"):
            self._sync_terminal_colors()
        self.plotter.set_background(theme_color("viewport_bg"))
        if hasattr(self, "tool_preview_plotter"):
            self.tool_preview_plotter.set_background(theme_color("viewport_bg"))
        self._update_status_indicator(
            self.grbl_controller is not None,
            self.grbl_capabilities is not None,
            self.grbl_status.state if self.grbl_status is not None else "",
        )
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)

    @staticmethod
    def _sync_terminal_colors() -> None:
        from basicam import terminal as terminal_module

        terminal_module.LEVEL_COLORS.update(
            {"warning": theme_color("warning"), "error": theme_color("error"), "info": theme_color("text")}
        )

    def _add_origin_axes(self, axis_lengths=None) -> None:
        base = max(self._scene_extent() * 0.4, 25.0)
        lengths = axis_lengths if axis_lengths is not None else (base, base, base)
        for index, (axis, color) in enumerate(zip("XYZ", self._axis_colors())):
            end = [0.0, 0.0, 0.0]
            end[index] = float(lengths[index])
            actor = self.plotter.add_mesh(
                pv.Line((0, 0, 0), tuple(end)), color=color, line_width=GRID_LINE_WIDTH,
                name=f"origin-axis-{axis}", pickable=False,
            )
            if actor is not None:
                actor.SetPickable(False)
        sphere = pv.Sphere(radius=base * 0.0075, center=(0, 0, 0))
        actor = self.plotter.add_mesh(
            sphere, color=theme_color("origin"), smooth_shading=True, name="origin-point", pickable=False
        )
        if actor is not None:
            actor.SetPickable(False)

    @staticmethod
    def _nice_step(span: float) -> float:
        raw = max(span, 1e-6) / 8.0
        magnitude = 10 ** math.floor(math.log10(raw))
        for factor in (1, 2, 5, 10):
            if raw <= factor * magnitude:
                return factor * magnitude
        return 10 * magnitude

    def _grid_bounds_and_labels(self) -> tuple[list[float], list[int]]:
        """Grid bounds on round multiples of one step so the grid always contains/starts at 0."""
        lows = np.zeros(3)
        highs = np.zeros(3)
        axis_len = max(self._scene_extent() * 0.4, 25.0)
        highs[:] = axis_len
        for part in self.parts:
            if part.mesh is not None and part.mesh.n_cells > 0:
                b = np.asarray(part.mesh.bounds, dtype=float)
                lows = np.minimum(lows, b[0::2])
                highs = np.maximum(highs, b[1::2])
        step = self._nice_step(float((highs - lows).max()))
        bounds: list[float] = []
        labels: list[int] = []
        for low, high in zip(lows, highs):
            lo = math.floor(low / step) * step
            hi = math.ceil(high / step) * step
            if hi <= lo:
                hi = lo + step
            bounds += [lo, hi]
            labels.append(int(round((hi - lo) / step)) + 1)
        return bounds, labels

    def _grid_wall_lines(self, bounds: list[float], step: float) -> pv.PolyData:
        """Grid lines on the three walls at the lower bounds.

        Segments that lie on a positive coloured axis are left out so only the coloured
        axis line exists there; the negative side of an axis keeps its grey grid line.
        """
        lo = np.array(bounds[0::2], dtype=float)
        hi = np.array(bounds[1::2], dtype=float)
        ticks = [np.arange(lo[k], hi[k] + step / 2, step) for k in range(3)]
        points: list[np.ndarray] = []
        cells: list[int] = []
        seen: set[tuple] = set()

        def add_segment(a: np.ndarray, b: np.ndarray) -> None:
            on_axis = [abs(a[k]) < 1e-9 and abs(b[k]) < 1e-9 for k in range(3)]
            if sum(on_axis) == 2:
                k = on_axis.index(False)
                low, high = sorted((a[k], b[k]))
                if low >= 0:
                    return  # fully on the coloured axis
                if high > 0:
                    a, b = a.copy(), b.copy()
                    a[k], b[k] = low, 0.0
            key = tuple(np.round(np.concatenate([np.minimum(a, b), np.maximum(a, b)]), 6))
            if key in seen:
                return
            seen.add(key)
            base = len(points)
            points.extend([a, b])
            cells.extend([2, base, base + 1])

        for wall in range(3):
            u, v = [k for k in range(3) if k != wall]
            for t in ticks[u]:
                a, b = np.zeros(3), np.zeros(3)
                a[wall] = b[wall] = lo[wall]
                a[u] = b[u] = t
                a[v], b[v] = lo[v], hi[v]
                add_segment(a, b)
            for t in ticks[v]:
                a, b = np.zeros(3), np.zeros(3)
                a[wall] = b[wall] = lo[wall]
                a[v] = b[v] = t
                a[u], b[u] = lo[u], hi[u]
                add_segment(a, b)
        mesh = pv.PolyData()
        if points:
            mesh.points = np.array(points)
            mesh.lines = np.array(cells)
        return mesh

    def _add_reference_grid_and_axes(self) -> None:
        if hasattr(self, "grid_toggle") and self.grid_toggle.isChecked():
            bounds, labels = self._grid_bounds_and_labels()
            step = (bounds[1] - bounds[0]) / max(labels[0] - 1, 1)
            self._add_origin_axes((bounds[1], bounds[3], bounds[5]))
            lines = self._grid_wall_lines(bounds, step)
            if lines.n_cells:
                actor = self.plotter.add_mesh(
                    lines, color=theme_color("grid"), line_width=GRID_LINE_WIDTH, name="reference-grid", pickable=False
                )
                if actor is not None:
                    actor.SetPickable(False)
            # VTK's cube axes are kept only for the number labels: no grid, axis lines hidden.
            self.plotter.show_grid(
                grid=False, color=theme_color("grid_label"), xtitle="X", ytitle="Y", ztitle="Z",
                bounds=bounds, n_xlabels=labels[0], n_ylabels=labels[1], n_zlabels=labels[2],
                minor_ticks=False, fmt="%g", font_size=self._grid_font_px(),
            )
            cube = self.plotter.renderer.cube_axes_actor
            if cube is not None:
                # Constant on-screen size for grid numbers (Settings > Interface > Font).
                try:
                    cube.SetUseTextActor3D(False)
                    cube.SetScreenSize(float(self._grid_font_px()))
                    for axis in range(3):
                        cube.GetLabelTextProperty(axis).SetFontSize(self._grid_font_px())
                        cube.GetTitleTextProperty(axis).SetFontSize(self._grid_font_px())
                except AttributeError:
                    pass
                for prop in (
                    cube.GetXAxesLinesProperty(), cube.GetYAxesLinesProperty(), cube.GetZAxesLinesProperty()
                ):
                    prop.SetOpacity(0.0)
        else:
            self._add_origin_axes()

    def _grid_font_px(self) -> int:
        try:
            return max(6, min(48, int(self.settings.value("interface/grid_font_px", 11))))
        except (TypeError, ValueError):
            return 11

    def _toggle_grid(self, _visible: bool) -> None:
        self._refresh_scene(reset_camera=False)
        
    def _extract_non_coplanar_edges(self, mesh: pv.PolyData) -> pv.PolyData | None:
        if mesh is None or mesh.n_cells == 0:
            return None
        try:
            clean = mesh.clean(point_merging=True, tolerance=1e-4, inplace=False)
            with_normals = clean.compute_normals(
                cell_normals=True,
                point_normals=False,
                auto_orient_normals=False,
                inplace=False,
            )
            edges = with_normals.extract_feature_edges(
                boundary_edges=True,
                feature_edges=True,
                manifold_edges=False,
                non_manifold_edges=False,
                feature_angle=35.0,
            )
            if edges is not None and edges.n_cells > 0:
                return edges
        except Exception:
            pass
        return None
    def build_rough_tool_mesh(
        diameter_mm: float = 6.0,
        length_mm: float = 25.0,
        tool_type: str = "Endmill",
    ) -> pv.PolyData:
        radius = max(float(diameter_mm) * 0.5, 0.05)
        height = max(float(length_mm), radius * 2.0)
        # Cylinder in PyVista defaults to axis along X; rotate 90 deg around Y to align with Z
        body = pv.Cylinder(
        center=(0.0, 0.0, height * 0.5),
        direction=(0.0, 0.0, 1.0),
        radius=radius,
        height=height,
        resolution=24,
        capping=True,
        )
        if tool_type.casefold() in {"drill", "center drill", "spot drill"}:
            tip_height = min(radius * 1.2, height * 0.35)
            tip = pv.Cone(
                center=(0.0, 0.0, -tip_height * 0.5),
                direction=(0.0, 0.0, -1.0),
                height=tip_height,
                radius=radius,
                resolution=24,
                capping=True,
            )
            return body.merge(tip)
        return body

    def _orientation_rotation_matrix(self, orientation: str) -> np.ndarray:
        # Maps tool +Z axis to desired setup tool axis
        axis_map = {
            "Top (+Z)": np.array([0.0, 0.0, 1.0]),
            "Bottom (-Z)": np.array([0.0, 0.0, -1.0]),
            "Front (-Y)": np.array([0.0, -1.0, 0.0]),
            "Back (+Y)": np.array([0.0, 1.0, 0.0]),
            "Left (-X)": np.array([-1.0, 0.0, 0.0]),
            "Right (+X)": np.array([1.0, 0.0, 0.0]),
        }
        target = axis_map.get(orientation, np.array([0.0, 0.0, 1.0]))
        z_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(target, z_axis):
            return np.eye(4)
        if np.allclose(target, -z_axis):
            return np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        rot_axis = np.cross(z_axis, target)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = math.acos(float(np.clip(np.dot(z_axis, target), -1.0, 1.0)))
        c, s = math.cos(angle), math.sin(angle)
        x, y, z = rot_axis
        r3 = np.array(
            [
                [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
            ]
        )
        mat = np.eye(4)
        mat[:3, :3] = r3
        return mat
    
    def _find_tool_definition(self, op: Operation) -> ToolDefinition:
        for tool in self.tools:
            if tool.name == op.tool:
                return tool
        return ToolDefinition(
            name=op.tool or "End mill",
            tool_type="End mill",
            diameter_mm=float(op.tool_diameter_mm or 6.0),
            shank_diameter_mm=float(op.tool_diameter_mm or 6.0),
        )

    def _draw_stock_outline(self, part_index: int, part_model) -> None:
        """Draw a stock/billet as clean outline edges only (no triangle wireframe, no surface)."""
        mesh = part_model.mesh
        self.stock_dataset_keys.add(self._vtk_address(mesh))
        self.dataset_part_indexes[self._vtk_address(mesh)] = part_index
        outline = None
        edge_mesh = getattr(part_model, "edge_mesh", None)
        if edge_mesh is not None and edge_mesh.n_cells > 0:
            outline = edge_mesh
        else:
            try:
                outline = mesh.extract_feature_edges(
                    feature_angle=30.0,
                    boundary_edges=True,
                    feature_edges=True,
                    manifold_edges=False,
                    non_manifold_edges=False,
                )
            except Exception:
                outline = None
        if outline is None or outline.n_cells == 0:
            return
        self.stock_dataset_keys.add(self._vtk_address(outline))
        self.dataset_part_indexes[self._vtk_address(outline)] = part_index
        self.plotter.add_mesh(
            outline,
            color="#8c9a9e",
            opacity=0.6,
            line_width=1.4,
            name=f"stock-outline-{part_index}",
        )

    def _lighting_value(self, key: str) -> int:
        try:
            return int(self.settings.value(f"view/light_{key}", LIGHTING_DEFAULTS[key]))
        except (TypeError, ValueError):
            return LIGHTING_DEFAULTS[key]

    def _current_lighting(self):
        return lighting_params(*(self._lighting_value(k) for k in ("brightness", "contrast", "azimuth", "elevation")))

    def _apply_lighting(self, render: bool = True) -> None:
        """Replace the default lights with an off-screen key light plus fill that follow the camera."""
        params = self._current_lighting()
        self.plotter.remove_all_lights()
        key = pv.Light(position=params.key_position, focal_point=(0, 0, 0), light_type="camera light")
        key.intensity = params.key_intensity
        self.plotter.add_light(key)
        fill_dir = (-params.key_position[0], -0.3 * params.key_position[1], params.key_position[2])
        fill = pv.Light(position=fill_dir, focal_point=(0, 0, 0), light_type="camera light")
        fill.intensity = params.fill_intensity
        self.plotter.add_light(fill)
        for name, actor in self.plotter.renderer.actors.items():
            if str(name).startswith("part-surface-") and actor.prop.lighting:
                actor.prop.ambient = params.ambient
                actor.prop.diffuse = params.diffuse
                actor.prop.specular = params.specular
        if render:
            self.plotter.render()

    def _show_lighting_panel(self) -> None:
        """Non-modal sliders for brightness, contrast and light direction."""
        panel = getattr(self, "_lighting_panel", None)
        if panel is not None:
            panel.show()
            panel.raise_()
            return
        panel = QDialog(self)
        panel.setWindowTitle("Lighting")
        form = QFormLayout(panel)
        specs = (
            ("brightness", "Brightness", 10, 200, " %", "Overall light level."),
            ("contrast", "Contrast", 0, 100, " %", "Higher contrast makes faces at different angles easier to tell apart."),
            ("azimuth", "Direction", -180, 180, "°", "Where the light comes from, around the view."),
            ("elevation", "Height", -80, 80, "°", "Light from above (positive) or below (negative) the view."),
        )
        self._lighting_sliders = {}
        for key, title, lo, hi, suffix, tip in specs:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(self._lighting_value(key))
            slider.setToolTip(tip)
            readout = QLabel(f"{slider.value()}{suffix}")
            readout.setFixedWidth(48)

            def changed(value, k=key, r=readout, s=suffix):
                r.setText(f"{value}{s}")
                self.settings.setValue(f"view/light_{k}", int(value))
                self._apply_lighting()

            slider.valueChanged.connect(changed)
            row = QHBoxLayout()
            row.addWidget(slider, 1)
            row.addWidget(readout)
            form.addRow(title, row)
            self._lighting_sliders[key] = slider
        reset = QPushButton("Reset lighting")

        def reset_all():
            for key, slider in self._lighting_sliders.items():
                slider.setValue(LIGHTING_DEFAULTS[key])

        reset.clicked.connect(reset_all)
        form.addRow(reset)
        self._lighting_panel = panel
        panel.show()

    def _on_display_mode_changed(self, mode: str) -> None:
        """Through-selection is only offered when hidden geometry can be seen."""
        see_through = not mode_flags(mode).show_surface or mode_flags(mode).opacity < 1.0 or mode_flags(mode).surface_is_mask
        if hasattr(self, "through_toggle"):
            self.through_toggle.setEnabled(see_through)
            if not see_through:
                self.through_toggle.setChecked(False)
        self._refresh_scene(reset_camera=False)

    def _hidden_edge_renderer(self):
        """Overlay renderer (layer 1) that draws dashed hidden edges on top of the scene."""
        renderer = getattr(self, "_hidden_layer", None)
        if renderer is None:
            from vtkmodules.vtkRenderingCore import vtkRenderer

            ren_win = self.plotter.ren_win
            ren_win.SetNumberOfLayers(max(2, ren_win.GetNumberOfLayers()))
            renderer = vtkRenderer()
            renderer.SetLayer(1)
            renderer.InteractiveOff()
            renderer.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
            ren_win.AddRenderer(renderer)
            self._hidden_layer = renderer
        renderer.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
        return renderer

    def _clear_hidden_edge_layer(self) -> None:
        renderer = getattr(self, "_hidden_layer", None)
        if renderer is not None:
            renderer.RemoveAllViewProps()
        self.hidden_edge_meshes: list[pv.PolyData] = []

    def _add_hidden_edges(self, edges: pv.PolyData, part_index: int) -> None:
        bounds = np.asarray(edges.bounds, dtype=float)
        size = float(np.linalg.norm(bounds[1::2] - bounds[0::2])) or 10.0
        dashed = dash_lines(edges, dash=size / 120.0, gap=size / 160.0)
        if dashed.n_cells == 0:
            return
        try:
            from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(dashed)
            actor = vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*[c / 255.0 for c in self._hex_to_rgb(theme_color("edge_hidden"))])
            prop.SetLineWidth(1.0)
            prop.SetOpacity(0.8)
            actor.PickableOff()
            self._hidden_edge_renderer().AddActor(actor)
            self.hidden_edge_meshes.append(dashed)
        except Exception as error:  # rendering helper only; scene still shows visible edges
            self.terminal_core.log("warning", f"Hidden-line overlay unavailable: {error}")

    def _refresh_scene(self, reset_camera: bool = True) -> None:
        self.plotter.clear()
        self.dataset_part_indexes.clear()
        self.stock_dataset_keys.clear()
        self._add_reference_grid_and_axes()

        mode = normalize_mode(self.display_mode.currentText() if hasattr(self, "display_mode") else "Shaded")
        flags = mode_flags(mode)
        opacity = flags.opacity
        show_surface = flags.show_surface
        show_edges = flags.show_edges
        self._clear_hidden_edge_layer()

        for part_index, part_model in enumerate(self.parts):
            mesh = part_model.mesh
            if mesh is None or mesh.n_cells == 0:
                continue
            if part_model.role is PartRole.STOCK:
                self._draw_stock_outline(part_index, part_model)
                continue

            self.dataset_part_indexes[self._vtk_address(mesh)] = part_index
            has_brep_edges = (
                getattr(part_model, "edge_mesh", None) is not None
                and part_model.edge_mesh.n_cells > 0
            )

            # Default shaded grey part appearance; setup colors only on assigned features
            default_grey = np.array(self._hex_to_rgb(theme_color("face")), dtype=np.uint8)
            cell_colors = np.tile(default_grey, (mesh.n_cells, 1))

            if "feature_id" in mesh.cell_data:
                fids = mesh.cell_data["feature_id"]
                for fid in np.unique(fids):
                    setup_hex = self._setup_color_for_feature(part_index, int(fid))
                    if setup_hex:
                        cell_colors[fids == fid] = np.array(
                            self._hex_to_rgb(setup_hex), dtype=np.uint8
                        )

            active_cells: set[int] = set(self.selected_cells.get(part_index, set()))
            if (
                self.selected_feature is not None
                and self.selected_feature[0] == part_index
                and "feature_id" in mesh.cell_data
            ):
                active_cells.update(
                    int(idx)
                    for idx in np.flatnonzero(mesh.cell_data["feature_id"] == self.selected_feature[1])
                )
            if active_cells:
                sel_rgb = np.array(
                    self._hex_to_rgb(theme_color("selection")), dtype=np.uint8
                )
                valid_idx = [i for i in active_cells if 0 <= i < mesh.n_cells]
                if valid_idx:
                    cell_colors[valid_idx] = sel_rgb

            if show_surface:
                display_mesh = self._build_smooth_display_mesh(mesh, smooth=flags.lighting)
                self.dataset_part_indexes[self._vtk_address(display_mesh)] = part_index
                if flags.surface_is_mask:
                    # Hidden lines: opaque background-coloured surface hides edges behind it.
                    self.plotter.add_mesh(
                        display_mesh,
                        color=theme_color("viewport_bg"),
                        lighting=False,
                        show_edges=False,
                        name=f"part-surface-{part_index}",
                    )
                else:
                    display_mesh.cell_data["setup_rgb"] = cell_colors
                    self.plotter.add_mesh(
                        display_mesh,
                        scalars="setup_rgb",
                        rgb=True,
                        opacity=opacity,
                        smooth_shading=flags.lighting,
                        lighting=flags.lighting,
                        show_edges=False,  # never draw triangle edges; only real face boundaries below
                        name=f"part-surface-{part_index}",
                    )

            if show_edges:
                # Real face boundaries: native B-rep edges, else edges between non-coplanar facets.
                edge_source = (
                    part_model.edge_mesh
                    if has_brep_edges
                    else self._extract_non_coplanar_edges(mesh)
                )
                if edge_source is not None and edge_source.n_cells > 0:
                    self.dataset_part_indexes[self._vtk_address(edge_source)] = part_index
                    self.plotter.add_mesh(
                        edge_source,
                        color=theme_color("edge"),
                        opacity=1.0 if flags.surface_is_mask or not show_surface else 0.72,
                        line_width=1.4 if has_brep_edges else 1.1,
                        name=f"part-edges-{part_index}",
                    )
                    if flags.dashed_hidden_edges:
                        self._add_hidden_edges(edge_source, part_index)

        self._draw_sketch_entities()
        self._draw_schematic_path()
        self._draw_ghost_tool()
        self._apply_lighting(render=False)

        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()
        
    def _prompt_import_metadata(self, default_part_name: str, replace_workspace: bool) -> tuple[str, str, str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Part & Job Setup")
        form = QFormLayout(dialog)

        job_input = QLineEdit(self.project_name_label.text())
        job_input.setToolTip("Name of the overall machining job/project")
        form.addRow("Job name", job_input)

        part_input = QLineEdit(default_part_name)
        part_input.setToolTip("Display name for this imported part")
        form.addRow("Part name", part_input)

        material_combo = QComboBox()
        material_combo.addItems(
            [
                "Aluminum",
                "Brass",
                "Bronze",
                "Mild steel",
                "Stainless steel",
                "Plastic",
                "Wood (soft)",
                "Wood (hard)",
                "Other",
                "Unspecified",
            ]
        )
        current_mat = self.stock_material.currentText()
        if current_mat and material_combo.findText(current_mat) >= 0:
            material_combo.setCurrentText(current_mat)
        material_combo.setToolTip("Stock material used for cutting parameter recommendations")
        form.addRow("Material", material_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        job_name = job_input.text().strip() or ("Untitled job" if replace_workspace else self.project_name_label.text())
        part_name = part_input.text().strip() or default_part_name
        return job_name, part_name, material_combo.currentText()

    def _draw_schematic_path(self) -> None:
        if not self._has_any_operations() or not self.parts:
            return
        bounds = self.parts[0].mesh.bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        inset_x = (xmax - xmin) * 0.15
        inset_y = (ymax - ymin) * 0.15
        for s_idx, setup in enumerate(self.setups):
            if not setup.operations:
                continue
            ox, oy, oz = setup.origin_mm
            z = oz + zmax + max((zmax - zmin) * 0.04, 0.5) + s_idx * 0.4
            points = np.array(
                [
                    [ox + xmin + inset_x, oy + ymin + inset_y, z],
                    [ox + xmax - inset_x, oy + ymin + inset_y, z],
                    [ox + xmax - inset_x, oy + ymax - inset_y, z],
                    [ox + xmin + inset_x, oy + ymax - inset_y, z],
                    [ox + xmin + inset_x, oy + ymin + inset_y, z],
                ]
            )
            path = pv.lines_from_points(points)
            self.plotter.add_mesh(
                path,
                color=setup.color,
                line_width=2.0,
                name=f"preview-path-setup-{s_idx}",
            )
            last_op = setup.operations[-1]
            tool_def = self._find_tool_definition(last_op)
            try:
                tool_mesh = create_tool_mesh(tool_def, resolution=24)
                tool_mesh.translate(
                    (ox + xmin + inset_x, oy + ymin + inset_y, z),
                    inplace=True,
                )
                self.plotter.add_mesh(
                    tool_mesh,
                    color=setup.color,
                    opacity=0.65 if s_idx == self.active_setup_index else 0.35,
                    show_edges=False,
                    name=f"preview-tool-setup-{s_idx}",
                )
            except Exception:
                pass

    def add_operation(self) -> None:
        target_feat = next((f for f in self.sketch_workflow.features if f.name == self.operation_target), None)
        if target_feat is not None:
            self._solve_sketches()
            allowed, reason = can_assign_tool(target_feat)
            if not allowed:  # tools only on fully defined sketch entities
                self.terminal_core.log("warning", f"Operation not added: {reason}")
                QMessageBox.warning(self, "Cannot assign a tool", reason)
                return
        self._push_history("Add operation")
        count = len(self.operations) + 1
        kind = self.operation_kind.currentText()
        operation = Operation(
            name=f"{count:02d} · {kind}",
            kind=kind,
            tool=self.tool_name.currentText(),
            rpm=self.spindle.value(),
            feed_mm_min=self.feed.value(),
            stepdown_mm=self.stepdown.value(),
            depth_mm=self.depth.value(),
            direction=self.direction.currentText().casefold(),
            tool_diameter_mm=self._tool_diameter_mm(),
            target=self.operation_target,
        )
        operation.overlap_pct = float(self.overlap.value())
        operation.finish_quality = self.finish_quality.currentText()
        self._read_entry_fields(operation)
        self.active_setup.operations.append(operation)
        if target_feat is not None:
            self.assignment_ledger.record(target_feat, self.setups)
        self.generate_action.setEnabled(True)
        self.export_button.setEnabled(True)
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)

    ENTRY_WIDGETS = (
        ("first_pass_count", "first_pass_count"), ("first_pass_depth_mm", "first_pass_depth"),
        ("final_pass_depth_mm", "final_pass_depth"), ("ramp_mode", "ramp_mode"), ("ramp_angle_deg", "ramp_angle"),
        ("ramp_length_mm", "ramp_length"), ("lead_in_mm", "lead_in"), ("lead_out_mm", "lead_out"),
    )

    def _read_entry_fields(self, operation) -> None:
        for field_name, attr in self.ENTRY_WIDGETS:
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            value = widget.currentText() if isinstance(widget, QComboBox) else widget.value()
            setattr(operation, field_name, value)

    def _apply_entry_defaults(self) -> None:
        values = default_entry_params(self._stock_material_name(), self.finish_quality.currentText(), self._tool_diameter_mm())
        for field_name, attr in self.ENTRY_WIDGETS:
            widget = getattr(self, attr)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(values[field_name]))
            else:
                widget.setValue(values[field_name])

    def trace_curved_face(self) -> None:
        """Create surface roughing + finishing operations for the selected face(s)."""
        if not self.selected_features:
            QMessageBox.information(self, "Trace curved face", "Select the face(s) to trace in the viewer first.")
            return
        part_index = sorted(self.selected_features)[0][0]
        feature_ids = sorted(f for p, f in self.selected_features if p == part_index)
        params = trace_surface_dialog(self, self.tools, self._tool_diameter_mm())
        if params is None:
            return
        surface = self._surface_mesh_for(part_index, feature_ids)
        if surface is None:
            return
        finer, deviation = suggest_finer_tool(surface, params, [t.diameter_mm for t in self.tools])
        if finer is not None:
            self.terminal_core.log(
                "warning", f"A {params.tool_diameter_mm:g} mm tool leaves up to {deviation:.2f} mm in the detail; "
                f"a {finer:g} mm tool can reach it. Add a finishing operation with the smaller tool."
            )
            QMessageBox.information(
                self, "Finer tool suggested",
                f"The {params.tool_diameter_mm:g} mm tool cannot reach all the detail (up to {deviation:.2f} mm left). "
                f"Consider a {finer:g} mm tool for finishing.",
            )
        self._push_history("Trace curved face")
        target = f"{self.parts[part_index].name} · Surface"
        kinds = (["Surface rough"] if params.remove_bulk else []) + ["Surface finish"]
        for kind in kinds:
            op = Operation(
                name=f"{len(self.operations) + 1:02d} · {kind}", kind=kind, tool=self.tool_name.currentText(),
                rpm=self.spindle.value(), feed_mm_min=self.feed.value(), stepdown_mm=params.stepdown_mm,
                depth_mm=max(params.stepdown_mm, 0.1), direction=self.direction.currentText().casefold(),
                tool_diameter_mm=params.tool_diameter_mm, target=target,
            )
            op.surface = {"part": part_index, "features": feature_ids, "params": params.to_dict(), "finish": kind == "Surface finish"}
            self.active_setup.operations.append(op)
        self.generate_action.setEnabled(True)
        self.export_button.setEnabled(True)
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)

    def _surface_mesh_for(self, part_index: int, feature_ids) -> pv.PolyData | None:
        if not 0 <= part_index < len(self.parts):
            return None
        mesh = self.parts[part_index].mesh
        if mesh is None or mesh.n_cells == 0:
            return None
        if "feature_id" in mesh.cell_data and feature_ids:
            cells = np.flatnonzero(np.isin(mesh.cell_data["feature_id"], list(feature_ids)))
        else:
            cells = np.asarray(list(feature_ids) or range(mesh.n_cells))
        return mesh.extract_cells(cells).extract_surface() if cells.size else None

    def _selected_operation_ref(self) -> tuple[int, int] | None:
        ref = getattr(self, "selected_operation", None)
        if ref is not None and 0 <= ref[0] < len(self.setups) and 0 <= ref[1] < len(self.setups[ref[0]].operations):
            return ref
        ops = self.active_setup.operations
        return (self.active_setup_index, len(ops) - 1) if ops else None

    def duplicate_operation(self) -> None:
        ref = self._selected_operation_ref()
        if ref is None:
            self.statusBar().showMessage("Add an operation first; there is nothing to duplicate.", 5000)
            return
        self._push_history("Duplicate operation")
        ops = self.setups[ref[0]].operations
        clone = copy.deepcopy(ops[ref[1]])
        ops.insert(ref[1] + 1, clone)
        renumber(ops)
        self.selected_operation = (ref[0], ref[1] + 1)
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)

    def _operation_targets_from_selection(self) -> list[str]:
        targets: list[str] = []
        if hasattr(self, "sketch_tree"):
            for item in self.sketch_tree.selectedItems():
                data = item.data(0, Qt.ItemDataRole.UserRole)
                feat_id = data[1] if isinstance(data, (tuple, list)) and len(data) > 1 else None
                feat = self.sketch_workflow.get_feature(feat_id) if feat_id else None
                if feat is not None:
                    targets.append(feat.name)
        for part_index, feature_id in sorted(self.selected_features):
            if 0 <= part_index < len(self.parts):
                targets.append(f"{self.parts[part_index].name} · Feature {feature_id + 1}")
        return targets

    def assign_operation_to_selection(self) -> None:
        ref = self._selected_operation_ref()
        targets = self._operation_targets_from_selection()
        if ref is None or not targets:
            self.statusBar().showMessage("Select an operation and one or more features or sketches first.", 6000)
            return
        self._push_history("Assign operation to targets")
        ops = self.setups[ref[0]].operations
        source = ops[ref[1]]
        source.target = targets[0]
        for target in targets[1:]:
            clone = copy.deepcopy(source)
            clone.target = target
            ops.append(clone)
        renumber(ops)
        self.terminal_core.log("info", f"Assigned {source.kind} to {len(targets)} target(s).")
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)

    def _remove_operation(self, index: int) -> None:
        ops = self.active_setup.operations
        if not 0 <= index < len(ops):
            return
        self._push_history("Remove operation")
        ops.pop(index)
        has_ops = self._has_any_operations()
        self.generate_action.setEnabled(has_ops)
        self.export_button.setEnabled(has_ops)
        for i, op in enumerate(ops, start=1):
            op.name = f"{i:02d} · {op.kind}"
        self._refresh_operation_stack()
        self._refresh_scene()

    def _run_generation_gate(self, single_operation: bool = False) -> bool:
        """Every sketch must be Construction, Future operation, or have a tool path (unless a single op runs)."""
        self._solve_sketches()
        manual = bool(getattr(self, "manual_control_active", False))
        ok, blocking = generation_gate(self.sketch_workflow.features, self.setups, single_operation=single_operation, manual_control=manual)
        if ok:
            return True
        names = "\n".join(f"• {f.name}" for f in blocking[:12])
        answer = QMessageBox.question(
            self,
            "Untooled sketches",
            f"These sketches have no tool path:\n\n{names}\n\nMark all untooled sketches as Future operation and proceed?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.terminal_core.log("warning", "Generation cancelled: untooled sketches.")
            return False
        self._push_history("Mark untooled sketches as Future operation")
        mark_future(blocking)
        self._refresh_sketch_tree()
        self.terminal_core.log("info", f"Marked {len(blocking)} sketch(es) as Future operation.")
        return True

    def _prepare_gcode(self, single_op_ref: tuple[int, int] | None = None):
        """All generation checks, then the program. Returns (text, settings) or None."""
        if self.sketch_workflow.has_errors():
            self._critical_box(
                self,
                "Unresolved sketch errors",
                "Fix or revert red sketch errors before exporting G-code:\n\n"
                + "\n".join(self.sketch_workflow.blocking_errors()),
            )
            return None
        undefined = self._underdefined_tooled_sketches()
        if undefined:
            self._critical_box(
                self,
                "Under-defined sketches",
                "No code can be generated from under-defined geometry. Fully define these sketches "
                "(dimensions and relations, anchored to the origin, model or coordinates):\n\n" + "\n".join(undefined),
            )
            return None
        if not self._run_generation_gate(single_operation=single_op_ref is not None):
            return None
        if not self._generation_fit_allowed():
            return None
        if not self.operations:
            QMessageBox.warning(self, "No operations", "Add at least one operation before exporting G-code.")
            return None
        settings = self._gcode_settings()
        try:
            gcode_text = self._build_full_gcode_program(settings, single_op_ref=single_op_ref)
        except Exception as error:
            self._critical_box(self, "G-code generation failed", str(error))
            return None
        scale = 1.0 / 25.4 if settings.units_mode == "G20" else 1.0
        issues = validate_gcode_program(
            gcode_text,
            stock_bounds=tuple(v * scale for v in (settings.x_min_mm, settings.x_max_mm, settings.y_min_mm, settings.y_max_mm, 0.0, settings.stock_top_z_mm)),
            expected_units=settings.units_mode,
        )
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            self._critical_box(
                self,
                "G-code validation failed",
                "\n".join(f"Line {e.line_number}: {e.message}" for e in errors),
            )
            return None
        if issues:
            proceed = QMessageBox.warning(
                self,
                "G-code validation warnings",
                "Warnings detected:\n\n"
                + "\n".join(f"Line {w.line_number}: {w.message}" for w in issues[:8])
                + "\n\nExport G-code anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if proceed != QMessageBox.StandardButton.Yes:
                return None
        return gcode_text, settings

    def _gcode_settings(self) -> GCodeSettings:
        if self.stock_mesh is not None and not self.stock_mesh.is_empty:
            xmin, xmax, ymin, ymax, _, zmax = self.stock_mesh.bounds
        else:
            xmin, ymin, zmax = 0.0, 0.0, 0.0
            xmax = float(self.stock_dimensions["X"].value())
            ymax = float(self.stock_dimensions["Y"].value())
        rec = recommend_cutting_params(
            material=self.stock_material.currentText(),
            diameter_mm=self._tool_diameter_mm(),
            flutes=self._current_flute_count(),
            multiplier=1.0,  # programs are nominal; the multiplier is applied live
            operation_kind=self.operation_kind.currentText(),
        )
        setup = self.active_setup
        return GCodeSettings(
            units_mode="G20" if self.units.display is LengthUnit.INCHES else "G21",
            safe_z_mm=float(zmax) + max(float(getattr(setup, "safe_z_mm", 5.0)), 1.0),
            stock_top_z_mm=float(zmax),
            x_min_mm=float(xmin),
            x_max_mm=float(xmax),
            y_min_mm=float(ymin),
            y_max_mm=float(ymax),
            manual_spindle=self.manual_spindle_box.isChecked(),
            recommended_rpm=rec.rpm,
        )

    def export_gcode(self, single_op_ref: tuple[int, int] | None = None) -> None:
        if isinstance(single_op_ref, bool):  # QAction.triggered(bool)
            single_op_ref = None
        prepared = self._prepare_gcode(single_op_ref)
        if prepared is None:
            return
        gcode_text, _settings = prepared
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export G-code",
            "",
            "G-code (*.nc *.gcode *.tap);;All files (*.*)",
        )
        if not path:
            return
        Path(path).write_text(gcode_text, encoding="utf-8")
        self.statusBar().showMessage(f"Exported G-code to {Path(path).name}", 5000)
    def recognize_dynamic_features(self):
        part_meshes = list(getattr(self, "part_meshes", []))
        stock_mesh = getattr(self, "stock_mesh", None)
        sketch_features = list(getattr(self, "sketch_features", []))
        return self.feature_recognizer.analyze(
            part_meshes=part_meshes,
            stock_mesh=stock_mesh,
            sketch_features=sketch_features,
        )

    def _underdefined_tooled_sketches(self) -> list[str]:
        """Non-construction sketch entities that have an operation but are not fully defined (or broken)."""
        self._solve_sketches()
        targets = {op.target for setup in self.setups for op in setup.operations}
        return [
            f"{f.name}: {f.status_label}"
            for f in self.sketch_workflow.features
            if not f.construction and (f.has_error or f.is_underdefined)
            and (f.name in targets or f.feature_id in targets or f.assigned_operation_name)
        ]

    def _operation_path_xy(self, op) -> list[tuple[float, float]] | None:
        """Closed XY contour of the sketch entity an operation targets (None: use the stock bounds)."""
        feat = next((f for f in self.sketch_workflow.features if op.target in (f.name, f.feature_id)), None)
        if feat is None or feat.construction or feat.has_error or feat.state != "defined":
            return None
        pts = np.asarray(feat.active_points, dtype=float)
        if len(pts) < 2:
            return None
        return [(float(x), float(y)) for x, y, _z in pts]

    def _operation_segments(self, op, setup, settings) -> list:
        bounds_xy = (float(settings.x_min_mm), float(settings.x_max_mm), float(settings.y_min_mm), float(settings.y_max_mm))
        surface = getattr(op, "surface", None)
        passes = None
        if surface:
            mesh = self._surface_mesh_for(int(surface.get("part", 0)), surface.get("features", []))
            if mesh is not None:
                params = SurfaceParams.from_dict(surface.get("params"))
                finish = bool(surface.get("finish", True))
                params.remove_bulk = not finish
                plan = plan_surface(mesh, params, stock_top_z=float(settings.stock_top_z_mm), finish=finish)
                passes = plan.passes or None
        return self.toolpath_dynamics.build_operation_segments(
            op, setup=setup, stock_top_z_mm=float(settings.stock_top_z_mm), bounds_xy=bounds_xy,
            path_xy=self._operation_path_xy(op), surface_passes=passes,
        )

    def _build_full_gcode_program(self, settings, single_op_ref: tuple[int, int] | None = None) -> str:
        if self.sketch_workflow.has_errors() or self._underdefined_tooled_sketches():
            raise ValueError("Sketch errors or under-defined tooled sketches; G-code is not generated.")
        manual = bool(getattr(self, "manual_control_active", False))
        ok, blocking = generation_gate(
            self.sketch_workflow.features, self.setups, single_operation=single_op_ref is not None, manual_control=manual
        )
        if not ok:
            raise ValueError("Untooled sketches: " + ", ".join(f.name for f in blocking[:8]))
        setups = list(getattr(self, "setups", []))
        if single_op_ref is not None:
            setup = setups[single_op_ref[0]]
            ops = [setup.operations[single_op_ref[1]]]
        else:
            setup = setups[getattr(self, "active_setup_index", 0)] if setups else None
            ops = list(getattr(self, "operations", []))
        all_segments = []
        for op in ops:
            all_segments.extend(self._operation_segments(op, setup, settings))
        self.toolpath_player.set_segments(all_segments)
        program = self.post_compiler.compile_segments(
            all_segments,
            units_code=str(settings.units_mode),
            manual_spindle=bool(settings.manual_spindle),
            work_offset=str(getattr(setup, "work_offset", "G54")),
        )
        violations = self.motion_guard.check_program_text(program)
        if violations:
            self.motion_guard.report(violations, "G-code generation", show_dialog=False)
            raise ValueError(
                "Motion guard refused the program:\n" + "\n".join(v.text() for v in violations[:10])
            )
        if not self.motion_guard.work_offset_known():
            self.terminal_core.log(
                "info", "Work offset unknown: machine limits will be checked again before streaming."
            )
        return program

    def _stock_dims_mm(self) -> dict[str, float]:
        """Stock X/Y/Z as numbers (the window stores them as spin boxes)."""
        dims = getattr(self, "stock_dimensions", None) or {"X": 100.0, "Y": 80.0, "Z": 20.0}
        return {axis: float(v.value()) if hasattr(v, "value") else float(v) for axis, v in dims.items()}

    def _stock_material_name(self) -> str:
        material = getattr(self, "stock_material", None)
        if hasattr(material, "currentText"):
            return material.currentText() or "Unspecified"
        return str(material or "Unspecified")

    def export_setup_sheet(self, out_dir):
        summary = build_setup_sheet_summary(
            job_name=getattr(self, "job_name", "Untitled job"),
            job_notes=getattr(self, "job_notes", ""),
            stock_material=self._stock_material_name(),
            stock_dimensions=self._stock_dims_mm(),
            setups=list(getattr(self, "setups", [])),
            dynamics=self.toolpath_dynamics,
            post_name=self.post_compiler.spec.name,
        )
        return write_setup_sheet_files(summary, out_dir)
    def apply_tool_library_entry(self, tool_id: str) -> ToolLibraryEntry | None:
        entry = next((t for t in self.tool_library if t.tool_id == tool_id), None)
        if entry is None:
            return None
        if hasattr(self, "tool_name"):
            idx = self.tool_name.findText(entry.name)
            if idx >= 0:
                self.tool_name.setCurrentIndex(idx)
            else:
                self.tool_name.setCurrentText(entry.name)
        if hasattr(self, "tool_diameter"):
            self._set_tool_diameter_mm(float(entry.diameter_mm))
        if hasattr(self, "spindle"):
            self.spindle.setValue(int(entry.default_rpm))
        if hasattr(self, "feed"):
            self.feed.setValue(float(entry.default_feed_mm_min))
        if hasattr(self, "stepdown"):
            self.stepdown.setValue(float(entry.default_stepdown_mm))
        return entry

    def validate_gcode_and_toolpaths(self, gcode_text: str = "") -> list[GCodeValidationIssue]:
        return self.gcode_validator.validate(
            gcode_text=gcode_text,
            operations=list(getattr(self, "operations", [])),
            setups=list(getattr(self, "setups", [])),
            stock_dimensions=self._stock_dims_mm(),
        )
    def recommend_feeds_and_speeds(self, flutes: int = 2, max_rpm: int = 24000):
        material = self._stock_material_name()
        dia = float(self._tool_diameter_mm()) if hasattr(self, "tool_diameter") else 6.0
        kind = self.operation_kind.currentText() if hasattr(self, "operation_kind") else "Rough"
        rec = self.feeds_speeds_calc.recommend(
            material=material,
            tool_diameter_mm=dia,
            flutes=flutes,
            operation_kind=kind,
            max_rpm=max_rpm,
        )
        if hasattr(self, "spindle"):
            self.spindle.setValue(int(rec.rpm))
        if hasattr(self, "feed"):
            self.feed.setValue(float(rec.feed_mm_min))
        if hasattr(self, "stepdown"):
            self.stepdown.setValue(float(rec.stepdown_mm))
        return rec

    def check_toolpath_collisions(self, segments):
        dia = float(self._tool_diameter_mm()) if hasattr(self, "tool_diameter") else 6.0
        dims = self._stock_dims_mm()
        return self.fixture_manager.check_segments(
            segments=segments,
            tool_diameter_mm=dia,
            stock_dimensions=dims,
        )

    def export_full_cam_package(self, out_dir, gcode_text: str, segments=None):
        summary_paths = self.export_setup_sheet(out_dir)
        md_text = summary_paths["markdown"].read_text(encoding="utf-8")
        json_data = json.loads(summary_paths["json"].read_text(encoding="utf-8"))
        collision_report = self.check_toolpath_collisions(segments) if segments is not None else None
        return export_cam_package(
            out_dir=out_dir,
            job_name=getattr(self, "job_name", "Untitled job"),
            gcode_text=gcode_text,
            setup_sheet_md=md_text,
            setup_sheet_json=json_data,
            collision_report=collision_report,
        )
    def check_machine_envelope(self, segments):
        return self.envelope_checker.check_segments(segments)

    def estimate_stock_removal(self):
        return self.stock_removal_estimator.estimate(
            stock_dimensions=self._stock_dims_mm(),
            operations=list(getattr(self, "operations", [])),
            setups=list(getattr(self, "setups", [])),
        )
    def _current_flute_count(self) -> int:
        idx = self.tool_name.currentIndex()
        if 0 <= idx < len(self.tools):
            return max(int(self.tools[idx].flute_count), 1)
        return 2

    def _on_manual_spindle_toggled(self, checked: bool) -> None:
        self.recommended_rpm_label.setVisible(checked)
        self._apply_recommended_defaults()

    def _on_speed_slider_changed(self, value: int) -> None:
        """Speed multiplier: confirm once per session above 100%; GRBL overrides up to 200%;
        beyond that only with the gate (F words scaled while streaming, clamped to the machine maximum)."""
        requested = slider_to_multiplier(value)
        if requested > 1.0 and not self.multiplier_state.confirmed_above_100:
            answer = QMessageBox.question(
                self, "Speed above 100%",
                "Run faster than the programmed feeds? This is asked once per session.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._set_slider_quietly(1.0)
                requested = 1.0
            else:
                self.multiplier_state.confirmed_above_100 = True
        program_max = max((s.feed_mm_min for s in getattr(self.toolpath_player, "segments", []) if s.kind not in ("rapid", "retract")), default=None)
        effective = clamp_multiplier(
            requested, self.safety.allows("speed_multiplier_over_limit"),
            machine_max_feed(self.grbl_settings, self.current_machine), program_max,
        )
        if effective < requested - 1e-9:
            self.terminal_core.log(
                "warning",
                f"Speed multiplier capped at {effective:.2f}x "
                + ("(GRBL override limit; enable it in Settings > Advanced to go further)." if effective <= 2.0 else "(machine maximum rate)."),
            )
            self._set_slider_quietly(effective)
        self.speed_multiplier = effective
        self.multiplier_state.history.append(effective)
        override, self.stream_feed_scale = split_multiplier(effective)
        self.speed_factor_label.setText(f"{effective:.2f}x")
        self._send_overrides(override)
        self._update_cycle_estimate()

    def _set_slider_quietly(self, multiplier: float) -> None:
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(multiplier_to_slider(multiplier))
        self.speed_slider.blockSignals(False)
        self.speed_factor_label.setText(f"{multiplier:.2f}x")

    def _send_overrides(self, override: float) -> None:
        """Feed override (and spindle, unless the spindle speed is manual); rapids stay at 100%."""
        if self.grbl_controller is None or self.grbl_capabilities is None:
            return
        try:
            for byte in override_bytes(override, "feed"):
                self.grbl_controller.send_realtime(byte)
            if not self.manual_spindle_box.isChecked():
                for byte in override_bytes(override, "spindle"):
                    self.grbl_controller.send_realtime(byte)
            self.terminal_core.log("command", f"Overrides: feed {override * 100:.0f}%" + ("" if self.manual_spindle_box.isChecked() else f", spindle {override * 100:.0f}%"))
        except Exception as error:
            self.terminal_core.log("error", f"Override not sent: {error}")

    def _update_cycle_estimate(self) -> None:
        """Current operation (elapsed / remaining) and estimated total, from the run or preview position."""
        if not hasattr(self, "cycle_current_label"):
            return
        segments = list(getattr(self.toolpath_player, "segments", []) or [])
        if not segments:
            self.cycle_current_label.setText("Current operation: -")
            self.cycle_total_label.setText("Estimated total: generate or preview a program")
            return
        if self.cycle_estimator is None or self.cycle_estimator.segments != segments:
            self.cycle_estimator = CycleEstimator(segments, self.current_machine)
        est = self.cycle_estimator
        position = self.stream_index if self._program_running() else int(getattr(self.toolpath_player, "current_index", 0))
        if self._program_running():
            est.tick()
        op, done, left = est.operation_times(position, self.speed_multiplier)
        elapsed = est.elapsed_s if self._program_running() else done
        self.cycle_current_label.setText(f"Current operation: {op or '-'} · {format_duration(elapsed)} / {format_duration(left)} left")
        total = est.elapsed_s + est.remaining(position, self.speed_multiplier) if self._program_running() else est.total(self.speed_multiplier)
        self.cycle_total_label.setText(f"Estimated total: {format_duration(total)} at {self.speed_multiplier:.2f}x")

    def _apply_recommended_defaults(self, *_args) -> None:
        if (
            not hasattr(self, "stock_material")
            or not hasattr(self, "tool_diameter")
            or not hasattr(self, "overlap")
            or not hasattr(self, "manual_spindle_box")
        ):
            return
        material = self.stock_material.currentText()
        rec = recommend_cutting_params(
            material=material,
            diameter_mm=self._tool_diameter_mm(),
            flutes=self._current_flute_count(),
            multiplier=1.0,  # programs are nominal; the multiplier is applied live
            operation_kind=self.operation_kind.currentText(),
        )
        if hasattr(self, "recommended_rpm_label"):
            self.recommended_rpm_label.setText(f"Recommended spindle: {rec.rpm} RPM ({material})")
        if not self.manual_spindle_box.isChecked():
            self.spindle.blockSignals(True)
            self.spindle.setValue(rec.rpm)
            self.spindle.blockSignals(False)
        self.feed.blockSignals(True)
        self.stepdown.blockSignals(True)
        self.feed.setValue(rec.feed_mm_min)
        self.stepdown.setValue(rec.stepdown_mm)
        self.feed.blockSignals(False)
        self.stepdown.blockSignals(False)
        self._apply_default_overlap()

    def _add_new_setup(self) -> None:
        next_num = len(self.setups) + 1
        default_orientations = ["Top (+Z)", "Front (-Y)", "Right (+X)", "Back (+Y)", "Left (-X)", "Bottom (-Z)"]
        orient = default_orientations[(next_num - 1) % len(default_orientations)]
        color = SETUP_PALETTE[(next_num - 1) % len(SETUP_PALETTE)]
        self.setups.append(
            MachiningSetup(
                name=f"Setup {next_num}",
                orientation=orient,
                origin_mm=(0.0, 0.0, 0.0),
                color=color,
            )
        )
        self.active_setup_index = len(self.setups) - 1
        self.setup_orientation_combo.blockSignals(True)
        self.setup_orientation_combo.setCurrentText(orient)
        self.setup_orientation_combo.blockSignals(False)
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)
        self.statusBar().showMessage(f"Created {self.active_setup.name} ({orient})", 4000)

    def _on_setup_orientation_changed(self, orientation: str) -> None:
        self.active_setup.orientation = orientation
        self._refresh_operation_stack()

    def _apply_selection_to_active_setup(self) -> None:
        if self.selected_feature is None or not self.parts:
            self.statusBar().showMessage("Pick a face or feature first to set setup orientation/origin.", 4000)
            return
        part_index, feature_id = self.selected_feature
        if not 0 <= part_index < len(self.parts):
            return
        mesh = self.parts[part_index].mesh
        cells = (
            np.flatnonzero(mesh.cell_data["feature_id"] == feature_id)
            if "feature_id" in mesh.cell_data
            else np.asarray(sorted(self.selected_cells.get(part_index, set())), dtype=int)
        )
        if len(cells) == 0:
            return
        points = np.concatenate([mesh.get_cell(int(c)).points for c in cells])
        center = points.mean(axis=0)
        self.active_setup.origin_mm = (float(center[0]), float(center[1]), float(center[2]))
        self.active_setup.orientation = "Custom face"
        self.setup_orientation_combo.blockSignals(True)
        self.setup_orientation_combo.setCurrentText("Custom face")
        self.setup_orientation_combo.blockSignals(False)
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)
        self.statusBar().showMessage(
            f"{self.active_setup.name} origin set to ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})",
            4000,
        )

    def _refresh_operation_stack(self) -> None:
        self._update_operation_form_visibility()
        while self.operation_stack.count():
            item = self.operation_stack.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not self._has_any_operations() and len(self.setups) == 1:
            empty = QLabel("No operations yet")
            empty.setObjectName("mutedText")
            self.operation_stack.addWidget(empty)

        for s_idx, setup in enumerate(self.setups):
            is_active = s_idx == self.active_setup_index
            header_btn = QPushButton(
                f"{'▸ ' if is_active else '  '}{setup.name} · {setup.orientation} ({len(setup.operations)})"
            )
            header_btn.setStyleSheet(
                f"text-align: left; font-weight: 700; border-left: 4px solid {setup.color};"
            )
            header_btn.clicked.connect(lambda checked=False, idx=s_idx: self._select_setup(idx))
            self.operation_stack.addWidget(header_btn)

            for op_idx, operation in enumerate(setup.operations):
                row = OperationRow(s_idx, op_idx, self._select_operation_row)
                row.setObjectName("operationRow")
                row.base_color = setup.color
                self._style_operation_row(row, getattr(self, "selected_operation", None) == (s_idx, op_idx))
                row.setToolTip("Click to select; drag to reorder.")
                row_layout = QVBoxLayout(row)
                row_layout.setContentsMargins(9, 7, 9, 7)
                row_layout.setSpacing(3)
                title = QLabel(operation.name)
                title.setObjectName("operationTitle")
                row_layout.addWidget(title)
                detail = QLabel(
                    f"{operation.tool} · {operation.feed_mm_min:g} mm/min · "
                    f"{operation.depth_mm:g} mm depth · {operation.target}"
                )
                detail.setObjectName("mutedText")
                detail.setWordWrap(True)
                row_layout.addWidget(detail)

                controls = QHBoxLayout()
                controls.addStretch(1)
                run_one = QPushButton("G-code")
                run_one.setToolTip("Generate G-code for this operation only (single-operation run).")
                run_one.clicked.connect(lambda checked=False, si=s_idx, oi=op_idx: self.export_gcode((si, oi)))
                controls.addWidget(run_one)
                up = QPushButton("↑")
                up.setFixedWidth(28)
                up.clicked.connect(
                    lambda checked=False, si=s_idx, oi=op_idx: self._move_setup_operation(si, oi, -1)
                )
                controls.addWidget(up)
                down = QPushButton("↓")
                down.setFixedWidth(28)
                down.clicked.connect(
                    lambda checked=False, si=s_idx, oi=op_idx: self._move_setup_operation(si, oi, 1)
                )
                controls.addWidget(down)
                remove = QPushButton("×")
                remove.setFixedWidth(28)
                remove.clicked.connect(
                    lambda checked=False, si=s_idx, oi=op_idx: self._remove_setup_operation(si, oi)
                )
                controls.addWidget(remove)
                row_layout.addLayout(controls)
                self.operation_stack.addWidget(row)

        warnings = operation_warnings(self.operations)
        self.warning_label = QLabel("\n".join(f"⚠ {warning}" for warning in warnings))
        self.warning_label.setObjectName("warningText")
        self.warning_label.setWordWrap(True)
        if warnings:
            self.operation_stack.addWidget(self.warning_label)
        self.operation_stack.addStretch(1)
        if hasattr(self, "sketch_tree"):
            self._refresh_sketch_tree()

    def _select_setup(self, index: int) -> None:
        if 0 <= index < len(self.setups):
            self.active_setup_index = index
            self.setup_orientation_combo.blockSignals(True)
            self.setup_orientation_combo.setCurrentText(self.active_setup.orientation)
            self.setup_orientation_combo.blockSignals(False)
            self._load_setup_fields()
            self._refresh_operation_stack()
            self._refresh_scene(reset_camera=False)

    @staticmethod
    def _style_operation_row(row, selected: bool) -> None:
        color = getattr(row, "base_color", "#68b5e8")
        row.setStyleSheet(
            f"margin-left: 14px; border-left: {5 if selected else 3}px solid {color};"
            + (" background: #2a3638;" if selected else "")
        )

    def _select_operation_row(self, setup_idx: int, op_idx: int) -> None:
        self.selected_operation = (setup_idx, op_idx)
        for row in self.operation_stack_area.findChildren(OperationRow):
            self._style_operation_row(row, (row.setup_index, row.op_index) == self.selected_operation)

    def _drop_operation(self, src: tuple[int, int], dst: tuple[int, int]) -> None:
        if src == dst or (src[0] == dst[0] and dst[1] == src[1] + 1):
            return
        self._push_history("Reorder operations")
        if move_operation(self.setups, src, dst):
            self.selected_operation = None
            self._refresh_operation_stack()
            self._refresh_scene(reset_camera=False)

    def _move_setup_operation(self, setup_idx: int, op_idx: int, offset: int) -> None:
        ops = self.setups[setup_idx].operations
        target = op_idx + offset
        if not 0 <= target < len(ops):
            return
        self._push_history("Reorder operations")
        ops[op_idx], ops[target] = ops[target], ops[op_idx]
        for i, op in enumerate(ops, start=1):
            op.name = f"{i:02d} · {op.kind}"
        self._refresh_operation_stack()

    def _remove_setup_operation(self, setup_idx: int, op_idx: int) -> None:
        ops = self.setups[setup_idx].operations
        ops.pop(op_idx)
        for i, op in enumerate(ops, start=1):
            op.name = f"{i:02d} · {op.kind}"
        has_ops = self._has_any_operations()
        self.generate_action.setEnabled(has_ops)
        self.export_button.setEnabled(has_ops)
        self._refresh_operation_stack()
        self._refresh_scene(reset_camera=False)

    def _renumber_operations(self) -> None:
        for index, operation in enumerate(self.operations, start=1):
            operation.name = f"{index:02d} · {operation.kind}"

    def start_measurement(self) -> None:
        if not self.parts:
            self.measure_result.setText("Import a part to measure")
            return
        self.measure_active = True
        self.measure_points.clear()
        self.measure_button.setText("Pick two points…")
        self.measure_result.setText("Click two points on the model")
        self.plotter.enable_point_picking(
            callback=self._on_measure_point,
            show_message=False,
            use_mesh=True,
            show_point=True,
            left_clicking=True,
        )

    def _on_measure_point(self, point: np.ndarray | None) -> None:
        if not self.measure_active or point is None:
            return
        self.measure_points.append(np.asarray(point, dtype=float))
        if len(self.measure_points) == 1:
            self.measure_result.setText("First point set · click the second point")
            return
        first, second = self.measure_points
        deltas = np.abs(second - first)
        distance = math.dist(first, second)
        unit = self.units.display
        scale = 1.0 / 25.4 if unit is LengthUnit.INCHES else 1.0
        formatted = (
            f"Shortest {distance * scale:.3f} {unit.value}   ·   "
            f"ΔX {deltas[0] * scale:.3f}   Y {deltas[1] * scale:.3f}   "
            f"Z {deltas[2] * scale:.3f} {unit.value}"
        )
        self.measure_result.setText(formatted)
        self.measure_button.setText("Measure")
        self.measure_active = False
        self.plotter.disable_picking()

    def _toggle_display_unit(self, checked: bool) -> None:
        previous_mm = self._tool_diameter_mm() if hasattr(self, "tool_diameter") else None
        self.units.display = LengthUnit.INCHES if checked else LengthUnit.MILLIMETERS
        self.display_button.setText(self.units.display.value)
        self._refresh_unit_labels(previous_mm)
        self.measure_result.setText("Display units changed · model size unchanged")

    def _toggle_import_unit(self, checked: bool) -> None:
        new_unit = LengthUnit.INCHES if checked else LengthUnit.MILLIMETERS
        factor = self.units.set_import_unit(new_unit)
        for part in self.parts:
            if part.imported:
                part.mesh.scale((factor,) * 3, inplace=True)
                if part.edge_mesh is not None:
                    part.edge_mesh.scale((factor,) * 3, inplace=True)
                if part.brep_shape is not None:
                    part.brep_shape = scale_solid_shape(part.brep_shape, factor)
                    if part.solid_shape is not None:
                        part.solid_shape = part.brep_shape
        if self.stock_mesh is not None and self.stock_imported:
            self.stock_mesh.scale((factor,) * 3, inplace=True)
        self.import_button.setText(new_unit.value)
        self.measure_result.setText(f"Imported geometry rescaled × {factor:g}")
        self._refresh_scene()

    def start_feature_pick(self, checked: bool = True) -> None:
        self.feature_pick_active = True
        self.pick_feature_button.setChecked(True)
        self.pick_feature_button.setText("Click a sketch feature…")
        self.statusBar().showMessage("Click a sketch feature in the viewer to target it.")

    def _select_feature_at(self, position, additive: bool = False) -> None:
        picked = self._pick_at(position)
        if picked is None:
            self.statusBar().showMessage("No part face was picked. Try clicking the model surface.", 4000)
            return
        data_key, cell_id = picked
        part_index = self.dataset_part_indexes.get(data_key)
        if part_index is None:
            self.statusBar().showMessage("Select a part feature, not the stock outline.", 4000)
            return
        part = self.parts[part_index]
        if cell_id >= part.mesh.n_cells:
            return
        if "feature_id" in part.mesh.cell_data:
            feature_id = int(part.mesh.cell_data["feature_id"][cell_id])
            face_cells = set(
                np.flatnonzero(part.mesh.cell_data["feature_id"] == feature_id).tolist()
            )
        else:
            feature_id = cell_id
            face_cells = {cell_id}

        target_key = (part_index, feature_id)
        already_selected = target_key in self.selected_features

        if not additive and already_selected and len(self.selected_features) == 1:
            self.selected_cells.pop(part_index, None)
            self.selected_features.clear()
            self.selected_feature = None
            self.operation_target = "Whole part"
            self.target_label.setText("Target: Whole part")
            self.statusBar().showMessage("Deselected feature", 3000)
            self._refresh_scene(reset_camera=False)
            return

        if not additive:
            self.selected_cells.clear()
            self.selected_features.clear()

        if additive and already_selected:
            self.selected_features.discard(target_key)
            part_cells = self.selected_cells.get(part_index, set())
            part_cells.difference_update(face_cells)
            if not part_cells:
                self.selected_cells.pop(part_index, None)
        else:
            self.selected_cells.setdefault(part_index, set()).update(face_cells)
            self.selected_features.add(target_key)

        self.selected_feature = (
            next(iter(self.selected_features)) if len(self.selected_features) == 1 else None
        )
        if len(self.selected_features) == 0:
            self.operation_target = "Whole part"
        elif len(self.selected_features) == 1:
            only_part_idx, only_feat_id = next(iter(self.selected_features))
            self.operation_target = f"{self.parts[only_part_idx].name} · Feature {only_feat_id + 1}"
        else:
            self.operation_target = f"{len(self.selected_features)} features selected"
        self.target_label.setText(f"Target: {self.operation_target}")

        if len(self.selected_features) == 1:
            only_part_idx, only_feat_id = next(iter(self.selected_features))
            self.known_feature_targets.add((only_part_idx, only_feat_id))
            target_data = ("feature", only_part_idx, only_feat_id)
            target_index = self.note_target.findData(target_data)
            if target_index < 0:
                self.note_target.addItem(
                    f"Feature {only_feat_id + 1} · {self.parts[only_part_idx].name}", target_data
                )
                target_index = self.note_target.count() - 1
            self.note_target.setCurrentIndex(target_index)

        self.statusBar().showMessage(f"Target: {self.operation_target}", 4000)
        self._refresh_scene(reset_camera=False)

    def _pick_at(self, position) -> tuple[str, int] | None:
        native_hit = self._pick_native_brep_face(position)
        if native_hit is not None:
            part_index, feature_id, exact_distance = native_hit
            native_part = self.parts[part_index]
            face_cells = np.flatnonzero(
                native_part.mesh.cell_data["feature_id"] == feature_id
            )
            if len(face_cells):
                self.cell_picker.Pick(
                    int(round(position.x() * self.plotter.interactor.devicePixelRatioF())),
                    int(
                        self.plotter.render_window.GetSize()[1]
                        - round(position.y() * self.plotter.interactor.devicePixelRatioF())
                    ),
                    0,
                    self.plotter.renderer,
                )
                picked_data = self.cell_picker.GetDataSet()
                picked_cell = self.cell_picker.GetCellId()
                if picked_data is not None and picked_cell >= 0:
                    picked_key = self._vtk_address(picked_data)
                    picked_part_index = self.dataset_part_indexes.get(picked_key)
                    if picked_part_index is not None and not self.parts[picked_part_index].native_brep:
                        picked_point = np.asarray(self.cell_picker.GetPickPosition())
                        ray = self._viewport_ray(position)
                        if ray is not None:
                            near, _ = ray
                            if np.linalg.norm(picked_point - near) < exact_distance:
                                return picked_key, picked_cell
                return self._vtk_address(native_part.mesh), int(face_cells[0])

        scale = self.plotter.interactor.devicePixelRatioF()
        render_height = self.plotter.render_window.GetSize()[1]
        x = int(round(position.x() * scale))
        y = int(round(render_height - position.y() * scale))
        self.cell_picker.Pick(x, y, 0, self.plotter.renderer)
        dataset = self.cell_picker.GetDataSet()
        cell_id = self.cell_picker.GetCellId()
        if dataset is None or cell_id < 0:
            return None
        return self._vtk_address(dataset), cell_id

    def _viewport_ray(self, position: QPoint) -> tuple[np.ndarray, np.ndarray] | None:
        scale = self.plotter.interactor.devicePixelRatioF()
        display_x = float(position.x() * scale)
        display_y = float(self.plotter.render_window.GetSize()[1] - position.y() * scale)
        renderer = self.plotter.renderer
        points = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(display_x, display_y, depth)
            renderer.DisplayToWorld()
            world = np.asarray(renderer.GetWorldPoint(), dtype=float)
            if abs(world[3]) < 1e-12:
                return None
            points.append(world[:3] / world[3])
        return points[0], points[1]

    def _pick_native_brep_face(self, position: QPoint) -> tuple[int, int, float] | None:
        ray = self._viewport_ray(position)
        if ray is None:
            return None
        near, far = ray
        direction = far - near
        ray_length = float(np.linalg.norm(direction))
        if ray_length <= 1e-12:
            return None
        try:
            from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopoDS import TopoDS
            from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
        except ImportError:
            return None

        line = gp_Lin(
            gp_Pnt(float(near[0]), float(near[1]), float(near[2])),
            gp_Dir(float(direction[0]), float(direction[1]), float(direction[2])),
        )
        best_hit: tuple[int, int, float] | None = None
        for part_index, part in enumerate(self.parts):
            if not part.native_brep or part.brep_shape is None:
                continue
            intersector = IntCurvesFace_ShapeIntersector()
            intersector.Load(part.brep_shape, max(ray_length * 1e-9, 1e-7))
            intersector.Perform(line, 0.0, ray_length)
            if intersector.NbPnt() == 0:
                continue
            hit_index = min(
                range(1, intersector.NbPnt() + 1), key=intersector.WParameter
            )
            distance = intersector.WParameter(hit_index)
            hit_face = intersector.Face(hit_index)
            explorer = TopExp_Explorer(part.brep_shape, TopAbs_FACE)
            feature_id = 0
            while explorer.More():
                if TopoDS.Face(explorer.Current()).IsSame(hit_face):
                    break
                feature_id += 1
                explorer.Next()
            if explorer.More() and (
                best_hit is None or distance < best_hit[2]
            ):
                best_hit = (part_index, feature_id, distance)
        return best_hit

    def _over_orientation_cube(self, position: QPoint) -> bool:
        scale = self.plotter.interactor.devicePixelRatioF()
        render_height = self.plotter.render_window.GetSize()[1]
        x = int(round(position.x() * scale))
        y = int(round(render_height - position.y() * scale))
        representation = self.orientation_widget.GetRepresentation()
        representation.ComputeInteractionState(x, y, 0)
        return (
            representation.GetInteractionStateAsEnum()
            != representation.InteractionStateType.Outside
        )

    @staticmethod
    def _vtk_address(dataset) -> str:
        return dataset.GetAddressAsString("")

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.parts_list
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Delete
        ):
            self.delete_layered()
            return True
        machine_widget = getattr(self, "machine_dock", None)
        if (
            event.type() == QEvent.Type.KeyPress
            and self._keyboard_machine_active()
            and hasattr(self, "plotter")
            and (watched is self.plotter.interactor or (machine_widget is not None and watched is machine_widget.widget()))
        ):
            if self._handle_machine_key(event):
                return True
        if hasattr(self, "plotter") and watched is self.plotter.interactor:
            event_type = event.type()
            if event_type == QEvent.Type.Resize:
                self._position_view_overlay()
            if event_type == QEvent.Type.KeyPress and event.key() in (
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down
            ):
                dx = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(event.key(), 0)
                dy = {Qt.Key.Key_Up: 1, Qt.Key.Key_Down: -1}.get(event.key(), 0)
                self._jog_selection(dx, dy, self._jog_step_mm(event.modifiers()))
                return True
            if event_type == QEvent.Type.MouseButtonPress:
                self.drag_origin = event.position().toPoint()
                self.drag_current = self.drag_origin
                if event.button() == Qt.MouseButton.LeftButton:
                    self.drag_additive = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if self._over_orientation_cube(self.drag_origin):
                        self.drag_mode = "orientation"
                        return False
                    if self.sketch_mode.isChecked():
                        hit_handle = self._hit_test_sketch_handle(self.drag_origin)
                        if hit_handle is not None:
                            self.dragging_sketch = hit_handle
                            self.active_sketch_id = hit_handle[0]
                            self.drag_mode = "sketch_drag"
                            self._refresh_sketch_tree()
                            return True
                    self.drag_mode = "select"
                    if self.drag_rubber_band is None:
                        self.drag_rubber_band = QRubberBand(
                            QRubberBand.Shape.Rectangle, self.plotter.interactor
                        )
                    self.drag_rubber_band.setGeometry(QRect(self.drag_origin, self.drag_origin))
                    return False
                if event.button() == Qt.MouseButton.MiddleButton:
                    self.drag_mode = "pan"
                    return True
                if event.button() == Qt.MouseButton.RightButton:
                    self.drag_mode = "orbit"
                    return True
            elif event_type == QEvent.Type.MouseMove:
                position = event.position().toPoint()
                if self.drag_mode == "orientation":
                    return False
                if self.drag_mode == "sketch_drag" and event.buttons() & Qt.MouseButton.LeftButton:
                    if self.dragging_sketch is not None:
                        new_pt = self._world_at_work_plane(position)
                        if new_pt is not None:
                            feat_id, pt_idx = self.dragging_sketch
                            self.sketch_workflow.move_feature_point(feat_id, pt_idx, new_pt)
                            self._refresh_sketch_tree()
                            self._refresh_scene(reset_camera=False)
                    return True
                if self.drag_mode == "select" and event.buttons() & Qt.MouseButton.LeftButton:
                    self.drag_current = position
                    if (position - self.drag_origin).manhattanLength() >= self.drag_threshold:
                        self.drag_rubber_band.setGeometry(QRect(self.drag_origin, position).normalized())
                        self.drag_rubber_band.show()
                    return True
                if self.drag_mode == "pan" and event.buttons() & Qt.MouseButton.MiddleButton:
                    delta = position - self.drag_current
                    self.drag_current = position
                    self._pan_camera(delta)
                    return True
                if self.drag_mode == "orbit" and event.buttons() & Qt.MouseButton.RightButton:
                    delta = position - self.drag_current
                    self.drag_current = position
                    self._orbit_camera(delta)
                    return True
                self._show_hover_note(event)
            elif event_type == QEvent.Type.MouseButtonRelease:
                if self.drag_mode == "sketch_drag" and event.button() == Qt.MouseButton.LeftButton:
                    self.dragging_sketch = None
                    self.drag_mode = ""
                    self._refresh_sketch_tree()
                    self._refresh_scene(reset_camera=False)
                    return True
                if self.drag_mode == "select" and event.button() == Qt.MouseButton.LeftButton:
                    end = event.position().toPoint()
                    moved = (end - self.drag_origin).manhattanLength() >= self.drag_threshold
                    if self.drag_rubber_band is not None:
                        self.drag_rubber_band.hide()
                    if moved:
                        self._select_box(self.drag_origin, end, self.drag_additive)
                    elif self.sketch_mode.isChecked():
                        self._add_sketch_click(end)
                    else:
                        self._select_click(end, self.drag_additive)
                    self.drag_mode = ""
                    if self.feature_pick_active:
                        self.feature_pick_active = False
                        self.pick_feature_button.setChecked(False)
                        self.pick_feature_button.setText("Pick sketch feature")
                    return False
                if self.drag_mode == "orientation" and event.button() == Qt.MouseButton.LeftButton:
                    self.drag_mode = ""
                    return False
                if self.drag_mode == "pan" and event.button() == Qt.MouseButton.MiddleButton:
                    self.drag_mode = ""
                    return True
                if self.drag_mode == "orbit" and event.button() == Qt.MouseButton.RightButton:
                    self.drag_mode = ""
                    return True
            elif event_type == QEvent.Type.Wheel:
                self._zoom_camera(event.angleDelta().y())
                return True
        return super().eventFilter(watched, event)

    def _show_hover_note(self, event) -> None:
        picked = self._pick_at(event.position().toPoint())
        note = ""
        if picked is not None:
            data_key, cell_id = picked
            part_index = self.dataset_part_indexes.get(data_key)
            if part_index is not None:
                part = self.parts[part_index]
                feature_id = cell_id
                if "feature_id" in part.mesh.cell_data and cell_id < part.mesh.n_cells:
                    feature_id = int(part.mesh.cell_data["feature_id"][cell_id])
                note = part.feature_notes.get(feature_id, "") or part.note
            elif data_key in self.stock_dataset_keys:
                note = self.stock_note
        if note != self.last_hover_note:
            self.last_hover_note = note
            if note:
                QToolTip.showText(event.globalPosition().toPoint(), note, self.plotter.interactor)
            else:
                QToolTip.hideText()

    def _refresh_note_targets(self) -> None:
        current = self.note_target.currentData()
        self.note_target.blockSignals(True)
        self.note_target.clear()
        for index, part in enumerate(self.parts):
            self.note_target.addItem(f"Whole part · {part.name}", ("part", index, -1))
        for part_index, feature_id in sorted(self.known_feature_targets):
            if part_index < len(self.parts):
                self.note_target.addItem(
                    f"Feature {feature_id + 1} · {self.parts[part_index].name}",
                    ("feature", part_index, feature_id),
                )
        for feat in self.sketch_workflow.features:
            self.note_target.addItem(f"Sketch · {feat.name}", ("sketch", feat.feature_id, -1))
        selected = self.note_target.findData(current)
        self.note_target.setCurrentIndex(selected if selected >= 0 else 0)
        self.note_target.blockSignals(False)
        self._load_note_target()

    def _load_note_target(self, _index: int = -1) -> None:
        if not hasattr(self, "note_editor"):
            return
        target = self.note_target.currentData()
        if target is None:
            self.note_editor.clear()
            return
        kind, part_index, feature_id = target
        if kind == "sketch":
            note = self.sketch_notes.get(part_index, "")
        elif kind == "part":
            note = self.parts[part_index].note
        else:
            note = self.parts[part_index].feature_notes.get(feature_id, "")
        self.note_editor.setPlainText(note)

    def save_note(self) -> None:
        target = self.note_target.currentData()
        if target is None:
            return
        kind, part_index, feature_id = target
        note = self.note_editor.toPlainText().strip()
        if kind == "sketch":
            self.sketch_notes[part_index] = note
        elif kind == "part":
            self.parts[part_index].note = note
        else:
            self.parts[part_index].feature_notes[feature_id] = note
        self._mark_dirty()
        self._refresh_notes_list()
        self.last_hover_note = ""
        self.statusBar().showMessage("Hover note saved", 3000)

    def show_gcode_readiness(self) -> None:
        items = gcode_readiness(
            self.parts,
            self.stock_mesh is not None,
            self.stock_material.currentText(),
            self.operations,
            work_origin_set=self.work_origin_set,
            machine_profile_set=self.machine_profile_set,
            postprocessor_set=self.postprocessor_set,
            workholding_checked=self.workholding_checked,
            setup=self.active_setup if self.setups else None,
            untooled_sketches=[f.name for f in untooled(self.sketch_workflow.features, self.setups)],
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("G-code setup checklist")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        heading = QLabel("Information required before a job can be posted")
        heading.setObjectName("panelTitle")
        layout.addWidget(heading)
        for item in items:
            status = "READY" if item.ready else "NEEDED"
            row = QLabel(f"{status}  ·  {item.title}\n{item.detail}")
            row.setObjectName("readinessReady" if item.ready else "readinessMissing")
            row.setWordWrap(True)
            layout.addWidget(row)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def _show_axes_assignment(self) -> None:
        """Axes button: origin and edge-based work-frame assignment in the Position tab."""
        self._show_tab("Position")
        if hasattr(self, "frame_group"):
            self.frame_group.setFocus()

    def _show_tab(self, title: str) -> None:
        index = next((i for i in range(self.cam_tabs.count()) if self.cam_tabs.tabText(i) == title), -1)
        if index >= 0:
            self.cam_tabs.setCurrentIndex(index)

    def _fit_view(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()


    def set_standard_view(self, name: str) -> None:
        directions = {
            "Front": ((0, -1, 0), (0, 0, 1)),
            "Back": ((0, 1, 0), (0, 0, 1)),
            "Left": ((-1, 0, 0), (0, 0, 1)),
            "Right": ((1, 0, 0), (0, 0, 1)),
            "Top": ((0, 0, 1), (0, 1, 0)),
            "Bottom": ((0, 0, -1), (0, 1, 0)),
            "Isometric": ((1, -1, 1), (0, 0, 1)),
        }
        if name not in directions:
            return
        direction, view_up = directions[name]
        bounds = np.asarray(self.plotter.bounds).reshape(3, 2)
        center = bounds.mean(axis=1)
        diagonal = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
        distance = max(diagonal * 2.2, 10.0)
        direction_vector = np.asarray(direction, dtype=float)
        direction_vector /= np.linalg.norm(direction_vector)
        camera = self.plotter.camera
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + direction_vector * distance))
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        camera.ParallelProjectionOn()
        camera.SetParallelScale(max(diagonal * 0.58, 1.0))
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def set_work_plane(self, name: str) -> None:
        origin = (
            np.asarray(self.active_setup.origin_mm, dtype=float)
            if hasattr(self, "active_setup")
            else np.zeros(3, dtype=float)
        )
        self.sketch_workflow.active_plane = build_standard_sketch_plane(name, origin=origin)
        plane_views = {"XY plane": "Top", "XZ plane": "Front", "YZ plane": "Right"}
        view = plane_views.get(name)
        if view:
            self.set_standard_view(view)

    def start_sketch_on_selection(self) -> bool:
        """Creates a sketch plane from the currently selected planar face or 30° manifold curved surface."""
        if self.selected_feature is not None:
            part_idx, fid = self.selected_feature
            if 0 <= part_idx < len(self.parts):
                mesh = self.parts[part_idx].mesh
                if mesh is not None and "feature_id" in mesh.cell_data:
                    fids, cells = flood_select_manifold_features(mesh, int(fid), max_angle_deg=30.0)
                    if len(cells) > 0:
                        self.selected_cells[part_idx] = set(int(c) for c in cells)
                        plane = build_sketch_plane_from_surface(
                            mesh,
                            cells,
                            part_index=part_idx,
                            feature_ids=fids,
                            label=f"Part {part_idx + 1} · Surface ({len(fids)} faces)",
                        )
                        self.sketch_workflow.active_plane = plane
                        return True
        if self.selected_cells:
            part_idx = next(iter(self.selected_cells))
            cells = np.asarray(sorted(self.selected_cells[part_idx]), dtype=int)
            if 0 <= part_idx < len(self.parts) and len(cells) > 0:
                mesh = self.parts[part_idx].mesh
                plane = build_sketch_plane_from_surface(
                    mesh,
                    cells,
                    part_index=part_idx,
                    feature_ids=[],
                    label=f"Part {part_idx + 1} · Selected Faces",
                )
                self.sketch_workflow.active_plane = plane
                return True
        return False

    def _toggle_sketch_mode(self, enabled: bool) -> None:
        self.pending_sketch_points.clear()
        self.sketch_mode.setText("Finish sketch" if enabled else "New sketch")
        if hasattr(self, "clear_sketch_btn"):
            self.clear_sketch_btn.setVisible(enabled)
        if hasattr(self, "sketch_tools_bar"):
            self.sketch_tools_bar.setVisible(enabled)
        if enabled:
            on_face = self.start_sketch_on_selection()
            if not on_face:
                self.set_work_plane(self.work_plane.currentText())
            plane = self.sketch_workflow.active_plane
            sketch_id = self.sketch_workflow.begin_sketch(plane.name if plane else self.work_plane.currentText())
            if plane is not None and plane.source_part_index is not None:
                self.sketch_workflow.sketches[sketch_id]["part_index"] = int(plane.source_part_index)
            mode_note = (
                f"Z-max curved surface plane ({plane.name})"
                if plane and plane.is_curved_surface
                else (plane.name if plane else self.work_plane.currentText())
            )
            self.statusBar().showMessage(
                f"Sketch active on {mode_note}. Click to place points locked to plane.",
                5000,
            )
            self._refresh_scene(reset_camera=False)
        else:
            self.sketch_workflow.end_sketch()
            self.statusBar().showMessage("Sketch mode ended.", 3000)

    def _world_at_work_plane(
        self, position: QPoint, plane_origin: np.ndarray | None = None
    ) -> np.ndarray | None:
        if self.sketch_workflow.active_plane is None:
            self.set_work_plane(self.work_plane.currentText())
        plane = self.sketch_workflow.active_plane
        normal = plane.normal
        origin = np.asarray(plane_origin, dtype=float) if plane_origin is not None else plane.origin

        scale = self.plotter.interactor.devicePixelRatioF()
        render_height = self.plotter.render_window.GetSize()[1]
        display_x = float(position.x() * scale)
        display_y = float(render_height - position.y() * scale)

        renderer = self.plotter.renderer
        ray_points = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(display_x, display_y, depth)
            renderer.DisplayToWorld()
            world = np.asarray(renderer.GetWorldPoint(), dtype=float)
            if abs(world[3]) < 1e-12:
                return None
            ray_points.append(world[:3] / world[3])
        near, far = ray_points
        ray = far - near
        denominator = float(np.dot(ray, normal))
        if abs(denominator) < 1e-12:
            return None
        distance = float(np.dot(origin - near, normal)) / denominator
        raw_pt = near + ray * distance

        anchor = self.pending_sketch_points[-1] if self.pending_sketch_points else None
        snapped_pt, _snap_cand, _constraints = self.sketch_workflow.resolve_snap(
            raw_pt,
            anchor_point=anchor,
            part_meshes=(p.mesh for p in self.parts if p.mesh is not None),
        )
        return plane.project_point_to_plane(snapped_pt)

    def _finish_pending_spline(self) -> None:
        if len(self.pending_sketch_points) < 2:
            self.pending_sketch_points.clear()
            return
        plane = self.sketch_workflow.active_plane
        plane_label = plane.name if plane else self.work_plane.currentText()
        is_const = bool(getattr(self, "sketch_construction_toggle", None) and self.sketch_construction_toggle.isChecked())
        ctrl = np.asarray(self.pending_sketch_points, dtype=float)
        sampled = sample_catmull_rom_spline(ctrl) if len(ctrl) >= 3 else ctrl
        self._push_history("Add sketch feature")
        feat = self.sketch_workflow.create_feature(
            "Spline",
            plane_label,
            sampled,
            construction=is_const,
            parent_feature_id=None,
        )
        self.active_sketch_id = feat.feature_id
        self.pending_sketch_points.clear()
        self.sketch_visible.setChecked(True)
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)

    def _sketch_clicks_needed(self, kind: str) -> int:
        variant = self.sketch_variant.currentText() if hasattr(self, "sketch_variant") else ""
        if kind == "Point":
            return 1
        if kind == "Curve" or (kind in {"Rectangle", "Circle"} and variant == "3 point"):
            return 3
        return 2

    def _add_sketch_click(self, position: QPoint) -> None:
        if self.sketch_workflow.has_errors():
            QMessageBox.warning(
                self,
                "Sketch error present",
                "Resolve or revert existing red sketch errors before adding new sketch geometry.",
            )
            return
        point = self._world_at_work_plane(position)
        if point is None:
            self.statusBar().showMessage("The view ray is parallel to the active sketch plane.", 3500)
            return

        kind = self.sketch_kind.currentText()
        plane = self.sketch_workflow.active_plane
        plane_label = plane.name if plane else self.work_plane.currentText()
        is_const = bool(getattr(self, "sketch_construction_toggle", None) and self.sketch_construction_toggle.isChecked())

        if kind == "Point":
            self._push_history("Add sketch feature")
            feat = self.sketch_workflow.create_feature(
                "Point",
                plane_label,
                point.reshape(1, 3),
                construction=is_const,
                parent_feature_id=None,
            )
            self.active_sketch_id = feat.feature_id
        elif kind == "Spline":
            self.pending_sketch_points.append(point)
            self.statusBar().showMessage(
                f"Spline point {len(self.pending_sketch_points)} placed — click next point or click 'Finish Spline'.",
                4000,
            )
            return
        elif kind in {"Curve", "Rectangle", "Circle"} and self._sketch_clicks_needed(kind) == 3:
            self.pending_sketch_points.append(point)
            variant = self.sketch_variant.currentText()
            if len(self.pending_sketch_points) < 3:
                self.statusBar().showMessage(f"{kind} ({variant}): point {len(self.pending_sketch_points)}/3 set.")
                return
            p1, p2, p3 = self.pending_sketch_points[-3:]
            if kind == "Curve":
                pts = sample_center_arc(p1, p2, p3, plane) if variant == "Center arc" else sample_three_point_arc(p1, p2, p3, plane)
            elif kind == "Rectangle":
                pts = sample_rectangle_three_points(p1, p2, p3, plane)
            else:
                pts = sample_circle_three_points(p1, p2, p3, plane)
                if pts is None:
                    self.pending_sketch_points.clear()
                    self.statusBar().showMessage("The three points are in a line; no circle passes through them.", 4000)
                    return
            self._push_history("Add sketch feature")
            feat = self.sketch_workflow.create_feature(
                kind,
                plane_label,
                pts,
                construction=is_const,
                parent_feature_id=None,
                variant=variant,
            )
            self.active_sketch_id = feat.feature_id
            self.pending_sketch_points.clear()
        else:
            self.pending_sketch_points.append(point)
            if len(self.pending_sketch_points) < 2:
                self.statusBar().showMessage(f"First point set for {kind.lower()}; pick the second point.")
                return
            first, second = self.pending_sketch_points[-2:]
            variant = self.sketch_variant.currentText()
            if kind == "Rectangle":
                rect_pts = sample_rectangle_points(first, second, plane, mode="Center" if variant == "Center" else "Corner")
                self._push_history("Add sketch feature")
                feat = self.sketch_workflow.create_feature(
                    "Rectangle",
                    plane_label,
                    rect_pts,
                    construction=is_const,
                    parent_feature_id=None,
                    variant=variant,
                )
            elif kind == "Circle":
                circ_pts = sample_circle_points(first, second, plane)
                self._push_history("Add sketch feature")
                feat = self.sketch_workflow.create_feature(
                    "Circle",
                    plane_label,
                    circ_pts,
                    construction=is_const,
                    parent_feature_id=None,
                    variant=variant,
                )
            elif kind == "Construction line" or is_const:
                bounds = np.asarray(self.plotter.bounds).reshape(3, 2)
                extent = max(float(np.linalg.norm(bounds[:, 1] - bounds[:, 0])) * 2, 10.0)
                self._push_history("Add sketch feature")
                feat = self.sketch_workflow.create_feature(
                    "Construction line",
                    plane_label,
                    construction_line(first, second, extent),
                    construction=True,
                    parent_feature_id=None,
                )
            else:
                self._push_history("Add sketch feature")
                feat = self.sketch_workflow.create_feature(
                    "Line",
                    plane_label,
                    np.vstack((first, second)),
                    construction=False,
                    parent_feature_id=None,
                )
            self.active_sketch_id = feat.feature_id
            self.pending_sketch_points.clear()

        self.sketch_visible.setChecked(True)
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)

    def _hit_test_sketch_handle(
        self, position: QPoint, tolerance_px: float = 12.0
    ) -> tuple[str, int] | None:
        scale = self.plotter.interactor.devicePixelRatioF()
        render_height = self.plotter.render_window.GetSize()[1]
        click_x = float(position.x() * scale)
        click_y = float(render_height - position.y() * scale)
        best: tuple[float, str, int] | None = None
        for feat in self.sketch_workflow.features:
            for pt_idx, pt in enumerate(feat.points):
                dx, dy, _ = self._project_world_point(pt)
                dist = math.hypot(dx - click_x, dy - click_y)
                if dist <= tolerance_px * scale and (best is None or dist < best[0]):
                    best = (dist, feat.feature_id, pt_idx)
        return (best[1], best[2]) if best is not None else None

    def clear_sketch(self) -> None:
        self._push_history("Clear sketch")
        self.sketch_workflow.clear()
        self.sketch_entities.clear()
        self.pending_sketch_points.clear()
        self.active_sketch_id = None
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)

    def revert_sketch_change(self) -> None:
        if not self.sketch_workflow.revert_last():
            self.statusBar().showMessage("Nothing to revert in sketch workflow.", 3000)
            return
        if self.active_sketch_id and self.sketch_workflow.get_feature(self.active_sketch_id) is None:
            self.active_sketch_id = (
                self.sketch_workflow.features[-1].feature_id
                if self.sketch_workflow.features
                else None
            )
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)
        self.statusBar().showMessage("Reverted last sketch modification.", 3500)

    def _refresh_sketch_tree(self) -> None:
        if not hasattr(self, "sketch_tree"):
            return
        if not self.sketch_tree.property("_context_menu_wired"):
            self.sketch_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.sketch_tree.customContextMenuRequested.connect(self._show_sketch_tree_context_menu)
            self.sketch_tree.setProperty("_context_menu_wired", True)

        self.sketch_workflow.sync_tooling_assignments(self.setups)
        self._solve_sketches()
        self._check_assignment_memory()
        self.sketch_tree.blockSignals(True)
        self.sketch_tree.clear()
        unit = self.units.display.value
        error_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
        for feat in self.sketch_workflow.features:
            owner = self.sketch_workflow.sketches.get(feat.owner_sketch_id, {}).get("name", "")
            star = "*" if feat.is_underdefined else ""
            label = f"{owner} › {feat.name}{star}" if owner else f"{feat.name}{star}"
            status = sketch_status(feat, self.setups)
            state_text = feat.status_label if feat.has_error or status is SketchStatus.UNTOOLED else f"{feat.status_label} · {status.value}"
            item = QTreeWidgetItem([label, state_text])
            item.setData(0, Qt.ItemDataRole.UserRole, ("feature", feat.feature_id, ""))
            color = QColor(feat.display_color)
            item.setForeground(0, color)
            item.setForeground(1, color)
            if feat.has_error:
                item.setIcon(0, error_icon)
                tint = QColor(theme_color("error"))
                tint.setAlpha(60)
                item.setBackground(0, tint)
                item.setBackground(1, tint)
            elif feat.is_underdefined and not feat.construction:
                tint = QColor(theme_color("warning"))
                tint.setAlpha(40)
                item.setBackground(0, tint)
                item.setBackground(1, tint)
            tip = f"Entity {feat.entity_id}"
            if feat.is_underdefined and feat.construction:
                tip = "Undefined construction features · " + tip
            if feat.owner_sketch_id:
                tip += f" · sketch {feat.owner_sketch_id}"
            if feat.parent_feature_id:
                tip += f" · driven by {feat.parent_feature_id}"
            item.setToolTip(0, tip)
            for dim in feat.dimensions:
                shown = format_length(dim.value, unit, dim.sig_figs) if dim.kind not in {"X", "Y", "Z"} else format_length(dim.value, unit, 6)
                dim_item = QTreeWidgetItem([f"  ↳ {dim.name}", shown + ("" if dim.driving else " (driven)")])
                dim_item.setData(0, Qt.ItemDataRole.UserRole, ("dim", feat.feature_id, dim.dim_id))
                if not dim.valid:
                    dim_item.setForeground(0, QColor(COLORS["red"]))
                    dim_item.setForeground(1, QColor(COLORS["red"]))
                item.addChild(dim_item)
            self.sketch_tree.addTopLevelItem(item)
            item.setExpanded(True)
            if feat.feature_id == self.active_sketch_id:
                self.sketch_tree.setCurrentItem(item)
        self.sketch_tree.blockSignals(False)

        if hasattr(self, "revert_sketch_btn"):
            self.revert_sketch_btn.setEnabled(self.sketch_workflow.can_revert())
        if hasattr(self, "sketch_error_banner"):
            if self.sketch_workflow.has_errors():
                self.sketch_error_banner.setText(
                    "⚠ Sketch error — fix dimension or click Revert:\n"
                    + "\n".join(self.sketch_workflow.blocking_errors())
                )
                self.sketch_error_banner.show()
            else:
                self.sketch_error_banner.hide()

    def _show_sketch_tree_context_menu(self, pos: QPoint) -> None:
        item = self.sketch_tree.itemAt(pos)
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind, feat_id, dim_id = payload
        self.active_sketch_id = feat_id
        feat = self.sketch_workflow.get_feature(feat_id)
        if feat is None:
            return

        menu = QMenu(self)
        assign_act = menu.addAction("Assign Tool Operation to Sketch")
        view_op_act = menu.addAction(
            f"View Assigned Operation ({feat.assigned_operation_name})"
            if feat.assigned_operation_name
            else "View Assigned Operation (None)"
        )
        view_op_act.setEnabled(bool(feat.assigned_operation_name))
        jump_op_act = menu.addAction("Jump to Operation in Stack")
        jump_op_act.setEnabled(bool(self._operation_ref_for_target(feat.name)))
        project_act = menu.addAction("Project onto Active Curved Surface")
        project_act.setEnabled(
            self.sketch_workflow.active_plane is not None
            and self.sketch_workflow.active_plane.surface_mesh is not None
        )
        dim_act = menu.addAction(
            "Edit Dimension…" if kind == "dim" and dim_id else "Define / Edit Dimensions…"
        )
        notes_act = menu.addAction("Open Note…")
        notes_act.setEnabled(hasattr(self, "note_target"))
        relation_menu = menu.addMenu("Add relation")
        relation_actions = {}
        for kind in RELATION_KINDS:
            relation_actions[relation_menu.addAction(kind.replace("_", " ").capitalize())] = kind
        relation_menu.addSeparator()
        clear_rel = relation_menu.addAction("Remove relations on selection")
        future_act = menu.addAction("Clear Future operation" if feat.status == "future" else "Mark as Future operation")

        chosen = menu.exec(self.sketch_tree.viewport().mapToGlobal(pos))
        if chosen == assign_act:
            self.assign_selected_sketch_to_target()
        elif chosen == view_op_act and feat.assigned_operation_name:
            self.statusBar().showMessage(
                f"{feat.name} → {feat.assigned_setup_name} / {feat.assigned_operation_name}",
                5000,
            )
        elif chosen == project_act:
            self._push_history("Project sketch to surface")
            if self.sketch_workflow.project_feature_to_active_surface(feat.feature_id):
                self._refresh_sketch_tree()
                self._refresh_scene(reset_camera=False)
        elif chosen == dim_act:
            if kind == "dim" and dim_id:
                self._edit_single_sketch_dimension(feat_id, dim_id)
            else:
                self.edit_selected_sketch_dimensions()
        elif chosen in relation_actions:
            self._add_relation(relation_actions[chosen])
        elif chosen == clear_rel:
            self._remove_relations_on_selection()
        elif chosen == future_act:
            self._set_future_operation(feat.status != "future")
        elif chosen == jump_op_act:
            self._jump_to_operation(self._operation_ref_for_target(feat.name))
        elif chosen == notes_act:
            self._open_note_for(("sketch", feat.feature_id, -1))

    def _selected_entity_ids(self) -> list[str]:
        ids = [
            item.data(0, Qt.ItemDataRole.UserRole)[1]
            for item in self.sketch_tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole) and item.data(0, Qt.ItemDataRole.UserRole)[0] == "feature"
        ]
        if not ids and self.active_sketch_id:
            ids = [self.active_sketch_id]
        return list(dict.fromkeys(ids))

    def _add_relation(self, kind: str, ids: list[str] | None = None) -> Relation | None:
        """Create a relation from the selected entities (see RELATION_KINDS)."""
        manager = self.sketch_workflow
        ids = ids or self._selected_entity_ids()
        feats = [f for f in (manager.get_feature(i) for i in ids) if f is not None]
        if not feats or not all(manager.can_edit(f) for f in feats):
            self.statusBar().showMessage("Select entities of the sketch being edited.", 4000)
            return None
        refs: list[tuple[str, int | None]] = []
        value = None
        if kind in {"horizontal", "vertical", "fix"}:
            refs = [(feats[0].feature_id, None)]
        elif kind == "anchor_origin":
            refs = [(feats[0].feature_id, 0)]
        elif kind == "anchor_model":
            point = np.asarray(feats[0].points[0], dtype=float)
            vertices = [np.asarray(p.mesh.points) for p in self.parts if p.mesh is not None and p.mesh.n_points]
            if not vertices:
                self.statusBar().showMessage("No model geometry to anchor to.", 4000)
                return None
            allv = np.vstack(vertices)
            value = allv[int(np.argmin(np.linalg.norm(allv - point, axis=1)))].tolist()
            refs = [(feats[0].feature_id, 0)]
        elif len(feats) >= 2:
            a, b = feats[0], feats[1]
            if kind == "coincident":
                ia, ib = closest_reference_points(a, b)
                refs = [(a.feature_id, ia), (b.feature_id, ib)]
            elif kind == "concentric":
                refs = [(a.feature_id, 0), (b.feature_id, 0)]
            elif kind == "midpoint":
                point, line = (a, b) if a.kind == "Point" else (b, a)
                refs = [(point.feature_id, 0), (line.feature_id, None)]
            else:
                refs = [(a.feature_id, None), (b.feature_id, None)]
        else:
            self.statusBar().showMessage(f"'{kind}' needs two selected entities.", 4000)
            return None
        self._push_history(f"Add {kind} relation")
        relation = Relation(kind, refs, value)
        manager.relations.append(relation)
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)
        return relation

    def _remove_relations_on_selection(self) -> None:
        ids = set(self._selected_entity_ids())
        keep = [r for r in self.sketch_workflow.relations if not set(r.entity_ids()) & ids]
        if len(keep) == len(self.sketch_workflow.relations):
            return
        self._push_history("Remove relations")
        self.sketch_workflow.relations = keep
        self._refresh_sketch_tree()
        self._refresh_scene(reset_camera=False)

    def _operation_ref_for_target(self, target: str) -> tuple[int, int] | None:
        for s_idx, setup in enumerate(self.setups):
            for o_idx, op in enumerate(setup.operations):
                if op.target == target:
                    return (s_idx, o_idx)
        return None

    def _jump_to_operation(self, ref: tuple[int, int] | None) -> None:
        if ref is None:
            return
        self.selected_operation = ref
        self.active_setup_index = ref[0]
        self._show_tab("Operations")
        self._refresh_operation_stack()
        for row in self.operation_stack_area.findChildren(OperationRow):
            if (row.setup_index, row.op_index) == ref:
                row.setFocus()
        self.statusBar().showMessage(f"Operation: {self.setups[ref[0]].operations[ref[1]].name}", 4000)

    def _on_sketch_tree_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        _kind, feat_id, _dim_id = payload
        self.active_sketch_id = feat_id
        feat = self.sketch_workflow.get_feature(feat_id)
        if feat is not None and hasattr(self, "sketch_construction_toggle"):
            self.sketch_construction_toggle.blockSignals(True)
            self.sketch_construction_toggle.setChecked(bool(feat.construction))
            self.sketch_construction_toggle.blockSignals(False)
        self._refresh_scene(reset_camera=False)

    def _on_sketch_tree_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind, feat_id, dim_id = payload
        self.active_sketch_id = feat_id
        if kind == "dim":
            self._edit_single_sketch_dimension(feat_id, dim_id)
        else:
            self.edit_selected_sketch_dimensions()

    def _edit_single_sketch_dimension(self, feat_id: str, dim_id: str) -> None:
        feat = self.sketch_workflow.get_feature(feat_id)
        if feat is None:
            return
        target_dim = next((d for d in feat.dimensions if d.dim_id == dim_id), None)
        if target_dim is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit {target_dim.name}")
        form = QFormLayout(dialog)
        spin = QDoubleSpinBox()
        spin.setRange(-100000.0, 100000.0)
        spin.setDecimals(3)
        spin.setValue(target_dim.value)
        form.addRow(f"{target_dim.name} (mm)", spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._push_history("Edit sketch dimension")
            ok = self.sketch_workflow.update_dimension_value(feat_id, dim_id, spin.value())
            self._refresh_sketch_tree()
            self._refresh_scene(reset_camera=False)
            if not ok:
                QMessageBox.warning(
                    self,
                    "Dimension error",
                    "Invalid dimension caused an error (marked in red). Fix the value or click Revert.",
                )

    def edit_selected_sketch_dimensions(self) -> None:
        if not self.active_sketch_id:
            self.statusBar().showMessage("Select a sketch feature in the workflow tree first.", 3500)
            return
        feat = self.sketch_workflow.get_feature(self.active_sketch_id)
        if feat is None:
            return
        if not feat.dimensions:
            self._push_history("Define sketch")
            self.sketch_workflow.define_feature(feat.feature_id)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Define & Dimension · {feat.name}")
        form = QFormLayout(dialog)
        inputs: list[tuple[SketchDimension, QDoubleSpinBox]] = []
        for dim in feat.dimensions:
            spin = QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)
            spin.setDecimals(3)
            spin.setValue(dim.value)
            form.addRow(f"{dim.name} (mm)", spin)
            inputs.append((dim, spin))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated_dims = [
                SketchDimension(
                    dim_id=dim.dim_id,
                    name=dim.name,
                    kind=dim.kind,
                    value=spin.value(),
                )
                for dim, spin in inputs
            ]
            self._push_history("Define sketch")
            ok = self.sketch_workflow.define_feature(feat.feature_id, updated_dims)
            self._refresh_sketch_tree()
            self._refresh_scene(reset_camera=False)
            if not ok:
                QMessageBox.warning(
                    self,
                    "Sketch constraint error",
                    "Invalid dimension caused a sketch error (marked in red). Fix it or click Revert.",
                )

    def assign_selected_sketch_to_target(self) -> None:
        if not self.active_sketch_id:
            self.statusBar().showMessage("Select a sketch feature in the workflow tree first.", 3500)
            return
        feat = self.sketch_workflow.get_feature(self.active_sketch_id)
        if feat is None:
            return
        self._solve_sketches()
        allowed, reason = can_assign_tool(feat)
        if not allowed:
            self.terminal_core.log("warning", f"Tool not assigned: {reason}")
            QMessageBox.warning(self, "Cannot assign a tool", reason)
            return
        self.operation_target = feat.name
        self.target_label.setText(f"Target: {self.operation_target}")
        self._update_operation_form_visibility()
        self.statusBar().showMessage(
            f"Operation target set to {feat.name}. Click 'Add operation to stack' to assign tooling.",
            5000,
        )

    def export_sketches_dxf_dialog(self) -> None:
        if not self.sketch_workflow.features:
            QMessageBox.information(self, "No sketches", "Create at least one sketch before exporting DXF.")
            return
        if self.sketch_workflow.has_errors():
            self._critical_box(
                self,
                "Unresolved sketch errors",
                "Fix or revert red sketch errors before exporting DXF:\n\n"
                + "\n".join(self.sketch_workflow.blocking_errors()),
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export sketches as DXF",
            "",
            "DXF drawing (*.dxf);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".dxf"):
            path += ".dxf"
        include_const = (
            self.construction_visible.isChecked()
            if hasattr(self, "construction_visible")
            else False
        )
        count = export_sketches_to_dxf(
            self.sketch_workflow.features,
            path,
            include_construction=include_const,
        )
        self.statusBar().showMessage(f"Exported {count} sketch entity(s) to {Path(path).name}", 5000)

    def view_normal_to_selected(self) -> None:
        if self.selected_feature is None:
            self.statusBar().showMessage("Select a face or curve first.", 4000)
            return
        part_index, feature_id = self.selected_feature
        if not 0 <= part_index < len(self.parts):
            return
        mesh = self.parts[part_index].mesh
        if "feature_id" in mesh.cell_data:
            cells = np.flatnonzero(mesh.cell_data["feature_id"] == feature_id)
        else:
            cells = np.asarray(sorted(self.selected_cells.get(part_index, set())), dtype=int)
        if len(cells) == 0:
            return
        cell_points = [mesh.get_cell(int(cell_id)).points for cell_id in cells]
        points = np.concatenate(cell_points)
        center = points.mean(axis=0)
        first_cell = mesh.get_cell(int(cells[0]))
        if first_cell.type in {pv.CellType.LINE, pv.CellType.POLY_LINE}:
            ordered_points = first_cell.points
            for next_points in cell_points[1:]:
                if len(next_points) and np.linalg.norm(ordered_points[-1] - next_points[0]) < 1e-6:
                    ordered_points = np.vstack((ordered_points, next_points[1:]))
            midpoint = len(ordered_points) // 2
            low = max(midpoint - 1, 0)
            high = min(midpoint + 1, len(ordered_points) - 1)
            direction = ordered_points[high] - ordered_points[low]
        else:
            normals = mesh.compute_normals(
                cell_normals=True, point_normals=False, inplace=False
            ).cell_data["Normals"][cells]
            direction = normals.mean(axis=0)
        magnitude = float(np.linalg.norm(direction))
        if magnitude < 1e-9:
            self.statusBar().showMessage("The selected geometry has no stable view direction.", 4000)
            return
        direction = direction / magnitude
        bounds = np.asarray(mesh.bounds).reshape(3, 2)
        diagonal = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
        camera = self.plotter.camera
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + direction * max(diagonal * 2.2, 10.0)))
        view_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(direction, view_up))) > 0.92:
            view_up = np.array([0.0, 1.0, 0.0])
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        camera.ParallelProjectionOn()
        camera.SetParallelScale(max(diagonal * 0.65, 1.0))
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _pan_camera(self, delta: QPoint) -> None:
        camera = self.plotter.camera
        up = np.asarray(camera.GetViewUp(), dtype=float)
        direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        right = np.cross(direction, up)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        height = max(self.plotter.interactor.height(), 1)
        if camera.GetParallelProjection():
            world_per_pixel = 2 * camera.GetParallelScale() / height
        else:
            distance = camera.GetDistance()
            view_angle = math.radians(camera.GetViewAngle())
            world_per_pixel = 2 * distance * math.tan(view_angle / 2) / height
        shift = (-delta.x() * right + delta.y() * up) * world_per_pixel
        camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint()) + shift))
        camera.SetPosition(*(np.asarray(camera.GetPosition()) + shift))
        self.plotter.render()

    def _orbit_camera(self, delta: QPoint) -> None:
        camera = self.plotter.camera
        camera.Azimuth(-delta.x() * self.orbit_sensitivity)
        camera.Elevation(delta.y() * self.orbit_sensitivity)
        camera.OrthogonalizeViewUp()
        self.plotter.render()

    def _zoom_camera(self, wheel_delta: int) -> None:
        camera = self.plotter.camera
        factor = 1.15 if wheel_delta > 0 else 1 / 1.15
        if camera.GetParallelProjection():
            camera.SetParallelScale(max(camera.GetParallelScale() / factor, 0.001))
        else:
            camera.Dolly(factor)
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _project_world_point(self, point: np.ndarray) -> tuple[float, float, float]:
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        display = renderer.GetDisplayPoint()
        return float(display[0]), float(display[1]), float(display[2])

    def _select_box(self, start: QPoint, end: QPoint, additive: bool = False) -> None:
        scale = self.plotter.interactor.devicePixelRatioF()
        start_x = int(round(start.x() * scale))
        end_x = int(round(end.x() * scale))
        start_y = int(round(start.y() * scale))
        end_y = int(round(end.y() * scale))
        left, right = sorted((start_x, end_x))
        top, bottom = sorted((start_y, end_y))
        contained = end.x() >= start.x()
        through = self.through_toggle.isChecked()
        height = self.plotter.render_window.GetSize()[1]
        selected = (
            {part_index: set(cell_ids) for part_index, cell_ids in self.selected_cells.items()}
            if additive
            else {}
        )
        selected_features = set(self.selected_features) if additive else set()
        for part_index, part in enumerate(self.parts):
            matching: set[int] = set()
            for cell_id in range(part.mesh.n_cells):
                cell = part.mesh.get_cell(cell_id)
                if cell.n_points == 0:
                    continue
                projected = [self._project_world_point(point) for point in cell.points]
                screen = [(x, height - y, z) for x, y, z in projected]
                xs = [point[0] for point in screen]
                ys = [point[1] for point in screen]
                fully_inside = all(
                    left <= x <= right and top <= y <= bottom
                    for x, y in zip(xs, ys)
                )
                intersects = (
                    max(xs) >= left and min(xs) <= right
                    and max(ys) >= top and min(ys) <= bottom
                )
                if (contained and not fully_inside) or (not contained and not intersects):
                    continue
                center_x = int(round(sum(xs) / len(xs)))
                center_y = int(round(sum(ys) / len(ys)))
                center_depth = sum(point[2] for point in screen) / len(screen)
                if not through:
                    visible_depth = self.plotter.renderer.GetZ(center_x, height - center_y)
                    if visible_depth < 0 or center_depth > visible_depth + 0.015:
                        continue
                matching.add(cell_id)
            if matching:
                selected.setdefault(part_index, set()).update(matching)
                feature_ids = part.mesh.cell_data.get("feature_id")
                for cell_id in matching:
                    feature_id = int(feature_ids[cell_id]) if feature_ids is not None else cell_id
                    selected_features.add((part_index, feature_id))
        self.selected_cells = selected
        self.selected_features = selected_features
        self.selected_feature = next(iter(selected_features)) if len(selected_features) == 1 else None
        total = sum(len(cells) for cells in selected.values())
        if total:
            self.operation_target = (
                f"{len(selected_features)} features selected"
                if selected_features
                else f"{total} selected cells"
            )
            self.target_label.setText(f"Target: {self.operation_target}")
            self.statusBar().showMessage(
                f"Selected {total} cells · {'contained' if contained else 'crossing'} · "
                f"{'through all geometry' if through else 'visible geometry'}",
                5000,
            )
        else:
            self.statusBar().showMessage("No geometry matched the selection window.", 3500)
        self._refresh_scene(reset_camera=False)

    def _select_click(self, position: QPoint, additive: bool = False) -> None:
        picked = self._pick_at(position)
        if picked is None:
            if not additive:
                self.selected_cells.clear()
                self.selected_features.clear()
                self.selected_feature = None
                self.operation_target = "Whole part"
                self.target_label.setText("Target: Whole part")
                self._refresh_scene(reset_camera=False)
            return
        data_key, cell_id = picked
        part_index = self.dataset_part_indexes.get(data_key)
        if part_index is None:
            return
        self._select_feature_at(position, additive=additive)

    @staticmethod
    def _panel() -> QFrame:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        return panel

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    @staticmethod
    def _divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("divider")
        return divider

    @staticmethod
    def _double_spin(value: float, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(decimals)
        field.setValue(value)
        field.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return field

    @staticmethod
    def _int_spin(value: int, minimum: int, maximum: int) -> QSpinBox:
        field = QSpinBox()
        field.setRange(minimum, maximum)
        field.setValue(value)
        field.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        return field

    @staticmethod
    def _stylesheet() -> str:
        return f"""
            QWidget {{ background: {COLORS['background']}; color: {COLORS['text']};
                      font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
            QMainWindow, QMenuBar {{ background: {COLORS['background']}; }}
            QMenuBar {{ padding: 4px 12px; }}
            QMenuBar::item {{ padding: 5px 9px; }}
            QMenu {{ background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; }}
            QMenu::item {{ padding: 7px 22px; }}
            QMenu::item:selected {{ background: {COLORS['panel_alt']}; }}
            #header {{ background: #192123; border-bottom: 1px solid {COLORS['line']}; }}
            #brand {{ font-size: 20px; font-weight: 700; }}
            #eyebrow, #sectionLabel {{ color: {COLORS['muted']}; font-size: 10px;
                                       font-weight: 700; letter-spacing: 1px; }}
            #sectionLabel {{ padding-top: 5px; }}
            #safeBadge {{ color: {COLORS['accent']}; border: 1px solid #53633a;
                          padding: 6px 9px; font-size: 11px; font-weight: 700; }}
            #sidePanel {{ background: {COLORS['panel']}; }}
            #subToolbar, #viewFooter {{ background: {COLORS['panel']};
                                       border-bottom: 1px solid {COLORS['line']}; }}
            #viewFooter {{ border-top: 1px solid {COLORS['line']}; border-bottom: 0; }}
            #panelTitle {{ font-size: 17px; font-weight: 650; }}
            #mutedText {{ color: {COLORS['muted']}; font-size: 11px; }}
            #warningText {{ color: {COLORS['warning']}; font-size: 11px; }}
            #readinessReady {{ color: {COLORS['accent']}; padding: 7px 4px;
                               border-bottom: 1px solid {COLORS['line']}; }}
            #readinessMissing {{ color: {COLORS['warning']}; padding: 7px 4px;
                                 border-bottom: 1px solid {COLORS['line']}; }}
            QTextEdit {{ background: #141b1d; border: 1px solid {COLORS['line']};
                         padding: 6px; }}
            #partRow {{ background: {COLORS['panel_alt']}; padding: 8px; }}
            #operationRow {{ background: {COLORS['panel_alt']};
                             border: 1px solid {COLORS['line']}; }}
            #operationTitle {{ font-weight: 700; }}
            #divider {{ color: {COLORS['line']}; max-height: 1px; }}
            QPushButton {{ background: {COLORS['panel_alt']}; border: 1px solid {COLORS['line']};
                           padding: 7px 9px; min-height: 19px; }}
            QPushButton:hover {{ border-color: {COLORS['accent']}; }}
            QPushButton:disabled {{ color: #687370; border-color: #30393b; }}
            QPushButton#primaryButton {{ background: {COLORS['accent']}; color: #172019;
                                         border: 0; font-weight: 700; }}
            QPushButton#primaryButton:hover {{ background: #d3eb75; }}
            QPushButton:checked {{ background: {COLORS['accent']}; color: #172019; }}
            QComboBox, QDoubleSpinBox, QSpinBox {{ background: #141b1d;
                border: 1px solid {COLORS['line']}; padding: 5px; min-height: 21px; }}
            QComboBox::drop-down {{ border: 0; width: 18px; }}
            QSplitter::handle {{ background: {COLORS['background']}; width: 2px; }}
            QScrollArea {{ background: transparent; }}
        """


def main() -> int:
    pv.set_plot_theme("dark")
    app = QApplication(sys.argv)
    app.setApplicationName("BasiCAM CNC")
    icon_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    icon_path = icon_root / "BCCIcon.ico"
    if icon_path.exists():
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(icon_path)))
    app.setFont(QFont("Segoe UI", 10))
    run_first_run_dialog(QSettings("BasiCAM", "BasiCAM CNC"))
    window = BasiCAMWindow()
    window.show()
    return app.exec()
