"""Machine inventory, soft/hard limits, keep-out zones, bed-fit report and the related dialogs.

All coordinates are GRBL machine coordinates (MPos) in millimetres. The Z hard
minimum is recorded with the longest tool mounted; other tools are compensated
by their stick-out difference (stick-out = overall length - 12 mm in the collet).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from basicam.cam_envelope_stock_removal import MachineKinematicsProfile
from basicam.tooling import COLLET_GRIP_MM

AXES = ("X", "Y", "Z")
DEFAULT_SOFT_MARGIN_MM = 20.0
MIN_SOFT_MARGIN_MM = 5.0


class LimitPolicyError(ValueError):
    """A soft-limit change that needs the `release_soft_limits` gate."""


@dataclass
class SoftLimit:
    """Per axis, per direction: hard limit (endstop), soft margin, Limited or Released."""

    hard_min: float | None = None
    hard_max: float | None = None
    margin_min: float = DEFAULT_SOFT_MARGIN_MM
    margin_max: float = DEFAULT_SOFT_MARGIN_MM
    released_min: bool = False
    released_max: bool = False

    @property
    def soft_min(self) -> float | None:
        return None if self.hard_min is None else self.hard_min + self.margin_min

    @property
    def soft_max(self) -> float | None:
        return None if self.hard_max is None else self.hard_max - self.margin_max

    def is_set(self) -> bool:
        return self.hard_min is not None or self.hard_max is not None

    @classmethod
    def from_dict(cls, data: dict) -> "SoftLimit":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(data or {}).items() if k in known})


def set_soft_margin(limit: SoftLimit, side: str, margin_mm: float, gate_allows: bool) -> None:
    """Margin defaults to 20 mm; under 5 mm needs the release gate."""
    if margin_mm < MIN_SOFT_MARGIN_MM and not gate_allows:
        raise LimitPolicyError(f"A soft-limit margin under {MIN_SOFT_MARGIN_MM:g} mm needs 'release_soft_limits'.")
    margin_mm = max(0.0, float(margin_mm))
    if side == "min":
        limit.margin_min = margin_mm
    else:
        limit.margin_max = margin_mm


def set_released(limit: SoftLimit, side: str, released: bool, gate_allows: bool) -> None:
    if released and not gate_allows:
        raise LimitPolicyError("Releasing a soft limit needs 'release_soft_limits'.")
    if side == "min":
        limit.released_min = bool(released)
    else:
        limit.released_max = bool(released)


def tool_stick_out(tool) -> float:
    if tool is None:
        return 0.0
    return max(0.0, float(getattr(tool, "overall_length_mm", 0.0)) - COLLET_GRIP_MM)


def z_min_for_tool(z_min_longest: float, longest_stick_out: float, tool_stick_out_mm: float) -> float:
    """A shorter tool may go lower by the stick-out difference."""
    return z_min_longest - max(0.0, longest_stick_out - tool_stick_out_mm)


@dataclass
class KeepOutZone:
    name: str
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    def normalized(self) -> "KeepOutZone":
        lo = tuple(min(a, b) for a, b in zip(self.min_xyz, self.max_xyz))
        hi = tuple(max(a, b) for a, b in zip(self.min_xyz, self.max_xyz))
        return KeepOutZone(self.name, lo, hi)

    @classmethod
    def from_dict(cls, data: dict) -> "KeepOutZone":
        return cls(
            str(data.get("name", "Keep-out")),
            tuple(float(v) for v in data.get("min_xyz", (0, 0, 0))),
            tuple(float(v) for v in data.get("max_xyz", (0, 0, 0))),
        ).normalized()


@dataclass
class MachineProfile:
    machine_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "New machine"
    controller: str = "GRBL"
    travel_mm: tuple[float, float, float] = (300.0, 300.0, 80.0)
    spindle_min_rpm: int = 0
    spindle_max_rpm: int = 24000
    max_feed_mm_min: tuple[float, float, float] = (3000.0, 3000.0, 1000.0)
    rapid_mm_min: float = 3000.0
    homing_direction: str = "max"  # GRBL default: homes to max, travel negative
    probe_present: bool = False
    tool_change_mode: str = "Manual"
    notes: str = ""
    keep_out_zones: list[KeepOutZone] = field(default_factory=list)
    soft_limits: dict[str, SoftLimit] = field(default_factory=lambda: {a: SoftLimit() for a in AXES})
    z_reference_tool: str = ""
    z_reference_stick_out_mm: float = 0.0

    # ---------------------------------------------------------------- limits
    def hard_bounds(self, axis: str) -> tuple[float, float]:
        """Recorded endstop limits, falling back to the travel range from the homing direction."""
        index = AXES.index(axis)
        travel = float(self.travel_mm[index])
        default = (-travel, 0.0) if self.homing_direction == "max" else (0.0, travel)
        limit = self.soft_limits.get(axis, SoftLimit())
        lo = limit.hard_min if limit.hard_min is not None else default[0]
        hi = limit.hard_max if limit.hard_max is not None else default[1]
        return (min(lo, hi), max(lo, hi))

    def soft_bounds(self, axis: str) -> tuple[float | None, float | None]:
        lo, hi = self.hard_bounds(axis)
        limit = self.soft_limits.get(axis, SoftLimit())
        return (lo + limit.margin_min, hi - limit.margin_max)

    def has_soft_limits(self) -> bool:
        return any(limit.is_set() for limit in self.soft_limits.values())

    def z_min_for(self, tool) -> float:
        lo, _hi = self.hard_bounds("Z")
        if self.z_reference_stick_out_mm > 0 and tool is not None:
            return z_min_for_tool(lo, self.z_reference_stick_out_mm, tool_stick_out(tool))
        return lo

    def z_top(self) -> float:
        return self.hard_bounds("Z")[1]

    def kinematics_profile(self) -> MachineKinematicsProfile:
        (x0, x1), (y0, y1), (z0, z1) = (self.hard_bounds(a) for a in AXES)
        return MachineKinematicsProfile(
            name=self.name,
            x_min_mm=x0, x_max_mm=x1, y_min_mm=y0, y_max_mm=y1, z_min_mm=z0, z_max_mm=z1,
            max_feed_mm_min=float(max(self.max_feed_mm_min)),
        )

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["keep_out_zones"] = [asdict(z) for z in self.keep_out_zones]
        data["soft_limits"] = {a: asdict(l) for a, l in self.soft_limits.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MachineProfile":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        for key in ("travel_mm", "max_feed_mm_min"):
            if key in values:
                values[key] = tuple(float(v) for v in values[key])
        values["keep_out_zones"] = [KeepOutZone.from_dict(z) for z in data.get("keep_out_zones", [])]
        raw_limits = data.get("soft_limits", {}) or {}
        values["soft_limits"] = {a: SoftLimit.from_dict(raw_limits.get(a, {})) for a in AXES}
        return cls(**values)


def fit_report(part_size: Iterable[float], machine: MachineProfile | None) -> list[str]:
    """Axes on which the part is larger than the machine travel."""
    if machine is None:
        return []
    size = list(part_size)
    return [axis for axis, s, t in zip(AXES, size, machine.travel_mm) if float(s) > float(t) + 1e-9]


def compatible_machines(machines: Iterable[MachineProfile], part_size: Iterable[float]) -> list[MachineProfile]:
    size = list(part_size)
    return [m for m in machines if not fit_report(size, m)]


def generation_fit_rule(part_size: Iterable[float], machine: MachineProfile | None, gate_allows: bool) -> tuple[bool, str]:
    """Exceeding in 1-2 dimensions warns; in all 3 refuses unless `generate_oversize_part` is on."""
    axes = fit_report(part_size, machine)
    if not axes:
        return True, ""
    message = f"Part exceeds the work area in {', '.join(axes)}. Check that it fits and the operation is possible."
    if len(axes) == 3 and not gate_allows:
        return False, message + " G-code is refused (all three dimensions); see Settings > Advanced."
    return True, message


class MachineStore:
    """Machine inventory and per-machine state in QSettings (JSON)."""

    INVENTORY_KEY = "machines/inventory"
    DEFAULT_KEY = "machines/default_id"

    def __init__(self, settings) -> None:
        self.settings = settings

    def list(self) -> list[MachineProfile]:
        raw = self.settings.value(self.INVENTORY_KEY, "[]")
        try:
            items = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        except (TypeError, ValueError):
            items = []
        return [MachineProfile.from_dict(item) for item in items if isinstance(item, dict)]

    def _write(self, machines: list[MachineProfile]) -> None:
        self.settings.setValue(self.INVENTORY_KEY, json.dumps([m.to_dict() for m in machines]))

    def get(self, machine_id: str | None) -> MachineProfile | None:
        return next((m for m in self.list() if m.machine_id == machine_id), None) if machine_id else None

    def save(self, machine: MachineProfile) -> None:
        machines = [m for m in self.list() if m.machine_id != machine.machine_id]
        machines.append(machine)
        self._write(machines)

    def delete(self, machine_id: str) -> None:
        self._write([m for m in self.list() if m.machine_id != machine_id])
        if self.default_id == machine_id:
            self.default_id = None

    @property
    def default_id(self) -> str | None:
        value = self.settings.value(self.DEFAULT_KEY, "")
        return str(value) if value else None

    @default_id.setter
    def default_id(self, machine_id: str | None) -> None:
        self.settings.setValue(self.DEFAULT_KEY, machine_id or "")

    def load_state(self, machine_id: str) -> dict[str, Any]:
        raw = self.settings.value(f"machines/{machine_id}/state", "{}")
        try:
            data = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            data = {}
        return data if isinstance(data, dict) else {}

    def save_state(self, machine_id: str, state: dict[str, Any]) -> None:
        self.settings.setValue(f"machines/{machine_id}/state", json.dumps(state))


# ====================================================================== Qt widgets
class MachinesTab(QWidget):
    """Machine inventory editor (a tab in the tool inventory window)."""

    machinesChanged = Signal()

    def __init__(self, store: MachineStore, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.machines = store.list()
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._select)
        left.addWidget(self.list_widget, 1)
        row = QHBoxLayout()
        add = QPushButton("New machine")
        add.clicked.connect(self._new)
        remove = QPushButton("Remove")
        remove.setToolTip("Delete this machine from the inventory; projects keep their own copy.")
        remove.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(remove)
        left.addLayout(row)
        root.addLayout(left, 1)

        form = QFormLayout()
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        self.controller = QComboBox()
        self.controller.addItems(["GRBL"])
        form.addRow("Controller", self.controller)
        self.travel = [self._spin(0, 5000, 300) for _ in AXES]
        self.max_feed = [self._spin(1, 100000, 3000) for _ in AXES]
        for axis, spin in zip(AXES, self.travel):
            form.addRow(f"Travel {axis} · mm", spin)
        for axis, spin in zip(AXES, self.max_feed):
            form.addRow(f"Max feed {axis} · mm/min", spin)
        self.rapid = self._spin(1, 100000, 3000)
        form.addRow("Rapid rate · mm/min", self.rapid)
        self.rpm_min = QSpinBox()
        self.rpm_min.setRange(0, 100000)
        self.rpm_max = QSpinBox()
        self.rpm_max.setRange(0, 100000)
        form.addRow("Spindle min · rpm", self.rpm_min)
        form.addRow("Spindle max · rpm", self.rpm_max)
        self.homing = QComboBox()
        self.homing.addItems(["max", "min"])
        self.homing.setToolTip("Direction the machine homes; GRBL default is max with negative travel.")
        form.addRow("Homing direction", self.homing)
        self.probe = QCheckBox("Probe present")
        form.addRow("", self.probe)
        self.tool_change = QComboBox()
        self.tool_change.addItems(["Manual", "Manual with tool length probe", "Automatic"])
        form.addRow("Tool change", self.tool_change)
        self.notes = QTextEdit()
        self.notes.setFixedHeight(60)
        form.addRow("Notes", self.notes)
        self.keep_out_table = QTableWidget(0, 7)
        self.keep_out_table.setHorizontalHeaderLabels(["Name", "X min", "Y min", "Z min", "X max", "Y max", "Z max"])
        self.keep_out_table.setToolTip("Boxes in machine coordinates the tool envelope may never enter.")
        self.keep_out_table.setMaximumHeight(130)
        ko_row = QHBoxLayout()
        add_ko = QPushButton("Add keep-out zone")
        add_ko.clicked.connect(lambda: self._add_keep_out_row(KeepOutZone("Keep-out", (0, 0, 0), (10, 10, 10))))
        del_ko = QPushButton("Remove keep-out zone")
        del_ko.clicked.connect(lambda: self.keep_out_table.removeRow(self.keep_out_table.currentRow()))
        ko_row.addWidget(add_ko)
        ko_row.addWidget(del_ko)
        form.addRow(QLabel("Keep-out zones"))
        form.addRow(self.keep_out_table)
        form.addRow(ko_row)
        save = QPushButton("Save machine")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        form.addRow(save)
        right = QWidget()
        right.setLayout(form)
        root.addWidget(right, 2)
        self._refresh_list()

    @staticmethod
    def _spin(lo: float, hi: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(1)
        spin.setValue(value)
        return spin

    def _refresh_list(self, select_id: str | None = None) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for machine in self.machines:
            self.list_widget.addItem(machine.name)
        self.list_widget.blockSignals(False)
        if self.machines:
            index = next((i for i, m in enumerate(self.machines) if m.machine_id == select_id), 0)
            self.list_widget.setCurrentRow(index)
            self._select(index)

    def _add_keep_out_row(self, zone: KeepOutZone) -> None:
        row = self.keep_out_table.rowCount()
        self.keep_out_table.insertRow(row)
        values = [zone.name, *zone.min_xyz, *zone.max_xyz]
        for col, value in enumerate(values):
            self.keep_out_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _select(self, index: int) -> None:
        if not 0 <= index < len(self.machines):
            return
        m = self.machines[index]
        self.name.setText(m.name)
        self.controller.setCurrentText(m.controller)
        for spin, value in zip(self.travel, m.travel_mm):
            spin.setValue(value)
        for spin, value in zip(self.max_feed, m.max_feed_mm_min):
            spin.setValue(value)
        self.rapid.setValue(m.rapid_mm_min)
        self.rpm_min.setValue(int(m.spindle_min_rpm))
        self.rpm_max.setValue(int(m.spindle_max_rpm))
        self.homing.setCurrentText(m.homing_direction)
        self.probe.setChecked(m.probe_present)
        self.tool_change.setCurrentText(m.tool_change_mode)
        self.notes.setPlainText(m.notes)
        self.keep_out_table.setRowCount(0)
        for zone in m.keep_out_zones:
            self._add_keep_out_row(zone)

    def current_machine(self) -> MachineProfile | None:
        index = self.list_widget.currentRow()
        return self.machines[index] if 0 <= index < len(self.machines) else None

    def _read_keep_outs(self) -> list[KeepOutZone]:
        zones = []
        for row in range(self.keep_out_table.rowCount()):
            try:
                cells = [self.keep_out_table.item(row, c).text() for c in range(7)]
                nums = [float(v) for v in cells[1:]]
            except (AttributeError, ValueError):
                continue
            zones.append(KeepOutZone(cells[0] or "Keep-out", tuple(nums[:3]), tuple(nums[3:])).normalized())
        return zones

    def _save(self) -> None:
        m = self.current_machine()
        if m is None:
            m = MachineProfile()
            self.machines.append(m)
        m.name = self.name.text().strip() or "Machine"
        m.controller = self.controller.currentText()
        m.travel_mm = tuple(s.value() for s in self.travel)
        m.max_feed_mm_min = tuple(s.value() for s in self.max_feed)
        m.rapid_mm_min = self.rapid.value()
        m.spindle_min_rpm = self.rpm_min.value()
        m.spindle_max_rpm = self.rpm_max.value()
        m.homing_direction = self.homing.currentText()
        m.probe_present = self.probe.isChecked()
        m.tool_change_mode = self.tool_change.currentText()
        m.notes = self.notes.toPlainText()
        m.keep_out_zones = self._read_keep_outs()
        self.store.save(m)
        self._refresh_list(m.machine_id)
        self.machinesChanged.emit()

    def _new(self) -> None:
        m = MachineProfile(name=f"Machine {len(self.machines) + 1}")
        self.machines.append(m)
        self.store.save(m)
        self._refresh_list(m.machine_id)
        self.machinesChanged.emit()

    def _remove(self) -> None:
        m = self.current_machine()
        if m is None:
            return
        self.store.delete(m.machine_id)
        self.machines = self.store.list()
        self._refresh_list()
        self.machinesChanged.emit()


class SoftLimitWizard(QDialog):
    """Record hard limits from the current machine position and set soft margins per axis and direction."""

    def __init__(self, machine: MachineProfile, gates, tools: list, position_provider, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Soft limits · {machine.name}")
        self.machine = machine
        self.gates = gates
        self.tools = tools
        self.position_provider = position_provider  # returns current MPos (x, y, z) or None
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Jog each axis with the controls until its endstop registers, then press Record. "
            "That position is the hard limit; the soft limit sits inside it by the margin (default 20 mm, minimum 5 mm)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        z_box = QGroupBox("Z: mounted tool")
        z_form = QFormLayout(z_box)
        z_note = QLabel(
            "Set the tool to the maximum length it will extend in any job. The longest tool sets the lowest Z; "
            f"stick-out = overall length - {COLLET_GRIP_MM:g} mm in the collet."
        )
        z_note.setWordWrap(True)
        z_form.addRow(z_note)
        self.tool_combo = QComboBox()
        for tool in tools:
            self.tool_combo.addItem(f"{tool.name} · stick-out {tool_stick_out(tool):g} mm")
        z_form.addRow("Mounted tool", self.tool_combo)
        layout.addWidget(z_box)
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        grid_box = QGroupBox("Limits (machine coordinates)")
        grid = QFormLayout(grid_box)
        for axis in AXES:
            limit = machine.soft_limits.setdefault(axis, SoftLimit())
            for side in ("min", "max"):
                hard = getattr(limit, f"hard_{side}")
                label = QLabel("not set" if hard is None else f"{hard:.3f}")
                record = QPushButton("Record")
                record.setToolTip(f"Store the current {axis} machine position as the {side} hard limit.")
                record.clicked.connect(lambda _=False, a=axis, s=side: self.record(a, s))
                margin = QDoubleSpinBox()
                margin.setRange(0.0, 500.0)
                margin.setValue(getattr(limit, f"margin_{side}"))
                margin.setSuffix(" mm")
                released = QCheckBox("Released")
                released.setChecked(getattr(limit, f"released_{side}"))
                released.setToolTip("Turn this soft limit off; needs the release gate in Settings > Advanced.")
                row = QHBoxLayout()
                for widget in (label, record, QLabel("margin"), margin, released):
                    row.addWidget(widget)
                grid.addRow(f"{axis} {side}", row)
                self.rows[(axis, side)] = {"label": label, "margin": margin, "released": released}
        layout.addWidget(grid_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def record(self, axis: str, side: str, value: float | None = None) -> None:
        if value is None:
            pos = self.position_provider() if self.position_provider else None
            if pos is None:
                QMessageBox.information(self, "Soft limits", "No machine position available; connect and jog first.")
                return
            value = float(pos[AXES.index(axis)])
        setattr(self.machine.soft_limits[axis], f"hard_{side}", float(value))
        self.rows[(axis, side)]["label"].setText(f"{value:.3f}")
        if axis == "Z" and side == "min" and self.tools:
            tool = self.tools[max(0, self.tool_combo.currentIndex())]
            self.machine.z_reference_tool = tool.name
            self.machine.z_reference_stick_out_mm = tool_stick_out(tool)

    def apply(self) -> None:
        allowed = bool(self.gates and self.gates.allows("release_soft_limits"))
        for (axis, side), row in self.rows.items():
            limit = self.machine.soft_limits[axis]
            set_soft_margin(limit, side, row["margin"].value(), allowed)
            set_released(limit, side, row["released"].isChecked(), allowed)

    def _save(self) -> None:
        try:
            self.apply()
        except LimitPolicyError as error:
            QMessageBox.warning(self, "Soft limits", str(error))
            return
        self.accept()


class PreflightDialog(QDialog):
    """Move each axis through its travel and tick it; motion stays disabled until every axis is ticked."""

    def __init__(self, machine: MachineProfile | None, axes: str = "XYZ", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preflight check")
        self.machine = machine
        self.wants_soft_limit_setup = False
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Before motion is enabled, move each axis through its full travel and confirm it moves freely. "
            "Tick each axis when done. Jog controls stay limited to preflight until every axis is ticked."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.boxes: dict[str, QCheckBox] = {}
        for axis in axes:
            box = QCheckBox(f"{axis} axis moved through its travel")
            box.toggled.connect(self._update)
            layout.addWidget(box)
            self.boxes[axis] = box
        self.soft_hint = QLabel("")
        self.soft_hint.setWordWrap(True)
        layout.addWidget(self.soft_hint)
        self.setup_soft = QCheckBox("Lead me through setting soft limits after this")
        layout.addWidget(self.setup_soft)
        if machine is None or not machine.has_soft_limits():
            self.soft_hint.setText("No soft limits are set for this machine. Setting them is recommended.")
            self.setup_soft.setChecked(True)
        else:
            self.setup_soft.hide()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update()

    def is_complete(self) -> bool:
        return all(box.isChecked() for box in self.boxes.values())

    def _update(self) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self.is_complete())

    def _accept(self) -> None:
        if self.is_complete():
            self.wants_soft_limit_setup = self.setup_soft.isVisible() and self.setup_soft.isChecked()
            self.accept()


class BedFitDialog(QDialog):
    """No machine selected: choose a compatible machine, add one, or skip."""

    def __init__(self, compatible: list[MachineProfile], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose a machine")
        self.choice: str = "skip"
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("These machines in the inventory are large enough for the part:"))
        self.combo = QComboBox()
        for machine in compatible:
            self.combo.addItem(machine.name, machine.machine_id)
        layout.addWidget(self.combo)
        row = QHBoxLayout()
        for text, choice in (("Use this machine", "use"), ("Add a machine...", "add"), ("Skip", "skip")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, c=choice: self._choose(c))
            row.addWidget(button)
        layout.addLayout(row)

    def _choose(self, choice: str) -> None:
        self.choice = choice
        self.accept()

    def selected_id(self) -> str | None:
        return self.combo.currentData()


__all__ = [
    "AXES", "SoftLimit", "KeepOutZone", "MachineProfile", "MachineStore", "MachinesTab", "SoftLimitWizard",
    "PreflightDialog", "BedFitDialog", "fit_report", "compatible_machines", "generation_fit_rule",
    "set_soft_margin", "set_released", "tool_stick_out", "z_min_for_tool", "LimitPolicyError",
]
