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
    QAbstractSpinBox,
    QGridLayout,
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
from basicam.auxiliaries import AuxOutput
from basicam.bar_feed import BarFeed
from basicam.window_fit import scrollable
from basicam.rotary_devices import DEVICE_KINDS, MOUNTS, RotaryDevice
from basicam.tool_changer import CHANGER_LABELS, ToolChanger, ToolTable
from basicam.machine_types import (
    CONTROLLERS,
    LAYOUT_LABELS,
    PROCESSES,
    AxisSpec,
    Head,
    axis_problems,
    default_axes,
)
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
    # v2 Block 10: process, kinematic layout, axes, heads (old machines load as a 3-axis mill)
    process: str = "mill"
    layout: str = "gantry3"
    axes: list[AxisSpec] = field(default_factory=list)  # empty: derived from the layout and travel
    heads: list[Head] = field(default_factory=list)
    rotary_axis: str = "A"  # 4th axis: "A" (about X) or "B" (about Y)
    tilt_axis: str = "A"  # trunnion tilt: "A" or "B"
    rotary_centre_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pivot_length_mm: float = 0.0
    diameter_mode: bool = True  # lathes: X reads the diameter
    wire_height_mm: float = 50.0  # wire EDM guide spacing
    # v2 Block 11: tool changer and the tool table (pocket -> tool)
    tool_changer: ToolChanger = field(default_factory=ToolChanger)
    tool_table: ToolTable = field(default_factory=ToolTable)
    # v2 Block 12: auxiliary outputs (air, gas, coolant, fluid, water, vacuum, dust, fumes)
    aux_outputs: list[AuxOutput] = field(default_factory=list)
    # v2 Block 13: lasers with fast beam gating (shutter / fly cut) may flash-cut rows of features
    fast_beam_gating: bool = False
    gate_method: str = "M4 S0"  # "shutter", "fast PWM / gate" or "M4 S0" between features
    min_gate_s: float = 0.002
    # v2 Block 14: rotary devices (one active)
    rotary_devices: list[RotaryDevice] = field(default_factory=list)
    active_rotary: int = 0
    bar_feed: BarFeed = field(default_factory=BarFeed)  # v2 Block 22
    # v3 Block 2: how BasiCAM reaches this machine (auto_connect is reserved; it stays off)
    port: str = ""
    baud: int = 115200
    auto_connect: bool = False
    boot_wait_s: float = 3.0  # v3 Block 7: how long to wait for the controller's start-up message
    has_homing_switches: bool = True  # v3 Block 8: without switches the owner sets the position by hand

    def rotary_device(self) -> RotaryDevice | None:
        if not self.rotary_devices:
            return None
        return self.rotary_devices[min(max(self.active_rotary, 0), len(self.rotary_devices) - 1)]

    def aux_output_list(self) -> list[AuxOutput]:
        """Configured auxiliaries; unconfigured machines get only the standard GRBL coolant
        outputs (M7 / M8), plus air assist on M8 for lasers."""
        if self.aux_outputs:
            return list(self.aux_outputs)
        outputs = [AuxOutput("flood"), AuxOutput("mist")]
        if self.process in ("laser", "rotary_laser", "laser_lathe"):
            outputs.append(AuxOutput("air_assist"))
        return outputs

    def axis_specs(self) -> list[AxisSpec]:
        if self.axes:
            return list(self.axes)
        axes = default_axes(self.layout, self.travel_mm, self.homing_direction)
        if self.layout == "gantry3_rotary" and self.rotary_axis == "B":
            axes[-1].name = "B"
        if self.layout == "table_table5" and self.tilt_axis == "B":
            axes[3].name = "B"
        return axes

    def axis_names(self) -> list[str]:
        return [a.name for a in self.axis_specs()]

    def controller_problems(self) -> list[str]:
        return axis_problems(self.axis_specs(), self.controller)

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
        data["tool_changer"] = asdict(self.tool_changer)
        data["tool_table"] = self.tool_table.to_dict()
        data["rotary_devices"] = [r.to_dict() for r in self.rotary_devices]
        data["bar_feed"] = self.bar_feed.to_dict()
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
        values["axes"] = [AxisSpec.from_dict(a) for a in data.get("axes", []) or []]
        values["tool_changer"] = ToolChanger.from_dict(data.get("tool_changer"))
        values["aux_outputs"] = [AuxOutput.from_dict(a) for a in data.get("aux_outputs", []) or []]
        values["rotary_devices"] = [RotaryDevice.from_dict(r) for r in data.get("rotary_devices", []) or []]
        values["bar_feed"] = BarFeed.from_dict(data.get("bar_feed"))
        values["tool_table"] = ToolTable.from_dict(data.get("tool_table"))
        values["heads"] = [Head.from_dict(h) for h in data.get("heads", []) or []]
        if "rotary_centre_mm" in values:
            values["rotary_centre_mm"] = tuple(float(v) for v in values["rotary_centre_mm"])
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
BAUD_RATES = ("115200", "250000", "230400", "57600", "38400", "19200", "9600")


class MachinesTab(QWidget):
    """Machine inventory editor (a tab in the tool inventory window).

    Edits are tracked: switching machine, New, Remove or closing with unsaved edits asks
    Save / Discard / Cancel. No button is a default button, so Enter never creates or removes
    a machine."""

    machinesChanged = Signal()
    dirtyChanged = Signal(bool)

    def __init__(self, store: MachineStore, parent=None, connect_callback=None, is_connected=None) -> None:
        super().__init__(parent)
        self.store = store
        self.machines = store.list()
        self.connect_callback = connect_callback  # main window: connect to a profile (normal path)
        self.is_connected = is_connected or (lambda _m: False)
        self._dirty = False
        self._loading = False
        self._row = -1
        self._unsaved_new: set[str] = set()  # created here, not yet saved by the owner
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._row_changed)
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
        self.controller.addItems(list(CONTROLLERS))
        self.controller.setToolTip("GRBL, grblHAL and FluidNC are run from BasiCAM; export-only controllers get a G-code file.")
        form.addRow("Controller", self.controller)
        self.port = QComboBox()
        self.port.setEditable(True)
        self.port.setToolTip("Serial port this machine is plugged into (Machine > Ports... lists them).")
        self._fill_ports()
        form.addRow("Port", self.port)
        self.baud = QComboBox()
        self.baud.setEditable(True)
        self.baud.addItems(list(BAUD_RATES))
        self.baud.setToolTip("Serial speed set in the controller firmware; GRBL and grblHAL usually use 115200.")
        form.addRow("Baud rate", self.baud)
        self.process = QComboBox()
        for key, label in PROCESSES.items():
            self.process.addItem(label, key)
        form.addRow("Process", self.process)
        self.layout_combo = QComboBox()
        for key, label in LAYOUT_LABELS.items():
            self.layout_combo.addItem(label, key)
        self.layout_combo.currentIndexChanged.connect(self._layout_changed)
        form.addRow("Axes layout", self.layout_combo)
        self.axis_table = QTableWidget(0, 5)
        self.axis_table.setHorizontalHeaderLabels(["Axis", "Type", "Min", "Max", "Continuous"])
        self.axis_table.setToolTip("Linear axes in mm, rotary axes in degrees; a continuous rotary axis has no end stops.")
        self.axis_table.setMinimumHeight(120)
        form.addRow(self.axis_table)
        self.changer_kind = QComboBox()
        for key, label in CHANGER_LABELS.items():
            self.changer_kind.addItem(label, key)
        form.addRow("Tool changer", self.changer_kind)
        self.changer_pockets = QSpinBox()
        self.changer_pockets.setRange(0, 200)
        self.changer_pockets.setToolTip("How many tools the changer holds (stations on a turret).")
        form.addRow("Pockets", self.changer_pockets)
        self.changer_time = self._spin(0, 600, 0)
        self.changer_time.setToolTip("Seconds per tool change for the cycle time (0 = 8 s with a changer, 60 s by hand).")
        form.addRow("Tool change time · s", self.changer_time)
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
        self.homing_switches = QCheckBox("Homing switches fitted")
        self.homing_switches.setToolTip("Untick for a machine without homing switches: you then tell BasiCAM where the tool is.")
        form.addRow("", self.homing_switches)
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
        rotary = QPushButton("Rotary axis setup...")
        rotary.setToolTip("Headstock, chuck / collet / rollers, jaws, tailstock, steady rest and speed control.")
        rotary.clicked.connect(self._edit_rotary)
        form.addRow(rotary)
        save = QPushButton("Save machine")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        self.save_button = save
        right = QWidget()
        right.setLayout(form)
        # The form scrolls; Save stays pinned below it so it is always reachable.
        right_column = QVBoxLayout()
        right_column.addWidget(scrollable(right), 1)
        self.dirty_label = QLabel("Unsaved changes")
        self.dirty_label.setObjectName("warningLabel")
        self.dirty_label.setVisible(False)
        bottom = QHBoxLayout()
        bottom.addWidget(self.dirty_label, 1)
        self.connect_button = QPushButton("Connect")
        self.connect_button.setToolTip("Connect to this machine on its port (asks first, like Machine > Connect).")
        self.connect_button.clicked.connect(self._connect_current)
        self.connect_button.setVisible(connect_callback is not None)
        bottom.addWidget(self.connect_button)
        bottom.addWidget(save)
        right_column.addLayout(bottom)
        root.addLayout(right_column, 2)
        for button in self.findChildren(QPushButton):  # Enter never presses New / Remove / Save
            button.setAutoDefault(False)
            button.setDefault(False)
        self._watch_edits(right)
        self._refresh_list()

    # ------------------------------------------------------------ unsaved edits (v3 Block 2)
    def _watch_edits(self, container: QWidget) -> None:
        for widget in container.findChildren(QLineEdit):
            if isinstance(widget.parent(), (QComboBox, QAbstractSpinBox)):
                continue
            widget.textEdited.connect(self._mark_dirty)
        for widget in container.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self._mark_dirty)
            if widget.isEditable():
                widget.editTextChanged.connect(self._mark_dirty)
        for widget in container.findChildren(QAbstractSpinBox):
            widget.valueChanged.connect(self._mark_dirty)
        for widget in container.findChildren(QCheckBox):
            widget.toggled.connect(self._mark_dirty)
        for widget in container.findChildren(QTextEdit):
            widget.textChanged.connect(self._mark_dirty)
        for widget in container.findChildren(QTableWidget):
            widget.itemChanged.connect(self._mark_dirty)
            widget.model().rowsInserted.connect(self._mark_dirty)
            widget.model().rowsRemoved.connect(self._mark_dirty)

    def is_dirty(self) -> bool:
        return self._dirty

    def _mark_dirty(self, *_args) -> None:
        if self._loading or self._dirty or self.current_machine() is None:
            return
        self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self.dirty_label.setVisible(dirty)
        row = self._row
        if 0 <= row < self.list_widget.count() and row < len(self.machines):
            self.list_widget.item(row).setText(self._list_text(self.machines[row]) + (" *" if dirty else ""))
        self.dirtyChanged.emit(dirty)

    def maybe_save(self) -> bool:
        """Unsaved edits: ask Save / Discard / Cancel. True means it is fine to move on."""
        if not self._dirty:
            return True
        m = self.current_machine()
        answer = QMessageBox.question(
            self,
            "Unsaved machine",
            f"Save the changes to {m.name if m else 'this machine'}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save()
        if answer == QMessageBox.StandardButton.Discard:
            self._select(self._row)  # back to the stored values
            return True
        return False

    def _row_changed(self, index: int) -> None:
        if index == self._row:
            return
        if self._dirty:
            self.list_widget.blockSignals(True)
            self.list_widget.setCurrentRow(self._row)  # stay until the owner decides
            self.list_widget.blockSignals(False)
            target_id = self.machines[index].machine_id if 0 <= index < len(self.machines) else None
            if not self.maybe_save():
                return
            index = next((i for i, m in enumerate(self.machines) if m.machine_id == target_id), -1)
            if index < 0:
                return
            self.list_widget.blockSignals(True)
            self.list_widget.setCurrentRow(index)
            self.list_widget.blockSignals(False)
        self._select(index)

    def _list_text(self, m: "MachineProfile") -> str:
        text = m.name
        if m.port:
            text += f" · {m.port}"
            if self.is_connected(m):
                text += " (connected)"
        return text

    def _fill_ports(self) -> None:
        from basicam.machine import available_serial_ports

        current = self.port.currentText()
        self.port.blockSignals(True)
        self.port.clear()
        self.port.addItem("")
        try:
            for device, _description in available_serial_ports():
                self.port.addItem(device)
        except Exception:  # noqa: BLE001 - a missing port list must not break the editor
            pass
        self.port.setCurrentText(current)
        self.port.blockSignals(False)

    def _connect_current(self) -> None:
        m = self.current_machine()
        if m is None or self.connect_callback is None:
            return
        if self._dirty and not self.maybe_save():
            return
        m = self.current_machine()
        if not m.port:
            QMessageBox.information(self, "Connect", "Set this machine's port first, then save it.")
            return
        self.connect_callback(m)
        self._refresh_list(m.machine_id)

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
            self.list_widget.addItem(self._list_text(machine))
        if self.machines:
            index = next((i for i, m in enumerate(self.machines) if m.machine_id == select_id), 0)
            self.list_widget.setCurrentRow(index)
            self.list_widget.blockSignals(False)
            self._select(index)
        else:
            self.list_widget.blockSignals(False)
            self._row = -1
            self._set_dirty(False)

    def _add_keep_out_row(self, zone: KeepOutZone) -> None:
        row = self.keep_out_table.rowCount()
        self.keep_out_table.insertRow(row)
        values = [zone.name, *zone.min_xyz, *zone.max_xyz]
        for col, value in enumerate(values):
            self.keep_out_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _select(self, index: int) -> None:
        if not 0 <= index < len(self.machines):
            return
        self._loading = True
        try:
            self._load_fields(index)
        finally:
            self._loading = False
        self._row = index
        self._set_dirty(False)

    def _load_fields(self, index: int) -> None:
        m = self.machines[index]
        self.port.setCurrentText(m.port)
        self.baud.setCurrentText(str(m.baud))
        self.name.setText(m.name)
        self.controller.setCurrentText(m.controller)
        index = self.process.findData(getattr(m, "process", "mill"))
        self.process.setCurrentIndex(max(0, index))
        self.layout_combo.blockSignals(True)
        self.layout_combo.setCurrentIndex(max(0, self.layout_combo.findData(getattr(m, "layout", "gantry3"))))
        self.layout_combo.blockSignals(False)
        self._fill_axes(m.axis_specs())
        self.changer_kind.setCurrentIndex(max(0, self.changer_kind.findData(m.tool_changer.kind)))
        self.changer_pockets.setValue(int(m.tool_changer.pockets))
        self.changer_time.setValue(float(m.tool_changer.change_time_s))
        for spin, value in zip(self.travel, m.travel_mm):
            spin.setValue(value)
        for spin, value in zip(self.max_feed, m.max_feed_mm_min):
            spin.setValue(value)
        self.rapid.setValue(m.rapid_mm_min)
        self.rpm_min.setValue(int(m.spindle_min_rpm))
        self.rpm_max.setValue(int(m.spindle_max_rpm))
        self.homing.setCurrentText(m.homing_direction)
        self.homing_switches.setChecked(bool(getattr(m, "has_homing_switches", True)))
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

    def _edit_rotary(self) -> None:
        m = self.current_machine()
        if m is None:
            return
        dialog = RotaryDeviceDialog(m.rotary_device(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if m.rotary_devices:
            m.rotary_devices[min(max(m.active_rotary, 0), len(m.rotary_devices) - 1)] = dialog.device
        else:
            m.rotary_devices.append(dialog.device)
            m.active_rotary = 0
        self.store.save(m)
        self.machinesChanged.emit()

    def _layout_changed(self, _index: int = 0) -> None:
        layout = self.layout_combo.currentData() or "gantry3"
        travel = tuple(s.value() for s in self.travel)
        self._fill_axes(default_axes(layout, travel, self.homing.currentText()))

    def _fill_axes(self, axes: list) -> None:
        self.axis_table.setRowCount(0)
        for axis in axes:
            row = self.axis_table.rowCount()
            self.axis_table.insertRow(row)
            for column, value in enumerate((axis.name, axis.kind, f"{axis.minimum:g}", f"{axis.maximum:g}", "yes" if axis.continuous else "no")):
                self.axis_table.setItem(row, column, QTableWidgetItem(value))

    def _read_axes(self) -> list:
        axes = []
        for row in range(self.axis_table.rowCount()):
            cells = [self.axis_table.item(row, c).text().strip() if self.axis_table.item(row, c) else "" for c in range(5)]
            try:
                lo, hi = float(cells[2]), float(cells[3])
            except ValueError:
                lo, hi = 0.0, 0.0
            axes.append(AxisSpec(cells[0] or "X", cells[1] if cells[1] in ("linear", "rotary") else "linear", lo, hi,
                                 continuous=cells[4].casefold() in ("yes", "true", "1")))
        return axes

    def _save(self) -> bool:
        m = self.current_machine()
        if m is None:
            m = MachineProfile()
            self.machines.append(m)
        was_new = m.machine_id in self._unsaved_new
        try:
            baud = int(self.baud.currentText().strip() or 115200)
        except ValueError:
            QMessageBox.warning(self, "Baud rate", "The baud rate must be a whole number, for example 115200.")
            return False
        m.port = self.port.currentText().strip()
        m.baud = baud
        m.name = self.name.text().strip() or "Machine"
        m.controller = self.controller.currentText()
        m.travel_mm = tuple(s.value() for s in self.travel)
        m.max_feed_mm_min = tuple(s.value() for s in self.max_feed)
        m.rapid_mm_min = self.rapid.value()
        m.spindle_min_rpm = self.rpm_min.value()
        m.spindle_max_rpm = self.rpm_max.value()
        m.homing_direction = self.homing.currentText()
        m.has_homing_switches = self.homing_switches.isChecked()
        m.probe_present = self.probe.isChecked()
        m.tool_change_mode = self.tool_change.currentText()
        m.notes = self.notes.toPlainText()
        m.keep_out_zones = self._read_keep_outs()
        m.process = self.process.currentData() or "mill"
        m.layout = self.layout_combo.currentData() or "gantry3"
        m.axes = self._read_axes()
        m.tool_changer.kind = self.changer_kind.currentData() or "none"
        m.tool_changer.pockets = int(self.changer_pockets.value())
        m.tool_changer.change_time_s = float(self.changer_time.value())
        problems = m.controller_problems()
        if problems:
            QMessageBox.warning(self, "Controller", "\n".join(problems))
            return False
        self.store.save(m)
        self._unsaved_new.discard(m.machine_id)
        self._dirty = False
        self._refresh_list(m.machine_id)
        self.machinesChanged.emit()
        if was_new and m.port and self.connect_callback is not None:
            answer = QMessageBox.question(
                self, "Connect now?", f"{m.name} is saved. Connect to it on {m.port} now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.connect_callback(m)
                self._refresh_list(m.machine_id)
        return True

    def _new(self) -> None:
        if not self.maybe_save():
            return
        m = MachineProfile(name=f"Machine {len(self.machines) + 1}")
        self.machines.append(m)
        self.store.save(m)
        self._unsaved_new.add(m.machine_id)
        self._refresh_list(m.machine_id)
        self.machinesChanged.emit()
        self.name.setFocus()
        self.name.selectAll()

    def _remove(self) -> None:
        m = self.current_machine()
        if m is None:
            return
        answer = QMessageBox.question(
            self, "Remove machine", f"Remove {m.name} from the inventory? Projects keep their own copy.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._dirty = False
        self.store.delete(m.machine_id)
        self.machines = self.store.list()
        self._refresh_list()
        self.machinesChanged.emit()


class SoftLimitWizard(QDialog):
    """Record hard limits from the current machine position and set soft margins per axis and direction."""

    def __init__(self, machine: MachineProfile, gates, tools: list, position_provider, parent=None,
                 pad=None, kill=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Soft limits · {machine.name}")
        self.pad = pad  # v3 Block 9: the jog pad (same guarded jogs as the machine panel)
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
        if pad is not None:
            layout.addWidget(pad)
        elif kill is not None:
            kill_button = QPushButton("KILL")
            kill_button.setToolTip("Feed hold, soft reset, spindle and beam off.")
            kill_button.setAutoDefault(False)
            kill_button.clicked.connect(kill)
            layout.addWidget(kill_button)
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

    def __init__(self, machine: MachineProfile | None, axes: str = "XYZ", parent=None, pad=None,
                 actions: dict | None = None) -> None:
        """``pad``: the host's jog pad (preflight jogs); ``actions``: home / unlock / hold / kill callables.
        With a pad, an axis's tick is offered once it has moved both ways during this preflight."""
        super().__init__(parent)
        self.setWindowTitle("Preflight check")
        self.setObjectName("preflightWindow")
        self.machine = machine
        self.wants_soft_limit_setup = False
        self.pad = pad
        self.moved: dict[str, set[int]] = {a: set() for a in axes}
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Before motion is enabled, move each axis through its full travel with the arrows below and "
            "confirm it moves freely. Tick each axis when done. Until then only these preflight jogs work."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        self.boxes: dict[str, QCheckBox] = {}
        self.position_labels: dict[str, QLabel] = {}
        self.pin_labels: dict[str, QLabel] = {}
        for row, axis in enumerate(axes):
            box = QCheckBox(f"{axis} axis moved through its travel")
            box.toggled.connect(self._update)
            if pad is not None:
                box.setEnabled(False)
                box.setToolTip("Move this axis both ways first.")
            self.position_labels[axis] = QLabel("-")
            self.pin_labels[axis] = QLabel("")
            grid.addWidget(box, row, 0)
            grid.addWidget(self.position_labels[axis], row, 1)
            grid.addWidget(self.pin_labels[axis], row, 2)
            self.boxes[axis] = box
        layout.addLayout(grid)
        self.action_buttons: dict[str, QPushButton] = {}
        if actions:
            row = QHBoxLayout()
            for key, text in (("home", "Home"), ("unlock", "Unlock ($X)"), ("hold", "Feed hold"), ("kill", "KILL")):
                if key not in actions:
                    continue
                button = QPushButton(text)
                button.setAutoDefault(False)
                button.clicked.connect(lambda _c=False, f=actions[key]: f())
                if key == "kill":
                    row.addSpacing(16)  # set apart from the other buttons
                    button.setToolTip("Feed hold, soft reset, spindle and beam off.")
                row.addWidget(button)
                self.action_buttons[key] = button
            layout.addLayout(row)
        if pad is not None:
            layout.addWidget(pad, 1)
        self.soft_hint = QLabel("")
        self.soft_hint.setWordWrap(True)
        layout.addWidget(self.soft_hint)
        self.setup_soft = QCheckBox("Set up soft limits after this")
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

    def note_move(self, axis: str, sign: int) -> None:
        """A preflight jog was sent; both directions offer the axis's tick."""
        axis = axis.upper()
        if axis not in self.moved:
            return
        self.moved[axis].add(1 if sign > 0 else -1)
        if self.moved[axis] == {1, -1}:
            self.boxes[axis].setEnabled(True)
            self.boxes[axis].setToolTip("")

    def update_status(self, positions: dict, pins: str = "", enabled: dict | None = None) -> None:
        """Live machine position and limit inputs per axis; which action buttons may be pressed."""
        for axis, label in self.position_labels.items():
            value = positions.get(axis)
            label.setText("-" if value is None else f"{value:.3f}")
        active = str(pins or "").upper()
        for axis, label in self.pin_labels.items():
            label.setText("limit ON" if axis in active else "")
        for key, allowed in (enabled or {}).items():
            button = self.action_buttons.get(key)
            if button is not None:
                button.setEnabled(bool(allowed))

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


class RotaryDeviceDialog(QDialog):
    """Rotary device setup: kind, headstock, throw, mount, jaws, tailstock, steady rest, speed."""

    def __init__(self, device: RotaryDevice | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rotary axis setup")
        self.device = RotaryDevice.from_dict(device.to_dict()) if device is not None else RotaryDevice()
        d = self.device
        form = QFormLayout(self)
        self.name = QLineEdit(d.name)
        form.addRow("Name", self.name)
        self.kind = QComboBox()
        for key, label in DEVICE_KINDS.items():
            self.kind.addItem(label, key)
        self.kind.setCurrentIndex(max(0, self.kind.findData(d.kind)))
        form.addRow("Type", self.kind)
        self.axis = QComboBox()
        self.axis.addItems(["A", "B", "C"])
        self.axis.setCurrentText(d.axis)
        form.addRow("Axis letter", self.axis)
        self.orientation = QComboBox()
        self.orientation.addItems(["X", "Y"])
        self.orientation.setCurrentText(d.orientation)
        form.addRow("Spins about", self.orientation)
        self.centre = [self._spin(-5000, 5000, v) for v in d.centre_mm]
        centre_row = QHBoxLayout()
        for spin in self.centre:
            centre_row.addWidget(spin)
        form.addRow("Face centre X / Y / Z · mm", centre_row)
        self.headstock = self._spin(1, 2000, d.headstock_diameter_mm)
        form.addRow("Headstock / chuck diameter · mm", self.headstock)
        self.throw = self._spin(1, 2000, d.throw_mm)
        self.throw.setToolTip("Largest diameter that turns clear of the bed or slide (swing).")
        form.addRow("Throw (swing) · mm", self.throw)
        self.mount = QComboBox()
        self.mount.addItems(list(MOUNTS))
        self.mount.setCurrentText(d.mount)
        form.addRow("Workholding", self.mount)
        self.clamp_min = self._spin(0, 2000, d.clamp_min_mm)
        self.clamp_max = self._spin(0, 2000, d.clamp_max_mm)
        form.addRow("Grips from · mm", self.clamp_min)
        form.addRow("Grips up to · mm", self.clamp_max)
        self.jaw_width = self._spin(0, 500, d.jaw_width_mm)
        self.jaw_height = self._spin(0, 500, d.jaw_height_mm)
        self.jaw_depth = self._spin(0, 500, d.jaw_depth_mm)
        form.addRow("Jaw width · mm", self.jaw_width)
        form.addRow("Jaw height (radial) · mm", self.jaw_height)
        form.addRow("Jaw depth (proud of face) · mm", self.jaw_depth)
        self.tailstock = QCheckBox("Tailstock")
        self.tailstock.setChecked(d.tailstock)
        form.addRow("", self.tailstock)
        self.steady = QCheckBox("Steady rest")
        self.steady.setChecked(d.steady_rest)
        form.addRow("", self.steady)
        self.speed = QComboBox()
        self.speed.addItems(["automatic", "manual"])
        self.speed.setCurrentText(d.speed_control)
        self.speed.setToolTip("Manual: the program pauses and asks for the spindle / rotation speed to be set by hand.")
        form.addRow("Speed control", self.speed)
        self.max_rpm = self._spin(0, 100000, d.max_rpm)
        form.addRow("Max RPM", self.max_rpm)
        self.continuous = QCheckBox("Can turn continuously (turning / wrapping)")
        self.continuous.setChecked(d.continuous)
        form.addRow("", self.continuous)
        self.error = QLabel("")
        self.error.setObjectName("errorText")
        self.error.setWordWrap(True)
        form.addRow(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _spin(lo: float, hi: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(3)
        spin.setValue(float(value))
        return spin

    def result_device(self) -> RotaryDevice:
        d = RotaryDevice.from_dict(self.device.to_dict())
        d.name = self.name.text().strip() or "Rotary"
        d.kind = self.kind.currentData()
        d.axis = self.axis.currentText()
        d.orientation = self.orientation.currentText()
        d.centre_mm = tuple(s.value() for s in self.centre)
        d.headstock_diameter_mm = self.headstock.value()
        d.throw_mm = self.throw.value()
        d.mount = self.mount.currentText()
        d.clamp_min_mm, d.clamp_max_mm = self.clamp_min.value(), self.clamp_max.value()
        d.clamp_diameter_mm = min(max(d.clamp_diameter_mm, d.clamp_min_mm), d.clamp_max_mm)
        d.jaw_width_mm, d.jaw_height_mm, d.jaw_depth_mm = self.jaw_width.value(), self.jaw_height.value(), self.jaw_depth.value()
        d.tailstock, d.steady_rest = self.tailstock.isChecked(), self.steady.isChecked()
        d.speed_control = self.speed.currentText()
        d.max_rpm = self.max_rpm.value()
        d.continuous = self.continuous.isChecked()
        return d

    def _accept(self) -> None:
        problems = self.result_device().problems()
        if problems:
            self.error.setText(" ".join(problems))
            return
        self.device = self.result_device()
        self.accept()


__all__ = [
    "RotaryDeviceDialog",
    "AXES", "SoftLimit", "KeepOutZone", "MachineProfile", "MachineStore", "MachinesTab", "SoftLimitWizard",
    "PreflightDialog", "BedFitDialog", "fit_report", "compatible_machines", "generation_fit_rule",
    "set_soft_margin", "set_released", "tool_stick_out", "z_min_for_tool", "LimitPolicyError",
]
