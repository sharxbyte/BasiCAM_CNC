"""Manual controller jog pad (roadmap v3 Block 9).

X / Y arrows in a cross (diagonals move X and Y together), Z arrows in a column beside it, one
increment selector per axis with the usual steps plus Custom..., a Link X/Y toggle, XY and Z feeds,
an optional hold-to-move mode, rows for rotary / extra axes, and a KILL button set apart from the
arrows. The pad never talks to the controller itself: every button calls the host's guarded jog
(``jog(moves, feed) -> bool``), ``kill()``, ``go_to_zero()`` and, for hold-to-move,
``jog_hold(axis, sign, feed) -> bool`` / ``jog_cancel()``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

STEPS_MM = (0.01, 0.1, 1.0, 10.0, 100.0)
STEPS_IN = (0.001, 0.01, 0.1, 1.0)
STEPS_DEG = (0.1, 1.0, 5.0, 15.0, 45.0, 90.0)
CUSTOM = "Custom..."
MM_PER_IN = 25.4
ROTARY_LETTERS = "ABC"


def _fmt(value: float) -> str:
    return f"{value:g}"


class AxisIncrement(QComboBox):
    """Increment for one axis: the usual steps, any custom ones, and Custom... (typed with units)."""

    changedMm = Signal(float)

    def __init__(self, axis: str, inches: bool = False, rotary: bool = False, parse=None, parent=None) -> None:
        super().__init__(parent)
        self.axis = axis
        self.inches = inches
        self.rotary = rotary
        self.parse = parse  # text -> mm (units, fractions); None: plain numbers in the display unit
        self.custom: list[float] = []  # mm (or degrees)
        self._last_index = 0
        self._fill(1.0 if not inches and not rotary else (MM_PER_IN * 0.1 if inches else 1.0))
        self.activated.connect(self._activated)

    def unit(self) -> str:
        return "°" if self.rotary else ("in" if self.inches else "mm")

    def _defaults_mm(self) -> list[float]:
        if self.rotary:
            return list(STEPS_DEG)
        if self.inches:
            return [v * MM_PER_IN for v in STEPS_IN]
        return list(STEPS_MM)

    def _shown(self, mm: float) -> str:
        if self.rotary:
            return f"{_fmt(mm)}°"
        return f"{_fmt(round(mm / MM_PER_IN, 6))} in" if self.inches else f"{_fmt(mm)} mm"

    def _fill(self, select_mm: float) -> None:
        self.blockSignals(True)
        self.clear()
        values = sorted(set(round(v, 6) for v in self._defaults_mm() + self.custom))
        for value in values:
            self.addItem(self._shown(value), float(value))
        self.addItem(CUSTOM, None)
        self.blockSignals(False)
        self.set_mm(select_mm)

    def set_custom(self, values: list[float]) -> None:
        current = self.mm()
        self.custom = [float(v) for v in values if float(v) > 0]
        self._fill(current)

    def set_mm(self, value: float) -> None:
        value = round(float(value), 6)
        index = self.findData(value)
        if index < 0:
            self.custom.append(value)
            self._fill(value)
            return
        self.setCurrentIndex(index)
        self._last_index = index

    def mm(self) -> float:
        data = self.itemData(self._last_index if self.currentData() is None else self.currentIndex())
        return float(data) if data is not None else 1.0

    def _activated(self, index: int) -> None:
        if self.itemData(index) is not None:
            self._last_index = index
            self.changedMm.emit(self.mm())
            return
        text, ok = QInputDialog.getText(self, f"{self.axis} increment", f"Increment ({self.unit()}), for example "
                                        + ("2.5" if self.rotary else ("0.05 or 1/8\"" if self.inches else "0.05, 2.5 or 0.1in")))
        value = None
        if ok and text.strip():
            try:
                if self.rotary or self.parse is None:
                    value = float(text) * (MM_PER_IN if (self.inches and not self.rotary) else 1.0)
                else:
                    value = float(self.parse(text))
            except (TypeError, ValueError):
                value = None
        if value is None or value <= 0:
            if ok:
                QMessageBox.warning(self, "Increment", "Type a positive number, with a unit if you like (mm, in, \").")
            self.setCurrentIndex(self._last_index)
            return
        self.set_mm(value)
        self.changedMm.emit(self.mm())


class JogPad(QWidget):
    settingsChanged = Signal()

    def __init__(self, jog=None, kill=None, go_to_zero=None, jog_hold=None, jog_cancel=None, parse=None,
                 inches: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.jog_callback = jog
        self.kill_callback = kill
        self.zero_callback = go_to_zero
        self.hold_callback = jog_hold
        self.cancel_callback = jog_cancel
        self.parse = parse
        self.inches = inches
        self.flip = {"X": False, "Y": False}
        self.buttons: dict[str, QPushButton] = {}
        self.increments: dict[str, AxisIncrement] = {}
        self.extra_axes: list[str] = []
        self._holding: tuple[str, int] | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        pad = QGridLayout()
        pad.setSpacing(4)
        arrows = {  # key: (row, col, moves, text)
            "X-Y+": (0, 0, {"X": -1, "Y": 1}, "↖"), "Y+": (0, 1, {"Y": 1}, "▲\nY+"), "X+Y+": (0, 2, {"X": 1, "Y": 1}, "↗"),
            "X-": (1, 0, {"X": -1}, "◀ X−"), "X+": (1, 2, {"X": 1}, "X+ ▶"),
            "X-Y-": (2, 0, {"X": -1, "Y": -1}, "↙"), "Y-": (2, 1, {"Y": -1}, "Y−\n▼"), "X+Y-": (2, 2, {"X": 1, "Y": -1}, "↘"),
            "Z+": (0, 4, {"Z": 1}, "▲\nZ+"), "Z-": (2, 4, {"Z": -1}, "Z−\n▼"),
        }
        self.arrow_moves: dict[str, dict[str, int]] = {}
        for key, (row, col, moves, text) in arrows.items():
            button = QPushButton(text)
            button.setMinimumSize(52, 44)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            button.setAutoDefault(False)
            button.setToolTip("Jog " + " and ".join(f"{a}{'+' if s > 0 else '−'}" for a, s in moves.items()))
            button.pressed.connect(lambda k=key: self._pressed(k))
            button.released.connect(lambda k=key: self._released(k))
            button.clicked.connect(lambda _c=False, k=key: self._clicked(k))
            pad.addWidget(button, row, col)
            self.buttons[key] = button
            self.arrow_moves[key] = moves
        self.zero_button = QPushButton("XY 0")
        self.zero_button.setMinimumSize(52, 44)
        self.zero_button.setAutoDefault(False)
        self.zero_button.setToolTip("Go to work zero in X and Y at a safe height (asks first).")
        self.zero_button.clicked.connect(lambda: self.zero_callback() if self.zero_callback else None)
        pad.addWidget(self.zero_button, 1, 1)
        spacer = QLabel("")
        spacer.setFixedWidth(10)
        pad.addWidget(spacer, 0, 3)
        self.z_label = QLabel("Z")
        self.z_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pad.addWidget(self.z_label, 1, 4)
        root.addLayout(pad, 1)

        self.extra_box = QGridLayout()
        root.addLayout(self.extra_box)

        steps = QGridLayout()
        for row, axis in enumerate("XYZ"):
            steps.addWidget(QLabel(f"{axis} step"), row, 0)
            combo = AxisIncrement(axis, inches=inches, parse=parse)
            combo.changedMm.connect(lambda _v, a=axis: self._increment_changed(a))
            steps.addWidget(combo, row, 1)
            self.increments[axis] = combo
        self.link_xy = QCheckBox("Link X/Y")
        self.link_xy.setChecked(True)
        self.link_xy.setToolTip("Keep the X and Y steps equal.")
        self.link_xy.toggled.connect(lambda on: self._increment_changed("X") if on else self.settingsChanged.emit())
        steps.addWidget(self.link_xy, 0, 2, 2, 1)
        root.addLayout(steps)

        feeds = QHBoxLayout()
        unit = "in/min" if inches else "mm/min"
        self.feed_xy = QDoubleSpinBox()
        self.feed_xy.setRange(1.0, 100000.0)
        self.feed_xy.setDecimals(1)
        self.feed_xy.setValue(1000.0)
        self.feed_xy.setSuffix(f" {unit}")
        self.feed_z = QDoubleSpinBox()
        self.feed_z.setRange(1.0, 100000.0)
        self.feed_z.setDecimals(1)
        self.feed_z.setValue(300.0)
        self.feed_z.setSuffix(f" {unit}")
        feeds.addWidget(QLabel("XY feed"))
        feeds.addWidget(self.feed_xy, 1)
        feeds.addWidget(QLabel("Z feed"))
        feeds.addWidget(self.feed_z, 1)
        self.feed_xy.valueChanged.connect(lambda *_: self.settingsChanged.emit())
        self.feed_z.valueChanged.connect(lambda *_: self.settingsChanged.emit())
        root.addLayout(feeds)

        options = QHBoxLayout()
        self.hold_mode = QCheckBox("Hold to move")
        self.hold_mode.setToolTip("Hold an arrow to move, release to stop (uses the controller's jog cancel).")
        self.hold_mode.toggled.connect(lambda *_: self.settingsChanged.emit())
        options.addWidget(self.hold_mode)
        self.flip_x = QCheckBox("Flip X arrows")
        self.flip_x.setToolTip("Swap the X arrows when + X moves the other way on this machine.")
        self.flip_x.toggled.connect(lambda on: self._set_flip("X", on))
        options.addWidget(self.flip_x)
        self.flip_y = QCheckBox("Flip Y arrows")
        self.flip_y.setToolTip("Swap the Y arrows when + Y moves the other way on this machine.")
        self.flip_y.toggled.connect(lambda on: self._set_flip("Y", on))
        options.addWidget(self.flip_y)
        options.addStretch(1)
        root.addLayout(options)

        bottom = QHBoxLayout()
        self.readout = QLabel("")
        self.readout.setObjectName("mutedText")
        self.readout.setWordWrap(True)
        bottom.addWidget(self.readout, 1)
        bottom.addSpacing(16)  # KILL is kept apart from the arrows
        self.kill_button = QPushButton("KILL")
        self.kill_button.setObjectName("padKillButton")
        self.kill_button.setMinimumSize(110, 48)
        self.kill_button.setAutoDefault(False)
        self.kill_button.setToolTip("Feed hold, soft reset, spindle and beam off.")
        self.kill_button.setStyleSheet("QPushButton#padKillButton { font-weight: 800; font-size: 15px; }")
        self.kill_button.clicked.connect(lambda: self.kill_callback() if self.kill_callback else None)
        bottom.addWidget(self.kill_button)
        root.addLayout(bottom)
        self.reason = QLabel("")
        self.reason.setObjectName("warningLabel")
        self.reason.setWordWrap(True)
        self.reason.setVisible(False)
        root.addWidget(self.reason)
        self._update_readout()

    # ------------------------------------------------------------ increments and settings
    def increment_mm(self, axis: str) -> float:
        combo = self.increments.get(axis.upper())
        return combo.mm() if combo is not None else 1.0

    def _increment_changed(self, axis: str) -> None:
        if self.link_xy.isChecked() and axis in ("X", "Y"):
            other = self.increments["Y" if axis == "X" else "X"]
            source = self.increments[axis]
            other.set_custom(source.custom)
            other.set_mm(source.mm())
        self._update_readout()
        self.settingsChanged.emit()

    def _set_flip(self, axis: str, on: bool) -> None:
        self.flip[axis] = bool(on)
        self.settingsChanged.emit()

    def feed_for(self, moves: dict[str, int]) -> float:
        scale = MM_PER_IN if self.inches else 1.0
        if any(a in ("X", "Y") for a in moves):
            return self.feed_xy.value() * scale
        return self.feed_z.value() * scale

    def set_feed_limits(self, xy_max: float | None, z_max: float | None) -> None:
        """Clamp the feeds to the machine's maximum rates (mm/min)."""
        scale = MM_PER_IN if self.inches else 1.0
        if xy_max:
            self.feed_xy.setMaximum(max(1.0, float(xy_max) / scale))
        if z_max:
            self.feed_z.setMaximum(max(1.0, float(z_max) / scale))

    def state(self) -> dict:
        """Per-machine pad settings (saved on the machine profile)."""
        return {
            "increments": {a: c.mm() for a, c in self.increments.items()},
            "custom": sorted({round(v, 6) for c in self.increments.values() for v in c.custom}),
            "feed_xy": self.feed_xy.value() * (MM_PER_IN if self.inches else 1.0),
            "feed_z": self.feed_z.value() * (MM_PER_IN if self.inches else 1.0),
            "link_xy": self.link_xy.isChecked(),
            "flip": dict(self.flip),
            "hold": self.hold_mode.isChecked(),
        }

    def apply_state(self, state: dict | None) -> None:
        state = dict(state or {})
        self.blockSignals(True)
        try:
            custom = [float(v) for v in state.get("custom", []) or []]
            for axis, combo in self.increments.items():
                if combo.rotary:
                    continue
                combo.set_custom([v for v in custom])
                value = (state.get("increments") or {}).get(axis)
                if value:
                    combo.set_mm(float(value))
            scale = MM_PER_IN if self.inches else 1.0
            if state.get("feed_xy"):
                self.feed_xy.setValue(float(state["feed_xy"]) / scale)
            if state.get("feed_z"):
                self.feed_z.setValue(float(state["feed_z"]) / scale)
            self.link_xy.setChecked(bool(state.get("link_xy", True)))
            self.hold_mode.setChecked(bool(state.get("hold", False)))
            flip = state.get("flip") or {}
            self.flip_x.setChecked(bool(flip.get("X", False)))
            self.flip_y.setChecked(bool(flip.get("Y", False)))
        finally:
            self.blockSignals(False)
        self._update_readout()

    def set_extra_axes(self, axes: str) -> None:
        """Rows of − / + for axes beyond X, Y, Z (degrees for A / B / C)."""
        axes = [a for a in str(axes or "").upper() if a not in "XYZ"]
        if axes == self.extra_axes:
            return
        while self.extra_box.count():
            item = self.extra_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for key in [k for k in self.buttons if k[0] in "ABCUVW"]:
            self.buttons.pop(key)
            self.arrow_moves.pop(key, None)
        for axis in [a for a in self.increments if a not in "XYZ"]:
            self.increments.pop(axis)
        self.extra_axes = axes
        for row, axis in enumerate(axes):
            minus, plus = QPushButton(f"{axis}−"), QPushButton(f"{axis}+")
            for key, button, sign in ((f"{axis}-", minus, -1), (f"{axis}+", plus, 1)):
                button.setAutoDefault(False)
                button.pressed.connect(lambda k=key: self._pressed(k))
                button.released.connect(lambda k=key: self._released(k))
                button.clicked.connect(lambda _c=False, k=key: self._clicked(k))
                self.buttons[key] = button
                self.arrow_moves[key] = {axis: sign}
            combo = AxisIncrement(axis, inches=self.inches, rotary=axis in ROTARY_LETTERS, parse=self.parse)
            combo.changedMm.connect(lambda _v: self.settingsChanged.emit())
            self.increments[axis] = combo
            self.extra_box.addWidget(minus, row, 0)
            self.extra_box.addWidget(plus, row, 1)
            self.extra_box.addWidget(QLabel(f"{axis} step"), row, 2)
            self.extra_box.addWidget(combo, row, 3)

    # ------------------------------------------------------------ enabled state
    def set_enabled_state(self, allowed: bool, axes: str, reason: str = "", connected: bool = False) -> None:
        axes = str(axes or "").upper()
        for key, button in self.buttons.items():
            ok = allowed and all(a in axes for a in self.arrow_moves[key])
            button.setEnabled(ok)
            button.setToolTip("" if ok or not reason else reason)
        self.zero_button.setEnabled(allowed)
        self.kill_button.setEnabled(bool(connected))  # KILL works whenever a port is open
        self.reason.setText("" if allowed else reason)
        self.reason.setVisible(bool(reason) and not allowed)

    def show_refusal(self, text: str) -> None:
        self.reason.setText(text)
        self.reason.setVisible(bool(text))

    def set_position(self, text: str) -> None:
        self._position = text
        self._update_readout()

    def _update_readout(self) -> None:
        steps = ", ".join(f"{a} {self.increments[a]._shown(self.increments[a].mm())}" for a in "XYZ" if a in self.increments)
        position = getattr(self, "_position", "")
        self.readout.setText((position + "\n" if position else "") + f"Step: {steps}")

    # ------------------------------------------------------------ jogging
    def moves_for(self, key: str) -> dict[str, float]:
        """Signed distances (mm / degrees) for one button, with flips and per-axis increments."""
        out: dict[str, float] = {}
        for axis, sign in self.arrow_moves[key].items():
            if self.flip.get(axis):
                sign = -sign
            out[axis] = sign * self.increment_mm(axis)
        return out

    def _clicked(self, key: str) -> None:
        if self.hold_mode.isChecked():
            return  # handled by press / release
        if self.jog_callback is None:
            return
        moves = self.moves_for(key)
        if not self.jog_callback(moves, self.feed_for(moves)):
            pass  # the host shows the reason (show_refusal)

    def _pressed(self, key: str) -> None:
        if not self.hold_mode.isChecked() or self.hold_callback is None:
            return
        moves = self.arrow_moves[key]
        if len(moves) != 1:
            return  # diagonals step only
        axis, sign = next(iter(moves.items()))
        if self.flip.get(axis):
            sign = -sign
        if self.hold_callback(axis, sign, self.feed_for(moves)):
            self._holding = (axis, sign)

    def _released(self, key: str) -> None:
        if self._holding is not None and self.cancel_callback is not None:
            self.cancel_callback()
        self._holding = None
