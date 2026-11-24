"""Guided limit-switch test (roadmap v3 Block 7).

The owner presses and releases each limit switch by hand; BasiCAM shows each pin toggle from the
controller's ``Pn:`` status field. A switch passes once it has been seen both released and pressed.
When every limit pin reads active at rest, the test may offer the ``$5`` change that inverts the
limit pins (decision 1 of the roadmap); after a change the test starts again so every switch is
proven to work. BasiCAM never turns the hard limits off.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from basicam.grbl_handshake import LIMIT_LETTERS, all_limits_active, inverted_limit_setting
from basicam.theme import color as theme_color
from basicam.window_fit import make_working_window


class LimitSwitchTest:
    """Pin bookkeeping, independent of the window."""

    def __init__(self, axes: str = "XYZ") -> None:
        self.axes = [a for a in str(axes or "XYZ").upper() if a in LIMIT_LETTERS]
        self.reset()

    def reset(self) -> None:
        self.seen_off = {a: False for a in self.axes}
        self.seen_on = {a: False for a in self.axes}
        self.current: dict[str, bool] = {a: False for a in self.axes}
        self.samples = 0
        self.first_pins: str | None = None

    def update(self, pins: str) -> None:
        pins = str(pins or "").upper()
        if self.first_pins is None:
            self.first_pins = pins
        self.samples += 1
        for axis in self.axes:
            on = axis in pins
            self.current[axis] = on
            if on:
                self.seen_on[axis] = True
            else:
                self.seen_off[axis] = True

    def passed(self, axis: str) -> bool:
        return self.seen_on.get(axis, False) and self.seen_off.get(axis, False)

    def complete(self) -> bool:
        return bool(self.axes) and all(self.passed(a) for a in self.axes)

    def all_active_at_rest(self) -> bool:
        return self.first_pins is not None and all_limits_active(self.first_pins, "".join(self.axes))


class LimitSwitchTestDialog(QDialog):
    """``send_setting(line)`` writes a setting to the controller (the main window's guarded path)."""

    def __init__(self, parent=None, axes: str = "XYZ", settings: dict | None = None, firmware: str = "grbl",
                 send_setting=None, allow_invert: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Limit-switch test")
        self.setObjectName("limitSwitchTest")
        make_working_window(self)
        self.test = LimitSwitchTest(axes)
        self.settings = dict(settings or {})
        self.firmware = firmware
        self.send_setting = send_setting
        self.allow_invert = allow_invert
        self.sent: list[str] = []
        layout = QVBoxLayout(self)
        intro = QLabel("Press and release each limit switch by hand. Each one must turn on, then off again. "
                       "The machine does not move during this test.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        self.state_labels: dict[str, QLabel] = {}
        self.pass_labels: dict[str, QLabel] = {}
        for row, axis in enumerate(self.test.axes):
            grid.addWidget(QLabel(f"{axis} limit"), row, 0)
            self.state_labels[axis] = QLabel("waiting for the controller")
            self.pass_labels[axis] = QLabel("")
            grid.addWidget(self.state_labels[axis], row, 1)
            grid.addWidget(self.pass_labels[axis], row, 2)
        layout.addLayout(grid)
        self.advice = QLabel("")
        self.advice.setWordWrap(True)
        layout.addWidget(self.advice)
        row = QHBoxLayout()
        self.invert_button = QPushButton("Invert limit pins ($5)")
        self.invert_button.setToolTip("Only when every limit reads on with no switch pressed.")
        self.invert_button.clicked.connect(self._invert)
        self.invert_button.setVisible(False)
        row.addWidget(self.invert_button)
        restart = QPushButton("Start again")
        restart.clicked.connect(self.restart)
        row.addWidget(restart)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        for button in (self.invert_button, restart, close):
            button.setAutoDefault(False)
        layout.addLayout(row)

    def restart(self) -> None:
        self.test.reset()
        self.refresh()

    def update_pins(self, pins: str) -> None:
        self.test.update(pins)
        self.refresh()

    def refresh(self) -> None:
        for axis in self.test.axes:
            if self.test.samples == 0:
                self.state_labels[axis].setText("waiting for the controller")
                self.pass_labels[axis].setText("")
                continue
            on = self.test.current[axis]
            self.state_labels[axis].setText("ON (pressed)" if on else "off")
            self.state_labels[axis].setStyleSheet(f"color: {theme_color('warning') if on else theme_color('muted')};")
            ok = self.test.passed(axis)
            self.pass_labels[axis].setText("works" if ok else "")
            self.pass_labels[axis].setStyleSheet(f"color: {theme_color('ok')};" if ok else "")
        line = inverted_limit_setting(self.settings, self.firmware, "".join(self.test.axes))
        stuck = self.test.all_active_at_rest() and not any(self.test.seen_off.values())
        self.invert_button.setVisible(bool(stuck and line and self.allow_invert))
        if stuck and self.firmware == "fluidnc":
            self.advice.setText("Every limit reads on at rest. FluidNC sets pin inversion in config.yaml "
                                "(add or remove ':low' on each limit pin), then restart the controller.")
        elif stuck:
            self.advice.setText("Every limit reads on with no switch pressed: the pins are probably inverted "
                                "(normally-closed switches). Invert them, then test every switch again.")
        elif self.test.complete():
            self.advice.setText("Every limit switch works.")
        else:
            self.advice.setText("")

    def _invert(self) -> None:
        line = inverted_limit_setting(self.settings, self.firmware, "".join(self.test.axes))
        if not line or self.send_setting is None:
            return
        answer = QMessageBox.question(
            self, "Invert limit pins",
            f"Send {line} to the controller? Afterwards press and release every limit switch again; "
            "a switch that does not respond means the change was wrong.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.send_setting(line)
        self.sent.append(line)
        key, _sep, value = line.partition("=")
        self.settings[int(key[1:])] = float(value)
        self.restart()  # prove each switch again after the change
