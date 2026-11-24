"""Headless UI wiring audit for BasiCAM CNC (not bundled).

Run from the project root:
    python tools/ui_audit.py            # static checks
    python tools/ui_audit.py --fire     # also fire every button/action with dialogs stubbed out

Reports controls with no connected signal, empty tool tips, duplicate visible text
in one container, handlers that raise when fired, labels over 40 characters, and
tool tips identical to the label.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import SIGNAL, QSettings  # noqa: E402
from PySide6.QtGui import QAction, QColor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QWidget,
)

MAX_LABEL = 40

SIGNALS = {
    QPushButton: ("clicked()", "clicked(bool)", "pressed()", "released()", "toggled(bool)"),
    QCheckBox: ("clicked()", "clicked(bool)", "toggled(bool)", "stateChanged(int)", "checkStateChanged(Qt::CheckState)"),
    QComboBox: ("currentIndexChanged(int)", "currentTextChanged(QString)", "activated(int)", "textActivated(QString)"),
    QAbstractSpinBox: ("valueChanged(int)", "valueChanged(double)", "editingFinished()", "textChanged(QString)"),
    QLineEdit: ("returnPressed()", "textChanged(QString)", "editingFinished()", "textEdited(QString)"),
    QTreeWidget: (
        "itemClicked(QTreeWidgetItem*,int)",
        "itemDoubleClicked(QTreeWidgetItem*,int)",
        "currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)",
        "itemSelectionChanged()",
        "itemChanged(QTreeWidgetItem*,int)",
        "customContextMenuRequested(QPoint)",
        "itemEntered(QTreeWidgetItem*,int)",
    ),
    QAction: ("triggered()", "triggered(bool)", "toggled(bool)"),
}
INPUT_TYPES = (QComboBox, QAbstractSpinBox, QLineEdit, QCheckBox)


def receivers(obj) -> int:
    total = 0
    for cls, sigs in SIGNALS.items():
        if isinstance(obj, cls):
            for sig in sigs:
                try:
                    total += obj.receivers(SIGNAL(sig))
                except (RuntimeError, TypeError):
                    pass
    return total


def label_of(obj) -> str:
    if isinstance(obj, (QPushButton, QCheckBox, QAction)):
        return obj.text().replace("&", "").strip()
    return ""


def describe(obj) -> str:
    name = obj.objectName() or ""
    text = label_of(obj)
    if isinstance(obj, QLineEdit):
        text = text or obj.placeholderText()
    parent = obj.parent()
    pname = type(parent).__name__ if parent is not None else "-"
    return f"{type(obj).__name__}[{name or text or '?'}] in {pname}"


def collect(window) -> list:
    # only controls of this window, not of dialogs it opened earlier (they stay as children)
    widgets = [w for w in window.findChildren(QWidget) if isinstance(w, tuple(SIGNALS)) and w.window() is window]
    actions = [a for a in window.findChildren(QAction) if not a.isSeparator() and a.text()]
    return widgets + actions


def attribute_ids(window) -> set[int]:
    ids = set()
    for value in vars(window).values():
        if isinstance(value, (QWidget, QAction)):
            ids.add(id(value))
        elif isinstance(value, dict):
            ids.update(id(v) for v in value.values() if isinstance(v, (QWidget, QAction)))
        elif isinstance(value, (list, tuple)):
            ids.update(id(v) for v in value if isinstance(v, (QWidget, QAction)))
    return ids


notes: list[tuple[str, str]] = []

# Reviewed findings that are intentional.
JUSTIFIED = {
    # Display and Import unit toggles both show the current unit; each has its own label beside it.
    ("duplicate-text", "QPushButton[mm] in QFrame"),
    ("duplicate-text", "QPushButton[in] in QFrame"),
}


def static_checks(window) -> list[tuple[str, str]]:
    report: list[tuple[str, str]] = []
    stored = attribute_ids(window)
    seen_text: dict[tuple[int, str], int] = {}
    for obj in collect(window):
        if isinstance(obj, QAction) and obj.menu() is not None:
            continue
        text = label_of(obj)
        wired = receivers(obj) > 0
        checkable = isinstance(obj, QPushButton) and obj.isCheckable()
        if (isinstance(obj, INPUT_TYPES) or checkable) and id(obj) in stored:
            wired = True  # state read on demand by the window
        if isinstance(obj, QLineEdit) and isinstance(obj.parent(), (QComboBox, QAbstractSpinBox)):
            continue  # internal editor of a combo/spin box
        if not wired:
            report.append(("unwired", describe(obj)))
        tip = obj.toolTip().strip()
        # Tool tips are only required where a control needs explaining; a missing one is a note.
        if not tip and not isinstance(obj, QAction):
            notes.append(("no-tooltip", describe(obj)))
        if text and tip and tip.casefold() == text.casefold() and not isinstance(obj, QAction):
            report.append(("tooltip-equals-label", describe(obj)))
        if len(text) > MAX_LABEL:
            report.append(("long-label", f"{describe(obj)}: {len(text)} chars"))
        if text and isinstance(obj, QWidget) and obj.parent() is not None:
            key = (id(obj.parent()), text.casefold())
            seen_text[key] = seen_text.get(key, 0) + 1
            if seen_text[key] == 2:
                report.append(("duplicate-text", describe(obj)))
    return [item for item in report if item not in JUSTIFIED]


def fire_checks(window) -> list[tuple[str, str]]:
    report: list[tuple[str, str]] = []
    errors: list[str] = []

    def hook(exc_type, exc, tb):
        errors.append("".join(traceback.format_exception_only(exc_type, exc)).strip())

    cancel = QMessageBox.StandardButton.Cancel
    patches = [
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No),
        mock.patch.object(QMessageBox, "warning", return_value=cancel),
        mock.patch.object(QMessageBox, "critical", return_value=cancel),
        mock.patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok),
        mock.patch.object(QMessageBox, "exec", return_value=int(cancel)),
        mock.patch.object(QDialog, "exec", return_value=0),
        mock.patch.object(QMenu, "exec", return_value=None),
        mock.patch.object(QFileDialog, "getOpenFileName", return_value=("", "")),
        mock.patch("PySide6.QtWidgets.QColorDialog.getColor", return_value=QColor()),
        mock.patch.object(QFileDialog, "getOpenFileNames", return_value=([], "")),
        mock.patch.object(QFileDialog, "getSaveFileName", return_value=("", "")),
        mock.patch.object(QFileDialog, "getExistingDirectory", return_value=""),
        mock.patch.object(QInputDialog, "getText", return_value=("", False)),
        mock.patch.object(QInputDialog, "getDouble", return_value=(0.0, False)),
        mock.patch.object(QInputDialog, "getInt", return_value=(0, False)),
        mock.patch.object(QInputDialog, "getItem", return_value=("", False)),
        mock.patch("PySide6.QtGui.QDesktopServices.openUrl", return_value=True),
    ]
    skip = {"quit", "exit", "close"}
    for p in patches:
        p.start()
    old_hook = sys.excepthook
    sys.excepthook = hook
    try:
        targets = [o for o in collect(window) if isinstance(o, (QPushButton, QAction))]
        for obj in targets:
            text = label_of(obj).casefold()
            if any(word in text for word in skip) or not obj.isEnabled():
                continue
            if isinstance(obj, QAction) and obj.menu() is not None:
                continue
            errors.clear()
            try:
                if isinstance(obj, QAction):
                    obj.trigger()
                    if obj.isCheckable():
                        obj.trigger()
                else:
                    obj.click()
                    if obj.isCheckable():
                        obj.click()
                QApplication.processEvents()
            except Exception as error:  # noqa: BLE001
                errors.append(f"{type(error).__name__}: {error}")
            for message in errors:
                report.append(("handler-raises", f"{describe(obj)}: {message}"))
    finally:
        sys.excepthook = old_hook
        for p in reversed(patches):
            p.stop()
    return report


# v3 Block 5 / 11: the visible commands change with the machine, so each type is audited in turn.
MACHINE_TYPES = (
    ("mill", dict(name="Audit mill", process="mill", layout="gantry3")),
    ("laser", dict(name="Audit laser", process="laser", layout="gantry3")),
    ("lathe", dict(name="Audit lathe", process="lathe", layout="lathe2", controller="grblHAL")),
    ("rotary laser", dict(name="Audit rotary laser", process="rotary_laser", layout="gantry3_rotary", controller="grblHAL")),
)


def load_machine(window, **kw) -> None:
    from basicam.machine_profiles import MachineProfile

    machine = MachineProfile(**kw)
    window.machine_store.save(machine)
    with mock.patch.object(window, "_maybe_ask_job_type"):
        window._set_current_machine(machine.machine_id, load_state=False)
        QApplication.processEvents()


def extra_windows(window) -> list:
    """Windows that are not children until opened: ports, preflight (with its pad), limit-switch test."""
    from basicam.limit_test import LimitSwitchTestDialog
    from basicam.machine_profiles import PreflightDialog
    from basicam.ports_dialog import PortsDialog

    ports = PortsDialog(window, details=lambda: [], test=lambda d, b: "", connect=lambda d, b: None,
                        use_for_machine=lambda d, b: "")
    ports.timer.stop()
    pad = window.make_jog_pad()
    actions = {"home": lambda: None, "unlock": lambda: None, "hold": lambda: None, "kill": lambda: None}
    preflight = PreflightDialog(None, "XYZ", window, pad=pad, actions=actions)
    limits = LimitSwitchTestDialog(window, axes="XYZ", settings={5: 0}, firmware="grbl", send_setting=lambda line: None)
    return [ports, preflight, limits]


def run(fire: bool = False) -> list[tuple[str, str]]:
    app = QApplication.instance() or QApplication([])
    import pyvista as pv

    pv.OFF_SCREEN = True
    from basicam import app as app_module

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = str(Path(tmp) / "settings.ini")

        def factory(*_a, **_k):
            return QSettings(path, QSettings.Format.IniFormat)

        # The audit never touches real hardware: no serial ports are listed and none can be opened.
        def no_port(*_a, **_k):
            raise OSError("ui_audit: serial ports are disabled")

        with mock.patch.object(app_module, "QSettings", side_effect=factory), mock.patch.dict(
            os.environ, {"LOCALAPPDATA": tmp}
        ), mock.patch("basicam.machine.list_ports.comports", return_value=[]),                 mock.patch("basicam.machine.serial.Serial", side_effect=no_port):
            report: list[tuple[str, str]] = []
            # each pass gets a fresh window: firing changes the workspace (adds setups, deletes operations)
            for label, kw in (("", None),) + MACHINE_TYPES:
                window = app_module.BasiCAMWindow()
                try:
                    if kw is not None:
                        load_machine(window, **kw)
                    found = static_checks(window)
                    if fire:
                        found += fire_checks(window)
                    if kw is None:
                        for dialog in extra_windows(window):
                            found += [(kind, f"[{dialog.windowTitle()}] {detail}") for kind, detail in static_checks(dialog)]
                            if fire:
                                found += [(kind, f"[{dialog.windowTitle()}] {detail}") for kind, detail in fire_checks(dialog)]
                            dialog.close()
                    report += [(kind, f"[{label}] {detail}" if label else detail) for kind, detail in found]
                finally:
                    window.terminal_core.close()
                    window.plotter.close()
                    window.close()
                    app.processEvents()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fire", action="store_true", help="fire every button and action with dialogs stubbed")
    parser.add_argument("--notes", action="store_true", help="also list controls without a tool tip")
    args = parser.parse_args()
    report = run(fire=args.fire)
    counts: dict[str, int] = {}
    for kind, detail in report:
        counts[kind] = counts.get(kind, 0) + 1
        print(f"{kind:<22} {detail}")
    print("-" * 60)
    print("clean" if not report else ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if notes:
        print(f"note: {len(notes)} controls have no tool tip (fine if self-explanatory; --notes lists them)")
        if "--notes" in sys.argv:
            for kind, detail in notes:
                print(f"{kind:<22} {detail}")
    return 1 if report else 0


if __name__ == "__main__":
    raise SystemExit(main())
