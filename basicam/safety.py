"""Safety gate registry, Settings > Advanced dialog and the first-run dialog.

`SafetyGates.allows(key)` is the only way code asks whether a rule may be broken.
Every gate defaults OFF. Hard limits, keep-out zones, collision alarms and
generation from under-defined sketches are not gates: `allows` is always False
for them, whatever is stored in the settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


@dataclass(frozen=True)
class SafetyRule:
    key: str
    title: str
    warning: str


SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        "speed_multiplier_over_limit",
        "Allow speed multiplier above 200%",
        "Scales feed words beyond GRBL's own 200% override, up to 10x, clamped to the machine's "
        "maximum rate. Faster than tested feeds can break tools, ruin the part or injure the operator.",
    ),
    SafetyRule(
        "generate_oversize_part",
        "Allow G-code for a part larger than the workspace",
        "Generates code for a part that exceeds the machine travel in all three dimensions. "
        "Moves may run into the frame or limit switches.",
    ),
    SafetyRule(
        "release_soft_limits",
        "Allow releasing soft limits or margins under 5 mm",
        "Lets an axis soft limit be switched off or set closer than 5 mm to the hard limit. "
        "The machine can then reach its mechanical end stops.",
    ),
    SafetyRule(
        "advanced_gcode_mode",
        "Enable Advanced G-code mode",
        "Enables the Advanced G-code editor, resume-from-line and live edits. Edited code is "
        "your responsibility; a wrong edit can crash the machine.",
    ),
)

RULES_BY_KEY: dict[str, SafetyRule] = {rule.key: rule for rule in SAFETY_RULES}

# Not gates: nothing may switch these off.
NEVER_OVERRIDABLE: frozenset[str] = frozenset(
    {"hard_limits", "keep_out_zones", "collision_alarms", "generate_from_underdefined_sketch"}
)

BEGINNER_KEY = "ui/beginner_mode"
FIRST_RUN_DONT_ASK_KEY = "first_run/dont_ask_again"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class SafetyGates:
    """Reads and writes gate state in QSettings under `safety/<key>`."""

    def __init__(self, settings) -> None:
        self.settings = settings

    # --- beginner mode -------------------------------------------------
    @property
    def beginner_mode(self) -> bool:
        return _as_bool(self.settings.value(BEGINNER_KEY, False))

    def set_beginner_mode(self, enabled: bool) -> None:
        self.settings.setValue(BEGINNER_KEY, bool(enabled))
        if enabled:
            self.reset_all()

    # --- gates -----------------------------------------------------------
    def allows(self, key: str) -> bool:
        """True only for a known gate that the user switched on (never in Beginner mode)."""
        if key in NEVER_OVERRIDABLE or key not in RULES_BY_KEY:
            return False
        if self.beginner_mode:
            return False
        return _as_bool(self.settings.value(f"safety/{key}", False))

    def set_allowed(self, key: str, enabled: bool, acknowledged: bool = False) -> bool:
        """Switch a gate. Turning one on needs the warning acknowledged; returns the new state."""
        if key in NEVER_OVERRIDABLE or key not in RULES_BY_KEY:
            return False
        if enabled and (not acknowledged or self.beginner_mode):
            return self.allows(key)
        self.settings.setValue(f"safety/{key}", bool(enabled))
        return self.allows(key)

    def reset_all(self) -> None:
        for rule in SAFETY_RULES:
            self.settings.setValue(f"safety/{rule.key}", False)

    def states(self) -> dict[str, bool]:
        return {rule.key: self.allows(rule.key) for rule in SAFETY_RULES}


def should_show_first_run(settings) -> bool:
    return not _as_bool(settings.value(FIRST_RUN_DONT_ASK_KEY, False))


def apply_first_run_answer(settings, new_user: bool, dont_ask_again: bool) -> None:
    SafetyGates(settings).set_beginner_mode(bool(new_user))
    if dont_ask_again:
        settings.setValue(FIRST_RUN_DONT_ASK_KEY, True)


class FirstRunDialog(QDialog):
    """'Are you a new user?' Yes sets Beginner mode; No leaves expert defaults (gates still off)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to BasiCAM CNC")
        self.new_user = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Are you a new user?"))
        info = QLabel(
            "Yes turns on Beginner mode: every warning and confirmation is shown, all safety "
            "overrides stay off and advanced features are hidden. Machine control stays available. "
            "You can turn Beginner mode off later in Options > Preferences."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self.dont_ask = QCheckBox("Don't ask again")
        self.dont_ask.setToolTip("Skip this question at the next start-up.")
        layout.addWidget(self.dont_ask)
        row = QHBoxLayout()
        self.yes_button = QPushButton("Yes")
        self.yes_button.setToolTip("Start in Beginner mode with all warnings on.")
        self.no_button = QPushButton("No")
        self.no_button.setToolTip("Start in normal mode; safety overrides stay off until you enable them.")
        self.yes_button.clicked.connect(lambda: self._answer(True))
        self.no_button.clicked.connect(lambda: self._answer(False))
        row.addStretch(1)
        row.addWidget(self.yes_button)
        row.addWidget(self.no_button)
        layout.addLayout(row)

    def _answer(self, new_user: bool) -> None:
        self.new_user = new_user
        self.accept()


def run_first_run_dialog(settings, parent=None) -> bool:
    """Show the first-run dialog if needed; returns the Beginner flag now in effect."""
    if should_show_first_run(settings):
        dialog = FirstRunDialog(parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            apply_first_run_answer(settings, dialog.new_user, dialog.dont_ask.isChecked())
    return SafetyGates(settings).beginner_mode


class AdvancedSettingsDialog(QDialog):
    """One checkbox per gate, each with its warning and an acknowledgement box."""

    def __init__(self, gates: SafetyGates, parent=None) -> None:
        super().__init__(parent)
        self.gates = gates
        self.setWindowTitle("Settings > Advanced")
        self.gate_boxes: dict[str, QCheckBox] = {}
        self.ack_boxes: dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        intro = QLabel("These switches allow rules to be broken. Each starts off and needs its warning acknowledged.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        for rule in SAFETY_RULES:
            group = QGroupBox(rule.title)
            box_layout = QVBoxLayout(group)
            warning = QLabel(rule.warning)
            warning.setWordWrap(True)
            box_layout.addWidget(warning)
            ack = QCheckBox("I have read this warning and accept the risk")
            ack.setToolTip("Required before this override can be switched on.")
            gate = QCheckBox("Enable")
            gate.setToolTip(rule.warning)
            on = gates.allows(rule.key)
            ack.setChecked(on)
            gate.setChecked(on)
            gate.setEnabled(on)
            ack.toggled.connect(lambda checked, g=gate: self._ack_changed(g, checked))
            box_layout.addWidget(ack)
            box_layout.addWidget(gate)
            self.gate_boxes[rule.key] = gate
            self.ack_boxes[rule.key] = ack
            layout.addWidget(group)
        not_gates = QLabel(
            "Never overridable: hard limits, keep-out zones, collision alarms, and code from under-defined sketches."
        )
        not_gates.setWordWrap(True)
        layout.addWidget(not_gates)
        self.reset_button = QPushButton("Reset all to safe")
        self.reset_button.setToolTip("Switch every override off and clear the acknowledgements.")
        self.reset_button.clicked.connect(self.reset_to_safe)
        layout.addWidget(self.reset_button)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _ack_changed(gate: QCheckBox, checked: bool) -> None:
        gate.setEnabled(checked)
        if not checked:
            gate.setChecked(False)

    def reset_to_safe(self) -> None:
        for key in self.gate_boxes:
            self.ack_boxes[key].setChecked(False)
            self.gate_boxes[key].setChecked(False)
        self.gates.reset_all()

    def apply(self) -> None:
        for key, gate in self.gate_boxes.items():
            self.gates.set_allowed(key, gate.isChecked(), acknowledged=self.ack_boxes[key].isChecked())

    def _save(self) -> None:
        self.apply()
        self.accept()
