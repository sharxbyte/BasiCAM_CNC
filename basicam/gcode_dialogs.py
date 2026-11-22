"""Dialogs for Advanced G-code mode: start disclaimer, resume, backup-first."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

DISCLAIMER = (
    "You are responsible for the machine setup, tooling, fixturing and work offsets, and for any "
    "financial loss, damage or personal injury that results from using this software. This applies "
    "ESPECIALLY to live edits of the code and to starting a program part-way through. Verify the code "
    "independently, keep the physical E-stop within reach, and stay at the machine."
)


class StartDisclaimerDialog(QDialog):
    """Heavy confirmation shown on every start path; OK stays disabled until acknowledged."""

    def __init__(self, title: str, detail: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        heading = QLabel(detail)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        text = QLabel(DISCLAIMER)
        text.setWordWrap(True)
        text.setStyleSheet("font-weight: 700;")
        layout.addWidget(text)
        self.ack = QCheckBox("I have read this and accept full responsibility")
        layout.addWidget(self.ack)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(False)
        self.ack.toggled.connect(ok.setEnabled)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)


class ResumeDialog(QDialog):
    """Resume code here: operation, tool, remaining estimate, disclaimers, home-location question."""

    def __init__(self, line: int, operation: str, tool: str, estimate: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Resume at line {line}")
        layout = QVBoxLayout(self)
        summary = QLabel(
            f"Operation about to resume: {operation or 'unnamed'}\n"
            f"Tool called out for this section: {tool or 'not stated in the code'}\n"
            f"Estimated remaining time: {estimate}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        self.tool_ok = QCheckBox(f"The mounted tool is {tool or 'the correct tool'} and its length offset is set")
        layout.addWidget(self.tool_ok)
        layout.addWidget(QLabel("Home (work zero) location:"))
        self.same_home = QRadioButton("Same as last used")
        self.new_home = QRadioButton("Set a new home location (opens the machine controls)")
        self.same_home.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.same_home)
        group.addButton(self.new_home)
        layout.addWidget(self.same_home)
        layout.addWidget(self.new_home)
        warning = QLabel(DISCLAIMER)
        warning.setWordWrap(True)
        warning.setStyleSheet("font-weight: 700;")
        layout.addWidget(warning)
        self.ack = QCheckBox("I accept full responsibility for resuming this program")
        layout.addWidget(self.ack)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setEnabled(False)
        self.tool_ok.toggled.connect(self._update)
        self.ack.toggled.connect(self._update)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _update(self) -> None:
        self.ok.setEnabled(self.tool_ok.isChecked() and self.ack.isChecked())

    def wants_new_home(self) -> bool:
        return self.new_home.isChecked()


class BackupFirstDialog(QDialog):
    """Before deleting prior operations: save a backup copy first?"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Delete prior operations")
        self.choice = "cancel"
        layout = QVBoxLayout(self)
        text = QLabel(
            "This removes every operation before the selection from the program. The setup sheet is rewritten "
            "to match what remains. Save a backup copy of the current program first?"
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        row = QHBoxLayout()
        for label, key, tip in (
            ("Save backup...", "backup", "Save the current program to a file, then delete."),
            ("Delete without backup", "delete", "Delete the prior operations now."),
            ("Cancel", "cancel", "Keep the program unchanged."),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, k=key: self._choose(k))
            row.addWidget(button)
        layout.addLayout(row)

    def _choose(self, key: str) -> None:
        self.choice = key
        self.accept()
