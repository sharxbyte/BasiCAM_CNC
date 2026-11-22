"""Dialogs for the sketch definition system: cascade delete and bad-sketch delete."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout


class _ChoiceDialog(QDialog):
    def __init__(self, title: str, text: str, items: list[str], choices: list[tuple[str, str, str]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.choice = choices[-1][1] if choices else ""
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        if items:
            listing = QListWidget()
            listing.addItems(items)
            listing.setMaximumHeight(140)
            layout.addWidget(listing)
        row = QHBoxLayout()
        row.addStretch(1)
        self.buttons: dict[str, QPushButton] = {}
        for text_, key, tip in choices:
            button = QPushButton(text_)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, k=key: self._choose(k))
            row.addWidget(button)
            self.buttons[key] = button
        layout.addLayout(row)

    def _choose(self, key: str) -> None:
        self.choice = key
        self.accept()


class CascadeDeleteDialog(_ChoiceDialog):
    """Shown before deleting something other entities depend on."""

    def __init__(self, target: str, dependents: list[str], parent=None) -> None:
        super().__init__(
            "Delete breaks other sketches",
            f"'{target}' is used by the entities below. Deleting it breaks them (shown red). "
            "If you continue you must immediately fix the broken sketch or delete the broken entities.",
            dependents,
            [
                ("Continue", "continue", "Delete it and mark the dependent entities as broken."),
                ("Cancel", "cancel", "Keep everything as it is."),
            ],
            parent,
        )


class CascadeStepDialog(_ChoiceDialog):
    """After each cascade step: Undo step / Undo all / Continue (delete the broken entities)."""

    def __init__(self, broken: list[str], step: int, parent=None) -> None:
        super().__init__(
            f"Broken entities · step {step}",
            "These entities lost a reference and are broken. Undo this step, undo the whole chain, "
            "delete the broken entities (which may break more), or close and fix them by hand.",
            broken,
            [
                ("Undo step", "undo_step", "Undo only the last deletion step."),
                ("Undo all", "undo_all", "Undo every step back to a state with no errors."),
                ("Continue", "continue", "Delete the broken entities as the next step."),
                ("Fix by hand", "fix", "Close; the broken entities stay red until fixed or deleted."),
            ],
            parent,
        )


class DeleteBadSketchDialog(_ChoiceDialog):
    """A new sketch that never had a good state: confirm deleting it."""

    def __init__(self, sketch_name: str, parent=None) -> None:
        super().__init__(
            "Sketch could not be solved",
            f"'{sketch_name}' has never been in a good state, so there is nothing to revert to. Delete it?",
            [],
            [
                ("Delete sketch", "delete", "Remove the broken sketch and its entities."),
                ("Keep editing", "keep", "Return and fix the sketch."),
            ],
            parent,
        )
