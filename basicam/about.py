"""Version, license and terms of use shown inside the program."""
from __future__ import annotations

from pathlib import Path

APP_NAME = "BasiCAM CNC"
APP_VERSION = "1.0.0"
LICENSE_NAME = "GNU General Public License v3.0 or later (GPL-3.0-or-later)"
REPO_URL = "https://github.com/sharxbyte/BasiCAM_CNC"

SUMMARY = (
    f"{APP_NAME} {APP_VERSION} is free software under the {LICENSE_NAME}. "
    "You may use it for hobby and commercial work, copy it, share it and change it. "
    "It comes with no warranty: you use it at your own risk and are responsible for checking every "
    "toolpath before running it on a machine."
)


def resource_root() -> Path:
    """Folder holding LICENSE / TERMS_OF_USE.md: the build folder, or the source checkout."""
    return Path(__file__).resolve().parent.parent


def read_document(name: str) -> str:
    path = resource_root() / name
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError:
        return f"{name} was not found next to the program. See {REPO_URL}."


def show_terms_dialog(parent=None):
    """Terms of use (plain English), the full license and third-party notices, in tabs."""
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QTabWidget, QTextBrowser, QVBoxLayout

    dialog = QDialog(parent)
    dialog.setWindowTitle("Terms of Use and License")
    dialog.setObjectName("termsDialog")
    layout = QVBoxLayout(dialog)
    summary = QLabel(SUMMARY)
    summary.setWordWrap(True)
    layout.addWidget(summary)
    tabs = QTabWidget()
    for title, name in (("Terms of use", "TERMS_OF_USE.md"), ("Third-party software", "THIRD_PARTY_NOTICES.md")):
        view = QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setMarkdown(read_document(name))
        tabs.addTab(view, title)
    license_view = QPlainTextEdit(read_document("LICENSE"))
    license_view.setReadOnly(True)
    tabs.insertTab(1, license_view, "License (GPL-3.0)")
    tabs.setCurrentIndex(0)
    layout.addWidget(tabs, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(760, 620)
    try:
        from basicam.window_fit import make_working_window

        make_working_window(dialog)
    except Exception:
        pass
    dialog.show()
    return dialog
