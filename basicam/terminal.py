"""Shared terminal: log model, history, command validation, and the two Qt views.

`TerminalCore.log(level, text)` is the single entry point. Every view built on a
core shows the same text. Output is written in full to a rotating log file
(5 files of 2 MB under the user's data folder by default).
"""

from __future__ import annotations

import html
import logging
import os
import re
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, Signal

from basicam.grbl_handshake import refused_setting
from basicam.theme import TOKENS
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

LEVELS = ("info", "command", "response", "warning", "error")
SCREEN_LINE_CAP = 5000
HISTORY_CAP = 50
PASTE_MAX_CHARS = 80
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 4  # current file + 4 backups = 5 files

# Colours per level; error uses the reserved error token. The window re-syncs these from the active theme.
LEVEL_COLORS = {
    "info": TOKENS["text"],
    "command": "#7fb7ff",
    "response": "#9fd8a8",
    "warning": TOKENS["warning"],
    "error": TOKENS["error"],
}
LEVEL_PREFIX = {"info": "", "command": "> ", "response": "< ", "warning": "! ", "error": "ERROR: "}

# GRBL 1.1 realtime commands, by typed alias.
REALTIME_COMMANDS: dict[str, bytes] = {
    "!": b"!",
    "hold": b"!",
    "~": b"~",
    "resume": b"~",
    "?": b"?",
    "reset": b"\x18",
    "ctrl-x": b"\x18",
    "feed100": b"\x90",
    "feed+10": b"\x91",
    "feed-10": b"\x92",
    "feed+1": b"\x93",
    "feed-1": b"\x94",
    "rapid100": b"\x95",
    "rapid50": b"\x96",
    "rapid25": b"\x97",
    "spindle100": b"\x99",
    "spindle+10": b"\x9a",
    "spindle-10": b"\x9b",
    "spindle+1": b"\x9c",
    "spindle-1": b"\x9d",
}

_MOTION_RE = re.compile(r"(?<![A-Z0-9])(?:G0?[0-3]|G28|G30|G38\.\d|M0?[34])(?![\d.])")


def default_log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "BasiCAM_CNC" / "logs"


def is_motion_command(command: str) -> bool:
    upper = command.upper().strip()
    return bool(_MOTION_RE.search(upper) or upper.startswith("$J=") or upper in {"$H"} or upper.startswith("$H"))


@dataclass
class ValidationResult:
    action: str  # "help", "status", "send", "realtime", "refuse", "empty"
    command: str = ""
    message: str = ""
    moves: bool = False
    realtime: bytes = b""


class TerminalCore(QObject):
    """Log model + history shared by every terminal view."""

    entry_added = Signal(str, str)

    def __init__(
        self,
        settings=None,
        log_dir: str | Path | None = None,
        max_bytes: int = LOG_MAX_BYTES,
        backup_count: int = LOG_BACKUPS,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.entries: list[tuple[str, str, float]] = []
        self.history: list[str] = []
        self.last_result = "No operation yet."
        self.log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        self.log_path = self.log_dir / "terminal.log"
        self._logger = logging.getLogger(f"basicam.terminal.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler: RotatingFileHandler | None = None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._handler = RotatingFileHandler(
                self.log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            self._handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._logger.addHandler(self._handler)
        except OSError:
            self._handler = None

    def log(self, level: str, text: str) -> None:
        level = level if level in LEVELS else "info"
        for line in str(text).splitlines() or [""]:
            self.entries.append((level, line, time.time()))
            if len(self.entries) > SCREEN_LINE_CAP:
                del self.entries[: len(self.entries) - SCREEN_LINE_CAP]
            if self._handler is not None:
                self._logger.info("[%s] %s", level, line)
            self.entry_added.emit(level, line)
        if level == "error":
            self.last_result = f"Error: {text}"

    def record_success(self, text: str) -> None:
        self.last_result = text

    @staticmethod
    def format_line(level: str, text: str) -> str:
        return f"{LEVEL_PREFIX.get(level, '')}{text}"

    def plain_text(self) -> str:
        return "\n".join(self.format_line(level, text) for level, text, _ in self.entries)

    def add_history(self, command: str) -> None:
        command = command.strip()
        if not command:
            return
        if self.history and self.history[-1] == command:
            return
        self.history.append(command)
        if len(self.history) > HISTORY_CAP:
            del self.history[: len(self.history) - HISTORY_CAP]

    def close(self) -> None:
        if self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None


class CommandValidator:
    """Decides what a typed or recalled line may do. Never sends anything itself."""

    HELP_TEXT = (
        "; Commands: help, status, GRBL commands ($$, $G, $I, $X, $H, $J=...), single G-code lines.",
        "; Realtime: hold (!), resume (~), reset (ctrl-x), ?, feed+10/feed-10/feed100,",
        ";   spindle+10/spindle-10/spindle100, rapid100/rapid50/rapid25.",
        "; Rules: one line per send; pasting is limited to one line of 80 printable characters.",
        "; Load longer programs in the G-code editor.",
        "; While a program runs only status, help and realtime commands are accepted.",
        "; Motion needs a connected machine with verified position, a confirmation, and must pass",
        ";   the soft-limit / hard-limit / keep-out check.",
        "; Up/Down arrows cycle the last 50 commands; recalled commands are checked again.",
    )

    def __init__(self, motion_check: Callable[[str], tuple[bool, str]] | None = None) -> None:
        self.motion_check = motion_check

    @staticmethod
    def check_paste(text: str) -> tuple[bool, str]:
        text = text.rstrip("\r\n")
        if "\n" in text or "\r" in text:
            return False, "Paste refused: only a single line can be pasted. Load programs in the G-code editor."
        if len(text) > PASTE_MAX_CHARS:
            return False, f"Paste refused: more than {PASTE_MAX_CHARS} characters. Load programs in the G-code editor."
        if any(not ch.isprintable() for ch in text):
            return False, "Paste refused: control or non-printable characters. Load programs in the G-code editor."
        return True, ""

    def validate(
        self,
        command: str,
        *,
        connected: bool,
        running: bool,
        position_verified: bool,
    ) -> ValidationResult:
        text = command.strip()
        if not text:
            return ValidationResult("empty")
        lowered = text.casefold()
        if lowered == "help":
            return ValidationResult("help", text)
        if lowered == "status":
            return ValidationResult("status", text)
        if any(not ch.isprintable() for ch in text) or len(text) > 256:
            return ValidationResult("refuse", text, "Refused: control characters or an over-long line.")
        if lowered in REALTIME_COMMANDS:
            if not connected:
                return ValidationResult("refuse", text, "No machine connected; nothing was sent.")
            return ValidationResult("realtime", text, realtime=REALTIME_COMMANDS[lowered])
        if running:
            return ValidationResult(
                "refuse", text, "A program is running: only status, help and realtime commands are accepted."
            )
        if not connected:
            return ValidationResult("refuse", text, "No machine connected; nothing was sent.")
        refusal = refused_setting(text)  # $21=0 / $20=0 never; $5 only in the limit-switch test (v3 Block 7)
        if refusal:
            return ValidationResult("refuse", text, refusal)
        moves = is_motion_command(text)
        if moves:
            if not position_verified and not text.upper().startswith("$H"):
                return ValidationResult(
                    "refuse", text, "Home the machine or verify its position before sending motion commands.", True
                )
            if self.motion_check is not None:
                ok, reason = self.motion_check(text)
                if not ok:
                    return ValidationResult("refuse", text, reason or "Refused by the motion guard.", True)
        return ValidationResult("send", text, moves=moves)


class HistoryLineEdit(QLineEdit):
    """Input line with Up/Down history and a guarded paste."""

    pasteRefused = Signal(str)

    def __init__(self, core: TerminalCore, parent=None) -> None:
        super().__init__(parent)
        self.core = core
        self._index: int | None = None
        self.setAcceptDrops(False)
        self.setDragEnabled(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    def guarded_paste(self, text: str) -> bool:
        ok, message = CommandValidator.check_paste(text)
        if not ok:
            self.pasteRefused.emit(message)
            return False
        self.insert(text.rstrip("\r\n"))
        return True

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.matches(QKeySequence.StandardKey.Paste):
            self.guarded_paste(QGuiApplication.clipboard().text())
            return
        history = self.core.history
        if event.key() == Qt.Key.Key_Up and history:
            self._index = len(history) - 1 if self._index is None else max(0, self._index - 1)
            self.setText(history[self._index])
            return
        if event.key() == Qt.Key.Key_Down and history:
            if self._index is None:
                return
            if self._index >= len(history) - 1:
                self._index = None
                self.clear()
            else:
                self._index += 1
                self.setText(history[self._index])
            return
        super().keyPressEvent(event)

    def reset_history_cursor(self) -> None:
        self._index = None


class TerminalView(QWidget):
    """Scrolling output above one input line; several views can share one core."""

    submitted = Signal(str)

    def __init__(self, core: TerminalCore, parent=None, placeholder: str = "help, status, GRBL command or one G-code line") -> None:
        super().__init__(parent)
        self.core = core
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.document().setMaximumBlockCount(SCREEN_LINE_CAP)
        layout.addWidget(self.output, 1)
        row = QHBoxLayout()
        self.input_line = HistoryLineEdit(core)
        self.input_line.setPlaceholderText(placeholder)
        self.input_line.setToolTip("Type a command and press Enter; Up/Down recall earlier commands.")
        self.input_line.returnPressed.connect(self._submit)
        self.input_line.pasteRefused.connect(lambda message: core.log("warning", message))
        row.addWidget(self.input_line, 1)
        self.send_button = QPushButton("Send")
        self.send_button.setToolTip("Check the command and send it to the controller if it is allowed.")
        self.send_button.clicked.connect(self._submit)
        row.addWidget(self.send_button)
        layout.addLayout(row)
        for level, text, _ in core.entries:
            self._append(level, text)
        core.entry_added.connect(self._append)

    def _append(self, level: str, text: str) -> None:
        color = LEVEL_COLORS.get(level, LEVEL_COLORS["info"])
        line = html.escape(TerminalCore.format_line(level, text))
        self.output.appendHtml(f'<span style="color:{color}; white-space:pre">{line}</span>')

    def plain_text(self) -> str:
        return self.output.toPlainText()

    def _submit(self) -> None:
        text = self.input_line.text().strip()
        self.input_line.reset_history_cursor()
        if text:
            self.submitted.emit(text)

    def clear_input(self) -> None:
        self.input_line.clear()


class TerminalDock(QDockWidget):
    """Bottom terminal dock. Collapses to its title bar during console takeover."""

    def __init__(self, core: TerminalCore, parent=None) -> None:
        super().__init__("Terminal", parent)
        self.setObjectName("bottomTerminalDock")
        self.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.view = TerminalView(core, self)
        self.setWidget(self.view)
        self.manual_collapsed = False
        self.collapsed = False

    def set_collapsed(self, collapsed: bool, manual: bool = False) -> None:
        if manual:
            self.manual_collapsed = collapsed
        self.collapsed = collapsed
        self.view.setVisible(not collapsed)
        if collapsed:
            self.setMaximumHeight(max(24, self.fontMetrics().height() + 12))
        else:
            self.setMaximumHeight(16777215)

    def takeover(self, machine_dock_visible: bool) -> None:
        """Machine console takes input while its dock is open; restore unless collapsed by hand."""
        if machine_dock_visible:
            self.set_collapsed(True)
        elif not self.manual_collapsed:
            self.set_collapsed(False)
