"""Advanced G-code editor pieces: program map, syntax colours, editor widget, resume preamble.

`parse_program` reuses the motion guard's modal interpreter (G90/G91, G20/G21, G0-G3,
G53, G54-G59) so the viewer, the guard and the resume preamble agree on positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from pathlib import Path

from PySide6.QtCore import QRect, QRegularExpression, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTextFormat
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from basicam.cam_features import GrblModalState
from basicam.motion_guard import GuardState, interpret_line
from basicam.window_fit import make_working_window

OPERATION_HEADER = re.compile(r"^\s*;\s*Operation:\s*(?P<name>.+?)(?:\s+·\s+Tool:\s*(?P<tool>.+?))?\s*$")


@dataclass
class LineInfo:
    number: int  # 1-based
    text: str
    modal: GrblModalState
    position: tuple[float, float, float]  # work coordinates (mm) after this line
    operation: str
    tool: str
    spindle_rpm: float
    spindle_on: bool


@dataclass
class OperationSection:
    name: str
    tool: str
    start: int  # 1-based first line (the header)
    end: int  # 1-based last line


@dataclass
class ProgramMap:
    lines: list[LineInfo] = field(default_factory=list)
    sections: list[OperationSection] = field(default_factory=list)

    def info(self, number: int) -> LineInfo | None:
        return self.lines[number - 1] if 1 <= number <= len(self.lines) else None

    def section_at(self, number: int) -> OperationSection | None:
        return next((s for s in self.sections if s.start <= number <= s.end), None)

    def segments(self, start: int = 1, end: int | None = None) -> list:
        """ToolpathSegments between lines (for the cycle-time estimate)."""
        from basicam.cam_dynamics_post import ToolpathSegment

        end = end or len(self.lines)
        out = []
        prev = self.lines[start - 2].position if start > 1 else (0.0, 0.0, 0.0)
        for info in self.lines[start - 1 : end]:
            if info.position != prev:
                kind = "rapid" if info.modal.motion == "G0" else "cut"
                feed = info.modal.feed_rate * (25.4 if info.modal.units == "G20" else 1.0) or 1.0
                out.append(ToolpathSegment(kind, prev, info.position, feed, int(info.spindle_rpm), {"op": info.operation, "tool": info.tool}))
            prev = info.position
        return out


def parse_program(text: str) -> ProgramMap:
    state = GuardState()
    program = ProgramMap()
    op, tool, rpm, spindle_on = "", "", 0.0, False
    current: OperationSection | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        header = OPERATION_HEADER.match(raw)
        if header:
            if current is not None:
                current.end = number - 1
            op = header.group("name").strip()
            tool = (header.group("tool") or "").strip()
            current = OperationSection(op, tool, number, number)
            program.sections.append(current)
        code = re.sub(r"\([^)]*\)", "", raw).split(";", 1)[0].upper()
        s_word = re.search(r"S\s*([-+]?\d*\.?\d+)", code)
        if s_word:
            rpm = float(s_word.group(1))
        if re.search(r"(?<![A-Z0-9])M0?[34](?!\d)", code):
            spindle_on = True
        if re.search(r"(?<![A-Z0-9])M0?5(?!\d)|M30|M0?2(?!\d)", code):
            spindle_on = False
        t_word = re.search(r"(?<![A-Z])T\s*(\d+)", code)
        if t_word and not tool:
            tool = f"T{t_word.group(1)}"
        _points, state = interpret_line(raw, state)
        program.lines.append(LineInfo(number, raw, replace(state.modal), tuple(state.position), op, tool, rpm, spindle_on))
    if current is not None:
        current.end = len(program.lines)
    return program


def build_resume_preamble(program: ProgramMap, line: int, safe_z_mm: float, spindle_rpm: float | None = None, manual_spindle: bool = False) -> list[str]:
    """Safe start before resuming at `line`: units, work offset, spindle, safe Z, XY, plunge."""
    before = program.info(line - 1) if line > 1 else None
    target = program.info(line)
    if target is None:
        return []
    modal = before.modal if before else target.modal
    units = modal.units
    scale = 1.0 / 25.4 if units == "G20" else 1.0
    digits = 5 if units == "G20" else 4
    start = before.position if before else (0.0, 0.0, 0.0)
    rpm = spindle_rpm if spindle_rpm is not None else (before.spindle_rpm if before else target.spindle_rpm)
    feed = modal.feed_rate or 100.0
    lines = [
        "; ---- resume preamble (generated) ----",
        units,
        "G90",
        modal.wcs,
        "G17",
    ]
    if not manual_spindle and rpm > 0:
        lines.append(f"M3 S{int(rpm)}")
    lines.append(f"G0 Z{safe_z_mm * scale:.{digits}f}")
    lines.append(f"G0 X{start[0] * scale:.{digits}f} Y{start[1] * scale:.{digits}f}")
    plunge_feed = max(feed * 0.35, 1.0)
    lines.append(f"G1 Z{start[2] * scale:.{digits}f} F{plunge_feed:.1f}")
    lines.append(f"F{feed:.1f}")
    if modal.distance == "G91":
        lines.append("G91")
    lines.append("; ---- resume at line %d ----" % line)
    return lines


def section_setup_sheet(program: ProgramMap, start: int, end: int, estimate_s: float | None = None) -> dict:
    """What remains in lines start..end: operations, tools, work offsets, estimate."""
    sections = [s for s in program.sections if s.end >= start and s.start <= end]
    infos = program.lines[start - 1 : end]
    return {
        "lines": (start, end),
        "operations": [s.name for s in sections],
        "tools": list(dict.fromkeys(s.tool for s in sections if s.tool)) or list(dict.fromkeys(i.tool for i in infos if i.tool)),
        "work_offsets": list(dict.fromkeys(i.modal.wcs for i in infos)),
        "units": infos[0].modal.units if infos else "G21",
        "estimate_s": estimate_s,
    }


# ---------------------------------------------------------------------- syntax colours
COLORS = {
    "g": "#7fb7ff",
    "m": "#c58be8",
    "coord": "#e4e9e5",
    "feed": "#e89f54",
    "tool": "#5ad2b0",
    "comment": "#7a8a8c",
    "header": "#c2df54",
    "number": "#7a8a8c",
}


class GCodeHighlighter(QSyntaxHighlighter):
    """IDLE-like colours: G words, M words, coordinates, feed/speed, tool calls, comments, operation headers."""

    def __init__(self, document) -> None:
        super().__init__(document)

        def fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            f.setFontItalic(italic)
            return f

        self.rules = [
            (QRegularExpression(r"\bN\d+"), fmt(COLORS["number"])),
            (QRegularExpression(r"\b[Gg]\d+(\.\d+)?"), fmt(COLORS["g"], bold=True)),
            (QRegularExpression(r"\b[Mm]\d+"), fmt(COLORS["m"], bold=True)),
            (QRegularExpression(r"\b[XYZIJKRxyzijkr][-+]?\d*\.?\d+"), fmt(COLORS["coord"])),
            (QRegularExpression(r"\b[FSfs][-+]?\d*\.?\d+"), fmt(COLORS["feed"])),
            (QRegularExpression(r"\b[Tt]\d+"), fmt(COLORS["tool"], bold=True)),
            (QRegularExpression(r"\([^)]*\)|;.*$"), fmt(COLORS["comment"], italic=True)),
            (QRegularExpression(r"^\s*;\s*Operation:.*$"), fmt(COLORS["header"], bold=True)),
        ]

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt override
        for pattern, form in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), form)


class _LineNumbers(QWidget):
    def __init__(self, editor: "GCodeEditor") -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.editor.line_number_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802
        self.editor.paint_line_numbers(event)


class GCodeEditor(QPlainTextEdit):
    """Plain-text editor with line numbers, syntax colours, search and go-to-line."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFont(QFont("Consolas", 10))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.highlighter = GCodeHighlighter(self.document())
        self.numbers = _LineNumbers(self)
        self.blockCountChanged.connect(lambda _n: self.setViewportMargins(self.line_number_width(), 0, 0, 0))
        self.updateRequest.connect(self._update_numbers)
        self.setViewportMargins(self.line_number_width(), 0, 0, 0)
        self.highlight_lines: tuple[int, int] | None = None

    def line_number_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_numbers(self, rect, dy) -> None:
        if dy:
            self.numbers.scroll(0, dy)
        else:
            self.numbers.update(0, rect.y(), self.numbers.width(), rect.height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.numbers.setGeometry(QRect(cr.left(), cr.top(), self.line_number_width(), cr.height()))

    def paint_line_numbers(self, event) -> None:
        painter = QPainter(self.numbers)
        painter.fillRect(event.rect(), QColor("#1c2426"))
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor("#7a8a8c"))
                painter.drawText(0, top, self.numbers.width() - 4, self.fontMetrics().height(), Qt.AlignmentFlag.AlignRight, str(block.blockNumber() + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
        painter.end()

    def go_to_line(self, number: int) -> bool:
        block = self.document().findBlockByNumber(max(0, number - 1))
        if not block.isValid():
            return False
        cursor = QTextCursor(block)
        self.setTextCursor(cursor)
        self.centerCursor()
        return True

    def find_text(self, text: str) -> bool:
        if not text:
            return False
        if self.find(text):
            return True
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        return self.find(text)

    def selected_line_range(self) -> tuple[int, int]:
        cursor = self.textCursor()
        doc = self.document()
        a = doc.findBlock(cursor.selectionStart()).blockNumber() + 1
        b = doc.findBlock(cursor.selectionEnd()).blockNumber() + 1
        return (min(a, b), max(a, b))

    def mark_lines(self, start: int, end: int, color: str = "#2a3a20") -> None:
        selections = []
        for number in range(start, end + 1):
            block = self.document().findBlockByNumber(number - 1)
            if not block.isValid():
                continue
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(QColor(color))
            sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            sel.cursor = QTextCursor(block)
            selections.append(sel)
        self.setExtraSelections(selections)


# ---------------------------------------------------------------------- editor window
class AdvancedEditorWindow(QDialog):
    """Advanced G-code editor (needs the `advanced_gcode_mode` gate). `host` is the main window."""

    def __init__(self, host, text: str = "", source: str = "generated") -> None:
        super().__init__(host)
        self.host = host
        self.setWindowTitle("Advanced G-code")
        make_working_window(self)
        self.resize(900, 700)
        self.source = source
        self.edited = False
        self.blocked_reason = ""
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        for label, slot, tip in (
            ("Open file...", self.open_file, "Load a G-code file; it is checked by the motion guard."),
            ("Load generated", self.load_generated, "Generate the program from the operations and load it."),
            ("Save...", self.save_file, "Save the program; it is checked by the motion guard first."),
            ("Run program", self.run_program, "Stream the whole program after the full pre-check."),
            ("Stop", self.stop_program, "Feed hold; editing unlocks when the machine and spindle are stopped."),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        layout.addLayout(bar)
        find_row = QHBoxLayout()
        self.find_box = QLineEdit()
        self.find_box.setPlaceholderText("Search")
        self.find_box.returnPressed.connect(lambda: self.editor.find_text(self.find_box.text()))
        find_row.addWidget(self.find_box, 1)
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(1, 10_000_000)
        self.goto_spin.setPrefix("Line ")
        find_row.addWidget(self.goto_spin)
        goto = QPushButton("Go to line")
        goto.clicked.connect(lambda: self.editor.go_to_line(self.goto_spin.value()))
        find_row.addWidget(goto)
        layout.addLayout(find_row)
        self.editor = GCodeEditor(self)
        self.editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.editor.customContextMenuRequested.connect(self._context_menu)
        self.editor.setPlainText(text)
        self.editor.textChanged.connect(self._on_edited)
        layout.addWidget(self.editor, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.lock_timer = QTimer(self)
        self.lock_timer.setInterval(400)
        self.lock_timer.timeout.connect(self.update_lock)
        self.lock_timer.start()
        self.check_program("Loaded")
        self.update_lock()

    # ---------------------------------------------------------------- state
    def program_text(self) -> str:
        return self.editor.toPlainText()

    def program_map(self) -> ProgramMap:
        return parse_program(self.program_text())

    def update_lock(self) -> None:
        """No edits while a program runs, until the machine and spindle are stopped."""
        locked = self.host._program_running() or not self.host._machine_and_spindle_stopped()
        if self.editor.isReadOnly() != locked:
            self.editor.setReadOnly(locked)
        state = "LOCKED: stop the machine and spindle to edit" if locked else "editable"
        tags = [state]
        if self.edited:
            tags.append("edited (regenerate to clear)")
        if self.blocked_reason:
            tags.append(f"cannot run: {self.blocked_reason}")
        self.status.setText(" · ".join(tags))

    def _on_edited(self) -> None:
        if self.editor.isReadOnly():
            return
        if not self.edited:
            self.host.terminal_core.log("warning", "Advanced G-code: program edited by hand (marked edited until regenerated).")
        self.edited = True

    def check_program(self, action: str, lines: list[str] | None = None, start=None) -> list:
        """Motion guard on import, save and before running; HARD/KEEPOUT block running."""
        text = "\n".join(lines) if lines is not None else self.program_text()
        violations = self.host.motion_guard.check_program_text(text, start)
        blocking = [v for v in violations if v.kind.value in ("HARD", "KEEPOUT")]
        self.blocked_reason = blocking[0].text() if blocking else ""
        if violations:
            self.host.motion_guard.report(violations, f"Advanced G-code · {action}", show_dialog=False)
        return violations

    # ---------------------------------------------------------------- file actions
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open G-code", "", "G-code (*.nc *.gcode *.tap *.ngc);;All files (*.*)")
        if not path:
            return
        self.editor.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
        self.source, self.edited = path, False
        self.check_program("Import")
        self.update_lock()

    def load_generated(self) -> None:
        prepared = self.host._prepare_gcode()
        if prepared is None:
            return
        self.editor.setPlainText(prepared[0])
        self.source, self.edited = "generated", False
        self.check_program("Generated")
        self.update_lock()

    def save_file(self) -> None:
        violations = self.check_program("Save")
        if violations:
            QMessageBox.warning(self, "Motion guard", "The program has motion-guard violations and cannot run until fixed:\n\n"
                                + "\n".join(v.text() for v in violations[:8]))
        path, _ = QFileDialog.getSaveFileName(self, "Save G-code", "", "G-code (*.nc *.gcode *.tap);;All files (*.*)")
        if path:
            Path(path).write_text(self.program_text(), encoding="utf-8")
            self.host.terminal_core.log("info", f"Saved program to {path}")

    # ---------------------------------------------------------------- running
    def run_program(self) -> None:
        self.host.start_program_stream(self.program_text().splitlines(), "Run program", start_line=1, editor=self)

    def stop_program(self) -> None:
        self.host.stop_program_stream()

    # ---------------------------------------------------------------- context menu
    def _context_menu(self, pos) -> None:
        start, end = self.editor.selected_line_range()
        menu = QMenu(self)
        actions = {
            menu.addAction("Preview beginning here"): lambda: self.preview(start, None),
            menu.addAction("Preview highlighted only"): lambda: self.preview(start, end),
            menu.addAction("Delete prior operations"): lambda: self.delete_prior(start),
            menu.addAction("Resume code here"): lambda: self.resume_here(start),
            menu.addAction("Export highlighted section"): lambda: self.export_section(start, end),
        }
        chosen = menu.exec(self.editor.viewport().mapToGlobal(pos))
        if chosen in actions:
            actions[chosen]()

    def preview(self, start: int, end: int | None) -> None:
        program = self.program_map()
        segments = program.segments(start, end)
        self.editor.mark_lines(start, end or len(program.lines))
        self.host.preview_program_segments(segments, f"lines {start}-{end or len(program.lines)}")

    def delete_prior(self, start: int) -> None:
        from basicam.gcode_dialogs import BackupFirstDialog

        program = self.program_map()
        section = program.section_at(start)
        if section is None or not program.sections or section is program.sections[0]:
            self.host.terminal_core.log("info", "Nothing before the selected operation to delete.")
            return
        dialog = BackupFirstDialog(self)
        dialog.exec()
        if dialog.choice == "cancel":
            return
        if dialog.choice == "backup":
            path, _ = QFileDialog.getSaveFileName(self, "Save backup", "", "G-code (*.nc *.gcode *.tap);;All files (*.*)")
            if not path:
                return
            Path(path).write_text(self.program_text(), encoding="utf-8")
        lines = self.program_text().splitlines()
        first = program.sections[0].start
        kept = lines[: first - 1] + lines[section.start - 1 :]
        self.editor.setPlainText("\n".join(kept) + "\n")
        self.edited = True
        self.host.terminal_core.log("warning", f"Deleted the operations before '{section.name}' (lines {first}-{section.start - 1}).")
        new_map = self.program_map()
        self.host.show_section_setup_sheet(new_map, 1, len(new_map.lines))

    def resume_here(self, line: int) -> None:
        self.host.resume_program_at(self.program_map(), line, editor=self)

    def export_section(self, start: int, end: int) -> None:
        program = self.program_map()
        info = program.info(start)
        if info is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export highlighted section", "", "G-code (*.nc *.gcode *.tap);;All files (*.*)")
        if not path:
            return
        before = program.info(start - 1) if start > 1 else info
        header = ["; ---- exported section ----", before.modal.units, "G90", before.modal.wcs]
        body = self.program_text().splitlines()[start - 1 : end]
        Path(path).write_text("\n".join(header + body) + "\n", encoding="utf-8")
        sheet = self.host.show_section_setup_sheet(program, start, end)
        Path(path).with_suffix(".setup.md").write_text(sheet, encoding="utf-8")
        self.host.terminal_core.log("info", f"Exported lines {start}-{end} to {path} with its setup sheet.")
