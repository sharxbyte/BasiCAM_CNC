"""Drag-and-drop widgets for the operation stack, plus pure reorder helpers."""

from __future__ import annotations

import re
from typing import Callable

from PySide6.QtCore import QMimeData, QPoint, Qt
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QApplication, QFrame, QWidget

MIME_TYPE = "application/x-basicam-operation"
_NUMBERED = re.compile(r"^\d+ · ")


def renumber(operations) -> None:
    """Rename auto-numbered operations ("01 · Rough") to their new positions."""
    for index, op in enumerate(operations, start=1):
        if _NUMBERED.match(op.name) or not op.name:
            op.name = f"{index:02d} · {op.kind}"


def move_operation(setups, src: tuple[int, int], dst: tuple[int, int]) -> bool:
    """Move setups[src_setup].operations[src_op] to position dst (setup, index). Returns True if changed."""
    s_si, s_oi = src
    d_si, d_oi = dst
    if not (0 <= s_si < len(setups) and 0 <= s_oi < len(setups[s_si].operations) and 0 <= d_si < len(setups)):
        return False
    if (s_si, s_oi) == (d_si, d_oi):
        return False
    op = setups[s_si].operations.pop(s_oi)
    if s_si == d_si and d_oi > s_oi:
        d_oi -= 1
    d_oi = max(0, min(d_oi, len(setups[d_si].operations)))
    setups[d_si].operations.insert(d_oi, op)
    renumber(setups[s_si].operations)
    if d_si != s_si:
        renumber(setups[d_si].operations)
    return True


class OperationRow(QFrame):
    """One operation card; drag it onto another card to reorder."""

    def __init__(self, setup_index: int, op_index: int, on_click: Callable[[int, int], None] | None = None) -> None:
        super().__init__()
        self.setup_index = setup_index
        self.op_index = op_index
        self._on_click = on_click
        self._press_pos: QPoint | None = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            if self._on_click is not None:
                self._on_click(self.setup_index, self.op_index)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press_pos is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.position().toPoint() - self._press_pos).manhattanLength() < QApplication.startDragDistance():
            return
        mime = QMimeData()
        mime.setData(MIME_TYPE, f"{self.setup_index}:{self.op_index}".encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)
        self._press_pos = None


class OperationDropArea(QWidget):
    """Container for the operation stack that accepts OperationRow drops."""

    def __init__(self, on_drop: Callable[[tuple[int, int], tuple[int, int]], None]) -> None:
        super().__init__()
        self._on_drop = on_drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(MIME_TYPE):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(MIME_TYPE):
            event.acceptProposedAction()

    def target_for(self, pos: QPoint) -> tuple[int, int] | None:
        rows = sorted((w for w in self.findChildren(OperationRow) if w.isVisible()), key=lambda w: w.y())
        if not rows:
            return None
        for row in rows:
            if pos.y() < row.y() + row.height() / 2:
                return (row.setup_index, row.op_index)
        last = rows[-1]
        return (last.setup_index, last.op_index + 1)

    def dropEvent(self, event) -> None:  # noqa: N802
        raw = bytes(event.mimeData().data(MIME_TYPE)).decode()
        try:
            si, oi = (int(v) for v in raw.split(":"))
        except ValueError:
            return
        target = self.target_for(event.position().toPoint())
        if target is not None:
            self._on_drop((si, oi), target)
            event.acceptProposedAction()
