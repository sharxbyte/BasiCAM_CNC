"""Window sizing: every window fits the screen it opens on, can be resized and remembers its size.

`fit_to_screen` clamps a window to the available geometry (the screen minus the taskbar), keeps
its title bar on screen and lowers its minimum size so layouts can shrink. `WindowFitter` is an
application event filter that applies it to every dialog when it is first shown, and saves and
restores each dialog's geometry under ``windows/<key>`` in the main window's QSettings.
`scrollable` wraps a long form in a scroll area.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QEvent, QObject, QRect, QSize, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractButton,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFontDialog,
    QFrame,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

TITLE_BAR_PX = 32  # room kept for the native title bar above the client area
FRAME_PX = 8  # window border on each side
MIN_SIZE = QSize(320, 200)
# Small system dialogs keep their natural size and are not remembered.
_NATURAL = (QMessageBox, QInputDialog, QFileDialog, QProgressDialog, QColorDialog, QFontDialog)
_override_rect: QRect | None = None  # tests: simulate a smaller screen


def set_available_override(rect: QRect | None) -> None:
    """Tests only: pretend the available screen area is ``rect``."""
    global _override_rect
    _override_rect = QRect(rect) if rect is not None else None


def available_rect(widget: QWidget | None = None) -> QRect:
    """Usable area (screen minus taskbar) of the screen the widget, or its parent, is on."""
    if _override_rect is not None:
        return QRect(_override_rect)
    screen = None
    probe = widget
    while probe is not None and screen is None:
        try:
            if probe.isWindow() and probe.isVisible():
                screen = probe.screen()
        except RuntimeError:
            screen = None
        probe = probe.parentWidget()
    if screen is None and widget is not None and widget.parentWidget() is not None:
        screen = widget.parentWidget().window().screen()
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 720)


def client_limit(area: QRect) -> QSize:
    """Largest client size whose frame (borders and title bar) still fits ``area``."""
    return QSize(max(area.width() - 2 * FRAME_PX, MIN_SIZE.width()),
                 max(area.height() - TITLE_BAR_PX - FRAME_PX, MIN_SIZE.height()))


def fit_to_screen(window: QWidget, preferred: tuple[int, int] | None = None) -> None:
    """Clamp size and position to the available area; keep the title bar reachable."""
    area = available_rect(window)
    limit = client_limit(area)
    # The layout's own minimum is kept unless it is larger than the screen; long forms scroll.
    hint = window.minimumSizeHint()
    explicit = window.minimumSize()
    min_w = max(explicit.width(), hint.width() if hint.isValid() else 0)
    min_h = max(explicit.height(), hint.height() if hint.isValid() else 0)
    window.setMinimumSize(min(min_w, limit.width()), min(min_h, limit.height()))
    if preferred is not None:
        width, height = preferred
    else:
        size = window.size()
        hint = window.sizeHint()
        width = max(size.width(), hint.width()) if not window.isVisible() else size.width()
        height = max(size.height(), hint.height()) if not window.isVisible() else size.height()
    width = max(MIN_SIZE.width(), min(int(width), limit.width()))
    height = max(MIN_SIZE.height(), min(int(height), limit.height()))
    window.resize(width, height)
    clamp_position(window, area)


def clamp_position(window: QWidget, area: QRect | None = None) -> None:
    """Move the window so its whole frame, title bar included, is inside the area."""
    area = area or available_rect(window)
    pos = window.pos()
    width = window.width() + 2 * FRAME_PX
    height = window.height() + TITLE_BAR_PX + FRAME_PX
    x = min(max(pos.x(), area.left()), max(area.left(), area.right() + 1 - width))
    y = min(max(pos.y(), area.top()), max(area.top(), area.bottom() + 1 - height))
    if (x, y) != (pos.x(), pos.y()):
        window.move(x, y)


def make_working_window(dialog: QDialog) -> None:
    """A working window (inventory, editor, nesting...): minimise / maximise buttons and a size grip."""
    flags = dialog.windowFlags()
    flags |= Qt.WindowType.WindowMinMaxButtonsHint | Qt.WindowType.WindowCloseButtonHint
    flags &= ~Qt.WindowType.WindowContextHelpButtonHint
    dialog.setWindowFlags(flags)
    dialog.setSizeGripEnabled(True)


def scrollable(content: QWidget, parent: QWidget | None = None) -> QScrollArea:
    """Wrap a long form so it scrolls instead of making the window taller than the screen."""
    area = QScrollArea(parent)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setWidget(content)
    return area


def wrap_if_too_tall(dialog: QDialog, area: QRect | None = None) -> bool:
    """When a dialog's layout cannot shrink to the screen, move its content into a scroll area.

    A trailing button box (or a row of buttons) stays pinned below the scroll area so OK / Save /
    Cancel are always visible. Returns True when the dialog was wrapped."""
    layout = dialog.layout()
    if layout is None or getattr(dialog, "_fit_wrapped", False):
        return False
    limit = client_limit(area or available_rect(dialog))
    need = layout.totalMinimumSize()
    if need.height() <= limit.height() and need.width() <= limit.width():
        return False
    pinned = None
    count = layout.count()
    if count > 1:
        last = layout.itemAt(count - 1)
        widget = last.widget()
        inner = last.layout()
        if isinstance(widget, QDialogButtonBox):
            pinned = layout.takeAt(count - 1)
        elif inner is not None and inner.count() and all(
            isinstance(inner.itemAt(i).widget(), (QAbstractButton, type(None))) for i in range(inner.count())
        ):
            pinned = layout.takeAt(count - 1)
    content = QWidget()
    content.setLayout(layout)  # takes the dialog's layout (the dialog is left without one)
    outer = QVBoxLayout()
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(scrollable(content), 1)
    if pinned is not None:
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        if pinned.widget() is not None:
            holder_layout.addWidget(pinned.widget())
        elif pinned.layout() is not None:
            holder_layout.addLayout(pinned.layout())
        outer.addWidget(holder)
    dialog.setLayout(outer)
    dialog._fit_wrapped = True
    return True


def geometry_key(window: QWidget) -> str:
    name = window.objectName()
    if name and not name.startswith("qt_"):
        return name
    title = "".join(ch for ch in window.windowTitle() if ch.isalnum() or ch in " -_").strip()
    return f"{type(window).__name__}:{title or 'untitled'}"


class WindowFitter(QObject):
    """Application event filter: fit every dialog on first show, remember its geometry on hide."""

    def __init__(self, settings=None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._fitted: set[int] = set()

    def _wants(self, obj) -> bool:
        return isinstance(obj, QDialog) and obj.isWindow()

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        try:
            kind = event.type()
            if kind == QEvent.Type.Show and self._wants(obj) and id(obj) not in self._fitted:
                self._fitted.add(id(obj))
                obj.destroyed.connect(lambda *_a, i=id(obj): self._fitted.discard(i))
                self.on_first_show(obj)
            elif kind == QEvent.Type.Hide and self._wants(obj) and id(obj) in self._fitted:
                self.save(obj)
        except RuntimeError:  # widget already deleted
            pass
        return False

    def on_first_show(self, dialog: QDialog) -> None:
        if isinstance(dialog, _NATURAL):
            clamp_position(dialog)
            return
        wrap_if_too_tall(dialog)
        if not self.restore(dialog):
            fit_to_screen(dialog)
            parent = dialog.parentWidget()
            if parent is not None:  # centre over the parent, then keep on screen
                centre = parent.window().frameGeometry().center()
                dialog.move(centre.x() - dialog.width() // 2, centre.y() - dialog.height() // 2)
                clamp_position(dialog)

    def restore(self, dialog: QWidget) -> bool:
        if self.settings is None:
            return False
        data = self.settings.value(f"windows/{geometry_key(dialog)}")
        if not isinstance(data, (QByteArray, bytes)) or not data:
            return False
        if not dialog.restoreGeometry(QByteArray(data)):
            return False
        fit_to_screen(dialog, (dialog.width(), dialog.height()))  # pulls an off-screen window back
        return True

    def save(self, dialog: QWidget) -> None:
        if self.settings is None or isinstance(dialog, _NATURAL):
            return
        self.settings.setValue(f"windows/{geometry_key(dialog)}", dialog.saveGeometry())
