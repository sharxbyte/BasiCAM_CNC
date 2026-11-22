"""Keyboard machine control state machine (no Qt).

Direct keys: arrows move X/Y, Page Up / Page Down move Z (Shift = large, Ctrl = fine step),
Home homes, End stops. Axis entry: X/Y/Z, then + or -, type the distance, Enter shows a
ghost preview, Enter again confirms and sends. Esc backs up one step (or out of the move).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KState(str, Enum):
    IDLE = "Idle"
    AXIS_CHOSEN = "AxisChosen"
    SIGN_CHOSEN = "SignChosen"
    TYPING = "Typing"
    PREVIEW = "Preview"


@dataclass(frozen=True)
class MoveRequest:
    axis: str
    distance: float  # mm, signed
    feed: float  # mm/min
    is_cut: bool = False


@dataclass(frozen=True)
class KeyResult:
    action: str  # "none", "preview", "send", "home", "stop", "cancel", "status"
    request: MoveRequest | None = None
    message: str = ""


STEPS_MM = {"medium": 1.0, "fine": 0.1, "large": 10.0}
STEPS_IN = {"medium": 0.1, "fine": 0.01, "large": 1.0}

DIRECT = {
    "Left": ("X", -1), "Right": ("X", 1), "Up": ("Y", 1), "Down": ("Y", -1),
    "PageUp": ("Z", 1), "PageDown": ("Z", -1),
}


class KeyboardMachineControl:
    def __init__(self, jog_feed: float = 300.0, cut_feed: float = 200.0, inches: bool = False) -> None:
        self.jog_feed = jog_feed
        self.cut_feed = cut_feed
        self.inches = inches
        self.spindle_on = False
        self.reset()

    def reset(self) -> None:
        self.state = KState.IDLE
        self.axis: str | None = None
        self.sign = 1
        self.text = ""
        self.pending: MoveRequest | None = None

    # -------------------------------------------------------------------
    def step_mm(self, shift: bool = False, ctrl: bool = False) -> float:
        size = "fine" if ctrl else ("large" if shift else "medium")
        return STEPS_IN[size] * 25.4 if self.inches else STEPS_MM[size]

    def _request(self, axis: str, distance: float) -> MoveRequest:
        cut = bool(self.spindle_on)
        return MoveRequest(axis, distance, self.cut_feed if cut else self.jog_feed, cut)

    def prompt(self) -> str:
        if self.state is KState.IDLE:
            return ""
        unit = "in" if self.inches else "mm"
        sign = "+" if self.sign > 0 else "-"
        if self.state is KState.AXIS_CHOSEN:
            return f"{self.axis}: press + or -"
        if self.state in (KState.SIGN_CHOSEN, KState.TYPING):
            return f"{self.axis}{sign}{self.text or '_'} {unit}: type the distance, Enter to preview"
        if self.pending is not None:
            return f"{self.axis}{sign}{abs(self.pending.distance):g} mm: Enter to send, Esc to change"
        return ""

    def handle(self, key: str, shift: bool = False, ctrl: bool = False) -> KeyResult:
        if key == "Escape":
            return self._escape()
        if key == "Home" and self.state is KState.IDLE:
            return KeyResult("home", message="Home the machine")
        if key == "End":
            self.reset()
            return KeyResult("stop", message="Stop")
        if self.state is KState.IDLE:
            if key in DIRECT:
                axis, direction = DIRECT[key]
                return KeyResult("send", self._request(axis, direction * self.step_mm(shift, ctrl)), "Keyboard jog")
            if key.upper() in ("X", "Y", "Z"):
                self.state, self.axis = KState.AXIS_CHOSEN, key.upper()
                return KeyResult("status", message=self.prompt())
            return KeyResult("none")
        if self.state is KState.AXIS_CHOSEN:
            if key in ("+", "-"):
                self.sign = 1 if key == "+" else -1
                self.state = KState.SIGN_CHOSEN
                return KeyResult("status", message=self.prompt())
            if key.upper() in ("X", "Y", "Z"):
                self.axis = key.upper()
                return KeyResult("status", message=self.prompt())
            return KeyResult("none")
        if self.state in (KState.SIGN_CHOSEN, KState.TYPING):
            if key.isdigit() or (key == "." and "." not in self.text):
                self.text += key
                self.state = KState.TYPING
                return KeyResult("status", message=self.prompt())
            if key == "Backspace":
                self.text = self.text[:-1]
                self.state = KState.TYPING if self.text else KState.SIGN_CHOSEN
                return KeyResult("status", message=self.prompt())
            if key == "Enter" and self.state is KState.TYPING:
                try:
                    value = float(self.text)
                except ValueError:
                    return KeyResult("none", message="Type a number")
                if value <= 0:
                    return KeyResult("none", message="Distance must be more than zero")
                mm = value * (25.4 if self.inches else 1.0)
                self.pending = self._request(self.axis or "X", self.sign * mm)
                self.state = KState.PREVIEW
                return KeyResult("preview", self.pending, self.prompt())
            return KeyResult("none")
        if self.state is KState.PREVIEW:
            if key == "Enter" and self.pending is not None:
                request = self.pending
                self.reset()
                return KeyResult("send", request, "Confirmed")
            return KeyResult("none")
        return KeyResult("none")

    def _escape(self) -> KeyResult:
        """Back up one step; from Idle nothing happens."""
        if self.state is KState.PREVIEW:
            self.pending = None
            self.state = KState.TYPING
            return KeyResult("cancel", message=self.prompt())
        if self.state is KState.TYPING:
            self.text = ""
            self.state = KState.SIGN_CHOSEN
            return KeyResult("cancel", message=self.prompt())
        if self.state is KState.SIGN_CHOSEN:
            self.state = KState.AXIS_CHOSEN
            return KeyResult("cancel", message=self.prompt())
        if self.state is KState.AXIS_CHOSEN:
            self.reset()
            return KeyResult("cancel", message="Move cancelled")
        return KeyResult("none")


QT_KEY_NAMES = {
    "Key_Left": "Left", "Key_Right": "Right", "Key_Up": "Up", "Key_Down": "Down",
    "Key_PageUp": "PageUp", "Key_PageDown": "PageDown", "Key_Home": "Home", "Key_End": "End",
    "Key_Escape": "Escape", "Key_Return": "Enter", "Key_Enter": "Enter", "Key_Backspace": "Backspace",
    "Key_Plus": "+", "Key_Minus": "-", "Key_Period": ".",
}
