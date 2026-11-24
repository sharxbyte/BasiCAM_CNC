"""One readiness gate for everything that sends to the controller (roadmap v3 Block 8).

Buttons, menu items, the jog pad, keyboard control, the console, the preflight window and program
streaming all ask ``machine_readiness(action, ctx)`` whether they may send, and why not. Nothing that
is refused is silent: every refusal has a reason and the next step to offer.

Order of steps: connect -> identify -> clear the alarm (Unlock or Home) -> verify the position
(Home, or Verify position) -> preflight (with preflight jogs) -> ready.
"""

from __future__ import annotations

from dataclasses import dataclass

# Alarms that homing clears on its own (grblHAL 11 "homing required", FluidNC 14 "unhomed").
HOMING_ALARMS = {"grblhal": {11}, "fluidnc": {14}, "grbl": set()}

ACTIONS = ("kill", "hold", "resume", "unlock", "home", "verify", "preflight", "preflight_jog", "jog", "spindle",
           "console_motion", "console_setting", "stream", "limit_test")


@dataclass
class ReadinessContext:
    connected: bool = False
    identified: bool = False
    firmware: str = "grbl"
    state: str = ""  # controller state from the last status report ("Idle", "Alarm", "Hold:0", ...)
    alarm_code: int | None = None
    position_verified: bool = False
    preflight_complete: bool = False
    program_running: bool = False
    homing_enabled: bool = True  # $22 or the machine profile
    has_homing_switches: bool = True
    limit_pins_active: bool = False  # a limit input reads active at rest
    axes_match: bool = True  # controller axes cover the machine profile
    handshake_failed: bool = False


@dataclass
class Readiness:
    allowed: bool
    reason: str = ""
    next_step: str = ""  # Connect, Test connection, Unlock, Home, Verify position, Run preflight, Wait, Limit-switch test

    def __bool__(self) -> bool:
        return self.allowed


OK = Readiness(True)


def _state(ctx: ReadinessContext) -> str:
    return str(ctx.state or "").casefold()


def in_alarm(ctx: ReadinessContext) -> bool:
    return _state(ctx).startswith("alarm")


def homing_clears_alarm(ctx: ReadinessContext) -> bool:
    if not in_alarm(ctx):
        return False
    if ctx.alarm_code is not None and ctx.alarm_code in HOMING_ALARMS.get(ctx.firmware, set()):
        return True
    return bool(ctx.homing_enabled)  # GRBL with $22=1 starts in alarm until homed


def machine_readiness(action: str, ctx: ReadinessContext) -> Readiness:
    """May ``action`` be sent now? With the reason and the next step when not."""
    state = _state(ctx)
    if not ctx.connected:
        return Readiness(False, "No controller is connected.", "Connect")
    if action == "kill":
        return OK
    if not ctx.identified:
        if ctx.handshake_failed:
            return Readiness(False, "The controller did not answer like GRBL, grblHAL or FluidNC.", "Test connection")
        return Readiness(False, "Waiting for the controller to identify its firmware.", "Wait")
    if action == "hold":
        if state in ("run", "jog", "home"):
            return OK
        return Readiness(False, "Nothing is moving.", "")
    if action == "resume":
        if state.startswith("hold"):
            return OK
        return Readiness(False, "The machine is not in a feed hold.", "")
    if ctx.program_running:
        return Readiness(False, "A program is running.", "Wait")
    if not ctx.axes_match and action in ("preflight_jog", "jog", "spindle", "console_motion", "stream", "home"):
        return Readiness(False, "The controller's axes do not match the machine profile.", "Test connection")
    if action == "unlock":
        if in_alarm(ctx):
            return OK
        return Readiness(False, "The controller is not in alarm.", "")
    if action == "limit_test":
        if state in ("run", "jog", "home"):
            return Readiness(False, "Stop the machine first.", "Wait")
        return OK
    if action == "console_setting":
        if state in ("run", "jog", "home"):
            return Readiness(False, "Settings can be changed only while the machine is stopped.", "Wait")
        return OK
    if action == "home":
        if not ctx.has_homing_switches:
            return Readiness(False, "This machine has no homing switches.", "Verify position")
        if not ctx.homing_enabled:
            return Readiness(False, "Homing is turned off in the controller ($22=0).", "Verify position")
        if ctx.limit_pins_active:
            return Readiness(False, "A limit input reads active with nothing pressed; homing would fail.",
                             "Limit-switch test")
        if state == "idle" or homing_clears_alarm(ctx):
            return OK
        if in_alarm(ctx):
            return Readiness(False, "Clear the alarm first.", "Unlock")
        return Readiness(False, f"The controller is busy ({ctx.state}).", "Wait")
    # everything below moves the machine (or confirms its position)
    if in_alarm(ctx):
        if ctx.limit_pins_active:
            return Readiness(False, "The controller is in alarm and a limit input reads active.", "Limit-switch test")
        if homing_clears_alarm(ctx) and ctx.has_homing_switches:
            return Readiness(False, "The controller is in alarm until it is homed.", "Home")
        return Readiness(False, "The controller is in alarm.", "Unlock")
    if action == "verify":
        if state != "idle":
            return Readiness(False, f"The controller must be Idle ({ctx.state or 'no status yet'}).", "Wait")
        return OK
    if state not in ("idle", "jog"):
        return Readiness(False, f"The controller must be Idle ({ctx.state or 'no status yet'}).", "Wait")
    if not ctx.position_verified:
        step = "Home" if ctx.has_homing_switches and ctx.homing_enabled else "Verify position"
        return Readiness(False, "The machine position is not verified.", step)
    if action in ("preflight", "preflight_jog"):
        return OK
    if not ctx.preflight_complete:
        return Readiness(False, "The preflight check is not done.", "Run preflight")
    return OK


CHECKLIST = (
    ("connected", "Connected"),
    ("identified", "Firmware identified"),
    ("no_alarm", "Not in alarm"),
    ("verified", "Position verified"),
    ("preflight", "Preflight done"),
    ("ready", "Ready"),
)


def checklist(ctx: ReadinessContext) -> list[tuple[str, str, bool, str]]:
    """(key, label, done, action button text) for the ready checklist; the first unmet item is the next step."""
    ready = machine_readiness("jog", ctx).allowed
    no_alarm = ctx.connected and ctx.identified and not in_alarm(ctx)
    alarm_action = "Home" if (ctx.connected and homing_clears_alarm(ctx) and ctx.has_homing_switches
                              and not ctx.limit_pins_active) else "Unlock"
    if ctx.limit_pins_active and in_alarm(ctx):
        alarm_action = "Limit-switch test"
    verify_action = "Home" if ctx.has_homing_switches and ctx.homing_enabled else "Verify position"
    items = [
        ("connected", "Connected", ctx.connected, "Connect"),
        ("identified", "Firmware identified", ctx.connected and ctx.identified, "Test connection"),
        ("no_alarm", "Not in alarm", no_alarm, alarm_action),
        ("verified", "Position verified", no_alarm and ctx.position_verified, verify_action),
        ("preflight", "Preflight done", no_alarm and ctx.position_verified and ctx.preflight_complete, "Run preflight"),
        ("ready", "Ready", ready, ""),
    ]
    return items


def first_unmet(items) -> str | None:
    for key, _label, done, _action in items:
        if not done:
            return key
    return None
