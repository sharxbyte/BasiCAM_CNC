"""GRBL-protocol handshake: wait for the boot banner, identify the firmware, retry, diagnose.

Boards that reset when the port opens (Arduino-based GRBL) are still booting for 1-2 s and lose
anything sent at once, so the handshake first waits for the welcome banner. Without one it sends a
blank line and ``$I``, retrying ``$I`` up to three times a second apart, then asks for ``$$`` and a
status report. When nothing valid arrives it fails with a diagnostic listing the likely causes and
the raw bytes received. The state machine is driven by a clock and fed received lines, so it runs
from the GUI poll timer (no blocking sleeps) and is easy to test with a fake port.

Recognised firmware: GRBL 1.1 (``Grbl 1.1h ['$' for help]``), grblHAL (``GrblHAL 1.1f ...`` or
``[FIRMWARE:grblHAL]``) and FluidNC (``Grbl 3.7 [FluidNC v3.7.8 ...]`` / ``[VER:3.7 FluidNC ...]``).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

BOOT_WAIT_S = 3.0
QUERY_RETRIES = 3
QUERY_INTERVAL_S = 1.0
SETTINGS_WAIT_S = 1.5

LIKELY_CAUSES = (
    "the wrong port or baud rate",
    "the port is open in another program (another sender, the Arduino serial monitor)",
    "the board is not running GRBL, grblHAL or FluidNC firmware",
    "the board is still booting or is not powered",
    "a charge-only (power-only) USB cable",
)

_BANNER = re.compile(r"^(Grbl|GrblHAL|grblHAL)\s+([0-9][^\s\[]*)(.*)$", re.IGNORECASE)
_VER = re.compile(r"\[VER:([^\]]*)\]", re.IGNORECASE)
_OPT = re.compile(r"\[OPT:([^\]]*)\]", re.IGNORECASE)
_AXS = re.compile(r"\[AXS:\d+:([^\]]+)\]", re.IGNORECASE)


@dataclass
class FirmwareInfo:
    kind: str = ""  # "grbl", "grblhal" or "fluidnc"
    version: str = ""
    options: str = ""
    axes: str = ""


def detect_firmware(lines: list[str]) -> FirmwareInfo | None:
    """Firmware kind and version from the banner, ``[VER:]``, ``[OPT:]``, ``[AXS:]`` lines."""
    info = FirmwareInfo()
    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        match = _BANNER.match(line)
        if match:
            name, version, rest = match.group(1).lower(), match.group(2), match.group(3).lower()
            info.version = info.version or version
            if "fluidnc" in rest:
                info.kind = "fluidnc"
                fluid = re.search(r"fluidnc\s+v?([0-9][^\s\]\)]*)", rest)
                if fluid:
                    info.version = fluid.group(1)
            elif name == "grblhal" or "grblhal" in rest:
                info.kind = "grblhal"
            elif not info.kind:
                info.kind = "grbl"
        ver = _VER.search(line)
        if ver:
            text = ver.group(1).strip().rstrip(":")
            if "fluidnc" in text.lower():
                info.kind = "fluidnc"
                fluid = re.search(r"fluidnc\s+v?([0-9][^\s:]*)", text, re.IGNORECASE)
                info.version = fluid.group(1) if fluid else text.split()[0]
            else:
                info.version = text.split(":")[0].split()[0] if text else info.version
                info.kind = info.kind or "grbl"
        if "[firmware:grblhal" in lower or "grblhal" in lower and line.startswith("["):
            info.kind = "grblhal"
        opt = _OPT.search(line)
        if opt:
            info.options = opt.group(1).strip()
        axs = _AXS.search(line)
        if axs:
            info.axes = axs.group(1).strip().upper()
    if not info.kind:
        return None
    if info.kind == "grbl" and not info.version.startswith("1.1"):
        # Only GRBL 1.1 speaks the protocol BasiCAM uses; 0.9 and older are not identified.
        if not info.version.startswith(("1.", "2.")):
            return None
    return info


def looks_garbled(data: bytes) -> bool:
    """Bytes that do not look like GRBL text (usually a wrong baud rate)."""
    if len(data) < 8:
        return False
    printable = sum(1 for b in data if b in (9, 10, 13) or 32 <= b < 127)
    return printable / len(data) < 0.75


def describe_failure(raw: bytes, sent: list[str], baud: int, garbled: bool) -> str:
    lines = []
    if garbled:
        lines.append(f"Bytes arrived but they are not GRBL text: wrong baud rate? ({baud} was used)")
    else:
        lines.append("No reply that looks like GRBL, grblHAL or FluidNC arrived.")
    lines.append("Likely causes: " + "; ".join(LIKELY_CAUSES) + ".")
    if sent:
        lines.append("Sent: " + ", ".join(repr(s) for s in sent))
    preview = raw[:120]
    lines.append("Received: " + (preview.hex(" ") if garbled else repr(preview.decode("ascii", "replace"))) if raw else "Received: nothing")
    return "\n".join(lines)


@dataclass
class Handshake:
    """Connect-time identification. Call ``start(now)``, then ``feed(lines, raw)`` with what arrives and
    ``tick(now)`` from a timer; each returns the commands to send ("" = a blank line, b"?" realtime)."""

    baud: int = 115200
    boot_wait_s: float = BOOT_WAIT_S
    state: str = "idle"  # boot_wait, query, settings, done, failed
    lines: list[str] = field(default_factory=list)
    raw: bytearray = field(default_factory=bytearray)
    sent: list[str] = field(default_factory=list)
    firmware: FirmwareInfo | None = None
    message: str = ""
    started_at: float = 0.0
    _deadline: float = 0.0
    _queries: int = 0

    def start(self, now: float) -> list:
        self.state = "boot_wait"
        self.started_at = now
        self._deadline = now + self.boot_wait_s
        self.message = "Waiting for the controller to start"
        return []

    @property
    def done(self) -> bool:
        return self.state in ("done", "failed")

    @property
    def identified(self) -> bool:
        return self.firmware is not None

    def waited(self, now: float) -> float:
        return max(0.0, now - self.started_at)

    def status_text(self, now: float) -> str:
        if self.state == "boot_wait":
            return f"Waiting for the controller to start ({self.waited(now):.0f} s)"
        if self.state == "query":
            return f"Waiting for firmware reply ({self.waited(now):.0f} s, sent $I x{self._queries})"
        if self.state == "settings":
            return "Firmware identified; reading settings"
        if self.state == "failed":
            return "No GRBL reply"
        return "Identified" if self.firmware else ""

    def feed(self, lines: list[str], raw: bytes = b"") -> list:
        if raw:
            self.raw.extend(raw)
            del self.raw[:-4096]
        if not lines:
            return []
        self.lines.extend(lines)
        self.lines = self.lines[-200:]
        found = detect_firmware(self.lines)
        if found is not None:
            self.firmware = found
        if self.state == "boot_wait" and any(_BANNER.match(line.strip()) for line in lines):
            return self._ask(0.0, settings=self.firmware is not None)
        if self.state == "query" and self.firmware is not None:
            return self._ask(0.0, settings=True)
        return []

    def _ask(self, now: float, settings: bool) -> list:
        if settings:
            self.state = "settings"
            self._deadline = now + SETTINGS_WAIT_S if now else 0.0
            self.sent += ["$I", "$$", "?"]
            return ["$I", "$$", b"?"]
        self.state = "query"
        self._queries += 1
        self._deadline = now + QUERY_INTERVAL_S
        self.sent.append("$I")
        return ["$I"]

    def tick(self, now: float) -> list:
        if self.state == "boot_wait" and now >= self._deadline:
            # no banner: maybe the board did not reset; wake it with a blank line, then ask
            self.sent.append("(blank line)")
            return [""] + self._ask(now, settings=False)
        if self.state == "query" and now >= self._deadline:
            if self.firmware is not None:
                return self._ask(now, settings=True)
            if self._queries >= QUERY_RETRIES:
                self.state = "failed"
                self.message = describe_failure(bytes(self.raw), self.sent, self.baud, looks_garbled(bytes(self.raw)))
                return []
            return self._ask(now, settings=False)
        if self.state == "settings":
            if self._deadline == 0.0:
                self._deadline = now + SETTINGS_WAIT_S
            elif now >= self._deadline:
                self.state = "done"
                self.message = "Identified"
        return []

    def garbled(self) -> bool:
        return looks_garbled(bytes(self.raw))


@dataclass
class ProbeResult:
    ok: bool
    firmware: FirmwareInfo | None = None
    state: str = ""
    pins: str = ""
    round_trip_ms: float | None = None
    baud: int = 0
    lines: list[str] = field(default_factory=list)
    message: str = ""

    def summary(self) -> str:
        if not self.ok:
            return self.message
        fw = self.firmware
        name = {"grbl": "GRBL", "grblhal": "grblHAL", "fluidnc": "FluidNC"}.get(fw.kind, fw.kind) if fw else "?"
        parts = [f"Firmware: {name} {fw.version if fw else ''}".strip(), f"Baud rate: {self.baud}"]
        if fw and fw.options:
            parts.append(f"Options: {fw.options}")
        parts.append(f"Axes: {fw.axes or 'XYZ (not reported)'}" if fw else "Axes: ?")
        parts.append(f"State: {self.state or 'no status report'}")
        parts.append(f"Limit / input pins: {self.pins or 'none active'}")
        if self.round_trip_ms is not None:
            parts.append(f"Status round trip: {self.round_trip_ms:.0f} ms")
        return "\n".join(parts)


def _send(controller, command) -> None:
    if isinstance(command, bytes):
        controller.send_realtime(command)
    elif command == "":
        if hasattr(controller, "send_blank"):
            controller.send_blank()
    else:
        controller.send_line(command)


def run_handshake(controller, baud: int, clock=time.monotonic, wait=None, boot_wait_s: float = BOOT_WAIT_S,
                  limit_s: float = 12.0) -> Handshake:
    """Run the handshake to the end on an open controller (blocking; ``wait`` keeps a GUI alive)."""
    wait = wait or (lambda s: time.sleep(s))
    hs = Handshake(baud=baud, boot_wait_s=boot_wait_s)
    for command in hs.start(clock()):
        _send(controller, command)
    end = clock() + limit_s
    while not hs.done and clock() < end:
        raw = controller.read_raw() if hasattr(controller, "read_raw") else b""
        lines = controller.read_lines()
        for command in hs.feed(lines, raw):
            _send(controller, command)
        for command in hs.tick(clock()):
            _send(controller, command)
        wait(0.02)
    if not hs.done:
        hs.state = "failed"
        hs.message = describe_failure(bytes(hs.raw), hs.sent, baud, hs.garbled())
    return hs


def probe_controller(port: str, baud: int, factory=None, clock=time.monotonic, wait=None,
                     boot_wait_s: float = BOOT_WAIT_S) -> ProbeResult:
    """Machine > Test connection on a port that is not open: handshake, one status query, close.
    Never sends motion."""
    from basicam.machine import GrblController, parse_grbl_status

    factory = factory or GrblController
    wait = wait or (lambda s: time.sleep(s))
    try:
        controller = factory(port, baud)
    except Exception as error:  # noqa: BLE001 - shown to the owner
        return ProbeResult(False, baud=baud, message=f"Could not open {port}: {error}")
    try:
        hs = run_handshake(controller, baud, clock=clock, wait=wait, boot_wait_s=boot_wait_s)
        if not hs.identified:
            return ProbeResult(False, baud=baud, lines=list(hs.lines), message=hs.message)
        result = ProbeResult(True, hs.firmware, baud=baud, lines=list(hs.lines))
        for line in hs.lines:
            status = parse_grbl_status(line)
            if status is not None:
                result.state, result.pins = status.state, status.pins
        sent_at = clock()
        controller.send_realtime(b"?")
        end = sent_at + 1.0
        while clock() < end:
            got = False
            for line in controller.read_lines():
                result.lines.append(line)
                status = parse_grbl_status(line)
                if status is not None:
                    result.state, result.pins = status.state, status.pins
                    result.round_trip_ms = (clock() - sent_at) * 1000.0
                    got = True
            if got:
                break
            wait(0.01)
        return result
    finally:
        try:
            controller.close()
        except Exception:  # noqa: BLE001
            pass


def scan_baud_rates(port: str, rates, factory=None, clock=time.monotonic, wait=None,
                    boot_wait_s: float = BOOT_WAIT_S) -> ProbeResult:
    """Auto baud: try each rate in turn and keep the first that gives a valid GRBL reply."""
    last = ProbeResult(False, message="No baud rates to try.")
    garbled_at: list[int] = []
    for rate in rates:
        result = probe_controller(port, int(rate), factory=factory, clock=clock, wait=wait, boot_wait_s=boot_wait_s)
        if result.ok:
            return result
        if "wrong baud rate" in result.message:
            garbled_at.append(int(rate))
        last = result
    tried = ", ".join(str(r) for r in rates)
    last.message = f"No GRBL reply at any of {tried} baud.\n" + last.message
    return last


# ---------------------------------------------------------------------- limit and input pins
PIN_NAMES = {
    "X": "X limit", "Y": "Y limit", "Z": "Z limit", "A": "A limit", "B": "B limit", "C": "C limit",
    "U": "U limit", "V": "V limit", "P": "probe", "D": "door", "H": "feed hold", "R": "reset",
    "S": "cycle start", "E": "emergency stop",
}
LIMIT_LETTERS = "XYZABCUV"
AXIS_BITS = {axis: 1 << i for i, axis in enumerate("XYZABC")}


def describe_pins(pins: str) -> list[str]:
    """'XZP' -> ['X limit', 'Z limit', 'probe']."""
    return [PIN_NAMES.get(ch, ch) for ch in str(pins or "") if ch.strip()]


def all_limits_active(pins: str, axes: str = "XYZ") -> bool:
    axes = [a for a in str(axes or "XYZ").upper() if a in LIMIT_LETTERS]
    return bool(axes) and all(a in str(pins or "").upper() for a in axes)


def limit_diagnostic(pins: str, axes: str, settings: dict, alarm_after_reset: bool = False) -> str:
    """Explain a limit pin that reads active at rest (no switch pressed). Empty when nothing is wrong."""
    active = [a for a in str(pins or "").upper() if a in LIMIT_LETTERS]
    if not active and not alarm_after_reset:
        return ""
    every = all_limits_active(pins, axes)
    lines = []
    if every:
        lines.append(f"Every limit input reads active ({', '.join(active)}) while no switch should be pressed.")
    elif active:
        lines.append(f"Limit input(s) {', '.join(active)} read active.")
    if alarm_after_reset:
        lines.append("A hard-limit alarm arrived right after the reset, before any motion.")
    lines.append("Likely causes:")
    lines.append("- limit pins inverted ($5) - all of them reading active at rest is the usual sign;")
    lines.append("- normally-closed switches wired without inverting the pins;")
    lines.append("- unconnected (floating) limit inputs;")
    lines.append("- electrical noise on the switch wires (long or unshielded cables near the motors).")
    values = []
    for key, name in ((5, "limit pins inverted"), (21, "hard limits"), (22, "homing")):
        if key in settings:
            values.append(f"${key}={settings[key]:g} ({name})")
    if values:
        lines.append("Controller settings: " + ", ".join(values) + ".")
    lines.append("Run the limit-switch test in the machine panel to check each switch.")
    return "\n".join(lines)


def inverted_limit_setting(settings: dict, firmware: str, axes: str = "XYZ") -> str | None:
    """The `$5=` line that flips the limit-pin inversion, or None when it cannot be done this way.

    GRBL 1.1: $5 is one switch for all limit pins (0/1). grblHAL: $5 is a mask, one bit per axis.
    FluidNC configures pins in config.yaml, so there is no $5 to change."""
    if firmware == "fluidnc" or 5 not in settings:
        return None
    value = int(settings[5])
    if firmware == "grblhal":
        mask = 0
        for axis in str(axes or "XYZ").upper():
            mask |= AXIS_BITS.get(axis, 0)
        return f"$5={value ^ mask}"
    return f"$5={0 if value else 1}"


REFUSED_SETTINGS = {
    "$21=0": "Refused: BasiCAM never turns the controller's hard limits off ($21=0).",
    "$20=0": "Refused: BasiCAM never turns the controller's soft limits off ($20=0).",
}


def refused_setting(command: str) -> str:
    """Console refusal for settings that would switch a limit off, or change $5 outside the test."""
    match = re.match(r"^\$(\d+)=([-+]?\d*\.?\d+)$", "".join(str(command).split()))
    if not match:
        return ""
    key, value = int(match.group(1)), float(match.group(2))
    if key in (20, 21) and value == 0:
        return REFUSED_SETTINGS[f"${key}=0"]
    if key == 5:
        return "Limit-pin inversion ($5) is changed only through the limit-switch test in the machine panel."
    return ""
